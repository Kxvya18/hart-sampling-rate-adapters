import os
import sys
import json
import time
import argparse
import tempfile
import inspect
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Keep local-only loading for HF
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
from sklearn.metrics import accuracy_score, f1_score

# Use the finetuning HAR dataloader for exact preprocessing and LOSO ordering
FINETUNE_DIR = "/home/user/suyog/anonymous_github/finetuning_HART"
sys.path.insert(0, FINETUNE_DIR)
from HAR_dataloader import (  # noqa: E402
    load_dataset,
    get_input_columns,
    get_label_column,
    get_subject_column,
)

from tsfm_public.toolkit.dataset import ClassificationDFDataset  # noqa: E402
from tsfm_public.toolkit.time_series_classification_preprocessor import (  # noqa: E402
    TimeSeriesClassificationPreprocessor,
)
from safetensors.torch import load_file  # noqa: E402

# Reuse the finetuning model factory to match the saved state dict structure
from finetune import load_model  # noqa: E402


@dataclass
class EvalResult:
    subject_id: int
    fold_idx: int
    fp32_f1: float
    fp32_acc: float
    int8_f1: float
    int8_acc: float
    fp32_size_mb: float
    int8_size_mb: float
    retention_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PTQ evaluation for finetuned TSPulse HAR models")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=(
            "/home/user/suyog/quantization_pilot/finetuned_models_bestckpt/"
            "tspulse_dual_domain_pretrained_120_full_dataset_bestckpt_ep100_ctx120"
        ),
        help="Directory containing per-dataset fold checkpoints",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="wisdm,motion_sense,har_sense",
        help="Comma-separated dataset list",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="/home/user/suyog/tspulse_dual_domain_pretrained_120_full_dataset_bestckpt",
        help="Pretrained checkpoint path used to initialize the finetuned model class",
    )
    parser.add_argument("--context_length", type=int, default=120)
    parser.add_argument("--hop_length", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument(
        "--test_set_dir",
        type=str,
        default="/home/user/suyog/test_set",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/user/suyog/quantization_pilot/results",
    )
    parser.add_argument("--output_prefix", type=str, default="ptq_finetuned_bestckpt")
    return parser.parse_args()


def _set_cuda_visible(device: str) -> None:
    if device.startswith("cuda"):
        gpu_idx = device.split(":")[-1] if ":" in device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx


def _prepare_loso_dataset(
    dataset_name: str,
    context_length: int,
    hop_length: int,
    test_set_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    input_columns = get_input_columns(dataset_name)
    label_column = get_label_column(dataset_name)
    subject_col = get_subject_column(dataset_name)

    df_cleaned = load_dataset(
        dataset_name,
        window_length=context_length,
        hop_length=hop_length,
        test_set_dir=test_set_dir,
    )

    # Keep only labels common to all subjects (same as finetune.py)
    per_subject_labels = df_cleaned.groupby(subject_col)[label_column].apply(set)
    common_labels = set.intersection(*per_subject_labels)
    df_cleaned = df_cleaned[df_cleaned[label_column].isin(common_labels)].reset_index(drop=True)

    # Order matters: fold index is the order of unique subjects
    subjects = df_cleaned[subject_col].unique().tolist()

    all_classes = sorted(df_cleaned[label_column].unique())

    return df_cleaned, input_columns, label_column, subject_col, all_classes, subjects


def _build_test_dataset(
    df_train,
    df_test,
    input_columns,
    label_column,
    all_classes,
    context_length: int,
) -> Tuple[ClassificationDFDataset, TimeSeriesClassificationPreprocessor]:
    tsp = TimeSeriesClassificationPreprocessor(
        input_columns=input_columns,
        label_column=label_column,
        scaling=True,
    )
    tsp.train(df_train)
    tsp.label_encoder.classes_ = np.array(all_classes)

    df_test_prep = tsp.preprocess(df_test)

    ds_kwargs = dict(
        id_columns=[],
        timestamp_column=None,
        input_columns=input_columns,
        label_column=label_column,
        context_length=context_length,
        static_categorical_columns=[],
        stride=1,
        enable_padding=False,
        full_series=True,
    )

    test_dataset = ClassificationDFDataset(df_test_prep, **ds_kwargs)
    return test_dataset, tsp


def _get_logits(model_output):
    if hasattr(model_output, "prediction_outputs"):
        return model_output.prediction_outputs
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    raise ValueError("Unsupported model output format")


def _evaluate_model(model, dataloader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    accepted = set(inspect.signature(model.forward).parameters.keys())
    preds: List[int] = []
    labels: List[int] = []

    with torch.inference_mode():
        for batch in dataloader:
            y = batch.pop("target_values")
            batch = {k: v.to(device) for k, v in batch.items() if k in accepted}
            outputs = model(**batch)
            logits = _get_logits(outputs)
            pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            preds.extend(pred)
            labels.extend(y.numpy().tolist())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1


def _model_size_mb_from_path(model_path: str) -> float:
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)


def _quantized_size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    os.remove(tmp.name)
    return size_mb


def _load_fp32_model(
    checkpoint_dir: str,
    pretrained_checkpoint: str,
    tsp: TimeSeriesClassificationPreprocessor,
    num_targets: int,
    context_length: int,
    device: torch.device,
):
    model = load_model(
        pretrained_checkpoint,
        tsp,
        num_targets,
        context_length,
        device,
    )
    state = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    model.load_state_dict(state, strict=True)
    return model.to(device)


def _quantize_model(model) -> torch.nn.Module:
    model_cpu = model.to("cpu")
    model_cpu.eval()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


def evaluate_dataset(
    dataset_name: str,
    run_dir: str,
    pretrained_checkpoint: str,
    context_length: int,
    hop_length: int,
    batch_size: int,
    device: torch.device,
    test_set_dir: str,
) -> Dict[str, object]:
    df_cleaned, input_columns, label_column, subject_col, all_classes, subjects = _prepare_loso_dataset(
        dataset_name,
        context_length,
        hop_length,
        test_set_dir,
    )

    results: List[EvalResult] = []

    for fold_idx, subject_id in enumerate(subjects, start=1):
        df_train = df_cleaned[df_cleaned[subject_col] != subject_id].reset_index(drop=True)
        df_test = df_cleaned[df_cleaned[subject_col] == subject_id].reset_index(drop=True)

        test_dataset, tsp = _build_test_dataset(
            df_train,
            df_test,
            input_columns,
            label_column,
            all_classes,
            context_length,
        )

        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        fold_dir = os.path.join(run_dir, f"{dataset_name}_fold{fold_idx}")
        model_path = os.path.join(fold_dir, "model.safetensors")

        fp32_model = _load_fp32_model(
            fold_dir,
            pretrained_checkpoint,
            tsp,
            len(all_classes),
            context_length,
            device,
        )
        fp32_acc, fp32_f1 = _evaluate_model(fp32_model, dataloader, device)
        fp32_size_mb = _model_size_mb_from_path(model_path)

        int8_model = _quantize_model(fp32_model)
        int8_acc, int8_f1 = _evaluate_model(int8_model, dataloader, torch.device("cpu"))
        int8_size_mb = _quantized_size_mb(int8_model)

        retention = (int8_f1 / fp32_f1 * 100.0) if fp32_f1 > 0 else 0.0

        results.append(
            EvalResult(
                subject_id=subject_id,
                fold_idx=fold_idx,
                fp32_f1=fp32_f1,
                fp32_acc=fp32_acc,
                int8_f1=int8_f1,
                int8_acc=int8_acc,
                fp32_size_mb=fp32_size_mb,
                int8_size_mb=int8_size_mb,
                retention_pct=retention,
            )
        )

    fp32_f1s = [r.fp32_f1 for r in results]
    int8_f1s = [r.int8_f1 for r in results]
    fp32_accs = [r.fp32_acc for r in results]
    int8_accs = [r.int8_acc for r in results]
    retentions = [r.retention_pct for r in results]
    ratios = [r.fp32_size_mb / r.int8_size_mb if r.int8_size_mb > 0 else 0.0 for r in results]

    summary = {
        "dataset": dataset_name,
        "subjects": len(results),
        "fp32_f1_mean": float(np.mean(fp32_f1s)),
        "fp32_f1_std": float(np.std(fp32_f1s)),
        "int8_f1_mean": float(np.mean(int8_f1s)),
        "int8_f1_std": float(np.std(int8_f1s)),
        "fp32_acc_mean": float(np.mean(fp32_accs)),
        "int8_acc_mean": float(np.mean(int8_accs)),
        "retention_mean_pct": float(np.mean(retentions)),
        "compression_ratio_mean": float(np.mean(ratios)),
        "per_subject": [r.__dict__ for r in results],
    }

    return summary


def main() -> None:
    args = parse_args()
    _set_cuda_visible(args.device)

    device = torch.device("cuda:0" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": args.run_dir,
        "context_length": args.context_length,
        "hop_length": args.hop_length,
        "batch_size": args.batch_size,
        "device": args.device,
        "datasets": [],
    }

    for dataset_name in datasets:
        summary = evaluate_dataset(
            dataset_name,
            args.run_dir,
            args.pretrained_checkpoint,
            args.context_length,
            args.hop_length,
            args.batch_size,
            device,
            args.test_set_dir,
        )
        all_results["datasets"].append(summary)

    json_path = os.path.join(args.output_dir, f"{args.output_prefix}_results.json")
    csv_path = os.path.join(args.output_dir, f"{args.output_prefix}_summary.csv")
    txt_path = os.path.join(args.output_dir, f"{args.output_prefix}_report.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Summary CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "dataset,subjects,fp32_f1_mean,int8_f1_mean,retention_mean_pct,compression_ratio_mean,fp32_acc_mean,int8_acc_mean\n"
        )
        for ds in all_results["datasets"]:
            f.write(
                f"{ds['dataset']},{ds['subjects']},"
                f"{ds['fp32_f1_mean']:.6f},{ds['int8_f1_mean']:.6f},"
                f"{ds['retention_mean_pct']:.2f},{ds['compression_ratio_mean']:.2f},"
                f"{ds['fp32_acc_mean']:.6f},{ds['int8_acc_mean']:.6f}\n"
            )

    # Human-readable report
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("PTQ finetuned results\n")
        f.write(f"Run dir: {args.run_dir}\n")
        f.write(f"Context length: {args.context_length}\n")
        f.write(f"Hop length: {args.hop_length}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Device (fp32): {args.device}\n\n")
        for ds in all_results["datasets"]:
            f.write(f"Dataset: {ds['dataset']}\n")
            f.write(f"  Subjects: {ds['subjects']}\n")
            f.write(f"  FP32 F1: {ds['fp32_f1_mean']:.6f} (std {ds['fp32_f1_std']:.6f})\n")
            f.write(f"  INT8 F1: {ds['int8_f1_mean']:.6f} (std {ds['int8_f1_std']:.6f})\n")
            f.write(f"  Retention: {ds['retention_mean_pct']:.2f}%\n")
            f.write(f"  Compression: {ds['compression_ratio_mean']:.2f}x\n")
            f.write(f"  FP32 Acc: {ds['fp32_acc_mean']:.6f}\n")
            f.write(f"  INT8 Acc: {ds['int8_acc_mean']:.6f}\n")
            f.write("\n")

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
