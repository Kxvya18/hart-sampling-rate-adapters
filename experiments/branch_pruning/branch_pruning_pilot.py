import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, classification_report

REPO_ROOT = Path(__file__).resolve().parent
FINETUNE_SCRIPT = REPO_ROOT / "finetuning_HART" / "finetune.py"
LOGS_ROOT = REPO_ROOT / "results" / "logs_tspulse_dual_domain_pretrained_ep100_128"
OUTPUT_ROOT = REPO_ROOT / "outputs_branch_pruning"
RUNS_ROOT = OUTPUT_ROOT / "runs"

ID_FALLBACK = ["wisdm", "daphnet", "motion_sense"]
OOD_FALLBACK = ["har_sense"]


@dataclass
class DatasetRuntime:
    name: str
    train_runtime: float


def parse_train_runtime(log_path: Path) -> float | None:
    pattern = re.compile(r"'train_runtime':\s*([0-9.]+)")
    runtimes = []
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    for match in pattern.finditer(text):
        runtimes.append(float(match.group(1)))
    return min(runtimes) if runtimes else None


def collect_runtimes(logs_root: Path) -> Dict[str, DatasetRuntime]:
    runtimes = {}
    for log_path in logs_root.glob("*.log"):
        name = log_path.stem
        runtime = parse_train_runtime(log_path)
        if runtime is not None:
            runtimes[name] = DatasetRuntime(name=name, train_runtime=runtime)
    return runtimes


def select_datasets(runtimes: Dict[str, DatasetRuntime]) -> Tuple[List[str], List[str], List[str]]:
    id_candidates = [d for d in ID_FALLBACK if d in runtimes]
    ood_candidates = [d for d in OOD_FALLBACK if d in runtimes]

    if not id_candidates:
        id_candidates = ID_FALLBACK
    if not ood_candidates:
        ood_candidates = OOD_FALLBACK

    id_sorted = sorted(id_candidates, key=lambda d: runtimes.get(d, DatasetRuntime(d, float("inf"))).train_runtime)
    ood_sorted = sorted(ood_candidates, key=lambda d: runtimes.get(d, DatasetRuntime(d, float("inf"))).train_runtime)

    selected_id = id_sorted[:2]
    selected_ood = ood_sorted[:1]

    selected = selected_id + selected_ood
    return selected, selected_id, selected_ood


def run_finetune(args: List[str]) -> None:
    cmd = [sys.executable, str(FINETUNE_SCRIPT)] + args
    env = os.environ.copy()
    extra_paths = [str(REPO_ROOT)]
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + [env.get("PYTHONPATH", "")])
    subprocess.run(cmd, check=True, env=env)


def get_fold_output_dir(checkpoint_path: Path, epochs: int, ctx: int, dataset: str, fold_idx: int, variant: str, mode: str) -> Path:
    checkpoint_name = checkpoint_path.name
    run_output_dir = RUNS_ROOT / f"{checkpoint_name}_ep{epochs}_ctx{ctx}"
    return run_output_dir / f"{dataset}_fold{fold_idx}_{variant}_{mode}"


def load_har_dataloader():
    sys.path.append(str(REPO_ROOT / "finetuning_HART"))
    sys.path.append(str(REPO_ROOT))
    from HAR_dataloader import load_dataset, get_input_columns, get_label_column, get_subject_column
    return load_dataset, get_input_columns, get_label_column, get_subject_column


def load_finetune_module():
    sys.path.append(str(REPO_ROOT / "finetuning_HART"))
    sys.path.append(str(REPO_ROOT))
    import finetune as finetune_mod
    return finetune_mod


def measure_latency(model: torch.nn.Module, device: torch.device, seq_len: int, channels: int, iters: int = 200, warmup: int = 50) -> Dict[str, float]:
    model.eval()
    x = torch.randn(1, seq_len, channels, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(past_values=x)
        torch.cuda.synchronize() if device.type == "cuda" else None
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(past_values=x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    arr = np.array(times)
    return {
        "latency_mean_ms": float(arr.mean() * 1000),
        "latency_std_ms": float(arr.std() * 1000),
        "latency_median_ms": float(np.median(arr) * 1000),
    }


def model_size_mb(model: torch.nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 * 1024)


def evaluate_checkpoint(metadata_dir: Path, checkpoint_path: Path, dataset: str, context_length: int, hop_length: int, device: str, variant: str, mode: str, test_set_dir: str | None) -> Dict[str, float]:
    import json
    finetune_mod = load_finetune_module()
    load_dataset, get_input_columns, get_label_column, get_subject_column = load_har_dataloader()

    input_columns = get_input_columns(dataset)
    label_column = get_label_column(dataset)
    subject_col = get_subject_column(dataset)

    load_kwargs = dict(window_length=context_length, hop_length=hop_length)
    if test_set_dir:
        load_kwargs["test_set_dir"] = test_set_dir
    df_cleaned = load_dataset(dataset, **load_kwargs)
    per_subject_labels = df_cleaned.groupby(subject_col)[label_column].apply(set)
    common_labels = set.intersection(*per_subject_labels)
    df_cleaned = df_cleaned[df_cleaned[label_column].isin(common_labels)].reset_index(drop=True)
    all_classes = sorted(df_cleaned[label_column].unique())

    # ✅ FIX: Read metadata about which subject was held out during training
    metadata_path = metadata_dir / "train_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            sid = metadata.get("held_out_subject")
    else:
        sid = df_cleaned[subject_col].unique()[0]  # Fallback if metadata missing
    df_train = df_cleaned[df_cleaned[subject_col] != sid].reset_index(drop=True)
    df_test = df_cleaned[df_cleaned[subject_col] == sid].reset_index(drop=True)

    train_ds, val_ds, test_ds, tsp = finetune_mod.prepare_dataset(
        df_train, df_test, input_columns, label_column, all_classes, context_length=context_length
    )

    # Load finetuned checkpoint with the specific soft knockout variant applied
    model = finetune_mod.load_model(
        str(metadata_dir),
        tsp,
        len(all_classes),
        context_length,
        device,
        variant,  # Apply soft knockout variant here
    )

    trainer = finetune_mod.Trainer(
        model=model,
        args=finetune_mod.TrainingArguments(
            output_dir=str(metadata_dir / "eval_tmp"),
            per_device_eval_batch_size=256,
            report_to="none",
            label_names=["target_values"],
        ),
    )

    raw_preds = trainer.predict(test_ds).predictions
    preds = raw_preds[0] if isinstance(raw_preds, tuple) else raw_preds
    y_pred = np.argmax(preds, axis=1)

    dataloader = finetune_mod.DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        collate_fn=finetune_mod.RemoveColumnsCollator(
            data_collator=finetune_mod.default_data_collator,
            signature_columns=["target_values"],
            logger=None,
            description=None,
            model_name="temp",
        ),
    )

    y_true = np.concatenate([batch["target_values"].numpy() for batch in dataloader])

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class_f1 = {k: v["f1-score"] for k, v in report.items() if k.isdigit()}

    cm = confusion_matrix(y_true, y_pred)

    latency_device = device
    if isinstance(latency_device, str) and latency_device.startswith("cuda:"):
        latency_device = "cuda:0"
    device_obj = torch.device(latency_device)
    latency = measure_latency(model, device_obj, context_length, tsp.num_input_channels)

    metrics.update(latency)
    metrics["params_total"] = sum(p.numel() for p in model.parameters())
    metrics["params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics["model_size_mb"] = model_size_mb(model)

    metrics["per_class_f1_json"] = json.dumps(per_class_f1)
    metrics["confusion_matrix_json"] = json.dumps(cm.tolist())
    metrics["variant"] = variant
    metrics["mode"] = mode
    metrics["dataset"] = dataset
    metrics["fold_subject"] = int(sid)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--hop_length", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_folds", type=int, default=1)
    parser.add_argument("--test_set_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    args = parser.parse_args()

    global OUTPUT_ROOT, RUNS_ROOT
    if args.output_root:
        OUTPUT_ROOT = Path(args.output_root)
        RUNS_ROOT = OUTPUT_ROOT / "runs"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    runtimes = collect_runtimes(LOGS_ROOT)
    selected, selected_id, selected_ood = select_datasets(runtimes)

    baseline_rows = []
    direct_rows = []

    test_set_dir = args.test_set_dir
    if not test_set_dir:
        default_test_dir = REPO_ROOT / "test_set"
        if default_test_dir.exists():
            test_set_dir = str(default_test_dir)

    for dataset in selected:
        print(f"[Pilot] Baseline training for {dataset}")
        run_finetune([
            "--dataset_name", dataset,
            "--checkpoint_path", args.checkpoint_path,
            "--device", args.device,
            "--context_length", str(args.context_length),
            "--hop_length", str(args.hop_length),
            "--output_dir", str(RUNS_ROOT),
            "--branch_prune_variant", "full",
            "--branch_pruning_mode", "standard",
            "--max_folds", str(args.max_folds),
            "--test_set_dir", test_set_dir,
        ])

        fold_dir = get_fold_output_dir(Path(args.checkpoint_path), 100, args.context_length, dataset, 1, "full", "standard")

        baseline_metrics = evaluate_checkpoint(
            fold_dir,
            Path(args.checkpoint_path),
            dataset,
            args.context_length,
            args.hop_length,
            args.device,
            "full",
            "direct_eval",
            test_set_dir,
        )
        baseline_metrics["split"] = "OOD" if dataset in selected_ood else "ID"
        baseline_rows.append(baseline_metrics)

        for variant in ["full", "fft_soft", "reg_r2_soft", "reg_r0_soft", "decoder1_soft", "no_channelmix_soft"]:
            print(f"[Pilot] Direct eval {dataset} variant={variant}")
            metrics = evaluate_checkpoint(
                fold_dir,
                Path(args.checkpoint_path),
                dataset,
                args.context_length,
                args.hop_length,
                args.device,
                variant,
                "direct_eval",
                test_set_dir,
            )
            metrics["split"] = "OOD" if dataset in selected_ood else "ID"
            direct_rows.append(metrics)

    baseline_df = pd.DataFrame(baseline_rows)
    direct_df = pd.DataFrame(direct_rows)

    baseline_df.to_csv(OUTPUT_ROOT / "pilot_baseline_efficiency.csv", index=False)
    direct_df.to_csv(OUTPUT_ROOT / "pilot_soft_knockout_direct_eval.csv", index=False)

    summary_path = OUTPUT_ROOT / "branch_pruning_pilot_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Branch Pruning Pilot Summary\n\n")
        f.write("## Selected datasets\n")
        f.write(f"- ID: {', '.join(selected_id)}\n")
        f.write(f"- OOD: {', '.join(selected_ood)}\n\n")
        f.write("## Baseline efficiency\n")
        f.write(baseline_df[["dataset", "split", "macro_f1", "params_total", "model_size_mb", "latency_mean_ms"]].to_markdown(index=False))
        f.write("\n\n## Direct soft knockout results\n")
        f.write(direct_df[["dataset", "split", "variant", "macro_f1", "latency_mean_ms"]].to_markdown(index=False))


if __name__ == "__main__":
    main()
