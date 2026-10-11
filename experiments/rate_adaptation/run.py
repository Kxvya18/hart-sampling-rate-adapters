"""Architecture and representation-learning controls for sample-rate adaptation."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.rate_reliability.data import HARTH_LABELS, load_harth, split_subjects
from model import AdaptedHART, alignment_loss
from views import rate_view, shared_band
from diagnostics import class_diagnostics, training_order
from support_model import SupportHART


METHODS = ("clean_head", "aug_head", "rate_adapter", "full_align", "shared_align", "blind_adapter",
           "support_adapter", "support_projection", "support_context",
           "residual_support", "residual_support_blind")
CONDITIONS = {"clean30": (30, True), "aa15": (15, True), "aa10": (10, True),
              "unseen12": (12, True), "unseen6": (6, True), "naive15": (15, False), "naive10": (10, False)}


def build_model(args, channels, classes, method, device):
    from tsfm_public.models.tspulse import TSPulseForClassification
    loaded, loading = TSPulseForClassification.from_pretrained(
        str(args.checkpoint), num_input_channels=channels, num_targets=classes,
        context_length=120, loss="cross_entropy", channel_virtual_expand_scale=1,
        disable_mask_in_classification_eval=True, fft_time_consistent_masking=True,
        head_reduce_d_model=1, decoder_mode="common_channel", head_aggregation_dim="patch",
        ignore_mismatched_sizes=True, output_loading_info=True,
    )
    problems = [key for key in loading.get("missing_keys", []) if key.startswith("backbone.")]
    problems += [str(item) for item in loading.get("mismatched_keys", [])
                 if (item[0] if isinstance(item, (tuple, list)) else str(item)).startswith("backbone.")]
    if problems:
        raise RuntimeError(f"Pretrained backbone did not fully load: {problems}")
    if method.startswith("residual_support"):
        return SupportHART(loaded.backbone, channels, classes, args.rank,
                           project=False, contextual=method != "residual_support_blind",
                           residual_support=True).to(device)
    if method.startswith("support_"):
        return SupportHART(loaded.backbone, channels, classes, args.rank,
                           project=method != "support_context",
                           contextual=method != "support_projection").to(device)
    return AdaptedHART(loaded.backbone, channels, classes, args.rank,
                       method not in ("clean_head", "aug_head"), method != "blind_adapter").to(device)


def score(y, p, classes):
    pred = classes[p.argmax(axis=1)]
    confidence = p.max(axis=1)
    correct = pred == y
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(np.mean(bins == k) * abs(correct[bins == k].mean() - confidence[bins == k].mean())
              for k in range(10) if np.any(bins == k))
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
            "supported_macro_f1": float(f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, labels=classes, average="weighted", zero_division=0)),
            "supported_classes": int(len(np.unique(y))),
            "nll": float(log_loss(y, p, labels=classes)), "ece10": float(ece),
            "high_conf_error_fraction": float(np.mean((confidence >= .8) & ~correct))}


def predict(model, values, rate, batch_size, device):
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start:start + batch_size], device=device)
            rates = torch.full((len(batch),), float(rate), device=device)
            logits, _ = model(batch, rates)
            probabilities.append(logits.softmax(dim=-1).cpu().numpy())
    return np.concatenate(probabilities)


def fit(model, args, method, arrays, indices, y_index, labels, classes, device, log_rows):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-3)
    rng = np.random.default_rng(args.seed)
    train = indices["train"]
    best_loss, best, wait, steps, best_epoch = float("inf"), None, 0, 0, 0
    clock_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = training_order(train, labels, rng, getattr(args, "train_sampling", "uniform"))
        ce_values, align_values = [], []
        gradient_ratio = 0.
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            targets = torch.as_tensor(y_index[ids], device=device)
            clean = torch.as_tensor(arrays["clean30"][ids], device=device)
            optimizer.zero_grad()
            if method == "clean_head":
                logits, _ = model(clean, torch.full((len(ids),), 30., device=device))
                ce = F.cross_entropy(logits, targets)
                alignment = torch.zeros((), device=device)
            else:
                rate = int(rng.choice((15, 10)))
                lower = torch.as_tensor(arrays[f"aa{rate}"][ids], device=device)
                rates = torch.cat((torch.full((len(ids),), 30., device=device),
                                   torch.full((len(ids),), float(rate), device=device)))
                logits, representation = model(torch.cat((clean, lower)), rates)
                ce = F.cross_entropy(logits, targets.repeat(2))
                alignment = torch.zeros((), device=device)
                if method == "full_align":
                    alignment = alignment_loss(*representation.chunk(2))
                elif method == "shared_align":
                    common = torch.cat((torch.as_tensor(arrays[f"common_clean{rate}"][ids], device=device),
                                        torch.as_tensor(arrays[f"common_low{rate}"][ids], device=device)))
                    _, common_representation = model(common, rates)
                    alignment = alignment_loss(*common_representation.chunk(2))
            loss = ce + args.align_weight * alignment
            if start == 0 and method in ("full_align", "shared_align"):
                adapter_parameters = [p for p in model.adapters.parameters() if p.requires_grad]
                ce_grad = torch.autograd.grad(ce, adapter_parameters, retain_graph=True, allow_unused=True)
                align_grad = torch.autograd.grad(args.align_weight * alignment, adapter_parameters,
                                                 retain_graph=True, allow_unused=True)
                ce_norm = torch.sqrt(sum(g.square().sum() for g in ce_grad if g is not None))
                align_norm = torch.sqrt(sum(g.square().sum() for g in align_grad if g is not None))
                gradient_ratio = float((align_norm / ce_norm.clamp_min(1e-12)).detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            ce_values.append(float(ce.detach().cpu()))
            align_values.append(float(alignment.detach().cpu()))
            steps += 1
            if steps >= args.max_steps:
                break
        validation = []
        for condition, rate in (("clean30", 30), ("aa15", 15), ("aa10", 10)):
            val_ids = indices["val"]
            p = predict(model, arrays[condition][val_ids], rate, args.batch_size, device)
            validation.append(log_loss(labels[val_ids], p, labels=classes))
        val_loss = float(np.mean(validation))
        row = {"method": method, "epoch": epoch + 1, "steps": steps,
               "train_ce": float(np.mean(ce_values)), "alignment": float(np.mean(align_values)),
               "weighted_alignment": args.align_weight * float(np.mean(align_values)),
               "alignment_to_ce_gradient_ratio_first_batch": gradient_ratio,
               "validation_nll": val_loss, "elapsed_seconds": time.perf_counter() - clock_start}
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(args.output / "training.csv", index=False)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
        if steps >= args.max_steps or wait >= args.patience:
            break
    if getattr(args, "checkpoint_audit", False):
        final = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save({"model": final, "method": method, "validation_nll": val_loss,
                    "seed": args.seed}, args.output / f"{method}_final.pt")
    model.load_state_dict(best)
    torch.save({"model": best, "method": method, "validation_nll": best_loss,
                "seed": args.seed}, args.output / f"{method}.pt")
    reason = "max_steps" if steps >= args.max_steps else "patience" if wait >= args.patience else "epochs"
    return {"method": method, "steps": steps, "epochs": epoch + 1, "best_epoch": best_epoch,
            "best_validation_nll": best_loss, "stop_reason": reason,
            "final_validation_nll": val_loss,
            "best_at_final_epoch": best_epoch == epoch + 1,
            "elapsed_seconds": time.perf_counter() - clock_start}


def load_windows(args):
    sources = sorted(args.data.rglob("*.csv"))
    provenance = {"files": [{"path": str(p.resolve()), "bytes": p.stat().st_size,
                              "mtime_ns": p.stat().st_mtime_ns} for p in sources],
                  "data_seed": args.data_seed, "windows_per_subject": args.windows_per_subject,
                  "loader_sha256": hashlib.sha256((REPO / "experiments/rate_reliability/data.py").read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache = args.cache / f"windows_{fingerprint}.npz" if args.cache is not None else None
    if cache is not None and cache.is_file():
        print(f"Loading cached windows from {cache}", flush=True)
        with np.load(cache, allow_pickle=False) as saved:
            return saved["x"], saved["labels"], saved["groups"], provenance
    x, labels, groups = load_harth(args.data, 0, args.windows_per_subject, args.data_seed)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".part.npz")
        np.savez_compressed(temporary, x=x, labels=labels, groups=groups)
        temporary.replace(cache)
    return x, labels, groups, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--expected-subjects", type=int, default=22,
                        help="Required unique HARTH CSV subjects; fail on incomplete extraction")
    parser.add_argument("--smoke-test", action="store_true", help="Check real checkpoint and adapter gradients without a dataset")
    parser.add_argument("--smoke-method", choices=METHODS, default="shared_align")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "finetuning_HART/pretrained_checkpoint")
    parser.add_argument("--output", type=Path, default=REPO / "outputs_rate_adaptation/pilot")
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS[:6]))
    parser.add_argument("--windows-per-subject", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=80, help="Hard training-step cap per method")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--align-weight", type=float, default=.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-seed", type=int, default=7, help="Fixed window selection seed, independent of model seed")
    parser.add_argument("--split-seed", type=int, default=7, help="Fixed subject split seed, independent of model seed")
    parser.add_argument("--train-sampling", choices=("uniform", "sqrt-balanced"), default="uniform")
    parser.add_argument("--cache", type=Path, help="Cache unnormalized windows for subsequent runs")
    parser.add_argument("--checkpoint-audit", action="store_true",
                        help="Also evaluate final-step checkpoints, without selecting them on test results")
    parser.add_argument("--rate-audit", action="store_true",
                        help="Compare correct rate metadata with constant 30 Hz metadata at inference")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.max_steps, args.batch_size, args.rank, args.threads) <= 0:
        parser.error("epochs, max-steps, batch-size, rank, and threads must be positive")
    if args.windows_per_subject < 0 or args.align_weight < 0 or args.patience <= 0:
        parser.error("window cap and alignment weight must be nonnegative; patience must be positive")
    if len(args.methods) != len(set(args.methods)):
        parser.error("methods must not contain duplicates")
    if not args.smoke_test and (args.output / "protocol.json").exists():
        parser.error("output already contains a run; choose a new directory to preserve previous results")
    if not args.smoke_test and args.data is not None:
        subjects = {p.stem for p in args.data.rglob("S*.csv")}
        if len(subjects) != args.expected_subjects:
            parser.error(f"Expected {args.expected_subjects} subjects, found {len(subjects)}. Restore dataset before training.")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --device cpu")
    if args.smoke_test:
        torch.manual_seed(args.seed)
        model = build_model(args, 6, len(HARTH_LABELS), args.smoke_method, device).train()
        values = torch.randn(2, 120, 6, device=device)
        logits, embedding = model(values, torch.tensor([30., 10.], device=device))
        F.cross_entropy(logits, torch.tensor([0, 1], device=device)).backward()
        if not all(p.grad is None for p in model.backbone.parameters()):
            raise RuntimeError("Frozen HART weights received gradients")
        adapter_gradient = model.adapters["time"].up.weight.grad
        gradient = adapter_gradient.abs().sum().item() if adapter_gradient is not None else 0.
        if gradient <= 0:
            raise RuntimeError("Adapter received no training signal")
        print(json.dumps({"smoke_test": "PASS", "device": str(device),
                          "logits_shape": list(logits.shape), "embedding_shape": list(embedding.shape),
                          "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                          "adapter_gradient": gradient}), flush=True)
        model.close()
        return
    if args.data is None:
        parser.error("--data is required unless --smoke-test is used")
    x, labels, groups, data_provenance = load_windows(args)
    subject_splits, _ = split_subjects(groups, args.split_seed)
    subjects = {"train": subject_splits["probe"] + subject_splits["gate"],
                "val": subject_splits["val"], "test": subject_splits["test"]}
    indices = {part: np.flatnonzero(np.isin(groups, ids)) for part, ids in subjects.items()}
    classes = np.asarray(HARTH_LABELS)
    if not set(labels).issubset(set(classes)):
        raise ValueError("Unexpected HARTH activity labels")
    y_index = np.searchsorted(classes, labels)
    mean = x[indices["train"]].mean(axis=(0, 1), keepdims=True)
    std = np.maximum(x[indices["train"]].std(axis=(0, 1), keepdims=True), 1e-5)
    x = ((x - mean) / std).astype(np.float32)
    arrays = {condition: rate_view(x, rate, aa) for condition, (rate, aa) in CONDITIONS.items()}
    for rate in (15, 10):
        arrays[f"common_clean{rate}"] = shared_band(x, rate)
        arrays[f"common_low{rate}"] = shared_band(arrays[f"aa{rate}"], rate)
    protocol = {"source_repo": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "checkpoint_sha256": hashlib.sha256((args.checkpoint / "model.safetensors").read_bytes()).hexdigest(),
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "dataset_provenance": data_provenance,
                "sampling": "Uniform valid windows per subject; balancing applies only to training batches",
                "experiment_source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                             for name in ("run.py", "model.py", "views.py", "diagnostics.py", "support_model.py")},
                "support_method_scope": "Projection acts on normalized Fourier patch inputs; it does not recover missing or aliased signal content",
                "subjects": subjects, "train_rates": [30, 15, 10], "held_out_rates": [12, 6],
                "shared_cutoff": "0.8 times lower-rate Nyquist", "normalization": "training subjects only",
                "class_counts": {part: {str(c): int(np.sum(labels[idx] == c)) for c in classes}
                                 for part, idx in indices.items()}, "device": str(device)}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for part, counts in protocol["class_counts"].items():
        absent = [label for label, count in counts.items() if count == 0]
        print(f"{part}: {len(indices[part])} windows, absent classes={absent}", flush=True)
    rows, training, predictions, per_class, per_subject, summaries = [], [], [], [], [], []
    final_rows, final_subjects, rate_audit = [], [], []
    for method in args.methods:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = build_model(args, x.shape[-1], len(classes), method, device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"method={method} device={device} trainable={trainable}", flush=True)
        summaries.append(fit(model, args, method, arrays, indices, y_index, labels, classes, device, training))
        pd.DataFrame(summaries).to_csv(args.output / "training_summary.csv", index=False)
        test_ids = indices["test"]
        for condition, (rate, _) in CONDITIONS.items():
            p = predict(model, arrays[condition][test_ids], rate, args.batch_size, device)
            rows.append({"method": method, "condition": condition, "trainable_parameters": trainable,
                         **score(labels[test_ids], p, classes)})
            if args.rate_audit:
                constant = p if rate == 30 else predict(model, arrays[condition][test_ids], 30,
                                                       args.batch_size, device)
                correct_metrics = score(labels[test_ids], p, classes)
                constant_metrics = score(labels[test_ids], constant, classes)
                rate_audit.append({"method": method, "condition": condition,
                                   "correct_rate": rate, "constant_rate": 30,
                                   "prediction_disagreement_fraction": float(np.mean(p.argmax(1) != constant.argmax(1))),
                                   "mean_probability_l1_change": float(np.abs(p - constant).sum(1).mean()),
                                   "correct_metadata_accuracy_gain": correct_metrics["accuracy"] - constant_metrics["accuracy"],
                                   "correct_metadata_f1_gain": correct_metrics["macro_f1"] - constant_metrics["macro_f1"],
                                   "correct_metadata_nll_reduction": constant_metrics["nll"] - correct_metrics["nll"]})
            per_class.extend({"method": method, "condition": condition, **r}
                             for r in class_diagnostics(labels[test_ids], p, classes))
            for subject in subjects["test"]:
                selected = groups[test_ids] == subject
                per_subject.append({"method": method, "condition": condition, "subject": subject,
                                    **score(labels[test_ids][selected], p[selected], classes)})
            for j, idx in enumerate(test_ids):
                predictions.append({"method": method, "condition": condition, "subject": groups[idx],
                                    "window_index": int(idx), "true": int(labels[idx]),
                                    "predicted": int(classes[p[j].argmax()]), "confidence": float(p[j].max()),
                                    "p_true": float(p[j, y_index[idx]])})
        pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
        pd.DataFrame(predictions).to_csv(args.output / "predictions.csv", index=False)
        pd.DataFrame(per_class).to_csv(args.output / "per_class.csv", index=False)
        pd.DataFrame(per_subject).to_csv(args.output / "per_subject.csv", index=False)
        if args.rate_audit:
            pd.DataFrame(rate_audit).to_csv(args.output / "rate_metadata_audit.csv", index=False)
        if args.checkpoint_audit:
            saved = torch.load(args.output / f"{method}_final.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(saved["model"])
            for condition, (rate, _) in CONDITIONS.items():
                p = predict(model, arrays[condition][test_ids], rate, args.batch_size, device)
                final_rows.append({"method": method, "condition": condition,
                                   "trainable_parameters": trainable, **score(labels[test_ids], p, classes)})
                for subject in subjects["test"]:
                    selected = groups[test_ids] == subject
                    final_subjects.append({"method": method, "condition": condition, "subject": subject,
                                           **score(labels[test_ids][selected], p[selected], classes)})
            pd.DataFrame(final_rows).to_csv(args.output / "final_metrics.csv", index=False)
            pd.DataFrame(final_subjects).to_csv(args.output / "final_per_subject.csv", index=False)
            del saved
        print(pd.DataFrame(rows[-len(CONDITIONS):]).to_string(index=False), flush=True)
        model.close()
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    result = pd.DataFrame(rows).set_index(["method", "condition"])
    comparisons = []
    for proposed_method in ("rate_adapter", "shared_align", "support_adapter"):
        if proposed_method not in args.methods:
            continue
        for baseline in args.methods:
            if baseline == proposed_method:
                continue
            for condition in CONDITIONS:
                proposed, reference = result.loc[(proposed_method, condition)], result.loc[(baseline, condition)]
                comparisons.append({"proposed": proposed_method, "baseline": baseline, "condition": condition,
                                    "f1_gain": float(proposed.macro_f1 - reference.macro_f1),
                                    "supported_f1_gain": float(proposed.supported_macro_f1 - reference.supported_macro_f1),
                                    "accuracy_gain": float(proposed.accuracy - reference.accuracy),
                                    "nll_reduction": float(reference.nll - proposed.nll)})
    pd.DataFrame(comparisons).to_csv(args.output / "contrasts.csv", index=False)
    print(f"Saved architecture study to {args.output}", flush=True)


if __name__ == "__main__":
    main()
