"""Matched-seed summaries with descriptive held-out-subject uncertainty."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIRS = (("rate_adapter", "blind_adapter"), ("rate_adapter", "aug_head"),
         ("shared_align", "rate_adapter"), ("shared_align", "full_align"),
         ("support_adapter", "blind_adapter"), ("support_adapter", "rate_adapter"),
         ("support_adapter", "support_projection"), ("support_adapter", "support_context"),
         ("residual_support_blind", "blind_adapter"),
         ("residual_support_blind", "rate_adapter"),
         ("residual_support", "residual_support_blind"),
         ("residual_support", "rate_adapter"))
METRICS = ("accuracy", "macro_f1", "nll")
MATCH_ARGS = ("data_seed", "split_seed", "windows_per_subject", "epochs", "max_steps",
              "batch_size", "rank", "lr", "align_weight", "patience", "train_sampling")


def load_matched_runs(paths, min_seeds):
    runs, reference, seeds = [], None, set()
    for path in paths:
        protocol = json.loads((path / "protocol.json").read_text())
        args = protocol["arguments"]
        seed = args["seed"]
        if seed in seeds:
            raise ValueError(f"Repeated model seed {seed}; repeats are not independent trained seeds")
        seeds.add(seed)
        signature = {"arguments": {k: args[k] for k in MATCH_ARGS},
                     "subjects": protocol["subjects"], "dataset": protocol["dataset_provenance"],
                     "checkpoint": protocol["checkpoint_sha256"], "methods": sorted(args["methods"]),
                     "architecture": {k: v for k, v in protocol["experiment_source_sha256"].items()
                                      if k != "run.py"}}
        if reference is not None and signature != reference:
            raise ValueError("Runs differ in training protocol, data, subjects, experts, or architecture")
        reference = signature
        metrics = pd.read_csv(path / "metrics.csv")
        subjects = pd.read_csv(path / "per_subject.csv")
        expected = {(m, c) for m in args["methods"] for c in
                    ("clean30", "aa15", "aa10", "unseen12", "unseen6", "naive15", "naive10")}
        if metrics.duplicated(["method", "condition"]).any() or set(zip(metrics.method, metrics.condition)) != expected:
            raise ValueError(f"Incomplete or duplicate metric rows in {path}")
        expected_subjects = {(m, c, s) for m, c in expected for s in protocol["subjects"]["test"]}
        if subjects.duplicated(["method", "condition", "subject"]).any() or \
                set(zip(subjects.method, subjects.condition, subjects.subject)) != expected_subjects:
            raise ValueError(f"Incomplete or duplicate subject rows in {path}")
        runs.append((path, protocol, metrics.assign(seed=seed), subjects.assign(seed=seed)))
    if len(seeds) < min_seeds:
        raise ValueError(f"Need at least {min_seeds} distinct model seeds, found {len(seeds)}")
    return runs


def paired_summary(metrics, subjects, bootstrap, seed):
    records = []
    for proposed, baseline in PAIRS:
        if not {proposed, baseline}.issubset(set(metrics.method)):
            continue
        pooled = metrics[metrics.method == proposed].merge(
            metrics[metrics.method == baseline], on=["seed", "condition"], suffixes=("_p", "_b"))
        paired = subjects[subjects.method == proposed].merge(
            subjects[subjects.method == baseline], on=["seed", "condition", "subject"], suffixes=("_p", "_b"))
        for condition, frame in paired.groupby("condition", sort=False):
            overall = pooled[pooled.condition == condition]
            rng = np.random.default_rng(seed)
            subjects_order = sorted(frame.subject.unique())
            draws = rng.integers(0, len(subjects_order), (bootstrap, len(subjects_order)))
            for metric in METRICS:
                sign = -1 if metric == "nll" else 1
                values = sign * (frame[f"{metric}_p"] - frame[f"{metric}_b"])
                subject_gain = frame.assign(gain=values).groupby("subject").gain.mean().reindex(subjects_order)
                resamples = subject_gain.to_numpy()[draws].mean(axis=1)
                seed_gain = sign * (overall[f"{metric}_p"] - overall[f"{metric}_b"])
                lo, hi = np.percentile(resamples, [2.5, 97.5])
                records.append({"proposed": proposed, "baseline": baseline, "condition": condition,
                                "metric": metric, "pooled_gain_seed_mean": float(seed_gain.mean()),
                                "pooled_gain_seed_sd": float(seed_gain.std(ddof=1)) if len(seed_gain) > 1 else 0.,
                                "positive_seeds": int((seed_gain > 0).sum()), "seeds": len(seed_gain),
                                "mean_subject_gain": float(subject_gain.mean()),
                                "subject_bootstrap_lo": float(lo), "subject_bootstrap_hi": float(hi),
                                "positive_subjects": int((subject_gain > 0).sum()), "subjects": len(subject_gain)})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--min-seeds", type=int, default=3)
    args = parser.parse_args()
    if args.bootstrap <= 0 or args.min_seeds <= 0:
        parser.error("bootstrap and minimum seeds must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("report output is not empty; choose a new directory")
    runs = load_matched_runs(args.runs, args.min_seeds)
    metrics = pd.concat([r[2] for r in runs], ignore_index=True)
    subjects = pd.concat([r[3] for r in runs], ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "seed_metrics.csv", index=False)
    aggregate = {f"{m}_{stat}": (m, stat) for m in METRICS for stat in ("mean", "std")}
    metrics.groupby(["method", "condition"]).agg(**aggregate).reset_index().to_csv(
        args.output / "model_summary.csv", index=False)
    paired = paired_summary(metrics, subjects, args.bootstrap, 2026)
    paired.to_csv(args.output / "paired_summary.csv", index=False)
    audit_coverage = {}
    for filename in ("final_metrics.csv", "final_per_subject.csv", "rate_metadata_audit.csv", "training_summary.csv", "per_class.csv"):
        available = [r for r in runs if (r[0] / filename).exists()]
        audit_coverage[filename] = [r[1]["arguments"]["seed"] for r in available]
        if available:
            pd.concat([pd.read_csv(r[0] / filename).assign(seed=r[1]["arguments"]["seed"])
                       for r in available], ignore_index=True).to_csv(args.output / filename, index=False)
    (args.output / "audit_coverage.json").write_text(json.dumps(audit_coverage, indent=2))
    (args.output / "interpretation.md").write_text(
        "# Interpretation\n\nPositive gains favor the proposed method. NLL gains are reductions.\n\n"
        "Pooled gains and model-seed standard deviations summarize repeated training on a fixed split. "
        "Subject bootstrap intervals instead apply to mean per-subject paired gains after averaging "
        "the model seeds. They are not confidence intervals for pooled macro-F1. "
        "Resampling the few held-out subjects is descriptive, not evidence across independent datasets.\n\n"
        "Conditions reuse the same windows and must not be treated as independent samples. "
        "Three trained seeds test initialization stability, not subject-split generalization. "
        "The previously inspected seed 7 is exploratory. New seeds replicate the unchanged protocol "
        "but are not a fresh unseen-subject confirmation.\n\n"
        "Final-checkpoint and metadata audits include only runs that produced them. "
        "audit_coverage.json lists their seeds explicitly. Do not equate two-seed audits with the "
        "three-seed primary comparison. "
        "Check actual steps in training_summary.csv before treating final checkpoints as "
        "equal-budget controls; early stopping can make budgets different. "
        "The final policy is diagnostic, not test-selected. run.py hash differences may reflect new "
        "reporting options and should be inspected; architecture and training settings are checked. "
        "Rare or absent classes remain explicit in per_class.csv.\n")
    view = paired[(paired.proposed == "rate_adapter") & (paired.baseline == "blind_adapter")]
    print(view.to_string(index=False))
    print(f"Saved matched-seed report to {args.output}")


if __name__ == "__main__":
    main()
