# Interpretation

Positive gains favor the proposed method. NLL gains are reductions.

Pooled gains and model-seed standard deviations summarize repeated training on a fixed split. Subject bootstrap intervals instead apply to mean per-subject paired gains after averaging the model seeds. They are not confidence intervals for pooled macro-F1. Resampling the few held-out subjects is descriptive, not evidence across independent datasets.

Conditions reuse the same windows and must not be treated as independent samples. Three trained seeds test initialization stability, not subject-split generalization. The previously inspected seed 7 is exploratory. New seeds replicate the unchanged protocol but are not a fresh unseen-subject confirmation.

Final-checkpoint and metadata audits include only runs that produced them. audit_coverage.json lists their seeds explicitly. Do not equate two-seed audits with the three-seed primary comparison. Check actual steps in training_summary.csv before treating final checkpoints as equal-budget controls; early stopping can make budgets different. The final policy is diagnostic, not test-selected. run.py hash differences may reflect new reporting options and should be inspected; architecture and training settings are checked. Rare or absent classes remain explicit in per_class.csv.
