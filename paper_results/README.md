# Paper results: contextualization-aware adapter placement

Raw aggregated results behind the paper "Contextualization-Aware Adapter Placement
for Sampling-Rate Transfer in Wearable Machine Learning".

## Method names

| Paper label | Method name in the CSVs |
|---|---|
| Head only | `aug_head` |
| Early CE | `blind_adapter` |
| Early KD | `early_distill` |
| Late CE | `late_adapter` |
| Late KD | `late_distill` |
| Reverse KD | `late_reverse_distill` |
| Last block | `last_block` |

Splits 7 and 17 are exploratory (800 test windows). Split 37 is the confirmation
split (3,452 test windows; participants S015, S023, S025, S029). Seeds are 7, 17 and 27.
Conditions: `clean30`, `aa15`, `aa10` (training rates), `unseen12`, `unseen6`
(held out), `naive15`, `naive10` (aliasing controls).

## Files

- `paper_tables/all_metrics.csv`: per run metrics (accuracy, macro-F1, NLL, ECE) for every method, condition, split and seed.
- `paper_tables/all_per_subject.csv`, `all_per_class.csv`: participant and class breakdowns.
- `paper_tables/all_training_summary.csv`, `all_training.csv`: checkpoint selection and learning curves.
- `paper_tables/supplement_*.csv`: final-checkpoint contrasts, rate controls and class support.
- `paper_tables/table*.tex|csv`: Tables I-V of the paper, generated from the CSVs above.
- `paper_tables/source_manifest.json`: SHA-256 of every source run file.
- `scripts/`: table and figure export scripts. They expect the run folders under `outputs_rate_adaptation/`.
- `confirmation_split.json`, `CONFIRMATION_PLAN.md`: the confirmation roster and plan.
