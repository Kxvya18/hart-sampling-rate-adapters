# Technical interpretation of the completed ML study

## What changed in HART

The released time-series HART backbone remains frozen. Two rank-eight residual adapters are relocated from the temporal and FFT encodings to the corresponding token positions after the final encoder layer. Those late positions contain mixed contextual representations, not isolated domains. Register tokens are not adapted. Mean and population-standard-deviation pooling and the classifier are shared across methods.

Early and late models declare 9948 trainable parameters. The unconditioned implementation has 32 dormant context weights, so both have 9916 effective active parameters. Head-only and last-block controls have different budgets and are not factorial controls.

The paired objective is cross-entropy on reference and degraded views plus 0.1 times temperature-scaled KL with temperature 2. The reference target is detached only inside KL. It is an online view of the same evolving model, not a separate fixed teacher. Both branches still receive cross-entropy gradients. Reverse KD swaps student and target view.

## Main finding

The revised manuscript also reports the saved final-600-update audit. Confirmation late KD gains 6.7024 F1 points over early CE and has a 2.3172-point interaction, each positive in three seeds. Final-step interactions are positive in seven of nine runs across partitions, compared with eight of nine validation-selected runs. Checkpoint selection is therefore disclosed rather than conflated with fixed checkpoint age. Supplementary CSVs preserve every final-step condition and seed.

The primary contrast is `(late KD - late CE) - (early KD - early CE)` in twelve-label macro-F1. At 6 Hz, its partition means are 2.8103, 3.8624, and 2.7598 percentage points. It is positive in eight of nine split-seed pairs, including all three confirmation seeds.

Confirmation late KD minus early CE is 6.0353 F1 points, positive in all three seeds. Most of that difference is already explained by relocating CE adaptation, which gives 4.6999 points. The additional late-KD gain is 1.3354 points, while early KD loses 1.4244 points. These quantities justify an architecture-objective interaction rather than attributing the entire gain to distillation.

At 12 Hz, the confirmation endpoint gain is only 0.1777 F1 points. Incremental KD benefits are inconsistent across exploratory partitions. Clean-rate late KD is below late CE on confirmation. Reverse KD is competitive and wins F1 in one confirmation seed. Forward distillation direction is therefore not established as necessary.

## Evaluation scope

HARTH provides real 50 Hz recordings. Acquisition-rate shifts are anti-aliased resampling transformations followed by interpolation, not separately recorded low-rate devices. Training rates are 30, 15, and 10 Hz. Unseen test rates are 12 and 6 Hz.

The confirmation test roster excludes every previously tested participant but can include people used in earlier exploratory training. Backbone pretraining membership remains unknown. Confirmation has four participants and 3452 windows. All three seeds evaluate those same windows. They do not constitute three independent datasets. The exploratory test rosters overlap.

Fixed-twelve-label F1 assigns zero contributions to absent labels. Exploratory partitions support eleven and nine labels, and confirmation supports all twelve. Cross-partition absolute scores are not directly comparable. Support-threshold audits change which class scores are averaged without changing predictions. Participant-level F1 is nonlinear and does not average to pooled F1.

## Figures and tables

The four monochrome figures each have four quantitative panels. No panel uses a flowchart or artificial data. Window distributions mostly average seed losses before visualization. Figure 2d retains separate seed distributions. Class violins use density smoothing, not invented samples. Display ranges and heatmap limits can clip extreme values, as documented in captions and figure notes.

Figure 1d uses an NLL interaction, which is not the macro-F1 interaction used in the main claim. Training KL can rise as validation NLL improves. This is not proof of a decreasing disagreement mechanism. Confusion and reliability panels pool repeated predictions descriptively and do not establish statistical calibration.

The CSV tables are regenerated directly from nine saved runs. `source_manifest.json` contains source file hashes. Aggregate CSVs include negative controls and all saved conditions, not only favorable rows. Means and standard deviations characterize three model seeds, not population confidence intervals or significance tests.

The revised acquisition-control paragraph reports confirmation late-KD gains of 3.7862 and 0.5856 F1 points at anti-aliased 15 and 10 Hz, and -0.0206 points under naive 10 Hz decimation. These do not establish general aliasing robustness. Activity 140 has one training and one validation window. The added class-support supplement documents this severe sparsity. Per-channel normalization uses training-only statistics with a 1e-5 standard-deviation floor, and window selection uses seed 7.

## Limits of novelty and efficiency

Residual adaptation, online distillation, resampling augmentation, and HART are existing ingredients. The proposed contribution is their controlled architecture-objective interaction under severe rate shift, with a low-parameter late-adaptation intervention. Encoder-Jacobian reasoning motivates a hypothesis but is not a measured mechanistic discovery.

The frozen backbone still runs at inference. Training uses two forward views. No latency, energy, memory, or edge-device speedup was measured. No state-of-the-art or journal-readiness claim is justified from this single-dataset study. Additional datasets, acquired low-rate recordings, stronger tuning baselines, and pretraining audits are the most important extensions.
