# HART architecture adaptation across sampling rates

This is the main ML experiment. The earlier `rate_reliability` experiment is
retained as a diagnostic only. No confidence gate selects predictions here.

Two small residual adapters modify HART's time and FFT encoding outputs before
its shared pretrained encoder. Each has a rank-eight bottleneck conditioned on
the known sampling rate. The original HART parameters stay frozen, but gradients
pass through the encoder to train the adapters. The output is one activity
classifier with per-channel patch mean/standard-deviation pooling.

The method is a research hypothesis, not a confirmed discovery. Its distinguishing
test is whether aligning only shared-frequency representations outperforms
aligning full representations when low-rate sensing has lost high frequencies.
Resampling augmentation, adapters, and consistency learning already have prior
work and are not claimed as independently new.

## Objective

For the clean recording x and lower-rate observation x_f, classification uses
both full signals. A common-band operator B_f keeps frequencies at most
0.8 times the lower-rate Nyquist frequency. The loss is

`0.5 CE(y, model(x,30)) + 0.5 CE(y, model(x_f,f))`

`+ lambda * (1 - cosine(h(B_f x,30), h(B_f x_f,f)))`.

The alignment inputs are additional training views. Inference uses the observed
signal with its known rate in one forward pass. There is no reconstruction claim
for frequency content that the sensor did not record.

## Controls

- `clean_head` trains the shared classifier on clean recordings only.
- `aug_head` trains that classifier on clean and lower-rate observations.
- `rate_adapter` adds the proposed adapters with classification loss only.
- `full_align` uses the same adapters, aligning entire clean/lower-rate representations.
- `shared_align` aligns only common-band views while classifying full signals.
- Optional `blind_adapter` replaces sampling-rate conditioning with zeros.

The adapter controls use identical model initialization, splits, step caps, optimizer,
and validation criterion. The pretrained encoder remains in evaluation mode to
disable stochastic masking and dropout. Checkpoint loading rejects missing
backbone weights. Only validation labels select the checkpoint.

## Corrected controlled study

The new loader applies the 90 percent purity and finite-value filters before
uniformly selecting the requested number of valid windows per subject. The old
pilot capped candidates first, then discarded unstable windows. Therefore, new
runs use a changed sample and must not be combined directly with the old pilot.
Validation and test windows remain uniformly sampled, not class-balanced.

The optional `--train-sampling sqrt-balanced` uses replacement sampling weighted
by inverse square-root training-class frequency. Every model receives the same
training policy. It neither consults validation/test labels nor guarantees that
rare activities appear in every held-out subject split. No classes are merged.

`--data-seed` and `--split-seed` keep window selection and subjects fixed when
the independent model `--seed` changes. Cached unnormalized windows are keyed
by loader source, data seed, window cap, and source file paths/sizes/mtimes.
Do not alter input files during a run. Existing run directories are protected
against overwrite. Protocol records also hash the experiment source files.

The six default controls include the rate-blind adapter. `per_class.csv` exposes
support and recall failures, and `per_subject.csv` keeps held-out subjects
separate. `macro_f1` always covers the fixed twelve-class ontology, including
absent classes with zero F1. `supported_macro_f1` is a separately labelled
diagnostic over classes present in that evaluation subset, not a replacement
for the primary metric. `contrasts.csv` now includes both rate-adapter and
shared-alignment comparisons. Positive gains favor the proposed column.

`training_summary.csv` reports best epoch and the reason training stopped.
Best-at-final-epoch flags can indicate that a capped run has not converged.
`training.csv` additionally reports the weighted alignment objective and its
gradient norm relative to classification, measured on the first batch of each
epoch over adapter parameters. A small loss alone does not prove a weak gradient.
The alignment weight is unchanged; any sensitivity choices must use validation,
not the held-out-rate test results.

## Replication after controlled_v3

Keep the training configuration fixed rather than tuning it against the inspected
test set. Repeat model seeds 17 and 27 with the same data/split seeds as seed 7.
Retain all six methods, including negative alignment controls. The added
`--checkpoint-audit` saves and evaluates the last trained checkpoint separately
from the validation-selected one. Actual stopping steps are recorded and must
match before calling this an equal-budget comparison. `--rate-audit` compares
correct rate metadata with constant 30 Hz metadata, without retraining or changing
the observed signal. The blind control should be invariant. Neither audit chooses
a checkpoint or feature using test labels.

```bash
for seed in 17 27; do
  python experiments/rate_adaptation/run.py --data datasets/harth_raw --cache datasets/window_cache --windows-per-subject 200 --epochs 12 --max-steps 600 --patience 4 --batch-size 32 --threads 4 --train-sampling sqrt-balanced --device mps --seed "$seed" --data-seed 7 --split-seed 7 --checkpoint-audit --rate-audit --output "outputs_rate_adaptation/replication_seed_$seed" || break
done

python experiments/rate_adaptation/report.py --runs outputs_rate_adaptation/controlled_v3 outputs_rate_adaptation/replication_seed_17 outputs_rate_adaptation/replication_seed_27 --output outputs_rate_adaptation/replication_report
```

The report checks split, data, architecture, training hyperparameters, completeness,
and distinct model seeds. New reporting-only changes to `run.py` are allowed so
the saved controlled_v3 run can be retained. The report includes all conditions
and separates pooled metric seed variability from bootstrap intervals for mean
per-subject paired gains. With four test subjects, those intervals are descriptive.
Audits are available for the two new seeds only, recorded in `audit_coverage.json`.
Three initializations on one split do not establish generalization to new subjects.
Do not discard unfavorable seeds or reinterpret paired rate views as new samples.

HARTH subject files and the installation environment are the same as for the
earlier pilot. Training subjects now include both former probe and gate groups.
Training rates are 30, 15, and 10 Hz. Test-only rates 12 and 6 Hz examine
interpolation and extrapolation. Naive decimation is an additional stress test.
All windows retain four-second duration, sensor identities, and original labels.

## Commands

From `/Users/boolu/Desktop/simulations/hart_rate_reliability`:

```bash
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m unittest discover -s experiments/rate_adaptation -p 'test_*.py'
python experiments/rate_adaptation/run.py --smoke-test
python experiments/rate_adaptation/run.py --data datasets/harth_raw --windows-per-subject 40 --epochs 5 --max-steps 80 --batch-size 32 --output outputs_rate_adaptation/pilot
```

Use a new directory for the corrected bounded comparison:

```bash
python experiments/rate_adaptation/run.py --data datasets/harth_raw --cache datasets/window_cache --windows-per-subject 200 --epochs 12 --max-steps 600 --patience 4 --batch-size 32 --train-sampling sqrt-balanced --device mps --seed 7 --data-seed 7 --split-seed 7 --output outputs_rate_adaptation/controlled_v3
```

For a larger comparison after fixing the pilot configuration:

```bash
for seed in 7 17 27; do
  python experiments/rate_adaptation/run.py --data datasets/harth_raw --cache datasets/window_cache --windows-per-subject 200 --epochs 12 --max-steps 600 --patience 4 --batch-size 32 --train-sampling sqrt-balanced --seed "$seed" --data-seed 7 --split-seed 7 --output "outputs_rate_adaptation/controlled_seed_$seed"
done
```

The step cap bounds training per model. `training.csv` updates each epoch and
`metrics.csv` updates after every completed model. Saved model checkpoints,
`predictions.csv`, `protocol.json`, and `contrasts.csv` support reproducibility.
Positive contrast values favor shared-band alignment. Report negative results
and class counts too. Adjacent sensor windows and paired rate views are not
independent observations; uncertainty should be assessed by held-out subject.
Measure device latency before making efficiency claims. Freezing parameters
reduces trainable size but does not remove encoder-backpropagation cost.
