# HART rate-reliability pilot

This experiment uses HART's released pretrained weights, not new pretraining. The
backbone is frozen. Forward hooks collect the time, FFT, and fused embeddings in
one pass. Per-channel patch means and standard deviations preserve motion
variation and give equally sized probe inputs. Small linear probes are fitted
on subject-disjoint windows. A linear
reliability gate mixes time and FFT probe probabilities. This is **not** a claim
that HART has two independent classification heads or that the probes reproduce
the paper's fine-tuned HART classifier.

The primary dataset is the original [HARTH UCI release](https://archive.ics.uci.edu/dataset/779/harth).
HARTH is absent from HART's listed 26 pretraining datasets. Place the subject
CSV files anywhere under `datasets/harth_raw`. Each must have `back_x`,
`back_y`, `back_z`, `thigh_x`, `thigh_y`, `thigh_z`, and `label` columns.
The downloader is intentionally separate because the dataset has its own terms.

## Protocol

- Resample original 50 Hz recordings to HART's 30 Hz reference using a polyphase
  anti-aliasing filter. Build nonoverlapping four-second windows and retain only
  windows with at least 90 percent of one label.
- Split by entire subject into probe fitting, gate fitting, validation, and
  untouched test. Sensor normalization is fitted on probe subjects only. The
  12-class label ontology is taken from the UCI dataset documentation, not
  inferred from held-out labels.
- Train probes on clean 30 Hz and anti-aliased 15/10 Hz windows. Train the gate
  on different subjects. Select stopping point and fixed mixture on validation.
- Test clean 30 Hz, anti-aliased 15/10 Hz, and naive-decimated 15/10 Hz. Naive
  decimation is a stress condition, not a recommended preprocessing policy.
- Compare fused, time, FFT, fixed blend, per-rate fixed blend, rate-aware gate,
  and rate-blind gate.
  The rate-aware gate sees only deployment-observable features. No test labels
  enter model selection.
- Inspect `metrics.csv`, `contrasts.csv`, `predictions.csv`, and `protocol.json`. `macro_f1`,
  NLL, ECE, and high-confidence error should all be assessed. The full backbone
  runs for every method, so this pilot makes **no latency or energy savings claim**.

The low-compute pilot uses 100 windows per subject by default. Raise
`--windows-per-subject` to 400 and repeat at three seeds only after the pilot
shows a credible effect. Missing training classes remain in the evaluation
ontology and are recorded in `protocol.json`. Increase sampling if rare-class
support is poor. Do not frame positive pilot numbers as confirmed results.

## Run on macOS

The repository's `environment.yml` pins Linux CUDA packages and should not be
used on Apple Silicon. From the repository root, create a separate environment.
Use the Python and Transformers versions matched to the released HART checkpoint.

```bash
/opt/anaconda3/envs/cdg/bin/python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.4,<3' 'transformers==4.56.0' 'granite-tsfm==0.3.2' numpy pandas scipy scikit-learn safetensors
python -m unittest discover -s experiments/rate_reliability -p 'test_*.py'
```

Download the original 22-subject UCI release. The unzip layout may include a
nested directory; the loader searches recursively for subject CSVs.

```bash
mkdir -p datasets/harth_raw
curl -L --fail 'https://archive.ics.uci.edu/static/public/779/harth.zip' -o datasets/harth.zip
unzip -q datasets/harth.zip -d datasets/harth_raw
```

First run a bounded pilot using every subject but only 40 windows from each.
Using all subjects avoids fragile class support in a tiny subject subset.

```bash
python experiments/rate_reliability/run.py --data datasets/harth_raw --windows-per-subject 40 --batch-size 32 --output outputs_rate_reliability/pilot
```

If the pilot is coherent, increase to 200 windows and repeat subject splits.

```bash
for seed in 7 17 27; do python experiments/rate_reliability/run.py --data datasets/harth_raw --windows-per-subject 200 --batch-size 64 --seed "$seed" --output "outputs_rate_reliability/seed_$seed"; done
```

Keep the declared seeds regardless of test performance. Inspect class-support
counts when interpreting rare-activity scores.
