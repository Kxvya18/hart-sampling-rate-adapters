"""Frozen HART branch reliability experiment on real HARTH subjects."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import CONDITIONS, HARTH_LABELS, change_rate, high_band_fraction, load_harth, split_subjects


REPO = Path(__file__).resolve().parents[2]
FIT_CONDITIONS = ("clean30", "aa15", "aa10")
BRANCHES = ("time", "frequency", "fused")


def pool_features(tensor):
    # Preserve sensor location and patch variability after RevIN removes signal means.
    return torch.cat((tensor.mean(dim=2), tensor.std(dim=2, unbiased=False)), dim=-1).flatten(1)


def extract(model, x: np.ndarray, device: torch.device, batch_size: int):
    observed = {}

    def capture(name):
        def hook(_module, _inputs, output):
            observed[name] = output
        return hook

    handles = [
        model.time_encoding.register_forward_hook(capture("time")),
        model.fft_encoding.register_forward_hook(capture("frequency")),
    ]
    features = {name: [] for name in BRANCHES}
    try:
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
                output = model(batch, enable_masking=False, return_dict=True)
                raw = {"time": observed["time"], "frequency": observed["frequency"],
                       "fused": output.last_hidden_state}
                for name, tensor in raw.items():
                    if tensor.ndim != 4:
                        raise RuntimeError(f"Unexpected {name} embedding shape {tuple(tensor.shape)}")
                    features[name].append(pool_features(tensor).float().cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
    return {name: np.concatenate(parts) for name, parts in features.items()}


def aligned_proba(model, x, classes):
    raw = model.predict_proba(x)
    result = np.zeros((len(x), len(classes)), dtype=np.float64)
    positions = np.searchsorted(classes, model[-1].classes_)
    result[:, positions] = raw
    result = np.clip(result, 1e-9, 1.0)
    return result / result.sum(axis=1, keepdims=True)


def gate_features(probabilities, spectral, rate):
    pt, pf = probabilities["time"], probabilities["frequency"]
    entropy = lambda p: -(p * np.log(np.clip(p, 1e-9, 1))).sum(axis=1)
    disagreement = np.abs(pt - pf).sum(axis=1) / 2
    return np.column_stack((np.full(len(pt), rate / 30), spectral,
                            entropy(pt), entropy(pf), pt.max(axis=1),
                            pf.max(axis=1), disagreement)).astype(np.float32)


def fit_gate(train, val, classes, rate_aware, epochs):
    x_train, pt_train, pf_train, y_train = train
    x_val, pt_val, pf_val, y_val = val
    columns = slice(None) if rate_aware else slice(2, None)
    scaler = StandardScaler().fit(x_train[:, columns])
    a = torch.tensor(scaler.transform(x_train[:, columns]), dtype=torch.float32)
    b = torch.tensor(scaler.transform(x_val[:, columns]), dtype=torch.float32)
    yt = torch.tensor(np.searchsorted(classes, y_train), dtype=torch.long)
    pt = torch.tensor(pt_train, dtype=torch.float32)
    pf = torch.tensor(pf_train, dtype=torch.float32)
    linear = torch.nn.Linear(a.shape[1], 1)
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.02, weight_decay=1e-3)
    best, state = float("inf"), None
    wait = 0
    for _ in range(epochs):
        linear.train()
        optimizer.zero_grad()
        gate = torch.sigmoid(linear(a))
        mixture = (1 - gate) * pt + gate * pf
        loss = -torch.log(mixture[torch.arange(len(yt)), yt].clamp_min(1e-9)).mean()
        loss.backward()
        optimizer.step()
        linear.eval()
        with torch.no_grad():
            gv = torch.sigmoid(linear(b)).numpy()
        pv = (1 - gv) * pt_val + gv * pf_val
        score = log_loss(y_val, pv, labels=classes)
        if score < best - 1e-5:
            best = score
            state = {k: v.detach().clone() for k, v in linear.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 20:
                break
    linear.load_state_dict(state)
    return scaler, linear.eval(), columns


def predict_gate(fitted, x, pt, pf):
    scaler, linear, columns = fitted
    with torch.no_grad():
        g = torch.sigmoid(linear(torch.tensor(scaler.transform(x[:, columns]), dtype=torch.float32))).numpy()
    return (1 - g) * pt + g * pf, g[:, 0]


def metrics(y, p, classes):
    pred = classes[p.argmax(axis=1)]
    conf = p.max(axis=1)
    correct = pred == y
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(np.mean(bins == k) * abs(correct[bins == k].mean() - conf[bins == k].mean())
              for k in range(10) if np.any(bins == k))
    high = conf >= 0.8
    return {"n": len(y), "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=classes, average="macro", zero_division=0),
            "nll": log_loss(y, p, labels=classes), "ece10": ece,
            "high_conf_error_fraction": np.mean(high & ~correct),
            "high_conf_error_rate": np.mean(~correct[high]) if high.any() else None,
            "high_conf_count": int(high.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing original HARTH subject CSVs")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "finetuning_HART/pretrained_checkpoint")
    parser.add_argument("--output", type=Path, default=REPO / "outputs_rate_reliability/pilot")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-subjects", type=int, default=0, help="0 uses all subjects")
    parser.add_argument("--windows-per-subject", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gate-epochs", type=int, default=120)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; use --device cpu")
    x, y, groups = load_harth(args.data, args.max_subjects, args.windows_per_subject, args.seed)
    subjects, indices = split_subjects(groups, args.seed)
    classes = np.asarray(HARTH_LABELS)
    unexpected = sorted(set(y) - set(classes))
    if unexpected:
        raise ValueError(f"Unknown labels {unexpected}; expected the original UCI HARTH label ontology")
    missing = sorted(set(classes) - set(y[indices["probe"]]))
    if missing:
        print(f"Pilot training lacks classes {missing}; retaining them in evaluation with smoothed probability", flush=True)
    mean = x[indices["probe"]].mean(axis=(0, 1), keepdims=True)
    std = np.maximum(x[indices["probe"]].std(axis=(0, 1), keepdims=True), 1e-5)
    x = (x - mean) / std
    print(f"device={device}, subjects={len(subjects['probe'])}/{len(subjects['gate'])}/"
          f"{len(subjects['val'])}/{len(subjects['test'])}, windows={len(x)}", flush=True)
    try:
        from tsfm_public.models.tspulse import TSPulseForClassification
    except ImportError as exc:
        raise RuntimeError("Install granite-tsfm and the pinned dependencies from the experiment README") from exc
    model = TSPulseForClassification.from_pretrained(
        str(args.checkpoint), num_input_channels=len(x[0, 0]), num_targets=len(classes),
        context_length=120, loss="cross_entropy", channel_virtual_expand_scale=1,
        disable_mask_in_classification_eval=True, fft_time_consistent_masking=True,
        head_reduce_d_model=1, decoder_mode="common_channel", head_aggregation_dim="patch",
        ignore_mismatched_sizes=True,
    ).backbone.to(device).eval()
    if not hasattr(model, "fft_encoding"):
        raise RuntimeError("Checkpoint does not expose HART's FFT encoder")
    all_features = {}
    all_spectral = {}
    for condition in CONDITIONS:
        changed = change_rate(x, condition)
        all_spectral[condition] = high_band_fraction(changed)
        all_features[condition] = extract(model, changed, device, args.batch_size)
        print(f"extracted {condition}", flush=True)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    probes = {}
    for branch in BRANCHES:
        xp = np.concatenate([all_features[c][branch][indices["probe"]] for c in FIT_CONDITIONS])
        yp = np.tile(y[indices["probe"]], len(FIT_CONDITIONS))
        probes[branch] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
        probes[branch].fit(xp, yp)
    probabilities = {c: {b: aligned_proba(probes[b], all_features[c][b], classes)
                         for b in BRANCHES} for c in CONDITIONS}

    def collect(part):
        ids = indices[part]
        feat, pt, pf, yy = [], [], [], []
        for condition, rate in (("clean30", 30), ("aa15", 15), ("aa10", 10)):
            q = probabilities[condition]
            feat.append(gate_features({b: q[b][ids] for b in BRANCHES},
                                      all_spectral[condition][ids], rate))
            pt.append(q["time"][ids])
            pf.append(q["frequency"][ids])
            yy.append(y[ids])
        return np.concatenate(feat), np.concatenate(pt), np.concatenate(pf), np.concatenate(yy)

    gate_train, gate_val = collect("gate"), collect("val")
    rate_gate = fit_gate(gate_train, gate_val, classes, True, args.gate_epochs)
    blind_gate = fit_gate(gate_train, gate_val, classes, False, args.gate_epochs)
    _, vt, vf, vy = gate_val
    fixed_grid = np.linspace(0, 1, 21)
    fixed_alpha = min(fixed_grid, key=lambda a: log_loss(vy, (1 - a) * vt + a * vf, labels=classes))
    rate_alphas = {}
    for condition, rate in (("clean30", 30), ("aa15", 15), ("aa10", 10)):
        v_ids = indices["val"]
        pt, pf = probabilities[condition]["time"][v_ids], probabilities[condition]["frequency"][v_ids]
        rate_alphas[rate] = float(min(fixed_grid, key=lambda a: log_loss(y[v_ids], (1 - a) * pt + a * pf, labels=classes)))
    print(f"validation-selected fixed alpha={fixed_alpha:.2f}", flush=True)
    rows, predictions = [], []
    ids = indices["test"]
    for condition in CONDITIONS:
        q = probabilities[condition]
        rate = 30 if condition == "clean30" else (15 if condition.endswith("15") else 10)
        features = gate_features({b: q[b][ids] for b in BRANCHES}, all_spectral[condition][ids], rate)
        policies = {b: q[b][ids] for b in BRANCHES}
        policies["fixed_mix"] = (1 - fixed_alpha) * policies["time"] + fixed_alpha * policies["frequency"]
        policies["rate_constant"] = (1 - rate_alphas[rate]) * policies["time"] + rate_alphas[rate] * policies["frequency"]
        policies["rate_gate"], gate_values = predict_gate(rate_gate, features, policies["time"], policies["frequency"])
        policies["blind_gate"], _ = predict_gate(blind_gate, features, policies["time"], policies["frequency"])
        for policy, p in policies.items():
            rows.append({"condition": condition, "policy": policy, **metrics(y[ids], p, classes)})
            for j, idx in enumerate(ids):
                predictions.append({"subject": groups[idx], "window_index": int(idx),
                                    "condition": condition, "policy": policy,
                                    "true": int(y[idx]), "predicted": int(classes[p[j].argmax()]),
                                    "confidence": float(p[j].max()),
                                    "p_true": float(p[j, np.searchsorted(classes, y[idx])]),
                                    "gate": float(gate_values[j]) if policy == "rate_gate" else None})
    results = pd.DataFrame(rows)
    results.to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output / "predictions.csv", index=False)
    provenance = {"source_repo": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  "checkpoint": str(args.checkpoint), "dataset": str(args.data),
                  "subjects": subjects, "conditions": CONDITIONS, "fit_conditions": FIT_CONDITIONS,
                  "classes": classes.tolist(), "fixed_alpha": float(fixed_alpha),
                  "rate_alphas": rate_alphas, "missing_probe_classes": [int(c) for c in missing],
                  "class_counts": {part: {str(c): int(np.sum(y[idx] == c)) for c in classes}
                                   for part, idx in indices.items()},
                  "seed": args.seed, "windows_per_subject": args.windows_per_subject,
                  "normalization": "probe subjects only", "device": str(device)}
    (args.output / "protocol.json").write_text(json.dumps(provenance, indent=2))
    indexed = results.set_index(["condition", "policy"])
    contrasts = []
    for condition in CONDITIONS:
        gate = indexed.loc[(condition, "rate_gate")]
        for baseline in ("fused", "fixed_mix", "rate_constant", "blind_gate"):
            reference = indexed.loc[(condition, baseline)]
            contrasts.append({"condition": condition, "baseline": baseline,
                              "f1_gain": float(gate.macro_f1 - reference.macro_f1),
                              "nll_reduction": float(reference.nll - gate.nll),
                              "high_conf_error_reduction": float(reference.high_conf_error_fraction - gate.high_conf_error_fraction)})
    pd.DataFrame(contrasts).to_csv(args.output / "contrasts.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
