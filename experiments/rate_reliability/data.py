"""Subject-separated HARTH windows and controlled sample-rate interventions."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


CHANNELS = ("back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z")
CONDITIONS = ("clean30", "aa15", "aa10", "naive15", "naive10")
HARTH_LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140)


def stable_window_indices(clean, labels, limit, rng):
    """Apply the purity/finite filter before uniformly capping valid windows."""
    count = len(clean) // 120
    if count == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    window_labels = labels[:count * 120].reshape(count, 120)
    classes = np.unique(window_labels)
    votes = np.stack([(window_labels == c).sum(axis=1) for c in classes], axis=1)
    majority = votes.argmax(axis=1)
    finite = np.isfinite(clean[:count * 120].reshape(count, 120, -1)).all(axis=(1, 2))
    valid = np.flatnonzero((votes.max(axis=1) >= 108) & finite)
    if limit and len(valid) > limit:
        valid = np.sort(rng.choice(valid, limit, replace=False))
    return valid * 120, classes[majority[valid]].astype(int)


def load_harth(root: Path, max_subjects: int, windows_per_subject: int, seed: int):
    """Read original 50 Hz subject files, then create nonoverlapping 30 Hz windows."""
    paths = []
    for path in sorted(root.rglob("*.csv")):
        header = pd.read_csv(path, nrows=0).columns
        if set(CHANNELS).issubset(header) and "label" in header:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No HARTH subject CSVs with expected columns under {root}")
    if max_subjects:
        paths = paths[:max_subjects]
    xs, ys, groups = [], [], []
    rng = np.random.default_rng(seed)
    for path in paths:
        df = pd.read_csv(path, usecols=[*CHANNELS, "label"])
        raw = df.loc[:, CHANNELS].to_numpy(dtype=np.float32)
        labels = df["label"].to_numpy()
        clean = resample_poly(raw, 3, 5, axis=0).astype(np.float32)
        nearest = np.minimum(np.round(np.arange(len(clean)) * 5 / 3).astype(int), len(labels) - 1)
        clean_labels = labels[nearest]
        starts, targets = stable_window_indices(clean, clean_labels, windows_per_subject, rng)
        retained = 0
        for start, target in zip(starts, targets):
            window = clean[start:start + 120]
            xs.append(window)
            ys.append(int(target))
            groups.append(path.stem)
            retained += 1
        print(f"{path.stem}: {retained} retained windows", flush=True)
    if not xs:
        raise ValueError("No stable, finite windows found")
    return np.stack(xs), np.asarray(ys), np.asarray(groups)


def split_subjects(groups: np.ndarray, seed: int):
    unique = np.unique(groups)
    if len(unique) < 8:
        raise ValueError("At least eight subject files are needed for probe/gate/val/test splits")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(2, round(0.2 * len(unique)))
    n_val = max(2, round(0.2 * len(unique)))
    remaining = unique[n_test + n_val:]
    n_gate = max(2, round(0.25 * len(remaining)))
    subjects = {
        "test": unique[:n_test].tolist(),
        "val": unique[n_test:n_test + n_val].tolist(),
        "gate": remaining[:n_gate].tolist(),
        "probe": remaining[n_gate:].tolist(),
    }
    if not subjects["probe"]:
        raise ValueError("Subject split leaves no probe subjects")
    indices = {part: np.flatnonzero(np.isin(groups, ids)) for part, ids in subjects.items()}
    return subjects, indices


def change_rate(windows: np.ndarray, condition: str):
    """Keep 4-second duration and 120 model samples; intervene only on information rate."""
    if condition == "clean30":
        return windows.copy()
    if condition not in CONDITIONS:
        raise ValueError(condition)
    factor = 2 if condition.endswith("15") else 3
    if condition.startswith("aa"):
        low = resample_poly(windows, 1, factor, axis=1)
    else:
        low = windows[:, ::factor, :]
    low_grid = np.arange(low.shape[1]) * factor
    high_grid = np.arange(windows.shape[1])
    restored = np.empty_like(windows)
    for i in range(windows.shape[0]):
        for c in range(windows.shape[2]):
            restored[i, :, c] = np.interp(high_grid, low_grid, low[i, :, c])
    return restored


def high_band_fraction(windows: np.ndarray):
    spectrum = np.abs(np.fft.rfft(windows, axis=1)) ** 2
    spectrum[:, 0, :] = 0
    upper = spectrum[:, int(spectrum.shape[1] * 0.65):, :].sum(axis=(1, 2))
    total = spectrum.sum(axis=(1, 2))
    return (upper / np.maximum(total, 1e-8)).astype(np.float32)
