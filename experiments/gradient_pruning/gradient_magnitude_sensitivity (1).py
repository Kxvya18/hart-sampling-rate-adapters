"""
gradient_magnitude_sensitivity.py
-----------------------------------
Layer sensitivity analysis using GRADIENT MAGNITUDE (L1 norm of gradients).

Method
------
1. Load pretrained checkpoint + a small batch of dataset windows.
2. Run n_batches forward+backward passes (no weight update).
3. Per encoder layer: sum |grad| across all parameters -> GM importance score.
4. Normalise scores to [0, 1]  (max layer = 1.0).
5. For each pruning threshold (25 / 50 / 75 / 90%):
   - SKIP (identity pass-through) bottom-T% least-important layers by GM rank.
   - Run LOSO fine-tuning + evaluation -> real Acc and F1 at that sparsity.
   NOTE: layers are *skipped* (output = input), not zeroed. Zeroing hooks produce
         identical outputs across thresholds because zeroing L6 already cuts the
         signal path to L7; additional zeroes downstream have no further effect.
         Skipping correctly lets the remaining layers see meaningful signals.
6. Outputs saved to sensitivity_results_gm/{dataset}/:
     gm_bargraph.png         — single dual-panel: GM sensitivity + GM norm per layer
     gm_pruning_impact.png   — Acc AND F1 vs pruning threshold (real LOSO metrics)
     gm_param_table.png      — parameter count table per threshold
     gradient_magnitude_scores.csv
     param_count_by_threshold.csv
     pruning_eval_results.csv

Run from HART root:
    cd finetuning_HART
    python ../pruning/gradient_magnitude_sensitivity.py \\
        --dataset        uci_har \\
        --split          ID \\
        --checkpoint_path pretrained_checkpoint \\
        --test_set_dir   ../datasets/test_set \\
        --output_dir     ../sensitivity_results_gm \\
        --n_batches      10
"""

import os
import sys
import argparse
import warnings
import math
import gc
import tempfile
import numpy as np
import pandas as pd

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["PYTORCH_ALLOC_CONF"]   = "expandable_segments:True"

script_dir   = os.path.dirname(os.path.abspath(__file__))
finetune_dir = os.path.join(script_dir, "..", "finetuning_HART")
sys.path.insert(0, os.path.abspath(finetune_dir))

from tsfm_public.models.tspulse import TSPulseForClassification
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseLayer
from tsfm_public.toolkit.time_series_classification_preprocessor import (
    TimeSeriesClassificationPreprocessor,
)
from tsfm_public.toolkit.dataset import ClassificationDFDataset
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    Trainer, TrainingArguments, EarlyStoppingCallback, set_seed
)
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import RemoveColumnsCollator
from sklearn.metrics import accuracy_score, f1_score

from HAR_dataloader import load_dataset, get_input_columns, get_label_column, get_subject_column

set_seed(42)

NUM_LAYERS = 8
THRESHOLDS = [25, 50, 75, 90]   # % of least-important layers to zero

COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HART Gradient Magnitude Sensitivity")
    p.add_argument("--dataset",         type=str, default="uci_har")
    p.add_argument("--split",           type=str, default="ID", choices=["ID", "OOD"])
    p.add_argument("--checkpoint_path", type=str, default="pretrained_checkpoint")
    p.add_argument("--output_dir",      type=str, default="../sensitivity_results_gm")
    p.add_argument("--device",          type=str, default="cuda")
    p.add_argument("--test_set_dir",    type=str, default=None)
    p.add_argument("--context_length",  type=int, default=120)
    p.add_argument("--hop_length",      type=int, default=30)
    p.add_argument("--batch_size",      type=int, default=256)
    p.add_argument("--n_batches",       type=int, default=10,
                   help="Mini-batches for gradient accumulation.")
    p.add_argument("--epochs",          type=int, default=30)
    p.add_argument("--patience",        type=int, default=10)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_df(args, dataset):
    load_kw = dict(window_length=args.context_length, hop_length=args.hop_length)
    if args.test_set_dir:
        load_kw["test_set_dir"] = args.test_set_dir
    df = load_dataset(dataset, **load_kw)
    subject_col = get_subject_column(dataset)
    label_col   = get_label_column(dataset)
    per_subj    = df.groupby(subject_col)[label_col].apply(set)
    common      = set.intersection(*per_subj)
    df          = df[df[label_col].isin(common)].reset_index(drop=True)
    return df


def build_quick_dataloader(args, df, dataset):
    """Small dataloader for gradient computation only."""
    input_cols  = get_input_columns(dataset)
    label_col   = get_label_column(dataset)

    tsp = TimeSeriesClassificationPreprocessor(
        input_columns=input_cols, label_column=label_col, scaling=True
    )
    tsp.train(df)
    df_prep = tsp.preprocess(df)

    ds = ClassificationDFDataset(
        df_prep,
        id_columns=[], timestamp_column=None,
        input_columns=input_cols, label_column=label_col,
        context_length=args.context_length,
        static_categorical_columns=[], stride=1,
        enable_padding=False, full_series=True,
    )
    n_classes  = len(tsp.label_encoder.classes_)
    n_channels = tsp.num_input_channels
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True), n_classes, n_channels


def prepare_loso_dataset(df_train, df_test, input_cols, label_col, all_classes, ctx_len):
    tsp = TimeSeriesClassificationPreprocessor(
        input_columns=input_cols, label_column=label_col, scaling=True
    )
    tsp.train(df_train)
    tsp.label_encoder.classes_ = np.array(all_classes)

    ds_kw = dict(
        id_columns=[], timestamp_column=None,
        input_columns=input_cols, label_column=label_col,
        context_length=ctx_len,
        static_categorical_columns=[], stride=1,
        enable_padding=False, full_series=True,
    )
    base_ds = ClassificationDFDataset(tsp.preprocess(df_train), **ds_kw)
    test_ds = ClassificationDFDataset(tsp.preprocess(df_test),  **ds_kw)

    n_val   = max(1, int(0.1 * len(base_ds)))
    n_train = len(base_ds) - n_val
    train_ds, val_ds = random_split(base_ds, [n_train, n_val])
    return train_ds, val_ds, test_ds, tsp


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

import types as _types
from dataclasses import dataclass
from typing import Optional

@dataclass
class TSPulseOutput:
    loss:               Optional[torch.FloatTensor] = None
    logits:             Optional[torch.FloatTensor] = None
    prediction_outputs: Optional[torch.FloatTensor] = None

    def __iter__(self):
        yield self.loss
        yield self.logits

    def __getitem__(self, idx):
        return (self.loss, self.logits)[idx]


def _patch_encoder_skip(model, layers_to_skip: set):
    """
    Monkey-patch TSPulseBlock.forward so the specified TSPulseLayer indices are
    skipped (identity pass-through).  Returns the original bound method so it
    can be restored later.

    WHY skip instead of zero:
      Zeroing layer i's *output* means layer i+1 receives zeros and still
      transforms them with its own biases -> non-zero output.  So zeroing
      layer j>i has no additional effect once i is zeroed.  Skipping
      (output = input) keeps the signal alive for all downstream layers,
      so each extra skipped layer produces a distinct, measurable change.
    """
    block = model.backbone.encoder_block
    original_forward = block.forward   # bound method

    # Build a map: position in mixers -> TSPulseLayer index (or None)
    tsp_idx = 0
    mixer_tsp_map = []
    for m in block.mixers:
        if isinstance(m, TSPulseLayer):
            mixer_tsp_map.append(tsp_idx)
            tsp_idx += 1
        else:
            mixer_tsp_map.append(None)

    def _skipping_forward(self_block, hidden_state, output_hidden_states=False):
        embedding = hidden_state
        all_hidden_states = []
        for i, mod in enumerate(self_block.mixers):
            tsp_layer_idx = mixer_tsp_map[i]
            if tsp_layer_idx is not None and tsp_layer_idx in layers_to_skip:
                pass  # identity: embedding unchanged
            else:
                embedding = mod(embedding)
            if output_hidden_states:
                all_hidden_states.append(embedding)
        if self_block.return_tuple:
            return (embedding, all_hidden_states) if output_hidden_states else (embedding, None)
        return embedding

    block.forward = _types.MethodType(_skipping_forward, block)
    return original_forward   # caller must restore this


def _restore_encoder(model, original_forward):
    model.backbone.encoder_block.forward = original_forward


class TSPulseForHAR(torch.nn.Module):
    def __init__(self, checkpoint_path, num_classes, num_input_channels,
                 config_overrides, layers_to_skip: set = None):
        super().__init__()
        overrides = dict(config_overrides)
        overrides["loss"] = "cross_entropy"
        self._model = TSPulseForClassification.from_pretrained(
            checkpoint_path, **overrides
        )
        self._loss_fn = torch.nn.CrossEntropyLoss()

        for p in self._model.backbone.parameters():
            p.requires_grad = False
        for p in self._model.backbone.time_encoding.parameters():
            p.requires_grad = True
        for p in self._model.backbone.fft_encoding.parameters():
            p.requires_grad = True

        self._original_forward = None
        if layers_to_skip:
            self._original_forward = _patch_encoder_skip(self._model, layers_to_skip)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        skip_str  = str(sorted(layers_to_skip)) if layers_to_skip else "none"
        print(f"  Trainable: {trainable:,} / {total:,}  |  skipped layers: {skip_str}")

    def forward(self, past_values, target_values=None, past_observed_mask=None, **kwargs):
        out    = self._model(past_values=past_values, target_values=None,
                             past_observed_mask=past_observed_mask)
        logits = (out.prediction_outputs
                  if hasattr(out, "prediction_outputs") and out.prediction_outputs is not None
                  else out[1])
        loss = self._loss_fn(logits, target_values.long()) if target_values is not None else None
        return TSPulseOutput(loss=loss, logits=logits, prediction_outputs=logits)

    def restore_encoder(self):
        if self._original_forward is not None:
            _restore_encoder(self._model, self._original_forward)


def load_model_plain(checkpoint_path, num_classes, num_channels, device):
    overrides = {
        "head_gated_attention_activation":     "softmax",
        "channel_virtual_expand_scale":        1,
        "mask_ratio":                          0.3,
        "head_reduce_d_model":                 1,
        "disable_mask_in_classification_eval": True,
        "fft_time_consistent_masking":         True,
        "decoder_mode":                        "common_channel",
        "head_aggregation_dim":                "patch",
        "head_aggregation":                    None,
        "num_input_channels":                  num_channels,
        "num_targets":                         num_classes,
        "loss":                                "cross_entropy",
        "ignore_mismatched_sizes":             True,
    }
    model = TSPulseForClassification.from_pretrained(checkpoint_path, **overrides)
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient magnitude computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradient_magnitudes(model, dataloader, device, n_batches):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    model.zero_grad()

    processed = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break
        past_values = batch["past_values"].to(device)
        target      = batch["target_values"].long().to(device)
        out    = model(past_values=past_values, target_values=None)
        logits = (out.prediction_outputs
                  if hasattr(out, "prediction_outputs") and out.prediction_outputs is not None
                  else out[1])
        loss = loss_fn(logits, target)
        loss.backward()
        processed += 1
        print(f"  Batch {processed}/{n_batches}  loss={loss.item():.4f}")

    tspulse_layers = [m for m in model.backbone.encoder_block.mixers
                      if isinstance(m, TSPulseLayer)]
    layer_gm     = {}
    layer_params = {}
    for idx, layer in enumerate(tspulse_layers):
        total_grad = 0.0
        n_p        = 0
        for name, param in layer.named_parameters():
            if param.grad is not None:
                total_grad += param.grad.abs().sum().item()
                n_p        += param.numel()
        layer_gm[idx]    = total_grad
        layer_params[idx] = n_p

    model.zero_grad()
    return layer_gm, layer_params


# ─────────────────────────────────────────────────────────────────────────────
# LOSO evaluation with zeroed layers
# ─────────────────────────────────────────────────────────────────────────────

def run_loso_with_skipped_layers(df, input_cols, label_col, subject_col,
                                  all_classes, checkpoint_path, args,
                                  layers_to_skip: set, num_channels: int) -> dict:
    """Run full LOSO, skipping (identity pass-through) the given layer indices."""
    if layers_to_skip:
        print(f"  [skip] TSPulseLayers skipped: {sorted(layers_to_skip)}")
    else:
        print(f"  [skip] No layers skipped (baseline run)")
    fold_metrics = []
    subjects     = df[subject_col].unique()

    for fold_idx, sid in enumerate(subjects):
        df_train = df[df[subject_col] != sid].reset_index(drop=True)
        df_test  = df[df[subject_col] == sid].reset_index(drop=True)
        print(f"  Fold {fold_idx+1}/{len(subjects)}  |  test subject: {sid}")

        train_ds, val_ds, test_ds, tsp = prepare_loso_dataset(
            df_train, df_test, input_cols, label_col, all_classes, args.context_length
        )

        n_ch  = tsp.num_input_channels
        batch = min(args.batch_size,
                    3096 if n_ch <= 9 else 1024 if n_ch <= 18 else
                    512  if n_ch <= 36 else 256  if n_ch <= 72 else 64)

        config_overrides = {
            "head_gated_attention_activation":      "softmax",
            "channel_virtual_expand_scale":         1,
            "mask_ratio":                           0.3,
            "head_reduce_d_model":                  1,
            "disable_mask_in_classification_eval":  True,
            "fft_time_consistent_masking":          True,
            "decoder_mode":                         "common_channel",
            "head_aggregation_dim":                 "patch",
            "head_aggregation":                     None,
            "num_input_channels":                   tsp.num_input_channels,
            "num_targets":                          len(all_classes),
            "loss":                                 "cross_entropy",
            "ignore_mismatched_sizes":              True,
        }

        model = TSPulseForHAR(
            checkpoint_path=checkpoint_path,
            num_classes=len(all_classes),
            num_input_channels=tsp.num_input_channels,
            config_overrides=config_overrides,
            layers_to_skip=layers_to_skip,
        ).to(args.device)

        lr, model = optimal_lr_finder(model, train_ds, batch_size=batch)
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = OneCycleLR(
            optimizer, lr, epochs=args.epochs,
            steps_per_epoch=max(1, math.ceil(len(train_ds) / batch)),
        )

        training_args = TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            learning_rate=lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            eval_strategy="epoch", save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            label_names=["target_values"],
        )

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=val_ds,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
            optimizers=(optimizer, scheduler),
        )
        trainer.train()

        raw_preds = trainer.predict(test_ds).predictions
        preds     = raw_preds[0] if isinstance(raw_preds, tuple) else raw_preds
        y_pred    = np.argmax(preds, axis=1)

        dl = DataLoader(
            test_ds, batch_size=batch, shuffle=False,
            collate_fn=RemoveColumnsCollator(
                data_collator=default_data_collator,
                signature_columns=["target_values"],
                logger=None, description=None, model_name="tmp",
            ),
        )
        y_true = np.concatenate([b["target_values"].numpy() for b in dl])

        m = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "subject_id": sid,
        }
        fold_metrics.append(m)
        print(f"    Acc={m['accuracy']:.4f}  F1={m['f1_score']:.4f}")

        model.restore_encoder()
        del trainer, model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

    return pd.DataFrame(fold_metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter table
# ─────────────────────────────────────────────────────────────────────────────

def compute_param_table(layer_gm, layer_params_count, ranked, thresholds):
    n = NUM_LAYERS
    max_gm = max(layer_gm.values()) if max(layer_gm.values()) > 0 else 1.0
    rows = []
    for i in range(n):
        row = {
            "layer":           i,
            "params_original": layer_params_count[i],
            "gm_score":        round(layer_gm[i], 2),
            "gm_norm":         round(layer_gm[i] / max_gm, 4),
        }
        for t in thresholds:
            n_drop = max(1, int(round(n * t / 100)))
            zeroed = set(ranked[:n_drop])
            row[f"params_at_{t}pct"] = 0 if i in zeroed else layer_params_count[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    total = {"layer": "TOTAL", "params_original": df["params_original"].sum(),
             "gm_score": "", "gm_norm": ""}
    for t in thresholds:
        total[f"params_at_{t}pct"] = df[f"params_at_{t}pct"].sum()
    rows.append(total)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _ds_title(dataset, split):
    return dataset.upper().replace("_", "-") + f" ({split})"


def plot_gm_bargraph(layer_gm, dataset, split, out_path):
    """
    Single-panel bar chart: normalised GM score per layer with baseline line at 1.0.
    """
    n      = NUM_LAYERS
    max_gm = max(layer_gm.values()) if max(layer_gm.values()) > 0 else 1.0
    norms  = [layer_gm[i] / max_gm for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"HART Encoder -- Layer Sensitivity Analysis  [Gradient Magnitude]\n"
        f"Dataset: {_ds_title(dataset, split)}  |  "
        f"Baseline (normalised GM of most important layer): 1.0000",
        fontsize=13, fontweight="bold",
    )

    ax.bar([f"L{i}" for i in range(n)], norms,
           color="#1f77b4", edgecolor="black", linewidth=0.7, width=0.6,
           label="GM Score (normalised)")
    ax.axhline(1.0, color="#ff7f0e", linewidth=2, linestyle="--", label="Max (1.0000)")
    ax.set_xlabel("Encoder Layer", fontsize=11)
    ax.set_ylabel("Normalised Gradient Magnitude", fontsize=11)
    ax.set_title("GM Score per Layer", fontsize=11)
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=9)
    for bar, v in zip(ax.patches, norms):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_pruning_impact(eval_rows, baseline_acc, baseline_f1, dataset, split, out_path):
    """
    Dual-panel pruning impact plot.
    Left  — Acc & F1 line plot vs pruning threshold; x-tick labels show zeroed layers.
    Right — Acc drop & F1 drop grouped bar chart per threshold.
    """
    thresholds  = [r["threshold"]      for r in eval_rows]
    acc_vals    = [r["mean_acc"] * 100 for r in eval_rows]
    f1_vals     = [r["mean_f1"]  * 100 for r in eval_rows]
    acc_drops   = [r["acc_drop"] * 100 for r in eval_rows]
    f1_drops    = [r["f1_drop"]  * 100 for r in eval_rows]
    zeroed_lbls = [r["skipped_layers"]  for r in eval_rows]   # e.g. "[6, 7]"

    x_pts     = [0] + thresholds
    acc_line  = [baseline_acc * 100] + acc_vals
    f1_line   = [baseline_f1  * 100] + f1_vals

    # x-tick labels: "0%\n(baseline)" then "25%\nL[6,7]" etc.
    xtick_labels = ["0%\n(baseline)"]
    for t, zl in zip(thresholds, zeroed_lbls):
        xtick_labels.append(f"{t}%\nL{zl}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"Pruning Impact: Accuracy & F1 vs Sparsity  --  {_ds_title(dataset, split)}\n"
        f"Layers zeroed by GM rank (least important first)",
        fontsize=12, fontweight="bold",
    )

    # ── Left: line plot ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(range(len(x_pts)), acc_line, marker="o", markersize=7, linewidth=2.0,
            color="#1f77b4", label="Accuracy")
    ax.plot(range(len(x_pts)), f1_line,  marker="s", markersize=7, linewidth=2.0,
            color="#ff7f0e", label="F1 Score (macro)")

    ax.axhline(baseline_acc * 100, color="#1f77b4", linewidth=0.8,
               linestyle="--", alpha=0.4)
    ax.axhline(baseline_f1  * 100, color="#ff7f0e", linewidth=0.8,
               linestyle="--", alpha=0.4)

    for xi, ai, fi in zip(range(len(x_pts)), acc_line, f1_line):
        ax.annotate(f"{ai:.1f}", xy=(xi, ai), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color="#1f77b4")
        ax.annotate(f"{fi:.1f}", xy=(xi, fi), xytext=(0, -13),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color="#ff7f0e")

    all_vals = acc_line + f1_line
    y_min = max(0,   min(all_vals) - 6)
    y_max = min(100, max(all_vals) + 8)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.4, len(x_pts) - 0.6)
    ax.set_xticks(range(len(x_pts)))
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.set_xlabel("Pruning Threshold  (layers zeroed)", fontsize=11)
    ax.set_ylabel("Metric (%)", fontsize=11)
    ax.set_title("Acc & F1 vs Pruning Threshold", fontsize=11)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    # ── Right: grouped bar chart of drops ─────────────────────────────────────
    ax2 = axes[1]
    n_t   = len(thresholds)
    x_bar = np.arange(n_t)
    w     = 0.35

    bars_acc = ax2.bar(x_bar - w/2, acc_drops, w, color="#1f77b4",
                       edgecolor="black", linewidth=0.7, label="Acc Drop")
    bars_f1  = ax2.bar(x_bar + w/2, f1_drops,  w, color="#ff7f0e",
                       edgecolor="black", linewidth=0.7, label="F1 Drop")

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    for bar, v in zip(bars_acc, acc_drops):
        yoff = 0.3 if v >= 0 else -1.2
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yoff,
                 f"{v:+.1f}", ha="center", va="bottom", fontsize=8.5, color="#1f77b4")
    for bar, v in zip(bars_f1, f1_drops):
        yoff = 0.3 if v >= 0 else -1.2
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yoff,
                 f"{v:+.1f}", ha="center", va="bottom", fontsize=8.5, color="#ff7f0e")

    ax2.set_xticks(x_bar)
    ax2.set_xticklabels([f"{t}%\nL{zl}" for t, zl in zip(thresholds, zeroed_lbls)],
                        fontsize=8)
    ax2.set_xlabel("Pruning Threshold  (layers zeroed)", fontsize=11)
    ax2.set_ylabel("Drop from Baseline (%)", fontsize=11)
    ax2.set_title("Acc & F1 Drop vs Baseline", fontsize=11)
    drop_max = max(max(acc_drops), max(f1_drops)) if acc_drops else 1
    drop_min = min(min(acc_drops), min(f1_drops)) if acc_drops else 0
    ax2.set_ylim(min(0, drop_min) - 3, drop_max + 5)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_param_table(param_df, thresholds, out_path):
    data_rows = param_df[param_df["layer"] != "TOTAL"].copy()
    total_row = param_df[param_df["layer"] == "TOTAL"].copy()

    t_cols     = [f"Params\n@{t}% pruned" for t in thresholds]
    col_labels = ["Layer", "Params\n(original)", "GM Score", "GM Norm"] + t_cols

    def fmt_row(row):
        cells = [
            f"L{int(row['layer'])}" if row["layer"] != "TOTAL" else "TOTAL",
            f"{int(row['params_original']):,}",
            f"{row['gm_score']}" if row["gm_score"] != "" else "",
            f"{row['gm_norm']:.4f}" if row["gm_norm"] != "" else "",
        ]
        for t in thresholds:
            cells.append(f"{int(row[f'params_at_{t}pct']):,}")
        return cells

    cell_text = [fmt_row(r) for _, r in data_rows.iterrows()]
    cell_text.append(fmt_row(total_row.iloc[0]))

    n_data   = len(data_rows)
    row_cols = [["#f0f4f8"] * len(col_labels)] * n_data
    row_cols.append(["#d0e4f7"] * len(col_labels))

    fig, ax = plt.subplots(figsize=(15, 0.55 * (n_data + 3)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellLoc="center", loc="center", cellColours=row_cols)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for col_idx in range(len(col_labels)):
        tbl[(0, col_idx)].set_facecolor("#2c5f8a")
        tbl[(0, col_idx)].get_text().set_color("white")
        tbl[(0, col_idx)].get_text().set_fontweight("bold")

    ax.set_title(
        "HART Encoder -- Parameter Count per Layer at Each Pruning Threshold\n"
        "(Gradient Magnitude: bottom-N% layers zeroed  |  Thresholds: 25 / 50 / 75 / 90%)",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    dataset = args.dataset

    if args.device.startswith("cuda"):
        gpu_idx = args.device.split(":")[-1] if ":" in args.device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    args.device = device

    out_dir = os.path.join(args.output_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Gradient Magnitude Sensitivity")
    print(f"  Dataset : {dataset}  ({args.split})")
    print(f"  Output  : {out_dir}")
    print(f"{'='*60}")

    # ── [1] Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading dataset ...")
    df          = load_df(args, dataset)
    input_cols  = get_input_columns(dataset)
    label_col   = get_label_column(dataset)
    subject_col = get_subject_column(dataset)
    all_classes = sorted(df[label_col].unique().tolist())

    dataloader, num_classes, num_channels = build_quick_dataloader(args, df, dataset)
    n_use = min(args.n_batches, len(dataloader))
    print(f"  Batches: {len(dataloader)}  |  Using: {n_use}  "
          f"|  channels: {num_channels}  classes: {num_classes}")

    # ── [2] Compute gradient magnitudes ──────────────────────────────────────
    print(f"\n[2/4] Computing gradient magnitudes ({n_use} batches) ...")
    gm_model = load_model_plain(args.checkpoint_path, num_classes, num_channels, device)
    for p in gm_model.parameters():
        p.requires_grad = True

    layer_gm, layer_params_count = compute_gradient_magnitudes(
        gm_model, dataloader, device, n_use
    )
    del gm_model
    gc.collect()
    torch.cuda.empty_cache()

    max_gm = max(layer_gm.values()) if max(layer_gm.values()) > 0 else 1.0
    ranked = sorted(range(NUM_LAYERS), key=lambda i: layer_gm[i])  # least important first

    print(f"\n  GM scores (normalised):")
    for i in range(NUM_LAYERS):
        print(f"    Layer {i}: GM={layer_gm[i]:.2f}  norm={layer_gm[i]/max_gm:.4f}  "
              f"params={layer_params_count[i]:,}")

    scores_df = pd.DataFrame([
        {"layer": i, "gm_score": layer_gm[i], "gm_norm": layer_gm[i] / max_gm,
         "params": layer_params_count[i]}
        for i in range(NUM_LAYERS)
    ])
    scores_csv = os.path.join(out_dir, "gradient_magnitude_scores.csv")
    scores_df.to_csv(scores_csv, index=False)

    param_df  = compute_param_table(layer_gm, layer_params_count, ranked, THRESHOLDS)
    param_csv = os.path.join(out_dir, "param_count_by_threshold.csv")
    param_df.to_csv(param_csv, index=False)
    print(f"\n  Scores  -> {scores_csv}")
    print(f"  Params  -> {param_csv}")

    # ── [3] LOSO evaluation at each pruning threshold ─────────────────────────
    print(f"\n[3/4] Running LOSO evaluation at each pruning threshold ...")

    # Baseline (no skipping)
    print(f"\n  -- Baseline (no pruning) --")
    set_seed(42)
    baseline_df = run_loso_with_skipped_layers(
        df, input_cols, label_col, subject_col,
        all_classes, args.checkpoint_path, args,
        layers_to_skip=set(), num_channels=num_channels,
    )
    baseline_acc = baseline_df["accuracy"].mean()
    baseline_f1  = baseline_df["f1_score"].mean()
    baseline_df.to_csv(os.path.join(out_dir, "eval_baseline.csv"), index=False)
    print(f"  Baseline -> Acc={baseline_acc:.4f}  F1={baseline_f1:.4f}")

    eval_rows = []
    for t in THRESHOLDS:
        n_drop    = max(1, int(round(NUM_LAYERS * t / 100)))
        skip_set  = set(ranked[:n_drop])
        csv_path  = os.path.join(out_dir, f"eval_threshold_{t}pct.csv")
        print(f"\n  -- Threshold {t}%: skipping layers {sorted(skip_set)} --")
        set_seed(42)
        fold_df = run_loso_with_skipped_layers(
            df, input_cols, label_col, subject_col,
            all_classes, args.checkpoint_path, args,
            layers_to_skip=skip_set, num_channels=num_channels,
        )
        mean_acc = fold_df["accuracy"].mean()
        mean_f1  = fold_df["f1_score"].mean()
        fold_df.to_csv(csv_path, index=False)
        print(f"  Threshold {t}% -> Acc={mean_acc:.4f}  F1={mean_f1:.4f}")
        eval_rows.append({
            "threshold": t,
            "n_layers_skipped": n_drop,
            "skipped_layers": str(sorted(skip_set)),
            "mean_acc": mean_acc,
            "mean_f1":  mean_f1,
            "acc_drop": baseline_acc - mean_acc,
            "f1_drop":  baseline_f1  - mean_f1,
        })

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(os.path.join(out_dir, "pruning_eval_results.csv"), index=False)

    # ── [4] Plots ─────────────────────────────────────────────────────────────
    print(f"\n[4/4] Generating 3 plots ...")

    plot_gm_bargraph(layer_gm, dataset, args.split,
                     os.path.join(out_dir, "gm_bargraph.png"))

    plot_pruning_impact(eval_rows, baseline_acc, baseline_f1, dataset, args.split,
                        os.path.join(out_dir, "gm_pruning_impact.png"))

    plot_param_table(param_df, THRESHOLDS,
                     os.path.join(out_dir, "gm_param_table.png"))

    # Summary
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}  ({args.split})")
    print(f"  Baseline  -> Acc={baseline_acc:.4f}  F1={baseline_f1:.4f}")
    print(f"  {'Threshold':<12} {'Skipped':<10} {'Acc':<10} {'F1':<10} {'Acc Drop':<10} {'F1 Drop'}")
    for r in eval_rows:
        print(f"  {r['threshold']:<12} {r['n_layers_skipped']:<10} "
              f"{r['mean_acc']:.4f}     {r['mean_f1']:.4f}     "
              f"{r['acc_drop']:+.4f}    {r['f1_drop']:+.4f}")
    print(f"\n  Most important  (DO NOT prune): layers {ranked[-2:]}")
    print(f"  Least important (safe to prune): layers {ranked[:2]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
