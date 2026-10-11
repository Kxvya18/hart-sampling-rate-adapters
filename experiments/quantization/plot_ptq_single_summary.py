#!/usr/bin/env python3
"""Create one slide-ready PTQ summary plot from the finetuned best-checkpoint CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


DEFAULT_CSV = Path(
    "/home/user/suyog/quantization_pilot/results/ptq_finetuned_bestckpt_summary.csv"
)
DEFAULT_OUTPUT = Path(
    "/home/user/suyog/quantization_pilot/results/plots/ptq_f1_retention_compression_single.png"
)


DISPLAY_NAMES = {
    "falldet": "FallDet",
    "utd_mhad": "UTD-MHAD",
    "uci_har": "UCI-HAR",
    "wisdm": "WISDM",
    "motion_sense": "MotionSense",
    "har_sense": "HARSense",
    "overall": "Overall",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a single PTQ figure showing F1, retention, and compression."
    )
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def format_dataset_names(series: pd.Series) -> list[str]:
    return [DISPLAY_NAMES.get(value, value.replace("_", " ").title()) for value in series]


def add_value_labels(ax, bars, fmt: str, dy: float = 0.01, fontsize: int = 9) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#2f3437",
        )


def make_plot(df: pd.DataFrame, output_path: Path) -> None:
    df = df.copy()
    dataset_rows = df[df["dataset"] != "overall"].copy()
    overall_rows = df[df["dataset"] == "overall"].copy()
    plot_df = pd.concat([dataset_rows, overall_rows], ignore_index=True)

    labels = format_dataset_names(plot_df["dataset"])
    x = np.arange(len(plot_df))
    width = 0.34
    overall_idx = labels.index("Overall") if "Overall" in labels else None

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#c9d1d3",
            "axes.linewidth": 0.9,
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(13.2, 6.9), dpi=240)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.35], hspace=0.10)
    ax_f1 = fig.add_subplot(gs[0])
    ax_comp = fig.add_subplot(gs[1], sharex=ax_f1)

    fp32_color = "#6f8fb8"
    int8_color = "#d9968b"
    retention_color = "#3d7c74"
    compression_color = "#b66a4f"
    overall_band = "#f2f5f5"

    if overall_idx is not None:
        for ax in (ax_f1, ax_comp):
            ax.axvspan(
                overall_idx - 0.55,
                overall_idx + 0.55,
                color=overall_band,
                zorder=0,
            )

    fp32_bars = ax_f1.bar(
        x - width / 2,
        plot_df["fp32_f1_mean"],
        width,
        color=fp32_color,
        edgecolor="#ffffff",
        linewidth=0.9,
        label="FP32 F1",
        zorder=3,
    )
    int8_bars = ax_f1.bar(
        x + width / 2,
        plot_df["int8_f1_mean"],
        width,
        color=int8_color,
        edgecolor="#ffffff",
        linewidth=0.9,
        label="INT8 F1",
        zorder=3,
    )

    add_value_labels(ax_f1, fp32_bars, "{:.3f}", dy=0.010, fontsize=8)
    add_value_labels(ax_f1, int8_bars, "{:.3f}", dy=0.010, fontsize=8)

    ax_ret = ax_f1.twinx()
    ax_ret.plot(
        x,
        plot_df["retention_mean_pct"],
        color=retention_color,
        marker="o",
        markersize=6.5,
        linewidth=2.4,
        label="F1 Retention",
        zorder=5,
    )
    ax_ret.axhline(100, color="#7c8588", linestyle=(0, (3, 3)), linewidth=1.2, alpha=0.8)
    for xi, value in zip(x, plot_df["retention_mean_pct"]):
        ax_ret.text(
            xi,
            value + (0.35 if value < 100 else -0.55),
            f"{value:.1f}%",
            ha="center",
            va="bottom" if value < 100 else "top",
            fontsize=8.5,
            color=retention_color,
            fontweight="semibold",
        )

    ax_f1.set_ylim(0, 0.98)
    ax_f1.set_ylabel("Macro F1 score", fontweight="semibold", color="#2f3437")
    ax_f1.grid(axis="y", color="#d9e0e2", linewidth=0.8, alpha=0.75)
    ax_f1.set_axisbelow(True)
    ax_f1.tick_params(axis="x", labelbottom=False)
    ax_f1.set_title(
        "PTQ INT8 Quantization: F1 Preservation with Model Compression",
        fontsize=15,
        color="#2f3437",
        pad=14,
    )
    ax_ret.set_ylim(94.5, 105.2)
    ax_ret.set_ylabel("F1 retention (%)", fontweight="semibold", color=retention_color)
    ax_ret.tick_params(axis="y", colors=retention_color)
    ax_ret.spines["right"].set_color("#bad2ce")

    comp_bars = ax_comp.bar(
        x,
        plot_df["compression_ratio_mean"],
        width=0.56,
        color=compression_color,
        edgecolor="#ffffff",
        linewidth=0.9,
        zorder=3,
    )
    for bar, value in zip(comp_bars, plot_df["compression_ratio_mean"]):
        ax_comp.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#2f3437",
            fontweight="bold",
        )

    ax_comp.axhline(
        plot_df["compression_ratio_mean"].mean(),
        color="#7a6258",
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        alpha=0.7,
    )
    ax_comp.set_ylim(0, max(2.35, plot_df["compression_ratio_mean"].max() + 0.25))
    ax_comp.set_ylabel("Compression\nratio", fontweight="semibold", color="#2f3437")
    ax_comp.grid(axis="y", color="#e3e8e9", linewidth=0.8, alpha=0.75)
    ax_comp.set_axisbelow(True)
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(labels, rotation=0, ha="center", fontsize=9.5)

    handles = [
        Patch(facecolor=fp32_color, edgecolor="none", label="FP32 F1"),
        Patch(facecolor=int8_color, edgecolor="none", label="INT8 F1"),
        plt.Line2D([0], [0], color=retention_color, marker="o", linewidth=2.4, label="F1 retention"),
        Patch(facecolor=compression_color, edgecolor="none", label="Compression ratio"),
    ]
    ax_f1.legend(
        handles=handles,
        loc="upper left",
        ncol=4,
        frameon=True,
        framealpha=1,
        facecolor="white",
        edgecolor="#d8dee0",
        fontsize=9.5,
    )

    if overall_idx is not None:
        ax_comp.text(
            overall_idx,
            -0.34,
            "aggregate",
            transform=ax_comp.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
            color="#687174",
        )

    overall = overall_rows.iloc[0] if not overall_rows.empty else plot_df.mean(numeric_only=True)
    note = (
        f"Overall: FP32 F1 {overall['fp32_f1_mean']:.3f} -> INT8 F1 "
        f"{overall['int8_f1_mean']:.3f} | retention {overall['retention_mean_pct']:.2f}% "
        f"| compression {overall['compression_ratio_mean']:.2f}x"
    )
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=10.5, color="#3f474a")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    make_plot(df, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
