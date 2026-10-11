import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report plots from PTQ summary CSV")
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="/home/user/suyog/quantization_pilot/results/ptq_finetuned_bestckpt_summary.csv",
        help="Path to the merged PTQ summary CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/user/suyog/quantization_pilot/results/plots",
        help="Directory where plots will be saved",
    )
    return parser.parse_args()


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate overall row if present
    df_datasets = df[df["dataset"] != "overall"].copy()
    df_overall = df[df["dataset"] == "overall"].copy()
    
    datasets = df_datasets["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.35

    plt.rcParams["font.size"] = 10
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#F8F9FA"

    # ============ 1) COMPRESSION RATIO - HORIZONTAL BARS (IMPRESSIVE) ============
    fig, ax = plt.subplots(figsize=(10, 6))
    compression_vals = df_datasets["compression_ratio_mean"].values
    colors_comp = ["#2E86DE" if c < 2.05 else "#27AE60" if c < 2.10 else "#E74C3C" for c in compression_vals]
    bars = ax.barh(datasets, compression_vals, height=0.6, color=colors_comp, edgecolor="black", linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, compression_vals)):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.2f}x", 
                va="center", fontweight="bold", fontsize=10)
    
    # Add overall average line
    if not df_overall.empty:
        overall_comp = df_overall["compression_ratio_mean"].values[0]
        ax.axvline(overall_comp, color="#E67E22", linestyle="--", linewidth=2.5, label=f"Overall Avg: {overall_comp:.2f}x")
        ax.legend(loc="lower right", frameon=True, fancybox=True, fontsize=11)
    
    ax.set_xlabel("Compression Ratio (Model Size Reduction)", fontsize=12, fontweight="bold")
    ax.set_title("INT8 Model Compression: Per-Dataset Breakdown", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(compression_vals) + 0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ptq_compression_ratio.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============ 2) F1 RETENTION - LINE PLOT WITH EMPHASIS ON 100% BASELINE ============
    fig, ax = plt.subplots(figsize=(11, 6))
    retention_vals = df_datasets["retention_mean_pct"].values
    
    # Plot 100% baseline
    ax.axhline(100, color="gray", linestyle="-", linewidth=2, alpha=0.5, label="Perfect Retention (100%)")
    
    # Plot retention line
    line = ax.plot(x, retention_vals, marker="o", markersize=10, linewidth=3, color="#C0392B", label="F1 Retention (%)", zorder=3)
    
    # Shade the region between 0 and retention
    ax.fill_between(x, 0, retention_vals, alpha=0.2, color="#C0392B")
    
    # Add value labels on points
    for i, (xi, val) in enumerate(zip(x, retention_vals)):
        offset = 2 if val < 100 else -3
        ax.text(xi, val + offset, f"{val:.2f}%", ha="center", fontweight="bold", fontsize=10)
    
    # Add overall average line
    if not df_overall.empty:
        overall_ret = df_overall["retention_mean_pct"].values[0]
        ax.axhline(overall_ret, color="#27AE60", linestyle="--", linewidth=2.5, 
                  label=f"Overall Avg: {overall_ret:.2f}%", zorder=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("F1 Score Retention (%)", fontsize=12, fontweight="bold")
    ax.set_title("INT8 Quantization: F1 Score Retention vs FP32", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(95, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, fancybox=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "ptq_f1_retention.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============ 3) RETENTION vs COMPRESSION SCATTER (EFFICIENCY FRONTIER) ============
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot for each dataset
    scatter = ax.scatter(df_datasets["compression_ratio_mean"], 
                        df_datasets["retention_mean_pct"],
                        s=300, alpha=0.7, c=range(len(datasets)), cmap="tab20", 
                        edgecolors="black", linewidth=1.5, zorder=3)
    
    # Annotate each point with dataset name
    for i, (comp, ret, dataset) in enumerate(zip(df_datasets["compression_ratio_mean"], 
                                                  df_datasets["retention_mean_pct"], 
                                                  datasets)):
        ax.annotate(dataset, (comp, ret), xytext=(8, 8), textcoords="offset points", 
                   fontsize=9, fontweight="bold")
    
    # Add overall point if present
    if not df_overall.empty:
        overall_comp = df_overall["compression_ratio_mean"].values[0]
        overall_ret = df_overall["retention_mean_pct"].values[0]
        ax.scatter(overall_comp, overall_ret, s=500, marker="*", color="#E67E22", 
                  edgecolors="black", linewidth=2, zorder=4, label=f"Overall Avg")
        ax.annotate("OVERALL", (overall_comp, overall_ret), xytext=(10, 10), 
                   textcoords="offset points", fontsize=10, fontweight="bold", color="#E67E22")
    
    # Add reference zones
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax.fill_between([1.9, 2.15], 95, 105, alpha=0.1, color="green", label="Ideal Zone")
    
    ax.set_xlabel("Compression Ratio (x)", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1 Retention (%)", fontsize=12, fontweight="bold")
    ax.set_title("Efficiency Frontier: Compression vs F1 Retention Trade-off", 
                fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=True, fancybox=True, fontsize=11)
    ax.set_xlim(1.95, 2.15)
    ax.set_ylim(95, 105)
    fig.tight_layout()
    fig.savefig(output_dir / "ptq_efficiency_frontier.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============ 4) PROFESSIONAL DASHBOARD (COMBINED) ============
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top-left: Compression
    ax1 = fig.add_subplot(gs[0, 0])
    colors_comp = ["#2E86DE" if c < 2.05 else "#27AE60" if c < 2.10 else "#E74C3C" for c in df_datasets["compression_ratio_mean"]]
    bars1 = ax1.bar(datasets, df_datasets["compression_ratio_mean"], color=colors_comp, 
                    edgecolor="black", linewidth=1, alpha=0.8)
    for bar, val in zip(bars1, df_datasets["compression_ratio_mean"]):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}x", 
                ha="center", va="bottom", fontweight="bold", fontsize=9)
    if not df_overall.empty:
        overall_comp = df_overall["compression_ratio_mean"].values[0]
        ax1.axhline(overall_comp, color="#E67E22", linestyle="--", linewidth=2, label=f"Avg: {overall_comp:.2f}x")
        ax1.legend(fontsize=9)
    ax1.set_ylabel("Compression Ratio (x)", fontweight="bold")
    ax1.set_title("Compression Ratio", fontweight="bold", fontsize=11)
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.3)
    
    # Top-right: F1 Retention
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axhline(100, color="gray", linestyle="-", linewidth=1.5, alpha=0.5)
    line2 = ax2.plot(x, df_datasets["retention_mean_pct"], marker="o", markersize=8, 
                     linewidth=2.5, color="#C0392B", label="F1 Retention")
    ax2.fill_between(x, 0, df_datasets["retention_mean_pct"], alpha=0.15, color="#C0392B")
    for xi, val in zip(x, df_datasets["retention_mean_pct"]):
        ax2.text(xi, val + 0.3, f"{val:.2f}%", ha="center", fontsize=8, fontweight="bold")
    if not df_overall.empty:
        overall_ret = df_overall["retention_mean_pct"].values[0]
        ax2.axhline(overall_ret, color="#27AE60", linestyle="--", linewidth=2, label=f"Avg: {overall_ret:.2f}%")
        ax2.legend(fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=20, ha="right")
    ax2.set_ylabel("F1 Retention (%)", fontweight="bold")
    ax2.set_title("F1 Score Retention vs FP32", fontweight="bold", fontsize=11)
    ax2.set_ylim(95.5, 105)
    ax2.grid(axis="y", alpha=0.3)
    
    # Bottom-left: Accuracy Retention
    ax3 = fig.add_subplot(gs[1, 0])
    acc_retention = (df_datasets["int8_acc_mean"] / df_datasets["fp32_acc_mean"] * 100).values
    colors_acc = ["#27AE60" if r > 99.5 else "#F39C12" if r > 98 else "#E74C3C" for r in acc_retention]
    bars3 = ax3.bar(datasets, acc_retention, color=colors_acc, edgecolor="black", linewidth=1, alpha=0.8)
    ax3.axhline(100, color="gray", linestyle="-", linewidth=1.5, alpha=0.5)
    for bar, val in zip(bars3, acc_retention):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.1, f"{val:.2f}%", 
                ha="center", va="bottom", fontweight="bold", fontsize=9)
    if not df_overall.empty:
        overall_acc_ret = (df_overall["int8_acc_mean"].values[0] / df_overall["fp32_acc_mean"].values[0] * 100)
        ax3.axhline(overall_acc_ret, color="#E67E22", linestyle="--", linewidth=2, label=f"Avg: {overall_acc_ret:.2f}%")
        ax3.legend(fontsize=9)
    ax3.set_ylabel("Accuracy Retention (%)", fontweight="bold")
    ax3.set_title("Accuracy Retention: INT8 vs FP32", fontweight="bold", fontsize=11)
    ax3.tick_params(axis="x", rotation=20)
    ax3.set_ylim(95, 105)
    ax3.grid(axis="y", alpha=0.3)
    
    # Bottom-right: Summary stats
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    
    summary_text = "PTQ QUANTIZATION SUMMARY\n" + "="*40 + "\n\n"
    summary_text += f"Datasets Evaluated: {len(datasets)}\n"
    summary_text += f"Total Subjects: {df_datasets['subjects'].sum()}\n\n"
    
    if not df_overall.empty:
        summary_text += "OVERALL METRICS:\n" + "-"*40 + "\n"
        summary_text += f"Avg Compression: {df_overall['compression_ratio_mean'].values[0]:.2f}x\n"
        summary_text += f"Avg F1 Retention: {df_overall['retention_mean_pct'].values[0]:.2f}%\n"
        summary_text += f"Avg F1 Score (FP32): {df_overall['fp32_f1_mean'].values[0]:.4f}\n"
        summary_text += f"Avg F1 Score (INT8): {df_overall['int8_f1_mean'].values[0]:.4f}\n"
        summary_text += f"Avg Accuracy (FP32): {df_overall['fp32_acc_mean'].values[0]:.4f}\n"
        summary_text += f"Avg Accuracy (INT8): {df_overall['int8_acc_mean'].values[0]:.4f}\n"
        acc_ret = (df_overall["int8_acc_mean"].values[0] / df_overall["fp32_acc_mean"].values[0] * 100)
        summary_text += f"Avg Accuracy Retention: {acc_ret:.2f}%\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11, 
            verticalalignment="top", fontfamily="monospace", 
            bbox=dict(boxstyle="round", facecolor="#F0F0F0", alpha=0.8, pad=1))
    
    fig.suptitle("PTQ INT8 Quantization Report: Compression & Retention Analysis", 
                fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(output_dir / "ptq_comprehensive_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)




def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir)
    df = pd.read_csv(summary_csv)
    make_plots(df, output_dir)
    print(f"\n✓ Saved plots to: {output_dir}")
    print(f"✓ Generated 4 professional visualization files:")
    print(f"  - ptq_compression_ratio.png (horizontal bars)")
    print(f"  - ptq_f1_retention.png (line plot with baselines)")
    print(f"  - ptq_efficiency_frontier.png (scatter efficiency plot)")
    print(f"  - ptq_comprehensive_dashboard.png (combined 2x2 dashboard)")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()