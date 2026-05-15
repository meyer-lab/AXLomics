"""
plots.py — Figure panels for the AXL-dosage bias analysis.

Each function saves a PDF to `save_dir` and returns the figure object.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Apply a consistent style once when the module is imported
import matplotlib
matplotlib.rcParams["font.sans-serif"] = "Arial"
sns.set(
    style="whitegrid", font_scale=1.3, color_codes=True, palette="colorblind",
    rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
)


def panel_a_axl_abundance(axl_wb: pd.Series, save_dir: str) -> plt.Figure:
    """Bar plot of normalised AXL protein abundance per cell line (Panel A)."""
    bar_colors = [
        "dimgray" if cl == "KO" else ("#1f77b4" if cl == "PAR" else "#d62728")
        for cl in axl_wb.index
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(axl_wb.index, axl_wb.values, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Cell Line")
    ax.set_ylabel("AXL / Rhodamine (a.u.)")
    ax.set_title("Panel A — AXL Protein Abundance (Western Blot)")
    ax.tick_params(axis="x", rotation=45)

    non_ko_mean = axl_wb.drop("KO").mean()
    ax.axhline(non_ko_mean, color="black", linestyle="--", linewidth=1,
               label=f"Mean (excl. KO) = {non_ko_mean:.2f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "Panel_A_AXL_abundance.pdf"), dpi=300, bbox_inches="tight")
    return fig


def panel_b_cluster_corr(corrs: list, pvals: list, save_dir: str) -> plt.Figure:
    """Bar chart of Spearman r (AXL abundance vs cluster centers) (Panel B)."""
    bar_cols = ["#d62728" if abs(r) >= 0.6 else "#1f77b4" for r in corrs]
    labels   = [f"C{k+1}" for k in range(len(corrs))]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, corrs, color=bar_cols, edgecolor="black", linewidth=0.5)
    ax.axhline(0,    color="black", linewidth=0.8)
    ax.axhline( 0.6, color="gray", linestyle="--", linewidth=0.9, label="|r| = 0.6 threshold")
    ax.axhline(-0.6, color="gray", linestyle="--", linewidth=0.9)

    for k, (r, p) in enumerate(zip(corrs, pvals)):
        offset = 0.04 if r >= 0 else -0.04
        va     = "bottom" if r >= 0 else "top"
        ax.text(k, r + offset, f"p={p:.2f}", ha="center", va=va, fontsize=9)

    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("DDMC Cluster")
    ax.set_ylabel("Spearman r")
    ax.set_title("Panel B — AXL Abundance vs. Cluster Centers\n(KO excluded)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "Panel_B_cluster_AXL_corr.pdf"), dpi=300, bbox_inches="tight")
    return fig


def panel_c_scatter(axl_vec: np.ndarray, centers: np.ndarray,
                    ms_order: list, save_dir: str) -> plt.Figure:
    """Scatter plots: AXL abundance vs each cluster center signal (Panel C)."""
    n_clusters = centers.shape[1]
    fig, axes = plt.subplots(1, n_clusters, figsize=(16, 3.5))

    for idx, ax in enumerate(axes):
        r, p = spearmanr(axl_vec, centers[:, idx])
        ax.scatter(axl_vec, centers[:, idx], color="#1f77b4", zorder=3,
                   s=55, edgecolors="black", linewidths=0.4)

        for j, label in enumerate(ms_order):
            ax.annotate(label, (axl_vec[j], centers[j, idx]),
                        textcoords="offset points", xytext=(4, 2), fontsize=6.5)

        m, b = np.polyfit(axl_vec, centers[:, idx], 1)
        xfit = np.linspace(axl_vec.min(), axl_vec.max(), 50)
        ax.plot(xfit, m * xfit + b, color="red", linewidth=1, linestyle="--")

        ax.set_xlabel("AXL / Rhodamine (a.u.)", fontsize=10)
        ax.set_ylabel("Cluster signal" if idx == 0 else "", fontsize=10)
        ax.set_title(f"Cluster {idx+1}\nr = {r:+.2f}, p = {p:.2f}", fontsize=10)

    fig.suptitle("Panel C — AXL Abundance vs. Cluster Centers (KO excluded)",
                 y=1.03, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "Panel_C_scatterplots.pdf"), dpi=300, bbox_inches="tight")
    return fig


def panel_d_center_comparison(centers: np.ndarray, centers_resid_perm: np.ndarray,
                               pearson_r: list, ms_order: list,
                               ari: float, save_dir: str) -> plt.Figure:
    """Overlay line plots: original vs AXL-regressed cluster centers (Panel D)."""
    n_clusters = centers.shape[1]
    fig, axes = plt.subplots(1, n_clusters, figsize=(16, 3.5), sharey=False)

    for idx, ax in enumerate(axes):
        ax.plot(ms_order, centers[:, idx], "o-", color="#1f77b4", linewidth=2,
                label="Original", markersize=7)
        ax.plot(ms_order, centers_resid_perm[:, idx], "s--", color="#ff7f0e", linewidth=2,
                label="AXL-regressed", markersize=7, alpha=0.85)
        ax.set_title(f"Cluster {idx+1}\nr = {pearson_r[idx]:.2f}", fontsize=10)
        ax.set_xlabel("Cell Line", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        if idx == 0:
            ax.set_ylabel("Cluster signal")
        if idx == n_clusters - 1:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Panel D — Original vs. AXL-Regressed Cluster Centers  (ARI = {ari:.2f})",
        y=1.03, fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "Panel_D_center_comparison.pdf"),
                dpi=300, bbox_inches="tight")
    return fig


def panel_e_q2(q2_df: pd.DataFrame, save_dir: str) -> plt.Figure:
    """Grouped bar chart of LOO Q² across models and phenotypes (Panel E)."""
    phenotype_names = list(q2_df.index)
    model_lbls = ["A: DDMC Centers", "B: AXL Only", "C: Centers + AXL"]
    model_cols = ["#1f77b4", "#d62728", "#2ca02c"]

    x     = np.arange(len(phenotype_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, (label, color) in enumerate(zip(model_lbls, model_cols)):
        ax.bar(x + (k - 1) * width, q2_df[label], width,
               label=label, color=color, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(phenotype_names, rotation=15)
    ax.set_xlabel("Phenotype (A/E treatment)")
    ax.set_ylabel("Q² (LOO cross-validation)")
    ax.set_title("Panel E — Phenotype Prediction: DDMC Clusters vs. AXL Dosage Alone")
    ax.legend(title="Model", fontsize=9, title_fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "Panel_E_Q2_comparison.pdf"),
                dpi=300, bbox_inches="tight")
    return fig
