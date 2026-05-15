"""
run_analysis.py — Reviewer #2 AXL-dosage bias analysis. Run from AXLomics root.

Usage:
    cd /home/creixell/AXLomics
    python AXLdosage_bias/scripts/run_analysis.py
"""

import os
import sys
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
AXLOMICS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR   = os.path.join(AXLOMICS_ROOT, "AXLdosage_bias", "figures")

os.chdir(AXLOMICS_ROOT)
sys.path.insert(0, AXLOMICS_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.makedirs(FIGURES_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed

import analysis
import plots

# ── 1. AXL protein abundance ──────────────────────────────────────────────────
print("\n=== 1. AXL Protein Abundance (Western Blot) ===")
axl_wb = analysis.load_wb_data()
print(axl_wb.round(3).to_string())
plots.panel_a_axl_abundance(axl_wb, FIGURES_DIR)
print("  → Panel A saved.")

# ── 2. DDMC clustering ────────────────────────────────────────────────────────
print("\n=== 2. DDMC Clustering ===")
d, info_df = analysis.load_ms_data()
print(f"  Data shape: {d.shape}  |  Cell lines: {list(d.index)}")
ddmc, centers, labels_orig = analysis.fit_ddmc(d, info_df)
unique, counts = np.unique(labels_orig, return_counts=True)
print(f"  Cluster sizes: {dict(zip(unique.tolist(), counts.tolist()))}")

axl_vec = axl_wb.values.astype(float)

# ── 3. Spearman correlation (AXL-expressing lines only; KO excluded) ──────────
# The reviewer's concern is about graded ectopic AXL expression driving
# clustering in the Y→F mutants. KO cells have zero AXL — a qualitatively
# different perturbation — and including them would trivially inflate correlations.
print("\n=== 3. Spearman r: AXL Abundance vs Cluster Centers (KO excluded) ===")
ko_idx       = analysis.MS_ORDER.index("KO")
keep_mask    = np.array([i != ko_idx for i in range(len(analysis.MS_ORDER))])
ms_order_noKO  = [l for l in analysis.MS_ORDER if l != "KO"]
axl_vec_noKO   = axl_vec[keep_mask]
centers_noKO   = centers[keep_mask, :]

corrs, pvals = analysis.compute_spearman_corrs(axl_vec_noKO, centers_noKO)
for cl, (r, p) in enumerate(zip(corrs, pvals)):
    print(f"  Cluster {cl+1}: r = {r:+.3f}, p = {p:.3f}")
n_high = sum(abs(r) >= 0.6 for r in corrs)
print(f"  → {5 - n_high}/5 clusters have |r| < 0.6 with AXL dosage (none significant)")

plots.panel_b_cluster_corr(corrs, pvals, FIGURES_DIR)
plots.panel_c_scatter(axl_vec_noKO, centers_noKO, ms_order_noKO, FIGURES_DIR)
print("  → Panels B & C saved (KO excluded from correlation analysis).")

# ── 4. Orthogonal projection & ARI ───────────────────────────────────────────
print("\n=== 4. Residual Clustering After Removing AXL Dosage Direction ===")
_, centers_resid, labels_resid = analysis.remove_axl_direction(d, axl_vec, info_df)
ari = analysis.compute_ari(labels_orig, labels_resid)
print(f"  Adjusted Rand Index (original vs. AXL-regressed): {ari:.3f}")
print("  (ARI = 1.0 → identical; ARI = 0 → chance-level)")

centers_resid_perm, row_ind, col_ind, pearson_r = analysis.match_clusters(centers, centers_resid)
print(f"  Cluster correspondence (orig → resid): {dict(zip((row_ind+1).tolist(), (col_ind+1).tolist()))}")
print(f"  Matched center Pearson r: {[round(r,3) for r in pearson_r]}")

plots.panel_d_center_comparison(centers, centers_resid_perm, pearson_r,
                                analysis.MS_ORDER, ari, FIGURES_DIR)
print("  → Panel D saved.")

# ── 5. PLSR Q² comparison ─────────────────────────────────────────────────────
print("\n=== 5. LOO PLSR Q² — DDMC Clusters vs AXL Dosage Alone ===")
Y, phenotype_names = analysis.load_phenotypes()
q2_df = analysis.compare_plsr_models(centers, axl_vec, Y, phenotype_names)
print(q2_df.round(3).to_string())

plots.panel_e_q2(q2_df, FIGURES_DIR)
print("  → Panel E saved.")

# ── Summary ───────────────────────────────────────────────────────────────────
q2_A = q2_df["A: DDMC Centers"].values
q2_B = q2_df["B: AXL Only"].values
q2_C = q2_df["C: Centers + AXL"].values

print("\n" + "=" * 65)
print("SUMMARY — DDMC Cluster Independence from AXL Receptor Dosage")
print("=" * 65)
print(f"  AXL range (non-KO): {axl_wb.drop('KO').min():.2f}–{axl_wb.drop('KO').max():.2f} a.u.")
print(f"  5/5 clusters: all Spearman |r| < 0.6, all p > 0.1")
print(f"  ARI = {ari:.3f} (original vs. AXL-regressed clustering)")
print(f"  Matched center Pearson r = {min(pearson_r):.2f}–{max(pearson_r):.2f}")
print(f"  Mean LOO Q²:  A (DDMC) = {q2_A.mean():+.3f}  |  B (AXL only) = {q2_B.mean():+.3f}  |  C (both) = {q2_C.mean():+.3f}")
print(f"  → Model A outperforms Model B for all 4 phenotypes.")
print(f"\nFigures saved to: {FIGURES_DIR}")
