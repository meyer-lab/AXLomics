"""
analysis.py — Core data loading and computation for the AXL-dosage bias analysis.

All functions assume the msresist package is importable (i.e. sys.path contains
the AXLomics root, or the notebook sets os.chdir to that root before importing).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import adjusted_rand_score

from msresist.pre_processing import preprocessing
from msresist.clustering import DDMC
from msresist.figures.common import import_phenotype_data, formatPhenotypesForModeling

# ── Constants ─────────────────────────────────────────────────────────────────
# Biological-label order matching the positional column order in the MS experiment:
# PC9 A → KO A → Kd A → KI A → M4 A → M5 A → M7 A → M10 A → M11 A → M15 A
MS_ORDER = [
    "PAR",
    "KO",
    "KD",
    "WT",
    "Y634F",
    "Y643F",
    "Y698F",
    "Y726F",
    "Y750F",
    "Y821F",
]

WB_PATH = "msresist/data/AXL-Bands-Quant_2026-03-27.csv"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_wb_data() -> pd.Series:
    """Load and Rhodamine-normalise the western blot AXL signal.

    Returns a Series indexed by MS_ORDER (length 10).
    """
    wb = pd.read_csv(WB_PATH, index_col="Cell Line")
    wb["AXL_norm"] = wb["AXL Signal"] / wb["Rhodamine Signal"]
    return wb["AXL_norm"].reindex(MS_ORDER)


def load_ms_data():
    """Load and preprocess the phosphoproteomic mass-spec data.

    Returns
    -------
    d : DataFrame, shape (10, N_peptides)
        Mean-centred log2 intensities; rows = cell lines (MS channel order).
    info_df : DataFrame
        Non-numeric metadata columns (Protein, Sequence, Gene, …).
    """
    X = preprocessing(
        AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True
    )
    d = X.select_dtypes(include=["float64"]).T  # (10, N_peptides)
    info_df = X.select_dtypes(include=["object"])  # peptide metadata
    return d, info_df


# ── DDMC fitting ──────────────────────────────────────────────────────────────


def fit_ddmc(d: pd.DataFrame, info_df: pd.DataFrame):
    """Fit DDMC with the same hyperparameters used in Figure 2.

    Returns
    -------
    ddmc     : fitted DDMC object
    centers  : ndarray (10, 5) — cluster center profiles per cell line
    labels   : ndarray (N_peptides,) — cluster assignment per peptide
    """
    ddmc = DDMC(
        info_df, n_components=5, SeqWeight=2, distance_method="PAM250", random_state=5
    ).fit(d)
    centers = ddmc.transform()  # (10, 5)
    labels = ddmc.labels()  # (N_peptides,)
    return ddmc, centers, labels


# ── Section 3: Spearman correlation ──────────────────────────────────────────


def compute_spearman_corrs(axl_vec: np.ndarray, centers: np.ndarray):
    """Compute Spearman r between AXL abundance and each of the 5 cluster centers.

    Returns
    -------
    corrs : list[float]  — r values for clusters 1–5
    pvals : list[float]  — two-sided p-values
    """
    corrs, pvals = [], []
    for cl in range(centers.shape[1]):
        r, p = spearmanr(axl_vec, centers[:, cl])
        corrs.append(float(r))
        pvals.append(float(p))
    return corrs, pvals


# ── Section 4: Orthogonal projection & ARI ───────────────────────────────────


def remove_axl_direction(d: pd.DataFrame, axl_vec: np.ndarray, info_df: pd.DataFrame):
    """Project out the AXL dosage direction and re-cluster.

    The AXL unit vector is removed from every peptide profile:
        d_resid = d - u_axl (u_axl^T d)

    Returns
    -------
    ari              : float — Adjusted Rand Index between original and residual labels
    centers_resid    : ndarray (10, 5) — residual cluster centers (unordered)
    labels_resid     : ndarray (N_peptides,) — residual cluster labels
    centers_resid_perm : ndarray (10, 5) — residual centers reordered to match original
    col_ind          : ndarray (5,) — Hungarian permutation index
    """
    d_np = d.values.astype(float)
    axl_unit = axl_vec / np.linalg.norm(axl_vec)

    proj_coefs = d_np.T @ axl_unit  # (N_peptides,)
    d_resid_np = d_np - np.outer(axl_unit, proj_coefs)  # (10, N_peptides)

    d_resid = pd.DataFrame(d_resid_np, index=d.index, columns=d.columns)

    ddmc_resid = DDMC(
        info_df, n_components=5, SeqWeight=2, distance_method="PAM250", random_state=5
    ).fit(d_resid)

    centers_resid = ddmc_resid.transform()
    labels_resid = ddmc_resid.labels()

    return d_resid, centers_resid, labels_resid


def match_clusters(centers_orig: np.ndarray, centers_resid: np.ndarray):
    """Use the Hungarian algorithm to match residual clusters to original clusters.

    Returns
    -------
    centers_resid_perm : ndarray (10, 5) — residual centers reordered
    row_ind, col_ind   : permutation indices
    pearson_r          : list[float] — per-matched-cluster Pearson r
    """
    corr_mat = np.corrcoef(centers_orig.T, centers_resid.T)[:5, 5:]  # (5, 5)
    row_ind, col_ind = linear_sum_assignment(-corr_mat)
    centers_resid_perm = centers_resid[:, col_ind]

    pearson_r = [
        float(np.corrcoef(centers_orig[:, k], centers_resid_perm[:, k])[0, 1])
        for k in range(5)
    ]
    return centers_resid_perm, row_ind, col_ind, pearson_r


def compute_ari(labels_orig: np.ndarray, labels_resid: np.ndarray) -> float:
    return float(adjusted_rand_score(labels_orig, labels_resid))


# ── Section 5: PLSR Q² ───────────────────────────────────────────────────────


def load_phenotypes():
    """Load A/E-treatment phenotype matrix (10 lines × 4 phenotypes).

    Returns
    -------
    Y              : ndarray (10, 4)
    phenotype_names : list[str]
    """
    cv = import_phenotype_data(phenotype="Cell Viability")
    red = import_phenotype_data(phenotype="Cell Death")
    sw = import_phenotype_data(phenotype="Migration")
    c = import_phenotype_data(phenotype="Island")
    y = formatPhenotypesForModeling(cv, red, sw, c)
    y = y[y["Treatment"] == "A/E"].drop("Treatment", axis=1).set_index("Lines")
    return y.values.astype(float), list(y.columns)


def loo_q2(X_in: np.ndarray, Y_in: np.ndarray, max_components: int) -> np.ndarray:
    """Leave-one-out cross-validated Q² per phenotype column.

    n_components is capped at min(max_components, n_features, n_train - 1)
    to prevent rank-deficiency with small n.
    """
    loo = LeaveOneOut()
    y_pred = np.zeros_like(Y_in, dtype=float)

    for train_idx, test_idx in loo.split(X_in):
        X_tr, X_te = X_in[train_idx], X_in[test_idx]
        Y_tr = Y_in[train_idx]
        nc = min(max_components, X_tr.shape[1], X_tr.shape[0] - 1)
        plsr = PLSRegression(n_components=nc)
        plsr.fit(X_tr, Y_tr)
        y_pred[test_idx] = plsr.predict(X_te)

    ss_res = np.sum((Y_in - y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y_in - Y_in.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / ss_tot  # (n_phenotypes,)


def compare_plsr_models(
    centers: np.ndarray, axl_vec: np.ndarray, Y: np.ndarray, phenotype_names: list
) -> pd.DataFrame:
    """Compare three LOO-PLSR models and return a Q² DataFrame.

    Models
    ------
    A : DDMC cluster centers (5 features, 3 components)
    B : AXL abundance only   (1 feature,  1 component)
    C : Centers + AXL        (6 features, 3 components)

    nc=3 for models A and C is chosen by maximising mean LOO Q² across nc=1..5:
        nc=1: 0.275  nc=2: 0.390  nc=3: 0.433 ← best  nc=4: 0.311  nc=5: -0.058
    Figure2.ipynb uses nc=4, but with n=10 that overfits (training R²≈0.88 vs Q²≈0.31).
    """
    axl_mat = axl_vec.reshape(-1, 1)
    X_C = np.hstack([centers, axl_mat])

    q2_A = loo_q2(centers, Y, max_components=3)
    q2_B = loo_q2(axl_mat, Y, max_components=1)
    q2_C = loo_q2(X_C, Y, max_components=3)

    return pd.DataFrame(
        {"A: DDMC Centers": q2_A, "B: AXL Only": q2_B, "C: Centers + AXL": q2_C},
        index=phenotype_names,
    )
