"""
This creates Figure 6I-H: AXL and YAP signatures in scRNA-seq of TKI-treated LUAD tumors
"""

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import gseapy as gp
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from venny4py.venny4py import venny4py
from matplotlib.patches import Patch
from scipy.stats import zscore, gmean
from .common import subplotLabel, getSetup, Introduce_Correct_DDMC_labels
from .figure3C_J import preprocess_maynard, annotate_maynard, plot_tsne
from ..pre_processing import preprocessing
from ..clustering import DDMC


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((14, 12), (3, 3), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Load scRNA-seq data
    adata = preprocess_maynard()
    adata, TvsNAT, cc = annotate_maynard(adata)

    # Subset to EGFR-driven tumors
    adata_egfr = adata[adata.obs["Driver gene"] == "EGFR"]
    cc_egfr = cc[(cc.obs["Driver gene"] == "EGFR")]
    TvsNAT_egfr = TvsNAT[TvsNAT.obs["Driver gene"] == "EGFR"]

    # Import AXL pY MS Signaling Data
    X = preprocessing(AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    d = X.select_dtypes(include=['float64']).T
    i = X.select_dtypes(include=['object'])

    # Fit DDMC
    ddmc = DDMC(i, n_components=5, SeqWeight=2, distance_method="PAM250", random_state=5).fit(d)

    X = Introduce_Correct_DDMC_labels(X, ddmc)
    C2_3 = X[(X["Cluster"] == 2) | (X["Cluster"] == 3)][["Cluster", "Gene", "Position", "PC9 A", "KO A"]]

    cordenonsi_genes, wp4534_genes, yap_up_genes = get_YAPsignature_genes()

    gs_dict = {
        "Cordenonsi YAP": sanitize_genes(TvsNAT, cordenonsi_genes),
        "WP4534":         sanitize_genes(TvsNAT, wp4534_genes),
        "YAP1_UP":        sanitize_genes(TvsNAT, yap_up_genes),
        "AXL C2":  sanitize_genes(TvsNAT, axl_clusters_dict[2]),
        "AXL C3":  sanitize_genes(TvsNAT, axl_clusters_dict[3])
    }


    return f



def get_YAPsignature_genes():
    cordenonsi_genes = gp.get_library(name="MSigDB_Oncogenic_Signatures", organism="Human")['CORDENONSI YAP CONSERVED SIGNATURE']
    wp4534_genes = gp.get_library(name="WikiPathways_2024_Human")['Mechanoreg And Pathology Of YAP TAZ Hippo And Non Hippo Mechs WP4534']
    yap_up_genes = gp.get_library(name="MSigDB_Oncogenic_Signatures", organism="Human")['YAP1 UP']
    return cordenonsi_genes, wp4534_genes, yap_up_genes


def violin_plot_YAPsignatures(cc_egfr, C2_3, savefig=False, print_mann_whitney_test=False):
    """
    Create violin plots for YAP signatures in EGFR-driven LUAD tumors.
    This function computes gene signature scores for several YAP-related gene sets
    (Cordenonsi YAP, WP4534 YAP, YAP1 UP) and an AXL signature, then visualizes
    their distributions across treatment timepoints using violin plots.
    Parameters
    ----------
    cc_egfr : AnnData
        Annotated data matrix (typically single-cell RNA-seq data) containing gene expression data for EGFR-driven LUAD tumors.
    C2_3 : pandas.DataFrame
        DataFrame containing at least a "Gene" column, representing the gene list for the AXL signature.
    savefig : bool or str, optional (default: False)
        If False, the plot is not saved. If a string (file path), the plot is saved to the specified file.

    Returns
    -------
    None
        The function creates and displays a violin plot. Optionally saves the figure if `savefig` is provided.
    """

    cordenonsi_genes, wp4534_genes, yap_up_genes = get_YAPsignature_genes()

    sc.tl.score_genes(cc_egfr, cordenonsi_genes, score_name="Cordenonsi YAP")
    sc.tl.score_genes(cc_egfr, wp4534_genes, score_name="WP4534 YAP")
    sc.tl.score_genes(cc_egfr, yap_up_genes, score_name="YAP1 UP")
    sc.tl.score_genes(cc_egfr, C2_3["Gene"].values, score_name="AXL signature")

    # Melt your data for seaborn
    df = sc.get.obs_df(cc_egfr, ["Treatment timepoint with TKI", "WP4534 YAP", "Cordenonsi YAP", "YAP1 UP"])

    # Melt for easier plotting
    df_melt = df.melt(id_vars=["Treatment timepoint with TKI"], 
                    value_vars=["WP4534 YAP", "Cordenonsi YAP", "YAP1 UP"],
                    var_name="Signature", value_name="Score")

    _, ax = plt.subplots(1, 1, figsize=(8, 7),)
    sns.violinplot(df_melt, x="Signature", y="Score", hue="Treatment timepoint with TKI", ax=ax)

    # Remove legend from each subplot since it's redundant
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        # Rotate x-tick labels by 90 degrees
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

    # Add a single legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=300)

    if print_mann_whitney_test:
        print_mann_whitney_results(df)


def print_mann_whitney_results(df):
    """
    Performs Mann-Whitney U tests to compare gene signature scores between treatment groups and prints the resulting p-values.

    This function compares the scores of three gene signatures ("Cordenonsi YAP", "WP4534 YAP", "YAP1 UP") between the following treatment groups:
    - Progressive Disease (PD) vs Residual Disease (RD)
    - Progressive Disease (PD) vs TKI naive (TN)

    For each signature, the function prints the p-values for both comparisons.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the following columns:
            - "Cordenonsi YAP": Numeric scores for the Cordenonsi YAP gene signature.
            - "WP4534 YAP": Numeric scores for the WP4534 YAP gene signature.
            - "YAP1 UP": Numeric scores for the YAP1 UP gene signature.
            - "Treatment timepoint with TKI": Categorical labels indicating treatment group ("Progressive disease (PD)", "Residual disease (RD)", "TKI naive (TN)").

    Returns
    -------
    None
        This function prints the p-values for each comparison and does not return any value.
    """

    cordenonsi_scores = df["Cordenonsi YAP"]
    yap1_up_scores = df["YAP1 UP"]
    wp4534_scores = df["WP4534 YAP"]
    groups = df["Treatment timepoint with TKI"].values

    PD_mask = groups == "Progressive disease (PD)"
    RD_mask = groups == "Residual disease (RD)"
    TN_mask = groups == "TKI naive (TN)"

    # Cordenonsi YAP
    _, pval3 = mannwhitneyu(cordenonsi_scores[PD_mask], cordenonsi_scores[RD_mask], alternative='two-sided')
    _, pval4 = mannwhitneyu(cordenonsi_scores[PD_mask], cordenonsi_scores[TN_mask], alternative='two-sided')
    print(f"Cordenonsi YAP: PD vs RD p={pval3:.3e}, PD vs TKI naive (TN) p={pval4:.3e}")

    # WP4534 YAP
    _, pval5 = mannwhitneyu(wp4534_scores[PD_mask], wp4534_scores[RD_mask], alternative='two-sided')
    _, pval6 = mannwhitneyu(wp4534_scores[PD_mask], wp4534_scores[TN_mask], alternative='two-sided')
    print(f"WP4534 YAP: PD vs RD p={pval5:.3e}, PD vs TKI naive (TN) p={pval6:.3e}")

    # YAP1 UP
    _, pval7 = mannwhitneyu(yap1_up_scores[PD_mask], yap1_up_scores[RD_mask], alternative='two-sided')
    _, pval8 = mannwhitneyu(yap1_up_scores[PD_mask], yap1_up_scores[TN_mask], alternative='two-sided')
    print(f"YAP1 UP: PD vs RD p={pval7:.3e}, PD vs TKI naive (TN) p={pval8:.3e}")


# ------------------------------
# 1) Correlation + permutation test at pseudobulk level
# ------------------------------
def permutation_corr_pval(x, y, adata, id_col='Subject ID', n_permutations=10000, random_state=None):
    """
    Pearson r and permutation p-value at pseudobulk (patient) level.
    """
    rng = np.random.default_rng(random_state)
    if id_col not in adata.obs.columns:
        raise KeyError(f"'{id_col}' not found in adata.obs")
    df = pd.DataFrame({'x': x, 'y': y, id_col: adata.obs[id_col].values})
    pseudobulk = df.groupby(id_col, as_index=False).mean()

    xpb = pseudobulk['x'].values
    ypb = pseudobulk['y'].values
    observed_r = np.corrcoef(xpb, ypb)[0, 1]

    if n_permutations and n_permutations > 0:
        permuted_rs = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            permuted_rs[i] = np.corrcoef(xpb, rng.permutation(ypb))[0, 1]
        pval = np.mean(np.abs(permuted_rs) >= np.abs(observed_r))
        return observed_r, pval, permuted_rs, xpb, ypb
    else:
        return observed_r, np.nan, None, xpb, ypb


# ------------------------------
# 2) Quick visualization for a single pair (patient-level)
# ------------------------------
def plot_perm_result(adata, x_name, y_name, n_permutations=10000, random_state=7, 
                     additional_y_names=None):
    """
    Scatter of pseudobulked patients + permutation null.
    Allows plotting multiple y variables (e.g., YAP signatures) with different colors.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing observations and variables
    x_name : str
        Name of x variable (e.g., "AXL signature")
    y_name : str
        Name of primary y variable (e.g., "YAP1 UP")
    additional_y_names : list
        List of additional y variables to plot (e.g., ["Cordenonsi YAP", "WP4534 YAP"])
    n_permutations : int
        Number of permutations for p-value calculation
    random_state : int
        Random seed for reproducibility
    """
    # Ensure additional_y_names is a list or None
    if additional_y_names is None:
        additional_y_names = []
    elif isinstance(additional_y_names, str):
        additional_y_names = [additional_y_names]
    
    # Initialize with primary variable
    x = adata.obs[x_name]
    y = adata.obs[y_name]
    r, p, null_r, xpb, ypb = permutation_corr_pval(x, y, adata, n_permutations=n_permutations, random_state=random_state)

    # Create a figure with 2 subplots side by side
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Set up color palette for different signatures
    all_signatures = [y_name]
    all_signatures.extend(additional_y_names)
    palette = sns.color_palette("tab10", len(all_signatures))
    
    # Scatter plot and fit line for primary variable
    ax1.scatter(xpb, ypb, s=50, alpha=0.8, color=palette[0], label=f"{y_name}, r={r:.2f}, p={p:.3g}")
    m, b = np.polyfit(xpb, ypb, 1)
    xs = np.linspace(xpb.min(), xpb.max(), 100)
    ax1.plot(xs, m*xs + b, color=palette[0], linewidth=2)
    
    # Add additional variables if provided
    additional_results = []
    for i, add_y_name in enumerate(additional_y_names, 1):
        add_y = adata.obs[add_y_name]
        add_r, add_p, add_null_r, add_xpb, add_ypb = permutation_corr_pval(
            x, add_y, adata, n_permutations=n_permutations, random_state=random_state
        )
        additional_results.append((add_r, add_p, add_null_r, add_xpb, add_ypb))
        
        # Add scatter and fit line for this variable with increased transparency
        ax1.scatter(add_xpb, add_ypb, s=50, alpha=0.2, color=palette[i], 
                    label=f"{add_y_name}, r={add_r:.2f}, p={add_p:.3g}")
        add_m, add_b = np.polyfit(add_xpb, add_ypb, 1)
        ax1.plot(xs, add_m*xs + add_b, color=palette[i], linewidth=2, alpha=0.4)
    
    ax1.set_title(f"Patient-level {x_name} vs YAP Signatures")
    ax1.set_xlabel(f"Pseudobulk {x_name}")
    ax1.set_ylabel("Pseudobulk YAP Signature Scores")
    ax1.legend(fontsize=9, loc='best')
    
    # Histogram in the second subplot
    ax2.hist(null_r, bins=40, alpha=0.7, color=palette[0])
    ax2.axvline(r, color=palette[0], linestyle='--', linewidth=2, label=f"{y_name}: r={r:.2f}")
    
    # Add vertical lines for additional variables
    for i, (add_y_name, result) in enumerate(zip(additional_y_names, additional_results), 1):
        add_r, _, add_null_r, _, _ = result
        # Optional: Also plot histogram for additional null distributions (commented out to avoid clutter)
        # ax2.hist(add_null_r, bins=40, alpha=0.3, color=palette[i])
        ax2.axvline(add_r, color=palette[i], linestyle='--', linewidth=2, alpha=0.6,
                   label=f"{add_y_name}: r={add_r:.2f}")
    
    ax2.set_xlabel("Permutation r")
    ax2.set_ylabel("Count")
    ax2.set_title("Permutation null distribution")
    ax2.legend(fontsize=9, loc='best')
    
    plt.tight_layout()


def plot_venn_diagram_YAP_AXL_signatures(C2_3, savefig=False, return_venn_dict=False):
    """
    Plots a 4-way Venn diagram showing the overlap between three YAP gene signatures and the AXL signature.

    Parameters
    ----------
    cordenonsi_genes : list or set
        List or set of gene names for the Cordenonsi YAP signature.
    wp4534_genes : list or set
        List or set of gene names for the WP4534 YAP signature.
    yap_up_genes : list or set
        List or set of gene names for the YAP1 UP signature.
    C2_3 : pandas.DataFrame
        DataFrame containing at least a "Gene" column and a "Cluster" column, used to define the AXL signature genes (clusters 2 and 3).
    savefig : bool or str, optional (default: False)
        If False, the plot is not saved. If a string (file path), the plot is saved to the specified file.
    return_venn_dict : bool
        If True, returns a dictionary of the gene lists used for each set.

    Returns
    -------
    venn_dict : dict, optional
        If return_venn_dict is True, returns a dictionary with gene lists for each set.
        Otherwise, displays the Venn diagram and optionally saves it to file.
    """

    cordenonsi_genes, wp4534_genes, yap_up_genes = get_YAPsignature_genes()

    axl_genes = list(dict.fromkeys(
    C2_3[C2_3["Cluster"] == 2]["Gene"].tolist() +
    C2_3[C2_3["Cluster"] == 3]["Gene"].tolist()
    ))
    
    # Convert gene lists to sets
    set_cordenonsi = set(cordenonsi_genes)
    set_wp4534 = set(wp4534_genes)
    set_yap_up = set(yap_up_genes)
    set_axl = set(axl_genes)

    # Your sets
    venn_dict = {
        "Cordenonsi YAP": set_cordenonsi,
        "WP4534 YAP": set_wp4534,
        "YAP1 UP": set_yap_up,
        "AXL signature": set_axl,
    }

    # Order of labels to control both legend and colors
    order = ["Cordenonsi YAP", "WP4534 YAP", "YAP1 UP", "AXL signature"]

    # Colors must be a list in the same order
    color_map = {
        "Cordenonsi YAP": "#4292c6",  # blue
        "WP4534 YAP":     "#41ab5d",  # green
        "YAP1 UP":        "#feb24c",  # orange
        "AXL signature":  "#d73027",  # red
    }
    color_list = [color_map[k] for k in order]

    # Rebuild an ordered dict for consistent labeling
    venn_ordered = {k: venn_dict[k] for k in order}

    plt.figure(figsize=(8, 8))
    venny4py(venn_ordered, colors=color_list)
    plt.title("Overlap between YAP signatures and AXL signature")

    if savefig:
        plt.savefig(savefig, bbox_inches='tight')

    if return_venn_dict:
        axl_clusters_dict = {
        2: C2_3[C2_3["Cluster"] == 2]["Gene"].tolist(),
        3: C2_3[C2_3["Cluster"] == 3]["Gene"].tolist()
        }

        venn_dict = {
            "AXL C2": list(dict.fromkeys(C2_3[C2_3["Cluster"] == 2]["Gene"].tolist())),
            "AXL C3": list(dict.fromkeys(C2_3[C2_3["Cluster"] == 3]["Gene"].tolist())),
            "WP4534 YAP Regulators": list(set_wp4534),
            "Cordenonsi YAP Regulators": list(set_cordenonsi),
            "YAP1 UP Regulators": list(set_yap_up),
        }

        return venn_dict


def load_custom_palettes():
    custom_palettes = {
        "Response to treatment": {
            "Residual disease (RD)": "#377eb8",     # medium blue
            "Progressive disease (PD)": "#e41a1c",  # bright red – stands out
            "Unassigned": "#ff7f00",                # orange
        },
        "Driver gene": {
            "EGFR": "#984ea3",       # strong violet – distinctive
            "KRAS": "#4daf4a",       # green
            "BRAF": "#f781bf",       # pink
            "ALK": "#a65628",        # brown
        },
        "InferCNV annotation": {
            "Cancer cell": "#00ced1",      # vivid cyan – highly visible
            "Noncancer cell": "#ffd700",   # golden yellow
        },
        "Therapy": {
            "Alectinib": "#1f78b4",               # blue
            "Ceritinib/trametinib": "#33a02c",    # green
            "Crizotinib": "#6a3d9a",              # purple
            "Dabrafenib/trametinib": "#a6cee3",   # light blue
            "Erlotinib": "#ff7f0e",               # bright orange – standout
            "Erlotinib/gemcitabine": "#b15928",   # dark brown
            "Osimertinib": "#fb9a99",             # light red-pink
        }
    }
    return custom_palettes


def plot_pseudobulk_heatmap(TvsNAT, C2_3):

    cordenonsi_genes, wp4534_genes, yap_up_genes = get_YAPsignature_genes()
    C2 = list(dict.fromkeys(C2_3[C2_3["Cluster"] == 2]["Gene"].tolist()))
    C3 = list(dict.fromkeys(C2_3[C2_3["Cluster"] == 3]["Gene"].tolist()))
    gs_dict = {
        "Cordenonsi YAP": sanitize_genes(TvsNAT, cordenonsi_genes),
        "WP4534":         sanitize_genes(TvsNAT, wp4534_genes),
        "YAP1_UP":        sanitize_genes(TvsNAT, yap_up_genes),
        "AXL C2":  sanitize_genes(TvsNAT, C2),
        "AXL C3":  sanitize_genes(TvsNAT, C3)
    }

    plot_scRNAseq_gene_set_heatmap(
        TvsNAT,
        gs_dict,
        features=("Therapy", "Treatment timepoint with TKI", "Treatment timepoint", "Driver gene", "InferCNV annotation", "Biopsy site", "Sampling site"),
        layer=None, use_raw=False,
        id_col="Sample name",
        agg="gmean",
        rows="signature_means",
        figsize=(12, 7),
        cmap="vlag",
        savefig=True,
        sort_columns_by=("Treatment timepoint with TKI","Therapy"),
        sort_orders={
            "Treatment timepoint with TKI": ["Progressive disease (PD)", "Residual disease (RD)", "Unassigned"],
            "Therapy": ["Erlotinib", "Osimertinib"]
        }
    )


def plot_scRNAseq_gene_set_heatmap(
    adata,
    gene_sets_dict,
    features=("Response to treatment", "Treatment time point", "Treatment status"),
    layer=None,
    use_raw=False,
    id_col="Sample name",
    agg="gmean",                 # 'gmean' or 'mean' (how to pseudobulk cells -> patients)
    rows="genes",                # 'genes' or 'signature_means'
    figsize=(14, 8),
    cmap="coolwarm",
    save_fig=None,
    save_fig_legends=None,
    custom_palettes=None,
    savefig=False,
    sort_columns_by=("Response to treatment", "Therapy"),
    sort_orders=None,  # e.g., {"Response to treatment": ["Progressive disease (PD)","Residual disease (RD)","Unassigned"],
                       #        "Therapy": ["Erlotinib","Osimertinib"]}
    ):
    """
    Heatmap of scRNA-seq gene-set signals vs. patients (pseudobulk).
    Columns are sorted by `sort_columns_by` (with optional explicit `sort_orders`).

    rows='genes'            -> rows are individual genes (within the provided sets).
    rows='signature_means'  -> rows are signatures (mean across that signature's genes).
    """

    # --- 1) Collect all unique genes from the provided sets ---
    all_genes_uc = set()
    for _, genes in gene_sets_dict.items():
        for g in genes:
            if isinstance(g, str):
                all_genes_uc.add(g.strip().upper())
    if len(all_genes_uc) == 0:
        print("No genes provided in gene_sets_dict.")
        return

    # --- 2) Match to adata.var (case-insensitive) ---
    var_idx_uc = pd.Index(adata.var_names.astype(str))
    map_upper = pd.Series(var_idx_uc, index=var_idx_uc.str.upper())
    matched_syms = map_upper.loc[list(all_genes_uc)].dropna().unique().tolist()
    if len(matched_syms) == 0:
        print("No matching genes found between gene sets and AnnData object.")
        return

    # --- 3) Pull expression table (cells x genes) ---
    plot_df = sc.get.obs_df(adata, keys=matched_syms, layer=layer, use_raw=use_raw)

    # --- 4) Filter very-low-expression genes (data-driven) ---
    gene_sums = plot_df.sum(axis=0)
    thr = gene_sums.quantile(0.15)
    keep = gene_sums.index[gene_sums >= thr]
    plot_df = plot_df.loc[:, keep]
    if plot_df.shape[1] == 0:
        print("No genes passed the expression filter.")
        return

    # --- 5) Pseudobulk per patient ---
    if id_col not in adata.obs.columns:
        raise KeyError(f"'{id_col}' not found in adata.obs.")
    plot_df[id_col] = adata.obs[id_col].values

    if agg == "gmean":
        pb = plot_df.groupby(id_col, observed=True).agg(
            lambda x: gmean(np.asarray(x, dtype=float) + 1e-8)
            if np.issubdtype(x.dtype, np.number) else x.iloc[0]
        )
    elif agg == "mean":
        pb = plot_df.groupby(id_col, observed=True).mean(numeric_only=True)
    else:
        raise ValueError("agg must be 'gmean' or 'mean'")

    # keep only gene columns (groupby may keep id_col if non-numeric)
    pb = pb.loc[:, [c for c in pb.columns if c != id_col]]  # patients x genes

    # --- 6) Build per-patient feature color bars (mode per patient) ---
    feature_luts, col_colors_dict, feat_values = {}, {}, {}
    patient_index = pb.index
    for feat in features:
        if feat in adata.obs.columns:
            feat_series = (
                adata.obs[[id_col, feat]]
                .astype({feat: str})
                .groupby(id_col, observed=True)[feat]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
                .reindex(patient_index)
            )
        else:
            feat_series = pd.Series(["NA"] * len(patient_index), index=patient_index)

        feat_values[feat] = feat_series

        if custom_palettes and feat in custom_palettes:
            lut = custom_palettes[feat]
        else:
            uniq = pd.Index(feat_series.dropna().unique())
            palette = sns.color_palette("tab20", max(len(uniq), 1))
            lut = dict(zip(uniq.tolist(), palette))
        feature_luts[feat] = lut
        col_colors_dict[feat] = feat_series.map(lut)

    col_colors = pd.DataFrame(col_colors_dict, index=patient_index)

    # --- 6b) Determine desired column order based on sort_columns_by ---
    if sort_columns_by:
        sort_df = pd.DataFrame(index=patient_index)
        for feat in sort_columns_by:
            if feat in feat_values:
                s = feat_values[feat]
            else:
                # if not computed above, try to derive now
                if feat in adata.obs.columns:
                    s = (
                        adata.obs[[id_col, feat]].astype({feat: str})
                        .groupby(id_col, observed=True)[feat]
                        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
                        .reindex(patient_index)
                    )
                else:
                    s = pd.Series([""] * len(patient_index), index=patient_index)
            # apply explicit ordering if provided
            if sort_orders and feat in sort_orders:
                cat = pd.Categorical(s, categories=list(sort_orders[feat]), ordered=True)
                sort_df[feat] = cat
            else:
                sort_df[feat] = s.astype(str)
        order_cols = list(sort_df.columns)
        col_order = sort_df.sort_values(by=order_cols, kind="mergesort").index  # stable
    else:
        col_order = patient_index

    # --- 7) Prepare matrix to plot depending on row mode ---
    if rows == "genes":
        # z-score genes across patients (columns = genes)
        pb_z = pd.DataFrame(
            zscore(pb, axis=0, nan_policy="omit"),
            index=pb.index, columns=pb.columns
        )

        # order genes by gene-set appearance
        ordered_genes = []
        for set_name, genes in gene_sets_dict.items():
            for g in genes:
                gu = g.strip().upper()
                if gu in map_upper.index:
                    sym = map_upper[gu]
                    if sym in pb_z.columns:
                        ordered_genes.append(sym)
        # unique while preserving order
        seen = set()
        ordered_genes = [g for g in ordered_genes if not (g in seen or seen.add(g))]
        # intersect with available
        ordered_genes = [g for g in ordered_genes if g in pb_z.columns]
        if len(ordered_genes) == 0:
            ordered_genes = pb_z.columns.tolist()

        pb_plot = pb_z[ordered_genes]           # patients x genes

        # row colors (gene-set membership)
        unique_sets = []
        gene_to_set = {}
        for set_name, genes in gene_sets_dict.items():
            for g in genes:
                gu = g.strip().upper()
                sym = map_upper.get(gu, None)
                if sym in ordered_genes and sym not in gene_to_set:
                    gene_to_set[sym] = set_name
                    if set_name not in unique_sets:
                        unique_sets.append(set_name)

        set_palette = sns.color_palette("Set1", len(unique_sets)) if unique_sets else sns.color_palette("Set1", 1)
        set2color = dict(zip(unique_sets, set_palette))
        row_colors_df = pd.DataFrame(
            {"Gene Set": [set2color.get(gene_to_set.get(g, unique_sets[0] if unique_sets else ""), (0.5,0.5,0.5)) for g in ordered_genes]},
            index=ordered_genes
        )

        plot_df_T = pb_plot.T.loc[:, col_order]       # rows=genes, cols=patients (sorted)
        col_colors_sorted = col_colors.loc[col_order]

        g = sns.clustermap(
            plot_df_T,
            row_cluster=False,
            col_cluster=False,  # keep the requested sort order
            row_colors=row_colors_df,
            col_colors=col_colors_sorted,
            cmap=cmap,
            center=0,
            figsize=figsize,
            robust=True,
            xticklabels=True,
            yticklabels=False,
            vmin=-2,
            vmax=+2,
        )

        # legend for gene sets
        if unique_sets:
            handles = [Patch(facecolor=set2color[s], label=s) for s in unique_sets]
            g.ax_row_dendrogram.legend(
                handles=handles, title="Gene Set",
                bbox_to_anchor=(1.05, 1), loc="lower right",
                fontsize=8, title_fontsize=8, frameon=False
            )
        g.cax.set_position([.97, .2, .02, .3])

    elif rows == "signature_means":
        # map each gene -> set, and build per-signature mean per patient
        sig_to_genes = {
            s: [map_upper[g.strip().upper()] for g in genes if map_upper.get(g.strip().upper(), None) in pb.columns]
            for s, genes in gene_sets_dict.items()
        }
        sig_to_genes = {s: gl for s, gl in sig_to_genes.items() if len(gl) > 0}
        if len(sig_to_genes) == 0:
            print("No signature has genes present after filtering.")
            return

        # patients x signatures: mean across each signature's genes
        sig_mat = pd.DataFrame(index=pb.index)
        for s, gl in sig_to_genes.items():
            sig_mat[s] = pb[gl].mean(axis=1)

        # z-score each signature across patients
        sig_z = pd.DataFrame(
            zscore(sig_mat, axis=0, nan_policy="omit"),
            index=sig_mat.index, columns=sig_mat.columns
        )

        plot_df_T = sig_z.T.loc[:, col_order]      # rows=signatures, cols=patients (sorted)

        # row colors for signatures
        sigs = list(sig_to_genes.keys())
        sig_palette = sns.color_palette("Set1", len(sigs))
        sig2color = dict(zip(sigs, sig_palette))
        row_colors_df = pd.DataFrame({"Signature": [sig2color[s] for s in sigs]}, index=sigs)
        col_colors_sorted = col_colors.loc[col_order]

        g = sns.clustermap(
            plot_df_T,
            row_cluster=True,
            col_cluster=False,  # keep the requested sort order
            row_colors=row_colors_df,
            col_colors=col_colors_sorted,
            cmap=cmap,
            center=0,
            figsize=figsize,
            robust=True,
            xticklabels=True,
            yticklabels=True,
            linewidths=0.5,
            linecolor='black',
            vmin=-2,
            vmax=+2,
        )

        # legend for signatures
        handles = [Patch(facecolor=sig2color[s], label=s) for s in sigs]
        g.ax_row_dendrogram.legend(
            handles=handles, title="Signature",
            bbox_to_anchor=(1.05, 1), loc="lower right",
            fontsize=8, title_fontsize=8, frameon=False
        )
        g.cax.set_position([.97, .2, .02, .3])

    else:
        raise ValueError("rows must be 'genes' or 'signature_means'")

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight", dpi=600)

    # --- Legends for clinical features (per-patient) ---
    if features:
        all_handles_titles = []
        for feat in features:
            lut = feature_luts.get(feat, {})
            if not lut:
                continue
            handles = [Patch(facecolor=lut[k], label=str(k)) for k in lut.keys()]
            all_handles_titles.append((handles, feat))

        if all_handles_titles:
            _, axes = plt.subplots(1, len(all_handles_titles), figsize=(3.2 * len(all_handles_titles), 2))
            if len(all_handles_titles) == 1:
                axes = [axes]
            for ax, (handles, feat) in zip(axes, all_handles_titles):
                ax.legend(handles=handles, title=feat, fontsize=9, title_fontsize=10, frameon=True)
                ax.axis("off")
            if save_fig_legends:
                plt.savefig(save_fig_legends, bbox_inches="tight")

def sanitize_genes(adata, genes):
    vs = pd.Index(adata.var_names.astype(str))
    up = pd.Index([g.strip().upper() for g in genes if isinstance(g, str)])
    map_upper = pd.Series(vs, index=vs.str.upper())
    return map_upper.loc[up.intersection(map_upper.index)].dropna().unique().tolist()
