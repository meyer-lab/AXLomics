import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import gseapy as gp
import scipy as sp
from venny4py.venny4py import venny4py
from matplotlib.patches import Patch
from scipy.stats import zscore, gmean
from .common import subplotLabel, getSetup

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((5, 5), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    adata = preprocess_maynard("/scratch4/creixell/Maynard/8091e3d90a9045a181b2fc11000c0dd9_PMID32822576.h5ad")
    adata, TvsNAT, cc = annotate_maynard(adata)

    adata_egfr = adata[adata.obs["Driver gene"] == "EGFR"]
    cc_egfr = cc[cc.obs["Driver gene"] == "EGFR"]
    TvsNAT_egfr = TvsNAT[TvsNAT.obs["Driver gene"] == "EGFR"]

    # Filter out group, practically no cells
    cc_egfr = cc_egfr[
        (cc_egfr.obs["sampling site (standardized)"] != "Anatomic Site of Origin for Matched Normal Tissue")
    ]

    sc.pl.violin(cc_egfr, ["AXL score"], rotation=45, groupby='Response to treatment')
    sc.pl.violin(cc_egfr, ["AXL score"], rotation=90, groupby='sampling site (standardized)')

    # Plot t-SNE for EGFRm patients
    plot_tsne(adata_egfr)

    # Plot normalized AXL expression by treatment response and sampling site
    sc.tl.score_genes(cc_egfr, ["AXL"], score_name="AXL score")
    sc.pl.violin(cc_egfr, ["AXL score"], rotation=45, groupby='Response to treatment')
    sc.pl.violin(cc_egfr, ["AXL score"], rotation=90, groupby='sampling site (standardized)', order=["Site of Metastasis", "Primary Tumor Site"])

    # Plot AXL downstream signature by clinical features
    sc.pl.violin(cc, ["AXL signature"], rotation=45, groupby="Driver gene", order=["EGFR", "BRAF", "ALK"]) # Very few ells in KRAS and ROS1
    sc.pl.violin(cc_egfr, ["AXL signature"], rotation=45, groupby="Driver mutation", order=["EGFR L858R", "Del19"]) 
    sc.pl.violin(cc_egfr, ["AXL signature"], rotation=45, groupby="Response to treatment") 
    sc.pl.violin(cc_egfr, ["AXL signature"], rotation=90, groupby='sampling site (standardized)', order=["Site of Metastasis", "Primary Tumor Site"])

    return f


def plot_tsne(adata_egfr):
    """Plot t-SNE for EGFRm patients."""
    sc.tl.pca(adata_egfr, svd_solver='arpack')
    sc.tl.tsne(adata_egfr, n_pcs=20, perplexity=30)
    sc.pl.tsne(adata_egfr, color=["Author's cell type"], ncols=1)


def preprocess_maynard():
    """Load the Maynard dataset, calculate QC matrics, and filter cells and genes."""
    adata = sc.read_h5ad("/scratch4/creixell/Maynard/8091e3d90a9045a181b2fc11000c0dd9_PMID32822576.h5ad")
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata = adata[adata.obs["total_counts"] != 0]

    # Calculate percentage of mitochondrial genes
    mt_gene_mask = [gene.startswith('MT') for gene in adata.var.index]
    mt_counts = adata.X[:, np.array(mt_gene_mask)].sum(1)
    perc_mito_ot = mt_counts.flatten() / adata.obs['total_counts'].values
    adata.obs['perc_mito'] = perc_mito_ot.transpose()

    # Filter cells and genes based on quality metrics
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_cells(adata, max_counts=20000)
    adata = adata[adata.obs.perc_mito < .3, :]

    # Save raw counts
    adata.layers["counts"] = adata.X.copy()

    # Normalize counts and get log-counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Subset to highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata

    return adata


def annotate_maynard(adata):
    """
    Annotate the Maynard dataset with cell type and inferCNV annotations.
    """
    # Load cell type annotations
    cca = pd.read_csv("/scratch4/creixell/Maynard/cancer_cell_annotation.csv")
    IDtoCNV = dict(zip(list(cca["cell_id"]), list(list(cca["inferCNV_annotation"]))))
    adata.obs.insert(3, "inferCNV_annotation", [IDtoCNV[c_id] if c_id in list(IDtoCNV.keys()) else "nonepithelial" for c_id in list(adata.obs.index)])

    # Tumor vs NAT
    TvsNAT = adata[(adata.obs["Author's cell type"] == "Epithelial") & (adata.obs["inferCNV_annotation"] != "NA")]
    cc = adata[(adata.obs["Author's cell type"] == "Epithelial") & (adata.obs["inferCNV_annotation"] == "cancer cell")]
    adata.obs["Driver mutation"] = adata.obs["Driver mutation"].replace("del19", "Exon 19del")

    # Load AXL signature
    c123 = pd.read_csv("/home/creixell/AXLomics/msresist/out/results/C123.csv").dropna().iloc[:, -1].to_list()
    sc.tl.score_genes(adata, c123, score_name="AXL signature")
    sc.tl.score_genes(TvsNAT, c123, score_name="AXL signature")
    sc.tl.score_genes(cc, c123, score_name="AXL signature")

    return adata, TvsNAT, cc


def plot_count_depth(adata, min_count):
    fig, (ax1, _) = plt.subplots(1, 2, sharey=True)

    #produce elbow plot of count depths
    count_data = adata.obs['total_counts'].copy()
    count_data.sort_values(inplace=True, ascending=False)
    order =  range(1, len(count_data)+1)
    ax1.semilogy(order, count_data, 'b-')
    #draw min_count threshold
    ax1.axhline(min_count, 0,1, color='red')
    ax1.set_xlabel("Barcode rank")
    ax1.set_ylabel("Count depth")
    ax1.set_title('Distribution of Count Depth')

    #plot distibution of count depths
    max_counts = max(adata.obs['total_counts'])
    log_bins=np.logspace(0, np.log10(max_counts), 100)
    count_hist = sns.distplot(adata.obs['total_counts'], kde=False, bins=log_bins, vertical=True)
    count_hist.set_yscale('log')
    count_hist.set_xlabel("Frequency")
    count_hist.set_ylabel('')
    #draw min_count threshold
    count_hist.axhline(min_count, 0,1, color='red')
    count_hist.get_figure()

    fig.tight_layout(pad=0)
    plt.show()


def plot_genes_per_cell(adata, min_genes):
    #plot distribution of genes per cell
    gene_hist = sns.distplot(adata.obs['n_genes_by_counts'], kde=False)
    gene_hist.set_xlabel("Number of genes")
    gene_hist.set_ylabel("Frequency")
    gene_hist.set_title('Genes per cell')
    #draw min_genes threshold
    gene_hist.axvline(min_genes, 0,1, color='red')
    gene_hist.get_figure()
    plt.show()
