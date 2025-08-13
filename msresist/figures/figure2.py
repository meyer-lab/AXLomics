"""
This creates Figure 2: Model figure
"""

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
import gseapy as gp
from .common import subplotLabel, getSetup, import_phenotype_data, formatPhenotypesForModeling, plot_AllSites
from ..pre_processing import preprocessing
from ..clustering import DDMC
from ..plsr import plotStripActualVsPred, plotScoresLoadings

lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((14, 12), (3, 3), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Import siganling data
    X = preprocessing(AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    d = X.select_dtypes(include=['float64']).T
    i = X.select_dtypes(include=['object'])

    # Fit DDMC
    ddmc = DDMC(i, n_components=5, SeqWeight=2, distance_method="PAM250", random_state=5).fit(d)
    centers = ddmc.transform()

    # Import phenotypes
    cv = import_phenotype_data(phenotype="Cell Viability")
    red = import_phenotype_data(phenotype="Cell Death")
    sw = import_phenotype_data(phenotype="Migration")
    c = import_phenotype_data(phenotype="Island")
    y = formatPhenotypesForModeling(cv, red, sw, c)
    y = y[y["Treatment"] == "A/E"].drop("Treatment", axis=1).set_index("Lines")

    # Mass spec clustermap
    d.index = lines
    # sns.clustermap(data=d, cmap="bwr", row_cluster=True, col_cluster=True, square=True, robust=True, figsize=(12, 6))
    ax[0].axis("off") # Clustermap doesn't have an ax argument

    # Pipeline diagram
    ax[1].axis("off")

    # Plot DDMC centers
    X.columns = X.columns = list(X.iloc[:, :7].columns) + ["PAR", "KO", "KD", "WT", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    plotCenters_together(ddmc, X, ax[2], drop=2)

    # AXL phosphosites
    # plot_AllSites("", X, "AXL", "AXL", ylim=False, type="Heatmap")
    ax[3].axis("off") # Clustermap doesn't have an ax argument

    # Clusters GSEA
    ax[4].axis("off") # Ran with ClusterProfiler in R

    # Clustermap with DDMC cluster members
    # create_protein_clustermap(X.loc[top_prot].reset_index().set_index(["Gene", "Position"]), protein_classes)
    ax[5].axis("off") # Clustermap doesn't have an ax argument

    # Predictions
    Xs, models = ComputeCenters(X, d, i, ddmc)
    Xs.append(centers)
    models.append("DDMC mix")
    plotStripActualVsPred(ax[6], [3, 4, 2, 3, 4], Xs, y, models)

    # PLSR Scores & Loadings
    plsr = PLSRegression(n_components=4)
    plotScoresLoadings(ax[7:9], plsr.fit(centers, y), centers, y, ddmc.n_components, lines, pcX=1, pcY=2)

    return f


def plotCenters_together(ddmc, X, ax, drop=None):
    """Plot Cluster Centers together in same plot"""
    centers = pd.DataFrame(ddmc.transform()).T
    centers.columns = X.columns[7:]
    centers["Cluster"] = list(np.arange(ddmc.n_components) + 1)
    if drop:
        centers = centers.set_index("Cluster").drop(drop, axis=0).reset_index()
    m = pd.melt(centers, id_vars=["Cluster"], value_vars=list(centers.columns), value_name="p-signal", var_name="Lines")
    m["p-signal"] = m["p-signal"].astype("float64")
    sns.set_context("paper", rc={'lines.linewidth': 1}) 
    palette ={1: "C0", 2: "C1", 3: "C2", 4: "C3", 5: "k"}
    sns.lineplot(x="Lines", y="p-signal", data=m, hue="Cluster", ax=ax, palette=palette, **{"linewidth": 2}, marker="o", markersize=10)
    ax.tick_params(axis='x', rotation=45)


def ComputeCenters(X, d, i, ddmc):
    """Calculate cluster centers of  different algorithms."""
    # k-means
    labels = KMeans(n_clusters=ddmc.n_components).fit(d.T).labels_
    x_ = d.copy().T
    x_.insert(0, "Cluster", labels)
    c_kmeans = x_.groupby("Cluster").mean().T

    # GMM
    ddmc_data = DDMC(i, n_components=ddmc.n_components, SeqWeight=0, distance_method=ddmc.distance_method, random_state=ddmc.random_state).fit(d)
    c_gmm = ddmc_data.transform()

    # DDMC seq
    ddmc_seq = DDMC(i, n_components=ddmc.n_components, SeqWeight=ddmc.SeqWeight + 150, distance_method=ddmc.distance_method, random_state=ddmc.random_state).fit(d)
    ddmc_seq_c = ddmc_seq.transform()

    return [d, c_kmeans, c_gmm, ddmc_seq_c], ["Unclustered", "k-means", "GMM", "DDMC seq"]


def create_protein_clustermap(data, protein_classes, row_colors=True):
    """
    Create a clustermap for selected proteins with cluster colors and protein class colors.
    
    Parameters:
    ----------
    data : pandas.DataFrame
        DataFrame containing protein data, typically X.loc[top35_prot]
    row_colors : bool, default=True
        Whether to show cluster assignments as colors
    
    Returns:
    --------
    g : seaborn.matrix.ClusterGrid
        The clustermap object
    """
    # Sort data by cluster before creating the row_colors_df
    # Sort data first by cluster and then by protein name
    data = data.sort_values(by=['Cluster', 'Protein', 'PAR'], ascending=[True, True, False])

    # Extract only the numeric columns for plotting
    plot_data = data.iloc[:, 6:].copy()
    
    # Prepare row colors
    if row_colors:
        # Create a color palette for clusters
        n_clusters = data['Cluster'].nunique()
        cluster_colors = sns.color_palette("Set1", n_clusters)
        lut_cluster = dict(zip(range(1, n_clusters + 1), cluster_colors))
        
        # Create a color palette for protein classes
        class_categories = list(protein_classes.keys())
        class_palette = sns.color_palette("Set2", len(class_categories))
        lut_class = dict(zip(class_categories, class_palette))
        
        # Create a row color DataFrame
        row_colors_df = pd.DataFrame(index=data.index)
        
        # Add cluster colors
        row_colors_df['Cluster'] = data['Cluster'].map(lambda x: lut_cluster[x])
        
        # Add protein class colors
        # First map genes to their protein class
        gene_to_class = {}
        for class_name, genes in protein_classes.items():
            for gene in genes:
                gene_to_class[gene] = class_name

        # Use a color palette with more distinct colors for protein classes
        class_categories = list(protein_classes.keys())
        class_palette = sns.color_palette("tab20", len(class_categories))
        lut_class = dict(zip(class_categories, class_palette))

        # Extract gene names from the index
        if isinstance(data.index, pd.MultiIndex):
            genes = data.index.get_level_values('Gene')
        else:
            genes = data.index
            
        # Map genes to their class
        row_colors_df['Class'] = [lut_class.get(gene_to_class.get(gene, "Unknown"), "#CCCCCC") for gene in genes]
    else:
        row_colors_df = None

    # Create the clustermap
    g = sns.clustermap(
        plot_data,
        row_cluster=False,  # Don't cluster rows
        col_cluster=True,   # Cluster columns
        cmap="bwr",        # Red-blue divergent colormap
        row_colors=row_colors_df if row_colors else None,  # Use DataFrame for row colors
        figsize=(7, 19),   # Slightly wider to accommodate legend
        linewidths=0.1,
        method="median",  # Options: 'average', 'complete', 'single', 'centroid', 'median', 'ward', 'weighted'
        xticklabels=True,
        yticklabels=True,  # Show row labels to identify proteins
        robust=True,
        cbar_kws={"shrink": 0.1}  # Make the colorbar smaller
    )
    
    # Create a legend for clusters and protein classes
    if row_colors:
        # Create legend for clusters
        handles_cluster = [plt.Rectangle((0,0),1,1, color=lut_cluster[i]) for i in range(1, n_clusters + 1)]
        
        # Create legend for protein classes
        handles_class = [plt.Rectangle((0,0),1,1, color=lut_class[c]) for c in class_categories]
        
        # Add both legends to the plot with better positioning to avoid overlap
        cluster_leg = Legend(g.ax_heatmap, handles_cluster, [f'Cluster {i}' for i in range(1, n_clusters + 1)],
                            title='Clusters', bbox_to_anchor=(1.05, 1), loc=0)
        class_leg = Legend(g.ax_heatmap, handles_class, class_categories,
                          title='Protein Classes', bbox_to_anchor=(1.05, 0.6), loc=0)
        
        g.ax_heatmap.add_artist(cluster_leg)
        g.ax_heatmap.add_artist(class_leg)
    
    plt.tight_layout()
    return g


# Extracted from top nodes in /home/creixell/AXLomics/msresist/out/results/CytoNodes_Clusters_WTvsKO_pY.csv
top_prot = [
    "TCP1", "SRRM2", "GAB1", "GAB2", "CDK2", "TNK2", "SPTBN1",
    "DAPP1", "TOM1", "EPHA5", "BAG3", "EPS8", "ABI2", "WASL", "ALDH1A3", "CBLC", "EPB41L2", "GPRC5B",
    "SOS1", "CBLB", "PAG1", "PTPN11", "SIRPA", "ANXA8", "GRB2", "CDK1", "ZNF185", "AFDN", "PRKCD", "MAPK1", "MAPK3", "INPPL1", "SDCBP", "ARHGAP27", "RPS10",
    "MAPK10", "DYRK3", "GRB2", "DOK1", "MPZL1", "SH2D3A", "EIF2S2", "PKP3", "TNS1", "RASA1", "PITPNA", "TJP2", "ARAP1", "ACTN1", 
]

protein_classes = {
    "Kinases": [
        "CDK1", "CDK2", "MAPK1", "MAPK3", "MAPK10", "DYRK3", "TNK2", "PRKCD"
    ],
    "Phosphatases": [
        "PTPN11", "INPPL1"
    ],
    "Adaptor/Scaffold Proteins": [
        "GRB2", "GAB1", "GAB2", "CBLC", "CBLB", "PAG1", "DAPP1", "SDCBP", "AFDN"
    ],
    "Receptors (including RTKs)": [
        "EPHA2", "EPHA5", "EPHB3", "EPHB4", "GPRC5B"
    ],
    "Cytoskeletal/Structural Proteins": [
        "WASL", "SPTBN1", "EPB41L2", "ZNF185", "ANXA8", "ARHGAP27"
    ],
    "Signaling Regulators / GTPase-related": [
        "SOS1"
    ],
    "Chaperones / Co-chaperones": [
        "TCP1", "BAG3"
    ],
    "RNA Splicing / Processing": [
        "SRRM2"
    ],
    "Oxidoreductases / Enzymes": [
        "ALDH1A3"
    ],
    "Other / Unknown / Multifunctional": [
        "TOM1", "SIRPA", "RPS10", "ABI2"
    ]
}


