"""
This creates Figure 2: Model figure
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
import gseapy as gp
from .common import subplotLabel, getSetup, import_phenotype_data, formatPhenotypesForModeling, plotDistanceToUpstreamKinase
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

    # Pipeline diagram
    ax[0].axis("off")

    # Mass spec clustermap
    ax[1].axis("off") # Clustermap doesn't have an ax argument

    # Predictions
    Xs, models = ComputeCenters(X, d, i, ddmc)
    Xs.append(centers)
    models.append("DDMC mix")
    plotStripActualVsPred(ax[2], [3, 4, 2, 3, 4], Xs, y, models)

    # AXL p-sites clustermap
    # plot_AllSites("", X, "AXL", "AXL", ylim=False, type="Heatmap", figsize=(3.5, 5)).T
    ax[3].axis("off")

    # Centers
    plotCenters_together(ddmc, X, ax[4], drop=5)

    # Plot upstream kinases heatmap
    plotDistanceToUpstreamKinase(ddmc, [1, 2, 3, 4, 5], ax[7], num_hits=10)

    # Scores & Loadings
    plsr = PLSRegression(n_components=4)
    plotScoresLoadings(ax[5:7], plsr.fit(centers, y), centers, y, ddmc.n_components, lines, pcX=1, pcY=2)

    # EGFR TKI res signature
    # gseaplot_EGFRres_signature(X, ddmc) gseaplot doesn't have an ax argument

    # WT vs KO Heatmap of specific p-sites
    WTvsKO_heatmap_psites(X, ddmc)

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


def ComputeCenters(X, d, i, ddmc):
    """Calculate cluster centers of  different algorithms."""
    # k-means
    labels = KMeans(n_clusters=ddmc.n_components).fit(d.T).labels_
    x_ = X.copy()
    x_["Cluster"] = labels
    c_kmeans = x_.groupby("Cluster").mean().T

    # GMM
    ddmc_data = DDMC(i, n_components=ddmc.n_components, SeqWeight=0, distance_method=ddmc.distance_method, random_state=ddmc.random_state).fit(d)
    c_gmm = ddmc_data.transform()

    # DDMC seq
    ddmc_seq = DDMC(i, n_components=ddmc.n_components, SeqWeight=ddmc.SeqWeight + 150, distance_method=ddmc.distance_method, random_state=ddmc.random_state).fit(d)
    ddmc_seq_c = ddmc_seq.transform()

    return [d, c_kmeans, c_gmm, ddmc_seq_c], ["Unclustered", "k-means", "GMM", "DDMC seq"]

def gseaplot_EGFRres_signature(MS, ddmc):
    """ Calculate EGFR TKI resistance enrichment signature using WT vs KO fold change of MS data """
    MS.insert(0, "Cluster", ddmc.labels())
    cl3 = MS[(MS["Cluster"] == 3)]

    cl3_fc = pd.DataFrame()
    cl3_fc["Gene"] = cl3["Gene"]
    cl3_fc["log2(FC)"] = cl3["PC9 A"] - cl3["KO A"]
    rnk = cl3_fc.sort_values(by="log2(FC)", ascending=False).set_index("Gene")

    pre_res = gp.prerank(rnk=rnk,
                     gene_sets='WikiPathway_2021_Human',
                     threads=4,
                     min_size=5,
                     max_size=1000,
                     permutation_num=1000, # reduce number to speed up testing
                     outdir="/home/marcc/AXLomics/msresist/data/RNAseq/GSEA/Kurpa_gsea", # don't write to disk
                     seed=6,
                     verbose=True, # see what's going on behind the scenes
                    )
    term = "EGFR Tyrosine Kinase Inhibitor Resistance WP4806"
    gp.gseaplot(rank_metric=pre_res.ranking,
         term=term,
         **pre_res.results[term])

def WTvsKO_heatmap_psites(MS, ddmc, figsize=(4, 10)):
    """Make heatmap of specific p-sites of AXL WT EA vs KO labeling cluster membership"""

    kin = [
    "EGFR Y1172-p",
    "EPHA2 Y594-p",
    "ERBB2 Y877-p",
    "ERBB3 Y1328-p",
    "MET S988-p",
    "GAB1 Y659-p",
    "GAB1 Y406-p",
    "GAB2 Y293-p",
    "GRB2 Y160-p",
    "SOS1 Y1196-p",
    "DAPP1 Y139-p",
    "CBLB Y363-p",
    "CBLB Y802-p;T798-p",
    "CBLB Y889-p;T885-p",
    "FLNA Y2379-p",
    "FLNB Y904-p",
    "AHNAK S115-p",
    "FRK Y132-p",
    "LYN Y397-p",
    "LCK Y192-p",
    "LCK Y394-p",
    "YES1 Y222-p",
    "YES1 S195-p",
    "ABL1 Y185-p",
    "ABI2 Y213-p",
    "MAPK1 Y187-p",
    "MAPK3 Y204-p",
    "MAPK9 Y185-p;T183-p",
    "MAPK10 Y223-p",
    "MAPK10 Y223-p;T221-p",
    "MAPK13 Y182-p",
    "MAPK14 Y182-p",
    "ICK S152-p",
    "TNK2 Y859-p",
    "PRKCD Y313-p",
    "CDK1 Y15-p",
    "CDK2 Y15-p",
    ]
    MS.insert(0, "Phosphosite", [g + " " + p for g, p in zip(list(MS["Gene"]), list(MS["Position"]))])
    MS.insert(0, "Cluster", ddmc.labels())
    s_ea = MS.set_index("Phosphosite").loc[kin][["PC9 A", "KO A", "Cluster"]]
    s_ea.columns = ["WT", "KO", "Cluster"]
    clusters = s_ea.pop("Cluster")
    lut = dict(zip(clusters.unique(), ["darkgreen", "violet", "black", "yellow"]))
    row_colors = clusters.map(lut)
    sns.clustermap(data=s_ea, cmap="coolwarm", row_cluster=False, col_cluster=False, row_colors=row_colors, linewidth=1, linecolor='w', square=True, robust=True, figsize=figsize)


# AXL p-sites in WT UT vs WT E
# MS = preprocessing(AXLwt_GF=True, Vfilter=True, FCtoUT=False, mc_col=True, mc_row=True)
# axl_e = MS[MS["Gene"] == "AXL"][["Position", "PC9", "Erl"]]
# axl_e.columns = ["AXL p-site", "PC9 UT", "PC9 +E"]
# axl_e = pd.melt(frame=axl_e, value_vars=["PC9 UT", "PC9 +E"], id_vars="AXL p-site", value_name="norm log(p-signal)", var_name="Condition")

# _, ax = plt.subplots(1, 1, figsize=(5, 4))
# sns.barplot(axl_e, x="AXL p-site", y="norm log(p-signal)", hue="Condition", ax=ax)

# from scipy.stats import ttest_rel

# a = axl_e.iloc[:5, -1].values
# b = axl_e.iloc[5:, -1].values
# _, pval = ttest_rel(a, b, axis=0)

# from scipy.stats import ttest_rel

# a = axl_e.iloc[:5, -1].values
# b = axl_e.iloc[5:, -1].values
# ttest_rel(a, b, axis=0)

# kin_gf = [
#     "EGFR Y1197-p", 
#     "EPHA2 Y594-p",
#     "ERBB2 Y877-p",
#     "MET S988-p",
#     "GAB1 Y659-p",
#     "GAB1 Y406-p",
#     "GAB2 Y266-p",
#     "FLNB Y904-p",
#     "AHNAK Y836-p",
#     "AHNAK Y715-p",
#     "FRK Y132-p",
#     "FRK Y497-p",
#     "LYN Y193-p",
#     "LCK Y192-p",
#     "LCK Y394-p",
#     "YES1 Y194-p",
#     "ABL1 Y185-p",
#     "ABI2 Y213-p",
#     "MAPK1 Y187-p",
#     "MAPK3 Y204-p",
#     "MAPK9 Y185-p;T183-p",
#     "MAPK10 Y223-p;T221-p",
#     "MAPK13 T180-p",
#     "MAPK14 Y182-p",
#     "ICK Y159-p",
#     "PRKCD Y374-p",
#     "CDK1 Y15-p;T14-p",
# ]

# MSgf = preprocessing(AXLwt_GF=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
# dgf = MSgf.select_dtypes(include=['float64']).T
# MSgf.insert(0, "Phosphosite", [g + " " + p for g, p in zip(list(MSgf["Gene"]), list(MSgf["Position"]))])
# s_gf = MSgf.set_index("Phosphosite").loc[kin_gf][["PC9", "Erl", "R428", "Erl/R428", "KO Erl"]]
# cg = sns.clustermap(data=s_gf, cmap="coolwarm", row_cluster=False, col_cluster=False, linewidth=1, linecolor='w', square=True, robust=True, figsize=(4, 7))
# plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=35)
