"""
This creates Figure 2: Model figure
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from .common import subplotLabel, getSetup, import_phenotype_data, formatPhenotypesForModeling, plotDistanceToUpstreamKinase
from ..pre_processing import preprocessing
from ..clustering import DDMC
from ..plsr import plotStripActualVsPred, plotScoresLoadings


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
    ax[1].axis("off")

    # AXL p-sites clustermap
    # plot_AllSites("", X, "AXL", "AXL", ylim=False, type="Heatmap")
    ax[2].axis("off")

    # Centers
    plotCenters_together(ddmc, X, ax[3])

    # Predictions
    Xs, models = ComputeCenters(X, d, i, ddmc, 5)
    Xs.append(centers)
    models.append("DDMC mix")
    plotStripActualVsPred(ax[4], [3, 4, 2, 3, 4], Xs, y, models)

    # Scores & Loadings
    lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]
    plsr = PLSRegression(n_components=4)
    plotScoresLoadings(ax[5:7], plsr.fit(centers, y), centers, y, ddmc.n_components, lines, pcX=1, pcY=2)

    # Plot upstream kinases heatmap
    plotDistanceToUpstreamKinase(ddmc, [1, 2, 3, 4, 5], ax[7], num_hits=10)

    return f


def plotCenters_together(ddmc, X, ax):
    """Plot Cluster Centers together in same plot"""
    centers = pd.DataFrame(ddmc.transform()).T
    centers.columns = X.columns[7:]
    centers["Cluster"] = list(np.arange(ddmc.n_components) + 1)
    m = pd.melt(centers, id_vars=["Cluster"], value_vars=list(centers.columns), value_name="p-signal", var_name="Lines")
    m["p-signal"] = m["p-signal"].astype("float64")
    sns.set_context("paper", rc={'lines.linewidth': 1}) 
    palette ={1: "C0", 2: "C1", 3: "C2", 4: "C3", 5: "k"}
    sns.pointplot(x="Lines", y="p-signal", data=m, hue="Cluster", ax=ax, palette=palette, dashes=False, **{"linewidth": 0})


def ComputeCenters(X, d, i, ddmc, n_components):
    """Calculate cluster centers of  different algorithms."""
    # k-means
    labels = KMeans(n_clusters=n_components).fit(d.T).labels_
    x_ = X.copy()
    x_["Cluster"] = labels
    c_kmeans = x_.groupby("Cluster").mean().T

    # GMM
    ddmc_data = DDMC(i, n_components=n_components, SeqWeight=0, distance_method=ddmc.distance_method, random_state=ddmc.random_state).fit(d)
    c_gmm = ddmc_data.transform()

    # DDMC seq
    ddmc_seq = DDMC(i, n_components=n_components, SeqWeight=ddmc.SeqWeight + 20, distance_method=ddmc.distance_method, random_state=ddmc.random_state).fit(d)
    ddmc_seq_c = ddmc_seq.transform()

    # DDMC mix
    ddmc_c = ddmc.transform()
    return [c_kmeans, c_gmm, ddmc_seq_c, ddmc_c], ["Unclustered", "k-means", "GMM", "DDMC seq", "DDMC mix"]
