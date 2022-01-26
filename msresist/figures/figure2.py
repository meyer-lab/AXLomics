"""
This creates Figure 2: Model figure
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from .common import subplotLabel, getSetup, import_phenotype_data, formatPhenotypesForModeling
from ..pre_processing import preprocessing, MeanCenter
from ..clustering import MassSpecClustering
from ..motifs import KinToPhosphotypeDict
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
    ddmc = MassSpecClustering(i, ncl=5, SeqWeight=2, distance_method="PAM250", random_state=5).fit(d)
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
    centers["Cluster"] = [1, 2, 3, 4, 5]
    m = pd.melt(centers, id_vars=["Cluster"], value_vars=list(centers.columns), value_name="p-signal", var_name="Lines")
    m["p-signal"] = m["p-signal"].astype("float64")
    sns.set_context("paper", rc={'lines.linewidth': 1}) 
    palette ={1: "C0", 2: "C1", 3: "C2", 4: "C3", 5: "k"}
    sns.pointplot(x="Lines", y="p-signal", data=m, hue="Cluster", ax=ax, palette=palette, dashes=False, **{"linewidth": 0})


def ComputeCenters(X, d, i, ddmc, ncl):
    """Calculate cluster centers of  different algorithms."""
    # k-means
    labels = KMeans(n_clusters=ncl).fit(d.T).labels_
    x_ = X.copy()
    x_["Cluster"] = labels
    c_kmeans = x_.groupby("Cluster").mean().T

    # GMM
    ddmc_data = MassSpecClustering(i, ncl=ncl, SeqWeight=0, distance_method=ddmc.distance_method, random_state=ddmc.random_state).fit(d)
    c_gmm = ddmc_data.transform()

    # DDMC seq
    ddmc_seq = MassSpecClustering(i, ncl=ncl, SeqWeight=ddmc.SeqWeight + 20, distance_method=ddmc.distance_method, random_state=ddmc.random_state).fit(d)
    ddmc_seq_c = ddmc_seq.transform()

    # DDMC mix
    ddmc_c = ddmc.transform()
    return [c_kmeans, c_gmm, ddmc_seq_c, ddmc_c], ["Unclustered", "k-means", "GMM", "DDMC seq", "DDMC mix"]


def plotDistanceToUpstreamKinase(model, clusters, ax, kind="strip", num_hits=5, additional_pssms=False, add_labels=False, title=False, PsP_background=True):
    """Plot Frobenius norm between kinase PSPL and cluster PSSMs"""
    ukin = model.predict_UpstreamKinases(additional_pssms=additional_pssms, add_labels=add_labels, PsP_background=PsP_background)
    ukin_mc = MeanCenter(ukin, mc_col=True, mc_row=True)
    cOG = np.array(clusters).copy()
    if isinstance(add_labels, list):
        clusters += add_labels
    data = ukin_mc.sort_values(by="Kinase").set_index("Kinase")[clusters]
    if kind == "heatmap":
        sns.heatmap(data.T, ax=ax, xticklabels=data.index)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=7)
        ax.set_ylabel("Cluster")

    elif kind == "strip":
        data = pd.melt(data.reset_index(), id_vars="Kinase", value_vars=list(data.columns), var_name="Cluster", value_name="Frobenius Distance")
        if isinstance(add_labels, list):
            # Actual ERK predictions
            data["Cluster"] = data["Cluster"].astype(str)
            d1 = data[~data["Cluster"].str.contains("_S")]
            sns.stripplot(data=d1, x="Cluster", y="Frobenius Distance", ax=ax[0])
            print(cOG)
            AnnotateUpstreamKinases(model, list(cOG) + ["ERK2+"], ax[0], d1, 1)

            # Shuffled
            d2 = data[data["Kinase"] == "ERK2"]
            d2["Shuffled"] = ["_S" in s for s in d2["Cluster"]]
            d2["Cluster"] = [s.split("_S")[0] for s in d2["Cluster"]]
            sns.stripplot(data=d2, x="Cluster", y="Frobenius Distance", hue="Shuffled", ax=ax[1], size=8)
            ax[1].set_title("ERK2 Shuffled Positions")
            ax[1].legend(prop={'size': 10}, loc='lower left')
            DrawArrows(ax[1], d2)

        else:
            sns.stripplot(data=data, x="Cluster", y="Frobenius Distance", ax=ax)
            AnnotateUpstreamKinases(model, clusters, ax, data, num_hits)
            if title:
                ax.set_title(title)


def AnnotateUpstreamKinases(model, clusters, ax, data, num_hits=1):
    """Annotate upstream kinase predictions"""
    data.iloc[:, 1] = data.iloc[:, 1].astype(str)
    pssms, _ = model.pssms()
    for ii, c in enumerate(clusters, start=1):
        cluster = data[data.iloc[:, 1] == str(c)]
        hits = cluster.sort_values(by="Frobenius Distance", ascending=True)
        hits.index = np.arange(hits.shape[0])
        hits["Phosphoacceptor"] = [KinToPhosphotypeDict[kin] for kin in hits["Kinase"]]
        try:
            cCP = pssms[c - 1].iloc[:, 5].idxmax()
        except BaseException:
            cCP == "S/T"
        if cCP == "S" or cCP == "T":
            cCP = "S/T"
        hits = hits[hits["Phosphoacceptor"] == cCP]
        for jj in range(num_hits):
            ax.annotate(hits["Kinase"].iloc[jj], (ii - 1, hits["Frobenius Distance"].iloc[jj] - 0.01), fontsize=8)
    ax.legend().remove()
    ax.set_title("Kinase vs Cluster Motif")


def DrawArrows(ax, d2):
    data_shuff = d2[d2["Shuffled"]]
    actual_erks = d2[d2["Shuffled"] == False]
    arrow_lengths = np.add(data_shuff["Frobenius Distance"].values, abs(actual_erks["Frobenius Distance"].values)) * -1
    for dp in range(data_shuff.shape[0]):
        ax.arrow(dp,
                 data_shuff["Frobenius Distance"].iloc[dp] - 0.1,
                 0,
                 arrow_lengths[dp] + 0.3,
                 head_width=0.25,
                 head_length=0.15,
                 width=0.025,
                 fc='black',
                 ec='black')
