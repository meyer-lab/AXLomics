"""
This creates Supplemental Figure 4: Motifs
"""

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from .common import subplotLabel, getSetup
from ..clustering import MassSpecClustering
from ..pre_processing import preprocessing
import logomaker as lm


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 5), (2, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Load DDMC
    X = preprocessing(AXLwt_GF=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    data = X.select_dtypes(include=['float64']).T
    info = X.select_dtypes(include=['object'])
    model = MassSpecClustering(info, 5, SeqWeight=2, distance_method="PAM250").fit(X=data)
    lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]

    # Centers
    plotCenters(ax[:5], model, lines)

    # Plot motifs
    pssms, _ = model.pssms(PsP_background=True)
    plotMotifs([pssms[0], pssms[1], pssms[2], pssms[3], pssms[4]], axes=ax[5:10], titles=["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"], yaxis=[0, 11])

    return f


def plotCenters(ax, model, xlabels, yaxis=False, drop=False):
    centers = pd.DataFrame(model.transform()).T
    centers.columns = xlabels
    if drop:
        centers = centers.drop(drop)
    num_peptides = [np.count_nonzero(model.labels() == jj) for jj in range(1, model.n_components + 1)]
    for i in range(centers.shape[0]):
        cl = pd.DataFrame(centers.iloc[i, :]).T
        m = pd.melt(cl, value_vars=list(cl.columns), value_name="p-signal", var_name="Lines")
        m["p-signal"] = m["p-signal"].astype("float64")
        sns.lineplot(x="Lines", y="p-signal", data=m, color="#658cbb", ax=ax[i], linewidth=2)
        ax[i].set_xticklabels(xlabels, rotation=45)
        ax[i].set_xticks(np.arange(len(xlabels)))
        ax[i].set_ylabel("$log_{10}$ p-signal")
        ax[i].xaxis.set_tick_params(bottom=True)
        ax[i].set_xlabel("")
        ax[i].set_title("Cluster " + str(centers.index[i] + 1) + " Center " + "(" + "n=" + str(num_peptides[i]) + ")")
        if yaxis:
            ax[i].set_ylim([yaxis[0], yaxis[1]])


def plotMotifs(pssms, axes, titles=False, yaxis=False):
    """Generate logo plots of a list of PSSMs"""
    for i, ax in enumerate(axes):
        pssm = pssms[i].T
        if pssm.shape[0] == 11:
            pssm.index = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        elif pssm.shape[0] == 9:
            pssm.index = [-5, -4, -3, -2, -1, 1, 2, 3, 4]
        logo = lm.Logo(pssm,
                       font_name='Arial',
                       vpad=0.1,
                       width=.8,
                       flip_below=False,
                       center_values=False,
                       ax=ax)
        logo.ax.set_ylabel('log_{2} (Enrichment Score)')
        logo.style_xticks(anchor=1, spacing=1)
        if titles:
            logo.ax.set_title(titles[i] + " Motif")
        else:
            logo.ax.set_title('Motif Cluster ' + str(i + 1))
        if yaxis:
            logo.ax.set_ylim([yaxis[0], yaxis[1]])
