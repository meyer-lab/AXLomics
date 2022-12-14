"""
This creates Supplemental Figure 4: Motifs
"""

import numpy as np
import pandas as pd 
import matplotlib
import seaborn as sns
from msresist.figures.common import subplotLabel, getSetup, plotMotifs
from msresist.pca import plotBootPCA, bootPCA, preprocess_ID
from msresist.clustering import DDMC
from msresist.pre_processing import preprocessing
from msresist.figures.figure3 import lines


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 9), (3, 4), multz={4:1, 6:1, 8:1, 10:1})

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

    # Plot motifs
    pssms, _ = ddmc.pssms(PsP_background=False)
    plotMotifs([pssms[0], pssms[1], pssms[2], pssms[3]], axes=ax[0:4], titles=["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"], yaxis=[0, 11])

    bid = preprocess_ID(linear=True, npepts=7, FCcut=10)

    # Scores
    bootScor_m, bootScor_sd, bootLoad_m, bootLoad_sd, bootScor, varExp = bootPCA(bid, 4, "Gene", method="NMF", n_boots=100)
    plotBootPCA(ax[4], bootScor_m, bootScor_sd, varExp, title="NMF Scores", LegOut=False, annotate=False, colors=False)
    ax[4].legend(prop={'size': 10})

    plotBootPCA(ax[6], bootScor_m, bootScor_sd, varExp, title="NMF Scores", X="PC2", Y="PC3", LegOut=False, annotate=False, colors=False)
    ax[6].legend(prop={'size': 10})

    # Loadings
    plotBootPCA(ax[5], bootLoad_m, bootLoad_sd, varExp, title="NMF Loadings", LegOut=False, annotate=True, colors=False)
    ax[5].get_legend().remove()

    plotBootPCA(ax[7], bootLoad_m, bootLoad_sd, varExp, title="NMF Loadings", X="PC2", Y="PC3", LegOut=False, annotate=True, colors=False)
    ax[7].get_legend().remove()

    return f
