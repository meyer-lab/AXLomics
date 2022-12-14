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


def kinase_heatmap(X, prot_dict, ax, FC=False):
    """ Make a heatmap out of a dictionary wih p-sites """
    out = []
    for p, s in prot_dict:
        out.append(X.set_index(["Gene", "Position"]).loc[p, s])
    out = pd.concat(out).select_dtypes(include=[float])
    if FC:
        for ii in range(out.shape[1]):
            out.iloc[:, ii] /= out.iloc[:, 0]
    sns.heatmap(out, cmap="bwr", ax=ax)
    ax.set_xticklabels(lines)
    ax.set_xlabel("AXL Y—>F mutants")
    ax.set_ylabel("")


def kinases_clustermap(X):
    """Clustermap of all kinases showing a minimum variance acros mutants"""
    k = X[X["Protein"].str.contains("kinase")]
    XIDX = np.any(k.iloc[:, 8:-1] <= -0.5, axis=1) | np.any(k.iloc[:, 8:-1] >= 0.5, axis=1)
    k = k.iloc[list(XIDX), :].set_index(["Gene", "Position"]).select_dtypes(include=[float])
    k = k.drop("AXL")
    sns.clustermap(k, cmap="bwr", xticklabels=lines)


def AXL_volcanoplot(X):
    """AXL vs No AXL volcano plot"""
    axl_in = X[["PC9 A", "KI A"]].values
    axl_out = X[["KO A", "Kd A"]].values
    pvals = f_oneway(axl_in, axl_out, axis=1)[1]
    pvals = multipletests(pvals)[1]
    fc = axl_in.mean(axis=1) - axl_out.mean(axis=1)
    pv = pd.DataFrame()
    pv["Peptide"] = [g + ";" + p for g, p in list(zip(X["Gene"], X["Position"]))]
    pv["logFC"] = fc
    pv["p-values"] = pvals
    pv = pv.sort_values(by="p-values")
    visuz.GeneExpression.volcano(
        df=pv,
        lfc='logFC',
        pv='p-values',
        show=True,
        geneid="Peptide",
        lfc_thr=(
            0.5,
            0.5),
        genenames="deg",
        color=(
            "#00239CFF",
            "grey",
            "#E10600FF"),
        figtype="svg",
        gstyle=2,
        axtickfontname='Arial')
