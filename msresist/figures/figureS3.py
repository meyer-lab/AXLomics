"""
This creates Supplemental Figure 2: Cell migration and island
"""

import matplotlib
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup, IndividualTimeCourses, import_phenotype_data, barplot_UtErlAF154
from ..distances import BarPlotRipleysK, PlotRipleysK
from msresist.pca import plotBootPCA, bootPCA, preprocess_ID


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (4, 6), multz={0: 1, 12: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Read in migration data
    sw = import_phenotype_data(phenotype="Migration")

    # Labels
    lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    tr1 = ["-UT", "-E", "-A/E"]
    tr2 = ["Untreated", "Erlotinib", "Erl + AF154"]
    t1 = ["UT", "AF", "-E", "A/E"]
    t2 = ["Untreated", "AF154", "Erlotinib", "Erl + AF154"]
    colors = ["white", "windows blue", "scarlet"]
    mutants = ['PC9', 'KO', 'KIN', 'KD', 'M4', 'M5', 'M7', 'M10', 'M11', 'M15']
    itp = 24

    # Time courses
    for i, line in enumerate(lines):
        IndividualTimeCourses(sw, 24, lines, t1, t2, "RWD %", plot=line, ax_=ax[i + 1])
        PlotRipleysK('48hrs', mutants[i], ['ut', 'e', 'ae'], 6, ax=ax[i + 12], title=line)
        ax[i + 12].set_ylim(0, 40)

    # Bar plots
    # barplot_UtErlAF154(ax[0], lines, sw, 14, tr1, tr2, "Relative Wound Density (RWD)", "Cell Migration (14h)", TreatmentFC=False, colors=colors, TimePointFC=0)
    # BarPlotRipleysK(ax[11], '48hrs', mutants, lines, ['ut', 'e', 'ae'], tr2, 6, np.linspace(1.5, 14.67, 1), colors, TreatmentFC=False, ylabel="K estimate")
    # ax[11].set_title("Island effect (48h)")

    return f


def plot_bioID(ax):
    bid = preprocess_ID(linear=True, npepts=7, FCcut=10)

    # Scores
    bootScor_m, bootScor_sd, bootLoad_m, bootLoad_sd, _, varExp = bootPCA(bid, 4, "Gene", method="NMF", n_boots=100)
    plotBootPCA(ax[0], bootScor_m, bootScor_sd, varExp, title="NMF Scores", LegOut=False, annotate=False, colors=False)
    ax[0].legend(prop={'size': 10})

    plotBootPCA(ax[2], bootScor_m, bootScor_sd, varExp, title="NMF Scores", X="PC2", Y="PC3", LegOut=False, annotate=False, colors=False)
    ax[2].legend(prop={'size': 10})

    # Loadings
    plotBootPCA(ax[1], bootLoad_m, bootLoad_sd, varExp, title="NMF Loadings", LegOut=False, annotate=True, colors=False)
    ax[1].get_legend().remove()

    plotBootPCA(ax[3], bootLoad_m, bootLoad_sd, varExp, title="NMF Loadings", X="PC2", Y="PC3", LegOut=False, annotate=True, colors=False)
    ax[3].get_legend().remove()