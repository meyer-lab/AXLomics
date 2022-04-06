"""
This creates Figure 1: Phenotypic characterization of PC9 AXL mutants
"""

import os
import pandas as pd
import matplotlib
import seaborn as sns
from .common import subplotLabel, getSetup, import_phenotype_data, formatPhenotypesForModeling
from ..pca import plotPCA

sns.set(color_codes=True)


path = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.max_columns", 30)

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((16, 12), (3, 3), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Read in phenotype data
    cv = import_phenotype_data(phenotype="Cell Viability")
    red = import_phenotype_data(phenotype="Cell Death")
    sw = import_phenotype_data(phenotype="Migration")
    c = import_phenotype_data(phenotype="Island")
    y = formatPhenotypesForModeling(cv, red, sw, c)

    # AXL mutants cartoon
    ax[0].axis("off")

    # Phenotypes diagram
    ax[1].axis("off")

    # Heatmaps
    plot_phenotype_heatmap(ax[2], y[["Lines", "Treatment", "Viability"]])
    plot_phenotype_heatmap(ax[3], y[["Lines", "Treatment", "Apoptosis"]])
    plot_phenotype_heatmap(ax[4], y[["Lines", "Treatment", "Migration"]])
    plot_phenotype_heatmap(ax[5], y[["Lines", "Treatment", "Island"]])

    # PCA phenotypes
    y = formatPhenotypesForModeling(cv, red, sw, c)
    plotPCA(ax[6:8], y, 3, ["Lines", "Treatment"], "Phenotype", hue_scores="Lines", style_scores="Treatment", legendOut=True)

    return f


def plot_phenotype_heatmap(ax, d):
    """Make phenotype heatmap"""
    phe = pd.concat([d.iloc[:10, 0], d.iloc[:10, -1], d.iloc[10:20, -1], d.iloc[20:, -1]], axis=1)
    phe.columns = ["Cell Lines", "UT", "Erlotinib", "Erl + AF154"]
    sns.heatmap(phe.set_index("Cell Lines"), robust=True, cmap="bwr", ax=ax)
    ax.set_yticklabels(phe["Cell Lines"], rotation=0)