"""
This creates Figure 1: Phenotypic characterization of PC9 AXL mutants
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
from scipy.stats import ttest_ind
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
    plotPCA(ax[6:8], y, 3, ["Lines", "Treatment"], hue_scores="Lines", style_scores="Treatment", legendOut=True)

    return f


def plot_phenotype_heatmap(ax, d):
    """Make phenotype heatmap"""
    phe = pd.concat([d.iloc[:10, 0], d.iloc[:10, -1], d.iloc[10:20, -1], d.iloc[20:, -1]], axis=1)
    phe.columns = ["Cell Lines", "UT", "Erlotinib", "Erl + AF154"]
    sns.heatmap(phe.set_index("Cell Lines"), robust=True, cmap="bwr", ax=ax)
    ax.set_yticklabels(phe["Cell Lines"], rotation=0)


def pval_phenotypes(data, pheno, lines, all_lines, timepoint, fc=True):
    "For each phenotype Test: E vs EA across all cell lines and per cell line."
    out = np.empty(len(lines))
    for idx, line in enumerate(lines):
        aes = []
        es = []
        for d in data:
            d = d.set_index("Elapsed")
            l = d.loc[:, d.columns.str.contains(line)]
            aes.extend(l.loc[:, l.columns.str.contains("-A/E")].loc[timepoint].values)
            es.extend(l.loc[:, l.columns.str.contains("-E")].loc[timepoint].values)
        out[idx] = ttest_ind(es, aes)[1]

    table = pd.DataFrame()
    table["Cell Line"] = all_lines
    table[pheno] = out

    return table.set_index("Cell Line")


def Island_pvals(c, all_lines):
    mutants = list(set(c.index))
    muts = []
    out = np.empty(len(mutants))
    for idx, m in enumerate(mutants):
        mut = c.loc[m]
        e = mut[mut["Treatment"] == "e"]["K Estimate"].values
        ae = mut[mut["Treatment"] == "ae"]["K Estimate"].values
        out[idx] = ttest_ind(e, ae)[1]
        muts.append(m)

    table = pd.DataFrame()
    table["Cell Line"] = muts
    table["Island"] = out
    table = table.set_index("Cell Line")
    table = table.rename(index={"M7": "Y698F", "M4": "Y634F", "M5": "Y643F", "M10":"Y726F", "M11":"Y750F", "M15": "Y821F", "PC9": "WT", "KIN": "KI"})

    return table.T[all_lines].T



# —————Supplement—————:

# mutants = ['PC9', 'KO', 'KIN', 'KD', 'M4', 'M5', 'M7', 'M10', 'M11', 'M15']
# all_lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]

# # Cell Viability
# mutants = ['PC9', 'AXL KO', 'Kin', 'Kdead', 'M4', 'M5', 'M7', 'M10', 'M11', 'M15']
# cv[-1]["Elapsed"] = cv[-2]["Elapsed"]
# cv_pvals = pval_phenotypes(cv, "Cell Viability", mutants, all_lines, int(96))

# # Cell Death"
# red[-1]["Elapsed"] = red[-2]["Elapsed"]
# cd_pvals = pval_phenotypes(red, "Cell Death", mutants, all_lines, int(96))

# # Cell Migration
# mutants = ['PC9', 'KO', 'KIN', 'KD', 'M4', 'M5', 'M7', 'M10', 'M11', 'M15']
# sw_pvals = pval_phenotypes(sw, "Migration", mutants, all_lines, int(10))

# # Cell Island
# c_brs = DataFrameRipleysK('48hrs', mutants, ['ut', 'e', 'ae'], 6, np.linspace(1, 14.67, 1), merge=False).reset_index().set_index("Mutant")
# i_pvals = Island_pvals(c_brs, all_lines)

# pvals = pd.concat([cv_pvals, cd_pvals, sw_pvals, i_pvals], axis=1)

# sns.heatmap(pvals, vmax=(0.051))

# pval = []
# for p in y.columns[2:]:
#     e = y.set_index("Treatment").loc["-E"][p].values
#     ea =  y.set_index("Treatment").loc["A/E"][p].values
#     pval.append(ttest_ind(e, ea)[1])

# y = y[y["Treatment"] == "A/E"].drop("Treatment", axis=1).set_index("Lines")
# sns.clustermap(y, robust=True, cmap="bwr", figsize=(7, 5))