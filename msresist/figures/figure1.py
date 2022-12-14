"""
This creates Figure 1: CPTAC LUAD analysis of AXL hi vs AXL low tumors
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib
from .common import subplotLabel, getSetup
from ..clinical_data import *
from ..pre_processing import filter_NaNpeptides
from scipy.stats import ttest_ind
from .common import subplotLabel, getSetup, import_phenotype_data, formatPhenotypesForModeling
from ..pca import plotPCA

sns.set(color_codes=True)


path = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.max_columns", 30)


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (4, 3), multz={0: 1, 6:1})
    ax, f = getSetup((16, 12), (3, 3), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    phos = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=2)
    prot = pd.read_csv("msresist/data/MS/CPTAC/CPTAC_LUAD_Protein.csv").drop_duplicates(subset="geneSymbol").set_index("geneSymbol").select_dtypes(include=float).iloc[:, 4:].reset_index()
    rna = pd.read_csv("msresist/data/MS/CPTAC/CPTAC_LUAD_RNAseq.csv").drop_duplicates(subset="geneSymbol")

    _, phosR_tumor, _ = preprocess_phospho(phos)
    protR, protR_tumor, _ = preprocess_data(prot)
    _, rnaR_tumor, _ = preprocess_data(rna)
    y = formatPhenotypesForModeling(cv, red, sw, c)

    pmut = pd.read_csv("/home/marcc/AXLomics/msresist/data/MS/CPTAC/Patient_Mutations.csv")
    pmut = pmut[~pmut["Sample.ID"].str.contains("IR")]

    plot_AXLlevels_byStage(protR, pmut, ax[:2])
    # Phenotypes diagram
    ax[1].axis("off")

    # plot_GSEA_term(rna, prot, term="CORDENONSI YAP CONSERVED SIGNATURE") gseapy.gseaplot doesn't have an ax argument, add plot in illustrator    
    ax[2].axis("off")

    phos = phosR_tumor
    prot = protR_tumor
    rna = rnaR_tumor
    # Heatmaps
    plot_phenotype_heatmap(ax[2], y[["Lines", "Treatment", "Viability"]])
    plot_phenotype_heatmap(ax[3], y[["Lines", "Treatment", "Apoptosis"]])
    plot_phenotype_heatmap(ax[4], y[["Lines", "Treatment", "Migration"]])
    plot_phenotype_heatmap(ax[5], y[["Lines", "Treatment", "Island"]])

    pair_corr(rna, "AXL", "CYR61", ax=ax[3])
    pair_corr(rna, "AXL", "CTGF", ax=ax[4])

    phosHL = make_AXL_categorical_data(phos, prot, phospho=True, by_thres=True)
    sns.violinplot(data=phosHL.loc["YAP1", "S382-p"], x="AXL", y="p-site signal", color="white", ax=ax[5]).set_ylabel("YAP1 S382-p norm log(p-signal)")

    plot_YAPlevels_byStage(phosR_tumor, pmut, ax[6:8])

    protHL = make_AXL_categorical_data(prot, prot, phospho=False, by_thres=True)
    rnaHL = make_AXL_categorical_data(rna, prot, phospho=False, by_thres=True)
    sns.violinplot(data=rnaHL.loc["TWIST1"], x="AXL", y="log(expression)", color="white", ax=ax[0])
    sns.violinplot(data=rnaHL.loc["VIM"], x="AXL", y="log(expression)", color="white", ax=ax[1])
    sns.violinplot(data=rnaHL.loc["CDH11"], x="AXL", y="log(expression)", color="white", ax=ax[2])
    sns.violinplot(data=protHL.loc["CDH11"], x="AXL", y="log(expression)", color="white", ax=ax[3])
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
