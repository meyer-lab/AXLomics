"""
This creates Figure 1: CPTAC LUAD analysis of AXL hi vs AXL low tumors
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib
from ..clinical_data import *
from scipy.stats import ttest_ind
from .common import subplotLabel, getSetup, import_phenotype_data, IndividualTimeCourses, formatPhenotypesForModeling
from ..pca import plotPCA
from ..distances import PlotRipleysK

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

    lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    mutants = ['PC9', 'KO', 'KIN', 'KD', 'M4', 'M5', 'M7', 'M10', 'M11', 'M15']
    tr1 = ["-UT", "-E", "-A/E"]
    tr2 = ["Untreated", "Erlotinib", "Erl + AF154"]
    cv = import_phenotype_data(phenotype="Cell Viability")
    red = import_phenotype_data(phenotype="Cell Death")
    sw = import_phenotype_data(phenotype="Migration")
    c = import_phenotype_data(phenotype="Island")

    ax, f = getSetup((11, 17), (8, 5))
    cv_f, red_f, sw_f, ci_f = [], [], [], []
    for i, line in enumerate(lines):
        cv_c = IndividualTimeCourses(cv, 96, lines, tr1, tr2, "fold-change confluency", TimePointFC=24, TreatmentFC=False, plot=line, ax_=ax[i], ylim=[0.8, 10], out=True)
        cv_c["Lines"] = line
        cv_f.append(cv_c)
        red_c = IndividualTimeCourses(red, 96, lines, tr1, tr2, "fold-change apoptosis (YOYO+)", TimePointFC=24, plot=line, ax_=ax[i + 10], ylim=[0, 13], out=True)
        red_c["Lines"] = line
        red_f.append(red_c)
        sw_c = IndividualTimeCourses(sw, 24, lines, tr1, tr2, "RWD %", plot=line, ax_=ax[i + 20], out=True)
        sw_c["Lines"] = line
        sw_f.append(sw_c)
        ci_c = PlotRipleysK(mutant=mutants[i], ax=ax[i + 30], title=line, out=True)
        ci_c["Lines"] = line
        ci_f.append(ci_c)

    # PCA phenotypes
    y = formatPhenotypesForModeling(cv, red, sw, c)
    plotPCA(ax[6:8], y, 3, ["Lines", "Treatment"], hue_scores="Lines", style_scores="Treatment", legendOut=True)

    return f


def make_pval_table_AXLphenotypes(cv_f, red_f, sw_f, ci_f):
    cvF = pd.concat(cv_f).rename(columns={"fold-change confluency":"Data"})
    redF = pd.concat(red_f).rename(columns={"fold-change apoptosis (YOYO+)":"Data"})
    swF = pd.concat(sw_f).rename(columns={"RWD %":"Data"})
    ciF = pd.concat(ci_f).rename(columns={"Condition":"Treatments", "K Estimate":"Data"})
    ciF["Treatments"] = ciF["Treatments"].replace("AF154 + Erlotinib", "Erl + AF154")

    ds = [cvF, redF, swF, ciF]
    dsL = ["Cell Viability", "Apoptosis", "Migration", "Island"]

    df = pd.DataFrame()
    pvals = []
    lines = []
    phe = []
    for ii, d in enumerate(ds):
        for l in list(set(d["Lines"])):
            cl = d[d["Lines"] == l]
            c_erl = cl[cl["Treatments"] == "Erlotinib"].loc[:, "Data"].values
            c_ea = cl[cl["Treatments"] == "Erl + AF154"].loc[:, "Data"].values
            pvals.append(mannwhitneyu(c_erl, c_ea)[1])
            lines.append(l)
            phe.append(dsL[ii])

    df["Cell Line"] = lines
    df["Phenotype"] = phe
    df["p-value"] = pvals

    return df.sort_values(by=["Phenotype", "Cell Line"])



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





# pair_corr(rna, "AXL", "CYR61", ax=ax[3])
# pair_corr(rna, "AXL", "CTGF", ax=ax[4])

# phosHL = make_AXL_categorical_data(phos, prot, phospho=True, by_thres=True)
# sns.violinplot(data=phosHL.loc["YAP1", "S382-p"], x="AXL", y="p-site signal", color="white", ax=ax[5]).set_ylabel("YAP1 S382-p norm log(p-signal)")

# plot_YAPlevels_byStage(phosR_tumor, pmut, ax[6:8])

# protHL = make_AXL_categorical_data(prot, prot, phospho=False, by_thres=True)
# rnaHL = make_AXL_categorical_data(rna, prot, phospho=False, by_thres=True)
# sns.violinplot(data=rnaHL.loc["TWIST1"], x="AXL", y="log(expression)", color="white", ax=ax[0])
# sns.violinplot(data=rnaHL.loc["VIM"], x="AXL", y="log(expression)", color="white", ax=ax[1])
# sns.violinplot(data=rnaHL.loc["CDH11"], x="AXL", y="log(expression)", color="white", ax=ax[2])
# sns.violinplot(data=protHL.loc["CDH11"], x="AXL", y="log(expression)", color="white", ax=ax[3])

