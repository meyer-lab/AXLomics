"""
This creates Figure 1: CPTAC LUAD analysis of AXL hi vs AXL low tumors
"""

import pandas as pd
import seaborn as sns
import matplotlib
from .common import subplotLabel, getSetup
from ..clinical_data import *
from ..pre_processing import filter_NaNpeptides


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (4, 3), multz={0: 1, 6:1})

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

    pmut = pd.read_csv("/home/marcc/AXLomics/msresist/data/MS/CPTAC/Patient_Mutations.csv")
    pmut = pmut[~pmut["Sample.ID"].str.contains("IR")]

    plot_AXLlevels_byStage(protR, pmut, ax[:2])

    # plot_GSEA_term(rna, prot, term="CORDENONSI YAP CONSERVED SIGNATURE") gseapy.gseaplot doesn't have an ax argument, add plot in illustrator    
    ax[2].axis("off")

    phos = phosR_tumor
    prot = protR_tumor
    rna = rnaR_tumor

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

    return f
