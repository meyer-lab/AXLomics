"""
This creates Figure 3: ABL/SFK/YAP experimental validations
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from .common import subplotLabel, getSetup, TimePointFoldChange
from ..pre_processing import preprocessing
from ..clustering import DDMC


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 14), (4, 4), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Clustermap of dasatinib-responsive peptides
    # plot_dasatinib_MS_clustermaps(DR=True, AXLs=False) Export Figure 1A. Clustermap doesn't have ax argument
    ax[0].axis("off")

    # Enrichment of dasatinib-responsive peptides in clusters 2&3
    plotHyerGeomTestDasDRGenes(ax[1])

    # FAK pathway in AXL WT vs AXL KO
    FAKpatwhay_AXL_WTvsKO(ax[2], WTvsKO=True)

    # YAP enrichment E vs EA
    ax[3].axis("off")

    # YAP western blots +/- E and +/- A with varying cell density
    ax[4].axis("off")

    # Protein and phospho-AXL levels in KO and WT Dasatinib DR
    plot_protein_and_phosphoAXL_dasatinib(ax[5])

    # YAP blots with YAP DR
    ax[5].axis("off")

    # Immunofluorescence showing YAP and AXL in island and isolated cells
    ax[6].axis("off")

    return f


def plot_protein_and_phosphoAXL_dasatinib(ax):
    ad = pd.read_csv("/home/marcc/AXLomics/msresist/data/Validations/Luminex/102022-Luminex_AXL_Das.csv")
    ad.iloc[:, 1:] = np.log(ad.iloc[:, 1:])
    ad = pd.melt(ad, id_vars="Sample", value_vars=["tAXL", "pAXL"], var_name="t/p", value_name="log(abundance)")

    _, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.stripplot(ad, x="Sample", y="log(abundance)", hue="t/p", dodge=True, ax=ax)
    sns.boxplot(ad, x="Sample", y="log(abundance)", hue="t/p", width=0.5, color="white", ax=ax)
    ax.legend().remove()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)


def FAKpatwhay_AXL_WTvsKO(ax, WTvsKO=True, ds="AXL"):
    """ Plot the phosphorylation signal of identified proteins in Wilson et al Oncotarget 2014 as  members of FAK patwhay"""
    if ds == "AXL":
        # Import siganling data
        MS = preprocessing(AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
        xlabel = "Cell Line"
        fakS_phos = ["PTK2 Y570-p", "NEDD9 T185-p", "PEAK1 Y531-p", "CDK1 Y15-p", "AFAP1L2 S389-p", "PIK3R2 Y460-p;S457-p", "MYH9 Y9-p;Y11-p", "VCL Y692-p", "TNK2 Y859-p", "ACTN1 Y215-p", "BCAR3 Y212-p", "ABL1 Y185-p", "CNN3 Y261-p", "CAVIN1 Y308-p"] # FAK signature gene from Wilson et al 2014
        lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]

    elif ds == "Dasatinib":
        MS = preprocessing(AXL_Das_DR=True, Vfilter=True, FCfilter=False, log2T=True, mc_row=True)
        xlabel = "Sample"
        lines = MS.columns[6:]
        fakS_phos = ["PTK2 Y925-p", "NEDD9 Y345-p", "PEAK1 Y531-p", "CDK1 T14-p", "AFAP1L2 S389-p", "PIK3R2 Y605-p", "TNK2 Y859-p", "ACTN1 S244-p", "BCAR3 Y117-p", "ABL1 Y393-p"] # FAK signature gene from Wilson et al 2014

    # Fit DDMC
    MS.insert(0, "Phosphosite", [g + " " + p for g, p in zip(list(MS["Gene"]), list(MS["Position"]))])
    d = MS.set_index(["Gene", "Phosphosite"]).select_dtypes(include=[float])
    d.columns = lines
    d = d.reset_index()

    if WTvsKO: # plot just WT and KO
        phos_fak = d.set_index("Phosphosite").loc[:, ["WT", "KO"]].loc[fakS_phos].reset_index()
        phos_fak.columns = ["Phosphosite", "PC9 WT", "PC9 AXL KO"]
    else: # plot all samples
        phos_fak = d.set_index("Phosphosite").iloc[:, 1:].loc[fakS_phos].reset_index()

    phos_fak = pd.melt(phos_fak, value_vars=list(phos_fak.columns[1:]), id_vars="Phosphosite", var_name=xlabel, value_name="norm log(p-signal)")
    phos_fak.iloc[phos_fak[phos_fak["Phosphosite"].str.contains("CDK1")].index, -1] *= -1

    # plot
    sns.stripplot(phos_fak, x=xlabel, y="norm log(p-signal)", dodge=False, linewidth=1, hue=xlabel, ax=ax)
    sns.boxplot(phos_fak, x=xlabel, y="norm log(p-signal)", ax=ax, width=0.5, color="white").set_title("FAK pathway (Wilson et al 2014)")
    ax.legend().remove()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)


def plot_YAPinhibitorTimeLapse(ax, X, ylim=False):
    """For WT and KO, plot cell confluency in each treatment separately, hue are the inhibitor concentrations."""
    lines = ["WT", "KO"]
    treatments = ["UT", "E", "E/R", "E/A"]
    for i, line in enumerate(lines):
        for j, treatment in enumerate(treatments):
            if i > 0:
                j += 4
            m = X[X["Lines"] == line]
            m = m[m["Condition"] == treatment]
            sns.lineplot(x="Elapsed", y="Fold-change confluency", hue="Inh_concentration", data=m, ci=68, ax=ax[j])
            ax[j].set_title(line + "-" + treatment)
            if ylim:
                ax[j].set_ylim(ylim)
            if i != 0 or j != 0:
                ax[j].get_legend().remove()
            else:
                ax[j].legend(prop={'size': 10})


def transform_DRviability(data, units, itp):
    """Transform to initial time point and convert into seaborn format"""
    new = fold_change_acrossBRs(data, itp)
    c = pd.concat(new, axis=0)
    c = pd.melt(c, id_vars="Elapsed", value_vars=c.columns[1:], var_name="Lines", value_name="Fold-change confluency")
    c["Condition"] = [s.split(" ")[1].split(" ")[0] for s in c["Lines"]]
    c["Inh_concentration"] = [s[4:].split(" ")[1] for s in c["Lines"]]
    c["Lines"] = [s.split(" ")[0] for s in c["Lines"]]
    c = c[["Elapsed", "Lines", "Condition", "Inh_concentration", "Fold-change confluency"]]
    c = c[c["Elapsed"] >= itp]
    c["IC_n"] = [float(s.split(units)[0]) for s in c["Inh_concentration"]]
    return c.sort_values(by="IC_n").drop("IC_n", axis=1)


def fold_change_acrossBRs(data, itp):
    """Compute fold change to initial time point in every BR.
    Note that data should always be a list, even if just one BR."""
    new = []
    for _, mat in enumerate(data):
        new.append(TimePointFoldChange(mat, itp))
    return new


def MeanTRs(X):
    """Merge technical replicates of 2 BR by taking the mean."""
    idx = [np.arange(0, 6) + i for i in range(1, X.shape[1], 12)]
    for i in idx:
        for j in i:
            X.iloc[:, j] = X.iloc[:, [j, j + 6]].mean(axis=1)
            X.drop(X.columns[j + 6], axis="columns")

    return X.drop(X.columns[[j + 6 for i in idx for j in i]], axis="columns")


def plotHyerGeomTestDasDRGenes(ax):
    """Data from https://systems.crump.ucla.edu/hypergeometric/index.php where:
    - N = common peptides across both expts
    - M = cluster 4 among N
    - s = das responding among N
    - k = overlap
    Counts generated using GenerateHyperGeomTestParameters()."""
    hg = pd.DataFrame()
    hg["Cluster"] = np.arange(5) + 1
    hg["p_value"] = -np.log10[0.527, 0.003, 0.002, 0.404, 0.0557]
    hg["test"] = ["", "**", "**", "", "ns"]
    sns.barplot(data=hg, x="Cluster", y="p_value", ax=ax, color="darkblue", **{"linewidth": 1}, **{"edgecolor": "black"})
    ax.set_title("Enrichment of Das-responsive Peptides")
    ax.set_ylim((0, 0.55))
    for _, row in hg.iterrows():
        ax.text(row.Cluster - 1, row.p_value + 0.01, row.test, color='black', ha="center")


def GenerateHyperGeomTestParameters(A, X, dasG, cluster):
    """Generate parameters to calculate p-value for under- or over-enrichment based on CDF of the hypergeometric distribution."""
    N = list(set(A["Gene"]).intersection(set(X["Gene"])))
    cl = A[A["Cluster"] == cluster]
    M = list(set(cl["Gene"]).intersection(set(N)))
    s = list(set(dasG).intersection(set(N)))
    k = list(set(s).intersection(set(M)))
    return (len(k), len(s), len(M), len(N))


def plot_InhDR_timepoint(ax, inhibitor, cl="WT", itp=24):
    """Plot inhibitor DR at specified time point."""
    if inhibitor == "Dasatinib":
        inh = [merge_TRs("Dasatinib_Dose_BR3.csv", 2), merge_TRs("Dasatinib_2fixed.csv", 2)]
        units = "nM"
        time = 96
    elif inhibitor == "CX-4945":
        inh = [merge_TRs("CX_4945_BR1_dose.csv", 2), merge_TRs("02032022-CX_4945_BR3_dose.csv", 2)]
        units = "uM"
        time = 96
    elif inhibitor == "Volasertib":
        inh = [merge_TRs("Volasertib_Dose_BR1.csv", 2), merge_TRs("Volasertib_Dose_BR2.csv", 2)]
        units = "nM"
        time = 72
    data = transform_DRviability(inh, units, itp)
    tp = data[data["Elapsed"] == time]
    tp = tp[tp["Lines"] == cl]
    sns.lineplot(data=tp, x="Inh_concentration", y="Fold-change confluency", hue="Condition", ci=68, ax=ax)
    ax.set_xlabel("[" + inhibitor + "]")


def merge_TRs(filename, nTRs):
    """Merge technical replicates of an experiment"""
    path = "msresist/data/Validations/CellGrowth/"
    inh = pd.read_csv(path + filename)
    for i in range(1, nTRs):
        inh.columns = [col.split("." + str(i))[0].strip() for col in inh.columns]
    inh = inh.groupby(lambda x: x, axis=1).mean()
    return inh


def plot_dasatinib_MS_clustermaps(Full=False, DR=False, AXLs=False):
    """Generate clustermaps of PC9 WT/KO cells treated with an increasing concentration of dasatinib.
    Choose between entire data set, dose response only, or WT up KO down clusters.
    Note that sns.clustermap needs its own figure so these plots will be added manually."""
    X = preprocessing(AXL_Das_DR=True, Vfilter=True, log2T=True, mc_row=False)
    for i in range(X.shape[0]):
        X.iloc[i, 6:11] -= X.iloc[i, 6]
        X.iloc[i, 11:] -= X.iloc[i, 11]

    # axl_ms = preprocessing(AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    data = X.set_index(["Gene", "Position"]).select_dtypes(include=["float64"])
    lim = np.max(abs(data.values)) * 0.5

    g = sns.clustermap(data.T, method="centroid", cmap="bwr", robust=True, vmax=lim, vmin=-lim, figsize=(15, 6), yticklabels=True, col_cluster=True, row_cluster=False)
    if Full:
        plt.savefig("full.svg")
    dict(zip(X.iloc[g.dendrogram_row.reordered_ind[:55], 2].values, X.iloc[g.dendrogram_row.reordered_ind[:67], 3].values))

    data_dr = X.iloc[g.dendrogram_row.reordered_ind[:55], :].set_index(["Gene", "Position"]).select_dtypes(include=["float64"])
    sns.clustermap(data_dr.T, method="centroid", cmap="bwr", robust=True, vmax=lim, vmin=-lim, figsize=(15, 5), xticklabels=True, col_cluster=True, row_cluster=False)
    if DR:
        plt.savefig("DR.svg")

    data_ud = X.iloc[g.dendrogram_row.reordered_ind[413:427], :].set_index(["Gene", "Position"]).select_dtypes(include=["float64"])
    lim = np.max(abs(data_ud.values)) * 0.8
    g_ud = sns.clustermap(data_ud.T, cmap="bwr", method="centroid", robust=True, vmax=lim, vmin=-lim, figsize=(10, 5), xticklabels=True, row_cluster=False)
    if AXLs:
        plt.savefig("AXL.svg")


