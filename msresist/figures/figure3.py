"""
This creates Figure 3: ABL/SFK/YAP experimental validations
"""

import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
from .common import subplotLabel, getSetup, TimePointFoldChange, plot_IdSites
from msresist.pre_processing import preprocessing


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 14), (4, 4), multz={0: 1, 8: 2, 13: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Import Dasatinib DR MS data
    X = preprocessing(AXL_Das_DR=True, Vfilter=True, log2T=True, mc_row=False)
    for i in range(X.shape[0]):
        X.iloc[i, 6:11] -= X.iloc[i, 6]
        X.iloc[i, 11:] -= X.iloc[i, 11]

    # Das DR time point
    plot_InhDR_timepoint(ax[0], "Dasatinib", itp=24)

    # Luminex p-ASY Das DR
    plot_pAblSrcYap(ax[1:4])

    # Das DR Mass Spec Dose response cluster
    ax[4].axis("off")

    # AXL Mass Spec Cluster 4 enrichment of peptides in Das DR cluster
    plotHyerGeomTestDasDRGenes(ax[5])
    ax[5].set_ylim(0, 0.7)

    # Selected peptides within Dasatinib DR Cluster
    abl_sfk = {'LYN': 'Y397-p', 'YES1': 'Y223-p', 'ABL1': 'Y393-p', 'FRK': 'Y497-p', 'LCK': 'Y394-p'}
    plot_IdSites(ax[6], X, abl_sfk, "ABL&SFK", rn=False, ylim=False, xlabels=list(X.columns[6:]))

    # Das DR Mass Spec WT/KO dif
    ax[7].axis("off")

    return f


def plot_YAPinhibitorTimeLapse(ax, X, ylim=False):
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


def transform_DRviability(data, inhibitor, units, itp):
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
    for i, mat in enumerate(data):
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
    hg["p_value"] = [0.527, 0.003, 0.002, 0.404, 0.0557]
    sns.barplot(data=hg, x="Cluster", y="p_value", ax=ax, color="darkblue", **{"linewidth": 1}, **{"edgecolor": "black"})
    ax.set_title("Enrichment of Das-responsive Peptides")
    ax.set_ylim((0, 0.55))
    for index, row in hg.iterrows():
        ax.text(row.Cluster - 1, row.p_value + 0.01, round(row.p_value, 3), color='black', ha="center")


def GenerateHyperGeomTestParameters(A, X, dasG, cluster):
    """Generate parameters to calculate p-value for under- or over-enrichment based on CDF of the hypergeometric distribution."""
    N = list(set(A["Gene"]).intersection(set(X["Gene"])))
    cl = A[A["Cluster"] == cluster]
    M = list(set(cl["Gene"]).intersection(set(N)))
    s = list(set(dasG).intersection(set(N)))
    k = list(set(s).intersection(set(M)))
    return (len(k), len(s), len(M), len(N))


def plot_InhDR_timepoint(ax, inhibitor, itp=24):
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
    data = transform_DRviability(inh, inhibitor, units, itp)
    tp = data[data["Elapsed"] == time]
    sns.lineplot(data=tp, x="Inh_concentration", y="Fold-change confluency", hue="Lines", style="Condition", ci=68, ax=ax)
    ax.set_xlabel("[" + inhibitor + "]")


def merge_TRs(filename, nTRs):
    """Merge technical replicates of an experiment"""
    path = "msresist/data/Validations/CellGrowth/"
    inh = pd.read_csv(path + filename)
    for i in range(1, nTRs):
        inh.columns = [col.split("." + str(i))[0].strip() for col in inh.columns]
    inh = inh.groupby(lambda x: x, axis=1).mean()
    return inh


def transform_siRNA(data, itp):
    new = fold_change_acrossBRs(data, itp)
    c = pd.concat(new, axis=0)
    c = pd.melt(c, id_vars="Elapsed", value_vars=c.columns[1:], var_name="Lines", value_name="Fold-change confluency")
    c["Treatment"] = [s.split(" ")[1] for s in c["Lines"]]
    c["Construct"] = [s.split("-")[1] if "-" in s else "Non-T" for s in c["Lines"]]
    c["Lines"] = [s.split(" ")[0] for s in c["Lines"]]
    c = c[c["Elapsed"] >= itp]
    return c


def plot_siRNA_TimeLapse(ax, target, time=96, itp=24, trs=2):
    d = merge_TRs("NEK_siRNA_BR1.csv", trs)
    if target == "NEK6":
        d = [d.loc[:, ~d.columns.str.contains("7")]]
    elif target == "NEK7":
        d = [d.loc[:, ~d.columns.str.contains("6")]]
    d = transform_siRNA(d, itp)
    wt = d[d["Lines"] == "WT"]
    ko = d[d["Lines"] == "KO"]
    sns.lineplot(x="Elapsed", y="Fold-change confluency", data=wt, hue="Treatment", style="Construct", ax=ax[0]).set_title("PC9 WT-si" + target)
    sns.lineplot(x="Elapsed", y="Fold-change confluency", data=ko, hue="Treatment", style="Construct", ax=ax[1]).set_title("PC9 AXL KO-si" + target)


def plot_pAblSrcYap(ax):
    """Plot luminex p-signal of p-ABL, p-SRC, and p-YAP 127."""
    mfi_AS = pd.read_csv("msresist/data/Validations/Luminex/DasatinibDR_newMEK_lysisbuffer.csv")
    mfi_AS = pd.melt(mfi_AS, id_vars=["Treatment", "Line", "Lysis_Buffer"], value_vars=["p-MEK", "p-YAP", "p-ABL", "p-SRC"], var_name="Protein", value_name="p-Signal")
    mfi_YAP = pd.read_csv("msresist/data/Validations/Luminex/DasatinibDR_pYAP127_check.csv")
    mfi_YAP = pd.melt(mfi_YAP, id_vars=["Treatment", "Line", "Lysis_Buffer"], value_vars=["p-MEK", "p-YAP(S127)"], var_name="Protein", value_name="p-Signal")
    abl = mfi_AS[(mfi_AS["Protein"] == "p-ABL") & (mfi_AS["Lysis_Buffer"] == "RIPA")].iloc[:-1, :]
    abl["Treatment"] = [t.replace("A", "(A)") for t in abl["Treatment"]]
    abl["Treatment"][6:] = abl["Treatment"][1:6]
    src = mfi_AS[(mfi_AS["Protein"] == "p-SRC") & (mfi_AS["Lysis_Buffer"] == "NP-40")].iloc[:-2, :]
    src["Treatment"] = [t.replace("A", "(A)") for t in src["Treatment"]]
    src["Treatment"][6:] = src["Treatment"][1:6]
    yap = mfi_YAP[(mfi_YAP["Protein"] == "p-YAP(S127)") & (mfi_YAP["Lysis_Buffer"] == "RIPA")].iloc[:-1, :]
    yap["Treatment"] = [t.replace("A", "(A)") for t in yap["Treatment"]]
    yap["Treatment"][6:] = yap["Treatment"][1:6]

    sns.barplot(data=abl, x="Treatment", y="p-Signal", hue="Line", ax=ax[0])
    ax[0].set_title("p-ABL")
    ax[0].set_xticklabels(abl["Treatment"][:6], rotation=90)
    sns.barplot(data=src, x="Treatment", y="p-Signal", hue="Line", ax=ax[1])
    ax[1].set_title("p-SRC")
    ax[1].set_xticklabels(src["Treatment"][:6], rotation=90)
    sns.barplot(data=yap, x="Treatment", y="p-Signal", hue="Line", ax=ax[2])
    ax[2].set_title("p-YAP S127")
    ax[2].set_xticklabels(yap["Treatment"][:6], rotation=90)


def plot_pAblSrcYap2(ax, line="WT"):
    """Plot luminex p-signal of p-ABL, p-SRC, and p-YAP 127."""
    mfi_AS = pd.read_csv("msresist/data/Validations/Luminex/ABL_SRC_YAP_DasDR.csv")
    mfi_AS = pd.melt(mfi_AS, id_vars=["Treatment", "Line"], value_vars=["p-YAP S127", "p-SRC Y416", "p-ABL Y245"], var_name="Protein", value_name="p-Signal")
    mfi_AS = mfi_AS[mfi_AS["Line"] == line]

    abl = mfi_AS[(mfi_AS["Protein"] == "p-ABL Y245")]
    src = mfi_AS[(mfi_AS["Protein"] == "p-SRC Y416")]
    yap = mfi_AS[(mfi_AS["Protein"] == "p-YAP S127")]

    a = sns.barplot(data=abl, x="Treatment", y="p-Signal", ax=ax[0])
    a.set_title("p-ABL Y245")
    a.set_xticklabels(a.get_xticklabels(), rotation=90)
    s = sns.barplot(data=src, x="Treatment", y="p-Signal", ax=ax[1])
    s.set_title("p-SRC Y416")
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    y = sns.barplot(data=yap, x="Treatment", y="p-Signal", ax=ax[2])
    y.set_title("p-YAP S127")
    y.set_xticklabels(y.get_xticklabels(), rotation=90)


def plot_pAblSrcYap3(ax, line="WT"):
    """Plot luminex p-signal of p-ABL, p-SRC, and p-YAP 127."""
    mfi_AS = pd.read_csv("msresist/data/Validations/Luminex/20210730 ABL_SRC_YAP_DasDR 2_20210730_182250.csv")
    mfi_AS = pd.melt(mfi_AS, id_vars=["Treatment", "Line", "Experiment"], value_vars=["p-YAP S127", "p-SRC Y416", "p-ABL Y245"], var_name="Protein", value_name="p-Signal")
    mfi_AS = mfi_AS[mfi_AS["Line"] == line]

    abl = mfi_AS[(mfi_AS["Protein"] == "p-ABL Y245")]
    abl["Treatment"] = [t.replace("A", "(A)") for t in abl["Treatment"]]
    src = mfi_AS[(mfi_AS["Protein"] == "p-SRC Y416")]
    src["Treatment"] = [t.replace("A", "(A)") for t in src["Treatment"]]
    yap = mfi_AS[(mfi_AS["Protein"] == "p-YAP S127")]
    yap["Treatment"] = [t.replace("A", "(A)") for t in yap["Treatment"]]

    a = sns.barplot(data=abl, x="Treatment", y="p-Signal", ax=ax[0], hue="Experiment")
    a.set_title("p-ABL Y245")
    a.set_xticklabels(a.get_xticklabels(), rotation=90)
    s = sns.barplot(data=src, x="Treatment", y="p-Signal", ax=ax[1], hue="Experiment")
    s.set_title("p-SRC Y416")
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    y = sns.barplot(data=yap, x="Treatment", y="p-Signal", ax=ax[2], hue="Experiment")
    y.set_title("p-YAP S127")
    y.set_xticklabels(y.get_xticklabels(), rotation=90)


def plot_dasatinib_MS_clustermaps():
    """Generate clustermaps of PC9 WT/KO cells treated with an increasing concentration of dasatinib.
    Choose between entire data set, dose response only, or WT up KO down clusters.
    Note that sns.clustermap needs its own figure so these plots will be added manually."""
    X = preprocessing(AXL_Das_DR=True, Vfilter=True, log2T=True, mc_row=False)
    for i in range(X.shape[0]):
        X.iloc[i, 6:11] -= X.iloc[i, 6]
        X.iloc[i, 11:] -= X.iloc[i, 11]

    axl_ms = preprocessing(AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    data = X.set_index(["Gene", "Position"]).select_dtypes(include=["float64"])
    lim = np.max(abs(data.values)) * 0.5

    g = sns.clustermap(data, method="centroid", cmap="bwr", robust=True, vmax=lim, vmin=-lim, figsize=(10, 10), xticklabels=True, col_cluster=False)
    dict(zip(X.iloc[g.dendrogram_row.reordered_ind[:55], 2].values, X.iloc[g.dendrogram_row.reordered_ind[:67], 3].values))

    data_dr = X.iloc[g.dendrogram_row.reordered_ind[:55], :].set_index(["Gene", "Position"]).select_dtypes(include=["float64"])
    sns.clustermap(data_dr.T, method="centroid", cmap="bwr", robust=True, vmax=lim, vmin=-lim, figsize=(15, 5), xticklabels=True, col_cluster=True, row_cluster=False)

    data_ud = X.iloc[g.dendrogram_row.reordered_ind[413:427], :].set_index(["Gene", "Position"]).select_dtypes(include=["float64"])
    lim = np.max(abs(data_ud.values)) * 0.8
    g_ud = sns.clustermap(data_ud.T, cmap="bwr", method="centroid", robust=True, vmax=lim, vmin=-lim, figsize=(10, 5), xticklabels=True, row_cluster=False)
