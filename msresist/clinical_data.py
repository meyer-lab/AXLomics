""" All functions related to the CPTAC analysis comparing AXL high versus low tumors """

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gseapy as gp
import statannot
from bioinfokit import visuz
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from .RNAseq import YAP_gene_sets


def preprocess_data(X):
    xR = X.set_index("geneSymbol").select_dtypes(include=float)
    xR_tumor = xR.loc[:, ~xR.columns.str.endswith(".N")]
    xR_tumor = xR_tumor.reindex(sorted(xR_tumor.columns), axis=1)
    xR_nat = xR.loc[:, xR.columns.str.endswith(".N")]
    xR_nat = xR_nat.reindex(sorted(xR_nat.columns), axis=1)

    l1 = list(xR_tumor.columns)
    l2 = [s.split(".N")[0] for s in list(xR_nat.columns)]
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    xR_tumor = xR_tumor.drop(dif, axis=1)
    assert all(xR_tumor.columns.values == np.array(l2)), "Samples don't match"

    l1 = list(xR_tumor.index)
    xR_nat = xR_nat.drop(['SORBS2', 'PKP1', 'IGSF10', 'PANX1', 'PLCXD3', 'MYLK2'], axis=0)
    l2 = list(xR_nat.index)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    xR_tumor = xR_tumor.drop(dif, axis=0)
    assert all(xR_tumor.index == np.array(l2)), "Samples don't match"

    return xR, xR_tumor, xR_nat


def preprocess_phospho(X):
    xR = X.set_index("Gene").select_dtypes(include=float)
    xR_tumor = xR.loc[:, ~xR.columns.str.endswith(".N")]
    xR_tumor = xR_tumor.reindex(sorted(xR_tumor.columns), axis=1)
    xR_nat = xR.loc[:, xR.columns.str.endswith(".N")]
    xR_nat = xR_nat.reindex(sorted(xR_nat.columns), axis=1)

    l1 = list(xR_tumor.columns)
    l2 = [s.split(".N")[0] for s in list(xR_nat.columns)]
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    xR_tumor = xR_tumor.drop(dif, axis=1)
    assert all(xR_tumor.columns.values == np.array(l2)), "Samples don't match"

    l1 = list(xR_tumor.index)
    l2 = list(xR_nat.index)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    xR_tumor = xR_tumor.drop(dif, axis=0)
    assert all(xR_tumor.index == np.array(l2)), "Samples don't match"

    xR.insert(0, "Position", X["Position"].values)
    xR.insert(0, "Sequence", X["Sequence"].values)
    xR = xR.reset_index().set_index(["Gene", "Position", "Sequence"])
    xR_tumor.insert(0, "Position", X["Position"].values)
    xR_tumor.insert(0, "Sequence", X["Sequence"].values)
    xR_tumor = xR_tumor.reset_index().set_index(["Gene", "Position", "Sequence"])
    xR_nat.insert(0, "Position", X["Position"].values)
    xR_nat.insert(0, "Sequence", X["Sequence"].values)
    xR_nat = xR_nat.reset_index().set_index(["Gene", "Position", "Sequence"])

    return xR, xR_tumor, xR_nat

def TumorVsNAT_target(target, xR_nat, xR_tumor):
    out = pd.DataFrame()
    out["Patient_ID"] = list(xR_nat.columns) + list(xR_tumor.columns)
    out["Type"] = ["NAT"] * xR_nat.shape[1] + ["Tumor"] * xR_tumor.shape[1]
    out["Expression"] = list(np.squeeze(xR_nat.loc[target].values)) + list(np.squeeze(xR_tumor.loc[target].values))
    return out

def heatmap_ClusterVsTargets_Corr(targets, omic="Protein", title=False):
    """Plot correlations between clusters and targets"""
    tar = pd.DataFrame()
    for t in targets:
        try:
            if omic == "Protein":
                tar[t] = protC[t]
            elif omic == "RNA":
                tar[t] = rnaC[t]
        except BaseException:
            print(t + " not present in the data set")
            continue
    tar = tar.astype(float)
    tar.index += 1
    g = sns.clustermap(tar.astype(float), figsize=(10, 9), cmap="bwr")
    if title:
        g.fig.suptitle(title)


def find_axl_levels(prot):
    """ Find AXLhi & AXLlow """
    prot_axl = pd.DataFrame(prot.loc["AXL"])
    prot_axl["Levels"] = pd.qcut(prot.loc["AXL"], 3, labels=["Low", "Medium", "High"]).values
    return prot_axl


def volcano(X, prot, gene_label, export=False, multiple_testing=True, lfc_thr=(1, 1), pv_thr=(0.05, 0.05), genenames="deg", show=True):
    l1 = list(prot.loc["AXL"].index)
    l2 = list(X.columns)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    X = X.drop(dif, axis=1)
    assert all(X.columns.values == np.array(l1)), "Samples don't match"

    # Find AXLhi & AXLlow
    prot_axl = find_axl_levels_by_thres(prot)
    axl_hi = X[list(prot_axl[prot_axl["Levels"] == "High"].index)]
    axl_low = X[list(prot_axl[prot_axl["Levels"] == "Low"].index)]

    # Statistical Testing
    pvals = mannwhitneyu(axl_hi, axl_low, axis=1, nan_policy="omit")[1]
    if multiple_testing:
        pvals = multipletests(pvals)[1]

    # Plot
    prot_fc = pd.DataFrame()
    prot_fc["Protein"] = axl_hi.reset_index()[gene_label].values
    prot_fc["Position"] = axl_hi.reset_index()["Position"].values
    prot_fc["Prot_Pos"] = [pn + "-" + pos for pn, pos in list(zip(list(prot_fc["Protein"]), list(prot_fc["Position"])))]
    prot_fc["logFC"] = axl_hi.mean(axis=1).values - axl_low.mean(axis=1).values
    prot_fc["p-values"] = pvals
    prot_fc = prot_fc.sort_values(by="p-values")
    visuz.GeneExpression.volcano(df=prot_fc.dropna(), lfc='logFC', pv='p-values', lfc_thr=lfc_thr, pv_thr=pv_thr, show=show, geneid="Protein", genenames=genenames, figtype="png", dim=(10, 8))

    if export:
        return prot_fc

# enr = volcano(phos, prot, gene_label="Gene", multiple_testing=False, export=True)
    
def make_AXL_categorical_data(X, prot, phospho=False, by_samples=False, by_thres=False):
    if by_thres:
        X_axl = find_axl_levels_by_thres(prot)
    else:
        X_axl = find_axl_levels(prot)
    hi_and_low = X_axl[(X_axl["Levels"] == "High") | (X_axl["Levels"] == "Low")]
    axlHL = X[list(hi_and_low.index)]
    if by_samples:
        axlHL.columns = [l + "_" + c for c, l in zip(list(axlHL.columns.values) , list(hi_and_low["Levels"].values))]
        return axlHL.reindex(sorted(axlHL.columns), axis=1)

    axlHL.columns = hi_and_low["Levels"].values
    axlHL = axlHL.reset_index()
    if phospho:
        axlHL = pd.melt(frame=axlHL, id_vars=["Gene", "Position", "Sequence"], value_vars=axlHL.columns[2:], var_name="AXL", value_name="p-site signal").set_index(["Gene", "Position"])
    else:
        axlHL = pd.melt(frame=axlHL, id_vars=["geneSymbol"], value_vars=axlHL.columns[2:], var_name="AXL", value_name="log(expression)").set_index(["geneSymbol"])
    return axlHL


def find_axl_levels_by_thres(prot, up_thres=0.25, low_thresh=-0.5):
    """ Find AXLhi & AXLlow """
    prot_axl = pd.DataFrame(prot.loc["AXL"])
    axl = prot.loc["AXL"].values
    axl_levels = np.where(axl <= low_thresh, "Low", axl)
    axl_levels = np.where(axl >= up_thres, "High", axl_levels)
    axl_levels = np.where((axl >= low_thresh) & (axl <= up_thres), "Med", axl_levels)
    prot_axl["Levels"] = axl_levels
    return prot_axl


def annotate_pvals(ax, df, x, y, hue=None):
    statannot.add_stat_annotation(
        ax=ax, 
        data=df, 
        x=x, 
        y=y, 
        hue=hue, 
        box_pairs=[("High", "Low")], 
        test="t-test_ind", 
        text_format="star", 
        loc="outside",
        verbose=False)


def plot_psites_byAXLlevels(rows, cols, data, protein, figsize=(15, 12)):
    tot_plots = rows * cols
    fig = plt.figure(1, figsize=figsize)
    pos = range(1, tot_plots + 1)
    for ii, p_site in enumerate(list(set(data.loc[protein].index))):
        dd = data.loc[protein, p_site]
        ax = fig.add_subplot(rows, cols, pos[ii])
        sns.violinplot(data=dd, x="AXL", y="p-site signal", color="white", ax=ax).set_ylabel(p_site + " log(signal)")
        # sns.stripplot(data=dd, x="AXL", y="p-site signal", ax=ax).set_ylabel(p_site + " log(signal)")
        annotate_pvals(ax, dd, "AXL", "p-site signal", hue=None)


def add_feature(X_tumor, meta, features):
    df = X_tumor.T.reset_index().sort_values(by="index")
    m = meta[~meta["Sample.ID"].str.endswith(".N")].sort_values(by="Sample.ID")
    l1 = list(df["index"])
    l2 = list(m["Sample.ID"])
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    m = m.set_index("Sample.ID").drop(dif, axis=0)
    assert np.all(list(df["index"]) == list(m.index)), "Samples don't match"
    for feature in features:
        df.insert(0, feature, list(m[feature]))
    return df


def CPTAC_GSEA_YAP_by_AXLprot(rnaR_tumor, protR_tumor):
    rnaHL = make_AXL_categorical_data(rnaR_tumor, protR_tumor, phospho=False, by_samples=True)

    samples = [c.split("_")[0] for c in list(rnaHL.columns)]

    gs_res = gp.gsea(
                    data=rnaHL.reset_index(),
                    gene_sets=YAP_gene_sets(),
                    cls=samples, 
                    permutation_type='phenotype',
                    permutation_num=100, 
                    method='signal_to_noise',
                    processes=4, 
                    seed= 7,
                    )
    return gs_res


def plot_AXLlevels_byStage(X, pmut, ax):
    p = X.loc[:, ~X.columns.str.endswith(".N")]
    tumor_cf = add_feature(p, pmut, ["EGFR.mutation.status", "Stage"])
    tumor_cf = tumor_cf.set_index("Stage").drop(["1", "3"], axis=0).reset_index()
    sns.stripplot(tumor_cf, x="Stage", y="AXL", hue="EGFR.mutation.status", dodge=True, linewidth=1, ax=ax, order=["1A", "1B", "2B", "3A"])
    sns.boxplot(tumor_cf, x="Stage", y="AXL", hue="EGFR.mutation.status", ax=ax, width=0.5, order=["1A", "1B", "2B", "3A"])
    ax.legend().remove()
    ax.set_ylabel("log(protein abundance)")


def plot_YAPlevels_byStage(phosR_tumor, pmut, ax):
    phos_cf = phosR_tumor.reset_index()
    phos_cf.insert(0, "p-site", [p + " " + z for p, z in zip(list(phos_cf["Gene"]), list(phos_cf["Position"]))])
    phos_cf = phos_cf.drop(["Gene", "Position", "Sequence"], axis=1).set_index("p-site")
    pho_cf = add_feature(phos_cf, pmut, ["EGFR.mutation.status", "Stage"])

    sns.stripplot(pho_cf, x="Stage", y="YAP1 S382-p", dodge=True, linewidth=1, ax=ax, order=["1A", "1B", "2A", "2B", "3A"])
    sns.boxplot(pho_cf, x="Stage", y="YAP1 S382-p", ax=ax, width=0.5, color="white", order=["1A", "1B", "2A", "2B", "3A"]).set_title("YAP S382-p")
    ax.set_ylabel("norm log(p-signal)")
    ax.legend().remove()


def pair_corr(X, p1, p2, ax, title=""):
    corr = pd.DataFrame()
    corr[p1] = X.loc[p1].values
    corr[p2] = X.loc[p2].values
    sns.regplot(data=corr, x=p1, y=p2, ax=ax).set_title(title)