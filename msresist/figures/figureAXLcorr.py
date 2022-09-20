"""
Creates plots related to correlating phosphoclusters of LUAD patients and AXL expression
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .common import subplotLabel, getSetup

# rnaC = pd.read_csv("msresist/data/MS/CPTAC/Omics_results/mRNA_Cluster_Correlations.csv").drop("Unnamed: 0", axis=1)
# protC = pd.read_csv("msresist/data/MS/CPTAC/Omics_results/prot_Cluster_Correlations.csv").drop("Unnamed: 0", axis=1)

# # Change column labels to Symbol genes
# rnaC.columns = rnaC.iloc[-1, :]
# rnaC = rnaC.iloc[:-1, :]
# protC.columns = protC.iloc[-1, :]
# protC = protC.iloc[:-1, :]


# def heatmap_ClusterVsTargets_Corr(targets, omic="Protein", title=False):
#     """Plot correlations between clusters and targets"""
#     tar = pd.DataFrame()
#     for t in targets:
#         try:
#             if omic == "Protein":
#                 tar[t] = protC[t]
#             elif omic == "RNA":
#                 tar[t] = rnaC[t]
#         except BaseException:
#             print(t + " not present in the data set")
#             continue
#     tar = tar.astype(float)
#     tar.index += 1
#     g = sns.clustermap(tar.astype(float), figsize=(5, 10), cmap="bwr")
#     if title:
#         g.fig.suptitle(title)


# ddmc_targets = ["AXL", "ABL1", "ABL2", "SRC", "LCK", "LYN", "FRK", "YES1", "HCK", "YAP1", "BLK", "NEK6", "NEK7", "PLK1", "CLK2", "CSNK2A1", "MAPK3", "MAPK1", "BRCA1", "EGFR", "ALK", "INSR"]
# bioid_targets = ["AXL", "AHNAK", "FLNA", "SNX1", "ZFYVE16", "TFRC", "DLG5", "CLINT1", "COPG2", "ACSL3", "CTTN", "WWOX", "CTNND1", "TMPO", "EMD", "EGFR", "E41L2", "PLEC", "HSPA9"]

# heatmap_ClusterVsTargets_Corr(bioid_targets, omic="RNA", title="")
# heatmap_ClusterVsTargets_Corr(bioid_targets, omic="Protein", title="")

# ii = random.sample(range(rnaC.shape[1]), 100)
# targets = list(rnaC.columns[ii])
# heatmap_ClusterVsTargets_Corr(targets, omic="RNA", title="Random genes")


# def count_peptides_perCluster(gene, path, ncl, ax):
#     """Bar plot of peptide recurrences per cluster"""
#     occ = []
#     for clN in range(1, ncl + 1):
#         cl_genes = list(pd.read_csv(path + str(clN) + ".csv")["Gene"])
#         occ.append(cl_genes.count(gene) / len(cl_genes))
#     out = pd.DataFrame()
#     out["Fraction"] = occ
#     out["Cluster"] = np.arange(1, ncl + 1)
#     sns.barplot(data=out, x="Cluster", y="Fraction", color="darkblue", edgecolor=".2", ax=ax)
#     ax.set_title(gene)


# _, ax = plt.subplots(1, 2, figsize=(14, 5))
# path = "msresist/data/cluster_members/CPTACmodel_Members_C"
# count_peptides_perCluster("AHNAK", path, 24, ax[0])
# count_peptides_perCluster("CTNND1", path, 24, ax[1])

# _, ax = plt.subplots(1, 4, figsize=(25, 5))
# path = "msresist/data/cluster_members/AXLmodel_PAM250_Members_C"
# count_peptides_perCluster("AXL", path, 5, ax[0])
# count_peptides_perCluster("GAB1", path, 5, ax[1])
# count_peptides_perCluster("CTNND1", path, 5, ax[2])
# count_peptides_perCluster("MAPK1", path, 5, ax[3])

# _, ax = plt.subplots(1, 4, figsize=(25, 5))
# path = "msresist/data/cluster_members/AXLmodel_PAM250_Members_C"
# count_peptides_perCluster("FLNB", path, 5, ax[0])
# count_peptides_perCluster("FLNA", path, 5, ax[1])
# count_peptides_perCluster("AHNAK", path, 5, ax[2])
# count_peptides_perCluster("TNS1", path, 5, ax[3])


# def makeFigure():
#     """Get a list of the axis objects and create a figure"""
#     # Get list of axis objects
#     ax, f = getSetup((15, 5), (1, 3), multz={0:1})

#     # Add subplot labels
#     subplotLabel(ax)

#     # Set plotting format
#     sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

#     CorrViolin(ax[0])
#     CorrHeatmap(ax[1])

#     return f

path = 'msresist/data/MS/CPTAC/Omics_results/'

def CorrViolin(ax):
    """ Generate grouped violin plot showing correlation distributions per RNA/prot expression levels across clusters. """
    m_df = pd.read_csv(path+'mRNA_Cluster_Correlations.csv')
    p_df = pd.read_csv(path+'prot_Cluster_Correlations.csv')
    m_df = m_df[m_df.index != 30]
    p_df = p_df[p_df.index != 30]
    corr = np.array([], dtype=float)

    for col_m in m_df.columns[1:]:
        corr = np.concatenate((corr, np.asarray(m_df[col_m], dtype=float)))

    for col_p in p_df.columns[1:]:
        corr = np.concatenate((corr, np.asarray(p_df[col_p], dtype=float)))

    mol = ["mRNA" if idx < (len(m_df.columns) - 1) * 30 else "protein" for idx in range((len(m_df.columns) + len(p_df.columns) - 2) * 30)]
    clust = np.arange(0, len(corr)) % 30 + 1
    group1 = pd.DataFrame(np.asarray([corr, clust, mol]).T, columns=['Correlation', 'Cluster', 'Molecule'])
    group1 = group1.astype({'Correlation': 'float64', 'Cluster': 'str', 'Molecule': 'str'})
    sns.violinplot(x="Cluster", y="Correlation", data=group1, ax=ax, hue="Molecule", split=True)


def gen_csvs():
    """ Genereate phosphoproteomic cluster vs prot/RNA expression correlation. """
    clust_data = pd.read_csv(path + 'CPTAC_phosphoclusters.csv')
    prot_data = pd.read_csv(path + 'CPTAC_LUAD_Protein.csv').set_index("id")
    mRNA_data = pd.read_csv(path + 'CPTAC_LUAD_RNAseq.csv').set_index("gene_id")

    # clust import
    clust_data.index = clust_data['Patient_ID']
    clust_data.drop(clust_data.columns[0:2], axis=1, inplace=True)
    clust_data = clust_data.transpose()

    # construct predictor+response
    predictor = mRNA_data[mRNA_data.columns[6:]].T
    response = clust_data[mRNA_data.columns[6:]].T
    predictor_prot = prot_data[prot_data.columns[17:]].T
    response_prot = clust_data[prot_data.columns[17:]].T

    # get corrs
    corr = []
    for idx, gene in enumerate(predictor.columns):
        corr.append([])
        for clust in response.columns:
            cov = np.cov(list(predictor[gene].dropna()), list(response[clust].loc[predictor[gene].dropna().index]))
            corr[idx].append(cov[0][1] / ((cov[0][0] * cov[1][1])**.5))
    corr = np.asarray(corr).T

    prot = []
    for idx, gene in enumerate(predictor_prot.columns):
        prot.append([])
        for clust in response_prot.columns:
            cov = np.cov(list(predictor_prot[gene].dropna()), list(response_prot[clust].loc[predictor_prot[gene].dropna().index]))
            prot[idx].append(cov[0][1] / ((cov[0][0] * cov[1][1])**.5))
    prot = np.asarray(prot).T

    corr_out = pd.DataFrame(corr, columns=predictor.columns, index=response.columns)
    corr_out = corr_out.append(mRNA_data['geneSymbol'])
    corr_out.to_csv('mRNA_Cluster_Correlations.csv')
    prot_out = pd.DataFrame(prot, columns=predictor_prot.columns, index=response_prot.columns)
    prot_out = prot_out.append(prot_data['geneSymbol'])
    prot_out.to_csv('prot_Cluster_Correlations.csv')


def CorrHeatmap(ax):
    """ Correlation heatmap """
    clust_data = pd.read_csv(path+'CPTAC_phosphoclusters.csv')

    #import
    clust_data.index = clust_data['Patient_ID']
    clust_data.drop(clust_data.columns[0:2], axis=1, inplace=True)
    corr = clust_data.corr(method = 'pearson')
    corr.index = list(np.arange(30) + 1)
    corr.columns = list(np.arange(30) + 1)

    sns.heatmap(data=corr, vmin = -1, vmax = 1, cmap="RdBu", ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_ylabel('Cluster')
    ax.set_xlabel('Cluster')