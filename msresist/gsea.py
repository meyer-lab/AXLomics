"""
All functions relaed to GSEA analysis of clusters
"""

import numpy as np 
import mygene
from msresist.pre_processing import Linear


path = "/Users/creixell/Desktop/"


def translate_geneIDs(X, labels, toID="entrezgene", export=False, outpath="GSEA_Input.csv"):
    """ Generate GSEA clusterProfiler input data. Translate gene accessions.
    In this case to ENTREZID by default. """
    X["Clusters"] = labels
    X.index = list(range(X.shape[0]))
    mg = mygene.MyGeneInfo()
    gg = mg.querymany(list(X["Gene"]), scopes="symbol", fields=toID, species="human", returnall=False, as_dataframe=True)
    aa = dict(zip(list(gg.index), list(gg[toID])))
    for ii in range(X.shape[0]):
        X.loc[ii, "Gene"] = aa[X.loc[ii, "Gene"]]
    if export:
        X[["Gene", "Clusters"]].to_csv(outpath)
    return X[["Gene", "Clusters"]]


def cytoscape_input(ddmc, X):
    """Store cluster members wih specific formatting to analyze in cytoscape."""
    xL = Linear(X, X.columns[7:])
    xL.insert(7, "WTvsKO", np.log2((xL["PC9 A"] / xL["KO A"]).values))
    xL["Position"] = [s.replace("-p", "").replace(";", "/") for s in xL["Position"]]
    ddmc.store_cluster_members(xL, "DDMC_PAM250_5Cl_W2_RS5_C", ["UniprotAcc", "Gene", "Sequence", "Position", "WTvsKO"])
