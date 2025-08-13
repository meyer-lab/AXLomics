"""
All functions relaed to GSEA analysis of clusters
"""

from os import listdir
import pandas as pd
import numpy as np
import seaborn as sns
import mygene
import gseapy as gp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from msresist.pca import pca_dfs, plotPCA_scoresORloadings  
from msresist.pre_processing import Linear



path = "/Users/creixell/Desktop/"
gene_sets=["GO_Biological_Process_2021", "WikiPathway_2021_Human", "MSigDB_Oncogenic_Signatures", "KEGG_2021_Human"]


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


def preprocess_AXL_RNAseq_data():
    rna = pd.read_feather("/home/creixell/AXLomics/msresist/data/RNAseq/AXLrna/AXLmutants_RNAseq_merged.feather").iloc[:, 1:]
    idsT = pd.read_csv("/home/creixell/AXLomics/msresist/data/RNAseq/AXLrna/transcripts_to_genes.csv")
    ids = dict(zip(idsT["ENSEMBL1"], idsT["SYMBOL"]))
    rna.insert(0, "Cell Lines", [s[:3] if "M1" in s else s[:2] for s in rna["Cell Line"]])
    rna.insert(1, "Treatment", [s[-1] if s[-1] == "E" in s else s[-2:] for s in rna["Cell Line"]])
    rna = rna.drop("Cell Line", axis=1)
    XIDX = np.any(rna.iloc[:, 2:].values > 10, axis=0)
    rna_f = rna.iloc[:, [True, True] + list(XIDX)]
    rna_f.iloc[:, 2:] = pd.DataFrame(StandardScaler().fit_transform(rna_f.iloc[:, 2:]))
    rna_f.columns = ["Cell Lines", "Treatment"] + [ids[g] if g in ids.keys() else g for g in list(rna_f.columns[2:])]
    rna_f = rna_f.set_index("Cell Lines").T
    rna_f = rna_f.rename(columns={"KD":"KD", "PC9":"WT", "M4": "Y634F", "M5":"Y643F", "M7":"Y698F", "M10":"Y726F", "M11":"Y750F", "M15":"Y821F"}).T.reset_index()
    rna_f = rna_f.set_index(["Cell Lines", "Treatment"]).astype(float).reset_index()
    return rna_f


def import_RNAseq():
    names = listdir("msresist/data/RNAseq")
    tpm_table = pd.DataFrame()
    for name in names:
        data = pd.read_csv("msresist/data/RNAseq/" + name, delimiter="\t")
        condition = name[10:-4]
        data = data.set_index("target_id")
        tpm_table = tpm_table.append(data.iloc[:, -1].rename(condition))
    return tpm_table.T.sort_index(axis=1)


def filter_by_EvEAvar(rna_f, savefig=False, perCut=50):
    rnaE = rna_f[rna_f["Treatment"] == "E"].sort_values(by="Cell Lines")
    rnaEA = rna_f[(rna_f["Treatment"] == "EA") | (rna_f["Cell Lines"] == "KO") & ~(rna_f["Treatment"] == "UT")].sort_values(by="Cell Lines")
    ssd = []
    for ii in range(2, rnaE.shape[1]):
        ssd.append(np.sum(np.square(rnaEA.iloc[:, ii].values - rnaE.iloc[:, ii].values)))
    rna_fEA = rna_f.iloc[:, [True, True] + list(ssd >= np.percentile(ssd, perCut))]
    rna_fEA = rna_fEA[rna_fEA["Treatment"] != "UT"].T
    rna_fEA.columns = [c + "-" + t for c, t in list(zip(rna_fEA.iloc[0, :], rna_fEA.iloc[1, :]))]
    rna_fEA = rna_fEA.iloc[2:, :]
    if savefig:
        rna_fEA.to_csv("msresist/WGCNA/WGCNA_input_Filtered_Abundance_and_EvA.txt", sep="\t")        
    return rna_fEA


def run_standard_gsea_EvsEA(rna_f, gene_sets=gene_sets, outdir=None, out=False):
    st_gsea_d = rna_f[rna_f["Treatment"] != "UT"]
    st_gsea_d.insert(0, "Sample", [c + "_" + t for c, t in list(zip(list(st_gsea_d["Cell Lines"]), list(st_gsea_d["Treatment"])))])
    st_gsea_d = st_gsea_d.drop(["Cell Lines", "Treatment"], axis=1).set_index("Sample").T
    st_gsea_d = st_gsea_d[~st_gsea_d.index.duplicated()]
    m = st_gsea_d.columns.str.endswith("_EA")
    cols = st_gsea_d.columns[m].tolist() + st_gsea_d.columns[~m].tolist()
    st_gsea_d = st_gsea_d[cols]
    samples = [s.split("_")[1] for s in cols]

    gsea_rnaf_wp = gp.gsea(data=st_gsea_d, 
                gene_sets=gene_sets,
                cls= samples, 
                permutation_type='phenotype', # set permutation_type to phenotype if samples >=15
                permutation_num=100, 
                outdir=outdir,
                method='signal_to_noise',
                processes=4, seed= 7,
                format='png')

    if out:
        return gsea_rnaf_wp


def dotplot_std_EvsEA_gsea(rna_f):
    gsea_rnaf_wp = run_standard_gsea_EvsEA(rna_f, gene_sets="WikiPathway_2021_Human", outdir=None, out=True)
    wp_terms = [0, 1, 2, 3, 5, 6, 7, 8, 9, 13, 14]
    wp_selected = gsea_rnaf_wp.res2d.iloc[wp_terms, :]
    wp_selected["Term"] = [t.split("WP")[0] for t in wp_selected["Term"]]
    wp_selected["-log10(p-value)"] = [-np.log10(p + 0.001) for p in wp_selected["NOM p-val"]]
    wp_selected["Hit %"] = [np.round(((int(s.split("/")[0]) / int(s.split("/")[1]))), 2) for s in wp_selected["Tag %"]]
    g = sns.PairGrid(wp_selected, hue="Hit %",
                    x_vars="-log10(p-value)", y_vars=["Term"],
                    height=4, aspect=.75)
    g.map(sns.stripplot, size=10, orient="h", jitter=False,
        palette="flare_r", linewidth=1, edgecolor="w")

    g.set(xlim=(0, 4), xlabel="-log10(p-value)", ylabel="")

    # Make the grid horizontal instead of vertical
    g.axes.flat[0].xaxis.grid(False)
    g.axes.flat[0].yaxis.grid(True)
    g.axes.flat[0].set_title("GSEA (E vs EA)")
    g.axes.flat[0].legend()

    sns.despine(left=True, bottom=True)
    g.axes.flat[0].legend(title='Hit %', bbox_to_anchor=(1.02, 1))


def PCA_EA(rna_f, wUT=True, plot_pca=False, ax=None):
    rna_fEA = filter_by_EvEAvar(rna_f, savefig=False, perCut=50).astype(float)
    rna_fEA = rna_fEA.T.reset_index()
    rna_fEA = rna_fEA.rename(columns={"index": "Cell Lines"})
    rna_fEA.insert(1, "Treatment", [s.split("-")[1] for s in rna_fEA["Cell Lines"]])
    rna_fEA["Cell Lines"] = [s.split("-")[0] for s in rna_fEA["Cell Lines"]]

    if plot_pca:
        plotPCA_scoresORloadings(ax, rna_fEA, 2, ["Cell Lines", "Treatment"], hue_scores="Cell Lines", style_scores="Treatment", legendOut=True, plot="scores")
    n_components = 4
    scores_ind = ["Cell Lines"]

    if wUT:
        dd = rna_fEA
    else:
        dd = rna_f[rna_f["Treatment"] != "UT"]

    pca = PCA(n_components=n_components)
    dScor_EA = pca.fit_transform(dd.select_dtypes(include=["float64"]))
    dLoad_EA = pca.components_
    dScor_EA, dLoad_EA = pca_dfs(dScor_EA, dLoad_EA, dd, n_components, scores_ind)
    scoresEA = dScor_EA.set_index("Cell Lines")
    scoresEA.iloc[:, :] = StandardScaler().fit_transform(dScor_EA.set_index("Cell Lines"))

    dLoad_EA = dLoad_EA.reset_index()
    dLoad_EA = dLoad_EA.rename(columns={"index": "Gene"})
    rankPCS = dLoad_EA.iloc[:, 1:].abs().sum(axis=1)
    dLoad_EA["Sum"] = rankPCS
    PCrank = rankPCS.sort_values(ascending=False).index

    LoadR_pcs = dLoad_EA.iloc[PCrank, :]
    LoadR_pcs = LoadR_pcs[["Gene", "Sum"]].set_index("Gene")
    LoadR_pcs = LoadR_pcs[~LoadR_pcs.index.duplicated()]
    LoadR_pcs.iloc[:, :] = StandardScaler().fit_transform(LoadR_pcs)
    return dLoad_EA

def run_gsea_from_PCA_EA(rna_f, PCA="PC1", plot_pca=False, ax=None, outdir=None):
    if PCA == "PC1":
        dLoad_EA = PCA_EA(rna_f, True, plot_pca, ax)
        dPC1 = dLoad_EA.reset_index().sort_values(by="PC1", ascending=False)[["Gene", "PC1"]]
        dPC1 = dPC1[~dPC1.index.duplicated()].set_index("Gene")
        dPC1.iloc[:, :] = StandardScaler().fit_transform(dPC1)
        rnk = dPC1

    elif PCA == "Full":
        dLoad_F = PCA_EA(rna_f, wUT=False)
        dLoad_F = dLoad_F.reset_index()
        dLoad_F = dLoad_F.rename(columns={"index": "Gene"})
        rankPCS = dLoad_F.iloc[:, 1:].abs().sum(axis=1)
        dLoad_F["Sum"] = rankPCS
        PCrank = rankPCS.sort_values(ascending=False).index

        LoadF_pcs = dLoad_F.iloc[PCrank, :]
        LoadF_pcs = LoadF_pcs[["Gene", "Sum"]].set_index("Gene")
        LoadF_pcs = LoadF_pcs[~LoadF_pcs.index.duplicated()]
        LoadF_pcs.iloc[:, :] = StandardScaler().fit_transform(LoadF_pcs)
        rnk = LoadF_pcs

    res = gp.prerank(rnk=rnk,
                gene_sets=YAP_gene_sets(),
                threads=4,
                min_size=5,
                max_size=1000,
                permutation_num=1000,
                outdir=outdir,
                seed=6,
                verbose=True)

    return res


def YAP_gene_sets():
    """Get YAP related gene sets from MSigDB and WikiPathways."""

    onc_gs = gp.get_library("MSigDB_Oncogenic_Signatures")
    wp = gp.get_library("WikiPathway_2021_Human")

    keys = [
        "CORDENONSI YAP CONSERVED SIGNATURE",
        "YAP1 UP",
        "Mechanoregulation and pathology of YAP/TAZ via Hippo and non-Hippo mechanisms WP4534"
        ]

    yap_gs = {}

    for gs in [onc_gs, wp]:
        for key in gs.keys():
            if key in keys:
                yap_gs[key] = gs[key]

    return yap_gs