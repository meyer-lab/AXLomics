"""
This creates Figure 3B: CPTAC omics data stratified by AXL levels and EGFR mutational status.
"""

import pandas as pd
import seaborn as sns
import matplotlib
from scipy.stats import zscore
from .common import subplotLabel, getSetup, Introduce_Correct_DDMC_labels
from ..pre_processing import preprocessing, filter_NaNpeptides, preprocessing
from ..clinical_data import preprocess_data, preprocess_phospho, make_AXL_categorical_data
from ..clustering import DDMC

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((14, 12), (3, 3), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Import AXL pY MS Signaling Data
    X = preprocessing(AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    d = X.select_dtypes(include=['float64']).T
    i = X.select_dtypes(include=['object'])

    # Fit DDMC
    ddmc = DDMC(i, n_components=5, SeqWeight=2, distance_method="PAM250", random_state=5).fit(d)

    X = Introduce_Correct_DDMC_labels(X, ddmc)
    c123 = X[(X["Cluster"] == 1) | (X["Cluster"] == 2) | (X["Cluster"] == 3)][["Cluster", "Gene", "Position", "PC9 A", "KO A"]]

    # EGFR TKI res signature
    # gseaplot_EGFRres_signature(X, ddmc) gseaplot doesn't have an ax argument
    ax[0].axis("off")

    # Import CPTAC data
    phos = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=2)
    prot = pd.read_csv("msresist/data/MS/CPTAC/CPTAC_LUAD_Protein.csv").drop_duplicates(subset="geneSymbol").set_index("geneSymbol").select_dtypes(include=float).iloc[:, 4:].reset_index()
    rna = pd.read_csv("msresist/data/MS/CPTAC/CPTAC_LUAD_RNAseq.csv").drop_duplicates(subset="geneSymbol")
    _, phosR_tumor, _ = preprocess_phospho(phos)
    _, protR_tumor, _ = preprocess_data(prot)
    _, rnaR_tumor, _ = preprocess_data(rna)
    pmut = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    pmut = pmut[~pmut["Sample.ID"].str.contains("IR")]

    # DDMC Clusters CPTAC phosphorylation data
    pho_df = Phospho_by_AXL_levels(protR_tumor, phosR_tumor, pmut, c123)
    sns.heatmap(pho_df.set_index(["AXL levels", "EGFR mutational status"]).T, cmap="coolwarm", robust=True, ax=ax[1]).set_title("Phosphorylation")

    # DDMC Clusters CPTAC protein data
    prot_df = Protein_by_AXL_levels(protR_tumor, pmut, c123)
    sns.heatmap(prot_df.set_index(["AXL", "EGFR mutational status"]).T, cmap="coolwarm", robust=True, vmin=-1.4, vmax=1.4)

    # DDMC Clusters CPTAC RNA data
    rna_df = RNA_by_AXL_levels(rnaR_tumor, protR_tumor, pmut, c123)
    sns.heatmap(rna_df.set_index(["AXL", "EGFR mutational status"]).T, cmap="coolwarm", robust=True, vmin=-1.4, vmax=1.4)

    return f


def RNA_by_AXL_levels(rnaR_tumor, protR_tumor, pmut, c123):

    genes = filter_out_genes_not_included(c123, "RNA")
    rna_HL = make_AXL_categorical_data(rnaR_tumor, protR_tumor, phospho=False, by_samples=True, by_thres=False)
    df_rna = rna_HL.reset_index().set_index("geneSymbol").loc[genes].reset_index()
    rna_long = pd.melt(df_rna, "geneSymbol", df_rna.columns[1:], "Sample", "Log(protein expression)")
    rna_long.insert(1, "AXL", [s.split("_")[0] for s in rna_long["Sample"]])
    rna_long.insert(2, "Patient ID", [s.split("_")[1] for s in rna_long["Sample"]])
    id_to_egfr = dict(zip(list(pmut["Sample.ID"]), list(pmut["EGFR.mutation.status"])))
    rna_long.insert(3, "EGFR mutational status", [id_to_egfr[s] for s in rna_long["Patient ID"]])
    rna_long["EGFR mutational status"] = rna_long["EGFR mutational status"].replace(0, "EGFR WT")
    rna_long["EGFR mutational status"] = rna_long["EGFR mutational status"].replace(1, "EGFRm")

    ovprna_df = c123.set_index("Gene").loc[genes].reset_index()
    rna_long = rna_long.set_index("geneSymbol")

    rna_c1 = rna_long.loc[list(ovprna_df[ovprna_df["Cluster"] == 1]["Gene"])].reset_index()
    rna_c1["Cluster"] = 1

    rna_c2 = rna_long.loc[list(ovprna_df[ovprna_df["Cluster"] == 2]["Gene"])].reset_index()
    rna_c2["Cluster"] = 2

    rna_c3 = rna_long.loc[list(ovprna_df[ovprna_df["Cluster"] == 3]["Gene"])].reset_index()
    rna_c3["Cluster"] = 3

    rna_anno = pd.concat([rna_c1, rna_c2, rna_c3])
    rna_anno["Omic"] = "RNA"

    rna_df = pd.DataFrame(rna_anno.groupby(["AXL", "EGFR mutational status", "Cluster"])["Log(protein expression)"].median()).reset_index()
    rna_df = rna_df.pivot(index=["AXL", "EGFR mutational status"], columns="Cluster", values="Log(protein expression)").reset_index()
    rna_df.iloc[:, 2:] = zscore(rna_df.iloc[:, 2:], axis=0)

    return rna_df


def Protein_by_AXL_levels(protR_tumor, pmut, c123):
    """
    Create a DataFrame with median z-scored protein expression grouped by AXL levels, EGFR mutational status, and cluster.

    Args:
        protR_tumor (pd.DataFrame): Protein expression data for tumor samples.
        pmut (pd.DataFrame): Patient mutation data.
        c123 (pd.DataFrame): DataFrame with cluster assignments for genes.

    Returns:
        pd.DataFrame: Pivoted DataFrame with median z-scored protein expression grouped by AXL levels, EGFR mutational status, and cluster.
    """
    genes = filter_out_genes_not_included(c123, "Protein")
    protHL = make_AXL_categorical_data(protR_tumor, protR_tumor, phospho=False, by_samples=True, by_thres=False).reset_index().set_index("geneSymbol").loc[genes].reset_index()
    p_long = pd.melt(protHL, "geneSymbol", protHL.columns[1:], "Sample", "Log(protein expression)")
    p_long.insert(1, "AXL", [s.split("_")[0] for s in p_long["Sample"]])
    p_long.insert(2, "Patient ID", [s.split("_")[1] for s in p_long["Sample"]])

    id_to_egfr = dict(zip(list(pmut["Sample.ID"]), list(pmut["EGFR.mutation.status"])))

    p_long.insert(3, "EGFR mutational status", [id_to_egfr[s] for s in p_long["Patient ID"]])
    p_long["EGFR mutational status"] = p_long["EGFR mutational status"].replace(0, "EGFR WT")
    p_long["EGFR mutational status"] = p_long["EGFR mutational status"].replace(1, "EGFRm")

    ovprot_df = c123.set_index("Gene").loc[genes].reset_index()
    p_long = p_long.set_index("geneSymbol")

    prot_c1 = p_long.loc[list(ovprot_df[ovprot_df["Cluster"] == 1]["Gene"])].reset_index()
    prot_c1["Cluster"] = 1

    prot_c2 = p_long.loc[list(ovprot_df[ovprot_df["Cluster"] == 2]["Gene"])].reset_index()
    prot_c2["Cluster"] = 2

    prot_c3 = p_long.loc[list(ovprot_df[ovprot_df["Cluster"] == 3]["Gene"])].reset_index()
    prot_c3["Cluster"] = 3

    prot_anno = pd.concat([prot_c1, prot_c2, prot_c3])
    prot_anno["Omic"] = "Prot"

    prot_df = pd.DataFrame(prot_anno.groupby(["AXL", "EGFR mutational status", "Cluster"])["Log(protein expression)"].median()).reset_index()
    prot_df = prot_df.pivot(index=["AXL", "EGFR mutational status"], columns="Cluster", values="Log(protein expression)").reset_index()
    prot_df.iloc[:, 2:] = zscore(prot_df.iloc[:, 2:], axis=0)

    return prot_df


def Phospho_by_AXL_levels(protR_tumor, phosR_tumor, pmut, c123):
    """
    Create a DataFrame with the phosphorylation data grouped by AXL protein levels and EGFR mutational status

    Args:
        protR_tumor (pd.DataFrame): Protein expression data for tumor samples.
        phosR_tumor (pd.DataFrame): Phosphorylation data for tumor samples.
        pmut (pd.DataFrame): Patient mutation data.
        c123 (pd.DataFrame): DataFrame with cluster assignments for phosphosites.

    Returns:
        pd.DataFrame: Pivoted DataFrame with median z-scored phosphorylation signal grouped by AXL levels, EGFR mutational status, and cluster.
    """
    phosR_tumor = phosR_tumor.reset_index()
    phos_long = pd.melt(phosR_tumor, ["Gene", "Position"], phosR_tumor.columns[3:], "Patient ID", "norm p-signal")

    id_to_egfr = dict(zip(list(pmut["Sample.ID"]), list(pmut["EGFR.mutation.status"])))
    phos_long.insert(3, "EGFR mutational status", [id_to_egfr[s] for s in phos_long["Patient ID"]])

    axl_by_samples = list(make_AXL_categorical_data(protR_tumor, protR_tumor, phospho=False, by_samples=True, by_thres=False).columns)
    id_to_axl = {}
    for c in axl_by_samples:
        s = c.split("_")
        id_to_axl[s[1]] = s[0]
    phos_long.insert(5, "AXL levels", [id_to_axl[s] if s in id_to_axl.keys() else "High" for s in phos_long["Patient ID"]])

    psites = list(zip(list(c123["Gene"]), list(c123["Position"])))
    all_ps = list(zip(list(phos_long["Gene"]), list(phos_long["Position"])))
    ovp = [ps for ps in psites if ps in all_ps]

    df = phos_long.set_index(["Gene", "Position"])
    df["EGFR mutational status"] = df["EGFR mutational status"].replace(0, "EGFR WT")
    df["EGFR mutational status"] = df["EGFR mutational status"].replace(1, "EGFRm")
    phos_long = df.copy().reset_index()
    df = df.loc[ovp]

    cdk1 = phos_long[(phos_long["Gene"] == "CDK1") & (phos_long["Position"] == "Y15-p;T14-p")]
    cdk1["norm p-signal"] *= -1
    df = pd.concat([df.reset_index(), cdk1])
    df["EGFR mutational status"] = df["EGFR mutational status"].replace(0, "EGFR WT")
    df["EGFR mutational status"] = df["EGFR mutational status"].replace(1, "EGFRm")

    ovp_df = c123.set_index(["Gene", "Position"]).loc[ovp].reset_index()
    df = df.set_index(["Gene", "Position"])

    pho_c1 = df.loc[list(zip(list(ovp_df[ovp_df["Cluster"] == 1]["Gene"]),  list(ovp_df[ovp_df["Cluster"] == 1]["Position"])))].reset_index()
    pho_c1["Cluster"] = 1

    pho_c2 = df.loc[list(zip(list(ovp_df[ovp_df["Cluster"] == 2]["Gene"]),  list(ovp_df[ovp_df["Cluster"] == 2]["Position"])))].reset_index()
    pho_c2["Cluster"] = 2

    pho_c3 = df.loc[list(zip(list(ovp_df[ovp_df["Cluster"] == 3]["Gene"]),  list(ovp_df[ovp_df["Cluster"] == 3]["Position"])))].reset_index()
    pho_c3["Cluster"] = 3

    pho_anno = pd.concat([pho_c1, pho_c2, pho_c3])
    pho_anno["Omic"] = "Phospho"

    pho_df = pd.DataFrame(pho_anno.groupby(["AXL levels", "EGFR mutational status", "Cluster"])["norm p-signal"].median()).reset_index()
    pho_df = pho_df.pivot(index=["AXL levels", "EGFR mutational status"], columns="Cluster", values="norm p-signal").reset_index()
    pho_df.iloc[:, 2:] = zscore(pho_df.iloc[:, 2:], axis=0)

    return pho_df


def filter_out_genes_not_included(c123, data):
    if data == "Protein":
        rm_protein_list = ['HIPK1', 'PACS1', 'TCP1', 'SEPT2', 'FYB1', 'CAVIN1', 'KIRREL1', 'CFL1', 'CBLC', 'ICK', 'NME2', 'PSMA2', 'HSPA8', 'GPRC5B', 'ITCH', 'ABL1', 'RBM14', 'PIK3R3', 'SOS1', 'MTOR', 'HGS', 'EPHA5', 'SGMS2', 'SRSF1']
        genes = [*set(list(c123["Gene"]))]
        for _ in range(5):
            for p in genes:
                if p in rm_protein_list:
                    genes.remove(p)

    elif data == "RNA":
        rm_rna_list = ['HIPK1', 'PACS1', 'TCP1', 'SEPT2', 'FYB1', 'CAVIN1', 'KIRREL1', 'CFL1', 'CBLC', 'ICK', 'NME2', 'PSMA2', 'HSPA8', 'GPRC5B', 'ITCH', 'ABL1', 'RBM14', 'PIK3R3', 'SOS1', 'MTOR', 'HGS', 'EPHA5', 'SGMS2', 'SRSF1']
        genes = [*set(list(c123["Gene"]))]
        for _ in range(5):
            for p in genes:
                if p in rm_rna_list:
                    genes.remove(p)

    return genes