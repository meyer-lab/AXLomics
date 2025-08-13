import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from ..pre_processing import preprocessing
from ..clustering import DDMC
import gseapy as gp
from IPython.display import display
from msresist.figures.common import Introduce_Correct_DDMC_labels, plotMotifs

matplotlib.rcParams['font.sans-serif'] = "Arial"
sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

def plot_AXLspecificity_heatmap(ax=None, type="Heatmap"):
    """Plot AXL specificity heatmap or logo."""

    # Import OPLS-like AXL specificity screen results
    btn_logo = pd.read_csv("/home/creixell/AXLomics/msresist/data/AXL_screen/BTN_OPLS_mean.csv")
    btn_opls = btn_logo.set_index("Unnamed: 0")
    btn_opls = btn_opls.replace(np.nan, 0)

    if type == "Heatmap":
        g = sns.clustermap(
            btn_opls.drop("Y"),
            cmap="bwr",
            robust=True,
            col_cluster=False,
            row_cluster=True,
            figsize=(6, 6),
            cbar_kws={"label": "NES"},
            linewidths=0.003,
            linecolor="black"
        )

        g.figure.suptitle('AXL kinase specificity', y=0.84, fontsize=12)
        g.ax_heatmap.set_xticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        g.ax_heatmap.set_ylabel('');

    elif type == "Logo":
        plotMotifs([btn_opls], [ax], titles=["AXL BTN"], yaxis=[-5.5, 3.5])


def plotViolin_HuTyr_vs_AXLspikedIn():
    """AXL specificity screen results comparing libraries: 
    Annotated human tyrosine sites versus sites measured in this study."""

    # Import AXL specificity screen results
    pTyr = pd.read_csv("/home/creixell/AXLomics/msresist/data/AXL_screen/final_filtered.csv")
    pTyr["Library"] = ["AXL spiked-in" if v != "0" else "pTyr-Var" for v in pTyr["marc_check"]]

    pTyr["Library"] = ["AXL spiked-in" if v != "0" else "pTyr-Var" for v in pTyr["marc_check"]]
    pTyr["NES"] = pTyr[["enrich_b_r1", "enrich_b_r2"]].mean(axis=1)

    _, ax = plt.subplots(1, 1, figsize=(4, 5))
    # Create custom color palette for the violin plot
    custom_palette = {"AXL spiked-in": "peru", "pTyr-Var": "white"}

    # Create the violin plot
    sns.violinplot(
        x="Library", 
        y="NES", 
        data=pTyr, 
        inner="box", 
        palette=custom_palette,
        edgecolor="black",
        linewidth=3,
        ax=ax)

    # Extract NES values for each library
    axl_spiked = pTyr[pTyr["Library"] == "AXL spiked-in"]["NES"]
    ptyr_var = pTyr[pTyr["Library"] == "pTyr-Var"]["NES"]

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(axl_spiked, ptyr_var, equal_var=False)

    # Print results
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.8f}")
    print(f"Statistically significant difference: {p_value < 0.05}")

    # Calculate summary statistics
    axl_mean = axl_spiked.mean()
    ptyr_mean = ptyr_var.mean()
    axl_std = axl_spiked.std()
    ptyr_std = ptyr_var.std()

    print(f"\nAXL spiked-in: mean={axl_mean:.4f}, std={axl_std:.4f}, n={len(axl_spiked)}")
    print(f"pTyr-Var: mean={ptyr_mean:.4f}, std={ptyr_std:.4f}, n={len(ptyr_var)}")

    # Add statistical annotation
    x1, x2 = 0, 1
    y = max(pTyr["NES"].max(), axl_spiked.max(), ptyr_var.max()) + 0.5
    plt.plot([x1, x1, x2, x2], [y, y+0.1, y+0.1, y], lw=1.5, c='black')
    plt.text((x1+x2)*.5, y+0.2, f"p = {p_value:.2e}", ha='center', fontsize=10)

    plt.title("")
    plt.tight_layout()

    # Add statistical annotation
    x1, x2 = 0, 1
    y = max(pTyr["NES"].max(), axl_spiked.max(), ptyr_var.max()) + 0.5
    plt.plot([x1, x1, x2, x2], [y, y+0.1, y+0.1, y], lw=1.5, c='black')
    plt.text((x1+x2)*.5, y+0.2, f"p = {p_value:.2e}", ha='center', fontsize=10)


def plot_Specificity_Enrichment_by_DDMC_cluster():
    """Plot enrichment of AXL specificity by DDMC cluster."""

    X = label_AXLspecificity_peptides_by_DDMC()

    gene_sets = {}
    for n in np.arange(1, 6):
        gene_sets[str(n)] = list(X[X["Cluster"] == n]["Gene"].values)

    rnk = X[["Gene", "NES"]].sort_values(by="NES", ascending=False).dropna()
    rnk.columns = range(rnk.shape[1])
    rnk = rnk.drop_duplicates(subset=0, keep="first").set_index(0)

    pre_res = gp.prerank(rnk=rnk,
                    gene_sets=gene_sets,
                    threads=4,
                    min_size=5,
                    max_size=1000,
                    permutation_num=1000, # reduce number to speed up testing
                    outdir=None, # don't write to disk
                    seed=6,
                    verbose=True, # see what's going on behind the scenes
        )

    pre_res.plot(terms=["2", "3", "4"],
                   #legend_kws={'loc': (1.2, 0)}, # set the legend loc
                   show_ranking=True, # whether to show the second yaxis
                   figsize=(3,4)
                  )

    display(pre_res.res2d)

def label_AXLspecificity_peptides_by_DDMC():
    """Label AXL specificity peptides by DDMC cluster."""

    # Import AXL specificity screen results
    X = pd.read_csv("/home/creixell/AXLomics/msresist/data/AXL_screen/AXL_ms_data.csv")

    # Import siganling data
    X_ms = preprocessing(AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    d = X_ms.select_dtypes(include=['float64']).T
    i = X_ms.select_dtypes(include=['object'])

    # Fit DDMC
    ddmc = DDMC(i, n_components=5, SeqWeight=2, distance_method="PAM250", random_state=5).fit(d)
    X_ms = Introduce_Correct_DDMC_labels(X_ms, ddmc)

    # Filter X and X_ms to include only peptides with "y" in the 6th position of Sequence
    X = X.iloc[[seq[5] == "y" for seq in X["Sequence"]], :]
    X = X[X["Gene"] != "2-Sep"]
    X = X[X["Gene"] != "SEPT2"]

    X_ms = X_ms.iloc[[seq[5] == "y" for seq in X_ms["Sequence"]], :]
    X_ms = X_ms[X_ms["Gene"] != "SEPT2"].drop_duplicates(["Gene", "Position"])

    cluster_labels = [X_ms.set_index(["Gene", "Position"]).loc[g, p]["Cluster"] for g, p in list(zip(list(X["Gene"]), list(X["Position"])))]
    X.insert(5, "Cluster", cluster_labels)
    X.insert(6, "NES", X[['enrich_net', 'enrich_net_btn']].mean(axis=1))

    return X


def plot_AXL_Screen_Results():
    """Plot AXL screen results with enrichment scores and highlight PTK2 Y861-p."""

    # Import and rank AXL specificity screen results by NES

    X = label_AXLspecificity_peptides_by_DDMC()
    X.sort_values(by="NES", ascending=True, inplace=True)
    X.insert(0, "Rank", range(1, X.shape[0]+1))

    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    # Create a new column to indicate which point to highlight for PTK2 at position Y861-p
    X["highlight_PTK2_Y861"] = [(gene == "PTK2" and "Y861-p" in pos) for gene, pos in zip(X["Gene"], X["Position"])]

    # Create a custom palette where PTK2 at Y861-p is highlighted in red
    palette = {
        True: "red",
        False: "skyblue" 
    }

    # Create the scatter plot with different markers and sizes based on FAK1 status
    sns.scatterplot(
        data=X, 
        x="Rank", 
        y="NES", 
        hue="highlight_PTK2_Y861",
        palette=palette,
        sizes=(20, 100),
        legend="brief",
        ax=ax
    )

    # Add labels to PTK2 Y861-p point
    for idx, row in X[X["highlight_PTK2_Y861"]].iterrows():
        ax.annotate(
            f"PTK2 Y861-p\nNES: {row['NES']:.2f}",
            (row["Rank"], row["NES"]),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="black", lw=1)
        )

    # Set axis labels and title
    ax.set_xlabel("Rank")
    ax.set_ylabel("NES")
    ax.set_title("Enrichment Score vs Rank")
    ax.set_xticklabels([500, 400, 300, 200, 100, 1])

    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Other peptides", "PTK2 Y861-p", "Other", "FAK1"], 
            title="", loc="best")


    plt.tight_layout()
