"""
This creates Figure 3A: GSEA AXL downstream signaling
"""

import pandas as pd
import seaborn as sns
import matplotlib
import gseapy as gp
from .common import subplotLabel, getSetup, Introduce_Correct_DDMC_labels
from ..clustering import DDMC 
from ..pre_processing import preprocessing

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((5, 5), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Load data
    X = preprocessing(AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    d = X.select_dtypes(include=['float64']).T
    i = X.select_dtypes(include=['object'])

    # Fit DDMC
    ddmc = DDMC(i, n_components=5, SeqWeight=2, distance_method="PAM250", random_state=5).fit(d)

    # GSEA
    ax.axis("off")
    # gseaplot_EGFRres_signature(X, ddmc)

    return f


def gseaplot_EGFRres_signature(X, ddmc):
    """ Calculate EGFR TKI resistance enrichment signature using WT vs KO fold change of MS data """
    X = Introduce_Correct_DDMC_labels(X, ddmc)
    cl123 = X[(X["Cluster"] == 3) | (X["Cluster"] == 2) | (X["Cluster"] == 1)]
    cl123 = X.drop_duplicates(subset="Gene", keep="first")

    cl123_fc = pd.DataFrame()
    cl123_fc["Gene"] = cl123["Gene"]
    cl123_fc["log2(FC)"] = (cl123["PC9 A"] - cl123["KO A"])
    cl123_fc["log2(FC)"] = (cl123_fc["log2(FC)"] - cl123_fc["log2(FC)"].mean()) / cl123_fc["log2(FC)"].std()
    rnk = cl123_fc.sort_values(by="log2(FC)", ascending=False).set_index("Gene")

    pre_res = gp.prerank(rnk=rnk,
                     gene_sets='WikiPathway_2021_Human',
                     threads=4,
                     min_size=5,
                     max_size=1000,
                     permutation_num=1000, # reduce number to speed up testing
                     seed=6,
                     verbose=True, # see what's going on behind the scenes
                    )

    term = "EGFR Tyrosine Kinase Inhibitor Resistance WP4806"
    gp.gseaplot(rank_metric=pre_res.ranking,
         term=term,
         figsize=(6, 6.5),
         **pre_res.results[term])
