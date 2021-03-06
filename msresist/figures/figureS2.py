"""
This creates Supplemental Figure 2: Cell Viability and death
"""

import matplotlib
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup, IndividualTimeCourses, import_phenotype_data, barplot_UtErlAF154


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (4, 6), multz={0: 1, 12: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Read in viability and apoptosis data
    cv = import_phenotype_data(phenotype="Cell Viability")
    red = import_phenotype_data(phenotype="Cell Death")

    # Labels
    lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    tr1 = ["-UT", "-E", "-A/E"]
    tr2 = ["Untreated", "Erlotinib", "Erl + AF154"]
    colors = ["white", "windows blue", "scarlet"]
    itp = 24

    # Bar plots
    barplot_UtErlAF154(ax[0], lines, cv, 96, tr1, tr2, "fold-change confluency", "Cell Viability (t=96h)", colors, TreatmentFC="-E", TimePointFC=itp, loc='upper right')
    barplot_UtErlAF154(ax[11], lines, red, 72, tr1, tr2, "fold-change YOYO+ cells", "Cell Death (t=72h)", TreatmentFC="-E", colors=colors, TimePointFC=itp, loc='lower center')

    # Time courses
    for i, line in enumerate(lines):
        IndividualTimeCourses(cv, 96, lines, tr1, tr2, "fold-change confluency", TimePointFC=24, TreatmentFC="-E", plot=line, ax_=ax[i + 1], ylim=[0.8, 3.5])
        IndividualTimeCourses(red, 96, lines, tr1, tr2, "fold-change apoptosis (YOYO+)", TimePointFC=itp, plot=line, ax_=ax[i + 12], ylim=[0, 13])

    return f


# AXL expression data
# axl = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/AXLexpression.csv")
# axl = pd.melt(axl, value_vars=["AXL", "GFP"], id_vars="AXL mutants Y—>F", value_name="% Cells", var_name="Signal")
# sns.barplot(data=axl, x="AXL mutants Y—>F", y="% Cells", hue="Signal", ax=ax[1], palette=sns.xkcd_palette(["white", "darkgreen"]), **{"linewidth": 0.5}, **{"edgecolor": "black"})
# ax[1].set_title("Ectopic AXL expression")
# ax[1].legend(prop={'size': 8})