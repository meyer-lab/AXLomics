"""
This creates Supplemental Figure 2: Cell Viability and death
"""

import matplotlib
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup, IndividualTimeCourses, import_phenotype_data
from ..distances import PlotRipleysK


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((11, 17), (8, 5))

    # Read in phenotype data
    cv = import_phenotype_data(phenotype="Cell Viability")
    red = import_phenotype_data(phenotype="Cell Death")
    sw = import_phenotype_data(phenotype="Migration")

    # Labels
    lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    mutants = ['PC9', 'KO', 'KIN', 'KD', 'M4', 'M5', 'M7', 'M10', 'M11', 'M15']
    tr1 = ["-UT", "-E", "-A/E"]
    tr2 = ["Untreated", "Erlotinib", "Erl + AF154"]
    # t2 = ["Untreated", "AF154", "Erlotinib", "Erl + AF154"]
    itp = 24
    leg_idx = [0, 10, 20, 30, 40]

    # Time courses
    for i, line in enumerate(lines):
        IndividualTimeCourses(cv, 96, lines, tr1, tr2, "fold-change confluency", TimePointFC=24, TreatmentFC=False, plot=line, ax_=ax[i], ylim=[0.8, 10])
        IndividualTimeCourses(red, 96, lines, tr1, tr2, "fold-change apoptosis (YOYO+)", TimePointFC=24, plot=line, ax_=ax[i + 10], ylim=[0, 13])
        IndividualTimeCourses(sw, 24, lines, tr1, tr2, "RWD %", plot=line, ax_=ax[i + 20])
        PlotRipleysK('48hrs', mutants[i], ['ut', 'e', 'ae'], 6, ax=ax[i + 30], title=line)
        ax[i + 30].set_ylim(0, 40)
        if i not in leg_idx:
            ax[i].legend().remove()

    return f


# AXL expression data
# axl = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/AXLexpression.csv")
# axl = pd.melt(axl, value_vars=["AXL", "GFP"], id_vars="AXL mutants Y—>F", value_name="% Cells", var_name="Signal")
# sns.barplot(data=axl, x="AXL mutants Y—>F", y="% Cells", hue="Signal", ax=ax[1], palette=sns.xkcd_palette(["white", "darkgreen"]), **{"linewidth": 0.5}, **{"edgecolor": "black"})
# ax[1].set_title("Ectopic AXL expression")
# ax[1].legend(prop={'size': 8})