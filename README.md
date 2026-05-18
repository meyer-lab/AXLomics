# Dissecting AXL-mediated signaling and resistance

The project investigates how the AXL receptor tyrosine kinase drives resistance to EGFR-targeted therapies in lung cancer. It employs a systems biology approach, utilizing **Dual Data-Motif Clustering (DDMC)** to integrate phosphoproteomic abundance and sequence information, and **Partial Least Squares Regression (PLSR)** to link signaling clusters to phenotypic outcomes (survival, migration, and spatial clustering).

## Installation

This project uses `uv` for dependency management, and `git-lfs` for large data files.

```bash
# 1. Install git-lfs if not already installed (https://git-lfs.com)
git lfs install

# 2. Clone the repository and pull LFS files
git clone https://github.com/meyer-lab/AXLomics.git
cd AXLomics
git lfs pull

# 3. Install dependencies
uv sync
```

## Usage

### Reproducing Figures

Run any of the Jupyter notebooks (`Figure1.ipynb`–`FigureS2_motifs.ipynb`) directly in VS Code using the `.venv` kernel created by `uv sync`, or execute all at once from the repo root:

```bash
make notebooks
```

Generated figure files (SVG, PDF, PNG) are written to `output/` and computed results to `msresist/results/`.

### External Datasets

**Figure3C-J** requires the Maynard et al. single-cell RNA-seq dataset (PMID: [32822576](https://pubmed.ncbi.nlm.nih.gov/32822576/), DOI: [10.1016/j.cell.2020.07.017](https://doi.org/10.1016/j.cell.2020.07.017)). Generate an h5ad file using the study's deposited raw data and place it at:

```
msresist/data/Maynard/8091e3d90a9045a181b2fc11000c0dd9_PMID32822576.h5ad
```

The accompanying `cancer_cell_annotation.csv` (inferCNV-derived cancer cell annotations) is included in the repository.

### Running Analysis

To run the supplemental analysis regarding AXL receptor dosage bias:

```bash
uv run python AXLdosage_bias/scripts/run_analysis.py
```

### Testing and Linting

```bash
make test  # Run unit tests
make lint  # Run Ruff linter
```

## Repository Structure

- `msresist/`: Core Python package containing:
    - `clustering.py`: Implementation of DDMC (Gaussian Mixture Model variant).
    - `pca.py`, `plsr.py`: Dimensionality reduction and regression modeling.
    - `distances.py`: Implementation of Ripley's K function for cell island analysis.
    - `data/`: Raw and processed datasets (Mass Spec, phenotypic assays).
- `AXLdosage_bias/`: Scripts and figures for AXL dosage sensitivity analysis.
- `Figure*.ipynb`: Notebooks for main and supplemental figure generation.

## Authors

- **Marc Creixell** (creixell.marc@gmail.com)
- **Aaron S. Meyer** (git@ameyer.me)
