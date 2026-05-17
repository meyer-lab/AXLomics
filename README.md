# Dissecting AXL-mediated signaling and resistance

The project investigates how the AXL receptor tyrosine kinase drives resistance to EGFR-targeted therapies in lung cancer. It employs a systems biology approach, utilizing **Dual Data-Motif Clustering (DDMC)** to integrate phosphoproteomic abundance and sequence information, and **Partial Least Squares Regression (PLSR)** to link signaling clusters to phenotypic outcomes (survival, migration, and spatial clustering).

## Installation

This project uses `uv` for dependency management. To set up the environment, run:

```bash
uv sync
```

## Usage

### Generating Figures

The figures for the paper can be generated using the provided `makefile`. To generate a specific figure (e.g., Figure 1):

```bash
make output/figure1.svg
```

Alternatively, you can run the Jupyter notebooks (`Figure1.ipynb`, `Figure2.ipynb`, etc.) directly to reproduce the analysis.

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
