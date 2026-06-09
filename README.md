# Spurious Halo Classifier

[![CI](https://github.com/robmost/spurious_halo_classifier/actions/workflows/ci.yaml/badge.svg)](https://github.com/robmost/spurious_halo_classifier/actions/workflows/ci.yaml)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20521139.svg)](https://doi.org/10.5281/zenodo.20521139)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![DuckDB](https://img.shields.io/badge/DuckDB-1.3%2B-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8%2B-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-EE4C2C?logo=pytorch&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.11%2B-0194E2)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Scientific background

Warm dark matter (WDM) N-body simulations produce artefacts called *spurious haloes*, unphysical objects that arise from numerical noise rather than genuine gravitational collapse (see e.g. [Wang \& White 2007, MNRAS 380, 93](https://doi.org/10.1111/j.1365-2966.2007.12053.x); [Lovell et al. 2014, MNRAS 439, 300](https://doi.org/10.1093/mnras/stt2431)). They appear as regularly spaced beads along cosmic filaments and contaminate any analysis of the low-mass halo population.

The companion paper ([Mostoghiu Paun et al. 2025, MNRAS 542, 735](https://doi.org/10.1093/mnras/staf1229)) identifies spurious haloes using an empirical sphericity cut on protohalo Lagrangian volumes, following the [Lovell et al. 2014, MNRAS 439, 300](https://doi.org/10.1093/mnras/stt2431) method: haloes that formed from flattened initial volumes are flagged spurious. That cut is a single
threshold on a single feature, manually tuned, and shown to break down when the simulation's initial redshift changes.

This project explores the effects of replacing the empirical cut with a trained binary classifier. Labels are derived from WDM–CDM cross-correlation derived from a merit function, where a halo that exists in WDM but has no CDM counterpart is spurious by construction, decoupling the label from sphericity entirely. A follow-up SHAP analysis then asks which features actually drive classification, and whether sphericity is as informative as the paper assumes.

## Key results

### Model performance

Average Precision / ROC-AUC / F1 across all models and evaluation splits:

| Model | within-sim | cross-softening | cross-z_ini |
| --- | --- | --- | --- |
| Logistic Regression | 0.982 / 0.944 / 0.927 | 0.973 / 0.920 / 0.914 | 0.977 / 0.930 / 0.930 |
| Random Forest | 0.985 / 0.949 / 0.965 | 0.977 / 0.932 / 0.951 | 0.983 / 0.946 / 0.959 |
| Gradient Boosted Trees | 0.986 / 0.953 / 0.949 | 0.973 / 0.924 / 0.956 | 0.982 / 0.943 / 0.965 |
| Soft-voting Ensemble | 0.985 / 0.952 / 0.955 | 0.978 / 0.932 / 0.952 | 0.981 / 0.941 / 0.959 |
| MLP (impute) | 0.986 / 0.953 / 0.944 | 0.969 / 0.920 / 0.936 | 0.980 / 0.940 / 0.936 |
| MLP (mask) | 0.986 / 0.953 / 0.946 | 0.972 / 0.919 / 0.943 | 0.980 / 0.939 / 0.947 |

Where:
- `within_sim`: train and test on the same simulation
- `cross_softening`: train on fixed softening, test on tidal adaptive softening
- `cross_z_ini`: train on z_ini=39, test on z_ini=99

### SHAP feature importance

`log10_m200` dominates all models. This is expected, as spurious haloes are associated with artificial fragmentation near the WDM free-streaming mass scale, so low mass is already an informative regime. The CDM-match label also inherits mass dependence through counterpart availability below the free-streaming cutoff. Nothing surprising here.

`v_disp_sigv` (velocity dispersion) ranks third, or rather second among physically distinct features, since `log10_npart` is collinear with mass in non-zoom simulations; consistently across all model types. This is the non-trivial result. A plausible interpretation is that artificial-fragmentation haloes have systematically different internal kinematics than genuine haloes at similar mass.

`sphericity_s`, the paper's primary diagnostic, ranks near the bottom. Two explanations are consistent with this. First, the CDM-match label is not defined by sphericity, so once mass-linked structure is captured, sphericity adds limited incremental signal; and secondly, roughly 25% of haloes lack protohalo records (and therefore `sphericity_s`), which dilutes its measured importance relative to always-available features. This suggests that the paper's empirical cut is largely a mass cut in disguise.

### Generalisation

The expected structured failure on `cross_z_ini` did not materialise (RF average-precision only dropped 0.0024 from `within-sim`). The CDM-match label is robust to the initial-redshift shift because it is based on particle overlap, not morphology. Both WDM and CDM sphericities shift downward at z_ini=99, but the merit criterion is unaffected.

Counterintuitively, `cross_softening` degrades more (AP drop 0.0087). This is because tidal adaptive softening shifts halo formation times and increases the spurious fraction near the 100-particle mass limit, creating a harder test distribution.

Random Forest is the recommended model, as it generalises best and the soft-voting ensemble adds negligible diversity.

## Tech stack

- **Pipeline**: DuckDB, Polars, NumPy, SciPy
- **ML**: scikit-learn (LR, RF, GBM, Ensemble), PyTorch (MLP)
- **Interpretability**: SHAP
- **Experiment tracking**: MLflow
- **Data architecture**: DuckDB Databricks medallion (bronze -> silver -> gold)
- **Dev tooling**: ruff, basedpyright, pytest (95 tests), GitHub Actions CI

## Architecture

```text
Raw files (AHF catalogues, HDF5 shapes, MergerTree cross-correlations)
       |
       V  make gold
 bronze.*  -- raw ingestion into 4 tables, no transforms
 silver.*  -- column renames, merit filtering, and protohalo join into 4 tables
   gold.*  -- 15 ML features, CDM-match labels, train/val/test splits
       |
       V  make train
  models/  -- 12 sklearn + 6 PyTorch models, all tracked in MLflow
       |
       V  notebooks/
 EDA + SHAP + generalisation analysis (01–07)
```

## Getting started

### Prerequisites

- Python 3.13
- Conda (recommended) or any virtual environment

### Installation

```bash
git clone https://github.com/rmostoghiupaun/spurious_halo_classifier.git
cd spurious_halo_classifier
pip install -e ".[dev]"
```

### Environment setup (optional — for LaTeX rendering in notebooks)

```bash
cp .env.example .env
# Set LATEX_BIN_DIR to your TeXLive bin directory
```

### Data

Raw simulation data (~11 GB, 8 simulations) is available on Zenodo: [https://doi.org/10.5281/zenodo.20521139](https://doi.org/10.5281/zenodo.20521139). Download and unpack under `data/raw/` following the layout in `config.yaml`. The DuckDB database is built from scratch and does not need to be downloaded separately.

### Running the pipeline

```bash
make gold         # builds bronze -> silver -> gold in order
make train        # builds bronze -> silver -> gold, then trains all 18 models
make lint         # ruff + basedpyright
make test         # pytest with coverage
```

Each target depends on the previous layer. `make gold` automatically runs the full bronze -> silver -> gold chain, and `make train` runs everything through to model training.

To reset and rebuild from scratch:

```bash
make reset        # drop database + remove model artefacts
make train        # rebuild everything end-to-end
```

## Notebooks

All notebooks require the full pipeline (`make train`) to have been run first.

| Notebook | Description |
| --- | --- |
| `01_eda_bronze.ipynb` | Raw data: halo counts, AHF schema, protohalo coverage, merit score distribution |
| `02_eda_silver.ipynb` | Cleaned data: mass distributions, unit sanity checks, match statistics |
| `03_label_analysis.ipynb` | Label comparison: where do the two labels agree? Mass-dependence of disagreement |
| `04_feature_distributions.ipynb` | Feature completeness, per-feature distributions by spurious label |
| `05_model_comparison.ipynb` | PR curves, ROC curves, confusion matrices across models and splits |
| `06_shap_analysis.ipynb` | SHAP feature importance: which features drive classification across splits? |
| `07_generalisation.ipynb` | Why cross-z_ini did not fail: feature shift analysis, mass-binned error rates |

## Repository structure

```text
├── src/
│   ├── bronze/          # raw ingestion parsers
│   ├── silver/          # cleaning and joining
│   ├── gold/            # feature engineering, labels, splits
│   ├── models/          # sklearn and PyTorch training, evaluation, shared data loading
│   ├── utils/           # plotting helpers
│   ├── config.py        # config loader; parses and validates all config.yaml fields
│   └── db.py            # DuckDB connection and shared utilities (log_row_counts)
├── sql/
│   ├── schema/          # DDL reference (bronze.sql, silver.sql, gold.sql)
│   └── queries/         # analytical reference queries
├── notebooks/           # EDA and analysis (01–07)
├── tests/               # 95 pytest tests across 7 modules
│   └── fixtures/        # sample AHF and MergerTree test data
├── scripts/             # MergerTree cross-correlation runner
├── reports/
│   ├── figures/         # generated plots
│   └── spurious_halo_classifier.mplstyle
├── .github/
│   └── workflows/
│       └── ci.yaml      # lint, type-check, and test on every push
├── data/                # .gitkeep — database written here by make bronze
│   └── raw/             # .gitkeep — place Zenodo data here
├── models/              # trained model artefacts (tracked in MLflow)
├── config.yaml          # all configurable parameters
├── pyproject.toml       # package metadata and tool configuration
├── LICENSE              # MIT 2026
├── .env.example         # template for LATEX_BIN_DIR and MTREEBINPATH
└── Makefile             # pipeline entrypoints
```

## Tests

```bash
make test
```

95 tests across 7 modules covering config validation, parsers, silver transforms, gold labels and features, and model evaluation. CI runs `make lint` and `make test` on every push and pull request to `main`.

## Citation

```bibtex
@article{mostoghiupaun2025,
  author  = {Mostoghiu Paun, R.~A. and Croton, D. and Power, C. and
             Knebe, A. and Ussing, A.~J. and Duffy, A.~R.},
  title   = {Tidal adaptive softening and artificial fragmentation
             in cosmological simulations},
  journal = {MNRAS},
  year    = {2025},
  volume  = {542},
  pages   = {735--746},
  doi     = {10.1093/mnras/staf1229}
}
```
