# Noise-Aware Quantum Machine Learning

Noise-Aware Quantum Machine Learning: Experimental Analysis and Adaptive Circuit
Selection for Robust Classification.

> **Status:** This repository is being built incrementally. The current
> checkpoint implements the environment setup, project scaffolding, central
> configuration, the dataset module, and the classical (non-quantum)
> baselines, with automated tests for all of it. Quantum circuits, noise
> models, training/evaluation, adaptive selection, the experiment runner,
> plotting, and the research report are **not yet implemented** -- see
> `task1.md` for the full assignment and `PROGRESS.md` for exactly what has
> been completed so far and what remains.

## Setup

```powershell
# from the project root
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running tests

```powershell
pytest
```

## Project structure (current)

```
qml-noise-project/
├── src/qml_noise/
│   ├── __init__.py
│   ├── config.py       # central configuration (paths, seeds, shots, classical defaults)
│   ├── datasets.py      # dataset loading + preprocessing (Iris/Wine/Breast Cancer)
│   └── classical.py     # Logistic Regression + SVM baselines
├── tests/
│   ├── test_config.py
│   ├── test_datasets.py
│   └── test_classical.py
├── results/   # saved experiment results (CSV/JSON) -- populated by later modules
├── plots/     # saved PNG plots -- populated by later modules
├── report/    # Markdown research report -- populated by later modules
├── configs/   # optional external config files
├── pyproject.toml
├── requirements.txt
└── task1.md   # full assignment specification
```

## Datasets

- **Iris (binary, setosa vs. versicolor)** -- primary dataset, 4 features.
- **Wine** -- secondary/optional, 13 features (reducible via `n_features`).
- **Breast Cancer Wisconsin Diagnostic** -- secondary/optional, 30 features
  (reducible via `n_features`).

All loaders use a stratified 80:20 train/test split with a fixed seed and fit
preprocessing (`StandardScaler`) on the training data only. See
`src/qml_noise/datasets.py`.

## Classical baselines

Logistic Regression and SVM, trained/evaluated on the exact same split as the
(future) quantum classifiers. Reports accuracy, precision, recall, F1-score,
confusion matrix, and training time. See `src/qml_noise/classical.py`.
