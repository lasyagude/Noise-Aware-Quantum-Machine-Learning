# Progress — Noise-Aware QML Project

Tracking progress against `task1.md`. This checkpoint covers **only**
sections 1–4, 9, and 12 of the assignment, as scoped by the user. Sections
5–8, 10, 11, 13, and 14 (quantum circuits/VQC, noise models, metrics beyond
classical, adaptive selection, experiment matrix execution, plots, and the
research report) are **not started**.

## Environment

- Python 3.14.3 (existing `.venv` at project root, Windows/win_amd64).
- Verified all required packages have compatible wheels for this Python
  version and installed them into `.venv`:
  `qiskit 2.5.2`, `qiskit-aer 0.17.2`, `qiskit-machine-learning 0.9.1`,
  `numpy 2.5.4`, `pandas 3.0.6`, `scikit-learn 1.9.1`, `matplotlib 3.11.2`,
  `pytest 9.1.1`.
- Confirmed `qiskit_aer.AerSimulator()` instantiates correctly in this
  environment.
- `requirements.txt` created with these exact pinned versions.

## Project structure

```
qml-noise-project/
├── src/qml_noise/
│   ├── __init__.py
│   ├── config.py      # central configuration
│   ├── datasets.py     # dataset module
│   └── classical.py    # classical baselines
├── tests/
│   ├── test_config.py
│   ├── test_datasets.py
│   └── test_classical.py
├── results/   (empty, .gitkeep — for later experiment output)
├── plots/     (empty, .gitkeep — for later PNG plots)
├── report/    (empty, .gitkeep — for later Markdown report)
├── configs/   (empty — reserved for future external config files)
├── pyproject.toml   # pytest config (src on pythonpath)
├── requirements.txt
├── README.md
└── task1.md / task2.md
```

`pyproject.toml` sets `tool.pytest.ini_options.pythonpath = ["src"]` so the
package is importable by tests without an editable install.

## Central configuration (`src/qml_noise/config.py`)

- Project paths (`PROJECT_ROOT`, `RESULTS_DIR`, `PLOTS_DIR`, `REPORT_DIR`,
  `CONFIGS_DIR`), auto-created on import.
- `RANDOM_SEEDS = [0, 1, 2, 3, 4]` (the assignment's "five seeds where
  feasible" requirement) and `DEFAULT_SEED`.
- `TEST_SIZE = 0.2`, `SUPPORTED_DATASETS`, `PRIMARY_DATASET = "iris"`.
- `SHOTS = 1024` (mandatory assignment value).
- `ClassicalBaselineConfig` dataclass with Logistic Regression / SVM
  hyperparameter defaults.
- `get_logger()` factory for consistent logging across modules.
- **Not included yet** (out of scope for this checkpoint): qubit-count,
  ansatz-depth, and noise-channel parameters — those belong to the
  circuit/noise modules from sections 5–8 and will get their own config
  additions when those modules are built.

## Dataset module (`src/qml_noise/datasets.py`)

- `load_iris_binary()` — setosa vs. versicolor, 4 features, primary dataset.
- `load_wine_reduced()` — 13 features, optionally sliced to `n_features`.
- `load_breast_cancer_reduced()` — 30 features, optionally sliced to
  `n_features`.
- `load_dataset(name, ...)` — dispatcher by name.
- All loaders: stratified 80/20 split with fixed seed, `StandardScaler` fit
  on train only, return a `DatasetSplit` dataclass (train/test arrays,
  feature/class names, counts, seed).

## Classical baselines (`src/qml_noise/classical.py`)

- `run_logistic_regression(split)`, `run_svm(split)`,
  `run_all_classical_baselines(split)`.
- Each returns a `ClassicalResult` with accuracy, precision, recall,
  F1-score, confusion matrix, and training time (wall-clock, via
  `time.perf_counter`).
- Binary datasets use `average="binary"`; multi-class datasets fall back to
  `average="macro"`.

## Tests

23 tests across `test_config.py`, `test_datasets.py`, `test_classical.py`,
covering: path/seed/config sanity, stratified-split correctness and
reproducibility, scaler fit-on-train-only behavior, feature-reduction
bounds-checking, dataset dispatch, and classical-model accuracy/metrics
shape.

```
$ pytest
23 passed in 4.20s
```

No test failures were encountered — nothing needed fixing.

## How to run (this checkpoint)

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -v
```

## Explicitly not done (deferred to later work)

- Sections 5–8: VQC/quantum circuit module, noise-model module, the
  8-experiment matrix, required qubit/depth configurations.
- Section 10: quantum-specific metrics (degradation, robustness ratio,
  gate counts, circuit depth).
- Section 11: adaptive noise-aware circuit-selection module.
- Section 13: data-driven plots.
- Section 14: Markdown research report.
- Training/evaluation module, adaptive-selection module, experiment runner,
  plotting/report-generation module (remaining items from section 12's
  module list).
