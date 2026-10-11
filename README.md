# Noise-Aware Quantum Machine Learning

Noise-Aware Quantum Machine Learning: Experimental Analysis and Adaptive Circuit
Selection for Robust Classification.

> **Status:** This repository is being built incrementally. Implemented so
> far: environment setup, project scaffolding, central configuration, the
> dataset module, the classical baselines, the VQC circuit module, the Aer
> noise models, the trainable VQC, and the metrics module -- with automated
> tests for all of it. The eight-experiment matrix runner, adaptive circuit
> selection, plotting, and the research report are **not yet implemented**
> -- see `task1.md` for the full assignment and `PROGRESS.md` for exactly
> what has been completed and what remains.

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
│   ├── config.py       # central configuration (paths, seeds, shots, noise profiles)
│   ├── datasets.py      # dataset loading + preprocessing (Iris/Wine/Breast Cancer)
│   ├── classical.py     # Logistic Regression + SVM baselines
│   ├── circuits.py      # angle encoding, RY/RZ+CNOT ansatz, resource metrics
│   ├── noise.py         # Aer noise models (profiles + isolated channels)
│   ├── vqc.py           # trainable variational quantum classifier
│   └── metrics.py       # classification + noise-sensitivity metrics
├── tests/               # 126 tests
│   ├── test_config.py
│   ├── test_datasets.py
│   ├── test_classical.py
│   ├── test_circuits.py
│   ├── test_noise.py
│   ├── test_vqc.py
│   └── test_metrics.py
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
quantum classifiers. Reports accuracy, precision, recall, F1-score,
confusion matrix, and training time. See `src/qml_noise/classical.py`.

## Quantum classifier

A shot-based variational quantum classifier: angle encoding (`H` +
`RY(x_i)`), an ansatz of trainable `RY`/`RZ` rotations plus a `CNOT` ring
repeated `depth` times, all qubits measured, and a parity (or single-qubit)
readout mapping counts to a binary class. Trained with binary cross-entropy
and SciPy COBYLA. See `src/qml_noise/vqc.py` and `src/qml_noise/circuits.py`.

```python
from qml_noise.datasets import load_iris_binary
from qml_noise.noise import get_noise_model
from qml_noise.vqc import VariationalQuantumClassifier

split = load_iris_binary(seed=0)
vqc = VariationalQuantumClassifier(n_qubits=2, depth=2, seed=0).fit(
    split.X_train, split.y_train
)
ideal = vqc.predict(split.X_test)
noisy = vqc.set_noise_model(get_noise_model("medium")).predict(split.X_test)
```

## Noise models

Three combined profiles (`ideal`/`low`/`medium`) and four isolated channels
(`bit_flip`, `phase_flip`, `depolarizing`, `readout`). Errors attach to the
transpiled `sx`/`x`/`cx` gates and to `measure`; `rz` is left error-free
because it is a virtual frame change on real hardware. All probabilities are
configurable in `config.py` and are **chosen defaults**, not values mandated
by the assignment. See `src/qml_noise/noise.py`.
