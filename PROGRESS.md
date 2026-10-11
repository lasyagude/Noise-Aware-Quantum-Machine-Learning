# Progress — Noise-Aware QML Project

Tracking progress against `task1.md`.

**Completed so far:** sections 1–4, 9, 12 (stage 1) and sections 5, 6, 8, 10
(stage 2). **Not started:** sections 7 (run the eight-experiment matrix),
11 (adaptive circuit selection), 13 (plots), 14 (research report), plus the
training/evaluation runner and plotting modules from section 12's list.

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
│   ├── classical.py    # classical baselines
│   ├── circuits.py     # VQC circuit construction + resource metrics
│   ├── noise.py        # Aer noise models
│   ├── vqc.py          # trainable variational quantum classifier
│   └── metrics.py      # classification + noise-sensitivity metrics
├── tests/
│   ├── test_config.py
│   ├── test_datasets.py
│   ├── test_classical.py
│   ├── test_circuits.py
│   ├── test_noise.py
│   ├── test_vqc.py
│   └── test_metrics.py
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

Added in stage 2 (sections 6 and 8):

- Mandatory experimental parameters: `REQUIRED_QUBIT_COUNTS = (2, 4, 6)`,
  `REQUIRED_DEPTHS = (2, 4)`, `NOISE_CONDITIONS = ("ideal", "low", "medium")`,
  plus `OPTIONAL_QUBIT_COUNTS = (2, 4, 6, 8)` and
  `OPTIONAL_DEPTHS = (2, 4, 6)` for the optional extensions.
- `BASIS_GATES = ("rz", "sx", "x", "cx")`, `NOISY_SINGLE_QUBIT_GATES`,
  `NOISY_TWO_QUBIT_GATES`, `TRANSPILE_OPTIMIZATION_LEVEL`.
- `NOISE_PROFILES` (`ideal` / `low` / `medium`) as `NoiseProfile` dataclasses,
  and `ISOLATED_CHANNEL_PROBABILITIES` for single-channel sweeps.
- `VQCConfig` / `VQC_CONFIG` with the training hyperparameters.
- The module docstring now explicitly separates **mandatory** assignment
  values (1024 shots, 2/4/6 qubits, depth 2/4, the three noise conditions)
  from **chosen defaults** (every noise probability, the basis-gate set,
  and all training hyperparameters).

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

## VQC circuits (`src/qml_noise/circuits.py`, section 5 + 6)

- **Angle encoding:** `H` then `RY(scale * x_i)` on each qubit.
- **Ansatz block:** trainable `RY` on all qubits, trainable `RZ` on all
  qubits, then a `CNOT` ring `(i -> i+1 mod n)`. Logical ansatz depth = the
  number of repeated blocks, matching the assignment's definition.
- **Trainable parameter count:** `2 * n_qubits * depth`.
- **Readout:** all qubits measured; `parity` (default, XOR of all bits) or
  `single` (qubit 0 only) maps the bitstring distribution to a class.
  `P(class 1)` comes from the shot counts; label is 1 when `P > 0.5`.
- `transpile_circuit()` compiles into `BASIS_GATES`, which is what makes the
  noise model attach to real instructions.
- `circuit_resources()` records qubits, logical depth, transpiled depth,
  total gate count, two-qubit gate count, and parameter count (`barrier` and
  `measure` are excluded from gate totals as non-gates).

Measured resources for the required configurations:

| Qubits | Logical depth | Transpiled depth | Gates | 2-qubit gates | Parameters |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 19 | 32 | 4 | 8 |
| 2 | 4 | 33 | 56 | 8 | 16 |
| 4 | 2 | 23 | 64 | 8 | 16 |
| 4 | 4 | 41 | 112 | 16 | 32 |
| 6 | 2 | 27 | 96 | 12 | 24 |
| 6 | 4 | 49 | 168 | 24 | 48 |

## Noise models (`src/qml_noise/noise.py`, section 8)

- **Combined profiles:** `ideal` (returns `None` — no model attached),
  `low`, `medium`; depolarizing on 1q/2q gates plus symmetric readout error.
- **Isolated channels:** `bit_flip` (Pauli-X), `phase_flip` (Pauli-Z),
  `depolarizing`, `readout` — one mechanism at a time, so degradation can be
  attributed to a specific cause. Two-qubit Pauli errors are the tensor
  product of the single-qubit error.
- `get_noise_model(condition, probability)` resolves any profile or channel
  name; `get_backend(noise_model)` returns the `AerSimulator`.
- Probability inputs are validated to `[0, 1]`; zero probability returns
  `None` rather than a zero-strength model.

**Where noise lands (verified, not assumed).** Errors attach to `sx`, `x`,
`cx`, and `measure`. `rz` is deliberately excluded: on superconducting
hardware it is a virtual frame change and therefore error-free, so attaching
noise to it would overstate single-qubit error. Verified on a transpiled
4-qubit depth-2 circuit:

- transpiled instructions: `{rz, sx, cx, measure, barrier}`
- noise-carrying gates present in the circuit (`{sx, cx, measure}`) are all
  covered by the noise model
- `rz` is present in the circuit and carries no error

The noise model also declares an error for `x`, which this particular
transpilation happens not to emit; declaring an error for an absent gate is
a harmless no-op.

## Trainable VQC (`src/qml_noise/vqc.py`, section 5)

- Loss: binary cross-entropy on the shot-estimated `P(class 1)`, clipped so
  a 0/1024 outcome cannot give infinite loss.
- Optimizer: SciPy `COBYLA` (gradient-free, suited to noisy shot-based
  objectives). SciPy silently raises `maxiter` to at least
  `n_parameters + 2`.
- Initialization: `Uniform(-init_scale, init_scale)` from
  `np.random.default_rng(seed)` — reproducible per seed.
- `FeatureAdapter` matches feature count to qubit count: PCA to reduce
  (4 features → 2 qubits), cyclic tiling to expand (4 → 6). **Fit on
  training data only.**
- The parameterized circuit is transpiled **once**, then per-sample values
  are bound, so noise exposure is identical across samples and iterations;
  a whole batch is submitted to Aer in one `run` call (~0.3 s for 80
  circuits × 1024 shots).
- `set_noise_model()` swaps the noise model while keeping trained weights —
  this is how an ideally-trained model is evaluated under noise on the same
  circuit and weights.

### Confirmed: the VQC genuinely trains

Iris binary (setosa vs. versicolor), 2 qubits, depth 2, 1024 shots, seed 0,
COBYLA with 60 evaluations:

```
loss 1.3750 -> 0.2597
train accuracy 0.9125, test accuracy 0.9500
8 trainable parameters, transpiled depth 19
```

Loss falls by ~81% and test accuracy reaches 0.95 against a 0.50 chance
baseline, so the model is learning rather than emitting random or untrained
predictions. This is asserted in
`tests/test_vqc.py::test_vqc_actually_trains_on_iris_binary`.

### Confirmed: noise degrades behaviour measurably

4 qubits, depth 2, trained ideally (40 evaluations, so deliberately
undertrained), then evaluated under each condition with weights fixed:

| Condition | Test accuracy | D (pp) | D_rel (%) | R | TVD vs. ideal |
|---|---:|---:|---:|---:|---:|
| ideal | 0.8000 | — | — | — | — |
| low | 0.8500 | −5.00 | −6.25 | 1.0625 | 0.0615 |
| medium | 0.8000 | +0.00 | +0.00 | 1.0000 | 0.2285 |
| bit-flip p=0.10 | 0.4500 | +35.00 | +43.75 | 0.5625 | — |

Two things worth carrying into the report: accuracy under `low` noise came
out *above* ideal (negative degradation, `R > 1`), exactly the case the
assignment warns not to assume away — the metrics report it honestly rather
than clamping it. And output-distribution distance (TVD) increases
monotonically with noise strength even where accuracy does not, so the
distribution comparison is the more sensitive instrument at low noise.

## Metrics (`src/qml_noise/metrics.py`, section 10)

- `classification_metrics()` — accuracy, precision, recall, F1, confusion
  matrix (binary average for 2 classes, macro otherwise).
- `accuracy_degradation()` = `100 * (A_ideal - A_noisy)`;
  `relative_degradation()` = `100 * (A_ideal - A_noisy) / A_ideal`;
  `noise_robustness_ratio()` = `A_noisy / A_ideal`; `noise_sensitivity()`
  bundles all three.
- **Zero-denominator convention:** when `A_ideal == 0`, the two ratio
  metrics return `float("nan")` so the undefined case stays visible in
  tables instead of masquerading as a real number. Absolute degradation has
  no denominator and stays well defined.
- **Sign convention:** values are never clamped; noise can increase measured
  accuracy, giving negative degradation and `R > 1`.
- Output-distribution comparison: `total_variation_distance()`,
  `hellinger_distance()`, and `compare_distributions()` (which also counts
  outcomes that only appear under noise).
- `aggregate_across_seeds()` — mean/std/min/max for cross-seed error bars.
- Circuit resource metrics (qubits, logical/transpiled depth, gate counts,
  parameter count) live in `circuits.CircuitResources`.

## Environment compatibility on Python 3.14

Checked as the work progressed; no issues found:

- `qiskit 2.5.2`, `qiskit-aer 0.17.2` and `qiskit-machine-learning 0.9.1`
  all import and run on Python 3.14.3 (`VQC` and `COBYLA` from
  qiskit-machine-learning import cleanly against Qiskit 2.x).
- Parameterized circuits transpile and bind correctly, `AerSimulator`
  executes batched circuit lists, noise models attach as expected, and
  `qiskit.quantum_info.hellinger_fidelity` works.
- The VQC is implemented directly on Qiskit + Aer primitives (SciPy COBYLA
  as the classical optimizer) rather than via
  `qiskit_machine_learning.algorithms.VQC`, for full control over the
  readout strategy, the shot budget, and where the noise model attaches.
  Everything quantum still runs exclusively on Qiskit/Aer.
- One avoidable pitfall: passing `basis_gates` together with `backend` to
  `transpile()` emits a `UserWarning`, so `transpile_circuit()` passes
  `basis_gates` only.

## Tests

126 tests, all passing, no warnings:

```
$ pytest
126 passed in 23.35s
```

- `test_config.py` (6) — paths, seeds, mandatory shots, dataset names.
- `test_datasets.py` (12) — stratified split, reproducibility,
  fit-on-train-only scaling, feature reduction, dispatch.
- `test_classical.py` (5) — Logistic Regression / SVM accuracy and metrics.
- `test_circuits.py` (~35, parameterized) — parameter-count formula, all
  required **and** optional qubit/depth configurations, encoding and ansatz
  structure, depth scaling, transpiled-vs-logical depth, resource counting.
- `test_noise.py` (~30, parameterized) — which instructions carry noise,
  `rz` staying clean, full coverage of the circuit's noisy gates, isolated
  channels, probability validation, and empirical checks that noise shifts
  the output distribution and that `medium` shifts it more than `low`.
- `test_vqc.py` (~24) — readout mappings (including Qiskit's little-endian
  bit order), `FeatureAdapter` modes, reproducible initialization, error
  handling, the real training test, and noise interaction.
- `test_metrics.py` (~26) — metric formulas, negative degradation, the
  `nan` zero-denominator convention, distribution distances, aggregation.

Two warnings surfaced on the first run and were fixed in the tests: COBYLA
budgets below `n_params + 2` (raised to valid values) and a PCA fit on an
all-zeros array (replaced with random data). No source-code defects were
found.

## How to run (this checkpoint)

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -v
```

For ad-hoc scripts outside pytest, `src` must be on the path:

```powershell
$env:PYTHONPATH="src"; python -c "from qml_noise.circuits import describe_circuit; print(describe_circuit(4, 4).to_dict())"
```

## Explicitly not done (deferred to later work)

- Section 7: running the eight-experiment matrix (E1–E8). All circuit,
  noise, and metric machinery it needs now exists; the runner itself does
  not.
- Section 11: adaptive noise-aware circuit-selection module.
- Section 13: data-driven plots.
- Section 14: Markdown research report.
- Remaining section 12 modules: training/evaluation runner,
  adaptive-selection module, experiment runner, plotting/report generation,
  smoke/normal/full modes, and resumable incremental result saving.
- Optional thermal relaxation noise (section 8 defers it until the
  mandatory experiments work).
- No experiment results have been saved to `results/` and no plots to
  `plots/` yet — those come with the section 7/13 work.
