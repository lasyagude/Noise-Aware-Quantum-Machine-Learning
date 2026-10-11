# Progress — Noise-Aware QML Project

Tracking progress against `task1.md`.

| Stage | Assignment sections | Commit | Status |
|---|---|---|---|
| 1 | 1–4, 9, 12 — environment, structure, config, datasets, classical baselines | `9fd7d71` | done |
| 2 | 5, 6, 8, 10 — VQC circuits, experimental parameters, noise models, metrics | `8c0742c` | done |
| 3a | 7, 12 — training/evaluation module, experiment runner, smoke-mode run | — (uncommitted) | done |
| 3b | 7 (normal/full runs), 11, 13, 14 — full matrix execution, adaptive selection, plots, report | — | not started |

**Current state of `results/`:** the **smoke mode** of the E1–E8 matrix has
been run for real, for 1 seed, with a small COBYLA iteration budget, and its
output is genuinely saved in `results/vqc_experiment_results.jsonl`,
`results/vqc_experiment_results.csv`, and
`results/classical_baseline_results.jsonl`. These are real executions, not
mocked -- but they used a deliberately small iteration budget to prove the
pipeline works, not to produce numbers worth drawing conclusions from. The
**normal** (1 seed, full training budget) and **full** (5 seeds, full
training budget) runs have *not* been executed yet; exact commands to run
them are given below.

## Environment

- Python 3.14.3 (existing `.venv` at project root, Windows/win_amd64).
- Verified all required packages have compatible wheels for this Python
  version and installed them into `.venv`:
  `qiskit 2.5.2`, `qiskit-aer 0.17.2`, `qiskit-machine-learning 0.9.1`,
  `numpy 2.5.4`, `pandas 3.0.6`, `scikit-learn 1.9.1`, `matplotlib 3.11.2`,
  `pytest 9.1.1`.
- Confirmed `qiskit_aer.AerSimulator()` instantiates correctly, parameterized
  circuits transpile/bind correctly, and noise models attach as expected on
  this Python version.
- `requirements.txt` created with these exact pinned versions.

## Project structure

```
qml-noise-project/
├── src/qml_noise/
│   ├── __init__.py
│   ├── config.py      # central configuration (paths, seeds, shots, noise profiles)
│   ├── datasets.py     # dataset module
│   ├── classical.py    # classical baselines
│   ├── circuits.py     # VQC circuit construction + resource metrics
│   ├── noise.py        # Aer noise models
│   ├── vqc.py          # trainable variational quantum classifier
│   ├── metrics.py      # classification + noise-sensitivity metrics
│   ├── evaluation.py   # training/evaluation module (train once, eval under each noise condition)
│   └── experiments.py  # experiment runner: E1-E8 matrix, smoke/normal/full modes, resume
├── tests/               # 154 tests
│   ├── test_config.py
│   ├── test_datasets.py
│   ├── test_classical.py
│   ├── test_circuits.py
│   ├── test_noise.py
│   ├── test_vqc.py
│   ├── test_metrics.py
│   ├── test_evaluation.py
│   └── test_experiments.py
├── results/
│   ├── vqc_experiment_results.jsonl        # E1-E8, smoke mode, seed 0 (real, saved)
│   ├── vqc_experiment_results.csv          # scalar-only flattened view, regenerated each run
│   └── classical_baseline_results.jsonl    # Logistic Regression + SVM, seed 0 (real, saved)
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
- Mandatory experimental parameters (section 6): `REQUIRED_QUBIT_COUNTS =
  (2, 4, 6)`, `REQUIRED_DEPTHS = (2, 4)`,
  `NOISE_CONDITIONS = ("ideal", "low", "medium")`, plus
  `OPTIONAL_QUBIT_COUNTS = (2, 4, 6, 8)` and `OPTIONAL_DEPTHS = (2, 4, 6)`
  for the optional extensions.
- `BASIS_GATES = ("rz", "sx", "x", "cx")`, `NOISY_SINGLE_QUBIT_GATES`,
  `NOISY_TWO_QUBIT_GATES`, `TRANSPILE_OPTIMIZATION_LEVEL`.
- `NOISE_PROFILES` (`ideal` / `low` / `medium`) as `NoiseProfile` dataclasses,
  and `ISOLATED_CHANNEL_PROBABILITIES` for single-channel sweeps.
- `VQCConfig` / `VQC_CONFIG` with the training hyperparameters.
- The module docstring explicitly separates **mandatory** assignment values
  (1024 shots, 2/4/6 qubits, depth 2/4, the three noise conditions) from
  **chosen defaults** (every noise probability, the basis-gate set, and all
  training hyperparameters).

## Dataset module (`src/qml_noise/datasets.py`)

- `load_iris_binary()` — setosa vs. versicolor, 4 features, primary dataset.
- `load_wine_reduced()` — 13 features, optionally sliced to `n_features`.
- `load_breast_cancer_reduced()` — 30 features, optionally sliced to
  `n_features`.
- `load_dataset(name, ...)` — dispatcher by name.
- All loaders: stratified 80/20 split with fixed seed, `StandardScaler` fit
  on train only, return a `DatasetSplit` dataclass (train/test arrays,
  feature/class names, counts, seed).
- Wine and Breast Cancer load and reduce correctly (unit-tested) but have
  not yet been used to train a quantum or classical model — only Iris has.

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

One-off verification run (single seed, not saved to `results/`). Iris binary
(setosa vs. versicolor), 2 qubits, depth 2, 1024 shots, seed 0, COBYLA with
60 evaluations:

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

One-off verification run (single seed, not saved). 4 qubits, depth 2,
trained ideally (40 evaluations, so deliberately undertrained), then
evaluated under each condition with the trained weights held fixed:

| Condition | Test accuracy | D (pp) | D_rel (%) | R | TVD vs. ideal |
|---|---:|---:|---:|---:|---:|
| ideal | 0.8000 | — | — | — | — |
| low | 0.8500 | −5.00 | −6.25 | 1.0625 | 0.0615 |
| medium | 0.8000 | +0.00 | +0.00 | 1.0000 | 0.2285 |
| bit-flip p=0.10 | 0.4500 | +35.00 | +43.75 | 0.5625 | — |

Three things worth carrying into the report:

1. Accuracy under `low` noise came out *above* ideal (negative degradation,
   `R > 1`) — exactly the case the assignment warns not to assume away. The
   metrics report it honestly rather than clamping it.
2. The Iris test set has only 20 samples, so accuracy moves in 5-percentage-
   point steps and a single reclassified sample produces the entire ±5pp
   swing above. Single-seed accuracy differences of one or two steps are
   therefore within noise; this is precisely why the assignment asks for
   five seeds, and why the matrix results must be averaged before any
   degradation claim is made.
3. Output-distribution distance (TVD) rises monotonically with noise
   strength (0.062 → 0.229) even where accuracy does not move at all, so
   the distribution comparison is the more sensitive instrument at low
   noise.

## Training/evaluation module (`src/qml_noise/evaluation.py`, section 12)

**Design: train once per (qubits, depth), evaluate under every noise
condition in that group.** The matrix only ever varies noise condition
*within* a fixed architecture (E4/E5/E6 are all "4 qubits, depth 4"), which
is exactly the shape the assignment itself describes: establish a
noiseless baseline (section 2, item 2), then introduce noise and compare
against that baseline (items 3–4). So `train_ideal_vqc()` trains **once**
under ideal (noiseless) conditions per architecture, and
`evaluate_vqc_under_condition()` re-evaluates those exact trained weights
under a different noise model via `set_noise_model()` — it does not
retrain per noise condition.

- `TrainedVQCArtifact` — the trained model plus its ideal baseline metrics,
  pooled ideal measurement distribution, and training diagnostics.
- `ExperimentRecord` — every section 10 metric for one `(experiment_id,
  seed)` pair: classification metrics, confusion matrix, degradation /
  relative degradation / robustness ratio vs. the ideal baseline, all
  circuit resource metrics, training/eval/total runtime, and the pooled
  output-distribution comparison (TVD, Hellinger, outcomes unique to the
  noisy run).
- **Output distributions are pooled across the whole test set** (summed
  shot counts from every test-sample circuit), not kept per-sample — one
  readable ideal-vs-noisy distribution per experiment instead of 20+ tiny
  ones. Per-sample accuracy is computed separately and is unaffected.
- Reused training is tracked explicitly: a noisy row reusing another row's
  weights gets `training_time_seconds = 0.0` and
  `reused_training_from = "<ideal experiment id>"`, so runtime accounting
  never double-counts one training run across multiple matrix rows.

## Experiment runner (`src/qml_noise/experiments.py`, sections 7 + 12)

- `EXPERIMENT_MATRIX` — the exact, unmodified E1–E8 table from section 7,
  guarded by a test that fails if it is ever silently changed.
- `group_matrix_by_architecture()` groups rows sharing (qubits, depth):
  `(2,2):[E1]`, `(2,4):[E2]`, `(4,2):[E3]`, `(4,4):[E4,E5,E6]`,
  `(6,4):[E7,E8]` — five training runs cover all eight matrix rows.
- **Modes:** `smoke` (1 seed, maxiter=20 — proves the pipeline, not for
  conclusions), `normal` (1 seed, full maxiter=150), `full` (all 5 seeds,
  full maxiter=150).
- **Incremental saving:** every completed `(experiment_id, seed)` is
  appended to `results/vqc_experiment_results.jsonl` immediately, not
  buffered to the end of the run. A scalar-only
  `results/vqc_experiment_results.csv` is regenerated after each run.
  Classical baselines (Logistic Regression, SVM) are run once per seed and
  saved to `results/classical_baseline_results.jsonl`.
- **Resume:** on startup the runner reads the existing JSONL, builds the
  set of already-`"completed"` `(experiment_id, seed)` pairs, and skips
  them — rerunning the same command after an interruption only redoes what
  is missing. `--force` deletes the existing result files first instead of
  appending duplicates.
- **Caveat:** trained weights are not persisted across process runs. If a
  resumed run still has pending rows in a (qubits, depth) group, that
  architecture is retrained from scratch for that seed (deterministic,
  given the same seed and `--maxiter`) — only already-completed rows are
  skipped, not training itself.

### Confirmed: the smoke-mode pipeline runs end-to-end and saves real results

Ran for real against the actual `results/` directory (not a test, not
mocked): `PYTHONPATH=src python -m qml_noise.experiments --mode smoke`,
1 seed, maxiter=20 (SciPy's `n_params + 2` floor raised this to
20/20/20/34/50 per architecture). Total wall time **≈76 seconds**.

All 8 rows were saved, matching the assignment's matrix exactly:

| Experiment | Qubits | Depth | Noise | Accuracy | Degradation (pp) | Robustness | TVD vs. ideal | Runtime (s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| E1 | 2 | 2 | ideal | 1.00 | — | — | — | 7.14 |
| E2 | 2 | 4 | ideal | 1.00 | — | — | — | 9.15 |
| E3 | 4 | 2 | ideal | 0.80 | — | — | — | 9.62 |
| E4 | 4 | 4 | ideal | 0.85 | — | — | — | 18.12 |
| E5 | 4 | 4 | low | 0.85 | 0.00 | 1.0000 | 0.0638 | 0.52 |
| E6 | 4 | 4 | medium | 0.80 | 5.00 | 0.9412 | 0.2153 | 0.47 |
| E7 | 6 | 4 | ideal | 1.00 | — | — | — | 28.10 |
| E8 | 6 | 4 | medium | 1.00 | 0.00 | 1.0000 | 0.3202 | 1.06 |

Re-running the identical command confirmed resume works correctly: it
reported "8 already completed", skipped every group's training entirely,
and saved 0 new records.

**Read this table as a pipeline smoke test, not as science.** maxiter=20
is far below the normal/full budget of 150, several architectures were
visibly undertrained (E3's final loss was still 0.60, E4's 0.44), and this
is a single seed on a 20-sample test set, so accuracy moves in 5pp steps.
The one pattern that *is* already visible and expected to hold under full
training: output-distribution distance (TVD) still rises with noise
strength even at this iteration budget (E6 medium > E5 low: 0.215 > 0.064;
E8 medium TVD 0.320 at 6 qubits, larger circuit, more noisy gates). No
conclusions about accuracy degradation should be drawn from this table —
that requires the `full` run across five seeds.

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

154 tests, all passing, no warnings:

```
$ pytest
154 passed in ~2 minutes
```

Exact counts per file (`pytest --collect-only`, parameterized cases
expanded):

| File | Tests | Covers |
|---|---:|---|
| `test_config.py` | 6 | paths, seeds, mandatory shots, dataset names |
| `test_datasets.py` | 12 | stratified split, reproducibility, fit-on-train-only scaling, feature reduction, dispatch |
| `test_classical.py` | 5 | Logistic Regression / SVM accuracy and metrics |
| `test_circuits.py` | 31 | parameter-count formula, all required **and** optional qubit/depth configurations, encoding and ansatz structure, depth scaling, transpiled-vs-logical depth, resource counting |
| `test_noise.py` | 24 | which instructions carry noise, `rz` staying clean, full coverage of the circuit's noisy gates, isolated channels, probability validation, and empirical checks that noise shifts the output distribution and that `medium` shifts it more than `low` |
| `test_vqc.py` | 25 | readout mappings (including Qiskit's little-endian bit order), `FeatureAdapter` modes, reproducible initialization, error handling, the real training test, noise interaction |
| `test_metrics.py` | 23 | metric formulas, negative degradation, the `nan` zero-denominator convention, distribution distances, aggregation |
| `test_evaluation.py` | 9 | ideal-baseline training, evaluation under noise reproduces/changes metrics correctly, reused-training time accounting, resource metrics, pooled distributions, confusion matrix, JSON serializability |
| `test_experiments.py` | 19 | matrix integrity (exact table match, ≥6 experiments, ≥2 varied parameters), architecture grouping, CLI parsing, classical-baseline resume, a real full-matrix smoke run (section 10 metrics all present), resume/force/partial-completion bookkeeping (via a tiny synthetic matrix to stay fast) |
| **Total** | **154** | |

The resume/force/partial-completion tests use a tiny 2-row synthetic matrix
(passed via `run_matrix(..., matrix=...)`) rather than the real 8-row
matrix, so the bookkeeping logic is exercised repeatedly without paying for
6-qubit training each time. Exactly one test
(`test_smoke_run_produces_one_record_per_matrix_row`, via a module-scoped
fixture) runs the real E1–E8 matrix, once, to prove the actual pipeline
works end to end.

The COBYLA-floor warning (SciPy raising `maxiter` to `n_params + 2` for
these intentionally tiny test budgets) is suppressed via `filterwarnings`
in `pyproject.toml` — it is expected given the test budgets, not a defect.

## How to run

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -v
```

For any script outside pytest, `src` must be on the path (pytest gets this
from `pyproject.toml`; plain `python` does not):

```powershell
$env:PYTHONPATH="src"
```

### Running the experiment matrix

```powershell
# Smoke mode: 1 seed, small iteration budget. Proves the pipeline works.
# Already run for real -- see results above. Safe to rerun (resumes instantly).
python -m qml_noise.experiments --mode smoke

# Normal mode: 1 seed, full training budget (150 COBYLA iterations).
# NOT YET RUN. Estimated ~6 minutes based on smoke-run timing.
python -m qml_noise.experiments --mode normal

# Full mode: all 5 seeds, full training budget. The complete study.
# NOT YET RUN. Estimated ~25-35 minutes (5 x the normal-mode estimate).
python -m qml_noise.experiments --mode full

# Resume an interrupted run: identical command, already-completed
# (experiment, seed) pairs are skipped automatically.
python -m qml_noise.experiments --mode full

# Force a clean rerun, discarding existing results instead of resuming:
python -m qml_noise.experiments --mode full --force
```

Runtime estimates are extrapolated from the actual smoke run (maxiter=20,
~76s total) scaled to maxiter=150 assuming roughly linear per-iteration
cost; actual full/normal runtime has not yet been measured and may differ
if COBYLA converges before exhausting the iteration budget.

## Explicitly not done (stage 3b and beyond)

- **The `normal` and `full` runs have not been executed.** Only `smoke`
  (1 seed, maxiter=20) has been run for real; see the results table above.
  No result in `results/` yet reflects the full 150-iteration training
  budget or more than one seed.
- Section 11: adaptive noise-aware circuit-selection module.
- Section 13: data-driven plots (all 8 required figures) — not generated;
  `plots/` is still empty.
- Section 14: Markdown research report — not written; `report/` is still
  empty.
- Five-seed repetitions and cross-seed error bars: `aggregate_across_seeds()`
  exists in `metrics.py` but has not been applied to any real multi-seed
  result yet, since only 1 seed has been run.
- Optional thermal relaxation noise (section 8 explicitly defers it until
  the mandatory experiments work).
- Secondary datasets. Wine and Breast Cancer load and reduce correctly, but
  have only been exercised by unit tests, never run through the experiment
  matrix.

## Note on `task2.md`

`task2.md` now contains a follow-on task ("finish, run, verify, and deliver
the complete project") that picks up exactly where this file leaves off:
inspect existing work, run the E1–E8 matrix, build the adaptive-selection
module, generate plots, and write the report. Nothing in it has been acted
on yet — it is recorded here only because its presence explains what stage
3 will look like when it starts.
