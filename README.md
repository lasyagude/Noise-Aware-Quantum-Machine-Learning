# Noise-Aware Quantum Machine Learning

Noise-Aware Quantum Machine Learning: Experimental Analysis and Adaptive Circuit
Selection for Robust Classification.

> **Status: all mandatory requirements complete.** The VQC, noise models, metrics, experiment
> runner, the full E1-E8 x 5-seed Iris matrix, both extensions (noise sweep and Breast Cancer, 5
> seeds each), AN-VQC (3 seeds), all plots, and `report/report.md` are done; `report/report.md`
> has no outstanding placeholders. Not run: Wine (loader and tests only), and the full 6-config x
> 3-noise-level grid beyond the assignment's required 8-cell matrix. See the
> [requirement compliance checklist](#requirement-compliance-checklist) below for the exact
> PASS/PARTIAL/FAIL/NOT RUN status of every assignment requirement, and `PROGRESS.md` for
> turn-by-turn verification notes.

## Setup

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"      # needed for `python -m ...`; pytest sets it itself
```

Python 3.14 with `qiskit 2.5.2`, `qiskit-aer 0.17.2`,
`qiskit-machine-learning 0.9.1` (all verified compatible).

## Tests

```powershell
pytest          # runs the full automated test suite (training tests make a full run take several
                # minutes); 224+ test functions across tests/*.py as of the last count -- rerun
                # pytest for the exact current collected count, which this file does not assert
```

## Project structure

```
src/qml_noise/
  config.py       central config (seeds, shots=1024, noise profiles, extension defaults)
  datasets.py     Iris/Wine/Breast Cancer loaders, stratified 80:20, scaler fit on train only
  classical.py    Logistic Regression + SVM baselines
  circuits.py     angle encoding, RY/RZ + CNOT-ring ansatz, resource metrics
  noise.py        Aer noise models (ideal/low/medium + bit-flip/phase-flip/depolarizing/readout)
  vqc.py          trainable VQC (binary cross-entropy, COBYLA, seeded init)
  metrics.py      classification, degradation, robustness ratio, TVD/Hellinger
  evaluation.py   train once per (qubits, depth), evaluate under each noise model
  experiments.py  E1-E8 runner: smoke/normal/full modes, incremental save, resume
  extensions.py   EXTENSION experiments (separate result files, never touch E1-E8)
  an_vqc.py       adaptive noise-aware circuit selection (AN-VQC), 3-seed comparison vs 4 baselines
  plotting.py     all plots, from saved results only (skips incomplete/missing data)
tests/            automated tests (pytest; see "Tests" above for the last verified collected count)
results/          real experiment output (JSONL/CSV/JSON); results/smoke_checks/ = pipeline checks only
plots/            PNG figures + noise_comparison_table.csv/.md (regenerate: see below)
report/           report.md (complete, no outstanding placeholders)
```

## Main experiment matrix (E1-E8)

The assignment's exact matrix (2/4/6 qubits, depth 2/4, ideal/low/medium
noise, 1024 shots, seeds 0-4). A model is trained once per (qubits, depth)
under ideal conditions, then the same weights are re-evaluated under each
noise condition.

```powershell
python -m qml_noise.experiments --mode smoke    # 1 seed, tiny budget (~1 min), pipeline check only
python -m qml_noise.experiments --mode normal   # 1 seed, 150 iterations (~6 min)
python -m qml_noise.experiments --mode full     # 5 seeds, 150 iterations (~28 min; done)
```

Results: `results/vqc_experiment_results.jsonl` (+ `.csv`),
`results/classical_baseline_results.jsonl`. Every completed
`(experiment_id, seed)` is appended immediately; rerunning the same command
skips completed pairs (resume). `--force` deletes existing results first.

**Caution:** smoke rows count as "completed" for resume. Run smoke checks into
a scratch location, or use `--force` on the first real run.

## Extension experiments

Separate files, every record labeled with an `"extension"` field; the E1-E8
files are never written. All probabilities and sizes are **chosen defaults**
in `config.py`, not assignment values.

```powershell
# Read-only / seconds: split check + pooled analysis of the existing E1-E8 results
python -m qml_noise.extensions split-check        # -> results/extension_split_check.json
python -m qml_noise.extensions pooled-analysis    # -> results/extension_pooled_analysis.json

# Depolarizing noise sweep (4q/d4 and 6q/d4; 2-qubit p in 0,.001,.005,.01,.02,.05; 1-qubit p = p/10)
python -m qml_noise.extensions noise-sweep --mode full      # -> results/extension_noise_sweep.jsonl

# Breast Cancer E1-E8 (PCA to qubit count fit on train only; 100-sample stratified train subsample; full test set)
python -m qml_noise.extensions breast-cancer --mode full    # -> results/extension_breast_cancer_*.jsonl/.csv
```

Both long runs append per record and **resume** if you rerun the same
command. Options: `--seeds 0,1`, `--maxiter N`, `--force`; Breast Cancer also
`--train-samples N`. `smoke` mode = 1 seed, 20 iterations.

## Plots and report

```powershell
python -m qml_noise.plotting --noise-sweep-seeds 0,1,2,3,4 --breast-cancer-seeds 0,1,2,3,4
# draws every plot the saved results support
```

Reads only `results/*.jsonl`; no circuit is run. All 11 plots (1-8, the circuit
diagram, and both `extension_*.png`) are generated from the current results,
all at 5 seeds. `07_standard_vs_adaptive.png` reads `results/an_vqc_results.jsonl`
and `results/an_vqc_classical_results.jsonl` (3 seeds).
`--noise-sweep-seeds`/`--breast-cancer-seeds` only affect each extension's
completeness check; omit them to default to all of `config.RANDOM_SEEDS`.

## Quantum classifier

Angle encoding (`H` + `RY(x_i)`), `depth` repeated blocks of trainable `RY`/`RZ`
plus a `CNOT` ring, all qubits measured, parity readout. Noise attaches to
transpiled `sx`/`x`/`cx` and `measure`; `rz` is error-free (virtual frame
change). See `src/qml_noise/vqc.py`, `circuits.py`, `noise.py`.

## Noise levels (`config.NOISE_PROFILES`)

`low` and `medium` are **assignment-specified** -- taken directly from the
assignment's noise-level guideline table, not chosen by this project:

| Profile | Single-qubit | Two-qubit | Readout | Source |
|---|---:|---:|---:|---|
| low | 0.1% | 0.5% | 1% | assignment table, Level 1 |
| medium | 1% | 2% | 3% | assignment table, Level 2 |
| high (optional) | 3% | 5% | 5% | chosen within the table's stated "3-5% gate error" range for the optional Level 3 tier |

`low`/`medium` were corrected to these exact values on 2026-10-11 (previous
values were chosen defaults, since no guideline table had been supplied at
the time). Any saved result row evaluated under `"low"` or `"medium"` from
before that date used the old probabilities; see `PROGRESS.md` for which
files that affects and the exact regeneration commands.

## Requirement compliance checklist

Status key: **PASS** verified complete · **PARTIAL** implemented but incompletely evaluated ·
**FAIL** attempted and unsuccessful · **NOT RUN** not executed. Checked against `task1.md`,
`task2.md`, and the assignment's noise-level guideline table. No experiment was run to produce this
checklist -- every row below is read off already-saved code and result files.

### Mandatory tasks (task1.md Section 2)

| # | Requirement | Status | Module | Output |
|---|---|---|---|---|
| 1 | Implement a QML classifier using Qiskit | PASS | `circuits.py`, `vqc.py` | `results/vqc_experiment_results.jsonl` |
| 2 | Establish a noiseless baseline | PASS | `evaluation.py` | ideal-condition rows of `results/vqc_experiment_results.jsonl` |
| 3 | Introduce bit-flip, phase-flip, or depolarizing noise (readout optional) | PASS | `noise.py` (`build_profile_noise_model`: depolarizing + readout) | `low`/`medium` rows of `results/vqc_experiment_results.jsonl` |
| 3a | Isolated bit-flip / phase-flip channels specifically | PARTIAL | `noise.py` (`build_bit_flip_noise_model`, `build_phase_flip_noise_model`), unit-tested in `tests/test_noise.py` | implemented and tested only -- no experiment result file uses these channels; only the combined ideal/low/medium profiles were run |
| 4 | Compare classification accuracy and output distributions | PASS | `metrics.py` (`total_variation_distance`, `hellinger_distance`) | `results/vqc_experiment_results.jsonl` (`distribution_total_variation_distance`, `distribution_hellinger_distance`); `plots/02_accuracy_vs_depth.png`, `plots/05_output_distributions.png` |
| 5 | Analyze noise sensitivity | PASS | `metrics.py` (`noise_sensitivity`, `accuracy_degradation`, `relative_degradation`, `noise_robustness_ratio`) | `report/report.md` Sections 9-10 |
| 6 | Conduct at least six controlled experiments | PASS | `experiments.py` (E1-E8) | 8 experiments x 5 seeds = 40 runs, `results/vqc_experiment_results.jsonl` |
| 7 | Vary at least two experimental parameters | PASS | same | qubits (2/4/6), depth (2/4), and noise (ideal/low/medium) all varied |
| 8 | Use 1024 shots for measurement-based experiments | PASS | `config.py` (`SHOTS = 1024`) | every row's `shots` field in `results/*.jsonl` |
| 9 | Record qubit count and circuit depth | PASS | `circuits.py` (`circuit_resources`) | `n_qubits`, `logical_depth`, `transpiled_depth` columns |
| 10 | Compare results in tables and graphs | PASS | `plotting.py`, `report.md` | `plots/*.png`, `plots/noise_comparison_table.csv`/`.md` |
| 11 | Identify limitations and propose a research improvement | PASS | `report.md` Section 11 (Limitations), Section 12 (AN-VQC) | `report/report.md` |

### Expected results (task1.md Section 2)

| Requirement | Status | Module | Output |
|---|---|---|---|
| Accuracy-degradation graphs | PASS | `plotting.py` (`plot_degradation_vs_noise`) | `plots/01_degradation_vs_noise.png` |
| A noise-comparison table | PASS | `plotting.py` | `plots/noise_comparison_table.csv`/`.md` |
| Circuit-depth analysis | PASS | `plotting.py` (`plot_accuracy_vs_depth`) | `plots/02_accuracy_vs_depth.png` |
| Suggestions for improving noise resilience | PASS | `report.md` Section 10 | `report/report.md`, "Suggestions for improving noise resilience" |
| A classical baseline and ideal quantum baseline | PASS | `classical.py`; ideal rows of `evaluation.py` output | `results/classical_baseline_results.jsonl`; ideal rows of `results/vqc_experiment_results.jsonl` |
| Noise sensitivity analysis across models or configurations | PASS | `extensions.py` (`pooled-analysis`) | `results/extension_pooled_analysis.json`, `report/report.md` Section 9 |

### Required experimental parameters and matrix (task1.md Sections 6-7)

| Requirement | Status | Module | Output |
|---|---|---|---|
| Qubits 2, 4, 6; depth 2, 4; noise ideal/low/medium; 1024 shots | PASS | `experiments.py`, `config.py` | `results/vqc_experiment_results.jsonl` |
| Exact recommended 8-experiment matrix (E1-E8) | PASS | `experiments.py` | 8/8 experiments x 5 seeds (0-4) = 40 runs, `results/vqc_experiment_results.jsonl` |
| Full grid: all 6 (qubits, depth) configurations under all 3 noise levels (18 cells) | **NOT RUN** | `experiments.py` supports arbitrary `(qubits, depth, noise)` combinations, but only the 8 assignment-mandated cells were ever executed | only 8 of 18 possible `(n_qubits, logical_depth, noise_condition)` combinations appear in `results/vqc_experiment_results.jsonl`; e.g. 6-qubit depth-2, and every low/medium row outside 4-qubit depth-4 and 6-qubit depth-4, were never run |
| Optional 8-qubit / depth-6 configurations | **NOT RUN** | `config.py` (`OPTIONAL_*` settings) accepts these values | no result row uses `n_qubits=8` or `logical_depth=6` |
| Five random seeds where computationally feasible | PASS (main matrix + both extensions); PARTIAL (AN-VQC) | `config.py` (`RANDOM_SEEDS = [0,1,2,3,4]`; `AN_VQC_MODE_SEEDS["full"] = [0,1,2]`) | main matrix and both extensions: 5 seeds each; AN-VQC: 3 seeds (six candidates are trained and validated per seed, which is why it was not extended to 5) |

### Technology stack (task1.md Section 3)

| Requirement | Status | Module | Output |
|---|---|---|---|
| Qiskit exclusively for quantum circuits/simulation; no PennyLane/Cirq | PASS | all of `src/qml_noise/` | `requirements.txt` |
| Qiskit Aer as simulator | PASS | `noise.py` (`AerSimulator`) | `requirements.txt` |
| Qiskit Machine Learning | **PARTIAL** | `qiskit-machine-learning` is installed and import-checked for compatibility, but the VQC itself is built directly on Qiskit/Aer primitives (`vqc.py`, `circuits.py`) rather than `qiskit_machine_learning.algorithms.VQC`, for full control over the readout-to-class mapping, the 1024-shot budget, and swapping the noise model between train-once and evaluate-many -- reason documented in full | `requirements.txt`; `report/report.md` Section 5, "Implementation note" |
| NumPy, Pandas, scikit-learn, Matplotlib, pytest | PASS | used throughout | `requirements.txt` |

### Datasets (task1.md Section 4)

| Requirement | Status | Module | Output |
|---|---|---|---|
| Iris binary (setosa vs. versicolor) as primary dataset | PASS | `datasets.py` (`load_iris_binary`) | `results/vqc_experiment_results.jsonl` |
| Breast Cancer Wisconsin Diagnostic as secondary dataset | PASS, 5 seeds | `datasets.py` (`load_breast_cancer_reduced`) | `results/extension_breast_cancer_results.jsonl` |
| Wine as secondary dataset | **NOT RUN** | `datasets.py` (`load_wine_reduced`) -- loader implemented and unit-tested | `tests/test_datasets.py`; no result file exists, Breast Cancer was used as the secondary dataset instead |
| Stratified 80:20 split, fixed seeds, identical splits for comparable models | PASS | `datasets.py` | `results/extension_split_check.json` |
| Preprocessing fit on training data only; test labels never used for tuning | PASS | `datasets.py` (`StandardScaler`/PCA fit on train split only) | `results/extension_split_check.json`, `tests/test_datasets.py` |

### QML architecture (task1.md Section 5)

| Requirement | Status | Module | Output |
|---|---|---|---|
| Angle encoding, trainable RY/RZ, CNOT entangling gates, documented readout, classical optimizer, suitable loss, reproducible init | PASS | `circuits.py`, `vqc.py` | `report/report.md` Section 5; `plots/circuit_4q_depth2.png` |
| Model actually trains and predicts (not random/untrained) | PASS | `vqc.py` | `training_time_seconds`, `n_training_iterations`, `initial_loss`/`final_loss` columns of `results/vqc_experiment_results.jsonl` |

### Noise models (task1.md Section 8)

| Requirement | Status | Module | Output |
|---|---|---|---|
| Ideal / low / medium combined profiles | PASS | `noise.py` (`build_profile_noise_model`) | `results/vqc_experiment_results.jsonl` |
| Bit-flip, phase-flip, depolarizing, readout as isolated channels | PARTIAL | `noise.py` (`build_bit_flip_noise_model`, `build_phase_flip_noise_model`, `build_depolarizing_noise_model`, `build_readout_noise_model`) | implemented and unit-tested (`tests/test_noise.py`); not used in any reported experiment |
| Assignment noise levels used as specified, not chosen defaults | PASS | `config.py` (`NOISE_PROFILES`) | `report/report.md` Section 6 |

### Classical baselines and metrics (task1.md Sections 9-10)

| Requirement | Status | Module | Output |
|---|---|---|---|
| Logistic Regression, SVM baselines | PASS | `classical.py` | `results/classical_baseline_results.jsonl`, `results/an_vqc_classical_results.jsonl` |
| Accuracy, precision, recall, F1, confusion matrix, training time | PASS | `classical.py`, `metrics.py` | same files |
| Accuracy degradation (pp), relative degradation (%), robustness ratio | PASS | `metrics.py` (`accuracy_degradation`, `relative_degradation`, `noise_robustness_ratio`) | `degradation_pp`, `relative_degradation_pct`, `robustness_ratio` columns |
| Robustness percentage (100 x R) per the guideline table | PASS | derived from `metrics.py` (`noise_robustness_ratio`) | `report/report.md` Section 8 |
| Qubit count, logical/transpiled depth, gate counts, parameter count, shots, runtime | PASS | `circuits.py`, `vqc.py` | `n_qubits`, `logical_depth`, `transpiled_depth`, `total_gate_count`, `two_qubit_gate_count`, `n_trainable_parameters`, `shots`, `runtime_seconds` columns |
| Safe zero-denominator handling; noisy accuracy not assumed lower | PASS | `metrics.py` | `tests/test_metrics.py` |

### Proposed improvement: AN-VQC (task1.md Section 11)

| Requirement | Status | Module | Output |
|---|---|---|---|
| Adaptive circuit-selection framework (validation accuracy, robustness, cost) | PASS | `an_vqc.py` | `results/an_vqc_results.jsonl`, `results/an_vqc_selection_log.jsonl` |
| Compared against fixed VQC, ideal-validation-best, shallow baseline, Logistic Regression, SVM | PASS | `an_vqc.py` | `results/an_vqc_results.jsonl`, `results/an_vqc_classical_results.jsonl`; `plots/07_standard_vs_adaptive.png` |
| No test-set leakage in selection | PASS | `an_vqc.py` (selection uses only the validation split) | `results/an_vqc_selection_log.jsonl` |
| Honest reporting of whether it improves results | PASS | `report.md` Section 12 reports AN-VQC did **not** improve on the simpler ideal-validation-best baseline | `report/report.md` Section 12 |
| Seeds | **PARTIAL (3 seeds, not 5)** | `config.py` (`AN_VQC_MODE_SEEDS["full"] = [0, 1, 2]`) | `results/an_vqc_results.jsonl` -- reason documented in `report/report.md` Limitation 9 |

### Project structure, plots, documentation (task1.md Sections 12-14; task2.md Section 8)

| Requirement | Status | Module | Output |
|---|---|---|---|
| Dataset/preprocessing, classical, circuit/VQC, noise, training/eval, adaptive-selection, experiment-runner, plotting modules; central config; tests; README; requirements.txt | PASS | see "Project structure" above | `src/qml_noise/*.py`, `tests/*.py`, this file, `requirements.txt` |
| Smoke / normal / full modes; incremental save; resume after interruption | PASS | `experiments.py`, `extensions.py` | `--mode`/`--force` flags (see commands above) |
| 8 required plots (degradation, depth, noise-comparison, qubits-vs-performance, distributions, classical-vs-QML, standard-vs-adaptive, mean+std) | PASS | `plotting.py` | `plots/01_degradation_vs_noise.png` through `plots/08_accuracy_mean_std.png` |
| Architecture diagram | PASS | `plotting.py` (`plot_circuit_diagram`) | `plots/circuit_4q_depth2.png` |
| Research report (all required sections) | PASS | -- | `report/report.md` |
| Reproducibility record: software versions, seeds, dataset source, preprocessing, noise parameters | PASS | -- | `report/report.md` Section 15 |
| Requirement-compliance checklist | PASS | -- | this section |

### Final verification (task2.md Section 9)

| Requirement | Status | Module | Output |
|---|---|---|---|
| Project installs; primary Iris pipeline runs; VQC trains and predicts | PASS | -- | `results/vqc_experiment_results.jsonl` |
| Noise model can be enabled; results saved incrementally; experiments resume | PASS | `experiments.py`, `extensions.py` | incremental JSONL append + resume logic |
| Plots regenerate from saved results only | PASS | `plotting.py` | `python -m qml_noise.plotting ...` (no circuit execution) |
| Automated test suite | **not re-run this turn** (see "Tests" above) | `tests/*.py` | last known collected count and pass/fail status are whatever the most recent `pytest` run reported, not re-verified here |
