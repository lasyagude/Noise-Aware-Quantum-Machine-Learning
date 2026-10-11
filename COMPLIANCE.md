# Requirement compliance checklist

A detailed, line-by-line check of this project against the assignment's requirements and its
noise-level guideline table. This is the technical backup for the plain-language summary in
[README.md](README.md) — useful if you want to verify a specific claim rather than take the
README's word for it.

Status key: **PASS** verified complete · **PARTIAL** implemented but incompletely evaluated ·
**FAIL** attempted and unsuccessful · **NOT RUN** not executed. No experiment was run to produce
this checklist — every row below is read off already-saved code and result files.

### Mandatory tasks

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

### Expected results

| Requirement | Status | Module | Output |
|---|---|---|---|
| Accuracy-degradation graphs | PASS | `plotting.py` (`plot_degradation_vs_noise`) | `plots/01_degradation_vs_noise.png` |
| A noise-comparison table | PASS | `plotting.py` | `plots/noise_comparison_table.csv`/`.md` |
| Circuit-depth analysis | PASS | `plotting.py` (`plot_accuracy_vs_depth`) | `plots/02_accuracy_vs_depth.png` |
| Suggestions for improving noise resilience | PASS | `report.md` Section 10 | `report/report.md`, "Suggestions for improving noise resilience" |
| A classical baseline and ideal quantum baseline | PASS | `classical.py`; ideal rows of `evaluation.py` output | `results/classical_baseline_results.jsonl`; ideal rows of `results/vqc_experiment_results.jsonl` |
| Noise sensitivity analysis across models or configurations | PASS | `extensions.py` (`pooled-analysis`) | `results/extension_pooled_analysis.json`, `report/report.md` Section 9 |

### Required experimental parameters and matrix

| Requirement | Status | Module | Output |
|---|---|---|---|
| Qubits 2, 4, 6; depth 2, 4; noise ideal/low/medium; 1024 shots | PASS | `experiments.py`, `config.py` | `results/vqc_experiment_results.jsonl` |
| Exact recommended 8-experiment matrix (E1-E8) | PASS | `experiments.py` | 8/8 experiments x 5 seeds (0-4) = 40 runs, `results/vqc_experiment_results.jsonl` |
| Full grid: all 6 (qubits, depth) configurations under all 3 noise levels (18 cells) | **NOT RUN** | `experiments.py` supports arbitrary `(qubits, depth, noise)` combinations, but only the 8 assignment-mandated cells were ever executed | only 8 of 18 possible `(n_qubits, logical_depth, noise_condition)` combinations appear in `results/vqc_experiment_results.jsonl`; e.g. 6-qubit depth-2, and every low/medium row outside 4-qubit depth-4 and 6-qubit depth-4, were never run |
| Optional 8-qubit / depth-6 configurations | **NOT RUN** | `config.py` (`OPTIONAL_*` settings) accepts these values | no result row uses `n_qubits=8` or `logical_depth=6` |
| Five random seeds where computationally feasible | PASS (main matrix + both extensions); PARTIAL (AN-VQC) | `config.py` (`RANDOM_SEEDS = [0,1,2,3,4]`; `AN_VQC_MODE_SEEDS["full"] = [0,1,2]`) | main matrix and both extensions: 5 seeds each; AN-VQC: 3 seeds (six candidates are trained and validated per seed, which is why it was not extended to 5) |

### Technology stack

| Requirement | Status | Module | Output |
|---|---|---|---|
| Qiskit exclusively for quantum circuits/simulation; no PennyLane/Cirq | PASS | all of `src/qml_noise/` | `requirements.txt` |
| Qiskit Aer as simulator | PASS | `noise.py` (`AerSimulator`) | `requirements.txt` |
| Qiskit Machine Learning | **PARTIAL** | `qiskit-machine-learning` is installed and import-checked for compatibility, but the VQC itself is built directly on Qiskit/Aer primitives (`vqc.py`, `circuits.py`) rather than `qiskit_machine_learning.algorithms.VQC`, for full control over the readout-to-class mapping, the 1024-shot budget, and swapping the noise model between train-once and evaluate-many -- reason documented in full | `requirements.txt`; `report/report.md` Section 5, "Implementation note" |
| NumPy, Pandas, scikit-learn, Matplotlib, pytest | PASS | used throughout | `requirements.txt` |

### Datasets

| Requirement | Status | Module | Output |
|---|---|---|---|
| Iris binary (setosa vs. versicolor) as primary dataset | PASS | `datasets.py` (`load_iris_binary`) | `results/vqc_experiment_results.jsonl` |
| Breast Cancer Wisconsin Diagnostic as secondary dataset | PASS, 5 seeds | `datasets.py` (`load_breast_cancer_reduced`) | `results/extension_breast_cancer_results.jsonl` |
| Wine as secondary dataset | **NOT RUN** | `datasets.py` (`load_wine_reduced`) -- loader implemented and unit-tested | `tests/test_datasets.py`; no result file exists, Breast Cancer was used as the secondary dataset instead |
| Stratified 80:20 split, fixed seeds, identical splits for comparable models | PASS | `datasets.py` | `results/extension_split_check.json` |
| Preprocessing fit on training data only; test labels never used for tuning | PASS | `datasets.py` (`StandardScaler`/PCA fit on train split only) | `results/extension_split_check.json`, `tests/test_datasets.py` |

### QML architecture

| Requirement | Status | Module | Output |
|---|---|---|---|
| Angle encoding, trainable RY/RZ, CNOT entangling gates, documented readout, classical optimizer, suitable loss, reproducible init | PASS | `circuits.py`, `vqc.py` | `report/report.md` Section 5; `plots/circuit_4q_depth2.png` |
| Model actually trains and predicts (not random/untrained) | PASS | `vqc.py` | `training_time_seconds`, `n_training_iterations`, `initial_loss`/`final_loss` columns of `results/vqc_experiment_results.jsonl` |

### Noise models

| Requirement | Status | Module | Output |
|---|---|---|---|
| Ideal / low / medium combined profiles | PASS | `noise.py` (`build_profile_noise_model`) | `results/vqc_experiment_results.jsonl` |
| Bit-flip, phase-flip, depolarizing, readout as isolated channels | PARTIAL | `noise.py` (`build_bit_flip_noise_model`, `build_phase_flip_noise_model`, `build_depolarizing_noise_model`, `build_readout_noise_model`) | implemented and unit-tested (`tests/test_noise.py`); not used in any reported experiment |
| Assignment noise levels used as specified, not chosen defaults | PASS | `config.py` (`NOISE_PROFILES`) | `report/report.md` Section 6 |

### Classical baselines and metrics

| Requirement | Status | Module | Output |
|---|---|---|---|
| Logistic Regression, SVM baselines | PASS | `classical.py` | `results/classical_baseline_results.jsonl`, `results/an_vqc_classical_results.jsonl` |
| Accuracy, precision, recall, F1, confusion matrix, training time | PASS | `classical.py`, `metrics.py` | same files |
| Accuracy degradation (pp), relative degradation (%), robustness ratio | PASS | `metrics.py` (`accuracy_degradation`, `relative_degradation`, `noise_robustness_ratio`) | `degradation_pp`, `relative_degradation_pct`, `robustness_ratio` columns |
| Robustness percentage (100 x R) per the guideline table | PASS | derived from `metrics.py` (`noise_robustness_ratio`) | `report/report.md` Section 8 |
| Qubit count, logical/transpiled depth, gate counts, parameter count, shots, runtime | PASS | `circuits.py`, `vqc.py` | `n_qubits`, `logical_depth`, `transpiled_depth`, `total_gate_count`, `two_qubit_gate_count`, `n_trainable_parameters`, `shots`, `runtime_seconds` columns |
| Safe zero-denominator handling; noisy accuracy not assumed lower | PASS | `metrics.py` | `tests/test_metrics.py` |

### Proposed improvement: AN-VQC

| Requirement | Status | Module | Output |
|---|---|---|---|
| Adaptive circuit-selection framework (validation accuracy, robustness, cost) | PASS | `an_vqc.py` | `results/an_vqc_results.jsonl`, `results/an_vqc_selection_log.jsonl` |
| Compared against fixed VQC, ideal-validation-best, shallow baseline, Logistic Regression, SVM | PASS | `an_vqc.py` | `results/an_vqc_results.jsonl`, `results/an_vqc_classical_results.jsonl`; `plots/07_standard_vs_adaptive.png` |
| No test-set leakage in selection | PASS | `an_vqc.py` (selection uses only the validation split) | `results/an_vqc_selection_log.jsonl` |
| Honest reporting of whether it improves results | PASS | `report.md` Section 12 reports AN-VQC did **not** improve on the simpler ideal-validation-best baseline | `report/report.md` Section 12 |
| Seeds | **PARTIAL (3 seeds, not 5)** | `config.py` (`AN_VQC_MODE_SEEDS["full"] = [0, 1, 2]`) | `results/an_vqc_results.jsonl` -- reason documented in `report/report.md` Limitation 9 |

### Project structure, plots, documentation

| Requirement | Status | Module | Output |
|---|---|---|---|
| Dataset/preprocessing, classical, circuit/VQC, noise, training/eval, adaptive-selection, experiment-runner, plotting modules; central config; tests; README; requirements.txt | PASS | see `src/qml_noise/README.md` | `src/qml_noise/*.py`, `tests/*.py`, `requirements.txt` |
| Smoke / normal / full modes; incremental save; resume after interruption | PASS | `experiments.py`, `extensions.py` | `--mode`/`--force` flags (see README) |
| 8 required plots (degradation, depth, noise-comparison, qubits-vs-performance, distributions, classical-vs-QML, standard-vs-adaptive, mean+std) | PASS | `plotting.py` | `plots/01_degradation_vs_noise.png` through `plots/08_accuracy_mean_std.png` |
| Architecture diagram | PASS | `plotting.py` (`plot_circuit_diagram`) | `plots/circuit_4q_depth2.png` (+ `circuit_2q_depth2.png`, `circuit_6q_depth4.png`) |
| Research report (all required sections) | PASS | -- | `report/report.md` |
| Reproducibility record: software versions, seeds, dataset source, preprocessing, noise parameters | PASS | -- | `report/report.md` Section 15 |
| Requirement-compliance checklist | PASS | -- | this file |

### Final verification

| Requirement | Status | Module | Output |
|---|---|---|---|
| Project installs; primary Iris pipeline runs; VQC trains and predicts | PASS | -- | `results/vqc_experiment_results.jsonl` |
| Noise model can be enabled; results saved incrementally; experiments resume | PASS | `experiments.py`, `extensions.py` | incremental JSONL append + resume logic |
| Plots regenerate from saved results only | PASS | `plotting.py` | `python -m qml_noise.plotting ...` (no circuit execution) |
| Automated test suite | PASS | `tests/*.py` | 244 passed, 0 failed (verified 2026-10-11, ~15.6 min) |
