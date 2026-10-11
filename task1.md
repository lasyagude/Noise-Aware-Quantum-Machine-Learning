# AUTONOMOUS TASK 1/2: BUILD THE COMPLETE NOISE-AWARE QML PROJECT

Act as a senior Quantum Machine Learning researcher and Python engineer. Work directly in the current repository. Build a complete, runnable project for the assignment below. Do not stop after presenting a plan, and do not ask me questions. Make sensible decisions, implement everything, run tests, and fix errors.

## 1. Project title

**Noise-Aware Quantum Machine Learning: Experimental Analysis and Adaptive Circuit Selection for Robust Classification**

## 2. Exact assignment requirements

Research problem: Investigate how different noise conditions affect quantum machine-learning classification performance.

Research objective: Measure noise sensitivity across quantum classifiers, circuit configurations, and noise conditions, then propose and evaluate a research improvement.

Mandatory tasks:
1. Implement a QML classifier using Qiskit.
2. Establish a noiseless baseline.
3. Introduce bit-flip, phase-flip, or depolarizing noise. Readout error may also be investigated.
4. Compare classification accuracy and output distributions.
5. Analyze noise sensitivity.
6. Conduct at least six controlled experiments.
7. Vary at least two experimental parameters.
8. Use 1024 shots for measurement-based experiments.
9. Record qubit count and circuit depth.
10. Compare results in tables and graphs.
11. Identify limitations and propose a research improvement.

Expected results:
- Accuracy-degradation graphs.
- A noise-comparison table.
- Circuit-depth analysis.
- Suggestions for improving noise resilience.
- A classical baseline and ideal quantum baseline.
- Noise sensitivity analysis across models or configurations.

Use five random seeds where computationally feasible.

## 3. Fixed technology stack

Use Python with:
- Qiskit.
- Qiskit Aer.
- Qiskit Machine Learning.
- NumPy.
- Pandas.
- scikit-learn.
- Matplotlib.
- pytest.

**Use Qiskit exclusively for quantum circuits, quantum algorithms, and quantum simulation.** Do not use PennyLane, Cirq, or alternative quantum frameworks. Do not require paid quantum hardware or API keys.

Use Qiskit Aer as the simulator. Inspect the installed environment, select compatible package versions, and create requirements.txt. Do not blindly install incompatible latest versions.

## 4. Datasets

Support these datasets using scikit-learn's built-in dataset loaders:
- Iris: 4 features, 3 original classes.
- Wine: 13 features, 3 original classes.
- Breast Cancer Wisconsin Diagnostic: 30 features, 2 classes.

Use Iris binary classification as the primary experiment, selecting Iris setosa and Iris versicolor. Support reduced-feature Wine and Breast Cancer experiments as secondary options.

Use a stratified 80:20 train/test split, fixed seeds, and identical splits for comparable models. Fit preprocessing only on training data. Use training/validation data for model selection, and never use test labels for tuning.

## 5. QML architecture

Implement a trainable Variational Quantum Classifier (VQC) using Qiskit.

Use:
- Angle encoding.
- Trainable RY/RZ rotations.
- CNOT entangling gates.
- A documented binary measurement/readout strategy.
- A classical optimizer.
- A suitable classification loss.
- Reproducible parameter initialization.

The model must actually train and produce predictions. Do not substitute random predictions or an untrained circuit.

Document the feature encoding, loss, optimizer, initialization, stopping criteria, parameter count, and measurement-to-class mapping.

## 6. Required experimental parameters

Primary required configurations:
- Qubits: 2, 4, and 6.
- Circuit depth: 2 and 4.
- Noise conditions: ideal, low, and medium.
- Shots: exactly 1024 for required shot-based evaluations.

Define logical ansatz depth as the number of repeated ansatz blocks. Depth 2 means two repetitions; depth 4 means four repetitions. Also record actual transpiled circuit depth separately.

The fixed-parameter guidance additionally recommends qubits 2, 4, 6, 8 and depths 2, 4, 6. Implement configuration support for these optional settings, but prioritize the required 2/4/6 qubits and depth 2/4 experiments. Treat 8 qubits and depth 6 as optional extensions, not mandatory blockers.

## 7. Implement the exact recommended eight-experiment matrix

Run this matrix first:

| Experiment | Qubits | Depth | Noise |
|---|---:|---:|---|
| E1 | 2 | 2 | Ideal |
| E2 | 2 | 4 | Ideal |
| E3 | 4 | 2 | Ideal |
| E4 | 4 | 4 | Ideal |
| E5 | 4 | 4 | Low |
| E6 | 4 | 4 | Medium |
| E7 | 6 | 4 | Ideal |
| E8 | 6 | 4 | Medium |

This is the assignment's recommended matrix. Do not replace it with a different matrix.

After completing it, optionally expand to additional noise/depth combinations to analyze interactions more thoroughly.

## 8. Noise models

Implement configurable Qiskit Aer noise models for:
- Ideal/no noise.
- Low noise.
- Medium noise.
- Bit-flip.
- Phase-flip.
- Depolarizing noise.
- Optional readout error.

The supplied assignment screenshots do not specify numerical probabilities for low and medium noise. Therefore, make the values configurable in a central configuration file, select and document sensible simulation defaults, and clearly distinguish these chosen defaults from the assignment's mandatory parameters.

Apply noise consistently to the intended gate instructions. Document how the model is constructed, including single-qubit and two-qubit gate errors, readout errors, and transpilation assumptions.

Run isolated-channel experiments separately from combined noise profiles where feasible. Optional thermal relaxation can be added after mandatory experiments work.

## 9. Classical baselines

Implement:
- Logistic Regression.
- Support Vector Machine.

Use the same train/test split and appropriate preprocessing. Record accuracy, precision, recall, F1-score, confusion matrix, and training time.

## 10. Metrics

Calculate:
- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Accuracy degradation relative to the ideal baseline.
- Relative accuracy degradation.
- Noise robustness ratio.
- Number of qubits.
- Logical ansatz depth.
- Transpiled circuit depth.
- Total gate count.
- Two-qubit gate count.
- Number of trainable parameters.
- Shots.
- Runtime.

Define accuracy degradation in percentage points as:

D = 100 × (A_ideal − A_noisy)

Define relative degradation as:

D_relative = 100 × (A_ideal − A_noisy) / A_ideal

Define the noise robustness ratio as:

R = A_noisy / A_ideal

Handle zero denominators safely and document the metric conventions. Do not assume noisy accuracy must always be lower than ideal accuracy.

Compare measurement output distributions as well as classification accuracy, particularly for ideal and noisy versions of the same circuit.

## 11. Proposed research improvement: Adaptive Noise-Aware VQC

Implement a lightweight adaptive circuit-selection framework.

Candidate circuits should include the required 2-, 4-, and 6-qubit configurations where feasible. Select among candidates using validation accuracy, estimated robustness under a target noise profile, and resource cost.

Use a documented objective such as:

Score(c) = α × ValidationAccuracy(c) + β × Robustness(c) − γ × NormalizedCost(c)

Document the weights, normalization, calibration procedure, and selection assumptions.

Compare the proposed method against:
- A fixed VQC configuration.
- The best circuit chosen using ideal validation accuracy.
- A shallow-circuit baseline.
- Logistic Regression.
- SVM.

Never use test-set information to select a circuit or tune the method. The proposed method is a research hypothesis; report honestly whether it improves results.

## 12. Project structure and implementation

Create a modular project containing:
- dataset and preprocessing module.
- classical baseline module.
- quantum circuit and VQC module.
- noise-model module.
- training and evaluation module.
- adaptive-selection module.
- experiment runner.
- plotting and report generation.
- central configuration file.
- automated tests.
- README and requirements.txt.
- results, plots, and report directories.

Use clean code, logging, reproducible seeds, informative errors, and incremental result saving.

Implement a smoke-test mode, a normal experiment mode, and a full experiment mode. Preserve completed results and allow interrupted experiments to resume.

## 13. Plots and result files

Generate actual data-driven plots:
1. Accuracy degradation versus noise level.
2. Accuracy versus circuit depth.
3. Noise comparison table/plot.
4. Qubit count versus performance.
5. Output-distribution comparison under ideal and noisy conditions.
6. Classical versus ideal QML versus noisy QML.
7. Standard VQC versus adaptive VQC.
8. Mean and standard deviation across seeds where feasible.

Save experiment data in CSV or JSON and plots as PNG files.

Never use fabricated measurements or placeholder numbers in final result plots.

## 14. Research documentation

Create a report in Markdown containing:
- Abstract.
- Problem statement.
- Objectives and research questions.
- Related work and research gap.
- Dataset and preprocessing.
- QNN architecture and circuit design.
- Noise methodology.
- Experimental matrix.
- Metrics.
- Results and graphs.
- Discussion.
- Limitations.
- Proposed improvement.
- Conclusion and future work.
- References.

Use verified references; do not invent citations or literature findings. Base conclusions on actual saved results.

## 15. Execution rules

Immediately inspect the repository and environment, then implement the project. Do not return a planning-only answer.

Run tests, debug failures, and execute the eight-experiment matrix where feasible. Prioritize the required experiments over optional extensions. Use five seeds where computationally feasible and report actual repetitions.

Do not claim that experiments, tests, or reports succeeded unless they actually did. If an experiment cannot complete within the available execution budget, save partial results, mark its status, and document the exact command needed to resume it.

At the end of this task, leave the repository in a runnable state and provide exact execution commands. Do not ask for approval before proceeding.

## ADDENDUM (supersedes section 8 on noise values)

The assignment guideline table DOES specify noise levels. Use these as assignment-specified values, not chosen defaults:
- Level 0: ideal, no noise.
- Level 1 (low): 0.1% single-qubit error, 0.5% two-qubit error, 1% readout error.
- Level 2 (moderate/medium): 1% single-qubit, 2% two-qubit, 3% readout error.
- Level 3 (optional high): 3-5% gate error, higher readout error (stress test).
- Noise model: depolarizing error + readout error; thermal relaxation optional if documented.
Other guideline items to follow: report "robustness percentage" (100 x R); record software versions, seeds, dataset source, preprocessing and noise parameters; primary dataset Iris binary, secondary WDBC reduced (Wine optional); minimum experiment set is 2 qubit/depth settings x 3 noise levels.
Wherever task1.md or task2.md says the noise probabilities are "chosen defaults", read it as "assignment-specified".

## ADDENDUM STATUS: implemented (2026-10-11)

The noise-level correction above has been applied in code, not just noted here:
- `config.NOISE_PROFILES["low"]` = 0.001/0.005/0.01 and `["medium"]` = 0.01/0.02/0.03 (single-qubit/
  two-qubit/readout depolarizing+readout probabilities), matching Level 1/Level 2 exactly.
- Optional `config.NOISE_PROFILES["high"]` = 0.03/0.05/0.05 added (a chosen point inside the stated
  3-5% range; the table gives no exact Level 3 numbers), in `config.OPTIONAL_NOISE_CONDITIONS`, not
  the mandatory `NOISE_CONDITIONS`.
- `config.py` and `src/qml_noise/noise.py` docstrings now describe low/medium as assignment-specified.
- All results computed under the old (pre-correction) probabilities were identified and regenerated:
  `results/vqc_experiment_results.jsonl` (E5/E6/E8), `results/extension_breast_cancer_results.jsonl`
  (BC-E5/E6/E8), and `results/extension_pooled_analysis.json` (derived). The noise-sweep extension was
  never affected (it sweeps its own independent probability list, never reads `NOISE_PROFILES`).
- `report/report.md` reflects the corrected values throughout; see its Section 6 for the exact table
  and `PROGRESS.md` for the verification trail (row counts, duplicate checks, ideal-baseline
  consistency checks).
- Not yet covered by this addendum: the "robustness percentage" (100×R) reporting convention, and
  recording software versions/seeds/dataset source/preprocessing/noise parameters as a formal
  checklist -- both are implicit in the saved result files and `report/report.md` but not yet
  collected into one explicit compliance list.