# AUTONOMOUS TASK 2/2: FINISH, RUN, VERIFY, AND DELIVER THE COMPLETE PROJECT

Continue working in the current repository from Task 1. You have already been instructed to build the Noise-Aware Quantum Machine Learning project.

**Do not restart the project, repeat the planning phase, or rewrite working code unnecessarily. Your job now is to finish all remaining work and deliver a verified, runnable project. Do not ask me questions or wait for approval.**

## 1. Inspect and complete the implementation

Inspect the existing files, experiment configurations, logs, tests, and saved results.

Identify incomplete modules, failing tests, incorrect Qiskit APIs, invalid noise configurations, data leakage, and missing output files. Fix the problems directly.

Preserve completed work and existing experimental results.

## 2. Execute the mandatory eight-experiment matrix

Run and verify the exact assignment matrix:

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

Use 1024 shots for measurement-based evaluations and five seeds where computationally feasible.

Prioritize this exact matrix over optional experiments. Do not replace it with a different matrix.

The assignment requires at least six controlled experiments and variation of at least two experimental parameters. Record the actual number of experiments and repetitions completed.

If some runs cannot finish within the available execution budget, save their progress, mark them incomplete, and provide exact commands to resume them.

## 3. Verify datasets and technology restrictions

Verify that the project supports:
- Iris binary classification as the primary dataset.
- Wine classification as a secondary option.
- Breast Cancer Wisconsin Diagnostic classification as a secondary option.

Verify that preprocessing is fitted on training data only, comparable models use consistent splits, and the test set is never used for model selection.

Use Qiskit for quantum circuits and Qiskit Aer for simulation. Do not introduce PennyLane, Cirq, or alternative quantum frameworks.

## 4. Validate the QML implementation

Verify that the VQC is actually trainable and produces predictions.

Check:
- Feature encoding.
- Trainable RY/RZ gates.
- CNOT entanglement.
- Measurement-to-class mapping.
- Loss function and optimizer.
- Logical ansatz depth.
- Transpiled circuit depth.
- Gate counts and trainable parameter counts.
- Reproducibility under fixed seeds.

Fix any circuit or training errors before interpreting experimental results.

## 5. Validate noise experiments

Verify ideal, low-noise, and medium-noise configurations.

Verify the supported bit-flip, phase-flip, and depolarizing noise channels. Check readout-error implementation if enabled.

Confirm that gate-specific errors are applied consistently and that the transpilation and noise-model assumptions are documented.

Use configurable noise probabilities and explicitly distinguish chosen simulation defaults from values supplied by the assignment.

## 6. Verify all evaluation metrics

Ensure that the result files contain, where applicable:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Output measurement distributions.
- Accuracy degradation relative to ideal.
- Relative degradation.
- Noise robustness ratio.
- Qubit count.
- Logical ansatz depth.
- Transpiled depth.
- Total gate count.
- Two-qubit gate count.
- Trainable parameter count.
- Shots.
- Runtime.

Use correct equations, consistent units, and safe handling of zero denominators.

Do not assume that measured noisy accuracy must always be lower than ideal accuracy.

## 7. Finish and evaluate the proposed improvement

Complete the Adaptive Noise-Aware VQC (AN-VQC) circuit-selection framework.

The method must select candidates using validation accuracy, estimated robustness under the target noise profile, and circuit-resource cost.

Compare it against:
1. A fixed VQC.
2. The best circuit selected by ideal validation accuracy.
3. A shallow-circuit baseline.
4. Logistic Regression.
5. SVM.

Prevent test-set leakage.

Report whether AN-VQC actually improves the measured objective. Include negative or inconclusive findings where applicable. Do not invent a successful result.

## 8. Generate the final research outputs

Generate or update the following using actual experimental data:

- CSV/JSON experiment results.
- Accuracy degradation versus noise plots.
- Accuracy versus circuit depth plots.
- Noise-comparison tables.
- Output-distribution comparisons.
- Classical versus ideal/noisy QML comparisons.
- Standard VQC versus AN-VQC comparisons.
- Mean and standard deviation across seeds where available.
- Architecture diagram.
- Research report in Markdown.
- README with installation, execution, and resumption instructions.
- requirements.txt.
- Automated tests.
- Requirement-compliance checklist.

The report must include the problem statement, objectives, related work, research gap, methodology, datasets, QNN architecture, noise models, experiment matrix, metrics, actual results, discussion, limitations, proposed improvement, conclusion, future work, and references.

Use verified literature sources. Do not invent citations, measurements, graphs, or conclusions.

## 9. Final verification

Run the smoke test and automated tests. Fix failures where possible.

Verify that:
- The project can be installed.
- The primary Iris pipeline runs.
- The VQC trains and predicts.
- The noise model can be enabled.
- Results are saved incrementally.
- Experiments can resume after interruption.
- Plots can be regenerated from saved results.
- The report reflects actual experimental findings.
- Generated files exist in the expected directories.

In README.md, create a compliance checklist using these statuses:

- PASS: verified complete.
- PARTIAL: implemented but incompletely evaluated.
- FAIL: attempted and unsuccessful.
- NOT RUN: not executed.

For each item, identify the relevant module or output file.

## 10. Final response requirements

Do not respond with another plan. Finish implementation and verification within the available execution budget.

Your final response must state:

1. What is implemented.
2. Exact commands to install, test, and run the project.
3. Which experiments completed and how many seeds were used.
4. Actual results available, without fabricated numbers.
5. Which tests passed and failed.
6. Locations of result files, plots, and report.
7. Whether AN-VQC improved performance.
8. Remaining limitations and incomplete experiments.
9. Exact commands for resuming incomplete experiments.

If execution limits prevent complete evaluation, leave a runnable, resumable project and clearly report what remains. Never claim that an experiment or test succeeded unless it actually did.

**Begin by inspecting the current workspace and continue directly from the existing implementation. Finish the project now.**