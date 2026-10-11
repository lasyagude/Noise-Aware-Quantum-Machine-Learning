# Noise-Aware Quantum Machine Learning

Noise-Aware Quantum Machine Learning: Experimental Analysis and Adaptive Circuit
Selection for Robust Classification.

> **Status:** the VQC, noise models, metrics, experiment runner, and the full
> E1-E8 x 5-seed Iris matrix are done and saved in `results/`. Plotting code,
> the 4-qubit circuit diagram, and a **draft report** (`report/report.md`) exist;
> extension runs (noise sweep, Breast Cancer) were still running when the draft
> was written, and adaptive circuit selection (AN-VQC) is **not implemented**.
> The report marks every unfinished part with `PLACEHOLDER`. See `PROGRESS.md`.

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
pytest          # 205 tests collected (training tests make a full run take several minutes)
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
  plotting.py     all plots, from saved results only (skips incomplete/missing data)
tests/            205 tests
results/          real experiment output (JSONL/CSV/JSON); results/smoke_checks/ = pipeline checks only
plots/            PNG figures + noise_comparison_table.csv/.md (regenerate: see below)
report/           report.md (draft; PLACEHOLDERs mark unfinished parts)
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
python -m qml_noise.plotting     # draws every plot the saved results support
```

Reads only `results/*.jsonl`; no circuit is run. Plots 1-6 and 8 and the
circuit diagram (`plots/circuit_4q_depth2.png`, needs `pylatexenc`) are
generated. `extension_*.png` are drawn only once the extension result files
are **complete** (all experiment/seed pairs present); `07_standard_vs_adaptive.png`
needs `results/adaptive_selection_results.jsonl` (provisional schema:
`method`, `seed`, `accuracy`), which does not exist until AN-VQC is built.
Re-run the command after the extension runs finish.

## Quantum classifier

Angle encoding (`H` + `RY(x_i)`), `depth` repeated blocks of trainable `RY`/`RZ`
plus a `CNOT` ring, all qubits measured, parity readout. Noise attaches to
transpiled `sx`/`x`/`cx` and `measure`; `rz` is error-free (virtual frame
change). See `src/qml_noise/vqc.py`, `circuits.py`, `noise.py`.
