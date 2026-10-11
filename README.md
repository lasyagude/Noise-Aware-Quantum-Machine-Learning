# Noise-Aware Quantum Machine Learning

Noise-Aware Quantum Machine Learning: Experimental Analysis and Adaptive Circuit
Selection for Robust Classification.

> **Status: complete.** The VQC, noise models, metrics, experiment runner, the full E1-E8 x 5-seed
> Iris matrix, both extensions (noise sweep and Breast Cancer, 5 seeds each), AN-VQC (3 seeds), all
> plots, and `report/report.md` are done. No `PLACEHOLDER` markers remain in the report. See
> `PROGRESS.md` for details and verification notes.

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
report/           report.md (complete, no PLACEHOLDERs remaining)
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
