# `src/qml_noise/`

This is the actual project code. Each file does one job:

| File | What it does |
|---|---|
| `config.py` | Central settings — random seeds, number of measurement shots (1024), noise-level numbers, extension defaults. Change something here and it applies everywhere. |
| `datasets.py` | Loads Iris, Wine, and Breast Cancer data; splits it into train/test the same way every time; scales features using only the training data (so the test set is never "peeked at"). |
| `classical.py` | The classical baselines — Logistic Regression and an SVM — for comparison against the quantum model. |
| `circuits.py` | Builds the quantum circuit itself: how input data is encoded into qubits, the trainable gates, how many gates/parameters it uses. |
| `noise.py` | Builds the simulated hardware noise (ideal / low / medium, plus individual bit-flip, phase-flip, depolarizing, and readout-error channels). |
| `vqc.py` | The trainable quantum classifier — trains the circuit's parameters on data using a classical optimizer. |
| `metrics.py` | All the scoring: accuracy, precision/recall/F1, how much accuracy drops under noise, and how different the noisy output looks from the ideal output. |
| `evaluation.py` | Trains a model once (under ideal conditions), then re-tests the same trained model under each noise level. |
| `experiments.py` | Runs the required 8-experiment matrix (E1–E8) across all 5 seeds, saving results as it goes so a run can be safely interrupted and resumed. |
| `extensions.py` | The two extra experiments (noise sweep, Breast Cancer dataset) — kept separate so they never touch the main required results. |
| `an_vqc.py` | The proposed improvement: automatically picks which circuit to use based on validation accuracy, estimated noise robustness, and cost, then compares it against simpler baselines. |
| `plotting.py` | Generates every chart and diagram from the saved result files. Doesn't run any quantum circuit itself. |

## Running things

```powershell
# the required experiment matrix (2/4/6 qubits, depth 2/4, ideal/low/medium noise, 5 seeds)
python -m qml_noise.experiments --mode smoke    # quick pipeline check, ~1 min
python -m qml_noise.experiments --mode normal   # 1 seed, ~6 min
python -m qml_noise.experiments --mode full     # all 5 seeds, ~28 min

# the two extra experiments
python -m qml_noise.extensions split-check
python -m qml_noise.extensions pooled-analysis
python -m qml_noise.extensions noise-sweep --mode full
python -m qml_noise.extensions breast-cancer --mode full

# regenerate every plot from whatever results are currently saved
python -m qml_noise.plotting --noise-sweep-seeds 0,1,2,3,4 --breast-cancer-seeds 0,1,2,3,4
```

All the long-running commands save results as they go and can be safely re-run — they'll skip
whatever's already finished and pick up where they left off. Add `--force` to start over instead.

## Noise levels used

`low` and `medium` come directly from the assignment's own noise guideline table, not numbers we
picked ourselves:

| Level | Single-qubit error | Two-qubit error | Readout error |
|---|---:|---:|---:|
| low | 0.1% | 0.5% | 1% |
| medium | 1% | 2% | 3% |
| high (optional, not part of the required matrix) | 3% | 5% | 5% |
