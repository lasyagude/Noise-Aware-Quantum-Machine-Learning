# Noise-Aware Quantum Machine Learning

A small quantum classifier, built with [Qiskit](https://www.ibm.com/quantum/qiskit) and tested to
see how it holds up when realistic hardware noise is added — plus a simple follow-up question:
can automatically picking a smarter circuit make it more noise-resistant?

## What's done

- A real, trainable quantum classifier (VQC) — it learns from data and makes predictions, it's
  not a random or untrained circuit.
- The full experiment matrix the assignment asked for: 2, 4, and 6 qubits, shallow and deep
  circuits, three noise levels (none / low / medium), 5 random seeds each — 40 runs in total,
  1024 measurement shots per run.
- Classical baseline comparisons (Logistic Regression, SVM) on the same data.
- Two extra experiments beyond the minimum: a finer-grained noise sweep, and the same pipeline
  run on a second, real-world medical dataset (Breast Cancer Wisconsin).
- A proposed improvement — automatically selecting the circuit expected to handle noise best —
  tested honestly against simpler baselines. It didn't actually win; that negative result is
  reported as-is rather than hidden.
- 13 result plots, a full written report, and 244 automated tests, all currently passing.

For the exact, requirement-by-requirement status of everything above, see
[COMPLIANCE.md](COMPLIANCE.md). For the full write-up — methodology, results, discussion,
limitations — see [report/report.md](report/report.md).

## Getting started

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"

pytest                          # run the test suite (244 tests, a few minutes)
python -m qml_noise.plotting    # regenerate all plots from the already-saved results (fast, no training)
```

Re-running the actual experiments from scratch takes longer (the full matrix takes about half an
hour) — see [`src/qml_noise/README.md`](src/qml_noise/README.md) for those commands.

## Project layout

Each folder below has its own short README explaining what's inside it:

| Folder | What's in it |
|---|---|
| [`src/qml_noise/`](src/qml_noise/README.md) | The actual code — the quantum classifier, noise models, experiment runner, plotting |
| [`tests/`](tests/README.md) | Automated tests |
| [`results/`](results/README.md) | Saved experiment output (the raw numbers behind every plot and claim) |
| [`plots/`](plots/README.md) | The generated charts and circuit diagrams |
| [`report/`](report/README.md) | The full written report |
| [`scripts/`](scripts/README.md) | A one-off data-cleanup script |
