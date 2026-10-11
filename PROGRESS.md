# Progress — Noise-Aware QML Project

| Stage | Sections | Commit | Status |
|---|---|---|---|
| 1–2 | env, config, datasets, classical, VQC, noise, metrics | `9fd7d71`, `8c0742c` | done |
| 3 | runner + full E1–E8 × 5 seeds (Iris) | `b65bf6a`, `47ec9aa` | done |
| 5 | EXTENSIONS: split check, pooled stats, noise sweep, Breast Cancer | uncommitted | code + tests done; **long runs pending (yours)** |
| 4 | 11 adaptive selection, 13 plots, 14 report | — | not started |

Python 3.14.3, qiskit 2.5.2 / aer 0.17.2 / qml 0.9.1. **192 tests pass** (~3 min).
Main-matrix numbers (E1–E8, 40 rows, shots=1024, full 150-iteration budget,
verified) are unchanged and live in `results/vqc_experiment_results.jsonl`;
accuracies: E1 .92, E2 1.00, E3 .86, E4 .96, E5 .95, E6 .95, E7 .88, E8 .91
(pooled over 100 test predictions each). Classical: 1.00 (LogReg, SVM).

## Extensions (all labeled `extension`, separate `results/extension_*` files)

**Code changes:** `extensions.py` (new); `evaluation.py` gained
`evaluate_vqc_with_noise_model` (the old function is now a thin wrapper);
`experiments.run_matrix` gained `prepare_split` / `experiment_id_prefix` /
`extra_record_fields` hooks (defaults keep E1–E8 behaviour identical);
`datasets.subsample_training_stratified`; new defaults in `config.py`
(noise-sweep levels, 0.1 one-qubit ratio, BC train subsample = 100), labeled
chosen defaults. E1–E8 result files are never written by extensions.

**1. Split check — done (real).** Every seed gives a *different* stratified
split (all 10 seed pairs differ; test-set Jaccard 0.05–0.18), so pooling is
valid. Pooled over seeds (`extension_pooled_analysis.json`), paired per-seed
degradation, bootstrap 95% CI of mean degradation (10 000 resamples, n = 5):

| Exp | Mean degradation (pp) | 95% bootstrap CI |
|---|---:|---|
| E5 (4q, low) | +1.0 | [0.0, 3.0] |
| E6 (4q, medium) | +1.0 | [−2.0, 4.0] |
| E8 (6q, medium) | −3.0 | [−5.0, −1.0] |

Honest reading: E5/E6 are inconclusive (CIs include or touch 0). E8's CI
excludes 0 on the *negative* side — medium noise did not hurt and nominally
helped. With 5 seeds on 20-sample test sets (accuracy moves in 5-pp steps) a
bootstrap over five values is coarse; treat this as suggestive, not proof.

**2. Noise sweep — smoke-checked only.** 1 seed, 20-iteration budget, both
architectures × 6 levels ran and saved correctly (TVD rose monotonically
0 → 0.29 for 6q); accuracy stayed flat. Smoke numbers are plumbing checks,
not findings, and were moved to `results/smoke_checks/` so they cannot
contaminate the full run's resume state (smoke rows count as "completed").

**3. Breast Cancer — smoke-checked only.** 1 seed, 20 iterations, 100-sample
train subsample, full 114-sample test set confirmed. Classical 0.956 (LogReg,
SVM); VQC 0.51–0.65 ≈ chance, as expected at 20 iterations on 30→2–6 PCA
features. **No conclusions possible until the full run**; a weak VQC result
there is plausible and will be reported as-is.

## Commands for you (run in order; all resume if interrupted — rerun the same command)

```powershell
.venv\Scripts\Activate.ps1; $env:PYTHONPATH="src"
python -m qml_noise.extensions noise-sweep --mode full     # ~15–25 min (5 seeds × 2 archs; 80–95 s training each)
python -m qml_noise.extensions breast-cancer --mode full   # ~60–75 min (≈14 min/seed; ~2× Iris cost/iteration, 114 test samples)
# Optional: --seeds 0,1 for a shorter first pass; --force wipes that extension's results and restarts.
```

Estimates come from measured Iris training times (E4 80 s, E7 95 s per seed)
and BC smoke timings scaled to 150 iterations; actual time may be lower if
COBYLA converges early.

## Not done
Section 11 (AN-VQC), 13 (plots), 14 (report); extension full runs; extension
changes not yet committed.
