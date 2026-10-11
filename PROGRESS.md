# Progress — Noise-Aware QML Project

| Stage | Sections | Status |
|---|---|---|
| 1–3 | env, config, data, classical, VQC, noise, metrics, runner; **full E1–E8 × 5 seeds** | done |
| 5 | extensions: split check, pooled stats, noise sweep (3 seeds), Breast Cancer (5 seeds) | done, corrected noise, verified |
| 6 | plots (11), circuit diagram, report | **done — plot 07 fixed, report fully filled** |
| 7 | noise-level correction to assignment table | done + verified |
| 4 | **section 11 AN-VQC** | **done — 3-seed run complete, verified, reported honestly** |

Python 3.14.3, qiskit 2.5.2 / aer 0.17.2. No experiments run this turn (verification, plotting fix,
report writing only — the 3-seed AN-VQC run was already complete when this turn started).

## This stage: AN-VQC verified, plot 07 fixed, report finished

**(1) AN-VQC verified** (`results/an_vqc_{results,selection_log,classical_results}.jsonl`): **3 seeds
(0,1,2), 1024 shots, no duplicates, all `status == "completed"`.** Frozen config confirmed live:
`alpha=0.5, beta=0.3, gamma=0.2, val_fraction=0.25, target_noise_profile="medium", maxiter=60`;
`medium` = 0.01/0.02/0.03 (single-qubit/two-qubit/readout). Selection used validation data only
(`tests/test_an_vqc.py::test_selection_never_touches_test_data` — a sentinel substituted for the
test split raises on any access; selection still completes).

**Selected per seed:** seed 0 → (2,4); seed 1 → (2,2); seed 2 → (2,2). The ideal-validation-best
architecture was (2,4) in **every** seed — AN-VQC agreed only in seed 0, trading it for the cheaper
(2,2) in seeds 1–2 when validation accuracy was close enough (0.90–0.95 vs 1.00) to let the cost
term win.

| Method | Architecture | Acc. ideal | Acc. medium | Degradation (pp) | Robustness | Gates | Depth |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed_vqc | (4,4) | 1.000 ± 0.000 | 0.983 ± 0.024 | +1.67 ± 2.36 | 0.983 ± 0.024 | 112 | 41 |
| ideal_best | (2,4) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.00 ± 0.00 | 1.000 ± 0.000 | 56 | 33 |
| shallow_vqc | (2,2) | 0.933 ± 0.024 | 0.950 ± 0.000 | −1.67 ± 2.36 | 1.019 ± 0.026 | 32 | 19 |
| an_vqc | varies | 0.950 ± 0.041 | 0.967 ± 0.024 | −1.67 ± 2.36 | 1.019 ± 0.026 | 32–56 | 19–33 |
| LogReg / SVM | — | 1.000 / 1.000 | — | — | — | — | — |

Dedup confirmed working on the real run: `an_vqc`'s training was reused from `ideal_best` (seed 0,
same arch) and from `shallow_vqc` (seeds 1–2), zero extra training cost either way.

**Honest verdict: AN-VQC did not help.** `ideal_best` matched or beat it on test accuracy in every
seed (no robustness/cost term at all, yet dominant). AN-VQC cut gate count in 2/3 seeds as designed,
but at an accuracy cost with no compensating robustness gain distinguishable from noise. Re-scoring
every method on test data under AN-VQC's own formula: `shallow_vqc` scores highest
(0.7722 ± 0.0039), narrowly beating `an_vqc` (0.7688 ± 0.0045) and `ideal_best` (0.7647 ± 0.0000) —
**AN-VQC did not even win under its own objective on test data.** Reported as negative/inconclusive,
not papered over. Limits stated in both PROGRESS and the report: 3 seeds, 20-sample test set
(5-pp resolution), ~20-point validation set.

**(2) Plot 07 fixed and regenerated.** Rewrote `plot_standard_vs_adaptive` to read
`results/an_vqc_results.jsonl` (methods `an_vqc`/`fixed_vqc`/`ideal_best`/`shallow_vqc`, conditions
`ideal`/`medium`) and `results/an_vqc_classical_results.jsonl` (LogReg/SVM), grouped bars with
mean±std over 3 seeds. Removed the old provisional `adaptive_selection_results.jsonl` schema
entirely. Fixed a text-overlap bug found on first render (classical baseline labels collided with
AN-VQC's own bar labels) before finalizing.
```powershell
python -m qml_noise.plotting --noise-sweep-seeds 0,1,2 --breast-cancer-seeds 0,1,2,3,4
# Generated 11 plot(s). Skipped 0.
```
`tests/test_plotting.py`: rewrote the adaptive-plot tests for the real schema (6 tests covering
drawn/empty/missing-column/missing-method/missing-condition/missing-classical) — **19/19 passed**,
run twice (before and after the overlap fix).

**(3) `report/report.md` finished.** Every remaining `PLACEHOLDER` (RQ5, Section 9.4, Section 12,
Abstract, Conclusion) filled from saved results only — no more markers of any kind in the file.
Figure 11 added (`plots/07_standard_vs_adaptive.png`).

## Not done
Full pytest suite re-run (only `test_plotting.py` was re-verified this turn, since it's the only
module touched; `an_vqc.py` itself was not modified); committing this stage.
