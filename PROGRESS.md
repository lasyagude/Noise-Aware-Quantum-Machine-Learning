# Progress — Noise-Aware QML Project

| Stage | Sections | Status |
|---|---|---|
| 1–3 | env, config, data, classical, VQC, noise, metrics, runner; **full E1–E8 × 5 seeds** | done, committed (`47ec9aa`) |
| 5 | extensions: split check + pooled stats (done); noise sweep + Breast Cancer | code committed (`c169bcb`); **full runs in progress in your terminal** |
| 6 | plots (13), circuit diagram, report draft | **new, uncommitted** (this stage) |
| 4 | section 11 AN-VQC (adaptive selection) | **not started** |

Python 3.14.3, qiskit 2.5.2 / aer 0.17.2. No experiments were run in this stage.

## This stage (plots + report)

- `src/qml_noise/plotting.py` + `python -m qml_noise.plotting`: reads only saved results.
  Generated: `plots/01_degradation_vs_noise`, `02_accuracy_vs_depth`, `03_noise_comparison`
  (+ `noise_comparison_table.csv/.md`), `04_qubits_vs_performance`, `05_output_distributions`,
  `06_classical_vs_qml`, `08_accuracy_mean_std`, and `circuit_4q_depth2.png`
  (Qiskit `draw("mpl")`; needed new dependency `pylatexenc==2.11`, added to `requirements.txt`).
- **Skipped on purpose** (logged with reasons): `07_standard_vs_adaptive` (no AN-VQC results; the
  schema it expects is provisional), `extension_noise_sweep` and `extension_breast_cancer`
  (drawn only when all expected experiment/seed pairs exist; at last check the sweep had 18/60
  and Breast Cancer 1/40). Re-run `python -m qml_noise.plotting` after your runs finish.
- Colours: validated categorical slots 1–3 (aqua is <3:1 contrast on light, so every chart also
  uses distinct markers/hatching, value labels and a CSV table). Fixed two layout defects found by
  viewing the PNGs (overlapping labels/legends; a 6-qubit line implying a non-existent "low" run).
- `report/report.md`: all 14 required sections. Real numbers from the 40 E1–E8 rows and the pooled
  analysis; **15 `PLACEHOLDER` markers** (noise sweep, Breast Cancer, AN-VQC, and the conclusion that
  depends on them) contain no numbers. 11 references, all located by web search (metadata checked;
  full texts not read; Du et al. DOI not verified so only arXiv + journal ref are given).
- Tests: `tests/test_plotting.py`, 13 passed (synthetic rows only inside tests). **205 tests are
  collected; I did not re-run the full suite** (its training tests count as experiments); the last
  full pass was 192 tests before this stage's 13 were added.

## Findings reported in the draft (E1–E8, 5 seeds; every number from `results/`)

Accuracy: E1 .92, E2 1.00, E3 .86, E4 .96, E5 .95, E6 .95, E7 .88, E8 .91; LogReg/SVM 1.00.
Depth 4 > depth 2; accuracy *fell* with qubits at depth 4 (confounded with parameter count and an
unconverged 150-iteration budget). Degradation: E5 +1.0, E6 +1.0, E8 −3.0 pp, all within test-set
resolution (5 pp steps); bootstrap CIs [0,3], [−2,4], [−5,−1]. TVD is the clean signal:
0.049 → 0.159 (4q low → medium), 0.271 (6q medium). The E8 improvement is reported as
suggestive only; its explanations are labelled untested hypotheses.

## Your next steps

```powershell
# when both extension runs finish (they resume if interrupted):
python -m qml_noise.plotting                 # draws the two extension figures if complete
```
Then fill the `PLACEHOLDER`s in `report/report.md` (Section 9.3 and the conclusion).

## Not done
AN-VQC (section 11), its plot and report section; extension results; README compliance checklist
(task2 §9); full-suite re-run; committing this stage.
