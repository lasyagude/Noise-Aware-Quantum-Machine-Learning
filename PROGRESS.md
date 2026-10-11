# Progress — Noise-Aware QML Project

| Stage | Sections | Status |
|---|---|---|
| 1–3 | env, config, data, classical, VQC, noise, metrics, runner; **full E1–E8 × 5 seeds** | done |
| 5 | extensions: split check, pooled stats, noise sweep (**5 seeds**), Breast Cancer (5 seeds) | done, corrected noise, verified |
| 6 | plots (11), circuit diagram, report | **done — all sections filled, no markers remain** |
| 7 | noise-level correction to assignment table | done + verified |
| 4 | section 11 AN-VQC | done — 3-seed run complete, verified, reported honestly |

Python 3.14.3, qiskit 2.5.2 / aer 0.17.2. No experiments run this turn — verification + report
text only (the noise sweep's extra seeds 3–4 were already run and committed before this turn).

## This stage: noise sweep extended to 5 seeds, report refreshed

**Verified** `results/extension_noise_sweep.jsonl`: **60 rows, 5 seeds (0–4), no duplicates,
`shots == 1024` everywhere, all `status == "completed"`.** Sanity check: the p₂=0 (ideal) rows match
the main matrix's E4 (0.960 ± 0.037) and E7 (0.880 ± 0.081) exactly — same architecture, same
training, same seeds.

**Refreshed numbers (5 seeds vs. the old 3-seed table):**

| Architecture | p₂ | Accuracy (old → new) | TVD (old → new) |
|---|---:|---|---|
| 4q, d4 | 0.05 | 0.983 → 0.960 ± 0.037 | 0.165 → 0.146 ± 0.032 |
| 6q, d4 | 0.05 | 0.850 → 0.860 ± 0.097 | 0.276 → 0.248 ± 0.054 |

(Full 12-row table with degradation and robustness is in `report/report.md` §9.3.1.) The extra data
changed the picture from "4-qubit accuracy completely flat at 0.983" to "both architectures move
within 1–2 pp of baseline with no consistent trend, well inside their own 2–5 pp standard
deviations" — a more honest and, with 5 seeds instead of 3, better-supported version of the same
conclusion. TVD stayed cleanly monotonic at both architectures, now with visibly tighter error bars.

**Updated everywhere `report/report.md`, `README.md`, `PROGRESS.md` said "3 seeds"/"n=3" about the
sweep:** Abstract, §9.3.1 (heading, table, prose, Figure 9 caption), the RQ3 discussion paragraph,
Limitations item 9, and the Appendix commands. **Left untouched:** every "3 seeds" reference to
AN-VQC (Section 9.4/12, RQ5) — that run is genuinely 3 seeds and out of scope this turn. Plot
`extension_noise_sweep.png` was already regenerated with 5 seeds before this turn (footnote already
read "n = 5 seed(s)"); verified, not redrawn. `README.md`'s top status banner and two other lines
claiming the report still had `PLACEHOLDER`s / AN-VQC was "not implemented" were also stale from an
earlier turn and fixed for consistency, since they sit in the same paragraphs this turn touched.

## Not done
Full pytest suite re-run (no code changed this turn, only docs — `test_extensions.py`'s existing
noise-sweep tests use synthetic data and don't depend on the real file's seed count); committing
this stage.
