# Progress — Noise-Aware QML Project

| Stage | Sections | Status |
|---|---|---|
| 1–3 | env, config, data, classical, VQC, noise, metrics, runner; **full E1–E8 × 5 seeds** | done, committed |
| 5 | extensions: split check + pooled stats; noise sweep + Breast Cancer (3 seeds) | done, committed |
| 6 | plots (13), circuit diagram, report draft | done, committed |
| 4 | section 11 AN-VQC | code + tests + 1-seed smoke done; 3-seed run not run |
| 7 | **noise-level correction** (this stage) | **done; invalidates some saved results — see below** |

Python 3.14.3, qiskit 2.5.2 / aer 0.17.2.

## This stage: noise levels corrected to the assignment's guideline table

**(1) What was in `config.py` vs. the table, before this change:**

| Profile | Had (1q / 2q / readout) | Table wants (1q / 2q / readout) | Matched? |
|---|---|---|---|
| low | 0.1% / 1% / 1% | 0.1% / 0.5% / 1% | 1q ✓, **2q ✗**, readout ✓ |
| medium | 0.5% / 5% / 3% | 1% / 2% / 3% | **1q ✗**, **2q ✗**, readout ✓ |
| high | *(did not exist)* | 3–5% / — / higher (optional) | — |

So `low`'s two-qubit rate and `medium`'s both gate rates were off; readout was already correct on
both; `high` didn't exist.

**(2) Fixed in `config.py`** (`NOISE_PROFILES`): `low` → 0.001/0.005/0.01, `medium` →
0.01/0.02/0.03, and a new optional `high` → 0.03/0.05/0.05 (a chosen point inside the
table's stated 3–5% range, documented as such; not in `NOISE_CONDITIONS`, added to new
`OPTIONAL_NOISE_CONDITIONS`). Docstrings in `config.py` and `noise.py`, the README, and
`report/report.md`'s Section 6 methodology table now say `low`/`medium` are **assignment-specified**,
not chosen defaults. `report/report.md` also got a **stale-data warning** (top of file + above
Section 9) instead of silently leaving outdated numbers unflagged — the old E5/E6/E8 numbers are
still there (they're real, just stale), not deleted or rewritten, since doing that honestly requires
rerunning the experiments, which this stage does not do.

**(3) Affected result files — not regenerated yet:**

| File | Stale rows | Why |
|---|---|---|
| `results/vqc_experiment_results.jsonl` (+`.csv`) | E5, E6, E8 (15 of 40) | evaluated under old low/medium |
| `results/extension_breast_cancer_results.jsonl` (+`.csv`) | BC-E5, BC-E6, BC-E8 (9 of 24) | same |
| `results/extension_pooled_analysis.json` | entire file | derived from the file above |
| `plots/01,03,05,06,08*.png` + `noise_comparison_table.*` | all | derived from the matrix file |
| `report/report.md` §9 (Results), Abstract, Discussion | numbers citing E5/E6/E8 | now marked stale in-place |

**Confirmed unaffected, no action needed:** `results/extension_noise_sweep.jsonl` (sweeps its own
independent depolarizing probabilities via `build_depolarizing_noise_model` directly — never reads
`NOISE_PROFILES`); `results/extension_split_check.json` (no noise involved); both classical-baseline
files (no noise concept); `results/smoke_checks/an_vqc_*` (already labeled smoke-only, never a real
result, harmless that it used the old "medium"); the AN-VQC 3-seed run (hasn't been executed at all
yet — when it is, it will automatically use the corrected `"medium"` profile).

**Regeneration — exact commands.** Resume logic keys on `(experiment_id, seed)` and can't tell old
probabilities from new ones, so a plain rerun would skip the stale rows as "already completed".
`scripts/remove_stale_noise_rows.py` removes only the stale rows (verified against copies of the
real files, not run for real — not an experiment, but still deferred per instruction); the
unaffected ideal rows are then resume-skipped and never retrained.

```powershell
.venv\Scripts\Activate.ps1; $env:PYTHONPATH="src"
python scripts\remove_stale_noise_rows.py
python -m qml_noise.experiments --mode full                # ~15 min: retrains only the (4,4)/(6,4) groups, 5 seeds
python -m qml_noise.extensions breast-cancer --mode full    # ~12 min: retrains only BC's (4,4)/(6,4) groups, 3 seeds
python -m qml_noise.extensions pooled-analysis              # seconds
python -m qml_noise.plotting                                # seconds; redraws everything from corrected data
```
Estimates are measured, not guessed: E4/E7 (4,4)/(6,4) ideal training took 80.4s/94.7s per seed in
the saved 5-seed run; BC's equivalents took ~112s/~120s per seed in the saved 3-seed run. (A simpler
but slower alternative is `--force` on each command instead of the scrub script — full 8-row ×
5/3-seed rebuild, ~28 min + ~14 min — if you'd rather not trust the reuse-of-ideal-rows reasoning.)
After these, fill the now-correct numbers into `report/report.md` (the `STALE`/`PLACEHOLDER` markers
show exactly where) and regenerate the extension plots (still need `extension_seeds=[0,1,2]` wired
into `plotting`'s CLI — a pre-existing gap, not new this stage).

**(4) Tests:** `tests/test_noise.py` — 4 new tests (exact table-match for low/medium, `high` within
its stated range and harsher than medium, `high` usable through `get_noise_model`). Checked every
other test file for hardcoded noise-dependent numeric assertions (accuracy/degradation/TVD values
under "medium"/"low"): none found — existing tests check structure/labels/relative ordering, not
exact magnitudes, so nothing else needed updating. **Full suite: `pytest -q` → 238 passed in 15m03s**
(includes the full AN-VQC/plotting/extensions suites from prior stages — no regressions from this
change).

## Not done
Regenerating the affected result files (commands above, not run); report.md's stale sections;
extension plots at 3 seeds; AN-VQC 3-seed run; committing this stage.
