# Progress — Noise-Aware QML Project

| Stage | Sections | Status |
|---|---|---|
| 1–3 | env, config, data, classical, VQC, noise, metrics, runner; **full E1–E8 × 5 seeds** | done, committed |
| 5 | extensions: split check + pooled stats; noise sweep | done, committed |
| 5 | extension: Breast Cancer | **regenerating under corrected noise (seeds 0–4) — in progress, not committed** |
| 6 | plots (13), circuit diagram, report draft | done, committed (plots/report not yet refreshed for corrected numbers) |
| 4 | section 11 AN-VQC | code + tests + 1-seed smoke done; **3-seed run not run** |
| 7 | noise-level correction to assignment table | **done + verified** (`8f32ea8`, `4e8cf8f`) |

Python 3.14.3, qiskit 2.5.2 / aer 0.17.2. `config.NOISE_PROFILES`: low=0.001/0.005/0.01,
medium=0.01/0.02/0.03 (assignment-specified), optional high=0.03/0.05/0.05 — confirmed live via
import, not just on disk.

## Main matrix — corrected and verified (this check, not run by me)

`results/vqc_experiment_results.jsonl` was already regenerated (commit `4e8cf8f`, not by me this
turn). Verified by reading the file directly:

- **40 rows, no duplicates, `shots == 1024` everywhere, `status == "completed"` everywhere.**
- **Ideal-baseline consistency:** for every seed, E5's and E6's `accuracy_ideal_baseline` exactly
  equals that seed's saved E4 accuracy, and E8's equals E7's — all 15 noisy rows checked, zero
  mismatches. Confirms E5/E6/E8 were evaluated against the correct (4,4)/(6,4) trained weights.

**Corrected degradation / robustness / TVD (mean ± population std, n=5):**

| Exp | Mean accuracy | Degradation (pp) | Robustness ratio | TVD vs ideal |
|---|---:|---:|---:|---:|
| E5 (4q, low) | 0.960 | +0.00 ± 3.16 | 0.9994 ± 0.034 | 0.0324 ± 0.009 |
| E6 (4q, medium) | 0.950 | +1.00 ± 2.00 | 0.9889 ± 0.022 | 0.1182 ± 0.026 |
| E8 (6q, medium) | 0.910 | −3.00 ± 2.45 | 1.0348 ± 0.029 | 0.2094 ± 0.047 |

Degradation/robustness are numerically identical to the pre-correction run (same seeds move the
same test samples either way — Iris is linearly separable enough that the corrected, *lower*
per-gate probabilities didn't flip any additional predictions). **TVD dropped visibly** at every
noisy row (E5 0.049→0.032, E6 0.159→0.118, E8 0.271→0.209) — expected, since medium's two-qubit
rate fell from 5%→2% (more than offsetting 1-qubit's 0.5%→1% rise) and low's two-qubit rate fell
from 1%→0.5%. `results/extension_pooled_analysis.json` is now stale again (reflects the
pre-correction bootstrap CIs) — rerun `python -m qml_noise.extensions pooled-analysis` once Breast
Cancer frees up, or now (it only reads the just-verified matrix file).

## Breast Cancer — in progress, check later

`results/extension_breast_cancer_results.jsonl` is mid-regeneration under the corrected noise
values (now targeting 5 seeds, not the original 3) and is actively changing on disk — not
inspected or touched this turn, per instruction.

## AN-VQC 3-seed run — command given, NOT run

```powershell
.venv\Scripts\Activate.ps1; $env:PYTHONPATH="src"
python -m qml_noise.an_vqc --mode full
```
Uses the corrected `"medium"` profile automatically (`AN_VQC_TARGET_NOISE_PROFILE = "medium"` reads
`NOISE_PROFILES["medium"]` live; no code change needed). **Estimate: ~18–24 minutes** for seeds
0,1,2 — measured from the real 1-seed smoke run (~6.4 min at the same `maxiter=60` the full run
uses; architecture overlap/dedup varies by seed, so not a clean 3×).

## Not done
Breast Cancer regeneration (external, in progress); AN-VQC 3-seed run; regenerating
`extension_pooled_analysis.json`, plots, and `report/report.md`'s stale sections from the now-correct
matrix data; committing any of this.
