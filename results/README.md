# `results/`

The raw output from every experiment — real numbers from real runs, nothing fabricated or
hand-edited. Every plot and every number in the report traces back to one of these files.

| File | What it is |
|---|---|
| `vqc_experiment_results.jsonl` (+ `.csv`) | The main required experiment matrix: quantum classifier accuracy, degradation, and all other metrics across qubits/depth/noise, 5 seeds each |
| `classical_baseline_results.jsonl` | Logistic Regression and SVM results on the same data, for comparison |
| `extension_noise_sweep.jsonl` | The extra, finer-grained noise sweep |
| `extension_breast_cancer_results.jsonl` (+ `.csv`), `extension_breast_cancer_classical_results.jsonl` | The same experiment pipeline re-run on the Breast Cancer Wisconsin dataset |
| `extension_split_check.json` | A sanity check confirming the train/test split is done correctly and consistently |
| `extension_pooled_analysis.json` | Combined statistics pooled across the main matrix |
| `an_vqc_results.jsonl`, `an_vqc_selection_log.jsonl`, `an_vqc_classical_results.jsonl` | Results for the proposed "pick the best circuit automatically" method and what it compared against |
| `smoke_checks/` | Output from quick pipeline sanity checks, not real experiment results — kept separate so they never get mixed up with the real data |

Each file is one row per run, in JSON Lines format (`.jsonl` — one JSON object per line), so new
runs can be appended without rewriting the whole file. This is also how interrupted experiments
resume cleanly: whatever's already in the file is treated as done.
