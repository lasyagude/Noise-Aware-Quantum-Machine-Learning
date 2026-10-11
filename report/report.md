# Noise-Aware Quantum Machine Learning: Experimental Analysis and Adaptive Circuit Selection for Robust Classification

**Status: DRAFT.** Sections 1–11 use only saved results for the mandatory E1–E8 matrix
(`results/vqc_experiment_results.jsonl`, 40 rows, commit `47ec9aa`) and the already-computed
pooled analysis (`results/extension_pooled_analysis.json`). Anything not yet run is marked
`PLACEHOLDER` and contains **no numbers**: the extension runs (noise sweep, Breast Cancer) were
still in progress when this draft was written, and the adaptive method (AN-VQC) is not implemented.
Search the file for `PLACEHOLDER` to find everything that must be filled in before submission.

---

## Abstract

We study how simulated hardware noise affects a small variational quantum classifier (VQC)
built and simulated entirely in Qiskit/Qiskit Aer. Using the assignment's eight-experiment matrix
(2/4/6 qubits, ansatz depth 2/4, ideal/low/medium noise, 1024 shots) on binary Iris
(setosa vs. versicolor), each architecture is trained once without noise, and the same trained
weights are then re-evaluated under noise. Across five seeds (each with its own stratified
80:20 split), mean ideal accuracy per architecture ranged from 0.86 to 1.00, and depth-4 circuits scored higher than
depth-2 circuits (2 qubits: 0.92 → 1.00; 4 qubits: 0.86 → 0.96). Accuracy was not monotone in
qubit count at depth 4 (1.00, 0.96, 0.88 for 2, 4, 6 qubits). Noise had a small and statistically
inconclusive effect on accuracy: mean degradation was +1.0 pp (4 qubits, low noise), +1.0 pp
(4 qubits, medium) and −3.0 pp (6 qubits, medium; noisy accuracy was *higher* on average).
Output-distribution distance (total variation distance, TVD) was a far clearer signal, rising
from 0.049 (low) to 0.159 (medium) at 4 qubits and reaching 0.271 at 6 qubits under medium noise.
Logistic regression and an RBF SVM both reached 1.00 accuracy, so no advantage of the quantum
classifier is claimed. Extension results and the proposed adaptive noise-aware circuit selection
are pending (`PLACEHOLDER`).

## 1. Problem statement

Near-term quantum devices are noisy, and noise limits how deep or wide a circuit can usefully be
(Preskill, 2018). Variational algorithms were proposed partly as a response to these limits
(Cerezo et al., 2021), but it is not obvious *how much* noise a trained quantum classifier
tolerates, nor whether accuracy alone is a sensitive enough probe. This project measures the
noise sensitivity of a VQC across circuit sizes and noise conditions, and compares accuracy with
the shift in the circuit's output distribution.

## 2. Objectives and research questions

Objective: measure noise sensitivity across quantum classifiers, circuit configurations and noise
conditions, then propose and evaluate an improvement.

- **RQ1.** In ideal simulation, how do accuracy and circuit cost change with qubit count and depth?
- **RQ2.** How much does low/medium depolarizing + readout noise degrade the accuracy of an
  ideally-trained VQC?
- **RQ3.** Is the output-distribution shift (TVD / Hellinger) a more sensitive indicator of noise
  than accuracy?
- **RQ4 (extension).** How does accuracy fall as the depolarizing probability is swept?
  `PLACEHOLDER — answered by the noise-sweep extension once complete.`
- **RQ5 (proposed method).** Does validation-based noise-aware circuit selection (AN-VQC) improve
  on a fixed circuit, the ideal-validation-best circuit, a shallow circuit, and classical
  baselines? `PLACEHOLDER — AN-VQC not implemented; no result exists.`

## 3. Related work and research gap

Only works located and checked by web search are cited (see References); statements below are
limited to what their abstracts report. Full texts were not read.

- **Noise limits NISQ circuits.** Preskill (2018) argues that gate noise limits the circuits
  near-term devices can run reliably. Cerezo et al. (2021) review variational quantum algorithms
  (a parameterized circuit trained by a classical optimizer), noting that noise limits circuit
  depth and that trainability, accuracy and efficiency remain challenges.
- **Variational quantum classifiers.** Havlíček et al. (2019) ran a variational quantum classifier
  and a quantum kernel estimator on a superconducting processor, which is the design family
  studied here.
- **Noise and trainability.** Wang et al. (2021) show that for local Pauli noise, gradients can
  shrink exponentially with qubit count when ansatz depth grows linearly with qubit number
  ("noise-induced barren plateaus"), a reason to expect deeper/wider circuits to be harder to
  train under noise.
- **Circuit design and accuracy.** Hubregtsen et al. (2021) evaluate 19 parameterized circuits on
  9 datasets and report a strong correlation between expressibility and classification accuracy
  and a weak one with entangling capability.
- **Noise can also help.** Du et al. (2021) prove that depolarizing noise yields a robustness bound
  against adversarial perturbations that improves as noise grows. This is a different setting
  (adversarial robustness), but it shows noise is not uniformly harmful, which is relevant to our
  E8 observation.
- **Noise-adaptive circuit selection.** QuantumNAS (Wang et al., 2021, arXiv preprint) searches for
  circuits and qubit mappings using device noise models and reports gains over noise-unaware
  search baselines. It is the closest published idea to our proposed AN-VQC.

**Gap addressed (modestly).** We do not claim novelty for noise-aware circuit search. This project
provides a small, fully reproducible, controlled study on a standard benchmark: the same trained
weights under each noise level, paired per seed, with output-distribution metrics beside accuracy,
and an explicit report of null and negative results.

## 4. Dataset and preprocessing

- **Primary: Iris, binary** (setosa vs. versicolor; 100 samples, 4 features) (Fisher, 1936),
  loaded with scikit-learn (Pedregosa et al., 2011). Stratified 80:20 split → 80 train / 20 test.
- **Splits vary with the seed.** Reproducing the split call for seeds 0–4 shows all 10 seed pairs
  have different test sets (test-set Jaccard overlap 0.05–0.18; `results/extension_split_check.json`).
  Each seed therefore evaluates on a different 20-sample test set, and results are comparable
  *within* a seed (identical split for all models) rather than identical across seeds.
- **Preprocessing:** `StandardScaler` fit on training data only. Test labels are never used for
  fitting or tuning.
- **Feature-to-qubit adaptation:** 4 features → 2 qubits by PCA (fit on training data only);
  4 → 4 identity; 4 → 6 by cyclic tiling (features 0,1,2,3,0,1), so 6-qubit circuits receive
  duplicated information.
- **Secondary (extension): Breast Cancer Wisconsin Diagnostic** (Street, Wolberg & Mangasarian,
  1993), 30 features, PCA to the qubit count, 100-sample stratified training subset, full ~114-sample
  test set. `PLACEHOLDER — see Section 9.3.`

## 5. QNN architecture and circuit design

![4-qubit depth-2 circuit](../plots/circuit_4q_depth2.png)

*Figure 1. Logical circuit for 4 qubits, depth 2 (generated by Qiskit `draw("mpl")`).*

| Item | Choice |
|---|---|
| Encoding | Angle encoding: `H` then `RY(x_i)` on qubit *i* |
| Ansatz block | trainable `RY` on every qubit, trainable `RZ` on every qubit, then a ring of CNOTs *(i → i+1 mod n)* |
| Logical depth | number of repeated blocks (2 or 4) |
| Parameters | 2 · qubits · depth (8, 16, 16, 32, 48 across the matrix) |
| Readout | all qubits measured; class 1 if the parity (XOR) of the bitstring is odd; label = 1 if P(class 1) > 0.5, estimated from 1024 shots |
| Loss | binary cross-entropy on the shot-estimated P(class 1) (clipped to avoid log 0) |
| Optimizer | SciPy COBYLA, `maxiter = 150`, `tol = 1e-4` (gradient-free, suited to shot noise) |
| Initialization | uniform in [−0.1, 0.1], seeded by the experiment seed |
| Simulator | Qiskit Aer, 1024 shots (mandatory), Qiskit 2.5.2 / Aer 0.17.2 on Python 3.14 |

The circuit is transpiled **once** into the basis `{rz, sx, x, cx}` (optimization level 1) with
parameters unbound; sample features and trained weights are then bound. Noise exposure is therefore
identical across samples and optimizer iterations. Transpiled depth is reported separately from the
logical depth:

| Qubits | Logical depth | Transpiled depth | Total gates | 2-qubit gates | Parameters |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 19 | 32 | 4 | 8 |
| 2 | 4 | 33 | 56 | 8 | 16 |
| 4 | 2 | 23 | 64 | 8 | 16 |
| 4 | 4 | 41 | 112 | 16 | 32 |
| 6 | 4 | 49 | 168 | 24 | 48 |

## 6. Noise methodology

Noise is implemented with Qiskit Aer noise models attached to specific *transpiled* instructions:
single-qubit depolarizing error on `sx`/`x`, two-qubit depolarizing error on `cx`, and a symmetric
readout error on `measure`. `rz` carries **no** error because on superconducting hardware it is a
virtual frame change. Isolated bit-flip, phase-flip, depolarizing and readout channels are also
implemented and unit-tested, but the mandatory matrix uses the combined profiles below.

| Profile | 1-qubit depolarizing | 2-qubit depolarizing | Readout flip |
|---|---:|---:|---:|
| ideal | 0 | 0 | 0 |
| low | 0.001 | 0.01 | 0.01 |
| medium | 0.005 | 0.05 | 0.03 |

**These probabilities are chosen simulation defaults, not values given by the assignment**
(documented in `src/qml_noise/config.py`). Conclusions are therefore about this noise family and
these levels, not about any specific device.

**Evaluation protocol.** For each (qubits, depth) and seed, the VQC is trained once under ideal
conditions. The same weights are then evaluated under each noise condition by swapping the noise
model, with the same simulator seed, so the ideal/noisy comparison is paired. This measures the
noise *sensitivity of an ideally-trained model*; it does not measure noise-aware training, which
is not performed.

## 7. Experimental matrix

| Experiment | Qubits | Depth | Noise |
|---|---:|---:|---|
| E1 | 2 | 2 | ideal |
| E2 | 2 | 4 | ideal |
| E3 | 4 | 2 | ideal |
| E4 | 4 | 4 | ideal |
| E5 | 4 | 4 | low |
| E6 | 4 | 4 | medium |
| E7 | 6 | 4 | ideal |
| E8 | 6 | 4 | medium |

The assignment's recommended matrix, unmodified. **Completed: 8 experiments × 5 seeds (0–4) = 40 runs,
1024 shots in every run**, parameters varied: qubits, depth, noise (three parameters). Five trainings
per seed cover all eight rows (5 architectures). Total compute was about 28 minutes. E3–E8 used the
full 150-iteration COBYLA budget on every seed; E1 and E2 stopped early on the tolerance criterion
(mean 92 and 123 iterations).

## 8. Metrics

Accuracy, precision, recall, F1 and the confusion matrix are computed on each seed's test set.
With A_ideal and A_noisy the accuracies of the same trained weights:

- accuracy degradation D = 100 · (A_ideal − A_noisy) (percentage points; **negative means the noisy
  model scored higher**),
- relative degradation D_rel = 100 · (A_ideal − A_noisy) / A_ideal,
- noise robustness ratio R = A_noisy / A_ideal.

D_rel and R are `NaN` if A_ideal = 0 (never occurred). Values are never clamped, because noise can
raise measured accuracy. Output distributions are pooled over all test circuits (summed shot
counts) and compared with total variation distance (TVD = ½ Σ|p − q|) and Hellinger distance.
Also recorded: qubits, logical and transpiled depth, total and two-qubit gate counts, parameter
count, shots, and runtime. Mean ± std throughout is across seeds, with the **population** standard
deviation (ddof = 0). With n = 5 and a 20-sample test set, accuracy moves in steps of 5 pp, so
single-seed differences of one or two steps are within noise.

## 9. Results

All numbers are read from `results/vqc_experiment_results.jsonl`; the full table is
`plots/noise_comparison_table.csv`. Figures are regenerated with
`python -m qml_noise.plotting`.

### 9.1 Accuracy and circuit cost (E1–E8, 5 seeds)

| Exp | Qubits | Depth | Noise | Accuracy | Precision | Recall | F1 | Transpiled depth | 2-qubit gates |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| E1 | 2 | 2 | ideal | 0.920 ± 0.040 | 0.925 | 0.920 | 0.921 | 19 | 4 |
| E2 | 2 | 4 | ideal | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 33 | 8 |
| E3 | 4 | 2 | ideal | 0.860 ± 0.066 | 0.903 | 0.820 | 0.855 | 23 | 8 |
| E4 | 4 | 4 | ideal | 0.960 ± 0.037 | 0.982 | 0.940 | 0.958 | 41 | 16 |
| E5 | 4 | 4 | low | 0.950 ± 0.055 | 0.982 | 0.920 | 0.945 | 41 | 16 |
| E6 | 4 | 4 | medium | 0.950 ± 0.055 | 0.982 | 0.920 | 0.945 | 41 | 16 |
| E7 | 6 | 4 | ideal | 0.880 ± 0.081 | 0.874 | 0.900 | 0.885 | 49 | 24 |
| E8 | 6 | 4 | medium | 0.910 ± 0.080 | 0.908 | 0.920 | 0.912 | 49 | 24 |

Logistic regression and SVM (RBF): **1.000 ± 0.000** on every seed.

![Accuracy mean ± std](../plots/08_accuracy_mean_std.png)
*Figure 2. Accuracy per experiment, mean ± std across seeds (dots = seeds).*

![Accuracy vs depth](../plots/02_accuracy_vs_depth.png)
*Figure 3. Accuracy vs logical depth (ideal).*

![Qubits vs performance](../plots/04_qubits_vs_performance.png)
*Figure 4. Accuracy and transpiled depth vs qubit count.*

### 9.2 Noise sensitivity (paired per seed)

| Exp | Condition | Degradation D (pp) | Relative D (%) | Robustness R | TVD vs ideal | Hellinger |
|---|---|---:|---:|---:|---:|---:|
| E5 | 4q, low | +1.00 ± 2.00 | +1.11 | 0.989 ± 0.022 | 0.049 ± 0.010 | 0.044 |
| E6 | 4q, medium | +1.00 ± 3.74 | +1.06 | 0.989 ± 0.039 | 0.159 ± 0.037 | 0.138 |
| E8 | 6q, medium | −3.00 ± 2.45 | −3.48 | 1.035 ± 0.029 | 0.271 ± 0.061 | 0.236 |

Per-seed degradation (pp), seeds 0–4: E5 = 0, 0, 0, 5, 0; E6 = 0, −5, 5, 5, 0; E8 = 0, −5, 0, −5, −5.

Pooled analysis (**extension**, computed from the saved E1–E8 rows; 10 000 bootstrap resamples over
the 5 per-seed degradations, seed 0): the 95% bootstrap CI for mean degradation is **[0.0, 3.0] pp
(E5)**, **[−2.0, 4.0] pp (E6)** and **[−5.0, −1.0] pp (E8)**. Pooled accuracies over the 100 test
predictions per experiment equal the per-seed means above (E1 0.92 … E8 0.91), since every test set
has 20 samples.

![Degradation vs noise](../plots/01_degradation_vs_noise.png)
*Figure 5. Degradation (left) and TVD (right) vs noise level. The 6-qubit line is dashed because the
matrix has no 6-qubit low-noise run.*

![Noise comparison](../plots/03_noise_comparison.png)
*Figure 6. Accuracy under each noise condition for the same trained weights.*

![Output distributions](../plots/05_output_distributions.png)
*Figure 7. Mean pooled output distribution, ideal vs noisy (top 10 bitstrings of the ideal run).*

Under medium noise, probability mass moves off the ten most likely bitstrings: at 4 qubits the
residual "other" mass is 0.240 (ideal), 0.263 (low), 0.319 (medium); at 6 qubits 0.585 (ideal)
vs. 0.759 (medium). Noise did not create new outcomes (all 16 / 64 bitstrings already occur
under ideal simulation); it flattens the distribution.

![Classical vs QML](../plots/06_classical_vs_qml.png)
*Figure 8. Classical baselines vs ideal and noisy VQC.*

### 9.3 Extension results — `PLACEHOLDER`

> **`PLACEHOLDER — EXTENSION: depolarizing noise sweep.`** Two-qubit probabilities
> {0, 0.001, 0.005, 0.01, 0.02, 0.05} (single-qubit = one tenth), 4-qubit and 6-qubit depth-4
> models, 5 seeds. Accuracy, degradation, robustness ratio and TVD per level. Insert
> `plots/extension_noise_sweep.png` and a summary table once
> `results/extension_noise_sweep.jsonl` holds all 60 (experiment, seed) pairs. *No numbers yet.*

> **`PLACEHOLDER — EXTENSION: Breast Cancer Wisconsin (E1–E8 matrix).`** Insert
> `plots/extension_breast_cancer.png`, per-experiment accuracy ± std, degradation for BC-E5/E6/E8,
> and Logistic Regression / SVM on the identical subsampled training split, once
> `results/extension_breast_cancer_results.jsonl` holds all 40 pairs. *No numbers yet. A
> weak quantum result is possible and must be reported as found.*

### 9.4 Standard vs adaptive VQC — `PLACEHOLDER`

> **`PLACEHOLDER — AN-VQC comparison.`** Not implemented; `plots/07_standard_vs_adaptive.png` does
> not exist. *No numbers yet.*

## 10. Discussion

**RQ1: depth and qubits.** At 2 and 4 qubits, depth 4 beat depth 2 (0.92 → 1.00 and 0.86 → 0.96),
consistent with more trainable parameters helping within a fixed iteration budget. Accuracy then
*fell* with qubit count at depth 4 (1.00, 0.96, 0.88). We cannot attribute this to qubit count alone:
parameters grow from 16 to 48 with the same 150-iteration budget (E4/E7 hit the cap on every seed and
their mean final loss, 0.243 and 0.363, is above E2's 0.047), the feature adapter changes (PCA, identity,
duplicated features), and test sets differ by seed. Under-training is a plausible explanation, but it
was not tested (e.g. with a larger budget).

**RQ2: accuracy degradation is small and inconclusive.** Mean degradation was +1.0 pp for E5 and E6 and
−3.0 pp for E8, with per-seed values in {−5, 0, +5} pp, i.e. a single test sample flipping. The E5 and
E6 bootstrap intervals include or touch zero. The E8 interval lies entirely below zero: medium noise
did not hurt and nominally helped the 6-qubit model on average. Possible explanations are untested
hypotheses: an under-trained model whose ideal decision boundary is not optimal on the test sample, or
noise pulling the parity-based P(class 1) toward 0.5 and changing borderline decisions. Du et al. (2021)
show noise can have a protective effect in a different (adversarial) setting, but nothing here shows
that mechanism applies. With 5 seeds and 20-sample test sets this is suggestive at most; it is
not evidence that noise improves quantum classifiers.

**RQ3: distributions are more sensitive than accuracy.** TVD rose with noise strength in every
comparison (0.049 → 0.159 at 4 qubits; 0.271 at 6 qubits) with small relative spread, while
accuracy barely moved. Part of the 6-qubit increase may reflect the larger outcome space (64 vs 16
bitstrings) and more noisy two-qubit gates (24 vs 16), which cannot be separated in this matrix.

**Classical comparison.** Iris setosa/versicolor is linearly separable; both classical models are
perfect. The VQC is competitive at 2 and 4 qubits (up to 1.00) but never exceeds them, and no
quantum advantage is claimed.

## 11. Limitations

1. Simple noise: depolarizing + symmetric readout error only (no thermal relaxation, crosstalk,
   coherent or device-specific noise); probabilities are chosen defaults.
2. Noise is applied at evaluation only; models are trained ideally, so noise-aware training is untested.
3. Small data: 20-sample test sets (5 pp resolution) and 5 seeds; differences of 1–2 samples are noise.
   The easy, linearly separable Iris task leaves little room for degradation to show.
4. Confounds in the matrix: qubit count, parameter count, and feature adaptation change together; the
   6-qubit row has no low-noise or depth-2 counterpart.
5. Fixed COBYLA budget (150); E3–E8 were still improving when stopped, so "ideal" accuracy is a lower
   bound for those architectures.
6. The same simulator seed is used for ideal and noisy runs (paired design), so shot noise is not averaged
   out within a seed.
7. Trained weights are not persisted; noise-sweep extensions retrain the ideal model rather than reusing E4/E7 weights.
8. Simulation only; no hardware run.
9. The extension and AN-VQC parts of this report are incomplete (see `PLACEHOLDER`s).

## 12. Proposed improvement: Adaptive Noise-Aware VQC (AN-VQC) — `PLACEHOLDER`

> **`PLACEHOLDER — AN-VQC is designed but NOT implemented. No results exist.`**
> Planned design (from the assignment): choose among candidate circuits (2/4/6 qubits, depth 2/4) by
> Score(c) = α · ValidationAccuracy(c) + β · Robustness(c) − γ · NormalizedCost(c), using **validation
> data only** (never test data), with Robustness estimated under a target noise profile and cost from
> gate counts/depth. To be compared against a fixed VQC, the best circuit by ideal validation accuracy,
> a shallow circuit, Logistic Regression and SVM. Weights α, β, γ, the normalization, and the
> calibration procedure must be documented when implemented. Whether it helps is **unknown** and will be
> reported as found, including negative or inconclusive outcomes. Related idea: QuantumNAS (Wang et al.,
> 2021).

## 13. Conclusion and future work

In a controlled Qiskit simulation of ideally-trained VQCs on binary Iris, depth-4 circuits outperformed
depth-2 circuits, accuracy did not improve with more qubits under a fixed training budget, and low/medium
depolarizing + readout noise changed accuracy by amounts indistinguishable from test-set sampling noise
(one case, 6 qubits medium, nominally improved), while clearly and monotonically shifting the output
distribution. Classical baselines were perfect. Accuracy alone is a weak noise probe on this task;
distribution distances are more informative.

Future work: finish the noise sweep and Breast Cancer extensions; implement and honestly evaluate AN-VQC;
use larger test sets and harder datasets; train under noise; add device-calibrated noise and thermal
relaxation; separate qubit count from parameter count and training budget.
`PLACEHOLDER — update this section after the extension and AN-VQC results.`

## 14. References

Each entry was located by web search on 2026-10-11 and its title, authors and venue checked against the
search results; full texts were not read, and descriptions above are limited to their abstracts.

1. Preskill, J. (2018). Quantum Computing in the NISQ era and beyond. *Quantum* 2, 79.
   https://doi.org/10.22331/q-2018-08-06-79 · https://arxiv.org/abs/1801.00862
2. Cerezo, M. et al. (2021). Variational quantum algorithms. *Nature Reviews Physics* 3, 625–644.
   https://doi.org/10.1038/s42254-021-00348-9 · https://arxiv.org/abs/2012.09265
3. Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., Gambetta, J. M. (2019).
   Supervised learning with quantum-enhanced feature spaces. *Nature* 567.
   https://doi.org/10.1038/s41586-019-0980-2 · https://arxiv.org/abs/1804.11326
4. Wang, S., Fontana, E., Cerezo, M., Sharma, K., Sone, A., Cincio, L., Coles, P. J. (2021).
   Noise-induced barren plateaus in variational quantum algorithms. *Nature Communications* 12, 6961.
   https://doi.org/10.1038/s41467-021-27045-6 · https://arxiv.org/abs/2007.14384
5. Hubregtsen, T., Pichlmeier, J., Stecher, P., Bertels, K. (2021). Evaluation of parameterized quantum
   circuits: on the relation between classification accuracy, expressibility and entangling capability.
   *Quantum Machine Intelligence* 3. https://doi.org/10.1007/s42484-021-00038-w ·
   https://arxiv.org/abs/2003.09887
6. Du, Y., Hsieh, M.-H., Liu, T., Tao, D., Liu, N. (2021). Quantum noise protects quantum classifiers against
   adversaries. *Physical Review Research* 3, 023153. https://arxiv.org/abs/2003.09416
7. Wang, H., Ding, Y., Gu, J., Lin, Y., Pan, D. Z., Chong, F. T., Han, S. (2021). QuantumNAS: Noise-Adaptive
   Search for Robust Quantum Circuits. arXiv:2107.10845. https://arxiv.org/abs/2107.10845
8. Javadi-Abhari, A. et al. (2024). Quantum computing with Qiskit. arXiv:2405.08810.
   https://arxiv.org/abs/2405.08810
9. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*
   12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
10. Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics* 7(2),
    179–188. https://doi.org/10.1111/j.1469-1809.1936.tb02137.x
11. Street, W. N., Wolberg, W. H., Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor
    diagnosis. *Proc. SPIE* 1905, 861–870. https://doi.org/10.1117/12.148698

## Appendix: reproduction

```powershell
.venv\Scripts\Activate.ps1; $env:PYTHONPATH="src"
python -m qml_noise.experiments --mode full          # E1-E8, 5 seeds (done)
python -m qml_noise.extensions pooled-analysis       # extension: pooled stats (done)
python -m qml_noise.extensions noise-sweep --mode full        # extension (pending)
python -m qml_noise.extensions breast-cancer --mode full      # extension (pending)
python -m qml_noise.plotting                          # regenerate every plot the saved results support
```
