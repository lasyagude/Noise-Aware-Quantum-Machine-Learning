# `plots/`

All charts, generated directly from the files in `results/` — nothing here is hand-drawn or
fabricated. Regenerate any of them with:

```powershell
python -m qml_noise.plotting
```

| File | What it shows |
|---|---|
| `01_degradation_vs_noise.png` | How much accuracy drops as noise increases |
| `02_accuracy_vs_depth.png` | Accuracy vs. how many layers the circuit has |
| `03_noise_comparison.png` | Side-by-side comparison across noise levels |
| `04_qubits_vs_performance.png` | Accuracy vs. number of qubits used |
| `05_output_distributions.png` | How the circuit's raw measurement outcomes shift under noise |
| `06_classical_vs_qml.png` | Classical models vs. the quantum classifier |
| `07_standard_vs_adaptive.png` | The proposed "auto-pick the best circuit" method vs. simpler baselines |
| `08_accuracy_mean_std.png` | Accuracy averaged across all 5 random seeds, with error bars |
| `circuit_2q_depth2.png`, `circuit_4q_depth2.png`, `circuit_6q_depth4.png` | Diagrams of the actual quantum circuit at a few different sizes |
| `extension_noise_sweep.png` | Results of the extra, finer-grained noise sweep |
| `extension_breast_cancer.png` | Results of the same experiments re-run on the Breast Cancer dataset |
| `noise_comparison_table.csv` / `.md` | The noise-comparison numbers as a plain table |
