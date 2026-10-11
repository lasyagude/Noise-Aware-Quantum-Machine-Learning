"""Tests for the plotting module.

Synthetic result rows exist ONLY inside these tests, written to tmp_path, to
exercise the plotting/skip logic. Real plots are produced from real saved
results by `python -m qml_noise.plotting`.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from qml_noise.config import NOISE_SWEEP_TWO_QUBIT_PROBABILITIES
from qml_noise.experiments import EXPERIMENT_MATRIX, VQC_RESULTS_FILE
from qml_noise.extensions import NOISE_SWEEP_ARCHITECTURES
from qml_noise.plotting import (
    PlotSkipped,
    generate_all_plots,
    noise_comparison_table,
    plot_circuit_diagram,
    plot_standard_vs_adaptive,
)

SEEDS = [0, 1]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _main_rows(seeds=SEEDS):
    rows = []
    for eid, q, d, noise in EXPERIMENT_MATRIX:
        for s in seeds:
            acc = 0.8 + 0.01 * q + 0.01 * s
            rows.append(
                {
                    "experiment_id": eid, "seed": s, "status": "completed", "accuracy": acc,
                    "precision": acc, "recall": acc, "f1": acc,
                    "degradation_pp": 0.0 if noise == "ideal" else 1.0 + s,
                    "relative_degradation_pct": 0.0, "robustness_ratio": 1.0,
                    "distribution_total_variation_distance": 0.0 if noise == "ideal" else 0.1 * q,
                    "distribution_hellinger_distance": 0.0,
                    "transpiled_depth": 10 * d + q, "total_gate_count": 5 * q * d,
                    "two_qubit_gate_count": q * d, "n_trainable_parameters": 2 * q * d,
                    "shots": 1024, "runtime_seconds": 1.5,
                    "pooled_output_distribution": {"00": 0.5, "01": 0.3, "10": 0.2}
                    if noise == "ideal" else {"00": 0.4, "01": 0.3, "10": 0.2, "11": 0.1},
                }
            )
    return rows


def _classical_rows(seeds=SEEDS):
    return [
        {"model_name": m, "seed": s, "accuracy": 1.0, "dataset_name": "iris"}
        for m in ("logistic_regression", "svm") for s in seeds
    ]


@pytest.fixture
def paths(tmp_path):
    return {
        "out_dir": tmp_path / "plots",
        "vqc_results_file": tmp_path / "vqc.jsonl",
        "classical_results_file": tmp_path / "classical.jsonl",
        "noise_sweep_file": tmp_path / "sweep.jsonl",
        "bc_results_file": tmp_path / "bc.jsonl",
        "bc_classical_file": tmp_path / "bc_classical.jsonl",
        "an_vqc_results_file": tmp_path / "an_vqc_results.jsonl",
        "an_vqc_classical_file": tmp_path / "an_vqc_classical.jsonl",
    }


def _names(report):
    return {p.name for p in report.generated}


def test_circuit_diagram_is_written_and_named(tmp_path):
    path = plot_circuit_diagram(tmp_path, n_qubits=4, depth=2)
    assert path.name == "circuit_4q_depth2.png"
    assert path.exists() and path.stat().st_size > 5_000


def test_all_main_plots_generated_from_synthetic_results(paths):
    _write(paths["vqc_results_file"], _main_rows())
    _write(paths["classical_results_file"], _classical_rows())
    report = generate_all_plots(**paths, noise_sweep_seeds=SEEDS, breast_cancer_seeds=SEEDS)
    expected = {
        "circuit_4q_depth2.png", "01_degradation_vs_noise.png", "02_accuracy_vs_depth.png",
        "03_noise_comparison.png", "04_qubits_vs_performance.png", "05_output_distributions.png",
        "06_classical_vs_qml.png", "08_accuracy_mean_std.png",
    }
    assert expected <= _names(report)
    for p in report.generated:
        assert p.stat().st_size > 2_000
    assert (paths["out_dir"] / "noise_comparison_table.csv").exists()
    assert (paths["out_dir"] / "noise_comparison_table.md").exists()


def test_extension_and_adaptive_plots_skip_gracefully_when_files_missing(paths):
    _write(paths["vqc_results_file"], _main_rows())
    _write(paths["classical_results_file"], _classical_rows())
    report = generate_all_plots(**paths, noise_sweep_seeds=SEEDS, breast_cancer_seeds=SEEDS)
    assert "07_standard_vs_adaptive" in report.skipped
    assert "extension_noise_sweep" in report.skipped
    assert "extension_breast_cancer" in report.skipped
    assert not any(n.startswith("extension_") for n in _names(report))


def test_missing_main_results_skips_data_plots_but_still_draws_circuit(paths):
    report = generate_all_plots(**paths)
    assert "circuit_4q_depth2.png" in _names(report)
    for name in ("01_degradation_vs_noise", "03_noise_comparison", "08_accuracy_mean_std"):
        assert name in report.skipped


def test_noise_comparison_table_values(paths):
    df = pd.DataFrame(_main_rows())
    table = noise_comparison_table(df)
    assert list(table.experiment) == [e[0] for e in EXPERIMENT_MATRIX]
    e4 = table[table.experiment == "E4"].iloc[0]
    # synthetic accuracy for q=4: 0.84 (seed 0) and 0.85 (seed 1)
    assert e4.accuracy_mean == pytest.approx(0.845)
    assert e4.n_seeds == 2
    assert e4.shots == 1024


def test_partial_main_results_skip_plots_needing_missing_experiments(paths):
    rows = [r for r in _main_rows() if r["experiment_id"] in ("E1", "E2", "E3", "E4")]
    _write(paths["vqc_results_file"], rows)
    report = generate_all_plots(**paths)
    assert "01_degradation_vs_noise" in report.skipped  # needs E5-E8
    assert "E5" in report.skipped["01_degradation_vs_noise"] or "E6" in report.skipped["01_degradation_vs_noise"]


# -- extension plots: only when COMPLETE -------------------------------------


def _sweep_rows(seeds, drop_last=False):
    rows = []
    for q, d in NOISE_SWEEP_ARCHITECTURES:
        for p in NOISE_SWEEP_TWO_QUBIT_PROBABILITIES:
            for s in seeds:
                rows.append(
                    {
                        "experiment_id": f"NSWEEP-Q{q}D{d}-P{p:g}", "seed": s, "status": "completed",
                        "architecture": f"{q}q_depth{d}", "two_qubit_probability": p,
                        "accuracy": 0.9 - p, "degradation_pp": 100 * p,
                        "distribution_total_variation_distance": 4 * p,
                    }
                )
    return rows[:-1] if drop_last else rows


def test_noise_sweep_plot_requires_every_pair(paths):
    _write(paths["vqc_results_file"], _main_rows())
    _write(paths["classical_results_file"], _classical_rows())
    _write(paths["noise_sweep_file"], _sweep_rows(SEEDS, drop_last=True))
    report = generate_all_plots(**paths, noise_sweep_seeds=SEEDS, breast_cancer_seeds=SEEDS)
    assert "extension_noise_sweep" in report.skipped
    assert "incomplete" in report.skipped["extension_noise_sweep"]


def test_noise_sweep_plot_drawn_when_complete(paths):
    _write(paths["vqc_results_file"], _main_rows())
    _write(paths["classical_results_file"], _classical_rows())
    _write(paths["noise_sweep_file"], _sweep_rows(SEEDS))
    report = generate_all_plots(**paths, noise_sweep_seeds=SEEDS, breast_cancer_seeds=SEEDS)
    assert "extension_noise_sweep.png" in _names(report)


def _bc_rows(seeds):
    rows = []
    for eid, q, d, noise in EXPERIMENT_MATRIX:
        for s in seeds:
            rows.append(
                {
                    "experiment_id": f"BC-{eid}", "seed": s, "status": "completed", "accuracy": 0.6,
                    "degradation_pp": 0.0 if noise == "ideal" else 2.0,
                    "train_subsample_size": 100, "extension": "breast_cancer",
                }
            )
    return rows


def test_breast_cancer_plot_drawn_only_when_vqc_and_classical_complete(paths):
    _write(paths["vqc_results_file"], _main_rows())
    _write(paths["classical_results_file"], _classical_rows())
    _write(paths["bc_results_file"], _bc_rows(SEEDS))
    # classical missing -> skipped even though VQC rows are complete
    report = generate_all_plots(**paths, noise_sweep_seeds=SEEDS, breast_cancer_seeds=SEEDS)
    assert "extension_breast_cancer" in report.skipped

    _write(paths["bc_classical_file"], _classical_rows(SEEDS))
    report = generate_all_plots(**paths, noise_sweep_seeds=SEEDS, breast_cancer_seeds=SEEDS)
    assert "extension_breast_cancer.png" in _names(report)


def test_breast_cancer_plot_skipped_when_one_seed_missing(paths):
    _write(paths["bc_results_file"], _bc_rows([0]))
    _write(paths["bc_classical_file"], _classical_rows([0]))
    report = generate_all_plots(**paths, noise_sweep_seeds=SEEDS, breast_cancer_seeds=SEEDS)
    assert "incomplete" in report.skipped["extension_breast_cancer"]


# -- AN-VQC (results/an_vqc_results.jsonl + an_vqc_classical_results.jsonl) --

_AN_VQC_QUANTUM_METHODS = ("fixed_vqc", "ideal_best", "shallow_vqc", "an_vqc")


def _an_vqc_rows(seeds=SEEDS, methods=_AN_VQC_QUANTUM_METHODS, conditions=("ideal", "medium")):
    return [
        {"method": m, "seed": s, "noise_condition": c, "accuracy": 0.9, "status": "completed"}
        for m in methods for s in seeds for c in conditions
    ]


def _an_vqc_classical_rows(seeds=SEEDS):
    return [
        {"model_name": m, "seed": s, "accuracy": 1.0}
        for m in ("logistic_regression", "svm") for s in seeds
    ]


def test_an_vqc_plot_drawn_with_real_schema(tmp_path):
    df = pd.DataFrame(_an_vqc_rows())
    cl = pd.DataFrame(_an_vqc_classical_rows())
    path = plot_standard_vs_adaptive(df, cl, tmp_path, "an_vqc_results.jsonl")
    assert path.name == "07_standard_vs_adaptive.png" and path.exists()


def test_an_vqc_plot_skips_on_empty_results(tmp_path):
    with pytest.raises(PlotSkipped):
        plot_standard_vs_adaptive(pd.DataFrame(), pd.DataFrame(_an_vqc_classical_rows()), tmp_path, "x")


def test_an_vqc_plot_skips_on_missing_columns(tmp_path):
    bad = pd.DataFrame([{"method": "an_vqc", "seed": 0}])
    with pytest.raises(PlotSkipped):
        plot_standard_vs_adaptive(bad, pd.DataFrame(_an_vqc_classical_rows()), tmp_path, "x")


def test_an_vqc_plot_skips_when_a_method_is_missing():
    rows = _an_vqc_rows(methods=("fixed_vqc", "ideal_best", "shallow_vqc"))  # an_vqc missing
    with pytest.raises(PlotSkipped):
        plot_standard_vs_adaptive(pd.DataFrame(rows), pd.DataFrame(_an_vqc_classical_rows()), Path("."), "x")


def test_an_vqc_plot_skips_when_a_condition_is_missing():
    rows = _an_vqc_rows(conditions=("ideal",))  # medium missing for every method
    with pytest.raises(PlotSkipped):
        plot_standard_vs_adaptive(pd.DataFrame(rows), pd.DataFrame(_an_vqc_classical_rows()), Path("."), "x")


def test_an_vqc_plot_skips_on_empty_classical(tmp_path):
    with pytest.raises(PlotSkipped):
        plot_standard_vs_adaptive(pd.DataFrame(_an_vqc_rows()), pd.DataFrame(), tmp_path, "x")


# -- real saved results (read-only) ------------------------------------------


def test_real_results_produce_main_plots_without_modifying_them(tmp_path):
    if not VQC_RESULTS_FILE.exists():
        pytest.skip("real E1-E8 results not present")
    before = VQC_RESULTS_FILE.read_bytes()
    report = generate_all_plots(tmp_path / "plots")
    assert VQC_RESULTS_FILE.read_bytes() == before
    assert {"01_degradation_vs_noise.png", "08_accuracy_mean_std.png"} <= _names(report)


def test_extensions_can_require_independent_seed_counts(tmp_path):
    """Two extensions finishing with different seed counts (e.g. a 3-seed noise
    sweep alongside a 5-seed Breast Cancer run) must each be checked against
    their own actual seed count, not one shared value."""
    _write(tmp_path_vqc := tmp_path / "vqc.jsonl", _main_rows())
    _write(tmp_path / "classical.jsonl", _classical_rows())
    sweep_file = tmp_path / "sweep.jsonl"
    bc_file = tmp_path / "bc.jsonl"
    bc_classical_file = tmp_path / "bc_classical.jsonl"
    _write(sweep_file, _sweep_rows([0, 1, 2]))  # only 3 seeds
    _write(bc_file, _bc_rows([0, 1, 2, 3, 4]))  # 5 seeds
    _write(bc_classical_file, _classical_rows([0, 1, 2, 3, 4]))

    report = generate_all_plots(
        tmp_path / "plots",
        vqc_results_file=tmp_path_vqc,
        classical_results_file=tmp_path / "classical.jsonl",
        noise_sweep_file=sweep_file,
        bc_results_file=bc_file,
        bc_classical_file=bc_classical_file,
        noise_sweep_seeds=[0, 1, 2],
        breast_cancer_seeds=[0, 1, 2, 3, 4],
    )
    assert "extension_noise_sweep.png" in _names(report)
    assert "extension_breast_cancer.png" in _names(report)
    assert "extension_noise_sweep" not in report.skipped
    assert "extension_breast_cancer" not in report.skipped


def test_mismatched_seed_requirement_still_correctly_skips(tmp_path):
    """Sanity check of the fix: requiring 5 seeds from a genuinely 3-seed
    extension must still report it incomplete, not silently pass."""
    sweep_file = tmp_path / "sweep.jsonl"
    _write(sweep_file, _sweep_rows([0, 1, 2]))
    report = generate_all_plots(
        tmp_path / "plots",
        vqc_results_file=tmp_path / "missing_vqc.jsonl",
        classical_results_file=tmp_path / "missing_classical.jsonl",
        noise_sweep_file=sweep_file,
        bc_results_file=tmp_path / "missing_bc.jsonl",
        bc_classical_file=tmp_path / "missing_bc_classical.jsonl",
        noise_sweep_seeds=[0, 1, 2, 3, 4],  # wrong on purpose -- sweep only has 0,1,2
    )
    assert "incomplete" in report.skipped["extension_noise_sweep"]
