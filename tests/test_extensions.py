"""Tests for the extension experiments: split check, pooled analysis,
noise sweep, and the Breast Cancer extension.

All result files are redirected to tmp_path -- these tests never write
into (or read from, except where explicitly noted and read-only) the real
project `results/` directory.
"""

import json

import numpy as np
import pytest

from qml_noise.extensions import (
    EXTENSION_MODES,
    build_arg_parser,
    check_split_uniqueness,
    pooled_analysis,
    run_breast_cancer_extension,
    run_noise_sweep_extension,
)


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# -- 1a. split-uniqueness check ----------------------------------------------


def test_split_uniqueness_finds_distinct_splits_across_real_seeds():
    result = check_split_uniqueness()
    assert result["dataset"] == "iris"
    assert result["seeds_checked"] == [0, 1, 2, 3, 4]
    assert result["all_splits_identical"] is False
    assert len(result["pairwise_comparisons"]) == 10  # C(5,2)
    assert all(not p["identical"] for p in result["pairwise_comparisons"])
    assert "DIFFERENT" in result["finding"]


def test_split_uniqueness_detects_identical_splits_for_repeated_seed():
    """Sanity check of the detector itself: same seed twice must compare identical."""
    result = check_split_uniqueness(seeds=[0, 0])
    assert result["all_splits_identical"] is True
    assert result["pairwise_comparisons"][0]["identical"] is True
    assert result["pairwise_comparisons"][0]["jaccard_similarity"] == pytest.approx(1.0)


def test_split_uniqueness_jaccard_in_valid_range():
    result = check_split_uniqueness()
    for pair in result["pairwise_comparisons"]:
        assert 0.0 <= pair["jaccard_similarity"] <= 1.0


# -- 1b. pooled analysis ------------------------------------------------------


def _write_synthetic_vqc_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _synthetic_row(experiment_id, seed, degradation_pp, cm):
    return {
        "experiment_id": experiment_id,
        "seed": seed,
        "confusion_matrix": cm,
        "degradation_pp": degradation_pp,
        "status": "completed",
    }


def test_pooled_analysis_on_synthetic_results(tmp_path):
    results_file = tmp_path / "synthetic_vqc.jsonl"
    # Perfect confusion matrices except a few misclassifications, 2 classes,
    # 10 test samples each.
    rows = [
        _synthetic_row("E4", 0, 0.0, [[5, 0], [0, 5]]),
        _synthetic_row("E4", 1, 0.0, [[5, 0], [0, 5]]),
        _synthetic_row("E5", 0, 10.0, [[5, 0], [1, 4]]),
        _synthetic_row("E5", 1, -10.0, [[5, 0], [0, 5]]),
    ]
    _write_synthetic_vqc_results(results_file, rows)

    result = pooled_analysis(results_file=results_file, noisy_experiments=("E5",), n_bootstrap=500)

    assert result["n_source_records"] == 4
    # E4: 2 seeds x 10 correct / 20 total = 1.0
    assert result["pooled_accuracy_per_experiment"]["E4"]["pooled_accuracy"] == pytest.approx(1.0)
    # E5: (9 correct out of 10) + (10 correct out of 10) = 19/20
    assert result["pooled_accuracy_per_experiment"]["E5"]["pooled_accuracy"] == pytest.approx(0.95)

    paired = result["paired_degradation"]["E5"]
    assert paired["seeds"] == [0, 1]
    assert paired["degradation_pp"] == [10.0, -10.0]
    assert paired["mean"] == pytest.approx(0.0)

    ci = result["bootstrap_ci_degradation_pp"]["E5"]
    assert ci["mean_degradation_pp"] == pytest.approx(0.0)
    assert ci["ci95_low"] <= ci["mean_degradation_pp"] <= ci["ci95_high"]
    assert ci["n_bootstrap"] == 500


def test_pooled_analysis_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pooled_analysis(results_file=tmp_path / "does_not_exist.jsonl")


def test_pooled_analysis_skips_experiments_with_no_records(tmp_path, caplog):
    results_file = tmp_path / "synthetic_vqc.jsonl"
    rows = [_synthetic_row("E4", 0, 0.0, [[5, 0], [0, 5]])]
    _write_synthetic_vqc_results(results_file, rows)

    result = pooled_analysis(
        results_file=results_file, noisy_experiments=("E5", "E6", "E8"), n_bootstrap=100
    )
    assert result["paired_degradation"] == {}
    assert result["bootstrap_ci_degradation_pp"] == {}


def test_pooled_analysis_reads_the_real_existing_results_without_writing(tmp_path):
    """Integration check against the real project results -- read-only."""
    from qml_noise.experiments import VQC_RESULTS_FILE

    if not VQC_RESULTS_FILE.exists():
        pytest.skip("real E1-E8 results not present in this environment")

    before = VQC_RESULTS_FILE.read_text()
    result = pooled_analysis(n_bootstrap=200)
    after = VQC_RESULTS_FILE.read_text()

    assert before == after  # never modified
    assert result["n_source_records"] > 0
    assert set(result["pooled_accuracy_per_experiment"]) <= {
        "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
    }


# -- 2. noise sweep (small synthetic config for speed) -----------------------


@pytest.fixture(scope="module")
def tiny_sweep_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sweep")
    results_file = tmp / "sweep.jsonl"
    records = run_noise_sweep_extension(
        "smoke",
        seeds=[0],
        maxiter=5,
        architectures=((2, 2),),
        two_qubit_probabilities=(0.0, 0.05),
        results_file=results_file,
    )
    return {"results_file": results_file, "records": records}


def test_noise_sweep_produces_one_record_per_probability_level(tiny_sweep_dir):
    records = tiny_sweep_dir["records"]
    assert len(records) == 2
    ids = {r["experiment_id"] for r in records}
    assert ids == {"NSWEEP-Q2D2-P0", "NSWEEP-Q2D2-P0.05"}


def test_noise_sweep_records_are_labeled_as_extension(tiny_sweep_dir):
    for r in tiny_sweep_dir["records"]:
        assert r["extension"] == "noise_sweep"
        assert r["architecture"] == "2q_depth2"
        assert r["shots"] == 1024


def test_noise_sweep_baseline_has_zero_probability_and_no_reuse(tiny_sweep_dir):
    by_id = {r["experiment_id"]: r for r in tiny_sweep_dir["records"]}
    baseline = by_id["NSWEEP-Q2D2-P0"]
    assert baseline["two_qubit_probability"] == 0.0
    assert baseline["single_qubit_probability"] == 0.0
    assert baseline["reused_training_from"] is None
    assert baseline["training_time_seconds"] > 0.0


def test_noise_sweep_noisy_level_reuses_baseline_training(tiny_sweep_dir):
    by_id = {r["experiment_id"]: r for r in tiny_sweep_dir["records"]}
    noisy = by_id["NSWEEP-Q2D2-P0.05"]
    assert noisy["reused_training_from"] == "NSWEEP-Q2D2-P0"
    assert noisy["training_time_seconds"] == 0.0
    assert noisy["two_qubit_probability"] == pytest.approx(0.05)
    assert noisy["single_qubit_probability"] == pytest.approx(0.005)


def test_noise_sweep_resume_skips_completed(tmp_path):
    results_file = tmp_path / "sweep.jsonl"
    kwargs = dict(
        seeds=[0],
        maxiter=5,
        architectures=((2, 2),),
        two_qubit_probabilities=(0.0, 0.05),
        results_file=results_file,
    )
    first = run_noise_sweep_extension("smoke", **kwargs)
    assert len(first) == 2
    second = run_noise_sweep_extension("smoke", **kwargs)
    assert second == []
    assert len(_read_jsonl(results_file)) == 2


def test_noise_sweep_invalid_mode_raises():
    with pytest.raises(ValueError):
        run_noise_sweep_extension("ludicrous")


# -- 3. breast cancer extension (one real, small integration run) -----------


@pytest.fixture(scope="module")
def bc_run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bc")
    results_file = tmp / "bc.jsonl"
    results_csv = tmp / "bc.csv"
    classical_file = tmp / "bc_classical.jsonl"
    records = run_breast_cancer_extension(
        "smoke",
        seeds=[0],
        maxiter=5,
        n_train_samples=20,
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_file,
    )
    return {
        "results_file": results_file,
        "results_csv": results_csv,
        "classical_file": classical_file,
        "records": records,
    }


def test_bc_extension_produces_all_eight_prefixed_experiments(bc_run_dir):
    records = bc_run_dir["records"]
    assert len(records) == 8
    ids = {r["experiment_id"] for r in records}
    assert ids == {f"BC-E{i}" for i in range(1, 9)}


def test_bc_extension_uses_breast_cancer_dataset_and_full_test_set(bc_run_dir):
    for r in bc_run_dir["records"]:
        assert r["dataset"] == "breast_cancer"
        assert r["shots"] == 1024
        # Full held-out Breast Cancer test set is ~114 samples (20% of 569);
        # confusion matrix must sum to that, not to the 20-sample subsample.
        total = sum(sum(row) for row in r["confusion_matrix"])
        assert total > 50  # definitely not the 20-sample training subsample


def test_bc_extension_records_labeled_as_extension(bc_run_dir):
    for r in bc_run_dir["records"]:
        assert r["extension"] == "breast_cancer"
        assert r["train_subsample_size"] == 20


def test_bc_extension_classical_baselines_use_same_subsample(bc_run_dir):
    records = _read_jsonl(bc_run_dir["classical_file"])
    assert {r["model_name"] for r in records} == {"logistic_regression", "svm"}
    for r in records:
        assert r["dataset_name"] == "breast_cancer"
        assert r["extension"] == "breast_cancer"
        assert r["train_subsample_size"] == 20


def test_bc_extension_writes_summary_csv(bc_run_dir):
    assert bc_run_dir["results_csv"].exists()


def test_bc_extension_resume_skips_completed(bc_run_dir):
    # Rerunning with the same files must find everything already completed.
    second = run_breast_cancer_extension(
        "smoke",
        seeds=[0],
        maxiter=5,
        n_train_samples=20,
        results_file=bc_run_dir["results_file"],
        results_csv=bc_run_dir["results_csv"],
        classical_results_file=bc_run_dir["classical_file"],
    )
    assert second == []


def test_bc_extension_never_touches_real_results_files(bc_run_dir):
    from qml_noise.experiments import CLASSICAL_RESULTS_FILE, VQC_RESULTS_FILE

    assert bc_run_dir["results_file"] != VQC_RESULTS_FILE
    assert bc_run_dir["classical_file"] != CLASSICAL_RESULTS_FILE


def test_bc_extension_invalid_mode_raises():
    with pytest.raises(ValueError):
        run_breast_cancer_extension("ludicrous")


# -- CLI ----------------------------------------------------------------------


def test_extension_modes_are_smoke_and_full_only():
    assert EXTENSION_MODES == ("smoke", "full")


def test_arg_parser_requires_a_subcommand():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_arg_parser_split_check():
    parser = build_arg_parser()
    args = parser.parse_args(["split-check"])
    assert args.extension == "split-check"


def test_arg_parser_noise_sweep_requires_mode():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["noise-sweep"])
    args = parser.parse_args(["noise-sweep", "--mode", "smoke"])
    assert args.mode == "smoke"


def test_arg_parser_breast_cancer_accepts_train_samples():
    parser = build_arg_parser()
    args = parser.parse_args(["breast-cancer", "--mode", "full", "--train-samples", "150"])
    assert args.train_samples == 150
