"""Tests for the experiment runner: matrix integrity, resume, and smoke runs.

These tests always redirect result files to `tmp_path` so they never write
into the real project `results/` directory.
"""

import json

import pytest

from qml_noise.experiments import (
    EXPERIMENT_MATRIX,
    MODE_MAXITER,
    MODE_SEEDS,
    MODES,
    build_arg_parser,
    group_matrix_by_architecture,
    run_classical_baselines_for_seed,
    run_matrix,
)


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# A tiny 2-row, 2-qubit matrix for exercising resume/force bookkeeping
# quickly, without paying for the real matrix's 6-qubit architecture.
_TINY_MATRIX = (
    ("T1", 2, 1, "ideal"),
    ("T2", 2, 1, "medium"),
)


# -- matrix integrity --------------------------------------------------------


def test_matrix_matches_the_assignment_table_exactly():
    """Guards against ever silently replacing the mandatory E1-E8 matrix."""
    assert EXPERIMENT_MATRIX == (
        ("E1", 2, 2, "ideal"),
        ("E2", 2, 4, "ideal"),
        ("E3", 4, 2, "ideal"),
        ("E4", 4, 4, "ideal"),
        ("E5", 4, 4, "low"),
        ("E6", 4, 4, "medium"),
        ("E7", 6, 4, "ideal"),
        ("E8", 6, 4, "medium"),
    )


def test_matrix_has_at_least_six_experiments_and_two_varied_parameters():
    assert len(EXPERIMENT_MATRIX) >= 6
    qubit_values = {row[1] for row in EXPERIMENT_MATRIX}
    depth_values = {row[2] for row in EXPERIMENT_MATRIX}
    noise_values = {row[3] for row in EXPERIMENT_MATRIX}
    varied_params = sum(len(v) > 1 for v in (qubit_values, depth_values, noise_values))
    assert varied_params >= 2


def test_group_matrix_by_architecture():
    groups = group_matrix_by_architecture()
    assert groups[(2, 2)] == [("E1", 2, 2, "ideal")]
    assert groups[(2, 4)] == [("E2", 2, 4, "ideal")]
    assert groups[(4, 2)] == [("E3", 4, 2, "ideal")]
    assert groups[(4, 4)] == [
        ("E4", 4, 4, "ideal"),
        ("E5", 4, 4, "low"),
        ("E6", 4, 4, "medium"),
    ]
    assert groups[(6, 4)] == [("E7", 6, 4, "ideal"), ("E8", 6, 4, "medium")]
    assert sum(len(v) for v in groups.values()) == len(EXPERIMENT_MATRIX)


def test_every_group_has_exactly_one_ideal_row():
    """Every noisy row needs an ideal row in its group to reuse training from."""
    for (n_qubits, depth), rows in group_matrix_by_architecture().items():
        ideal_rows = [r for r in rows if r[3] == "ideal"]
        assert len(ideal_rows) == 1, f"group ({n_qubits},{depth}) has {len(ideal_rows)} ideal rows"


def test_modes_are_consistent():
    assert set(MODE_SEEDS) == set(MODES) == set(MODE_MAXITER)
    assert MODE_SEEDS["full"] == [0, 1, 2, 3, 4]
    assert len(MODE_SEEDS["smoke"]) == 1
    assert len(MODE_SEEDS["normal"]) == 1
    assert MODE_MAXITER["smoke"] < MODE_MAXITER["normal"]


# -- CLI parsing --------------------------------------------------------------


def test_arg_parser_requires_mode():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_arg_parser_accepts_overrides():
    parser = build_arg_parser()
    args = parser.parse_args(["--mode", "smoke", "--seeds", "0,1", "--maxiter", "5", "--force"])
    assert args.mode == "smoke"
    assert args.seeds == "0,1"
    assert args.maxiter == 5
    assert args.force is True


def test_arg_parser_rejects_unknown_mode():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "ludicrous"])


# -- classical baselines ------------------------------------------------------


def test_classical_baselines_saved_and_resumable(tmp_path):
    results_file = tmp_path / "classical.jsonl"
    saved = run_classical_baselines_for_seed(0, "iris", results_file)
    assert {r["model_name"] for r in saved} == {"logistic_regression", "svm"}

    records = _read_jsonl(results_file)
    assert len(records) == 2

    # Second call with the same seed must skip (already completed).
    saved_again = run_classical_baselines_for_seed(0, "iris", results_file)
    assert saved_again == []
    assert len(_read_jsonl(results_file)) == 2


# -- full smoke run (real pipeline, tiny budget, isolated files) -------------


@pytest.fixture(scope="module")
def smoke_run_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("smoke_results")
    results_file = tmp_path / "vqc.jsonl"
    results_csv = tmp_path / "vqc.csv"
    classical_file = tmp_path / "classical.jsonl"
    new_records = run_matrix(
        "smoke",
        seeds=[0],
        maxiter=5,
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_file,
    )
    return {
        "results_file": results_file,
        "results_csv": results_csv,
        "classical_file": classical_file,
        "new_records": new_records,
    }


def test_smoke_run_produces_one_record_per_matrix_row(smoke_run_dir):
    records = _read_jsonl(smoke_run_dir["results_file"])
    assert len(records) == len(EXPERIMENT_MATRIX) == 8
    assert {r["experiment_id"] for r in records} == {row[0] for row in EXPERIMENT_MATRIX}
    assert all(r["status"] == "completed" for r in records)
    assert all(r["seed"] == 0 for r in records)


def test_smoke_run_writes_summary_csv(smoke_run_dir):
    csv_path = smoke_run_dir["results_csv"]
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "experiment_id" in content.splitlines()[0]
    # Nested fields must not leak into the flat CSV.
    assert "confusion_matrix" not in content.splitlines()[0]
    assert "pooled_output_distribution" not in content.splitlines()[0]
    assert len(content.splitlines()) == 1 + 8  # header + 8 rows


def test_smoke_run_saves_classical_baselines_too(smoke_run_dir):
    records = _read_jsonl(smoke_run_dir["classical_file"])
    assert {r["model_name"] for r in records} == {"logistic_regression", "svm"}


def test_smoke_run_noisy_rows_reused_training(smoke_run_dir):
    records = {r["experiment_id"]: r for r in _read_jsonl(smoke_run_dir["results_file"])}
    assert records["E5"]["reused_training_from"] == "E4"
    assert records["E6"]["reused_training_from"] == "E4"
    assert records["E8"]["reused_training_from"] == "E7"
    assert records["E4"]["reused_training_from"] is None
    assert records["E5"]["training_time_seconds"] == 0.0
    assert records["E4"]["training_time_seconds"] > 0.0


def test_smoke_run_all_section_10_metrics_present(smoke_run_dir):
    required_fields = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
        "accuracy_ideal_baseline",
        "degradation_pp",
        "relative_degradation_pct",
        "robustness_ratio",
        "n_qubits",
        "logical_depth",
        "transpiled_depth",
        "total_gate_count",
        "two_qubit_gate_count",
        "n_trainable_parameters",
        "shots",
        "runtime_seconds",
        "pooled_output_distribution",
        "distribution_total_variation_distance",
    }
    for record in _read_jsonl(smoke_run_dir["results_file"]):
        assert required_fields <= set(record), f"missing fields in {record['experiment_id']}"
        assert record["shots"] == 1024


# -- resume behavior (separate isolated run) ---------------------------------


def test_resume_skips_already_completed_pairs(tmp_path):
    results_file = tmp_path / "vqc.jsonl"
    results_csv = tmp_path / "vqc.csv"
    classical_file = tmp_path / "classical.jsonl"

    first = run_matrix(
        "smoke",
        seeds=[0],
        maxiter=5,
        matrix=_TINY_MATRIX,
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_file,
    )
    assert len(first) == 2

    second = run_matrix(
        "smoke",
        seeds=[0],
        maxiter=5,
        matrix=_TINY_MATRIX,
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_file,
    )
    assert second == []
    assert len(_read_jsonl(results_file)) == 2  # no duplicates


def test_force_rerun_deletes_existing_results_first(tmp_path):
    results_file = tmp_path / "vqc.jsonl"
    results_csv = tmp_path / "vqc.csv"
    classical_file = tmp_path / "classical.jsonl"

    run_matrix(
        "smoke",
        seeds=[0],
        maxiter=5,
        matrix=_TINY_MATRIX,
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_file,
    )
    assert len(_read_jsonl(results_file)) == 2

    rerun = run_matrix(
        "smoke",
        seeds=[0],
        maxiter=5,
        matrix=_TINY_MATRIX,
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_file,
        force=True,
    )
    assert len(rerun) == 2
    assert len(_read_jsonl(results_file)) == 2  # overwritten, not duplicated


def test_partial_completion_only_runs_missing_pairs(tmp_path):
    """Manually seed the file with 1/2 rows completed; only the 2nd should run."""
    results_file = tmp_path / "vqc.jsonl"
    results_csv = tmp_path / "vqc.csv"
    classical_file = tmp_path / "classical.jsonl"

    full = run_matrix(
        "smoke",
        seeds=[0],
        maxiter=5,
        matrix=_TINY_MATRIX,
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_file,
    )
    assert len(full) == 2

    # Remove the last line (T2) to simulate an interrupted run.
    lines = results_file.read_text().splitlines()
    results_file.write_text("\n".join(lines[:-1]) + "\n")
    assert len(_read_jsonl(results_file)) == 1

    resumed = run_matrix(
        "smoke",
        seeds=[0],
        maxiter=5,
        matrix=_TINY_MATRIX,
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_file,
    )
    assert len(resumed) == 1
    assert resumed[0]["experiment_id"] == "T2"
    assert len(_read_jsonl(results_file)) == 2


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        run_matrix("ludicrous")


def test_empty_seeds_raises(tmp_path):
    with pytest.raises(ValueError):
        run_matrix("smoke", seeds=[], results_file=tmp_path / "x.jsonl")


# -- generalization hooks: prepare_split / experiment_id_prefix / extra_record_fields --
# These exist so qml_noise.extensions can reuse run_matrix for a different
# dataset (Breast Cancer) without duplicating its grouping/resume logic.


def test_experiment_id_prefix_applies_to_saved_ids_and_resume_keys(tmp_path):
    results_file = tmp_path / "vqc.jsonl"
    results_csv = tmp_path / "vqc.csv"
    classical_file = tmp_path / "classical.jsonl"

    records = run_matrix(
        "smoke",
        seeds=[0],
        maxiter=5,
        matrix=_TINY_MATRIX,
        experiment_id_prefix="X-",
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_file,
    )
    ids = {r["experiment_id"] for r in records}
    assert ids == {"X-T1", "X-T2"}

    by_id = {r["experiment_id"]: r for r in records}
    assert by_id["X-T1"]["reused_training_from"] is None
    assert by_id["X-T2"]["reused_training_from"] == "X-T1"

    # Resume must recognize the prefixed ids already on disk.
    resumed = run_matrix(
        "smoke",
        seeds=[0],
        maxiter=5,
        matrix=_TINY_MATRIX,
        experiment_id_prefix="X-",
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_file,
    )
    assert resumed == []


def test_extra_record_fields_merged_into_vqc_and_classical_records(tmp_path):
    results_file = tmp_path / "vqc.jsonl"
    results_csv = tmp_path / "vqc.csv"
    classical_file = tmp_path / "classical.jsonl"

    run_matrix(
        "smoke",
        seeds=[0],
        maxiter=5,
        matrix=_TINY_MATRIX,
        extra_record_fields={"extension": "unit_test"},
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_file,
    )

    for record in _read_jsonl(results_file):
        assert record["extension"] == "unit_test"
    for record in _read_jsonl(classical_file):
        assert record["extension"] == "unit_test"


def test_prepare_split_hook_is_applied_before_training_and_classical_baselines(tmp_path):
    results_file = tmp_path / "vqc.jsonl"
    results_csv = tmp_path / "vqc.csv"
    classical_file = tmp_path / "classical.jsonl"
    calls = []

    def _spy_prepare(split, seed):
        calls.append((split.X_train.shape[0], seed))
        return split

    run_matrix(
        "smoke",
        seeds=[0],
        maxiter=5,
        matrix=_TINY_MATRIX,
        prepare_split=_spy_prepare,
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_file,
    )
    assert len(calls) == 1
    assert calls[0][1] == 0  # seed passed through
    assert calls[0][0] > 0  # a real training-set size was seen
