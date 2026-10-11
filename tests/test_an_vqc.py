"""Tests for the Adaptive Noise-Aware VQC (AN-VQC) module.

All result files are redirected to tmp_path -- these tests never write into
(or read from) the real project `results/` directory, and the existing
E1-E8 / extension result files are never opened by this module at all.
"""

import json
import math

import numpy as np
import pytest

from qml_noise.an_vqc import (
    AN_VQC_MODES,
    METHOD_ANVQC,
    METHOD_FIXED,
    METHOD_IDEAL_BEST,
    METHOD_SHALLOW,
    QUANTUM_METHODS,
    ANVQCSelection,
    CandidateValidationResult,
    _method_architectures,
    build_arg_parser,
    candidate_normalized_costs,
    run_an_vqc_comparison,
    select_candidate,
)
from qml_noise.config import (
    AN_VQC_CANDIDATES,
    AN_VQC_FIXED_BASELINE,
    AN_VQC_SHALLOW_BASELINE,
)
from qml_noise.datasets import load_dataset


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# A reduced 3-candidate set for tests that would otherwise repeatedly pay for
# the 6-qubit/depth-4 candidate (COBYLA's n_params+2 floor forces ~50
# iterations regardless of a tiny maxiter override). Exactly one test below
# (the `selection` fixture) exercises the real, full 6-candidate default.
_FAST_CANDIDATES = ((2, 2), (2, 4), (4, 2))


# -- resource cost ------------------------------------------------------------


def test_candidate_costs_are_min_max_normalized():
    costs = candidate_normalized_costs()
    assert set(costs) == set(AN_VQC_CANDIDATES)
    assert min(costs.values()) == pytest.approx(0.0)
    assert max(costs.values()) == pytest.approx(1.0)
    # More qubits/depth -> more gates -> higher cost.
    assert costs[(2, 2)] < costs[(4, 4)] < costs[(6, 4)]


def test_candidate_costs_degenerate_case_returns_zero():
    costs = candidate_normalized_costs(candidates=((2, 2),))
    assert costs == {(2, 2): 0.0}


# -- selection never touches test data ----------------------------------------


class _Landmine:
    """Raises on any access, proving a value was never read."""

    def __getattr__(self, name):
        raise AssertionError(f"test data was accessed: .{name}")

    def __array__(self, *a, **k):
        raise AssertionError("test data was accessed via __array__ (e.g. passed to numpy)")

    def __len__(self):
        raise AssertionError("test data length was accessed")

    def __getitem__(self, i):
        raise AssertionError("test data was indexed")

    def __iter__(self):
        raise AssertionError("test data was iterated")

    def __eq__(self, other):
        raise AssertionError("test data was compared")


def test_selection_never_touches_test_data():
    """Direct proof: substituting a landmine for X_test/y_test must not raise."""
    from dataclasses import replace

    real_split = load_dataset("iris", seed=0)
    landmine_split = replace(real_split, X_test=_Landmine(), y_test=_Landmine())

    # If select_candidate (or anything it calls) ever reads split.X_test or
    # split.y_test, the landmine raises AssertionError and this test fails.
    selection = select_candidate(landmine_split, seed=0, candidates=_FAST_CANDIDATES, maxiter=8)

    assert isinstance(selection, ANVQCSelection)
    assert len(selection.candidates) == len(_FAST_CANDIDATES)
    assert selection.selected in _FAST_CANDIDATES
    assert selection.ideal_best in _FAST_CANDIDATES


def test_carve_validation_itself_never_touches_test_data():
    from dataclasses import replace

    from qml_noise.datasets import carve_validation

    real_split = load_dataset("iris", seed=0)
    landmine_split = replace(real_split, X_test=_Landmine(), y_test=_Landmine())
    val_split = carve_validation(landmine_split, val_fraction=0.25, seed=0)
    assert val_split.X_train.shape[0] == 60  # 80 * 0.75
    assert val_split.X_test.shape[0] == 20  # the carved-out validation set


# -- select_candidate (real, small, fast) -------------------------------------


@pytest.fixture(scope="module")
def iris_split():
    return load_dataset("iris", seed=0)


@pytest.fixture(scope="module")
def selection(iris_split):
    return select_candidate(iris_split, seed=0, maxiter=8)


def test_selection_covers_all_six_candidates(selection):
    assert len(selection.candidates) == 6
    assert {(c.n_qubits, c.depth) for c in selection.candidates} == set(AN_VQC_CANDIDATES)


def test_selection_picks_from_candidates(selection):
    assert selection.selected in AN_VQC_CANDIDATES
    assert selection.ideal_best in AN_VQC_CANDIDATES


def test_selection_score_formula(selection):
    for c in selection.candidates:
        robustness = 0.0 if math.isnan(c.robustness) else c.robustness
        expected = (
            selection.alpha * c.val_accuracy_ideal
            + selection.beta * robustness
            - selection.gamma * c.normalized_cost
        )
        assert c.score == pytest.approx(expected)


def test_ideal_best_has_highest_validation_accuracy(selection):
    best = max(c.val_accuracy_ideal for c in selection.candidates)
    picked = next(c for c in selection.candidates if (c.n_qubits, c.depth) == selection.ideal_best)
    assert picked.val_accuracy_ideal == pytest.approx(best)


def test_selected_has_highest_score(selection):
    best = max(c.score for c in selection.candidates)
    picked = next(c for c in selection.candidates if (c.n_qubits, c.depth) == selection.selected)
    assert picked.score == pytest.approx(best)


def test_selection_to_dict_roundtrip(selection):
    d = selection.to_dict()
    json.dumps(d)  # must not raise
    restored = ANVQCSelection.from_dict(d)
    assert restored.selected == selection.selected
    assert restored.ideal_best == selection.ideal_best
    assert len(restored.candidates) == len(selection.candidates)


def test_different_seed_may_give_different_selection():
    sel0 = select_candidate(load_dataset("iris", seed=0), seed=0, candidates=_FAST_CANDIDATES, maxiter=8)
    sel1 = select_candidate(load_dataset("iris", seed=1), seed=1, candidates=_FAST_CANDIDATES, maxiter=8)
    # Not asserting they differ (they might coincide); just that both are valid.
    assert sel0.selected in _FAST_CANDIDATES
    assert sel1.selected in _FAST_CANDIDATES


# -- method architecture mapping ----------------------------------------------


def test_method_architectures_uses_frozen_baselines(selection):
    arch = _method_architectures(selection)
    assert set(arch) == set(QUANTUM_METHODS)
    assert arch[METHOD_FIXED] == AN_VQC_FIXED_BASELINE
    assert arch[METHOD_SHALLOW] == AN_VQC_SHALLOW_BASELINE
    assert arch[METHOD_IDEAL_BEST] == selection.ideal_best
    assert arch[METHOD_ANVQC] == selection.selected


# -- full comparison run (real, small, isolated files) ------------------------


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("an_vqc_smoke")
    selection_file = tmp / "selection.jsonl"
    results_file = tmp / "results.jsonl"
    classical_file = tmp / "classical.jsonl"
    records = run_an_vqc_comparison(
        "smoke",
        seeds=[0],
        maxiter=8,
        candidates=_FAST_CANDIDATES,
        selection_file=selection_file,
        results_file=results_file,
        classical_file=classical_file,
    )
    return {
        "selection_file": selection_file,
        "results_file": results_file,
        "classical_file": classical_file,
        "records": records,
    }


def test_smoke_run_writes_one_selection_row(smoke_run):
    rows = _read_jsonl(smoke_run["selection_file"])
    assert len(rows) == 1
    assert rows[0]["seed"] == 0
    # smoke_run uses _FAST_CANDIDATES (3), not the full AN_VQC_CANDIDATES (6).
    assert len(rows[0]["candidates"]) == len(_FAST_CANDIDATES)


def test_smoke_run_writes_classical_baselines(smoke_run):
    rows = _read_jsonl(smoke_run["classical_file"])
    assert {r["model_name"] for r in rows} == {"logistic_regression", "svm"}
    assert all(r["method"] == r["model_name"] for r in rows)


def test_smoke_run_writes_quantum_methods_ideal_and_noisy(smoke_run):
    rows = _read_jsonl(smoke_run["results_file"])
    methods = {r["method"] for r in rows}
    assert methods <= set(QUANTUM_METHODS)
    assert len(methods) >= 1  # at least some methods ran (dedup via reuse is allowed)
    conditions = {r["noise_condition"] for r in rows}
    assert conditions == {"ideal", "medium"}
    for r in rows:
        assert r["shots"] == 1024
        assert r["status"] == "completed"
        assert "selected_qubits" in r and "selected_depth" in r


def test_smoke_run_dedups_shared_architectures(smoke_run):
    """If two methods share an architecture, only one pays real training time."""
    rows = _read_jsonl(smoke_run["results_file"])
    by_arch_ideal = {}
    for r in rows:
        if r["noise_condition"] != "ideal":
            continue
        arch = (r["selected_qubits"], r["selected_depth"])
        by_arch_ideal.setdefault(arch, []).append(r)
    for arch, group in by_arch_ideal.items():
        real_training = [r for r in group if r["reused_training_from"] is None]
        assert len(real_training) == 1, f"architecture {arch} trained more than once: {group}"


def test_smoke_run_resume_skips_everything_on_rerun(smoke_run):
    second = run_an_vqc_comparison(
        "smoke",
        seeds=[0],
        maxiter=8,
        candidates=_FAST_CANDIDATES,
        selection_file=smoke_run["selection_file"],
        results_file=smoke_run["results_file"],
        classical_file=smoke_run["classical_file"],
    )
    assert second == []
    assert len(_read_jsonl(smoke_run["selection_file"])) == 1


def test_force_rerun_deletes_existing_results(tmp_path):
    kwargs = dict(
        seeds=[0],
        maxiter=8,
        candidates=_FAST_CANDIDATES,
        selection_file=tmp_path / "sel.jsonl",
        results_file=tmp_path / "res.jsonl",
        classical_file=tmp_path / "cls.jsonl",
    )
    run_an_vqc_comparison("smoke", **kwargs)
    first_len = len(_read_jsonl(kwargs["selection_file"]))
    assert first_len == 1

    run_an_vqc_comparison("smoke", force=True, **kwargs)
    assert len(_read_jsonl(kwargs["selection_file"])) == 1  # overwritten, not duplicated


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        run_an_vqc_comparison("ludicrous")


def test_empty_seeds_raises(tmp_path):
    with pytest.raises(ValueError):
        run_an_vqc_comparison("smoke", seeds=[], selection_file=tmp_path / "x.jsonl")


# -- never touches the real E1-E8 / extension result files --------------------


def test_default_result_paths_are_an_vqc_specific():
    from qml_noise.an_vqc import AN_VQC_CLASSICAL_FILE, AN_VQC_RESULTS_FILE, AN_VQC_SELECTION_FILE
    from qml_noise.experiments import CLASSICAL_RESULTS_FILE, VQC_RESULTS_FILE
    from qml_noise.extensions import BC_RESULTS_FILE, NOISE_SWEEP_FILE

    an_vqc_paths = {AN_VQC_SELECTION_FILE, AN_VQC_RESULTS_FILE, AN_VQC_CLASSICAL_FILE}
    other_paths = {VQC_RESULTS_FILE, CLASSICAL_RESULTS_FILE, BC_RESULTS_FILE, NOISE_SWEEP_FILE}
    assert an_vqc_paths.isdisjoint(other_paths)
    assert all(p.name.startswith("an_vqc_") for p in an_vqc_paths)


# -- CLI ----------------------------------------------------------------------


def test_arg_parser_requires_mode():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_arg_parser_accepts_overrides():
    parser = build_arg_parser()
    args = parser.parse_args(["--mode", "full", "--seeds", "0,1,2", "--maxiter", "60", "--force"])
    assert args.mode == "full"
    assert args.seeds == "0,1,2"
    assert args.maxiter == 60
    assert args.force is True


def test_modes_match_assignment_seed_requirement():
    from qml_noise.an_vqc import AN_VQC_MODE_SEEDS

    assert AN_VQC_MODE_SEEDS["full"] == [0, 1, 2]
    assert len(AN_VQC_MODE_SEEDS["smoke"]) == 1
