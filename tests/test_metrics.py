"""Tests for the metrics module."""

import math

import numpy as np
import pytest

from qml_noise.metrics import (
    accuracy_degradation,
    aggregate_across_seeds,
    classification_metrics,
    compare_distributions,
    counts_to_distribution,
    hellinger_distance,
    noise_robustness_ratio,
    noise_sensitivity,
    relative_degradation,
    total_variation_distance,
)


# -- classification metrics -------------------------------------------------


def test_perfect_classification():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, y)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.confusion_matrix == [[2, 0], [0, 2]]


def test_partial_classification():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    m = classification_metrics(y_true, y_pred)
    assert m.accuracy == 0.75
    assert m.recall == 1.0
    assert m.precision == pytest.approx(2 / 3)


def test_multiclass_uses_macro_average():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 2, 1])
    m = classification_metrics(y_true, y_pred)
    assert m.accuracy == 0.75
    assert len(m.confusion_matrix) == 3


def test_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        classification_metrics(np.array([0, 1]), np.array([0, 1, 1]))


def test_metrics_on_empty_input_raises():
    with pytest.raises(ValueError):
        classification_metrics(np.array([]), np.array([]))


def test_metrics_to_dict():
    y = np.array([0, 1])
    d = classification_metrics(y, y).to_dict()
    assert set(d) == {"accuracy", "precision", "recall", "f1", "confusion_matrix"}


# -- degradation conventions ------------------------------------------------


def test_accuracy_degradation_in_percentage_points():
    assert accuracy_degradation(0.95, 0.80) == pytest.approx(15.0)


def test_relative_degradation_formula():
    assert relative_degradation(0.80, 0.60) == pytest.approx(25.0)


def test_robustness_ratio_formula():
    assert noise_robustness_ratio(0.80, 0.60) == pytest.approx(0.75)


def test_no_degradation_gives_zero_and_unit_ratio():
    assert accuracy_degradation(0.9, 0.9) == pytest.approx(0.0)
    assert relative_degradation(0.9, 0.9) == pytest.approx(0.0)
    assert noise_robustness_ratio(0.9, 0.9) == pytest.approx(1.0)


def test_noise_may_improve_accuracy_giving_negative_degradation():
    """Noisy accuracy above ideal is a real outcome, not an error."""
    assert accuracy_degradation(0.80, 0.85) == pytest.approx(-5.0)
    assert relative_degradation(0.80, 0.85) < 0.0
    assert noise_robustness_ratio(0.80, 0.85) > 1.0


def test_zero_ideal_accuracy_yields_nan_not_an_exception():
    assert math.isnan(relative_degradation(0.0, 0.0))
    assert math.isnan(noise_robustness_ratio(0.0, 0.5))
    # Absolute degradation has no denominator, so it stays well defined.
    assert accuracy_degradation(0.0, 0.5) == pytest.approx(-50.0)


def test_noise_sensitivity_bundle():
    ns = noise_sensitivity(0.90, 0.72)
    assert ns.degradation_pp == pytest.approx(18.0)
    assert ns.relative_degradation_pct == pytest.approx(20.0)
    assert ns.robustness_ratio == pytest.approx(0.8)
    assert set(ns.to_dict()) == {
        "accuracy_ideal",
        "accuracy_noisy",
        "degradation_pp",
        "relative_degradation_pct",
        "robustness_ratio",
    }


# -- distribution comparison ------------------------------------------------


def test_counts_to_distribution_normalizes():
    dist = counts_to_distribution({"00": 768, "11": 256})
    assert dist["00"] == pytest.approx(0.75)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_counts_to_distribution_strips_spaces():
    assert "0011" in counts_to_distribution({"00 11": 10})


def test_empty_counts_raise():
    with pytest.raises(ValueError):
        counts_to_distribution({})


def test_identical_distributions_have_zero_distance():
    counts = {"00": 512, "11": 512}
    assert total_variation_distance(counts, counts) == pytest.approx(0.0)
    assert hellinger_distance(counts, counts) == pytest.approx(0.0)


def test_disjoint_distributions_have_maximal_distance():
    assert total_variation_distance({"00": 100}, {"11": 100}) == pytest.approx(1.0)
    assert hellinger_distance({"00": 100}, {"11": 100}) == pytest.approx(1.0)


def test_partial_overlap_distance_is_between_zero_and_one():
    tvd = total_variation_distance({"00": 512, "11": 512}, {"00": 256, "11": 768})
    assert 0.0 < tvd < 1.0
    assert tvd == pytest.approx(0.25)


def test_compare_distributions_counts_noise_induced_outcomes():
    ideal = {"00": 1024}
    noisy = {"00": 900, "01": 60, "10": 64}
    cmp = compare_distributions(ideal, noisy)
    assert cmp.n_outcomes_ideal == 1
    assert cmp.n_outcomes_noisy == 3
    assert cmp.n_outcomes_only_in_noisy == 2
    assert cmp.total_variation_distance > 0.0
    assert set(cmp.to_dict()) == {
        "total_variation_distance",
        "hellinger_distance",
        "n_outcomes_ideal",
        "n_outcomes_noisy",
        "n_outcomes_only_in_noisy",
    }


# -- aggregation ------------------------------------------------------------


def test_aggregate_across_seeds():
    agg = aggregate_across_seeds([0.8, 0.9, 1.0])
    assert agg["mean"] == pytest.approx(0.9)
    assert agg["min"] == pytest.approx(0.8)
    assert agg["max"] == pytest.approx(1.0)
    assert agg["n_runs"] == 3
    assert agg["std"] > 0.0


def test_aggregate_single_run_has_zero_std():
    agg = aggregate_across_seeds([0.85])
    assert agg["mean"] == pytest.approx(0.85)
    assert agg["std"] == pytest.approx(0.0)


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_across_seeds([])
