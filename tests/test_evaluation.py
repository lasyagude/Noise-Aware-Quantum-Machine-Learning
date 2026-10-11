"""Tests for the training/evaluation module."""

import pytest

from qml_noise.datasets import load_iris_binary
from qml_noise.evaluation import (
    TrainedVQCArtifact,
    evaluate_vqc_under_condition,
    evaluate_vqc_with_noise_model,
    train_ideal_vqc,
)
from qml_noise.noise import get_noise_model


@pytest.fixture(scope="module")
def iris_split():
    return load_iris_binary(seed=0)


@pytest.fixture(scope="module")
def ideal_artifact(iris_split):
    return train_ideal_vqc(2, 2, seed=0, split=iris_split, maxiter=15)


def test_train_ideal_vqc_returns_populated_artifact(ideal_artifact):
    assert isinstance(ideal_artifact, TrainedVQCArtifact)
    assert ideal_artifact.resources.n_qubits == 2
    assert ideal_artifact.resources.logical_depth == 2
    assert ideal_artifact.resources.n_trainable_parameters == 8
    assert 0.0 <= ideal_artifact.ideal_test_metrics.accuracy <= 1.0
    assert ideal_artifact.training_time_seconds > 0.0
    assert ideal_artifact.n_training_iterations > 0
    assert sum(ideal_artifact.ideal_pooled_counts.values()) == 20 * 1024  # 20 test samples


def test_evaluate_under_ideal_condition_reproduces_ideal_metrics(ideal_artifact):
    """Re-evaluating under 'ideal' with the same seed must reproduce the baseline exactly."""
    record = evaluate_vqc_under_condition(ideal_artifact, "E_TEST", "ideal")
    assert record.accuracy == pytest.approx(ideal_artifact.ideal_test_metrics.accuracy)
    assert record.degradation_pp == pytest.approx(0.0)
    assert record.relative_degradation_pct == pytest.approx(0.0)
    assert record.robustness_ratio == pytest.approx(1.0)
    assert record.distribution_total_variation_distance == pytest.approx(0.0)
    assert record.distribution_hellinger_distance == pytest.approx(0.0)
    assert record.n_outcomes_only_in_noisy == 0


def test_evaluate_under_medium_noise_changes_distribution(ideal_artifact):
    record = evaluate_vqc_under_condition(
        ideal_artifact, "E_TEST", "medium", reused_training_from="E_IDEAL"
    )
    assert record.distribution_total_variation_distance > 0.0
    assert record.reused_training_from == "E_IDEAL"
    # Noise evaluation reuses training -> zero additional training cost attributed.
    assert record.training_time_seconds == 0.0
    assert record.eval_time_seconds > 0.0
    assert record.runtime_seconds == pytest.approx(record.eval_time_seconds)


def test_evaluate_without_reuse_attributes_full_training_time(ideal_artifact):
    record = evaluate_vqc_under_condition(ideal_artifact, "E_TEST", "ideal")
    assert record.training_time_seconds == pytest.approx(ideal_artifact.training_time_seconds)
    assert record.runtime_seconds == pytest.approx(
        record.training_time_seconds + record.eval_time_seconds
    )


def test_resource_metrics_match_section_10_requirements(ideal_artifact):
    record = evaluate_vqc_under_condition(ideal_artifact, "E_TEST", "ideal")
    assert record.n_qubits == 2
    assert record.logical_depth == 2
    assert record.transpiled_depth > 0
    assert record.total_gate_count > 0
    assert record.two_qubit_gate_count > 0
    assert record.n_trainable_parameters == 8
    assert record.shots == 1024


def test_pooled_output_distribution_sums_to_approximately_one(ideal_artifact):
    record = evaluate_vqc_under_condition(ideal_artifact, "E_TEST", "medium")
    assert sum(record.pooled_output_distribution.values()) == pytest.approx(1.0)


def test_confusion_matrix_is_present_and_well_formed(ideal_artifact):
    record = evaluate_vqc_under_condition(ideal_artifact, "E_TEST", "ideal")
    assert len(record.confusion_matrix) == 2
    assert all(len(row) == 2 for row in record.confusion_matrix)
    assert sum(sum(row) for row in record.confusion_matrix) == 20  # test set size


def test_to_dict_is_json_serializable_shape(ideal_artifact):
    record = evaluate_vqc_under_condition(ideal_artifact, "E_TEST", "low")
    d = record.to_dict()
    import json

    json.dumps(d)  # must not raise
    assert d["experiment_id"] == "E_TEST"
    assert d["noise_condition"] == "low"
    assert d["status"] == "completed"


def test_different_architecture_trains_independently(iris_split):
    artifact = train_ideal_vqc(4, 2, seed=0, split=iris_split, maxiter=10)
    assert artifact.resources.n_qubits == 4
    assert artifact.resources.n_trainable_parameters == 16


# -- evaluate_vqc_with_noise_model (the noise-model-agnostic core) ----------


def test_evaluate_with_noise_model_matches_condition_wrapper_for_ideal(ideal_artifact):
    """evaluate_vqc_under_condition is a thin wrapper; results must match exactly."""
    via_condition = evaluate_vqc_under_condition(ideal_artifact, "E_TEST", "ideal")
    via_model = evaluate_vqc_with_noise_model(
        ideal_artifact, "E_TEST", None, noise_label="ideal"
    )
    assert via_model.accuracy == pytest.approx(via_condition.accuracy)
    assert via_model.distribution_total_variation_distance == pytest.approx(
        via_condition.distribution_total_variation_distance
    )


def test_evaluate_with_noise_model_matches_condition_wrapper_for_medium(ideal_artifact):
    via_condition = evaluate_vqc_under_condition(ideal_artifact, "E_TEST", "medium")
    via_model = evaluate_vqc_with_noise_model(
        ideal_artifact, "E_TEST", get_noise_model("medium"), noise_label="medium"
    )
    assert via_model.accuracy == pytest.approx(via_condition.accuracy)
    assert via_model.distribution_total_variation_distance == pytest.approx(
        via_condition.distribution_total_variation_distance
    )


def test_evaluate_with_noise_model_accepts_custom_noise_model(ideal_artifact):
    """Extensions pass a hand-built noise model with a descriptive label."""
    from qml_noise.noise import build_depolarizing_noise_model

    custom = build_depolarizing_noise_model(0.002, two_qubit_probability=0.02)
    record = evaluate_vqc_with_noise_model(
        ideal_artifact,
        "NSWEEP-TEST",
        custom,
        noise_label="depolarizing(p1=0.002,p2=0.02)",
        reused_training_from="E_IDEAL",
    )
    assert record.noise_condition == "depolarizing(p1=0.002,p2=0.02)"
    assert record.reused_training_from == "E_IDEAL"
    assert record.distribution_total_variation_distance >= 0.0
