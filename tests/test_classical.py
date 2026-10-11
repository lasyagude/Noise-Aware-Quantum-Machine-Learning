"""Basic tests for the classical baseline module."""

from qml_noise.classical import (
    ClassicalResult,
    run_all_classical_baselines,
    run_logistic_regression,
    run_svm,
)
from qml_noise.datasets import load_iris_binary


def test_logistic_regression_on_iris_is_accurate():
    split = load_iris_binary(seed=0)
    result = run_logistic_regression(split)
    assert isinstance(result, ClassicalResult)
    # Iris setosa-vs-versicolor is linearly separable; expect near-perfect accuracy.
    assert result.accuracy >= 0.9
    assert result.training_time_seconds >= 0.0
    assert len(result.confusion_matrix) == 2
    assert len(result.confusion_matrix[0]) == 2


def test_svm_on_iris_is_accurate():
    split = load_iris_binary(seed=0)
    result = run_svm(split)
    assert result.accuracy >= 0.9


def test_run_all_classical_baselines_returns_both_models():
    split = load_iris_binary(seed=0)
    results = run_all_classical_baselines(split)
    names = {r.model_name for r in results}
    assert names == {"logistic_regression", "svm"}


def test_classical_result_to_dict_is_json_serializable_shape():
    split = load_iris_binary(seed=0)
    result = run_logistic_regression(split)
    d = result.to_dict()
    assert d["model_name"] == "logistic_regression"
    assert d["dataset_name"] == "iris"
    assert 0.0 <= d["accuracy"] <= 1.0
    assert 0.0 <= d["precision"] <= 1.0
    assert 0.0 <= d["recall"] <= 1.0
    assert 0.0 <= d["f1"] <= 1.0


def test_metrics_consistent_across_seeds():
    results = [run_logistic_regression(load_iris_binary(seed=s)) for s in range(5)]
    assert len(results) == 5
    for r in results:
        assert r.accuracy >= 0.8
