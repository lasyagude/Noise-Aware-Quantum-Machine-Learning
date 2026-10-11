"""Tests for the trainable variational quantum classifier.

The slowest test here (`test_vqc_actually_trains_on_iris_binary`) is the one
that matters most: it asserts the circuit really learns rather than emitting
random predictions. Mechanical plumbing tests use fewer shots/iterations to
keep the suite fast; the mandatory 1024-shot setting is the module default
and is used by the training test.
"""

import numpy as np
import pytest

from qml_noise.config import SHOTS
from qml_noise.datasets import load_iris_binary
from qml_noise.metrics import classification_metrics, total_variation_distance
from qml_noise.noise import get_noise_model
from qml_noise.vqc import (
    FeatureAdapter,
    VariationalQuantumClassifier,
    bitstring_to_class,
    counts_to_class1_probability,
)


# -- readout strategies -----------------------------------------------------


def test_parity_readout_mapping():
    assert bitstring_to_class("00", "parity") == 0
    assert bitstring_to_class("01", "parity") == 1
    assert bitstring_to_class("11", "parity") == 0
    assert bitstring_to_class("111", "parity") == 1


def test_single_qubit_readout_reads_qubit_zero_little_endian():
    """Qiskit bitstrings are little-endian: qubit 0 is the rightmost char."""
    assert bitstring_to_class("10", "single") == 0
    assert bitstring_to_class("01", "single") == 1
    assert bitstring_to_class("1110", "single") == 0


def test_readout_strategy_ignores_spaces():
    assert bitstring_to_class("01 10", "parity") == 0


def test_unknown_readout_strategy_raises():
    with pytest.raises(ValueError):
        bitstring_to_class("01", "telepathy")


def test_counts_to_class1_probability():
    counts = {"00": 768, "01": 256}  # parity: 00 -> 0, 01 -> 1
    assert counts_to_class1_probability(counts, "parity") == pytest.approx(0.25)
    assert counts_to_class1_probability(counts, "single") == pytest.approx(0.25)


def test_counts_to_class1_probability_empty_raises():
    with pytest.raises(ValueError):
        counts_to_class1_probability({}, "parity")


# -- feature adapter --------------------------------------------------------


def test_adapter_identity_when_features_match_qubits():
    X = np.random.default_rng(0).normal(size=(10, 4))
    adapter = FeatureAdapter(4)
    out = adapter.fit_transform(X)
    assert adapter.mode_ == "identity"
    assert np.array_equal(out, X)


def test_adapter_reduces_with_pca():
    X = np.random.default_rng(0).normal(size=(20, 4))
    adapter = FeatureAdapter(2)
    out = adapter.fit_transform(X)
    assert adapter.mode_ == "pca"
    assert out.shape == (20, 2)


def test_adapter_expands_by_cyclic_tiling():
    X = np.arange(8, dtype=float).reshape(2, 4)
    adapter = FeatureAdapter(6)
    out = adapter.fit_transform(X)
    assert adapter.mode_ == "tile"
    assert out.shape == (2, 6)
    # indices should cycle 0,1,2,3,0,1
    assert np.array_equal(out[0], np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0]))


def test_adapter_pca_is_fit_on_training_data_only():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(30, 4))
    X_test = rng.normal(size=(10, 4))
    adapter = FeatureAdapter(2).fit(X_train)
    # Transforming test data must work without re-fitting and keep shape.
    assert adapter.transform(X_test).shape == (10, 2)


def test_adapter_requires_fit_before_transform():
    with pytest.raises(RuntimeError):
        FeatureAdapter(2).transform(np.zeros((2, 4)))


def test_adapter_rejects_wrong_feature_count():
    X = np.random.default_rng(0).normal(size=(5, 4))
    adapter = FeatureAdapter(2).fit(X)
    with pytest.raises(ValueError):
        adapter.transform(X[:, :3])


# -- construction -----------------------------------------------------------


def test_vqc_reports_parameter_count_and_resources():
    vqc = VariationalQuantumClassifier(4, 4, shots=64)
    assert vqc.n_parameters == 32
    res = vqc.resources()
    assert res.n_qubits == 4
    assert res.logical_depth == 4
    assert res.n_trainable_parameters == 32
    assert res.transpiled_depth > 0


def test_default_shots_is_the_mandatory_1024():
    assert VariationalQuantumClassifier(2, 2).shots == SHOTS == 1024


def test_invalid_vqc_arguments_raise():
    with pytest.raises(ValueError):
        VariationalQuantumClassifier(2, 2, readout_strategy="vibes")
    with pytest.raises(ValueError):
        VariationalQuantumClassifier(2, 2, shots=0)


def test_predict_before_fit_raises():
    vqc = VariationalQuantumClassifier(2, 2, shots=64)
    with pytest.raises(RuntimeError):
        vqc.predict(np.zeros((2, 4)))


def test_initialization_is_reproducible_for_a_given_seed():
    split = load_iris_binary(seed=0)
    X, y = split.X_train[:12], split.y_train[:12]
    a = VariationalQuantumClassifier(2, 1, shots=64, maxiter=8, seed=7).fit(X, y)
    b = VariationalQuantumClassifier(2, 1, shots=64, maxiter=8, seed=7).fit(X, y)
    assert np.array_equal(a.initial_weights_, b.initial_weights_)


def test_different_seeds_give_different_initialization():
    split = load_iris_binary(seed=0)
    X, y = split.X_train[:12], split.y_train[:12]
    a = VariationalQuantumClassifier(2, 1, shots=64, maxiter=8, seed=1).fit(X, y)
    b = VariationalQuantumClassifier(2, 1, shots=64, maxiter=8, seed=2).fit(X, y)
    assert not np.array_equal(a.initial_weights_, b.initial_weights_)


def test_non_binary_labels_are_rejected():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(9, 4))
    y = np.array([0, 1, 2] * 3)
    with pytest.raises(ValueError):
        VariationalQuantumClassifier(2, 1, shots=64, maxiter=2).fit(X, y)


# -- training ---------------------------------------------------------------


def test_vqc_actually_trains_on_iris_binary():
    """The headline requirement: real training, not random predictions."""
    split = load_iris_binary(seed=0)
    vqc = VariationalQuantumClassifier(n_qubits=2, depth=2, maxiter=40, seed=0)
    vqc.fit(split.X_train, split.y_train)

    record = vqc.training_record_
    assert record is not None
    assert record.n_iterations > 1
    # Loss must genuinely come down from its initial value.
    assert record.final_loss < record.initial_loss
    assert record.training_time_seconds > 0.0

    accuracy = classification_metrics(split.y_test, vqc.predict(split.X_test)).accuracy
    # Well above the 0.5 chance level for this balanced binary problem.
    assert accuracy >= 0.8

    assert vqc.weights_ is not None
    assert not np.array_equal(vqc.weights_, vqc.initial_weights_)


def test_predictions_use_original_class_labels():
    split = load_iris_binary(seed=0)
    vqc = VariationalQuantumClassifier(2, 1, shots=128, maxiter=8, seed=0)
    vqc.fit(split.X_train[:20], split.y_train[:20])
    predictions = vqc.predict(split.X_test[:6])
    assert set(np.unique(predictions)) <= set(np.unique(split.y_train))


def test_predict_proba_returns_valid_probabilities():
    split = load_iris_binary(seed=0)
    vqc = VariationalQuantumClassifier(2, 1, shots=128, maxiter=8, seed=0)
    vqc.fit(split.X_train[:20], split.y_train[:20])
    p = vqc.predict_proba(split.X_test[:6])
    assert p.shape == (6,)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_measurement_counts_respect_shot_budget():
    split = load_iris_binary(seed=0)
    vqc = VariationalQuantumClassifier(2, 1, shots=256, maxiter=8, seed=0)
    vqc.fit(split.X_train[:12], split.y_train[:12])
    counts = vqc.measurement_counts(split.X_test[:3])
    assert len(counts) == 3
    for c in counts:
        assert sum(c.values()) == 256


# -- noise interaction ------------------------------------------------------


def test_set_noise_model_keeps_trained_weights():
    split = load_iris_binary(seed=0)
    vqc = VariationalQuantumClassifier(2, 1, shots=128, maxiter=8, seed=0)
    vqc.fit(split.X_train[:16], split.y_train[:16])
    weights = vqc.weights_.copy()

    vqc.set_noise_model(get_noise_model("medium"))
    assert np.array_equal(vqc.weights_, weights)
    assert vqc.noise_model is not None
    # Still able to predict on the same trained weights under noise.
    assert vqc.predict(split.X_test[:4]).shape == (4,)


def test_noise_changes_the_output_distribution_of_a_trained_model():
    split = load_iris_binary(seed=0)
    vqc = VariationalQuantumClassifier(2, 2, shots=1024, maxiter=12, seed=0)
    vqc.fit(split.X_train[:16], split.y_train[:16])

    ideal_counts = vqc.measurement_counts(split.X_test[:1], seed_simulator=3)[0]
    vqc.set_noise_model(get_noise_model("medium"))
    noisy_counts = vqc.measurement_counts(split.X_test[:1], seed_simulator=3)[0]

    assert total_variation_distance(ideal_counts, noisy_counts) > 0.0
