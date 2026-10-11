"""Basic tests for the dataset module."""

import numpy as np
import pytest

from qml_noise.datasets import (
    DatasetSplit,
    load_breast_cancer_reduced,
    load_dataset,
    load_iris_binary,
    load_wine_reduced,
)


def test_load_iris_binary_shapes_and_classes():
    split = load_iris_binary(seed=0)
    assert isinstance(split, DatasetSplit)
    assert split.n_features == 4
    assert split.n_classes == 2
    assert set(np.unique(split.y_train)) <= {0, 1}
    assert set(np.unique(split.y_test)) <= {0, 1}
    # 100 samples (setosa + versicolor) * 0.8 train / 0.2 test
    assert split.X_train.shape[0] == 80
    assert split.X_test.shape[0] == 20
    assert split.X_train.shape[1] == 4


def test_iris_split_is_stratified():
    split = load_iris_binary(seed=0)
    train_counts = np.bincount(split.y_train)
    test_counts = np.bincount(split.y_test)
    # Balanced classes (50/50) should remain balanced after stratified split.
    assert train_counts[0] == train_counts[1] == 40
    assert test_counts[0] == test_counts[1] == 10


def test_iris_split_reproducible_with_same_seed():
    split_a = load_iris_binary(seed=42)
    split_b = load_iris_binary(seed=42)
    assert np.array_equal(split_a.X_train, split_b.X_train)
    assert np.array_equal(split_a.y_train, split_b.y_train)
    assert np.array_equal(split_a.X_test, split_b.X_test)


def test_iris_split_differs_with_different_seed():
    split_a = load_iris_binary(seed=0)
    split_b = load_iris_binary(seed=1)
    assert not np.array_equal(split_a.X_train, split_b.X_train)


def test_scaler_fit_on_train_only():
    split = load_iris_binary(seed=0)
    # Standardized training data should have ~0 mean / ~1 std per feature.
    assert np.allclose(split.X_train.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(split.X_train.std(axis=0), 1.0, atol=1e-10)
    # Test data uses the train-fitted scaler, so it generally will NOT be
    # perfectly standardized -- this is the expected, correct behavior.


def test_load_wine_reduced_features():
    split = load_wine_reduced(seed=0, n_features=4)
    assert split.n_features == 4
    assert split.X_train.shape[1] == 4
    assert len(split.feature_names) == 4
    assert split.n_classes == 3


def test_load_wine_full_features():
    split = load_wine_reduced(seed=0)
    assert split.n_features == 13


def test_load_wine_invalid_n_features_raises():
    with pytest.raises(ValueError):
        load_wine_reduced(seed=0, n_features=0)
    with pytest.raises(ValueError):
        load_wine_reduced(seed=0, n_features=100)


def test_load_breast_cancer_reduced_features():
    split = load_breast_cancer_reduced(seed=0, n_features=6)
    assert split.n_features == 6
    assert split.n_classes == 2


def test_load_dataset_dispatch():
    iris = load_dataset("iris", seed=0)
    wine = load_dataset("wine", seed=0, n_features=4)
    bc = load_dataset("breast_cancer", seed=0, n_features=6)
    assert iris.dataset_name == "iris"
    assert wine.dataset_name == "wine"
    assert bc.dataset_name == "breast_cancer"


def test_load_dataset_unknown_name_raises():
    with pytest.raises(ValueError):
        load_dataset("not_a_real_dataset")


def test_load_dataset_case_insensitive():
    split = load_dataset("IRIS", seed=0)
    assert split.dataset_name == "iris"
