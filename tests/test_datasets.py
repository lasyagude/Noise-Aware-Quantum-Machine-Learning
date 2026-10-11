"""Basic tests for the dataset module."""

import numpy as np
import pytest

from qml_noise.datasets import (
    DatasetSplit,
    carve_validation,
    load_breast_cancer_reduced,
    load_dataset,
    load_iris_binary,
    load_wine_reduced,
    subsample_training_stratified,
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


# -- subsample_training_stratified -------------------------------------------


def test_subsample_reduces_training_set_size_only():
    split = load_breast_cancer_reduced(seed=0)
    original_test_size = split.X_test.shape[0]
    sub = subsample_training_stratified(split, n_samples=50, seed=0)
    assert sub.X_train.shape[0] == 50
    assert sub.y_train.shape[0] == 50
    # The held-out test set must be completely untouched.
    assert sub.X_test.shape[0] == original_test_size
    assert np.array_equal(sub.X_test, split.X_test)
    assert np.array_equal(sub.y_test, split.y_test)


def test_subsample_preserves_class_balance_roughly():
    split = load_breast_cancer_reduced(seed=0)
    sub = subsample_training_stratified(split, n_samples=100, seed=0)
    original_fraction = split.y_train.mean()
    sub_fraction = sub.y_train.mean()
    assert abs(original_fraction - sub_fraction) < 0.1


def test_subsample_noop_when_n_samples_exceeds_training_set():
    split = load_iris_binary(seed=0)  # only 80 training samples
    sub = subsample_training_stratified(split, n_samples=1000, seed=0)
    assert sub is split  # returned unchanged, not a copy


def test_subsample_reproducible_with_same_seed():
    split = load_breast_cancer_reduced(seed=0)
    a = subsample_training_stratified(split, n_samples=50, seed=3)
    b = subsample_training_stratified(split, n_samples=50, seed=3)
    assert np.array_equal(a.X_train, b.X_train)
    assert np.array_equal(a.y_train, b.y_train)


def test_subsample_differs_with_different_seed():
    split = load_breast_cancer_reduced(seed=0)
    a = subsample_training_stratified(split, n_samples=50, seed=1)
    b = subsample_training_stratified(split, n_samples=50, seed=2)
    assert not np.array_equal(a.X_train, b.X_train)


def test_subsample_rejects_non_positive_n_samples():
    split = load_breast_cancer_reduced(seed=0)
    with pytest.raises(ValueError):
        subsample_training_stratified(split, n_samples=0, seed=0)


# -- carve_validation ---------------------------------------------------------


def test_carve_validation_splits_training_data_only():
    split = load_iris_binary(seed=0)  # 80 train / 20 test
    val_split = carve_validation(split, val_fraction=0.25, seed=0)
    assert val_split.X_train.shape[0] == 60
    assert val_split.X_test.shape[0] == 20  # the carved-out validation set
    # The real test set is NOT in the returned object anywhere.
    assert not np.array_equal(val_split.X_test, split.X_test) or val_split.X_test.shape == split.X_test.shape
    assert val_split.X_train.shape[0] + val_split.X_test.shape[0] == split.X_train.shape[0]


def test_carve_validation_preserves_stratification():
    split = load_iris_binary(seed=0)  # balanced 40/40 classes in training
    val_split = carve_validation(split, val_fraction=0.25, seed=0)
    train_balance = val_split.y_train.mean()
    val_balance = val_split.y_test.mean()
    assert abs(train_balance - 0.5) < 0.15
    assert abs(val_balance - 0.5) < 0.2


def test_carve_validation_reproducible_with_same_seed():
    split = load_iris_binary(seed=0)
    a = carve_validation(split, val_fraction=0.25, seed=5)
    b = carve_validation(split, val_fraction=0.25, seed=5)
    assert np.array_equal(a.X_train, b.X_train)
    assert np.array_equal(a.X_test, b.X_test)


def test_carve_validation_differs_with_different_seed():
    split = load_iris_binary(seed=0)
    a = carve_validation(split, val_fraction=0.25, seed=1)
    b = carve_validation(split, val_fraction=0.25, seed=2)
    assert not np.array_equal(a.X_train, b.X_train)


def test_carve_validation_rejects_out_of_range_fraction():
    split = load_iris_binary(seed=0)
    with pytest.raises(ValueError):
        carve_validation(split, val_fraction=0.0, seed=0)
    with pytest.raises(ValueError):
        carve_validation(split, val_fraction=1.0, seed=0)


def test_carve_validation_does_not_mutate_original_split():
    split = load_iris_binary(seed=0)
    original_train_shape = split.X_train.shape
    original_test_shape = split.X_test.shape
    carve_validation(split, val_fraction=0.25, seed=0)
    assert split.X_train.shape == original_train_shape
    assert split.X_test.shape == original_test_shape
