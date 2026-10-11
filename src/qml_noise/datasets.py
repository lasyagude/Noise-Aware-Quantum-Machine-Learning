"""Dataset loading and preprocessing module.

Supports the three assignment-mandated datasets via scikit-learn's
built-in loaders:
    - Iris (binary: setosa vs. versicolor) -- primary experiment dataset.
    - Wine (optionally reduced to a feature subset) -- secondary.
    - Breast Cancer Wisconsin Diagnostic (optionally reduced) -- secondary.

All loaders:
    - use a stratified 80:20 train/test split with a fixed seed,
    - fit preprocessing (StandardScaler) on the training data only,
    - never use test labels for any fitting/tuning decision,
    - return identical splits across calls for the same seed, so that
      classical and quantum models are compared on the same data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qml_noise.config import DEFAULT_SEED, SUPPORTED_DATASETS, TEST_SIZE, get_logger

logger = get_logger(__name__)


@dataclass
class DatasetSplit:
    """Container for a preprocessed train/test split."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    dataset_name: str
    n_features: int
    n_classes: int
    seed: int


def _stratified_scaled_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split followed by train-only-fit standard scaling."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def load_iris_binary(
    *, seed: int = DEFAULT_SEED, test_size: float = TEST_SIZE
) -> DatasetSplit:
    """Load Iris restricted to the setosa/versicolor classes (binary, 4 features).

    This is the assignment's primary experiment dataset.
    """
    data = load_iris()
    mask = data.target != 2  # keep only setosa (0) and versicolor (1)
    X = data.data[mask]
    y = data.target[mask]

    X_train, X_test, y_train, y_test = _stratified_scaled_split(
        X, y, seed=seed, test_size=test_size
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(data.feature_names),
        class_names=[data.target_names[0], data.target_names[1]],
        dataset_name="iris",
        n_features=X.shape[1],
        n_classes=2,
        seed=seed,
    )


def load_wine_reduced(
    *,
    seed: int = DEFAULT_SEED,
    test_size: float = TEST_SIZE,
    n_features: int | None = None,
) -> DatasetSplit:
    """Load the Wine dataset, optionally reduced to the first ``n_features``.

    Secondary/optional experiment dataset (13 features, 3 classes).
    """
    data = load_wine()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)

    if n_features is not None:
        if not (0 < n_features <= X.shape[1]):
            raise ValueError(
                f"n_features must be in (0, {X.shape[1]}], got {n_features}"
            )
        X = X[:, :n_features]
        feature_names = feature_names[:n_features]

    X_train, X_test, y_train, y_test = _stratified_scaled_split(
        X, y, seed=seed, test_size=test_size
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        class_names=list(data.target_names),
        dataset_name="wine",
        n_features=X.shape[1],
        n_classes=len(set(y)),
        seed=seed,
    )


def load_breast_cancer_reduced(
    *,
    seed: int = DEFAULT_SEED,
    test_size: float = TEST_SIZE,
    n_features: int | None = None,
) -> DatasetSplit:
    """Load Breast Cancer Wisconsin Diagnostic, optionally reduced to ``n_features``.

    Secondary/optional experiment dataset (30 features, 2 classes).
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)

    if n_features is not None:
        if not (0 < n_features <= X.shape[1]):
            raise ValueError(
                f"n_features must be in (0, {X.shape[1]}], got {n_features}"
            )
        X = X[:, :n_features]
        feature_names = feature_names[:n_features]

    X_train, X_test, y_train, y_test = _stratified_scaled_split(
        X, y, seed=seed, test_size=test_size
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        class_names=list(data.target_names),
        dataset_name="breast_cancer",
        n_features=X.shape[1],
        n_classes=len(set(y)),
        seed=seed,
    )


def subsample_training_stratified(
    split: DatasetSplit, *, n_samples: int, seed: int
) -> DatasetSplit:
    """Return a copy of ``split`` with the training set stratified-subsampled.

    Used by the Breast Cancer extension to keep VQC training runtime
    bounded on 30-feature data: training sees only ``n_samples`` training
    points, but evaluation always uses the full, untouched ``X_test`` /
    ``y_test`` -- the held-out test set is never subsampled, and this
    function never looks at it. If ``n_samples >= len(split.y_train)``,
    ``split`` is returned unchanged (nothing to subsample).
    """
    if n_samples >= len(split.y_train):
        return split
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    X_sub, _, y_sub, _ = train_test_split(
        split.X_train,
        split.y_train,
        train_size=n_samples,
        stratify=split.y_train,
        random_state=seed,
    )
    return replace(split, X_train=X_sub, y_train=y_sub)


def carve_validation(split: DatasetSplit, *, val_fraction: float, seed: int) -> DatasetSplit:
    """Carve a stratified validation set out of ``split.X_train``/``y_train`` ONLY.

    Returns a new ``DatasetSplit`` whose ``X_train``/``y_train`` is the
    remaining ``(1 - val_fraction)`` of the original training data, and
    whose ``X_test``/``y_test`` is the carved-out validation portion. This
    lets existing training/evaluation helpers (``train_ideal_vqc``,
    ``evaluate_vqc_under_condition``) be reused unchanged for
    validation-based model selection -- they always evaluate against
    whatever is in ``X_test``/``y_test``, which here is the validation set,
    never the real held-out test set.

    This function reads only ``split.X_train`` and ``split.y_train``. The
    original ``split.X_test``/``y_test`` (the real test set) is never
    touched and does not appear anywhere in the object this returns.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    X_inner, X_val, y_inner, y_val = train_test_split(
        split.X_train,
        split.y_train,
        test_size=val_fraction,
        stratify=split.y_train,
        random_state=seed,
    )
    return replace(split, X_train=X_inner, y_train=y_inner, X_test=X_val, y_test=y_val)


def load_dataset(
    name: str,
    *,
    seed: int = DEFAULT_SEED,
    test_size: float = TEST_SIZE,
    n_features: int | None = None,
) -> DatasetSplit:
    """Dispatch to the right loader by dataset name.

    Parameters
    ----------
    name:
        One of ``qml_noise.config.SUPPORTED_DATASETS`` (``"iris"``,
        ``"wine"``, ``"breast_cancer"``).
    n_features:
        Only used for ``wine`` and ``breast_cancer`` to request a reduced
        feature subset (e.g. to match a target qubit count via angle
        encoding). Ignored for ``iris``, which is always 4 features.
    """
    name = name.lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported datasets: {SUPPORTED_DATASETS}"
        )

    logger.info("Loading dataset '%s' (seed=%d, test_size=%.2f)", name, seed, test_size)

    if name == "iris":
        if n_features is not None:
            logger.warning("n_features is ignored for the iris dataset")
        return load_iris_binary(seed=seed, test_size=test_size)
    if name == "wine":
        return load_wine_reduced(seed=seed, test_size=test_size, n_features=n_features)
    if name == "breast_cancer":
        return load_breast_cancer_reduced(
            seed=seed, test_size=test_size, n_features=n_features
        )

    raise AssertionError("unreachable")  # pragma: no cover
