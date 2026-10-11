"""Classical baseline models: Logistic Regression and SVM.

These establish the classical performance reference that the quantum
classifiers (ideal and noisy) are compared against. Both models are
trained on the exact same train/test split produced by
``qml_noise.datasets.load_dataset`` so that comparisons are apples-to-apples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC

from qml_noise.config import CLASSICAL_CONFIG, get_logger
from qml_noise.datasets import DatasetSplit

logger = get_logger(__name__)


@dataclass
class ClassicalResult:
    """Evaluation result for a single classical baseline model."""

    model_name: str
    dataset_name: str
    seed: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    training_time_seconds: float

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "seed": self.seed,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix,
            "training_time_seconds": self.training_time_seconds,
        }


def _evaluate(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
    seed: int,
    training_time_seconds: float,
) -> ClassicalResult:
    accuracy = float(np.mean(y_true == y_pred))
    average = "binary" if len(set(y_true)) == 2 else "macro"
    precision = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    recall = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return ClassicalResult(
        model_name=model_name,
        dataset_name=dataset_name,
        seed=seed,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm,
        training_time_seconds=training_time_seconds,
    )


def run_logistic_regression(split: DatasetSplit) -> ClassicalResult:
    """Train and evaluate a Logistic Regression baseline."""
    model = LogisticRegression(
        max_iter=CLASSICAL_CONFIG.logistic_regression_max_iter,
        random_state=split.seed,
    )

    start = time.perf_counter()
    model.fit(split.X_train, split.y_train)
    training_time = time.perf_counter() - start

    y_pred = model.predict(split.X_test)
    result = _evaluate(
        "logistic_regression", split.y_test, y_pred, split.dataset_name, split.seed, training_time
    )
    logger.info(
        "LogisticRegression on %s (seed=%d): accuracy=%.4f, train_time=%.4fs",
        split.dataset_name,
        split.seed,
        result.accuracy,
        training_time,
    )
    return result


def run_svm(split: DatasetSplit) -> ClassicalResult:
    """Train and evaluate an SVM baseline."""
    model = SVC(
        kernel=CLASSICAL_CONFIG.svm_kernel,
        C=CLASSICAL_CONFIG.svm_C,
        random_state=split.seed,
    )

    start = time.perf_counter()
    model.fit(split.X_train, split.y_train)
    training_time = time.perf_counter() - start

    y_pred = model.predict(split.X_test)
    result = _evaluate(
        "svm", split.y_test, y_pred, split.dataset_name, split.seed, training_time
    )
    logger.info(
        "SVM on %s (seed=%d): accuracy=%.4f, train_time=%.4fs",
        split.dataset_name,
        split.seed,
        result.accuracy,
        training_time,
    )
    return result


def run_all_classical_baselines(split: DatasetSplit) -> list[ClassicalResult]:
    """Run every classical baseline model on the given dataset split."""
    return [run_logistic_regression(split), run_svm(split)]
