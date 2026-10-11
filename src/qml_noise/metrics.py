"""Metrics for noise-sensitivity analysis (assignment section 10).

Conventions
-----------

Accuracy degradation in **percentage points**::

    D = 100 * (A_ideal - A_noisy)

Relative degradation in **percent of the ideal accuracy**::

    D_relative = 100 * (A_ideal - A_noisy) / A_ideal

Noise robustness ratio::

    R = A_noisy / A_ideal

Both ``D_relative`` and ``R`` divide by ``A_ideal``. When ``A_ideal == 0``
the quantity is undefined, so these functions return ``float("nan")``
rather than raising or silently substituting zero -- ``nan`` propagates
visibly into tables instead of masquerading as a real result.

Sign conventions are deliberately **not** clamped: noise can occasionally
*increase* measured accuracy (shot noise, or a noisy decision boundary that
happens to suit the test set), which yields a negative ``D`` and ``R > 1``.
That is a real observation and is reported as-is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class ClassificationMetrics:
    """Standard classification metrics for one evaluation."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


def classification_metrics(y_true, y_pred) -> ClassificationMetrics:
    """Accuracy, precision, recall, F1, and confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; "
            f"got {y_true.shape} and {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")

    average = "binary" if len(np.unique(y_true)) <= 2 else "macro"
    return ClassificationMetrics(
        accuracy=float(np.mean(y_true == y_pred)),
        precision=float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


# ---------------------------------------------------------------------------
# Noise sensitivity
# ---------------------------------------------------------------------------


def accuracy_degradation(accuracy_ideal: float, accuracy_noisy: float) -> float:
    """``D = 100 * (A_ideal - A_noisy)``, in percentage points.

    Negative values are legitimate and mean the noisy model scored higher.
    """
    return 100.0 * (accuracy_ideal - accuracy_noisy)


def relative_degradation(accuracy_ideal: float, accuracy_noisy: float) -> float:
    """``D_rel = 100 * (A_ideal - A_noisy) / A_ideal``; ``nan`` if ``A_ideal == 0``."""
    if accuracy_ideal == 0.0:
        return float("nan")
    return 100.0 * (accuracy_ideal - accuracy_noisy) / accuracy_ideal


def noise_robustness_ratio(accuracy_ideal: float, accuracy_noisy: float) -> float:
    """``R = A_noisy / A_ideal``; ``nan`` if ``A_ideal == 0``."""
    if accuracy_ideal == 0.0:
        return float("nan")
    return accuracy_noisy / accuracy_ideal


@dataclass
class NoiseSensitivity:
    """The three degradation/robustness quantities for one comparison."""

    accuracy_ideal: float
    accuracy_noisy: float
    degradation_pp: float
    relative_degradation_pct: float
    robustness_ratio: float

    def to_dict(self) -> dict:
        return {
            "accuracy_ideal": self.accuracy_ideal,
            "accuracy_noisy": self.accuracy_noisy,
            "degradation_pp": self.degradation_pp,
            "relative_degradation_pct": self.relative_degradation_pct,
            "robustness_ratio": self.robustness_ratio,
        }


def noise_sensitivity(accuracy_ideal: float, accuracy_noisy: float) -> NoiseSensitivity:
    """Bundle all three noise-sensitivity metrics for one ideal/noisy pair."""
    return NoiseSensitivity(
        accuracy_ideal=accuracy_ideal,
        accuracy_noisy=accuracy_noisy,
        degradation_pp=accuracy_degradation(accuracy_ideal, accuracy_noisy),
        relative_degradation_pct=relative_degradation(accuracy_ideal, accuracy_noisy),
        robustness_ratio=noise_robustness_ratio(accuracy_ideal, accuracy_noisy),
    )


# ---------------------------------------------------------------------------
# Measurement output distribution comparison
# ---------------------------------------------------------------------------


def counts_to_distribution(counts: dict[str, int]) -> dict[str, float]:
    """Normalize shot counts into a probability distribution."""
    total = sum(counts.values())
    if total == 0:
        raise ValueError("counts dictionary is empty; cannot normalize")
    return {bits.replace(" ", ""): n / total for bits, n in counts.items()}


def total_variation_distance(counts_a: dict[str, int], counts_b: dict[str, int]) -> float:
    """TVD = 0.5 * sum |p_i - q_i|. 0 = identical, 1 = disjoint support."""
    p = counts_to_distribution(counts_a)
    q = counts_to_distribution(counts_b)
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def hellinger_distance(counts_a: dict[str, int], counts_b: dict[str, int]) -> float:
    """Hellinger distance between two count distributions (0 = identical)."""
    p = counts_to_distribution(counts_a)
    q = counts_to_distribution(counts_b)
    keys = set(p) | set(q)
    squared = sum(
        (math.sqrt(p.get(k, 0.0)) - math.sqrt(q.get(k, 0.0))) ** 2 for k in keys
    )
    return math.sqrt(squared / 2.0)


@dataclass
class DistributionComparison:
    """Comparison of ideal vs. noisy measurement output distributions."""

    total_variation_distance: float
    hellinger_distance: float
    n_outcomes_ideal: int
    n_outcomes_noisy: int
    n_outcomes_only_in_noisy: int

    def to_dict(self) -> dict:
        return {
            "total_variation_distance": self.total_variation_distance,
            "hellinger_distance": self.hellinger_distance,
            "n_outcomes_ideal": self.n_outcomes_ideal,
            "n_outcomes_noisy": self.n_outcomes_noisy,
            "n_outcomes_only_in_noisy": self.n_outcomes_only_in_noisy,
        }


def compare_distributions(
    counts_ideal: dict[str, int], counts_noisy: dict[str, int]
) -> DistributionComparison:
    """Compare two measurement distributions from the same circuit.

    ``n_outcomes_only_in_noisy`` counts bitstrings that noise introduced --
    outcomes with zero ideal probability, a direct signature of noise
    spreading probability mass across the computational basis.
    """
    ideal_keys = {k.replace(" ", "") for k in counts_ideal}
    noisy_keys = {k.replace(" ", "") for k in counts_noisy}
    return DistributionComparison(
        total_variation_distance=total_variation_distance(counts_ideal, counts_noisy),
        hellinger_distance=hellinger_distance(counts_ideal, counts_noisy),
        n_outcomes_ideal=len(ideal_keys),
        n_outcomes_noisy=len(noisy_keys),
        n_outcomes_only_in_noisy=len(noisy_keys - ideal_keys),
    )


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------


def aggregate_across_seeds(values: list[float]) -> dict[str, float]:
    """Mean/std/min/max over repeated runs (section 13's error bars).

    ``std`` is the population standard deviation (``ddof=0``) and is ``0.0``
    for a single run.
    """
    if not values:
        raise ValueError("cannot aggregate an empty list of values")
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_runs": len(values),
    }
