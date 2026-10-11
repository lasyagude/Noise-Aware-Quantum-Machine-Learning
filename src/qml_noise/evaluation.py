"""Training/evaluation module (assignment section 12).

Design: train once per (qubits, depth), evaluate under every noise condition
-------------------------------------------------------------------------

The required matrix (section 7) only ever varies noise condition *within*
a fixed (qubits, depth) architecture -- e.g. E4/E5/E6 are all "4 qubits,
depth 4", just under ideal/low/medium noise. That structure is exactly the
right shape for the assignment's own instructions: "establish a noiseless
baseline" (section 2, item 2), then "introduce noise" and "compare" against
that baseline (items 3-4).

So for each (qubits, depth) architecture, :func:`train_ideal_vqc` trains
**once**, under ideal (noiseless) conditions -- this is the mandatory
baseline. :func:`evaluate_vqc_under_condition` then re-evaluates that exact
trained circuit and weights under a different noise model via
``VariationalQuantumClassifier.set_noise_model()``. This measures noise
*sensitivity* of a fixed, already-trained model, rather than conflating it
with training a separate model under noise. Retraining is not performed per
noise condition.

Output distributions (section 10) are pooled across the whole test set
(summed shot counts from every test-sample circuit) rather than kept
per-sample, so that the ideal-vs-noisy comparison is a single, readable
distribution rather than 20+ small per-sample ones. Per-sample accuracy is
unaffected -- it still comes from each sample's own prediction.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

from qml_noise.circuits import CircuitResources
from qml_noise.config import SHOTS, VQC_CONFIG, get_logger
from qml_noise.datasets import DatasetSplit
from qml_noise.metrics import (
    ClassificationMetrics,
    classification_metrics,
    compare_distributions,
    noise_sensitivity,
)
from qml_noise.noise import get_noise_model
from qml_noise.vqc import VariationalQuantumClassifier

logger = get_logger(__name__)


def _pool_counts(counts_list: list[dict[str, int]]) -> dict[str, int]:
    """Sum per-sample shot counts into one pooled distribution."""
    pooled: Counter = Counter()
    for counts in counts_list:
        pooled.update(counts)
    return dict(pooled)


@dataclass
class TrainedVQCArtifact:
    """A VQC trained once under ideal noise, reused across noise conditions."""

    model: VariationalQuantumClassifier
    split: DatasetSplit
    seed: int
    resources: CircuitResources
    ideal_test_metrics: ClassificationMetrics
    ideal_pooled_counts: dict[str, int]
    training_time_seconds: float
    n_training_iterations: int
    initial_loss: float
    final_loss: float


def train_ideal_vqc(
    n_qubits: int,
    depth: int,
    seed: int,
    split: DatasetSplit,
    *,
    maxiter: int = VQC_CONFIG.maxiter,
    shots: int = SHOTS,
) -> TrainedVQCArtifact:
    """Train a VQC under ideal (noiseless) conditions -- the mandatory baseline."""
    model = VariationalQuantumClassifier(
        n_qubits=n_qubits,
        depth=depth,
        noise_model=None,
        shots=shots,
        maxiter=maxiter,
        seed=seed,
    )
    model.fit(split.X_train, split.y_train)
    record = model.training_record_
    assert record is not None

    predictions = model.predict(split.X_test, seed_simulator=seed)
    ideal_metrics = classification_metrics(split.y_test, predictions)
    ideal_counts = model.measurement_counts(split.X_test, seed_simulator=seed)
    ideal_pooled = _pool_counts(ideal_counts)

    logger.info(
        "Trained ideal baseline: qubits=%d depth=%d seed=%d test_accuracy=%.4f "
        "loss %.4f -> %.4f (%d evaluations, %.2fs)",
        n_qubits,
        depth,
        seed,
        ideal_metrics.accuracy,
        record.initial_loss,
        record.final_loss,
        record.n_iterations,
        record.training_time_seconds,
    )

    return TrainedVQCArtifact(
        model=model,
        split=split,
        seed=seed,
        resources=model.resources(),
        ideal_test_metrics=ideal_metrics,
        ideal_pooled_counts=ideal_pooled,
        training_time_seconds=record.training_time_seconds,
        n_training_iterations=record.n_iterations,
        initial_loss=record.initial_loss,
        final_loss=record.final_loss,
    )


@dataclass
class ExperimentRecord:
    """One fully-populated row of the experiment matrix -- every section 10 metric."""

    experiment_id: str
    seed: int
    dataset: str
    n_qubits: int
    logical_depth: int
    noise_condition: str
    shots: int

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]

    accuracy_ideal_baseline: float
    degradation_pp: float
    relative_degradation_pct: float
    robustness_ratio: float

    transpiled_depth: int
    total_gate_count: int
    two_qubit_gate_count: int
    n_trainable_parameters: int

    reused_training_from: str | None
    training_time_seconds: float
    eval_time_seconds: float
    runtime_seconds: float
    n_training_iterations: int
    initial_loss: float
    final_loss: float

    pooled_output_distribution: dict[str, float]
    distribution_total_variation_distance: float
    distribution_hellinger_distance: float
    n_outcomes_only_in_noisy: int

    status: str = "completed"

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "dataset": self.dataset,
            "n_qubits": self.n_qubits,
            "logical_depth": self.logical_depth,
            "noise_condition": self.noise_condition,
            "shots": self.shots,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix,
            "accuracy_ideal_baseline": self.accuracy_ideal_baseline,
            "degradation_pp": self.degradation_pp,
            "relative_degradation_pct": self.relative_degradation_pct,
            "robustness_ratio": self.robustness_ratio,
            "transpiled_depth": self.transpiled_depth,
            "total_gate_count": self.total_gate_count,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "n_trainable_parameters": self.n_trainable_parameters,
            "reused_training_from": self.reused_training_from,
            "training_time_seconds": self.training_time_seconds,
            "eval_time_seconds": self.eval_time_seconds,
            "runtime_seconds": self.runtime_seconds,
            "n_training_iterations": self.n_training_iterations,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "pooled_output_distribution": self.pooled_output_distribution,
            "distribution_total_variation_distance": self.distribution_total_variation_distance,
            "distribution_hellinger_distance": self.distribution_hellinger_distance,
            "n_outcomes_only_in_noisy": self.n_outcomes_only_in_noisy,
            "status": self.status,
        }


def evaluate_vqc_under_condition(
    artifact: TrainedVQCArtifact,
    experiment_id: str,
    noise_condition: str,
    *,
    dataset: str = "iris",
    reused_training_from: str | None = None,
    noise_probability: float | None = None,
) -> ExperimentRecord:
    """Evaluate a trained VQC's weights under a given noise condition.

    Swaps the noise model on the already-trained model (see module
    docstring); does not retrain. ``reused_training_from`` is purely
    informational bookkeeping -- the experiment id of the row whose
    training produced these weights, or ``None`` if this call is itself
    the baseline training's own ideal evaluation.
    """
    model = artifact.model
    start = time.perf_counter()
    model.set_noise_model(get_noise_model(noise_condition, noise_probability))
    predictions = model.predict(artifact.split.X_test, seed_simulator=artifact.seed)
    metrics = classification_metrics(artifact.split.y_test, predictions)
    counts = model.measurement_counts(artifact.split.X_test, seed_simulator=artifact.seed)
    pooled = _pool_counts(counts)
    eval_time = time.perf_counter() - start

    sensitivity = noise_sensitivity(artifact.ideal_test_metrics.accuracy, metrics.accuracy)
    dist_cmp = compare_distributions(artifact.ideal_pooled_counts, pooled)

    total_pooled = sum(pooled.values())
    pooled_fraction = (
        {k: v / total_pooled for k, v in pooled.items()} if total_pooled else {}
    )

    training_time = 0.0 if reused_training_from else artifact.training_time_seconds

    record = ExperimentRecord(
        experiment_id=experiment_id,
        seed=artifact.seed,
        dataset=dataset,
        n_qubits=artifact.resources.n_qubits,
        logical_depth=artifact.resources.logical_depth,
        noise_condition=noise_condition,
        shots=model.shots,
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        confusion_matrix=metrics.confusion_matrix,
        accuracy_ideal_baseline=artifact.ideal_test_metrics.accuracy,
        degradation_pp=sensitivity.degradation_pp,
        relative_degradation_pct=sensitivity.relative_degradation_pct,
        robustness_ratio=sensitivity.robustness_ratio,
        transpiled_depth=artifact.resources.transpiled_depth,
        total_gate_count=artifact.resources.total_gate_count,
        two_qubit_gate_count=artifact.resources.two_qubit_gate_count,
        n_trainable_parameters=artifact.resources.n_trainable_parameters,
        reused_training_from=reused_training_from,
        training_time_seconds=training_time,
        eval_time_seconds=eval_time,
        runtime_seconds=training_time + eval_time,
        n_training_iterations=artifact.n_training_iterations,
        initial_loss=artifact.initial_loss,
        final_loss=artifact.final_loss,
        pooled_output_distribution=pooled_fraction,
        distribution_total_variation_distance=dist_cmp.total_variation_distance,
        distribution_hellinger_distance=dist_cmp.hellinger_distance,
        n_outcomes_only_in_noisy=dist_cmp.n_outcomes_only_in_noisy,
    )

    logger.info(
        "%s seed=%d qubits=%d depth=%d noise=%s accuracy=%.4f "
        "degradation=%+.2fpp robustness=%.4f tvd=%.4f runtime=%.2fs",
        experiment_id,
        artifact.seed,
        record.n_qubits,
        record.logical_depth,
        noise_condition,
        record.accuracy,
        record.degradation_pp,
        record.robustness_ratio,
        record.distribution_total_variation_distance,
        record.runtime_seconds,
    )
    return record
