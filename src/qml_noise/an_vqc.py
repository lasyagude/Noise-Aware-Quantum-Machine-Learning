"""Adaptive Noise-Aware VQC (AN-VQC) -- the proposed improvement (assignment
section 11 / task2 section 7), on binary Iris (setosa vs. versicolor).

Selection phase (validation data only -- never test data)
-----------------------------------------------------------

Six candidates (``config.AN_VQC_CANDIDATES``): qubits in {2, 4, 6} x depth in
{2, 4}. For each candidate, :func:`select_candidate`:

1. Carves a stratified validation set out of the TRAINING data only, via
   :func:`qml_noise.datasets.carve_validation` (never touches the test set).
2. Trains the candidate on the remaining "inner" training data
   (``config.AN_VQC_MAXITER`` COBYLA iterations -- a small budget, by
   instruction, to keep runtime low) and reads its ideal accuracy on the
   validation set: ``ValAccuracy(c)``.
3. Evaluates the SAME trained weights under ``config.AN_VQC_TARGET_NOISE_PROFILE``
   ("medium") on the same validation set, via the project's standard
   ``evaluate_vqc_under_condition`` (swap noise model, don't retrain). Its
   ``robustness_ratio`` (noisy/ideal validation accuracy, NaN-safe on a zero
   denominator -- see ``metrics.noise_sensitivity``) is ``Robustness(c)``.
4. ``NormalizedCost(c)`` is the min-max normalization of
   ``config.AN_VQC_COST_METRIC`` (total gate count) across the six
   candidates -- a purely structural quantity from
   ``circuits.describe_circuit``, independent of training or the seed.
5. ``Score(c) = alpha*ValAccuracy(c) + beta*Robustness(c) - gamma*NormalizedCost(c)``,
   with alpha/beta/gamma frozen in ``config.py`` (section 11 explicitly asks
   for this, and freezing them in a file reviewed before any test-set call
   is the point: nothing here is tuned against Robustness(c) outcomes).

The selected candidate is ``argmax Score(c)``. The "best circuit by ideal
validation accuracy" baseline is ``argmax ValAccuracy(c)`` from the same
validation pass (no extra training).

**Leakage.** :func:`select_candidate` is given a ``DatasetSplit`` and reads
only ``.X_train``/``.y_train`` from it (through ``carve_validation``); it
never reads ``.X_test``/``.y_test``. ``tests/test_an_vqc.py::
test_selection_never_touches_test_data`` substitutes a sentinel object for
``X_test``/``y_test`` that raises on any access, and asserts selection still
completes normally.

Comparison phase (the real test set, once selection is already frozen)
-------------------------------------------------------------------------

Five methods are compared on the untouched test set, each (re)trained on the
FULL original training set (not the validation-carved subset -- the
validation split only exists to select an architecture, then is discarded):

1. ``fixed_vqc``    -- ``config.AN_VQC_FIXED_BASELINE`` (4 qubits, depth 4).
2. ``ideal_best``   -- the architecture with the best ideal validation accuracy.
3. ``shallow_vqc``  -- ``config.AN_VQC_SHALLOW_BASELINE`` (2 qubits, depth 2).
4. ``an_vqc``       -- the architecture ``select_candidate`` actually chose.
5. ``logistic_regression`` / ``svm`` -- classical baselines on the identical split.

Methods that happen to select the same architecture in a given seed (e.g.
``an_vqc`` picking the fixed baseline) reuse one trained model rather than
retraining -- the same "train once, evaluate many ways" pattern as
``qml_noise.experiments``, with the reused training's cost attributed to
whichever method trained first (see ``reused_training_from`` in the saved
records).

Incremental saving and resume
-------------------------------

Three files, resumable independently, never overlapping with the mandatory
E1-E8 or extension result files:

- ``results/an_vqc_selection_log.jsonl`` -- one row per seed: every
  candidate's validation metrics and score, which was selected, which was
  the ideal-validation best. Keyed by ``seed``.
- ``results/an_vqc_results.jsonl`` -- one row per ``(method, seed,
  noise_condition)`` for the four quantum methods, reusing the project's
  standard ``ExperimentRecord`` schema (accuracy, degradation, robustness,
  gate/depth counts, ...) plus ``method``, ``selected_qubits``,
  ``selected_depth``.
- ``results/an_vqc_classical_results.jsonl`` -- one row per ``(model_name,
  seed)`` for Logistic Regression and SVM.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from qml_noise.circuits import describe_circuit
from qml_noise.classical import run_logistic_regression, run_svm
from qml_noise.config import (
    AN_VQC_ALPHA,
    AN_VQC_BETA,
    AN_VQC_CANDIDATES,
    AN_VQC_COST_METRIC,
    AN_VQC_FIXED_BASELINE,
    AN_VQC_GAMMA,
    AN_VQC_MAXITER,
    AN_VQC_SHALLOW_BASELINE,
    AN_VQC_TARGET_NOISE_PROFILE,
    AN_VQC_VAL_FRACTION,
    RESULTS_DIR,
    SHOTS,
    get_logger,
)
from qml_noise.datasets import DatasetSplit, carve_validation, load_dataset
from qml_noise.evaluation import TrainedVQCArtifact, evaluate_vqc_under_condition, train_ideal_vqc
from qml_noise.experiments import append_jsonl, load_jsonl

logger = get_logger(__name__)

AN_VQC_MODES: tuple[str, ...] = ("smoke", "full")
AN_VQC_MODE_SEEDS: dict[str, list[int]] = {"smoke": [0], "full": [0, 1, 2]}

METHOD_FIXED = "fixed_vqc"
METHOD_IDEAL_BEST = "ideal_best"
METHOD_SHALLOW = "shallow_vqc"
METHOD_ANVQC = "an_vqc"
QUANTUM_METHODS: tuple[str, ...] = (METHOD_FIXED, METHOD_IDEAL_BEST, METHOD_SHALLOW, METHOD_ANVQC)

AN_VQC_SELECTION_FILE: Path = RESULTS_DIR / "an_vqc_selection_log.jsonl"
AN_VQC_RESULTS_FILE: Path = RESULTS_DIR / "an_vqc_results.jsonl"
AN_VQC_CLASSICAL_FILE: Path = RESULTS_DIR / "an_vqc_classical_results.jsonl"


# ---------------------------------------------------------------------------
# Resource cost (structural; no training needed)
# ---------------------------------------------------------------------------


def candidate_normalized_costs(
    candidates: tuple[tuple[int, int], ...] = AN_VQC_CANDIDATES,
    *,
    metric: str = AN_VQC_COST_METRIC,
) -> dict[tuple[int, int], float]:
    """Min-max normalized ``metric`` (from ``circuits.describe_circuit``) per candidate.

    Purely structural -- the same for every seed. If every candidate has
    the same cost (degenerate), every ``NormalizedCost`` is 0.0 rather than
    dividing by zero.
    """
    values = {c: getattr(describe_circuit(*c), metric) for c in candidates}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {c: 0.0 for c in candidates}
    return {c: (v - lo) / (hi - lo) for c, v in values.items()}


# ---------------------------------------------------------------------------
# Selection phase
# ---------------------------------------------------------------------------


@dataclass
class CandidateValidationResult:
    """One candidate's validation-set metrics and AN-VQC score."""

    n_qubits: int
    depth: int
    val_accuracy_ideal: float
    val_accuracy_noisy: float
    robustness: float
    normalized_cost: float
    score: float

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "val_accuracy_ideal": self.val_accuracy_ideal,
            "val_accuracy_noisy": self.val_accuracy_noisy,
            "robustness": self.robustness,
            "normalized_cost": self.normalized_cost,
            "score": self.score,
        }


@dataclass
class ANVQCSelection:
    """The full selection record for one seed: every candidate, and the two picks."""

    seed: int
    val_fraction: float
    target_noise_profile: str
    alpha: float
    beta: float
    gamma: float
    maxiter: int
    candidates: list[CandidateValidationResult]
    selected: tuple[int, int]
    ideal_best: tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "target_noise_profile": self.target_noise_profile,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "maxiter": self.maxiter,
            "candidates": [c.to_dict() for c in self.candidates],
            "selected": list(self.selected),
            "ideal_best": list(self.ideal_best),
        }

    @staticmethod
    def from_dict(d: dict) -> "ANVQCSelection":
        return ANVQCSelection(
            seed=d["seed"],
            val_fraction=d["val_fraction"],
            target_noise_profile=d["target_noise_profile"],
            alpha=d["alpha"],
            beta=d["beta"],
            gamma=d["gamma"],
            maxiter=d["maxiter"],
            candidates=[CandidateValidationResult(**c) for c in d["candidates"]],
            selected=tuple(d["selected"]),
            ideal_best=tuple(d["ideal_best"]),
        )


def select_candidate(
    split: DatasetSplit,
    seed: int,
    *,
    candidates: tuple[tuple[int, int], ...] = AN_VQC_CANDIDATES,
    val_fraction: float = AN_VQC_VAL_FRACTION,
    maxiter: int = AN_VQC_MAXITER,
    target_noise_profile: str = AN_VQC_TARGET_NOISE_PROFILE,
    alpha: float = AN_VQC_ALPHA,
    beta: float = AN_VQC_BETA,
    gamma: float = AN_VQC_GAMMA,
    shots: int = SHOTS,
) -> ANVQCSelection:
    """Score every candidate on a validation set carved from TRAINING data only.

    Reads only ``split.X_train``/``split.y_train`` (via ``carve_validation``)
    -- see the module docstring's leakage note.
    """
    val_split = carve_validation(split, val_fraction=val_fraction, seed=seed)
    costs = candidate_normalized_costs(candidates)

    results: list[CandidateValidationResult] = []
    for n_qubits, depth in candidates:
        artifact: TrainedVQCArtifact = train_ideal_vqc(
            n_qubits, depth, seed, val_split, maxiter=maxiter, shots=shots
        )
        val_accuracy_ideal = artifact.ideal_test_metrics.accuracy

        noisy_record = evaluate_vqc_under_condition(
            artifact, f"SEL-{n_qubits}q-d{depth}", target_noise_profile, dataset="iris"
        )
        robustness = noisy_record.robustness_ratio
        # Safe zero-denominator handling for the SCORE only: a NaN robustness
        # (A_ideal_val == 0) contributes nothing rather than poisoning the
        # comparison; the raw (possibly NaN) value is still recorded.
        robustness_for_score = 0.0 if math.isnan(robustness) else robustness
        cost = costs[(n_qubits, depth)]
        score = alpha * val_accuracy_ideal + beta * robustness_for_score - gamma * cost

        results.append(
            CandidateValidationResult(
                n_qubits=n_qubits,
                depth=depth,
                val_accuracy_ideal=val_accuracy_ideal,
                val_accuracy_noisy=noisy_record.accuracy,
                robustness=robustness,
                normalized_cost=cost,
                score=score,
            )
        )

    selected = max(results, key=lambda r: r.score)
    ideal_best = max(results, key=lambda r: r.val_accuracy_ideal)

    selection = ANVQCSelection(
        seed=seed,
        val_fraction=val_fraction,
        target_noise_profile=target_noise_profile,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        maxiter=maxiter,
        candidates=results,
        selected=(selected.n_qubits, selected.depth),
        ideal_best=(ideal_best.n_qubits, ideal_best.depth),
    )
    logger.info(
        "seed=%d: AN-VQC selected %dq/d%d (score=%.4f); ideal-best validation is %dq/d%d (acc=%.4f)",
        seed,
        selection.selected[0],
        selection.selected[1],
        selected.score,
        selection.ideal_best[0],
        selection.ideal_best[1],
        ideal_best.val_accuracy_ideal,
    )
    return selection


# ---------------------------------------------------------------------------
# Comparison phase (test set)
# ---------------------------------------------------------------------------


def _method_architectures(selection: ANVQCSelection) -> dict[str, tuple[int, int]]:
    return {
        METHOD_FIXED: AN_VQC_FIXED_BASELINE,
        METHOD_IDEAL_BEST: selection.ideal_best,
        METHOD_SHALLOW: AN_VQC_SHALLOW_BASELINE,
        METHOD_ANVQC: selection.selected,
    }


def _load_selection_for_seed(path: Path, seed: int) -> ANVQCSelection | None:
    for record in load_jsonl(path):
        if record.get("seed") == seed:
            return ANVQCSelection.from_dict(record)
    return None


def _completed_result_keys(path: Path) -> set[tuple[str, int, str]]:
    return {
        (r["method"], r["seed"], r["noise_condition"])
        for r in load_jsonl(path)
        if r.get("status") == "completed"
    }


def _completed_classical_keys(path: Path) -> set[tuple[str, int]]:
    return {(r["model_name"], r["seed"]) for r in load_jsonl(path)}


def run_an_vqc_comparison(
    mode: str,
    *,
    seeds: list[int] | None = None,
    maxiter: int | None = None,
    candidates: tuple[tuple[int, int], ...] = AN_VQC_CANDIDATES,
    target_noise_profile: str = AN_VQC_TARGET_NOISE_PROFILE,
    selection_file: Path = AN_VQC_SELECTION_FILE,
    results_file: Path = AN_VQC_RESULTS_FILE,
    classical_file: Path = AN_VQC_CLASSICAL_FILE,
    force: bool = False,
) -> list[dict]:
    """Run selection + the five-way test-set comparison. Returns newly-saved records.

    Resumable: a seed whose selection is already logged is not reselected
    (avoiding retraining the six candidates); a ``(method, seed,
    noise_condition)`` already marked ``"completed"`` is skipped.

    ``candidates`` defaults to the frozen ``AN_VQC_CANDIDATES`` (all 6); tests
    may pass a smaller set to avoid paying for the 6-qubit/depth-4 candidate
    repeatedly. The fixed and shallow baselines are evaluated regardless of
    this override -- they are comparison baselines, not search candidates.
    """
    if mode not in AN_VQC_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {AN_VQC_MODES}")

    seeds = seeds if seeds is not None else AN_VQC_MODE_SEEDS[mode]
    maxiter = maxiter if maxiter is not None else AN_VQC_MAXITER
    if not seeds:
        raise ValueError("seeds must be a non-empty list")

    if force:
        for path in (selection_file, results_file, classical_file):
            if path.exists():
                path.unlink()
                logger.info("--force: deleted existing %s", path)

    selected_seeds = {r["seed"] for r in load_jsonl(selection_file)} if not force else set()
    completed_results = set() if force else _completed_result_keys(results_file)
    completed_classical = set() if force else _completed_classical_keys(classical_file)

    new_records: list[dict] = []
    for seed in seeds:
        full_split = load_dataset("iris", seed=seed)

        if seed in selected_seeds:
            selection = _load_selection_for_seed(selection_file, seed)
            logger.info("seed=%d: selection already logged, loaded from disk", seed)
        else:
            selection = select_candidate(
                full_split, seed, candidates=candidates, maxiter=maxiter,
                target_noise_profile=target_noise_profile,
            )
            append_jsonl(selection_file, selection.to_dict())
            selected_seeds.add(seed)
            new_records.append(selection.to_dict())
        assert selection is not None

        for result in (run_logistic_regression(full_split), run_svm(full_split)):
            key = (result.model_name, seed)
            if key in completed_classical:
                logger.info("Skipping classical %s seed=%d (already completed)", result.model_name, seed)
                continue
            record = result.to_dict()
            record["method"] = result.model_name
            append_jsonl(classical_file, record)
            completed_classical.add(key)
            new_records.append(record)

        method_arch = _method_architectures(selection)
        trained: dict[tuple[int, int], TrainedVQCArtifact] = {}
        origin_id: dict[tuple[int, int], str] = {}

        for method, (n_qubits, depth) in method_arch.items():
            need_ideal = (method, seed, "ideal") not in completed_results
            need_noisy = (method, seed, target_noise_profile) not in completed_results
            if not need_ideal and not need_noisy:
                logger.info("Skipping %s seed=%d (already completed)", method, seed)
                continue

            arch = (n_qubits, depth)
            if arch not in trained:
                trained[arch] = train_ideal_vqc(n_qubits, depth, seed, full_split, maxiter=maxiter)
                origin_id[arch] = f"{method}-ideal"
                reused_for_ideal = None
            else:
                reused_for_ideal = origin_id[arch]
            artifact = trained[arch]

            if need_ideal:
                record = evaluate_vqc_under_condition(
                    artifact, f"{method}-ideal", "ideal", dataset="iris",
                    reused_training_from=reused_for_ideal,
                )
                d = record.to_dict()
                d.update(method=method, selected_qubits=n_qubits, selected_depth=depth)
                append_jsonl(results_file, d)
                completed_results.add((method, seed, "ideal"))
                new_records.append(d)

            if need_noisy:
                record = evaluate_vqc_under_condition(
                    artifact, f"{method}-{target_noise_profile}", target_noise_profile, dataset="iris",
                    reused_training_from=origin_id[arch],
                )
                d = record.to_dict()
                d.update(method=method, selected_qubits=n_qubits, selected_depth=depth)
                append_jsonl(results_file, d)
                completed_results.add((method, seed, target_noise_profile))
                new_records.append(d)

    logger.info("AN-VQC run complete: %d new record(s) saved", len(new_records))
    return new_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Adaptive Noise-Aware VQC (AN-VQC) circuit-selection comparison."
    )
    parser.add_argument("--mode", choices=AN_VQC_MODES, required=True, help="smoke (1 seed) or full (seeds 0,1,2)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed override, e.g. '0,1,2'")
    parser.add_argument("--maxiter", type=int, default=None, help="COBYLA iteration budget override")
    parser.add_argument("--force", action="store_true", help="Delete existing AN-VQC results and rerun everything")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    run_an_vqc_comparison(args.mode, seeds=seeds, maxiter=args.maxiter, force=args.force)


if __name__ == "__main__":
    main()
