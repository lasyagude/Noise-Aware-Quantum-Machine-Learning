"""Experiment runner: the mandatory E1-E8 matrix, smoke/normal/full modes,
incremental JSON-Lines saving, and resume support (assignment sections 7, 12).

Modes
-----

- ``smoke``: 1 seed, a small COBYLA iteration budget. Exists purely to
  prove the pipeline runs end-to-end and saves results -- not to produce
  results anyone should draw conclusions from.
- ``normal``: 1 seed, the full training budget (`VQC_CONFIG.maxiter`). A
  single credible pass through the whole matrix.
- ``full``: all five seeds (`config.RANDOM_SEEDS`), the full training
  budget. The complete study the assignment asks for.

Incremental saving and resume
------------------------------

Every completed ``(experiment_id, seed)`` pair is appended as one JSON
object per line to ``results/vqc_experiment_results.jsonl`` immediately
after it finishes -- not buffered until the end of the run. A companion
``results/vqc_experiment_results.csv`` (scalar fields only; confusion
matrices and output distributions stay in the JSONL) is regenerated after
each run for quick inspection.

On startup, the runner reads whatever is already in the JSONL file and
skips any ``(experiment_id, seed)`` pair already marked ``"completed"``.
This means an interrupted run (Ctrl-C, crash, timeout) can simply be
restarted with the same command and will only redo the missing pairs.
``--force`` discards this: it deletes the existing result files and reruns
everything from scratch, rather than appending duplicate rows.

Caveat: trained model weights are **not** persisted across process runs.
If a resumed run still has pending rows within a (qubits, depth) group,
that architecture is retrained from scratch for the current seed (fast,
and deterministic given the same seed and the same `--maxiter`) -- only
already-completed `(experiment_id, seed)` rows are skipped, not training
itself.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from qml_noise.classical import run_all_classical_baselines
from qml_noise.config import RANDOM_SEEDS, RESULTS_DIR, VQC_CONFIG, get_logger
from qml_noise.datasets import load_dataset
from qml_noise.evaluation import evaluate_vqc_under_condition, train_ideal_vqc

logger = get_logger(__name__)

# The assignment's recommended eight-experiment matrix (section 7).
# (experiment_id, n_qubits, logical_depth, noise_condition) -- do not reorder
# or replace; the assignment explicitly says not to replace this matrix.
EXPERIMENT_MATRIX: tuple[tuple[str, int, int, str], ...] = (
    ("E1", 2, 2, "ideal"),
    ("E2", 2, 4, "ideal"),
    ("E3", 4, 2, "ideal"),
    ("E4", 4, 4, "ideal"),
    ("E5", 4, 4, "low"),
    ("E6", 4, 4, "medium"),
    ("E7", 6, 4, "ideal"),
    ("E8", 6, 4, "medium"),
)

MODES: tuple[str, ...] = ("smoke", "normal", "full")

MODE_SEEDS: dict[str, list[int]] = {
    "smoke": list(RANDOM_SEEDS[:1]),
    "normal": list(RANDOM_SEEDS[:1]),
    "full": list(RANDOM_SEEDS),
}

MODE_MAXITER: dict[str, int] = {
    "smoke": 20,
    "normal": VQC_CONFIG.maxiter,
    "full": VQC_CONFIG.maxiter,
}

VQC_RESULTS_FILE: Path = RESULTS_DIR / "vqc_experiment_results.jsonl"
VQC_RESULTS_CSV: Path = RESULTS_DIR / "vqc_experiment_results.csv"
CLASSICAL_RESULTS_FILE: Path = RESULTS_DIR / "classical_baseline_results.jsonl"

_CSV_SCALAR_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "seed",
    "dataset",
    "n_qubits",
    "logical_depth",
    "noise_condition",
    "shots",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "accuracy_ideal_baseline",
    "degradation_pp",
    "relative_degradation_pct",
    "robustness_ratio",
    "transpiled_depth",
    "total_gate_count",
    "two_qubit_gate_count",
    "n_trainable_parameters",
    "reused_training_from",
    "training_time_seconds",
    "eval_time_seconds",
    "runtime_seconds",
    "n_training_iterations",
    "initial_loss",
    "final_loss",
    "distribution_total_variation_distance",
    "distribution_hellinger_distance",
    "n_outcomes_only_in_noisy",
    "status",
)


def group_matrix_by_architecture(
    matrix: tuple[tuple[str, int, int, str], ...] = EXPERIMENT_MATRIX,
) -> dict[tuple[int, int], list[tuple[str, int, int, str]]]:
    """Group matrix rows sharing (n_qubits, logical_depth) for shared training."""
    groups: dict[tuple[int, int], list[tuple[str, int, int, str]]] = {}
    for row in matrix:
        _, n_qubits, depth, _ = row
        groups.setdefault((n_qubits, depth), []).append(row)
    return groups


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _completed_vqc_pairs(path: Path) -> set[tuple[str, int]]:
    return {
        (r["experiment_id"], r["seed"])
        for r in _load_jsonl(path)
        if r.get("status") == "completed"
    }


def _completed_classical_pairs(path: Path) -> set[tuple[str, int]]:
    return {(r["model_name"], r["seed"]) for r in _load_jsonl(path)}


def run_classical_baselines_for_seed(
    seed: int,
    dataset: str,
    results_file: Path = CLASSICAL_RESULTS_FILE,
    *,
    force: bool = False,
) -> list[dict]:
    """Run + save Logistic Regression and SVM for one seed, skipping if done.

    These are independent of (qubits, depth), so they are keyed by
    ``(model_name, seed)`` rather than ``(experiment_id, seed)``.
    """
    completed = set() if force else _completed_classical_pairs(results_file)
    split = load_dataset(dataset, seed=seed)
    saved = []
    for result in run_all_classical_baselines(split):
        key = (result.model_name, seed)
        if key in completed:
            logger.info(
                "Skipping classical %s seed=%d (already completed)",
                result.model_name,
                seed,
            )
            continue
        record = result.to_dict()
        _append_jsonl(results_file, record)
        saved.append(record)
        logger.info(
            "Saved classical %s seed=%d accuracy=%.4f",
            result.model_name,
            seed,
            result.accuracy,
        )
    return saved


def _write_summary_csv(jsonl_path: Path, csv_path: Path) -> None:
    """Regenerate a scalar-only CSV view of the JSONL results."""
    records = _load_jsonl(jsonl_path)
    if not records:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(_CSV_SCALAR_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def run_matrix(
    mode: str,
    *,
    seeds: list[int] | None = None,
    maxiter: int | None = None,
    dataset: str = "iris",
    matrix: tuple[tuple[str, int, int, str], ...] | None = None,
    results_file: Path = VQC_RESULTS_FILE,
    results_csv: Path = VQC_RESULTS_CSV,
    classical_results_file: Path = CLASSICAL_RESULTS_FILE,
    force: bool = False,
) -> list[dict]:
    """Run the E1-E8 matrix for the given mode. Returns newly-saved records.

    Already-completed ``(experiment_id, seed)`` pairs are skipped. Pass
    ``force=True`` to delete existing result files and rerun everything.
    ``matrix`` defaults to the mandatory ``EXPERIMENT_MATRIX``; tests may
    override it with a smaller synthetic matrix to exercise the resume/force
    bookkeeping without paying for the full 6-qubit architecture.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {MODES}")

    matrix = matrix if matrix is not None else EXPERIMENT_MATRIX
    seeds = seeds if seeds is not None else MODE_SEEDS[mode]
    maxiter = maxiter if maxiter is not None else MODE_MAXITER[mode]
    if not seeds:
        raise ValueError("seeds must be a non-empty list")
    if maxiter < 1:
        raise ValueError(f"maxiter must be >= 1, got {maxiter}")

    if force:
        for path in (results_file, classical_results_file):
            if path.exists():
                path.unlink()
                logger.info("--force: deleted existing %s", path)

    groups = group_matrix_by_architecture(matrix)
    completed = set() if force else _completed_vqc_pairs(results_file)

    total_pairs = len(matrix) * len(seeds)
    logger.info(
        "Starting '%s' run: seeds=%s, maxiter=%d, %d experiments x %d seeds "
        "= %d (experiment, seed) pairs, %d already completed",
        mode,
        seeds,
        maxiter,
        len(matrix),
        len(seeds),
        total_pairs,
        len(completed),
    )

    new_records: list[dict] = []
    for seed in seeds:
        run_classical_baselines_for_seed(seed, dataset, classical_results_file, force=force)
        split = load_dataset(dataset, seed=seed)

        for (n_qubits, depth), rows in groups.items():
            pending = [row for row in rows if (row[0], seed) not in completed]
            if not pending:
                logger.info(
                    "qubits=%d depth=%d seed=%d: all experiments already "
                    "completed, skipping training",
                    n_qubits,
                    depth,
                    seed,
                )
                continue

            artifact = train_ideal_vqc(n_qubits, depth, seed, split, maxiter=maxiter)
            ideal_experiment_id = next(
                (row[0] for row in rows if row[3] == "ideal"), None
            )

            for experiment_id, _, _, noise_condition in rows:
                if (experiment_id, seed) in completed:
                    logger.info("Skipping %s seed=%d (already completed)", experiment_id, seed)
                    continue
                reused_from = None if noise_condition == "ideal" else ideal_experiment_id
                record = evaluate_vqc_under_condition(
                    artifact,
                    experiment_id,
                    noise_condition,
                    dataset=dataset,
                    reused_training_from=reused_from,
                )
                record_dict = record.to_dict()
                _append_jsonl(results_file, record_dict)
                completed.add((experiment_id, seed))
                new_records.append(record_dict)

    if new_records:
        _write_summary_csv(results_file, results_csv)

    logger.info(
        "Run complete: %d new (experiment, seed) record(s) saved to %s",
        len(new_records),
        results_file,
    )
    return new_records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the noise-aware QML experiment matrix (E1-E8)."
    )
    parser.add_argument(
        "--mode", choices=MODES, required=True, help="smoke, normal, or full"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed override, e.g. '0,1,2'. Defaults to the mode's seeds.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="COBYLA iteration budget override. Defaults to the mode's budget.",
    )
    parser.add_argument("--dataset", type=str, default="iris")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing results and rerun everything instead of resuming.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    run_matrix(
        args.mode,
        seeds=seeds,
        maxiter=args.maxiter,
        dataset=args.dataset,
        force=args.force,
    )


if __name__ == "__main__":
    main()
