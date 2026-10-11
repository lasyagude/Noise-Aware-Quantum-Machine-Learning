"""Extension experiments -- run only on request, never part of the
mandatory E1-E8 matrix, and never written into its result files
(``results/vqc_experiment_results.jsonl`` /
``results/classical_baseline_results.jsonl`` are opened read-only, if at
all, by this module).

Every record this module saves carries an explicit ``"extension"`` field
and lives in its own ``results/extension_*`` file, per the instruction to
label extension output everywhere and never overwrite the existing E1-E8
results.

Three independent pieces:

1. **Split-uniqueness check + pooled analysis**
   (:func:`check_split_uniqueness`, :func:`pooled_analysis`). A read-only
   check of whether each seed's stratified train/test split actually
   differs, and a pooled/bootstrap analysis computed from the EXISTING,
   already-saved E1-E8 results. No retraining, no new quantum circuits.
2. **Depolarizing noise sweep** (:func:`run_noise_sweep_extension`).
   Trains the 4-qubit-depth-4 and 6-qubit-depth-4 architectures once per
   seed under ideal conditions (exactly as the main matrix does -- weights
   are not persisted across process runs, so this is a fresh ideal
   training, not a reload of E4/E7's saved weights), then evaluates those
   same weights, with no further retraining, under a sweep of depolarizing
   probabilities.
3. **Breast Cancer extension** (:func:`run_breast_cancer_extension`).
   Reruns the E1-E8 matrix *shape* on Breast Cancer Wisconsin. Feature
   reduction to the qubit count is PCA fit on training data only -- this
   is already how ``VariationalQuantumClassifier``'s internal
   ``FeatureAdapter`` works (reused unchanged, see :mod:`qml_noise.vqc`).
   Training is restricted to a stratified subsample of the training set
   (``config.BREAST_CANCER_TRAIN_SUBSAMPLE_SIZE``) to keep runtime
   bounded; evaluation always uses the full held-out test set. Logistic
   Regression and SVM are run on the exact same split.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as sk_train_test_split

from qml_noise.config import (
    BREAST_CANCER_TRAIN_SUBSAMPLE_SIZE,
    NOISE_SWEEP_SINGLE_TO_TWO_QUBIT_RATIO,
    NOISE_SWEEP_TWO_QUBIT_PROBABILITIES,
    RANDOM_SEEDS,
    RESULTS_DIR,
    TEST_SIZE,
    VQC_CONFIG,
    get_logger,
)
from qml_noise.datasets import load_dataset, subsample_training_stratified
from qml_noise.evaluation import evaluate_vqc_with_noise_model, train_ideal_vqc
from qml_noise.experiments import append_jsonl, load_jsonl, run_matrix
from qml_noise.noise import build_depolarizing_noise_model

logger = get_logger(__name__)

EXTENSION_MODES: tuple[str, ...] = ("smoke", "full")
EXTENSION_MODE_SEEDS: dict[str, list[int]] = {
    "smoke": list(RANDOM_SEEDS[:1]),
    "full": list(RANDOM_SEEDS),
}
EXTENSION_MODE_MAXITER: dict[str, int] = {
    "smoke": 20,
    "full": VQC_CONFIG.maxiter,
}

SPLIT_CHECK_FILE: Path = RESULTS_DIR / "extension_split_check.json"
POOLED_ANALYSIS_FILE: Path = RESULTS_DIR / "extension_pooled_analysis.json"
NOISE_SWEEP_FILE: Path = RESULTS_DIR / "extension_noise_sweep.jsonl"
BC_RESULTS_FILE: Path = RESULTS_DIR / "extension_breast_cancer_results.jsonl"
BC_RESULTS_CSV: Path = RESULTS_DIR / "extension_breast_cancer_results.csv"
BC_CLASSICAL_FILE: Path = RESULTS_DIR / "extension_breast_cancer_classical_results.jsonl"

NOISE_SWEEP_ARCHITECTURES: tuple[tuple[int, int], ...] = ((4, 4), (6, 4))


# ---------------------------------------------------------------------------
# 1a. Split-uniqueness check
# ---------------------------------------------------------------------------


def check_split_uniqueness(
    *, seeds: Sequence[int] = tuple(RANDOM_SEEDS), test_size: float = TEST_SIZE
) -> dict:
    """Check whether each seed's stratified Iris train/test split actually differs.

    Reproduces the exact ``train_test_split(..., random_state=seed,
    stratify=y)`` call ``datasets.load_iris_binary`` makes, additionally
    splitting a plain index array in parallel (stratified on the same
    ``y``), so we can compare *which samples* land in the test set across
    seeds without needing to change ``datasets.py``'s public return type.
    """
    data = load_iris()
    mask = data.target != 2
    y = data.target[mask]
    indices = np.arange(len(y))

    test_index_sets: dict[int, frozenset[int]] = {}
    for seed in seeds:
        _, test_idx = sk_train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=y
        )
        test_index_sets[seed] = frozenset(test_idx.tolist())

    seed_list = list(seeds)
    pairs = []
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            a, b = seed_list[i], seed_list[j]
            set_a, set_b = test_index_sets[a], test_index_sets[b]
            union = set_a | set_b
            jaccard = len(set_a & set_b) / len(union) if union else 1.0
            pairs.append(
                {
                    "seed_a": a,
                    "seed_b": b,
                    "identical": set_a == set_b,
                    "jaccard_similarity": jaccard,
                }
            )

    all_identical = all(p["identical"] for p in pairs) if pairs else True
    finding = (
        "All seeds produced the SAME train/test split -- stratified "
        "splitting did not vary with the seed as expected."
        if all_identical
        else "Each seed produces a DIFFERENT stratified train/test split "
        "(train_test_split's random_state is set to the seed value): "
        f"confirmed by comparing test-set sample membership across all "
        f"{len(pairs)} seed pairs -- none are identical."
    )

    result = {
        "dataset": "iris",
        "seeds_checked": seed_list,
        "test_set_size": len(next(iter(test_index_sets.values()))) if test_index_sets else 0,
        "pairwise_comparisons": pairs,
        "all_splits_identical": all_identical,
        "finding": finding,
    }
    logger.info("Split-uniqueness check: all_splits_identical=%s", all_identical)
    return result


# ---------------------------------------------------------------------------
# 1b. Pooled analysis across seeds, from the EXISTING E1-E8 results
# ---------------------------------------------------------------------------


def pooled_analysis(
    *,
    results_file: Path | None = None,
    noisy_experiments: Sequence[str] = ("E5", "E6", "E8"),
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict:
    """Pooled accuracy, paired per-seed degradation, and bootstrap CIs.

    Reads the existing (already-saved, real) E1-E8 results; never writes
    to that file. Raises if it is missing or empty, rather than silently
    analyzing nothing.

    ``results_file`` defaults to the real
    ``results/vqc_experiment_results.jsonl`` (imported lazily from
    ``qml_noise.experiments`` so this module's own default stays in sync
    with that module's, without creating an import-time circular read).
    """
    if results_file is None:
        from qml_noise.experiments import VQC_RESULTS_FILE

        results_file = VQC_RESULTS_FILE

    records = load_jsonl(results_file)
    if not records:
        raise FileNotFoundError(
            f"No records found in {results_file} -- run the E1-E8 matrix first "
            "(python -m qml_noise.experiments --mode full)"
        )

    by_exp: dict[str, list[dict]] = {}
    for r in records:
        by_exp.setdefault(r["experiment_id"], []).append(r)

    pooled_accuracy: dict[str, dict] = {}
    for exp_id, rows in by_exp.items():
        n_correct = 0
        n_total = 0
        for r in rows:
            cm = r["confusion_matrix"]
            n_correct += sum(cm[i][i] for i in range(len(cm)))
            n_total += sum(sum(row) for row in cm)
        pooled_accuracy[exp_id] = {
            "pooled_accuracy": n_correct / n_total if n_total else float("nan"),
            "n_seeds": len(rows),
            "n_total_predictions": n_total,
            "n_correct_predictions": n_correct,
        }

    paired_degradation: dict[str, dict] = {}
    bootstrap_ci: dict[str, dict] = {}
    rng = np.random.default_rng(bootstrap_seed)
    for exp in noisy_experiments:
        rows = sorted(by_exp.get(exp, []), key=lambda r: r["seed"])
        if not rows:
            logger.warning("No records found for experiment '%s'; skipping", exp)
            continue

        degs = [r["degradation_pp"] for r in rows]
        paired_degradation[exp] = {
            "seeds": [r["seed"] for r in rows],
            "degradation_pp": degs,
            "mean": stats.mean(degs),
            "std": stats.pstdev(degs),
        }

        degs_arr = np.array(degs)
        boot_means = np.array(
            [
                rng.choice(degs_arr, size=len(degs_arr), replace=True).mean()
                for _ in range(n_bootstrap)
            ]
        )
        low, high = np.percentile(boot_means, [2.5, 97.5])
        bootstrap_ci[exp] = {
            "mean_degradation_pp": float(degs_arr.mean()),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "n_seeds": len(degs),
        }

    return {
        "source_results_file": str(results_file),
        "n_source_records": len(records),
        "pooled_accuracy_per_experiment": pooled_accuracy,
        "paired_degradation": paired_degradation,
        "bootstrap_ci_degradation_pp": bootstrap_ci,
    }


# ---------------------------------------------------------------------------
# 2. Depolarizing noise sweep (evaluation only; trains once per architecture+seed)
# ---------------------------------------------------------------------------


def run_noise_sweep_extension(
    mode: str,
    *,
    seeds: list[int] | None = None,
    maxiter: int | None = None,
    dataset: str = "iris",
    architectures: tuple[tuple[int, int], ...] = NOISE_SWEEP_ARCHITECTURES,
    two_qubit_probabilities: Sequence[float] = NOISE_SWEEP_TWO_QUBIT_PROBABILITIES,
    single_to_two_qubit_ratio: float = NOISE_SWEEP_SINGLE_TO_TWO_QUBIT_RATIO,
    results_file: Path = NOISE_SWEEP_FILE,
    force: bool = False,
) -> list[dict]:
    """Depolarizing-probability sweep on the 4q/d4 and 6q/d4 architectures.

    One ideal training per (architecture, seed); every probability level is
    then an evaluation-only call against those same weights -- no
    retraining per level. Incremental saving + resume, keyed by
    ``(experiment_id, seed)``, exactly like the main matrix.
    """
    if mode not in EXTENSION_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {EXTENSION_MODES}")

    seeds = seeds if seeds is not None else EXTENSION_MODE_SEEDS[mode]
    maxiter = maxiter if maxiter is not None else EXTENSION_MODE_MAXITER[mode]
    if not seeds:
        raise ValueError("seeds must be a non-empty list")

    if force and results_file.exists():
        results_file.unlink()
        logger.info("--force: deleted existing %s", results_file)

    completed = (
        set()
        if force
        else {
            (r["experiment_id"], r["seed"])
            for r in load_jsonl(results_file)
            if r.get("status") == "completed"
        }
    )

    def _experiment_id(n_qubits: int, depth: int, p2: float) -> str:
        return f"NSWEEP-Q{n_qubits}D{depth}-P{p2:g}"

    new_records: list[dict] = []
    for seed in seeds:
        split = load_dataset(dataset, seed=seed)

        for n_qubits, depth in architectures:
            baseline_id = _experiment_id(n_qubits, depth, 0.0)
            pending = [
                p2
                for p2 in two_qubit_probabilities
                if (_experiment_id(n_qubits, depth, p2), seed) not in completed
            ]
            if not pending:
                logger.info(
                    "qubits=%d depth=%d seed=%d: noise sweep already complete, "
                    "skipping training",
                    n_qubits,
                    depth,
                    seed,
                )
                continue

            artifact = train_ideal_vqc(n_qubits, depth, seed, split, maxiter=maxiter)

            for p2 in two_qubit_probabilities:
                experiment_id = _experiment_id(n_qubits, depth, p2)
                if (experiment_id, seed) in completed:
                    logger.info("Skipping %s seed=%d (already completed)", experiment_id, seed)
                    continue

                p1 = p2 * single_to_two_qubit_ratio
                noise_model = (
                    None if p2 <= 0.0 else build_depolarizing_noise_model(p1, two_qubit_probability=p2)
                )
                reused_from = None if p2 <= 0.0 else baseline_id

                record = evaluate_vqc_with_noise_model(
                    artifact,
                    experiment_id,
                    noise_model,
                    noise_label=f"depolarizing(p1={p1:g},p2={p2:g})",
                    dataset=dataset,
                    reused_training_from=reused_from,
                )
                record_dict = record.to_dict()
                record_dict.update(
                    {
                        "extension": "noise_sweep",
                        "architecture": f"{n_qubits}q_depth{depth}",
                        "two_qubit_probability": p2,
                        "single_qubit_probability": p1,
                    }
                )
                append_jsonl(results_file, record_dict)
                completed.add((experiment_id, seed))
                new_records.append(record_dict)

    logger.info(
        "Noise sweep run complete: %d new record(s) saved to %s", len(new_records), results_file
    )
    return new_records


# ---------------------------------------------------------------------------
# 3. Breast Cancer extension
# ---------------------------------------------------------------------------


def run_breast_cancer_extension(
    mode: str,
    *,
    seeds: list[int] | None = None,
    maxiter: int | None = None,
    n_train_samples: int | None = None,
    results_file: Path = BC_RESULTS_FILE,
    results_csv: Path = BC_RESULTS_CSV,
    classical_results_file: Path = BC_CLASSICAL_FILE,
    force: bool = False,
) -> list[dict]:
    """Run the E1-E8 matrix shape on Breast Cancer Wisconsin (binary).

    Reuses :func:`qml_noise.experiments.run_matrix` unchanged for the
    grouping/training/resume/incremental-save machinery, supplying:

    - ``dataset="breast_cancer"`` (30 features; the VQC's internal
      ``FeatureAdapter`` PCA-reduces to each architecture's qubit count,
      fit on training data only -- see :mod:`qml_noise.vqc`),
    - a ``prepare_split`` hook that stratified-subsamples the training set
      to ``n_train_samples`` (default
      ``config.BREAST_CANCER_TRAIN_SUBSAMPLE_SIZE``) while leaving the full
      test set untouched, applied identically to classical and quantum
      models,
    - ``experiment_id_prefix="BC-"`` and an explicit
      ``extension: "breast_cancer"`` field on every saved record.
    """
    if mode not in EXTENSION_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {EXTENSION_MODES}")

    seeds = seeds if seeds is not None else EXTENSION_MODE_SEEDS[mode]
    maxiter = maxiter if maxiter is not None else EXTENSION_MODE_MAXITER[mode]
    n_train_samples = (
        n_train_samples if n_train_samples is not None else BREAST_CANCER_TRAIN_SUBSAMPLE_SIZE
    )

    def _prepare(split, seed):
        return subsample_training_stratified(split, n_samples=n_train_samples, seed=seed)

    return run_matrix(
        mode,
        seeds=seeds,
        maxiter=maxiter,
        dataset="breast_cancer",
        prepare_split=_prepare,
        experiment_id_prefix="BC-",
        extra_record_fields={
            "extension": "breast_cancer",
            "train_subsample_size": n_train_samples,
        },
        results_file=results_file,
        results_csv=results_csv,
        classical_results_file=classical_results_file,
        force=force,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run extension experiments: split-check, pooled analysis, "
        "noise sweep, Breast Cancer."
    )
    sub = parser.add_subparsers(dest="extension", required=True)

    sub.add_parser(
        "split-check",
        help="Check whether each seed's Iris split differs; writes extension_split_check.json",
    )

    pooled = sub.add_parser(
        "pooled-analysis",
        help="Pooled/bootstrap analysis of the EXISTING E1-E8 results (read-only)",
    )
    pooled.add_argument("--n-bootstrap", type=int, default=10_000)
    pooled.add_argument("--bootstrap-seed", type=int, default=0)

    sweep = sub.add_parser("noise-sweep", help="Depolarizing noise probability sweep")
    sweep.add_argument("--mode", choices=EXTENSION_MODES, required=True)
    sweep.add_argument("--seeds", type=str, default=None)
    sweep.add_argument("--maxiter", type=int, default=None)
    sweep.add_argument("--force", action="store_true")

    bc = sub.add_parser(
        "breast-cancer", help="Run the E1-E8 matrix shape on Breast Cancer Wisconsin"
    )
    bc.add_argument("--mode", choices=EXTENSION_MODES, required=True)
    bc.add_argument("--seeds", type=str, default=None)
    bc.add_argument("--maxiter", type=int, default=None)
    bc.add_argument("--train-samples", type=int, default=None)
    bc.add_argument("--force", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.extension == "split-check":
        result = check_split_uniqueness()
        SPLIT_CHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_CHECK_FILE.write_text(json.dumps(result, indent=2))
        logger.info("Wrote %s", SPLIT_CHECK_FILE)
    elif args.extension == "pooled-analysis":
        result = pooled_analysis(n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)
        POOLED_ANALYSIS_FILE.parent.mkdir(parents=True, exist_ok=True)
        POOLED_ANALYSIS_FILE.write_text(json.dumps(result, indent=2))
        logger.info("Wrote %s", POOLED_ANALYSIS_FILE)
    elif args.extension == "noise-sweep":
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        run_noise_sweep_extension(args.mode, seeds=seeds, maxiter=args.maxiter, force=args.force)
    elif args.extension == "breast-cancer":
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        run_breast_cancer_extension(
            args.mode,
            seeds=seeds,
            maxiter=args.maxiter,
            n_train_samples=args.train_samples,
            force=args.force,
        )


if __name__ == "__main__":
    main()
