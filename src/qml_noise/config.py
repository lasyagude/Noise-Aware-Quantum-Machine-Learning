"""Central configuration for the noise-aware QML project.

All paths, seeds, and shared experimental constants that other modules
depend on live here so that they stay consistent across the dataset,
classical-baseline, quantum, noise, training, and reporting modules.

Quantum-circuit-specific settings (qubit counts, ansatz depths, noise
channel parameters, the experiment matrix) are intentionally NOT defined
here yet -- they belong to later modules (circuits/noise/experiments)
and will be added alongside those modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RESULTS_DIR: Path = PROJECT_ROOT / "results"
PLOTS_DIR: Path = PROJECT_ROOT / "plots"
REPORT_DIR: Path = PROJECT_ROOT / "report"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

for _dir in (RESULTS_DIR, PLOTS_DIR, REPORT_DIR, CONFIGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# The assignment asks for five random seeds "where computationally feasible".
RANDOM_SEEDS: list[int] = [0, 1, 2, 3, 4]
DEFAULT_SEED: int = RANDOM_SEEDS[0]

# ---------------------------------------------------------------------------
# Dataset / preprocessing defaults
# ---------------------------------------------------------------------------

TEST_SIZE: float = 0.2
SUPPORTED_DATASETS: tuple[str, ...] = ("iris", "wine", "breast_cancer")
PRIMARY_DATASET: str = "iris"

# ---------------------------------------------------------------------------
# Measurement / shots (mandatory assignment value)
# ---------------------------------------------------------------------------

SHOTS: int = 1024

# ---------------------------------------------------------------------------
# Classical baseline defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassicalBaselineConfig:
    """Hyperparameters for the classical baseline models."""

    logistic_regression_max_iter: int = 1000
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    random_state: int = DEFAULT_SEED


CLASSICAL_CONFIG: ClassicalBaselineConfig = ClassicalBaselineConfig()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured module-level logger.

    Using a shared factory keeps log formatting consistent across the
    dataset, classical, and (later) quantum/noise/experiment modules.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
