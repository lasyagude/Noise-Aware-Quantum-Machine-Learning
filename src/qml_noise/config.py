"""Central configuration for the noise-aware QML project.

All paths, seeds, and shared experimental constants that other modules
depend on live here so that they stay consistent across the dataset,
classical-baseline, quantum, noise, training, and reporting modules.

A note on which values are *mandatory* vs. *chosen*:

- Mandatory (fixed by the assignment): ``SHOTS = 1024``,
  ``REQUIRED_QUBIT_COUNTS = (2, 4, 6)``, ``REQUIRED_DEPTHS = (2, 4)``,
  ``NOISE_CONDITIONS = ("ideal", "low", "medium")``, the 80:20 split, and
  the five random seeds.
- Chosen defaults (NOT specified by the assignment, documented and
  configurable here): every numerical noise probability in
  ``NOISE_PROFILES`` and ``ISOLATED_CHANNEL_PROBABILITIES``, the basis-gate
  set, which gates carry noise, and all VQC training hyperparameters.
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
# Quantum experimental parameters (assignment section 6)
# ---------------------------------------------------------------------------

# Mandatory required configurations.
REQUIRED_QUBIT_COUNTS: tuple[int, ...] = (2, 4, 6)
REQUIRED_DEPTHS: tuple[int, ...] = (2, 4)
NOISE_CONDITIONS: tuple[str, ...] = ("ideal", "low", "medium")

# Optional extensions (supported but not prioritized).
OPTIONAL_QUBIT_COUNTS: tuple[int, ...] = (2, 4, 6, 8)
OPTIONAL_DEPTHS: tuple[int, ...] = (2, 4, 6)

# ---------------------------------------------------------------------------
# Transpilation / basis gates (chosen defaults)
# ---------------------------------------------------------------------------

# IBM-like basis set. Circuits are transpiled into these gates so that the
# Aer noise model attaches to concrete hardware-level instructions.
BASIS_GATES: tuple[str, ...] = ("rz", "sx", "x", "cx")

# Gates that carry noise. `rz` is deliberately EXCLUDED: on superconducting
# hardware it is a virtual frame change (implemented by shifting the phase of
# subsequent pulses) and is therefore error-free. Including it would
# overstate single-qubit noise.
NOISY_SINGLE_QUBIT_GATES: tuple[str, ...] = ("sx", "x")
NOISY_TWO_QUBIT_GATES: tuple[str, ...] = ("cx",)

TRANSPILE_OPTIMIZATION_LEVEL: int = 1

# ---------------------------------------------------------------------------
# Noise profiles (chosen defaults -- NOT mandated by the assignment)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseProfile:
    """A combined noise profile applied to a whole circuit.

    Probabilities are depolarizing-channel parameters for single- and
    two-qubit gates, plus a symmetric readout (measurement) error.
    """

    name: str
    single_qubit_depolarizing: float
    two_qubit_depolarizing: float
    readout_error: float
    description: str

    @property
    def is_ideal(self) -> bool:
        return (
            self.single_qubit_depolarizing == 0.0
            and self.two_qubit_depolarizing == 0.0
            and self.readout_error == 0.0
        )


NOISE_PROFILES: dict[str, NoiseProfile] = {
    "ideal": NoiseProfile(
        name="ideal",
        single_qubit_depolarizing=0.0,
        two_qubit_depolarizing=0.0,
        readout_error=0.0,
        description="Noiseless reference simulation (no noise model attached).",
    ),
    "low": NoiseProfile(
        name="low",
        single_qubit_depolarizing=0.001,
        two_qubit_depolarizing=0.01,
        readout_error=0.01,
        description=(
            "Chosen default approximating a good current superconducting "
            "device: 0.1% single-qubit, 1% two-qubit, 1% readout error."
        ),
    ),
    "medium": NoiseProfile(
        name="medium",
        single_qubit_depolarizing=0.005,
        two_qubit_depolarizing=0.05,
        readout_error=0.03,
        description=(
            "Chosen default roughly 5x worse than 'low', to make degradation "
            "clearly measurable: 0.5% single-qubit, 5% two-qubit, 3% readout."
        ),
    ),
}

# Probabilities used for the isolated single-channel sweeps (bit-flip,
# phase-flip, depolarizing run one channel at a time). Chosen defaults.
ISOLATED_CHANNEL_PROBABILITIES: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)

# ---------------------------------------------------------------------------
# Extension experiments (chosen defaults -- NOT mandated by the assignment;
# see src/qml_noise/extensions.py)
# ---------------------------------------------------------------------------

# Depolarizing noise sweep: two-qubit-gate probability levels. The
# single-qubit probability at each level is this value times the ratio
# below (real hardware's single-qubit gates are typically ~10x cleaner
# than its two-qubit gates, so the ratio keeps the sweep qualitatively
# realistic rather than scanning an unphysical 1:1 regime).
NOISE_SWEEP_TWO_QUBIT_PROBABILITIES: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
NOISE_SWEEP_SINGLE_TO_TWO_QUBIT_RATIO: float = 0.1

# Breast Cancer extension: training-set subsample size. Chosen purely to
# keep VQC training runtime bounded on 30-feature data; evaluation always
# uses the FULL held-out test set, never a subsample of it.
BREAST_CANCER_TRAIN_SUBSAMPLE_SIZE: int = 100

# ---------------------------------------------------------------------------
# VQC training defaults (chosen defaults)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VQCConfig:
    """Hyperparameters for the variational quantum classifier."""

    # "parity" = parity of all measured qubits; "single" = qubit 0 only.
    readout_strategy: str = "parity"
    optimizer: str = "COBYLA"
    maxiter: int = 150
    tol: float = 1e-4
    shots: int = SHOTS
    # Initial weights are drawn uniformly from [-init_scale, +init_scale].
    init_scale: float = 0.1
    # Feature scaling applied before angle encoding, so that standardized
    # features land in a sensible rotation range rather than wrapping.
    encoding_scale: float = 1.0


VQC_CONFIG: VQCConfig = VQCConfig()


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
