"""Central configuration for the noise-aware QML project.

All paths, seeds, and shared experimental constants that other modules
depend on live here so that they stay consistent across the dataset,
classical-baseline, quantum, noise, training, and reporting modules.

A note on which values are *mandatory* vs. *chosen*:

- Mandatory (fixed by the assignment): ``SHOTS = 1024``,
  ``REQUIRED_QUBIT_COUNTS = (2, 4, 6)``, ``REQUIRED_DEPTHS = (2, 4)``,
  ``NOISE_CONDITIONS = ("ideal", "low", "medium")``, the 80:20 split, and
  the five random seeds.
- **Assignment-specified** (added later, from the assignment's noise-level
  guideline table -- not chosen by this project): the ``low`` and
  ``medium`` entries of ``NOISE_PROFILES`` (0.1%/0.5%/1% and 1%/2%/3%
  single-qubit/two-qubit/readout error respectively).
- Chosen defaults (NOT specified by the assignment, documented and
  configurable here): the optional ``high`` entry of ``NOISE_PROFILES``
  (picked within the table's stated "3-5% gate error" range, since the
  table does not give it exact numbers), ``ISOLATED_CHANNEL_PROBABILITIES``,
  the basis-gate set, which gates carry noise, and all VQC training
  hyperparameters.

.. note::
   ``low``/``medium`` were corrected on 2026-10-11 to match the
   assignment's guideline table exactly (previous values: low
   0.1%/1%/1%, medium 0.5%/5%/3% -- chosen defaults at the time, since no
   table had been supplied). Any saved result row evaluated under the
   ``"low"`` or ``"medium"`` noise condition before that date used the old
   values and must be regenerated; see ``PROGRESS.md`` for the exact
   affected files and regeneration commands.
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
# Noise profiles
#
# "low" and "medium" are ASSIGNMENT-SPECIFIED: they match the assignment's
# noise-level guideline table exactly (Level 1 "low" and Level 2
# "moderate"). "ideal" is the trivial no-noise case. "high" is OPTIONAL --
# the table names a Level 3 "high" tier with "3-5% gate error, higher
# readout error" but gives no exact numbers, so its values here are a
# chosen default picked within that stated range, documented as such.
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
        two_qubit_depolarizing=0.005,
        readout_error=0.01,
        description=(
            "Assignment-specified Level 1 'low': 0.1% single-qubit, "
            "0.5% two-qubit, 1% readout error (per the noise-level "
            "guideline table)."
        ),
    ),
    "medium": NoiseProfile(
        name="medium",
        single_qubit_depolarizing=0.01,
        two_qubit_depolarizing=0.02,
        readout_error=0.03,
        description=(
            "Assignment-specified Level 2 'moderate': 1% single-qubit, "
            "2% two-qubit, 3% readout error (per the noise-level "
            "guideline table)."
        ),
    ),
    "high": NoiseProfile(
        name="high",
        single_qubit_depolarizing=0.03,
        two_qubit_depolarizing=0.05,
        readout_error=0.05,
        description=(
            "OPTIONAL Level 3 'high'. The assignment's guideline table "
            "names this tier and states a range ('3-5% gate error, higher "
            "readout error') but gives no exact numbers; 3% single-qubit, "
            "5% two-qubit, 5% readout error is a chosen default picked "
            "within that stated range, not a value the table specifies "
            "directly. Not part of the mandatory ideal/low/medium matrix."
        ),
    ),
}

# The three MANDATORY noise conditions (NOISE_CONDITIONS, above) vs. the
# optional ones supported but not required.
OPTIONAL_NOISE_CONDITIONS: tuple[str, ...] = ("high",)

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
# Adaptive Noise-Aware VQC (AN-VQC) selection (assignment section 11;
# chosen defaults -- NOT mandated by the assignment). These are FROZEN here
# -- read by src/qml_noise/an_vqc.py -- before any test-set evaluation is
# performed; see that module's docstring for the selection procedure and
# tests/test_an_vqc.py for a direct proof that selection never reads test
# data.
# ---------------------------------------------------------------------------

# Six candidates: the three required qubit counts x the two required depths.
AN_VQC_CANDIDATES: tuple[tuple[int, int], ...] = (
    (2, 2), (2, 4), (4, 2), (4, 4), (6, 2), (6, 4),
)

# Noise profile Robustness(c) is estimated under. Must be one of
# NOISE_PROFILES's keys.
AN_VQC_TARGET_NOISE_PROFILE: str = "medium"

# Fraction of the TRAINING data (never the test data) carved out, stratified,
# as a validation set for candidate scoring. With Iris's 80-sample training
# set this gives 60 "inner-train" / 20 validation.
AN_VQC_VAL_FRACTION: float = 0.25

# Small budget shared by the selection phase (candidate training) AND the
# final test-time retraining, to keep total runtime low as instructed.
AN_VQC_MAXITER: int = 60

# Score(c) = alpha*ValAccuracy(c) + beta*Robustness(c) - gamma*NormalizedCost(c).
# Weights sum to 1.0 for interpretability (not a requirement): accuracy
# weighted highest, robustness second, cost penalized least but non-zero.
AN_VQC_ALPHA: float = 0.5
AN_VQC_BETA: float = 0.3
AN_VQC_GAMMA: float = 0.2

# NormalizedCost(c): min-max normalization of this resource metric (from
# circuits.CircuitResources) across AN_VQC_CANDIDATES. Gate count is chosen
# over depth because it tracks both circuit size and noise exposure (more
# gates = more error-carrying instructions) more directly than depth alone.
# This is a purely structural quantity -- it does not depend on training or
# the seed, and is computed once.
AN_VQC_COST_METRIC: str = "total_gate_count"

# Comparison baselines (assignment section 11, items 1 and 3). Both are
# members of AN_VQC_CANDIDATES.
AN_VQC_FIXED_BASELINE: tuple[int, int] = (4, 4)
AN_VQC_SHALLOW_BASELINE: tuple[int, int] = (2, 2)

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
