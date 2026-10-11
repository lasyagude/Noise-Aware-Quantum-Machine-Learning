"""Configurable Qiskit Aer noise models (assignment section 8).

How the noise models are constructed
------------------------------------

Every model is built with ``NoiseModel(basis_gates=BASIS_GATES)`` and errors
are attached with ``add_all_qubit_quantum_error`` to specific *transpiled*
instruction names:

- single-qubit errors attach to ``sx`` and ``x``,
- two-qubit errors attach to ``cx``,
- readout errors attach to ``measure`` via ``add_all_qubit_readout_error``.

``rz`` deliberately carries **no** error: on superconducting hardware it is a
virtual frame change (realized by phase-shifting later pulses), so it is
error-free. Attaching noise to it would overstate single-qubit noise.

**Transpilation assumption.** Noise only attaches to instructions that
actually appear in the executed circuit, so circuits must be transpiled into
``BASIS_GATES`` *before* execution (see
:func:`qml_noise.circuits.transpile_circuit`). A circuit left in terms of
``ry``/``rz``/``h`` would pick up *no* single-qubit noise at all, because
those instruction names are not in the noise model.

Two families of models
----------------------

1. **Combined profiles** (``ideal`` / ``low`` / ``medium``): depolarizing
   errors on one- and two-qubit gates plus a symmetric readout error. These
   are the assignment's three required noise conditions.
2. **Isolated channels** (``bit_flip`` / ``phase_flip`` / ``depolarizing`` /
   ``readout``): exactly one error channel at a time, for attributing
   degradation to a specific mechanism.

For the two-qubit version of an isolated single-qubit Pauli channel, the
single-qubit error is tensored with itself so that both CNOT qubits are
affected independently.

All numerical probabilities are chosen defaults defined in
:mod:`qml_noise.config`, not values mandated by the assignment.
"""

from __future__ import annotations

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error

from qml_noise.config import (
    BASIS_GATES,
    NOISE_PROFILES,
    NOISY_SINGLE_QUBIT_GATES,
    NOISY_TWO_QUBIT_GATES,
    NoiseProfile,
    get_logger,
)

logger = get_logger(__name__)

ISOLATED_CHANNELS: tuple[str, ...] = (
    "bit_flip",
    "phase_flip",
    "depolarizing",
    "readout",
)


def _validate_probability(p: float, label: str) -> None:
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"{label} must be a probability in [0, 1], got {p}")


def _new_noise_model() -> NoiseModel:
    return NoiseModel(basis_gates=list(BASIS_GATES))


def _add_readout_error(noise_model: NoiseModel, probability: float) -> None:
    """Attach a symmetric readout error: P(1|0) = P(0|1) = probability."""
    _validate_probability(probability, "readout probability")
    if probability <= 0.0:
        return
    error = ReadoutError(
        [
            [1.0 - probability, probability],
            [probability, 1.0 - probability],
        ]
    )
    noise_model.add_all_qubit_readout_error(error)


def build_profile_noise_model(profile: NoiseProfile | str) -> NoiseModel | None:
    """Build a combined depolarizing + readout model for a named profile.

    Returns ``None`` for the ideal profile, which signals "attach no noise
    model at all" rather than "attach a zero-strength model".
    """
    if isinstance(profile, str):
        if profile not in NOISE_PROFILES:
            raise ValueError(
                f"Unknown noise profile '{profile}'. "
                f"Available profiles: {tuple(NOISE_PROFILES)}"
            )
        profile = NOISE_PROFILES[profile]

    if profile.is_ideal:
        logger.debug("Noise profile '%s' is ideal; no noise model built", profile.name)
        return None

    _validate_probability(profile.single_qubit_depolarizing, "single_qubit_depolarizing")
    _validate_probability(profile.two_qubit_depolarizing, "two_qubit_depolarizing")

    noise_model = _new_noise_model()
    if profile.single_qubit_depolarizing > 0.0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(profile.single_qubit_depolarizing, 1),
            list(NOISY_SINGLE_QUBIT_GATES),
        )
    if profile.two_qubit_depolarizing > 0.0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(profile.two_qubit_depolarizing, 2),
            list(NOISY_TWO_QUBIT_GATES),
        )
    _add_readout_error(noise_model, profile.readout_error)

    logger.debug(
        "Built '%s' noise model: 1q=%.4f, 2q=%.4f, readout=%.4f, instructions=%s",
        profile.name,
        profile.single_qubit_depolarizing,
        profile.two_qubit_depolarizing,
        profile.readout_error,
        noise_model.noise_instructions,
    )
    return noise_model


def build_pauli_channel_noise_model(pauli: str, probability: float) -> NoiseModel | None:
    """Build an isolated single-Pauli channel model (``"X"`` or ``"Z"``).

    ``X`` gives a bit-flip channel, ``Z`` gives a phase-flip channel. The
    two-qubit error is the tensor product of the single-qubit error, so each
    CNOT qubit flips independently.
    """
    pauli = pauli.upper()
    if pauli not in ("X", "Y", "Z"):
        raise ValueError(f"pauli must be one of 'X', 'Y', 'Z'; got '{pauli}'")
    _validate_probability(probability, f"{pauli} flip probability")

    if probability <= 0.0:
        return None

    single = pauli_error([(pauli, probability), ("I", 1.0 - probability)])
    noise_model = _new_noise_model()
    noise_model.add_all_qubit_quantum_error(single, list(NOISY_SINGLE_QUBIT_GATES))
    noise_model.add_all_qubit_quantum_error(
        single.tensor(single), list(NOISY_TWO_QUBIT_GATES)
    )
    return noise_model


def build_bit_flip_noise_model(probability: float) -> NoiseModel | None:
    """Isolated bit-flip (Pauli-X) channel on gate instructions."""
    return build_pauli_channel_noise_model("X", probability)


def build_phase_flip_noise_model(probability: float) -> NoiseModel | None:
    """Isolated phase-flip (Pauli-Z) channel on gate instructions."""
    return build_pauli_channel_noise_model("Z", probability)


def build_depolarizing_noise_model(
    probability: float, *, two_qubit_probability: float | None = None
) -> NoiseModel | None:
    """Isolated depolarizing channel.

    If ``two_qubit_probability`` is omitted the same probability is used for
    one- and two-qubit gates, which makes the channel easy to interpret as a
    single swept parameter.
    """
    _validate_probability(probability, "depolarizing probability")
    p2 = probability if two_qubit_probability is None else two_qubit_probability
    _validate_probability(p2, "two-qubit depolarizing probability")

    if probability <= 0.0 and p2 <= 0.0:
        return None

    noise_model = _new_noise_model()
    if probability > 0.0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(probability, 1), list(NOISY_SINGLE_QUBIT_GATES)
        )
    if p2 > 0.0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p2, 2), list(NOISY_TWO_QUBIT_GATES)
        )
    return noise_model


def build_readout_noise_model(probability: float) -> NoiseModel | None:
    """Isolated readout (measurement) error, no gate errors."""
    _validate_probability(probability, "readout probability")
    if probability <= 0.0:
        return None
    noise_model = _new_noise_model()
    _add_readout_error(noise_model, probability)
    return noise_model


def build_isolated_channel_noise_model(
    channel: str, probability: float
) -> NoiseModel | None:
    """Dispatch to an isolated-channel builder by name."""
    channel = channel.lower()
    if channel not in ISOLATED_CHANNELS:
        raise ValueError(
            f"Unknown isolated channel '{channel}'. Available: {ISOLATED_CHANNELS}"
        )
    if channel == "bit_flip":
        return build_bit_flip_noise_model(probability)
    if channel == "phase_flip":
        return build_phase_flip_noise_model(probability)
    if channel == "depolarizing":
        return build_depolarizing_noise_model(probability)
    return build_readout_noise_model(probability)


def get_noise_model(condition: str, probability: float | None = None) -> NoiseModel | None:
    """Resolve any noise condition name to a noise model (or ``None``).

    ``condition`` is either a combined profile name (``"ideal"``, ``"low"``,
    ``"medium"``) or an isolated channel name, in which case ``probability``
    is required.
    """
    condition = condition.lower()
    if condition in NOISE_PROFILES:
        if probability is not None:
            logger.warning(
                "probability=%s ignored for combined profile '%s'",
                probability,
                condition,
            )
        return build_profile_noise_model(condition)
    if condition in ISOLATED_CHANNELS:
        if probability is None:
            raise ValueError(
                f"Isolated channel '{condition}' requires an explicit probability"
            )
        return build_isolated_channel_noise_model(condition, probability)
    raise ValueError(
        f"Unknown noise condition '{condition}'. "
        f"Profiles: {tuple(NOISE_PROFILES)}; channels: {ISOLATED_CHANNELS}"
    )


def get_backend(noise_model: NoiseModel | None = None, *, seed: int | None = None) -> AerSimulator:
    """Build an Aer simulator, optionally with a noise model attached."""
    kwargs: dict = {}
    if noise_model is not None:
        kwargs["noise_model"] = noise_model
    if seed is not None:
        kwargs["seed_simulator"] = seed
    return AerSimulator(**kwargs)
