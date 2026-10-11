"""Tests for the Aer noise-model module.

These tests verify the property that actually matters for the whole study:
noise attaches to the concrete gate instructions that appear in the
*transpiled* circuit, and `rz` stays error-free.
"""

import pytest

from qml_noise.circuits import build_vqc_circuit, transpile_circuit
from qml_noise.config import NOISE_CONDITIONS, NOISE_PROFILES, OPTIONAL_NOISE_CONDITIONS
from qml_noise.metrics import total_variation_distance
from qml_noise.noise import (
    build_bit_flip_noise_model,
    build_depolarizing_noise_model,
    build_isolated_channel_noise_model,
    build_phase_flip_noise_model,
    build_profile_noise_model,
    build_readout_noise_model,
    get_backend,
    get_noise_model,
)


def _transpiled_ops(n_qubits=4, depth=2):
    qc, _, _ = build_vqc_circuit(n_qubits, depth)
    tqc = transpile_circuit(qc)
    return tqc, {str(k) for k in tqc.count_ops()}


# -- combined profiles ------------------------------------------------------


def test_ideal_profile_builds_no_noise_model():
    assert build_profile_noise_model("ideal") is None


@pytest.mark.parametrize("profile", ["low", "medium"])
def test_profiles_target_gate_and_measure_instructions(profile):
    nm = build_profile_noise_model(profile)
    instructions = set(nm.noise_instructions)
    assert {"sx", "x"} <= instructions
    assert "cx" in instructions
    assert "measure" in instructions


@pytest.mark.parametrize("profile", ["low", "medium"])
def test_rz_is_never_noisy(profile):
    """`rz` is a virtual frame change on real hardware, so it must stay clean."""
    nm = build_profile_noise_model(profile)
    assert "rz" not in nm.noise_instructions


@pytest.mark.parametrize("profile", ["low", "medium"])
def test_every_noise_carrying_gate_in_the_circuit_is_covered(profile):
    """The direction that matters: no noisy gate in the circuit is missed."""
    _, ops = _transpiled_ops()
    nm = build_profile_noise_model(profile)
    must_be_noisy = {"sx", "x", "cx", "measure"} & ops
    assert must_be_noisy  # the circuit really does contain noisy gates
    assert must_be_noisy <= set(nm.noise_instructions)


def test_transpiled_circuit_uses_only_basis_gates():
    _, ops = _transpiled_ops()
    assert ops <= {"rz", "sx", "x", "cx", "measure", "barrier"}


def test_unknown_profile_name_raises():
    with pytest.raises(ValueError):
        build_profile_noise_model("catastrophic")


def test_profile_probabilities_are_ordered_low_below_medium():
    low = NOISE_PROFILES["low"]
    medium = NOISE_PROFILES["medium"]
    assert low.single_qubit_depolarizing < medium.single_qubit_depolarizing
    assert low.two_qubit_depolarizing < medium.two_qubit_depolarizing
    assert low.readout_error < medium.readout_error
    assert NOISE_PROFILES["ideal"].is_ideal
    assert not low.is_ideal


def test_low_and_medium_match_the_assignment_guideline_table_exactly():
    """Level 1 'low' and Level 2 'moderate' from the assignment's noise-level
    guideline table, not values chosen by this project."""
    low = NOISE_PROFILES["low"]
    assert low.single_qubit_depolarizing == pytest.approx(0.001)  # 0.1%
    assert low.two_qubit_depolarizing == pytest.approx(0.005)  # 0.5%
    assert low.readout_error == pytest.approx(0.01)  # 1%

    medium = NOISE_PROFILES["medium"]
    assert medium.single_qubit_depolarizing == pytest.approx(0.01)  # 1%
    assert medium.two_qubit_depolarizing == pytest.approx(0.02)  # 2%
    assert medium.readout_error == pytest.approx(0.03)  # 3%


def test_optional_high_profile_exists_within_the_tables_stated_range():
    """Level 3 'high' is optional; the table gives a range (3-5% gate error),
    not exact numbers, so these specific values are a documented chosen
    default -- but they must fall inside that stated range."""
    high = NOISE_PROFILES["high"]
    assert 0.03 <= high.single_qubit_depolarizing <= 0.05
    assert 0.03 <= high.two_qubit_depolarizing <= 0.05
    assert "high" in OPTIONAL_NOISE_CONDITIONS
    assert "high" not in NOISE_CONDITIONS
    # high must be the harshest named profile.
    medium = NOISE_PROFILES["medium"]
    assert high.single_qubit_depolarizing > medium.single_qubit_depolarizing
    assert high.two_qubit_depolarizing > medium.two_qubit_depolarizing
    assert high.readout_error > medium.readout_error


def test_high_profile_is_usable_through_get_noise_model():
    nm = get_noise_model("high")
    assert nm is not None
    assert {"sx", "x", "cx", "measure"} <= set(nm.noise_instructions)


# -- isolated channels ------------------------------------------------------


def test_bit_flip_channel_targets_gates_but_not_measurement():
    nm = build_bit_flip_noise_model(0.05)
    instructions = set(nm.noise_instructions)
    assert {"sx", "x", "cx"} <= instructions
    assert "measure" not in instructions


def test_phase_flip_channel_targets_gates_but_not_measurement():
    nm = build_phase_flip_noise_model(0.05)
    assert "measure" not in set(nm.noise_instructions)
    assert "cx" in set(nm.noise_instructions)


def test_readout_channel_targets_only_measurement():
    nm = build_readout_noise_model(0.05)
    assert set(nm.noise_instructions) == {"measure"}


def test_depolarizing_channel_accepts_separate_two_qubit_probability():
    nm = build_depolarizing_noise_model(0.01, two_qubit_probability=0.08)
    assert {"sx", "cx"} <= set(nm.noise_instructions)


def test_zero_probability_builds_no_model():
    assert build_bit_flip_noise_model(0.0) is None
    assert build_phase_flip_noise_model(0.0) is None
    assert build_readout_noise_model(0.0) is None
    assert build_depolarizing_noise_model(0.0) is None


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_out_of_range_probabilities_raise(bad):
    with pytest.raises(ValueError):
        build_bit_flip_noise_model(bad)
    with pytest.raises(ValueError):
        build_readout_noise_model(bad)


def test_unknown_channel_name_raises():
    with pytest.raises(ValueError):
        build_isolated_channel_noise_model("cosmic_rays", 0.1)


# -- dispatch ---------------------------------------------------------------


def test_get_noise_model_dispatches_profiles_and_channels():
    assert get_noise_model("ideal") is None
    assert get_noise_model("medium") is not None
    assert get_noise_model("bit_flip", 0.05) is not None


def test_isolated_channel_requires_explicit_probability():
    with pytest.raises(ValueError):
        get_noise_model("bit_flip")


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        get_noise_model("very_bad_day")


# -- empirical effect -------------------------------------------------------


def test_noise_actually_changes_measurement_outcomes():
    tqc, _ = _transpiled_ops(n_qubits=2, depth=2)
    tqc = tqc.assign_parameters([0.3] * tqc.num_parameters)

    ideal = get_backend(None).run(tqc, shots=1024, seed_simulator=11).result().get_counts()
    noisy = (
        get_backend(get_noise_model("medium"))
        .run(tqc, shots=1024, seed_simulator=11)
        .result()
        .get_counts()
    )
    assert ideal != noisy
    assert total_variation_distance(ideal, noisy) > 0.0


def test_stronger_noise_shifts_the_distribution_further():
    tqc, _ = _transpiled_ops(n_qubits=2, depth=4)
    tqc = tqc.assign_parameters([0.3] * tqc.num_parameters)

    def counts(condition):
        backend = get_backend(get_noise_model(condition))
        return backend.run(tqc, shots=1024, seed_simulator=5).result().get_counts()

    ideal = counts("ideal")
    tvd_low = total_variation_distance(ideal, counts("low"))
    tvd_medium = total_variation_distance(ideal, counts("medium"))
    assert tvd_medium > tvd_low


def test_heavy_bit_flip_noise_severely_disturbs_output():
    tqc, _ = _transpiled_ops(n_qubits=2, depth=4)
    tqc = tqc.assign_parameters([0.3] * tqc.num_parameters)
    ideal = get_backend(None).run(tqc, shots=1024, seed_simulator=5).result().get_counts()
    heavy = (
        get_backend(get_noise_model("bit_flip", 0.2))
        .run(tqc, shots=1024, seed_simulator=5)
        .result()
        .get_counts()
    )
    assert total_variation_distance(ideal, heavy) > 0.1
