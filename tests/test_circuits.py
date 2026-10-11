"""Tests for the VQC circuit construction module."""

import pytest

from qml_noise.circuits import (
    build_ansatz,
    build_feature_map,
    build_vqc_circuit,
    circuit_resources,
    describe_circuit,
    n_trainable_parameters,
    transpile_circuit,
)
from qml_noise.config import (
    OPTIONAL_DEPTHS,
    OPTIONAL_QUBIT_COUNTS,
    REQUIRED_DEPTHS,
    REQUIRED_QUBIT_COUNTS,
)


def test_parameter_count_formula():
    # 2 trainable rotations (RY, RZ) per qubit per ansatz block.
    assert n_trainable_parameters(2, 2) == 8
    assert n_trainable_parameters(4, 4) == 32
    assert n_trainable_parameters(6, 4) == 48


@pytest.mark.parametrize("n_qubits", REQUIRED_QUBIT_COUNTS)
@pytest.mark.parametrize("depth", REQUIRED_DEPTHS)
def test_required_configurations_build(n_qubits, depth):
    qc, x, w = build_vqc_circuit(n_qubits, depth)
    assert qc.num_qubits == n_qubits
    assert len(x) == n_qubits
    assert len(w) == n_trainable_parameters(n_qubits, depth)
    assert qc.num_parameters == n_qubits + n_trainable_parameters(n_qubits, depth)


@pytest.mark.parametrize("n_qubits", OPTIONAL_QUBIT_COUNTS)
@pytest.mark.parametrize("depth", OPTIONAL_DEPTHS)
def test_optional_configurations_are_supported(n_qubits, depth):
    qc, _, w = build_vqc_circuit(n_qubits, depth)
    assert qc.num_qubits == n_qubits
    assert len(w) == 2 * n_qubits * depth


def test_feature_map_uses_angle_encoding():
    qc, x = build_feature_map(4)
    ops = {str(k): v for k, v in qc.count_ops().items()}
    assert ops["h"] == 4
    assert ops["ry"] == 4
    assert len(x) == 4


def test_ansatz_has_trainable_rotations_and_entanglers():
    qc, w = build_ansatz(4, 2)
    ops = {str(k): v for k, v in qc.count_ops().items()}
    assert ops["ry"] == 8  # 4 qubits x 2 blocks
    assert ops["rz"] == 8
    assert ops["cx"] == 8  # ring of 4 CNOTs x 2 blocks
    assert len(w) == 16


def test_logical_depth_scales_block_count():
    shallow, _ = build_ansatz(4, 2)
    deep, _ = build_ansatz(4, 4)
    shallow_ops = {str(k): v for k, v in shallow.count_ops().items()}
    deep_ops = {str(k): v for k, v in deep.count_ops().items()}
    # Depth 4 means exactly twice the blocks of depth 2.
    for gate in ("ry", "rz", "cx"):
        assert deep_ops[gate] == 2 * shallow_ops[gate]


def test_single_qubit_ansatz_has_no_entangler():
    qc, _ = build_ansatz(1, 2)
    ops = {str(k) for k in qc.count_ops()}
    assert "cx" not in ops


def test_vqc_circuit_measures_all_qubits():
    qc, _, _ = build_vqc_circuit(4, 2)
    assert qc.num_clbits == 4
    assert "measure" in {str(k) for k in qc.count_ops()}


def test_vqc_circuit_without_measurement():
    qc, _, _ = build_vqc_circuit(4, 2, measure=False)
    assert qc.num_clbits == 0


def test_invalid_circuit_arguments_raise():
    with pytest.raises(ValueError):
        build_feature_map(0)
    with pytest.raises(ValueError):
        build_ansatz(4, 0)
    with pytest.raises(ValueError):
        build_ansatz(0, 2)


def test_transpilation_preserves_two_qubit_gate_count():
    qc, _, _ = build_vqc_circuit(4, 2)
    tqc = transpile_circuit(qc)
    assert tqc.count_ops()["cx"] == 8


def test_resources_report_logical_and_transpiled_depth_separately():
    res = describe_circuit(4, 4)
    assert res.n_qubits == 4
    assert res.logical_depth == 4
    assert res.n_trainable_parameters == 32
    # Transpiling into a hardware basis expands each rotation, so the
    # transpiled depth is much larger than the logical block count.
    assert res.transpiled_depth > res.logical_depth
    assert res.two_qubit_gate_count == 16


def test_resources_exclude_non_gate_instructions_from_gate_count():
    qc, _, _ = build_vqc_circuit(2, 2)
    res = circuit_resources(qc, logical_depth=2, n_params=8)
    non_gate_total = sum(
        v for k, v in res.gate_counts.items() if k in ("barrier", "measure")
    )
    assert non_gate_total > 0
    assert res.total_gate_count == sum(res.gate_counts.values()) - non_gate_total


def test_resources_to_dict_is_serializable_shape():
    d = describe_circuit(2, 2).to_dict()
    assert set(d) == {
        "n_qubits",
        "logical_depth",
        "transpiled_depth",
        "total_gate_count",
        "two_qubit_gate_count",
        "n_trainable_parameters",
        "gate_counts",
    }


def test_deeper_circuits_have_greater_transpiled_depth():
    assert describe_circuit(4, 4).transpiled_depth > describe_circuit(4, 2).transpiled_depth
