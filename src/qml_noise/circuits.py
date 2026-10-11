"""Quantum circuit construction for the variational quantum classifier.

Architecture (assignment section 5)
-----------------------------------

**Feature encoding — angle encoding.** Each of the ``n_qubits`` qubits is
prepared with a Hadamard and then rotated by ``RY(s * x_i)`` where ``x_i`` is
the (standardized) i-th feature and ``s`` is ``encoding_scale``. The
Hadamard puts each qubit on the equator first so that an ``RY`` rotation
produces an immediately measurable amplitude change. Feature count is
matched to qubit count by :class:`~qml_noise.vqc.FeatureAdapter` (PCA to
reduce, cyclic tiling to expand).

**Ansatz — trainable RY/RZ rotations + CNOT entanglement.** One ansatz
block is:

1. ``RY(theta)`` on every qubit (trainable),
2. ``RZ(theta)`` on every qubit (trainable),
3. a ring of ``CNOT`` gates ``(i -> i+1 mod n)`` entangling all qubits.

For a single qubit no entangler is applied (a CNOT needs two qubits). The
"logical ansatz depth" is the number of these repeated blocks, exactly as
the assignment defines it: depth 2 means two repetitions.

**Trainable parameter count.** ``2 * n_qubits * depth`` (an RY and an RZ per
qubit per block). E.g. 4 qubits at depth 4 gives 32 trainable parameters.
The transpiled circuit depth is recorded separately via
:func:`circuit_resources`.

**Measurement / readout.** All qubits are measured in the computational
basis. Two documented binary readout strategies map the bitstring
distribution onto a class:

- ``"parity"`` (default): class 1 if the parity (XOR of all measured bits)
  is odd. Uses the whole register, so it is sensitive to noise on every
  qubit.
- ``"single"``: class 1 if the qubit-0 bit is 1; all other qubits act purely
  as an entangled register. More noise-robust, less expressive.

In both cases ``P(class 1)`` is estimated from the shot counts and the
predicted label is ``1`` if ``P(class 1) > 0.5``.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector

from qml_noise.config import (
    BASIS_GATES,
    NOISY_TWO_QUBIT_GATES,
    TRANSPILE_OPTIMIZATION_LEVEL,
    VQC_CONFIG,
    get_logger,
)

logger = get_logger(__name__)

READOUT_STRATEGIES: tuple[str, ...] = ("parity", "single")


@dataclass
class CircuitResources(object):
    """Resource/structure metrics for one circuit (assignment section 10)."""

    n_qubits: int
    logical_depth: int
    transpiled_depth: int
    total_gate_count: int
    two_qubit_gate_count: int
    n_trainable_parameters: int
    gate_counts: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "logical_depth": self.logical_depth,
            "transpiled_depth": self.transpiled_depth,
            "total_gate_count": self.total_gate_count,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "n_trainable_parameters": self.n_trainable_parameters,
            "gate_counts": dict(self.gate_counts),
        }


def n_trainable_parameters(n_qubits: int, depth: int) -> int:
    """Number of trainable weights: one RY and one RZ per qubit per block."""
    return 2 * n_qubits * depth


def build_feature_map(
    n_qubits: int, *, encoding_scale: float = VQC_CONFIG.encoding_scale
) -> tuple[QuantumCircuit, ParameterVector]:
    """Angle-encoding feature map: H then ``RY(scale * x_i)`` on each qubit."""
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")

    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="angle_encoding")
    for i in range(n_qubits):
        qc.h(i)
        qc.ry(encoding_scale * x[i], i)
    return qc, x


def build_ansatz(n_qubits: int, depth: int) -> tuple[QuantumCircuit, ParameterVector]:
    """Trainable ansatz: ``depth`` repetitions of RY/RZ rotations + CNOT ring."""
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if depth < 1:
        raise ValueError(f"depth (ansatz repetitions) must be >= 1, got {depth}")

    weights = ParameterVector("w", n_trainable_parameters(n_qubits, depth))
    qc = QuantumCircuit(n_qubits, name=f"ansatz_d{depth}")

    k = 0
    for _ in range(depth):
        for i in range(n_qubits):
            qc.ry(weights[k], i)
            k += 1
        for i in range(n_qubits):
            qc.rz(weights[k], i)
            k += 1
        if n_qubits > 1:
            for i in range(n_qubits):
                qc.cx(i, (i + 1) % n_qubits)
    return qc, weights


def build_vqc_circuit(
    n_qubits: int,
    depth: int,
    *,
    encoding_scale: float = VQC_CONFIG.encoding_scale,
    measure: bool = True,
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Assemble the full VQC circuit: feature map + ansatz + measurement.

    Returns the circuit together with its feature and weight parameter
    vectors, so callers can bind them independently.
    """
    feature_map, x = build_feature_map(n_qubits, encoding_scale=encoding_scale)
    ansatz, weights = build_ansatz(n_qubits, depth)

    qc = QuantumCircuit(n_qubits, name=f"vqc_q{n_qubits}_d{depth}")
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    if measure:
        qc.measure_all()
    return qc, x, weights


def transpile_circuit(
    circuit: QuantumCircuit,
    *,
    basis_gates: tuple[str, ...] = BASIS_GATES,
    optimization_level: int = TRANSPILE_OPTIMIZATION_LEVEL,
    seed_transpiler: int = 0,
) -> QuantumCircuit:
    """Transpile into the noise model's basis gates.

    Transpiling to the same basis set the noise model is built on is what
    makes the noise actually attach to the circuit's instructions. The
    circuit may still carry unbound parameters -- binding afterwards keeps
    the transpiled gate structure (and therefore the noise exposure)
    identical across all samples and optimizer iterations.
    """
    return transpile(
        circuit,
        basis_gates=list(basis_gates),
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )


def circuit_resources(
    circuit: QuantumCircuit,
    *,
    logical_depth: int,
    n_params: int,
    transpiled: QuantumCircuit | None = None,
) -> CircuitResources:
    """Collect resource metrics, excluding non-gate bookkeeping instructions.

    ``barrier`` and ``measure`` are not gates, so they are excluded from the
    gate counts but remain visible in ``gate_counts``.
    """
    tqc = transpile_circuit(circuit) if transpiled is None else transpiled
    counts = {str(k): int(v) for k, v in tqc.count_ops().items()}

    non_gates = {"barrier", "measure", "delay", "reset", "snapshot"}
    total_gates = sum(v for k, v in counts.items() if k not in non_gates)
    two_qubit = sum(v for k, v in counts.items() if k in NOISY_TWO_QUBIT_GATES)

    return CircuitResources(
        n_qubits=circuit.num_qubits,
        logical_depth=logical_depth,
        transpiled_depth=tqc.depth(),
        total_gate_count=total_gates,
        two_qubit_gate_count=two_qubit,
        n_trainable_parameters=n_params,
        gate_counts=counts,
    )


def describe_circuit(n_qubits: int, depth: int) -> CircuitResources:
    """Convenience: build a VQC circuit and report its resource metrics."""
    qc, _, _ = build_vqc_circuit(n_qubits, depth)
    return circuit_resources(
        qc, logical_depth=depth, n_params=n_trainable_parameters(n_qubits, depth)
    )
