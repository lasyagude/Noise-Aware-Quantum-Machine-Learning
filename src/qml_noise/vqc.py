"""Trainable Variational Quantum Classifier (assignment section 5).

Training specification
----------------------

- **Feature encoding:** angle encoding (see :mod:`qml_noise.circuits`).
  Feature count is matched to qubit count by :class:`FeatureAdapter`: PCA
  when there are more features than qubits, cyclic tiling when there are
  fewer. The adapter is fit on the **training data only**.
- **Loss:** binary cross-entropy on the shot-estimated ``P(class 1)``.
  Probabilities are clipped to ``[eps, 1-eps]`` so that a 0/1024 shot
  outcome cannot produce an infinite loss.
- **Optimizer:** SciPy ``COBYLA`` (gradient-free, which suits noisy
  shot-based objective values where finite-difference gradients are
  unreliable).
- **Initialization:** weights drawn from ``Uniform(-init_scale, +init_scale)``
  via ``numpy.random.default_rng(seed)``, so a given seed always reproduces
  the same starting point.
- **Stopping criteria:** ``maxiter`` COBYLA iterations (default 150) or
  convergence within ``tol`` (default 1e-4), whichever comes first. The
  loss at every iteration is recorded in ``loss_history_``. Note that SciPy
  silently raises ``maxiter`` to at least ``n_parameters + 2``, so very
  small budgets are not honoured exactly.
- **Measurement-to-class mapping:** ``parity`` (default) or ``single``, both
  documented in :mod:`qml_noise.circuits`. ``P(class 1)`` is the total
  probability mass of bitstrings mapping to class 1; the predicted label is
  ``1`` when ``P(class 1) > 0.5``.

Execution model
---------------

The parameterized circuit is transpiled **once** into the noise model's
basis gates, then the feature and weight values are bound per sample. This
keeps the transpiled gate structure -- and therefore the noise exposure --
identical across every sample and every optimizer iteration, and it is much
faster than re-transpiling. All circuits for a batch are submitted to Aer in
a single ``run`` call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel
from scipy.optimize import minimize
from sklearn.decomposition import PCA

from qml_noise.circuits import (
    READOUT_STRATEGIES,
    build_vqc_circuit,
    circuit_resources,
    n_trainable_parameters,
    transpile_circuit,
)
from qml_noise.config import SHOTS, VQC_CONFIG, get_logger
from qml_noise.noise import get_backend

logger = get_logger(__name__)

_PROBABILITY_EPS = 1e-7


class FeatureAdapter:
    """Match an input feature count to the circuit's qubit count.

    - ``n_features > n_qubits``: PCA down to ``n_qubits`` components.
    - ``n_features < n_qubits``: cyclic tiling (feature ``i mod n_features``).
    - equal: identity.

    PCA is fit on the training data only; ``transform`` then applies the
    fitted projection to any later data.
    """

    def __init__(self, n_qubits: int, *, seed: int = 0) -> None:
        self.n_qubits = n_qubits
        self.seed = seed
        self.n_features_in_: int | None = None
        self.mode_: str | None = None
        self._pca: PCA | None = None

    def fit(self, X: np.ndarray) -> "FeatureAdapter":
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]

        if self.n_features_in_ == self.n_qubits:
            self.mode_ = "identity"
        elif self.n_features_in_ > self.n_qubits:
            self.mode_ = "pca"
            self._pca = PCA(n_components=self.n_qubits, random_state=self.seed)
            self._pca.fit(X)
        else:
            self.mode_ = "tile"
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mode_ is None:
            raise RuntimeError("FeatureAdapter must be fit before transform")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        if self.mode_ == "identity":
            return X
        if self.mode_ == "pca":
            assert self._pca is not None
            return self._pca.transform(X)
        indices = [i % self.n_features_in_ for i in range(self.n_qubits)]
        return X[:, indices]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def bitstring_to_class(bitstring: str, strategy: str) -> int:
    """Map one measured bitstring to a binary class label.

    Qiskit returns bitstrings in little-endian order (qubit 0 is the
    *rightmost* character), so the ``single`` strategy reads the last
    character.
    """
    clean = bitstring.replace(" ", "")
    if strategy == "parity":
        return clean.count("1") % 2
    if strategy == "single":
        return int(clean[-1])
    raise ValueError(
        f"Unknown readout strategy '{strategy}'. Available: {READOUT_STRATEGIES}"
    )


def counts_to_class1_probability(counts: dict[str, int], strategy: str) -> float:
    """Estimate ``P(class 1)`` from shot counts under a readout strategy."""
    total = sum(counts.values())
    if total == 0:
        raise ValueError("counts dictionary is empty; cannot estimate a probability")
    class1 = sum(n for bits, n in counts.items() if bitstring_to_class(bits, strategy) == 1)
    return class1 / total


@dataclass
class VQCTrainingRecord:
    """Diagnostics captured during a single fit."""

    loss_history: list[float] = field(default_factory=list)
    n_iterations: int = 0
    initial_loss: float = float("nan")
    final_loss: float = float("nan")
    training_time_seconds: float = 0.0
    converged: bool = False
    optimizer_message: str = ""


class VariationalQuantumClassifier:
    """A shot-based, noise-aware binary VQC with an sklearn-style API."""

    def __init__(
        self,
        n_qubits: int,
        depth: int,
        *,
        noise_model: NoiseModel | None = None,
        shots: int = SHOTS,
        readout_strategy: str = VQC_CONFIG.readout_strategy,
        maxiter: int = VQC_CONFIG.maxiter,
        tol: float = VQC_CONFIG.tol,
        init_scale: float = VQC_CONFIG.init_scale,
        encoding_scale: float = VQC_CONFIG.encoding_scale,
        seed: int = 0,
    ) -> None:
        if readout_strategy not in READOUT_STRATEGIES:
            raise ValueError(
                f"Unknown readout strategy '{readout_strategy}'. "
                f"Available: {READOUT_STRATEGIES}"
            )
        if shots < 1:
            raise ValueError(f"shots must be >= 1, got {shots}")

        self.n_qubits = n_qubits
        self.depth = depth
        self.noise_model = noise_model
        self.shots = shots
        self.readout_strategy = readout_strategy
        self.maxiter = maxiter
        self.tol = tol
        self.init_scale = init_scale
        self.encoding_scale = encoding_scale
        self.seed = seed

        self.n_parameters = n_trainable_parameters(n_qubits, depth)
        self._circuit, self._x_params, self._w_params = build_vqc_circuit(
            n_qubits, depth, encoding_scale=encoding_scale
        )
        self._transpiled = transpile_circuit(self._circuit, seed_transpiler=seed)
        self._backend = get_backend(noise_model)

        self.adapter = FeatureAdapter(n_qubits, seed=seed)
        self.weights_: np.ndarray | None = None
        self.initial_weights_: np.ndarray | None = None
        self.training_record_: VQCTrainingRecord | None = None
        self.classes_: np.ndarray | None = None

    # -- circuit plumbing ---------------------------------------------------

    @property
    def transpiled_circuit(self) -> QuantumCircuit:
        return self._transpiled

    def resources(self):
        """Resource metrics for this model's circuit (section 10)."""
        return circuit_resources(
            self._circuit,
            logical_depth=self.depth,
            n_params=self.n_parameters,
            transpiled=self._transpiled,
        )

    def _bind(self, features: np.ndarray, weights: np.ndarray) -> list[QuantumCircuit]:
        weight_map = {self._w_params[j]: float(weights[j]) for j in range(self.n_parameters)}
        circuits = []
        for row in features:
            binding = dict(weight_map)
            for i in range(self.n_qubits):
                binding[self._x_params[i]] = float(row[i])
            circuits.append(self._transpiled.assign_parameters(binding))
        return circuits

    def _run_counts(
        self, features: np.ndarray, weights: np.ndarray, *, seed_simulator: int
    ) -> list[dict[str, int]]:
        circuits = self._bind(features, weights)
        result = self._backend.run(
            circuits, shots=self.shots, seed_simulator=seed_simulator
        ).result()
        return [result.get_counts(i) for i in range(len(circuits))]

    def _class1_probabilities(
        self, features: np.ndarray, weights: np.ndarray, *, seed_simulator: int
    ) -> np.ndarray:
        counts_list = self._run_counts(features, weights, seed_simulator=seed_simulator)
        return np.array(
            [counts_to_class1_probability(c, self.readout_strategy) for c in counts_list]
        )

    # -- training -----------------------------------------------------------

    def _loss(self, probabilities: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(probabilities, _PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VariationalQuantumClassifier":
        """Train the circuit weights by minimizing binary cross-entropy."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"VQC supports binary classification only; got classes {self.classes_}"
            )
        y_binary = (y == self.classes_[1]).astype(float)

        features = self.adapter.fit_transform(X)
        rng = np.random.default_rng(self.seed)
        initial = rng.uniform(-self.init_scale, self.init_scale, size=self.n_parameters)
        self.initial_weights_ = initial.copy()

        record = VQCTrainingRecord()
        # A fixed simulator seed per iteration keeps the stochastic objective
        # reproducible without freezing the shot noise to one single sample.
        iteration = {"n": 0}

        def objective(weights: np.ndarray) -> float:
            iteration["n"] += 1
            probabilities = self._class1_probabilities(
                features, weights, seed_simulator=self.seed * 10_000 + iteration["n"]
            )
            loss = self._loss(probabilities, y_binary)
            record.loss_history.append(loss)
            return loss

        start = time.perf_counter()
        result = minimize(
            objective,
            initial,
            method="COBYLA",
            tol=self.tol,
            options={"maxiter": self.maxiter},
        )
        record.training_time_seconds = time.perf_counter() - start

        self.weights_ = np.asarray(result.x, dtype=float)
        record.n_iterations = iteration["n"]
        record.initial_loss = record.loss_history[0] if record.loss_history else float("nan")
        record.final_loss = min(record.loss_history) if record.loss_history else float("nan")
        record.converged = bool(result.success)
        record.optimizer_message = str(result.message)
        self.training_record_ = record

        logger.info(
            "VQC(q=%d, d=%d, readout=%s, noise=%s) trained: "
            "loss %.4f -> %.4f over %d evaluations in %.2fs",
            self.n_qubits,
            self.depth,
            self.readout_strategy,
            "none" if self.noise_model is None else "model",
            record.initial_loss,
            record.final_loss,
            record.n_iterations,
            record.training_time_seconds,
        )
        return self

    # -- inference ----------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.weights_ is None or self.classes_ is None:
            raise RuntimeError("VQC must be fit before calling predict/predict_proba")

    def predict_proba(self, X: np.ndarray, *, seed_simulator: int | None = None) -> np.ndarray:
        """Return ``P(class 1)`` for each sample, estimated from shot counts."""
        self._check_fitted()
        assert self.weights_ is not None
        features = self.adapter.transform(np.asarray(X, dtype=float))
        seed = self.seed if seed_simulator is None else seed_simulator
        return self._class1_probabilities(features, self.weights_, seed_simulator=seed)

    def predict(self, X: np.ndarray, *, seed_simulator: int | None = None) -> np.ndarray:
        """Predict class labels (original label values, not 0/1 internally)."""
        self._check_fitted()
        assert self.classes_ is not None
        probabilities = self.predict_proba(X, seed_simulator=seed_simulator)
        return np.where(probabilities > 0.5, self.classes_[1], self.classes_[0])

    def measurement_counts(
        self, X: np.ndarray, *, seed_simulator: int | None = None
    ) -> list[dict[str, int]]:
        """Raw measurement counts per sample, for output-distribution analysis."""
        self._check_fitted()
        assert self.weights_ is not None
        features = self.adapter.transform(np.asarray(X, dtype=float))
        seed = self.seed if seed_simulator is None else seed_simulator
        return self._run_counts(features, self.weights_, seed_simulator=seed)

    def set_noise_model(self, noise_model: NoiseModel | None) -> "VariationalQuantumClassifier":
        """Swap the noise model, keeping trained weights.

        This is how a model trained under ideal conditions is evaluated
        under noise (and vice versa) on exactly the same circuit and
        weights.
        """
        self.noise_model = noise_model
        self._backend = get_backend(noise_model)
        return self
