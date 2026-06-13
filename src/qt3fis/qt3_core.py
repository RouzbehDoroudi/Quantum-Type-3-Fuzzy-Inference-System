"""Core utilities for Quantum Type-3 Fuzzy Inference Systems (QT3-FIS).

The functions in this module implement the public-release building blocks used by
all four benchmark scripts:

1. Type-3 upper/lower membership evaluation,
2. alpha-level slicing,
3. quantum-oracle output distribution estimation,
4. probability-weighted singleton aggregation.

The Qiskit path is optional because repeated circuit execution can be slow on a
standard laptop. If ``use_qiskit=False`` the same normalized oracle distribution
is evaluated analytically and, optionally, sampled with a multinomial RNG to
mimic finite-shot measurement. If ``use_qiskit=True`` and Qiskit Aer is
available, the rule-to-output mapping is realized through MCX gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple
import math
import numpy as np


def generalized_power_mf(
    x: np.ndarray | float,
    center: float,
    left_width: float,
    right_width: float,
    upper_power: float = 2.0,
    lower_power: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a power-based Type-3 UMF/LMF pair.

    Parameters are intentionally simple and match the cleaned public scripts:
    ``center``, ``left_width``, ``right_width``, ``upper_power`` and
    ``lower_power``. For this power form, the larger power produces the
    wider/higher curve on the normalized interval, so the larger of the two
    powers is used for the UMF and the smaller for the LMF. The output is
    clipped to ``[0, 1]`` and constrained so that ``LMF <= UMF``.
    """
    x = np.asarray(x, dtype=float)
    dx = x - float(center)
    lw = max(float(left_width), 1e-12)
    rw = max(float(right_width), 1e-12)
    base = np.empty_like(dx, dtype=float)
    mask = dx <= 0.0
    base[mask] = np.abs(dx[mask]) / lw
    base[~mask] = np.abs(dx[~mask]) / rw
    base = np.minimum(base, 1.0)
    # For 1 - base**p with base in [0, 1], a larger p gives a wider/higher curve.
    # Use ordered powers to keep a non-degenerate FOU and guarantee LMF <= UMF.
    p_upper = max(float(upper_power), float(lower_power))
    p_lower = min(float(upper_power), float(lower_power))
    umf = np.maximum(1.0 - base ** p_upper, 0.0)
    lmf = np.maximum(1.0 - base ** p_lower, 0.0)
    lmf = np.minimum(lmf, umf)
    return umf, lmf


def alpha_slice(mu: np.ndarray, alpha: float, fallback: bool = True) -> np.ndarray:
    """Return an alpha-gated membership vector and normalize it.

    If all values are removed by the alpha cut, the ungated vector is used as a
    safe fallback when ``fallback=True``.
    """
    mu = np.asarray(mu, dtype=float)
    out = np.where(mu >= float(alpha), mu, 0.0)
    if fallback and np.all(out <= 0.0):
        out = np.maximum(mu, 0.0)
    s = float(np.sum(out))
    if s <= 1e-15:
        return np.ones_like(out) / max(1, out.size)
    return out / s


def probability_centroid(prob: np.ndarray, centers: Sequence[float]) -> float:
    """Map an output-label probability vector to a crisp singleton output."""
    prob = np.asarray(prob, dtype=float)
    centers = np.asarray(centers, dtype=float)
    prob = np.maximum(prob, 0.0)
    prob = prob / (float(np.sum(prob)) + 1e-15)
    return float(prob @ centers)


def weighted_metrics(y: np.ndarray, dt: float, ref: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Common tracking metrics used in the benchmark scripts."""
    y = np.asarray(y, dtype=float)
    if ref is None:
        ref = np.zeros_like(y)
    else:
        ref = np.asarray(ref, dtype=float)
    e = y - ref
    t = np.arange(e.size) * float(dt)
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "IAE": float(np.sum(np.abs(e)) * dt),
        "ITAE": float(np.sum(t * np.abs(e)) * dt),
        "ISE": float(np.sum(e**2) * dt),
    }


def _analytic_distribution(
    weights: np.ndarray,
    output_indices: np.ndarray,
    n_outputs: int,
    shots: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).ravel()
    output_indices = np.asarray(output_indices, dtype=int).ravel()
    weights = np.maximum(weights, 0.0)
    dist = np.zeros(int(n_outputs), dtype=float)
    for w, k in zip(weights, output_indices):
        if 0 <= int(k) < int(n_outputs):
            dist[int(k)] += float(w)
    s = float(np.sum(dist))
    if s <= 1e-15:
        dist[:] = 1.0 / int(n_outputs)
    else:
        dist /= s
    if shots and shots > 0:
        if rng is None:
            rng = np.random.default_rng(42)
        dist = rng.multinomial(int(shots), dist) / float(shots)
    return dist


def quantum_oracle_distribution(
    weights: np.ndarray,
    output_indices: np.ndarray,
    n_outputs: int,
    shots: int = 1024,
    use_qiskit: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Estimate output-label probabilities from a rule oracle.

    Parameters
    ----------
    weights:
        Non-negative rule activation matrix/vector. Its flattened indices are the
        input basis states.
    output_indices:
        Same shape as ``weights``. Each entry gives the consequent label index.
    n_outputs:
        Number of output labels.
    shots:
        Finite-shot measurement budget. In analytic mode, ``shots`` still causes
        multinomial sampling. Use ``shots=0`` for exact probabilities.
    use_qiskit:
        If true, build a compact MCX oracle and sample it on Qiskit Aer. If
        Qiskit is not installed, the function falls back to analytic sampling.
    rng:
        Optional RNG for analytic finite-shot sampling.
    """
    weights = np.asarray(weights, dtype=float)
    output_indices = np.asarray(output_indices, dtype=int)
    n_inputs = weights.size
    n_outputs = int(n_outputs)

    if not use_qiskit:
        return _analytic_distribution(weights, output_indices, n_outputs, shots=shots, rng=rng)

    try:
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
        from qiskit_aer import Aer
    except Exception:
        return _analytic_distribution(weights, output_indices, n_outputs, shots=shots, rng=rng)

    flat_w = np.maximum(weights.ravel(), 0.0)
    flat_k = output_indices.ravel().astype(int)
    if float(np.sum(flat_w)) <= 1e-15:
        return np.ones(n_outputs, dtype=float) / n_outputs

    n_in_qubits = max(1, int(math.ceil(math.log2(max(1, n_inputs)))))
    n_out_qubits = max(1, int(math.ceil(math.log2(max(1, n_outputs)))))
    state_len = 2 ** n_in_qubits
    amps = np.zeros(state_len, dtype=complex)
    amps[:n_inputs] = np.sqrt(flat_w / (float(np.sum(flat_w)) + 1e-15))
    amps /= np.linalg.norm(amps) + 1e-15

    qin = QuantumRegister(n_in_qubits, "qin")
    qout = QuantumRegister(n_out_qubits, "qout")
    cout = ClassicalRegister(n_out_qubits, "cout")
    qc = QuantumCircuit(qin, qout, cout)
    qc.initialize(amps, qin)

    controls = [qin[i] for i in range(n_in_qubits)]
    for state_index in range(n_inputs):
        if flat_w[state_index] <= 0.0:
            continue
        k = int(flat_k[state_index])
        if k < 0 or k >= n_outputs:
            continue
        flipped = []
        for bit in range(n_in_qubits):
            if ((state_index >> bit) & 1) == 0:
                qc.x(qin[bit])
                flipped.append(bit)
        for obit in range(n_out_qubits):
            if ((k >> obit) & 1) == 1:
                if n_in_qubits == 1:
                    qc.cx(qin[0], qout[obit])
                else:
                    qc.mcx(controls, qout[obit])
        for bit in reversed(flipped):
            qc.x(qin[bit])

    qc.measure(qout, cout)
    backend = Aer.get_backend("qasm_simulator")
    tqc = transpile(qc, backend, optimization_level=1)
    result = backend.run(tqc, shots=int(shots)).result()
    counts = result.get_counts()

    dist = np.zeros(n_outputs, dtype=float)
    total = max(1, sum(counts.values()))
    for bitstr, ct in counts.items():
        bits = bitstr.replace(" ", "")
        measured = int(bits[::-1], 2)  # qiskit prints classical bits MSB first
        if measured < n_outputs:
            dist[measured] += ct
    dist /= float(total)
    if float(np.sum(dist)) <= 1e-15:
        return np.ones(n_outputs, dtype=float) / n_outputs
    return dist / float(np.sum(dist))
