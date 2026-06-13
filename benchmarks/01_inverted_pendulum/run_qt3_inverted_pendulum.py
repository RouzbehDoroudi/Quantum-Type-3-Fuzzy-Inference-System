"""QT3-FIS benchmark 1: inverted pendulum stabilization.

Only the proposed Quantum Type-3 FIS controller is included. Lower-order fuzzy
baselines and classical Type-3 comparison code are intentionally excluded from
this public release.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qt3fis import generalized_power_mf, alpha_slice, probability_centroid, quantum_oracle_distribution, weighted_metrics

# Plant and simulation parameters consistent with the article formulation, Eq. (39)
DT = 0.01
T_FINAL = 10.0
G = 9.81
M = 1.0
L = 1.0
THETA_MAX = np.pi / 4.0
OMEGA_MAX = np.pi
ALPHAS = np.linspace(0.1, 1.0, 10)
GAMMA = 2.0

# Labels: input N,Z,P; output NM,NS,Z,PS,PM
INPUT_CENTERS = np.array([-1.0, 0.0, 1.0])
OUTPUT_CENTERS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

# Rule table from the inverted-pendulum example:
# rows theta=[N,Z,P], columns omega=[N,Z,P], values output=[NM,NS,Z,PS,PM] -> indices 0..4
RULE_OUTPUT = np.array([
    [4, 3, 2],
    [3, 2, 1],
    [2, 1, 0],
], dtype=int)


def input_memberships(x_norm: float) -> tuple[np.ndarray, np.ndarray]:
    """UMF/LMF for N, Z, P on normalized axis [-1, 1]."""
    params = [
        (-1.0, 0.9, 0.45, 2.0, 3.0),
        ( 0.0, 0.6, 0.60, 2.0, 3.0),
        ( 1.0, 0.45, 0.9, 2.0, 3.0),
    ]
    mu_u, mu_l = [], []
    for p in params:
        u, l = generalized_power_mf(np.array([x_norm]), *p)
        mu_u.append(float(u[0])); mu_l.append(float(l[0]))
    return np.array(mu_u), np.array(mu_l)


def qt3_controller(theta: float, omega: float, *, shots: int, use_qiskit: bool, rng: np.random.Generator) -> float:
    """One QT3-FIS control step."""
    th_n = float(np.clip(theta / THETA_MAX, -1.0, 1.0))
    om_n = float(np.clip(omega / OMEGA_MAX, -1.0, 1.0))
    th_u, th_l = input_memberships(th_n)
    om_u, om_l = input_memberships(om_n)

    acc = 0.0
    den = 0.0
    for a in ALPHAS:
        w_alpha = float(a ** GAMMA)
        thU = alpha_slice(th_u, a); thL = alpha_slice(th_l, a)
        omU = alpha_slice(om_u, a); omL = alpha_slice(om_l, a)
        WU = np.outer(thU, omU)
        WL = np.outer(thL, omL)
        pU = quantum_oracle_distribution(WU, RULE_OUTPUT, len(OUTPUT_CENTERS), shots=shots, use_qiskit=use_qiskit, rng=rng)
        pL = quantum_oracle_distribution(WL, RULE_OUTPUT, len(OUTPUT_CENTERS), shots=shots, use_qiskit=use_qiskit, rng=rng)
        y_alpha = 0.5 * (probability_centroid(pU, OUTPUT_CENTERS) + probability_centroid(pL, OUTPUT_CENTERS))
        acc += w_alpha * y_alpha
        den += w_alpha
    return float(np.clip(acc / max(den, 1e-15), -1.0, 1.0))


def simulate(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    n = int(args.t_final / args.dt) + 1
    t = np.linspace(0.0, args.t_final, n)
    theta = np.zeros(n); omega = np.zeros(n); u = np.zeros(n)
    theta[0] = np.deg2rad(args.theta0_deg)
    for k in range(n - 1):
        u[k] = qt3_controller(theta[k], omega[k], shots=args.shots, use_qiskit=args.use_qiskit, rng=rng)
        # Article Eq. (39): theta_dot = omega, omega_dot = (g/L) sin(theta) + Fc/(m L^2).
        # The QT3-FIS output is the normalized bounded control action Fc in this public demo.
        acc = (G / L) * np.sin(theta[k]) + u[k] / (M * L * L)
        theta[k + 1] = theta[k] + omega[k] * args.dt + 0.5 * args.dt**2 * acc
        omega[k + 1] = omega[k] + args.dt * acc
    u[-1] = u[-2]
    metrics = weighted_metrics(theta, args.dt)
    return {"t": t, "theta": theta, "omega": omega, "u": u, "metrics": metrics}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--t-final", type=float, default=T_FINAL)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--theta0-deg", type=float, default=5.0)
    p.add_argument("--shots", type=int, default=512, help="Use 0 for exact analytic probabilities; positive value for finite-shot sampling.")
    p.add_argument("--use-qiskit", action="store_true", help="Run MCX oracle circuits on Qiskit Aer. Slower but closer to gate-level demonstration.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    out = simulate(args)
    print("QT3-FIS — Inverted Pendulum")
    print(json.dumps(out["metrics"], indent=2))
    os.makedirs(ROOT / "results", exist_ok=True)
    np.savez_compressed(ROOT / "results" / "qt3_inverted_pendulum.npz", **{k: v for k, v in out.items() if k != "metrics"}, metrics=json.dumps(out["metrics"]))
    if args.plot:
        plt.figure(); plt.plot(out["t"], out["theta"]); plt.xlabel("Time [s]"); plt.ylabel("theta [rad]"); plt.title("QT3-FIS inverted pendulum"); plt.grid(True)
        plt.figure(); plt.plot(out["t"], out["u"]); plt.xlabel("Time [s]"); plt.ylabel("normalized control"); plt.grid(True); plt.show()


if __name__ == "__main__":
    main()
