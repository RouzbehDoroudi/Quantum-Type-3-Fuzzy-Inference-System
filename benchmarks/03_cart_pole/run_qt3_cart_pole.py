"""QT3-FIS benchmark 3: Cart-Pole balancing.

Only the Quantum Type-3 FIS controller is included. The script uses the
optimized 5x5 rule table and rule-weight matrix from the supplied CartPole
notebook, with Type-3 alpha slicing and oracle-based consequent aggregation.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qt3fis import alpha_slice, probability_centroid, quantum_oracle_distribution, weighted_metrics

THETA_CENTERS = np.linspace(-1.0, 1.0, 5)
OMEGA_CENTERS = np.linspace(-1.0, 1.0, 5)
FC_CENTERS = np.linspace(-15.0, 15.0, 7)

# Optimized rule table and weights from the supplied CartPole notebook.
RULE_LABELS = np.array([
    [5, 1, 0, 5, 2],
    [0, 0, 2, 4, 3],
    [5, 4, 1, 1, 2],
    [1, 3, 3, 3, 5],
    [3, 4, 0, 4, 1],
], dtype=int)

RULE_WEIGHTS = np.clip(np.array([
    [0.5751, 0.4569, 0.1954, 0.1749, 0.3862],
    [0.3082, 0.8095, 0.3691, 0.3297, 0.6070],
    [0.3342, 0.4798, 0.1705, 0.1330, 0.2456],
    [0.3598, 0.7036, 0.6736, 0.3487, 0.8535],
    [0.5362, 0.2964, 0.4406, 0.6537, 0.4153],
], dtype=float), 0.0, 1.0)

THETA_MF = dict(sigma_u=0.701220456240653, sigma_l=1.1701806059620243,
                p_u=2.320656280922303, p_l=0.5850175003823717,
                h_l=0.7534225754950167, k_l=1.3717349205118827)
OMEGA_MF = dict(sigma_u=3.786961967104939, sigma_l=1.7121925651643966,
                p_u=3.1672253977225653, p_l=0.5353306838625767,
                h_l=0.5088510915680412, k_l=1.1037228048888712)

ALPHAS = np.linspace(0.1, 1.0, 10)
GAMMA = 2.0


def _gmf_scalar(x: float, center: float, left_w: float, right_w: float, power: float) -> float:
    dx = x - center
    base = abs(dx) / (left_w if dx <= 0.0 else right_w)
    base = min(base, 1.0)
    return float(max(1.0 - base ** power, 0.0))


def _mf_params(mfdict: dict) -> tuple[float, float, float, float, float, float]:
    ul = ur = max(float(mfdict["sigma_u"]), 1e-8)
    up = max(float(mfdict["p_u"]), 1e-8)
    ll = max(float(mfdict["sigma_l"]), 1e-8)
    lr = max(float(mfdict["sigma_l"]) * float(mfdict["k_l"]), 1e-8)
    lp = max(float(mfdict["p_l"]), 1e-8)
    return ul, ur, up, ll, lr, lp


def input_memberships(theta: float, omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    th_n = float(np.clip(theta / 0.3, -1.0, 1.0))
    om_n = float(np.clip(omega / 2.0, -1.0, 1.0))
    th_u = np.zeros(5); th_l = np.zeros(5)
    om_u = np.zeros(5); om_l = np.zeros(5)
    tul, tur, tup, tll, tlr, tlp = _mf_params(THETA_MF)
    oul, our, oup, oll, olr, olp = _mf_params(OMEGA_MF)
    for i, c in enumerate(THETA_CENTERS):
        th_u[i] = _gmf_scalar(th_n, c, tul, tur, tup)
        th_l[i] = _gmf_scalar(th_n, c, tll, tlr, tlp)
    for i, c in enumerate(OMEGA_CENTERS):
        om_u[i] = _gmf_scalar(om_n, c, oul, our, oup)
        om_l[i] = _gmf_scalar(om_n, c, oll, olr, olp)
    return th_u, th_l, om_u, om_l


def qt3_controller(theta: float, omega: float, *, shots: int, use_qiskit: bool, rng: np.random.Generator) -> float:
    th_u, th_l, om_u, om_l = input_memberships(theta, omega)
    num = 0.0; den = 0.0
    for a in ALPHAS:
        wa = float(a ** GAMMA)
        thU = alpha_slice(th_u, a); thL = alpha_slice(th_l, a)
        omU = alpha_slice(om_u, a); omL = alpha_slice(om_l, a)
        WU = np.outer(thU, omU) * RULE_WEIGHTS
        WL = np.outer(thL, omL) * RULE_WEIGHTS
        pU = quantum_oracle_distribution(WU, RULE_LABELS, len(FC_CENTERS), shots=shots, use_qiskit=use_qiskit, rng=rng)
        pL = quantum_oracle_distribution(WL, RULE_LABELS, len(FC_CENTERS), shots=shots, use_qiskit=use_qiskit, rng=rng)
        y = 0.5 * (probability_centroid(pU, FC_CENTERS) + probability_centroid(pL, FC_CENTERS))
        num += wa * y; den += wa
    return float(np.clip(num / max(den, 1e-15), -25.0, 25.0))


def cartpole_rhs(state: np.ndarray, force: float) -> np.ndarray:
    g, length, mp, mc = 9.81, 0.5, 0.1, 1.0
    x, xdot, theta, thetadot = state
    s, c = np.sin(theta), np.cos(theta)
    denom = mc + mp * s * s
    theta_dd = (g * s - c * (force + mp * length * thetadot * thetadot * s) / denom) / (length * (4.0 / 3.0 - mp * c * c / denom))
    x_dd = (force + mp * length * (thetadot * thetadot * s - theta_dd * c)) / denom
    return np.array([xdot, x_dd, thetadot, theta_dd], dtype=float)


def simulate(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    n = int(args.t_final / args.dt) + 1
    t = np.linspace(0.0, args.t_final, n)
    y = np.zeros((n, 4), dtype=float)
    u = np.zeros(n, dtype=float)
    y[0] = [0.0, 0.0, args.theta0, args.omega0]
    for k in range(n - 1):
        fc = qt3_controller(y[k, 2], y[k, 3], shots=args.shots, use_qiskit=args.use_qiskit, rng=rng)
        u[k] = float(np.clip(fc, -25.0, 25.0))
        h = args.dt
        k1 = cartpole_rhs(y[k], u[k])
        k2 = cartpole_rhs(y[k] + 0.5 * h * k1, u[k])
        k3 = cartpole_rhs(y[k] + 0.5 * h * k2, u[k])
        k4 = cartpole_rhs(y[k] + h * k3, u[k])
        y[k + 1] = y[k] + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    u[-1] = u[-2]
    theta = y[:, 2]
    omega = y[:, 3]
    metrics = weighted_metrics(theta, args.dt)
    metrics["control_effort"] = float(np.sum(u**2) * args.dt)
    return {"t": t, "theta": theta, "omega": omega, "x": y[:,0], "xdot": y[:,1], "u": u, "metrics": metrics}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--t-final", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--theta0", type=float, default=0.05)
    p.add_argument("--omega0", type=float, default=0.0)
    p.add_argument("--shots", type=int, default=512)
    p.add_argument("--use-qiskit", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    out = simulate(args)
    print("QT3-FIS — Cart-Pole")
    print(json.dumps(out["metrics"], indent=2))
    os.makedirs(ROOT / "results", exist_ok=True)
    np.savez_compressed(ROOT / "results" / "qt3_cart_pole.npz", **{k: v for k, v in out.items() if k != "metrics"}, metrics=json.dumps(out["metrics"]))
    if args.plot:
        plt.figure(); plt.plot(out["t"], out["theta"]); plt.xlabel("Time [s]"); plt.ylabel("theta [rad]"); plt.title("QT3-FIS Cart-Pole"); plt.grid(True)
        plt.figure(); plt.plot(out["t"], out["u"]); plt.xlabel("Time [s]"); plt.ylabel("Force [N]"); plt.grid(True); plt.show()


if __name__ == "__main__":
    main()
