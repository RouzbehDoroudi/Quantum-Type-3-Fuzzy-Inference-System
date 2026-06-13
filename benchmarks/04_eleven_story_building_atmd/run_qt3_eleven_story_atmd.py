"""QT3-FIS benchmark 4: eleven-story shear building with ATMD.

Only the Quantum Type-3 FIS controller is included. The script contains a
self-contained structural demo using the 11-story mass/stiffness arrays from the
supplied notebook and the QT3 5x5 -> 7 rule table from the article. For exact
paper reproduction, place the original earthquake record in ``data/elcentro.mat``
and, if desired, replace the default QT3/TMD settings with the optimized design
vector used in the working notebook.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qt3fis import generalized_power_mf, alpha_slice, probability_centroid, quantum_oracle_distribution

# 5 inputs: NL, NS, Z, PS, PL; 7 outputs: NL, NM, NS, Z, PS, PM, PL
INPUT_CENTERS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
OUTPUT_CENTERS = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
RULE_OUTPUT = np.array([
    [3, 5, 6, 6, 4],
    [4, 6, 6, 6, 3],
    [3, 2, 3, 4, 3],
    [3, 0, 0, 0, 2],
    [2, 0, 0, 1, 3],
], dtype=int)
RULE_WEIGHTS = np.ones((5, 5), dtype=float)
ALPHAS = np.linspace(0.1, 1.0, 10)
GAMMA = 2.0


def load_ground_motion(path: Optional[str], dt: float, t_final: float) -> tuple[np.ndarray, np.ndarray]:
    """Load an El-Centro-style record or generate a synthetic demo record."""
    if path and Path(path).exists():
        d = loadmat(path)
        if "e" in d:
            E = np.asarray(d["e"], dtype=float)
            if E.shape[0] == 2:
                t, a = E[0].ravel(), E[1].ravel()
            else:
                t, a = E[:, 0].ravel(), E[:, 1].ravel()
        elif "t" in d and "a" in d:
            t, a = np.asarray(d["t"]).ravel(), np.asarray(d["a"]).ravel()
        else:
            raise ValueError("MAT file must contain key 'e' or keys 't' and 'a'.")
        tu = np.arange(float(t[0]), min(float(t[-1]), t_final) + dt, dt)
        au = np.interp(tu, t, a)
        return tu, au
    t = np.arange(0.0, t_final + dt, dt)
    # Synthetic earthquake-like demo in m/s^2. Replace with El Centro for paper reproduction.
    env = np.exp(-0.18 * t) * (1.0 - np.exp(-1.2 * t))
    a = 2.2 * env * (np.sin(2*np.pi*1.15*t) + 0.45*np.sin(2*np.pi*2.7*t + 0.4))
    return t, a


def build_structure_11() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass = np.array([2.15e5, 2.01e5, 2.01e5, 2.00e5, 2.01e5, 2.01e5, 2.01e5, 2.03e5, 2.03e5, 2.03e5, 1.76e5], dtype=float)
    stiff = np.array([4.68e8, 4.76e8, 4.68e8, 4.50e8, 4.50e8, 4.50e8, 4.50e8, 4.37e8, 4.37e8, 4.37e8, 3.12e8], dtype=float)
    n = len(mass)
    M = np.diag(mass)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i == 0:
            K[i, i] = stiff[i] + stiff[i+1]; K[i, i+1] = -stiff[i+1]
        elif i == n - 1:
            K[i, i-1] = -stiff[i]; K[i, i] = stiff[i]
        else:
            K[i, i-1] = -stiff[i]; K[i, i] = stiff[i] + stiff[i+1]; K[i, i+1] = -stiff[i+1]
    lam, _ = eigh(K, M)
    omega = np.sqrt(np.clip(lam, 0.0, None))
    w1, w2 = float(omega[0]), float(omega[1])
    A = np.array([[1.0/w1, w1], [1.0/w2, w2]])
    b = np.array([0.10, 0.10])  # 2*zeta, zeta=5%
    a0, a1 = np.linalg.solve(A, b)
    C = a0 * M + a1 * K
    return M, C, K


def build_tmd(M: np.ndarray, K: np.ndarray, mu: float = 0.03, fr: float = 1.0, zeta: float = 0.07) -> dict:
    lam, _ = eigh(K, M)
    w1 = float(np.sqrt(max(lam[0], 0.0)))
    mtot = float(np.sum(np.diag(M)))
    md = mu * mtot
    wd = fr * w1
    kd = md * wd * wd
    cd = 2.0 * zeta * md * wd
    return {"mu": mu, "fr": fr, "zeta": zeta, "m_d": md, "k_d": kd, "c_d": cd, "w1": w1}


def augment_with_tmd(M: np.ndarray, C: np.ndarray, K: np.ndarray, tmd: dict, floor_index: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = M.shape[0]
    M2 = np.zeros((n+1, n+1)); C2 = np.zeros_like(M2); K2 = np.zeros_like(M2)
    M2[:n, :n] = M; C2[:n, :n] = C; K2[:n, :n] = K
    M2[n, n] = tmd["m_d"]
    kd, cd = tmd["k_d"], tmd["c_d"]
    j = int(floor_index)
    K2[j, j] += kd; K2[j, n] -= kd; K2[n, j] -= kd; K2[n, n] += kd
    C2[j, j] += cd; C2[j, n] -= cd; C2[n, j] -= cd; C2[n, n] += cd
    return M2, C2, K2


def input_memberships(xn: float) -> tuple[np.ndarray, np.ndarray]:
    mu_u = np.zeros(5); mu_l = np.zeros(5)
    for i, c in enumerate(INPUT_CENTERS):
        u, l = generalized_power_mf(np.array([xn]), center=float(c), left_width=0.55, right_width=0.55, upper_power=2.0, lower_power=3.0)
        mu_u[i] = float(u[0]); mu_l[i] = float(l[0])
    return mu_u, mu_l


def qt3_force_norm(x_norm: float, v_norm: float, *, shots: int, use_qiskit: bool, rng: np.random.Generator) -> float:
    xu, xl = input_memberships(x_norm)
    vu, vl = input_memberships(v_norm)
    num = 0.0; den = 0.0
    for a in ALPHAS:
        wa = float(a ** GAMMA)
        xU = alpha_slice(xu, a); xL = alpha_slice(xl, a)
        vU = alpha_slice(vu, a); vL = alpha_slice(vl, a)
        WU = np.outer(xU, vU) * RULE_WEIGHTS
        WL = np.outer(xL, vL) * RULE_WEIGHTS
        pU = quantum_oracle_distribution(WU, RULE_OUTPUT, len(OUTPUT_CENTERS), shots=shots, use_qiskit=use_qiskit, rng=rng)
        pL = quantum_oracle_distribution(WL, RULE_OUTPUT, len(OUTPUT_CENTERS), shots=shots, use_qiskit=use_qiskit, rng=rng)
        y = 0.5 * (probability_centroid(pU, OUTPUT_CENTERS) + probability_centroid(pL, OUTPUT_CENTERS))
        num += wa * y; den += wa
    return float(np.clip((num / max(den, 1e-15)) / 1.5, -1.0, 1.0))


def run_uncontrolled(t_eq: np.ndarray, a_eq: np.ndarray, M: np.ndarray, C: np.ndarray, K: np.ndarray) -> dict:
    n = M.shape[0]
    choM = cho_factor(M, check_finite=False)
    ones = np.ones(n)
    def ag(t): return float(np.interp(t, t_eq, a_eq))
    def rhs(t, z):
        q, qd = z[:n], z[n:]
        qdd = cho_solve(choM, -C @ qd - K @ q - (M @ ones) * ag(t), check_finite=False)
        return np.r_[qd, qdd]
    sol = solve_ivp(rhs, (t_eq[0], t_eq[-1]), np.zeros(2*n), t_eval=t_eq, rtol=1e-6, atol=1e-8, max_step=np.median(np.diff(t_eq)))
    q, qd = sol.y[:n].T, sol.y[n:].T
    return {"t": sol.t, "x_all": q, "v_all": qd, "peak_top": float(np.max(np.abs(q[:, -1])))}


def run_qt3_atmd(t_eq: np.ndarray, a_eq: np.ndarray, args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    M, C, K = build_structure_11()
    base = run_uncontrolled(t_eq, a_eq, M, C, K)
    tmd = build_tmd(M, K, mu=args.tmd_mu, fr=args.tmd_fr, zeta=args.tmd_zeta)
    M2, C2, K2 = augment_with_tmd(M, C, K, tmd, floor_index=10)
    n = 11
    choM = cho_factor(M2, check_finite=False)
    ones = np.ones(n + 1)
    D = np.zeros(n + 1); D[10] = 1.0; D[n] = -1.0
    u_max = float(args.force_ratio * np.sum(np.diag(M)) * 9.81)
    x_peak = max(1e-9, float(np.max(np.abs(base["x_all"][:, 10]))))
    v_peak = max(1e-9, float(np.max(np.abs(base["v_all"][:, 10]))))
    control_dt = float(args.control_dt)
    last_u = 0.0; next_t = -1.0
    u_log_t, u_log = [], []
    def ag(t): return float(np.interp(t, t_eq, a_eq))
    def rhs(t, z):
        nonlocal last_u, next_t
        q, qd = z[:n+1], z[n+1:]
        if t >= next_t:
            xnorm = float(np.clip(q[10] / x_peak, -1.0, 1.0))
            vnorm = float(np.clip(qd[10] / v_peak, -1.0, 1.0))
            last_u = u_max * qt3_force_norm(xnorm, vnorm, shots=args.shots, use_qiskit=args.use_qiskit, rng=rng)
            next_t = t + control_dt
        rhs_vec = -C2 @ qd - K2 @ q - (M2 @ ones) * ag(t) + D * last_u
        qdd = cho_solve(choM, rhs_vec, check_finite=False)
        u_log_t.append(t); u_log.append(last_u)
        return np.r_[qd, qdd]
    sol = solve_ivp(rhs, (t_eq[0], t_eq[-1]), np.zeros(2*(n+1)), t_eval=t_eq, rtol=1e-6, atol=1e-8, max_step=np.median(np.diff(t_eq)))
    q, qd = sol.y[:n+1].T, sol.y[n+1:].T
    u = np.interp(sol.t, np.asarray(u_log_t), np.asarray(u_log)) if u_log_t else np.zeros_like(sol.t)
    out = {
        "t": sol.t, "x11": q[:, 10], "v11": qd[:, 10], "u": u,
        "peak_disp": float(np.max(np.abs(q[:, 10]))),
        "rms_disp": float(np.sqrt(np.mean(q[:, 10]**2))),
        "u_max": u_max, "tmd": tmd,
        "uncontrolled_peak_disp": base["peak_top"],
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--earthquake", type=str, default=str(ROOT / "data" / "elcentro.mat"))
    p.add_argument("--t-final", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--control-dt", type=float, default=0.02)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--use-qiskit", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tmd-mu", type=float, default=0.03)
    p.add_argument("--tmd-fr", type=float, default=1.0)
    p.add_argument("--tmd-zeta", type=float, default=0.07)
    p.add_argument("--force-ratio", type=float, default=0.05)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    t_eq, a_eq = load_ground_motion(args.earthquake if Path(args.earthquake).exists() else None, args.dt, args.t_final)
    out = run_qt3_atmd(t_eq, a_eq, args)
    metrics = {"peak_disp": out["peak_disp"], "rms_disp": out["rms_disp"], "uncontrolled_peak_disp": out["uncontrolled_peak_disp"]}
    metrics["peak_disp_reduction_percent"] = 100.0 * (1.0 - out["peak_disp"] / max(out["uncontrolled_peak_disp"], 1e-15))
    print("QT3-FIS — Eleven-story ATMD")
    print(json.dumps(metrics, indent=2))
    os.makedirs(ROOT / "results", exist_ok=True)
    np.savez_compressed(ROOT / "results" / "qt3_eleven_story_atmd.npz", t=out["t"], x11=out["x11"], v11=out["v11"], u=out["u"], metrics=json.dumps(metrics))
    if args.plot:
        plt.figure(); plt.plot(out["t"], out["x11"]); plt.xlabel("Time [s]"); plt.ylabel("Top displacement [m]"); plt.title("QT3-FIS 11-story ATMD"); plt.grid(True)
        plt.figure(); plt.plot(out["t"], out["u"]); plt.xlabel("Time [s]"); plt.ylabel("Control force [N]"); plt.grid(True); plt.show()


if __name__ == "__main__":
    main()
