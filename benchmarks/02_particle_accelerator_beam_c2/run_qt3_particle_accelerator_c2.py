"""QT3-FIS benchmark 2: particle accelerator beam control, C2 case.

Only the Quantum Type-3 controller is included. The controller follows the C2
negative-feedback rule base N->P, Z->Z, P->N and uses Type-3 alpha/beta fusion
before quantum measurement of the output-label probabilities.
"""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qt3fis import quantum_oracle_distribution, probability_centroid, weighted_metrics

X_MM_MIN, X_MM_MAX = -5.0, 5.0
Y_URAD_MIN, Y_URAD_MAX = -140.0, 140.0
LABELS = ["N", "Z", "P"]
OUT_CENTERS_N = np.array([-1.0, 0.0, 1.0])
# Input state N,Z,P maps to output P,Z,N -> output indices 2,1,0.
RULE_OUTPUT = np.array([2, 1, 0], dtype=int)


def to_norm(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    return np.clip(2.0 * (x - xmin) / max(xmax - xmin, 1e-12) - 1.0, -1.0, 1.0)


def from_norm(z: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    return (z + 1.0) * 0.5 * (xmax - xmin) + xmin


@dataclass
class TriMF:
    a: float
    b: float
    c: float


def trimf_degree(x: np.ndarray, mf: TriMF) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    left = (mf.a < x) & (x <= mf.b)
    right = (mf.b < x) & (x < mf.c)
    y[left] = (x[left] - mf.a) / (mf.b - mf.a + 1e-12)
    y[right] = (mf.c - x[right]) / (mf.c - mf.b + 1e-12)
    y[x == mf.b] = 1.0
    return np.clip(y, 0.0, 1.0)


def base_partition() -> Dict[str, TriMF]:
    return {"N": TriMF(-1.0, -1.0, 0.0), "Z": TriMF(-1.0, 0.0, 1.0), "P": TriMF(0.0, 1.0, 1.0)}


def widen_mf(m: TriMF, d: float) -> TriMF:
    return TriMF(max(-1.0, m.a - d), m.b, min(1.0, m.c + d))


def shrink_mf(m: TriMF, d: float) -> TriMF:
    return TriMF(min(m.b - 1e-6, max(-1.0, m.a + d)), m.b, max(m.b + 1e-6, min(1.0, m.c - d)))


BASE_MFS = base_partition()
UMF_MFS = {k: widen_mf(v, 0.20) for k, v in BASE_MFS.items()}
LMF_MFS = {k: shrink_mf(v, 0.20) for k, v in BASE_MFS.items()}


def _slice_primary(umf: np.ndarray, lmf: np.ndarray, alpha: float) -> np.ndarray:
    u = np.maximum(umf - (1.0 - alpha), 0.0)
    l = np.maximum(lmf - (1.0 - alpha), 0.0)
    m = np.maximum(u, l)
    s = float(np.sum(m))
    if s <= 1e-15:
        m = 0.5 * (umf + lmf)
        s = float(np.sum(m))
    return m / max(s, 1e-15)


def _secondary_band(m: np.ndarray, spread: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    delta = spread * (2.0 * beta - 1.0)
    upper = np.clip(m * (1.0 + max(0.0, delta)), 0.0, 1.0)
    lower = np.clip(m * (1.0 + min(0.0, delta)), 0.0, 1.0)
    return lower, upper


def qt3_eval_single(xn: float, *, n_alpha: int, n_beta: int, beta_spread: float, gamma_alpha: float, gamma_beta: float, shots: int, use_qiskit: bool, rng: np.random.Generator) -> float:
    xarr = np.array([xn], dtype=float)
    mu_u = np.array([trimf_degree(xarr, UMF_MFS[L])[0] for L in LABELS])
    mu_l = np.array([trimf_degree(xarr, LMF_MFS[L])[0] for L in LABELS])
    alphas = np.linspace(0.0, 1.0, n_alpha)
    betas = np.linspace(0.0, 1.0, n_beta)
    wA = alphas ** gamma_alpha; wA[0] = 0.0
    wB = betas ** gamma_beta
    P = np.zeros(3, dtype=float)
    for a, wa in zip(alphas, wA):
        if wa <= 0.0:
            continue
        mA = _slice_primary(mu_u, mu_l, float(a))
        P_beta = np.zeros(3, dtype=float)
        for b, wb in zip(betas, wB):
            lb, ub = _secondary_band(mA, beta_spread, float(b))
            weights = 0.5 * (lb + ub)
            p_ab = quantum_oracle_distribution(weights, RULE_OUTPUT, 3, shots=shots, use_qiskit=use_qiskit, rng=rng)
            P_beta += float(wb) * p_ab
        P += float(wa) * P_beta / max(float(np.sum(wB)), 1e-15)
    P = P / max(float(np.sum(P)), 1e-15)
    return probability_centroid(P, OUT_CENTERS_N)


@dataclass
class PlantCfg:
    dt: float = 0.02
    a: float = 0.985
    b: float = 0.12
    d: float = 0.20
    f: float = 0.35
    noise_std: float = 0.02


def plant_step(x_mm: float, y_urad: float, t: float, cfg: PlantCfg, rng: np.random.Generator) -> float:
    w = cfg.d * np.sin(2 * np.pi * cfg.f * t) + cfg.noise_std * rng.normal()
    x_next = cfg.a * x_mm + cfg.b * (y_urad / 140.0) + w
    return float(np.clip(x_next, X_MM_MIN, X_MM_MAX))


def controller(x_mm: float, args: argparse.Namespace, rng: np.random.Generator) -> float:
    xn = float(to_norm(np.array([x_mm]), X_MM_MIN, X_MM_MAX)[0])
    yn = qt3_eval_single(xn, n_alpha=args.n_alpha, n_beta=args.n_beta, beta_spread=args.beta_spread, gamma_alpha=args.gamma_alpha, gamma_beta=args.gamma_beta, shots=args.shots, use_qiskit=args.use_qiskit, rng=rng)
    y = float(from_norm(np.array([yn]), Y_URAD_MIN, Y_URAD_MAX)[0])
    return float(np.clip(y, Y_URAD_MIN, Y_URAD_MAX))


def simulate_episode(x0: float, args: argparse.Namespace, rng: np.random.Generator) -> dict:
    cfg = PlantCfg(dt=args.dt)
    n = int(args.t_final / cfg.dt) + 1
    t = np.linspace(0.0, args.t_final, n)
    x = np.zeros(n); y = np.zeros(n)
    x[0] = float(np.clip(x0, X_MM_MIN, X_MM_MAX))
    for k in range(n - 1):
        y[k] = controller(x[k], args, rng)
        x[k + 1] = plant_step(x[k], y[k], t[k], cfg, rng)
    y[-1] = y[-2]
    return {"t": t, "x_mm": x, "y_urad": y, "metrics": weighted_metrics(x, cfg.dt)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--t-final", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--shots", type=int, default=512)
    p.add_argument("--use-qiskit", action="store_true")
    p.add_argument("--n-alpha", type=int, default=11)
    p.add_argument("--n-beta", type=int, default=5)
    p.add_argument("--beta-spread", type=float, default=0.25)
    p.add_argument("--gamma-alpha", type=float, default=1.0)
    p.add_argument("--gamma-beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    x0s = np.linspace(2.5, 4.0, args.episodes)
    rows = []
    last = None
    for i, x0 in enumerate(x0s, start=1):
        out = simulate_episode(float(x0), args, rng)
        last = out
        row = {"episode": i, "x0_mm": float(x0)}
        row.update(out["metrics"])
        rows.append(row)
    df = pd.DataFrame(rows)
    print("QT3-FIS — Particle Accelerator C2")
    print(df.to_string(index=False))
    print("\nMean metrics:")
    print(df.drop(columns=["episode", "x0_mm"]).mean().to_string())
    os.makedirs(ROOT / "results", exist_ok=True)
    df.to_csv(ROOT / "results" / "qt3_particle_accelerator_c2_metrics.csv", index=False)
    if last is not None:
        np.savez_compressed(ROOT / "results" / "qt3_particle_accelerator_c2_last_episode.npz", t=last["t"], x_mm=last["x_mm"], y_urad=last["y_urad"], metrics=json.dumps(last["metrics"]))
    if args.plot and last is not None:
        plt.figure(); plt.plot(last["t"], last["x_mm"]); plt.xlabel("Time [s]"); plt.ylabel("BPM displacement [mm]"); plt.title("QT3-FIS C2 last episode"); plt.grid(True); plt.show()


if __name__ == "__main__":
    main()
