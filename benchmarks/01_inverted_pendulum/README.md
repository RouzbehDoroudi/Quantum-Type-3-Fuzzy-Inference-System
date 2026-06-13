# Benchmark 1 — Inverted Pendulum

This folder contains the QT3-FIS-only implementation for the inverted pendulum benchmark. The controller uses three linguistic labels for angle and angular velocity, five singleton output labels, UMF/LMF Type-3 memberships, alpha slicing, oracle-based rule mapping, measurement probabilities, and alpha-level fusion.

Run:

```bash
python benchmarks/01_inverted_pendulum/run_qt3_inverted_pendulum.py --shots 512
```

For gate-level Qiskit Aer execution:

```bash
python benchmarks/01_inverted_pendulum/run_qt3_inverted_pendulum.py --use-qiskit --shots 1024
```


## Article-consistency note

The plant update follows the article formulation `theta_dot = omega` and `omega_dot = (g/L) sin(theta) + Fc/(m L^2)`.
