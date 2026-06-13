# Benchmark 2 — Particle Accelerator Beam Control C2

This implementation keeps only the QT3-FIS controller for the QFIE-style C2 beam-steering example. The rule mapping is the compact negative-feedback rule base:

```text
N -> P
Z -> Z
P -> N
```

Run:

```bash
python benchmarks/02_particle_accelerator_beam_c2/run_qt3_particle_accelerator_c2.py --shots 512
```

For Qiskit Aer:

```bash
python benchmarks/02_particle_accelerator_beam_c2/run_qt3_particle_accelerator_c2.py --use-qiskit --shots 1024
```
