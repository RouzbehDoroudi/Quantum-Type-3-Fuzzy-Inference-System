# Benchmark 3 — Cart-Pole

This folder contains the QT3-FIS-only implementation for the Cart-Pole benchmark. It uses the optimized 5×5 rule matrix and rule weights from the supplied CartPole notebook. The output has seven singleton force levels.

Run:

```bash
python benchmarks/03_cart_pole/run_qt3_cart_pole.py --shots 512
```

Optional Qiskit Aer execution:

```bash
python benchmarks/03_cart_pole/run_qt3_cart_pole.py --use-qiskit --shots 1024
```
