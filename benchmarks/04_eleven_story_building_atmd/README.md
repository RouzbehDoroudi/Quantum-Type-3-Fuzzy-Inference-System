# Benchmark 4 — Eleven-story Shear Building with ATMD

This folder contains a QT3-FIS-only structural-control implementation. It uses the 11-story mass/stiffness arrays from the supplied working notebook, a roof ATMD, a 5×5 rule base, seven output labels, Type-3 UMF/LMF alpha slicing, and oracle-based output aggregation.

For a self-contained demo, the script generates a synthetic earthquake-like record if `data/elcentro.mat` is not available. For paper-level reproduction, place the original El Centro record in:

```text
data/elcentro.mat
```

Run demo mode:

```bash
python benchmarks/04_eleven_story_building_atmd/run_qt3_eleven_story_atmd.py --shots 4096
```

Optional Qiskit Aer execution is possible but slow because the controller is evaluated many times inside the structural time-history simulation:

```bash
python benchmarks/04_eleven_story_building_atmd/run_qt3_eleven_story_atmd.py --use-qiskit --shots 4096
```


## Shot budget note

The article reports finite-shot behavior for multiple shot budgets, including 64, 128, 256, 512, 1024, and 4096 shots. Use 128 shots only for a quick smoke test. Use 4096 shots, or the full shot sweep, for paper-level reporting.
