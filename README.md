# QT3-FIS — Quantum Type-3 Fuzzy Inference System

Public code package for the proposed **Quantum Type-3 Fuzzy Inference System (QT3-FIS)** associated with:

> Rouzbeh Doroudi and Siyamak Doroudi, **"QT3-FIS: A Quantum Type-3 fuzzy inference system for nonlinear, industrial, and structural applications"**, *Applied Soft Computing*, 200, 115406, 2026. DOI: https://doi.org/10.1016/j.asoc.2026.115406.

## Scope of this public release

This repository intentionally contains **only the proposed Quantum Type-3 FIS (QT3-FIS)** implementation.

The following comparison codes are **not included** in this public release:

- Type-1 FIS
- Type-2 FIS
- Classical Type-3 FIS
- Quantum Type-1 FIS
- Quantum Type-2 FIS
- all-method comparison notebooks

The included implementation focuses on the QT3-FIS workflow:

1. Type-3 UMF/LMF membership handling
2. alpha-level slicing
3. quantum-oracle rule mapping
4. finite-shot measurement/probability-based consequent aggregation
5. alpha-level fusion
6. Four benchmark examples from the article

## Repository structure

```text
QT3-FIS-Quantum-Only-Public-Release/
├── src/qt3fis/
│   ├── __init__.py
│   └── qt3_core.py
├── benchmarks/
│   ├── 01_inverted_pendulum/
│   ├── 02_particle_accelerator_beam_c2/
│   ├── 03_cart_pole/
│   └── 04_eleven_story_building_atmd/
├── data/
├── results/
├── docs/
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Conda alternative:

```bash
conda env create -f environment.yml
conda activate qt3fis-quantum-only
```

## Quick smoke test

These commands are intended only to verify that the package runs on a normal computer. They are **not** intended to reproduce the paper-level finite-shot tables.

```bash
python benchmarks/01_inverted_pendulum/run_qt3_inverted_pendulum.py --t-final 0.2 --shots 128
python benchmarks/02_particle_accelerator_beam_c2/run_qt3_particle_accelerator_c2.py --episodes 1 --t-final 0.2 --shots 128
python benchmarks/03_cart_pole/run_qt3_cart_pole.py --t-final 0.2 --shots 128
python benchmarks/04_eleven_story_building_atmd/run_qt3_eleven_story_atmd.py --t-final 0.2 --shots 128
```

## Paper-level finite-shot runs

For reporting results consistent with the article, use a sufficiently large shot budget or run the full shot sweep. The article evaluates finite-shot behavior over multiple shot budgets, including:

```text
64, 128, 256, 512, 1024, 4096
```

Recommended single-run command with 4096 shots:

```bash
python benchmarks/01_inverted_pendulum/run_qt3_inverted_pendulum.py --shots 4096
python benchmarks/02_particle_accelerator_beam_c2/run_qt3_particle_accelerator_c2.py --shots 4096
python benchmarks/03_cart_pole/run_qt3_cart_pole.py --shots 4096
python benchmarks/04_eleven_story_building_atmd/run_qt3_eleven_story_atmd.py --shots 4096
```

## Optional Qiskit Aer gate-level demonstration

The default scripts use a finite-shot probability-sampling implementation that follows the same QT3-FIS probability aggregation logic and runs quickly. The `--use-qiskit` option enables a gate-level Qiskit Aer demonstration of the oracle path. It is slower because QT3-FIS requires repeated alpha-slice evaluations.

```bash
python benchmarks/01_inverted_pendulum/run_qt3_inverted_pendulum.py --use-qiskit --shots 1024
python benchmarks/02_particle_accelerator_beam_c2/run_qt3_particle_accelerator_c2.py --use-qiskit --shots 1024
python benchmarks/03_cart_pole/run_qt3_cart_pole.py --use-qiskit --shots 1024
python benchmarks/04_eleven_story_building_atmd/run_qt3_eleven_story_atmd.py --use-qiskit --shots 1024
```

## Structural benchmark data note

The eleven-story ATMD example can run in demo mode without external files. For paper-level reproduction, place the original El Centro record in:

```text
data/elcentro.mat
```

The public package does not include publisher PDF files, private notebook outputs, local paths, or external data files with uncertain redistribution rights.

## Reproducibility notes

- All scripts accept `--seed` for finite-shot sampling reproducibility.
- Results are written to the `results/` directory.
- The public package is designed as a clean QT3-FIS release, not as a full comparison repository.
- For exact reproduction of all article tables, use the same earthquake record, benchmark settings, shot budgets, and seed sweep used in the article.

## Citation
**Software DOI:** https://doi.org/10.5281/zenodo.20676187
If you use this software, please cite both the software DOI and the associated Applied Soft Computing article below.
```bibtex
@article{Doroudi2026QT3FIS,
  title   = {QT3-FIS: A Quantum Type-3 fuzzy inference system for nonlinear, industrial, and structural applications},
  author  = {Doroudi, Rouzbeh and Doroudi, Siyamak},
  journal = {Applied Soft Computing},
  volume  = {200},
  pages   = {115406},
  year    = {2026},
  doi     = {10.1016/j.asoc.2026.115406},
  url     = {https://doi.org/10.1016/j.asoc.2026.115406}
}
```

## License

MIT License.
