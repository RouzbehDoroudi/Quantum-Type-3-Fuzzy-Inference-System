# Provenance and filtering

This public package was prepared from the author's supplied notebooks and the published QT3-FIS article.

## Source notebooks used

- `invert pendum_ article_based.ipynb`
- `Type3_2_1_FIS_C2_from_QFIE_QuantumBase.ipynb`
- `CartPole_optimal.ipynb`
- `benchmark_11strory_QT3_new_five membership_based on matlab.ipynb`

## Public-release filtering rule

Only the proposed QT3-FIS workflow is retained:

- Type-3 UMF/LMF memberships
- alpha slicing
- quantum-oracle rule mapping
- measurement/probability output aggregation
- alpha fusion

The following are excluded:

- Type-1, Type-2, QT1, QT2 baselines
- classical Type-3 comparison code
- all-method comparison plots
- private local file paths and notebook outputs
- publisher PDF files

## Benchmark mapping

| Benchmark | Public script |
|---|---|
| Inverted pendulum | `benchmarks/01_inverted_pendulum/run_qt3_inverted_pendulum.py` |
| Particle accelerator beam C2 | `benchmarks/02_particle_accelerator_beam_c2/run_qt3_particle_accelerator_c2.py` |
| Cart-Pole | `benchmarks/03_cart_pole/run_qt3_cart_pole.py` |
| Eleven-story building ATMD | `benchmarks/04_eleven_story_building_atmd/run_qt3_eleven_story_atmd.py` |
