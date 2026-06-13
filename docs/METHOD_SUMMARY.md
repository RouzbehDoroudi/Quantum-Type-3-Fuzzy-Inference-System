# Method summary

QT3-FIS maps Type-3 fuzzy inference to a quantum-inspired rule-evaluation pipeline.

## Main computational steps

1. Normalize system inputs.
2. Evaluate Type-3 upper and lower membership functions.
3. Apply alpha-level slicing.
4. Build rule activation weights for each alpha slice.
5. Pass the rule weights through a quantum-oracle output mapping.
6. Estimate output-label probabilities by measurement or finite-shot sampling.
7. Compute a probability-weighted singleton output.
8. Fuse outputs across alpha levels.
9. Apply the crisp controller output to the plant.

## Quantum execution modes

The package supports two modes:

- `use_qiskit=False`: fast probability path with optional finite-shot multinomial sampling.
- `use_qiskit=True`: Qiskit Aer circuit path using MCX oracle gates.

The first mode is useful for fast public reproducibility. The second mode demonstrates the gate-level oracle mechanism but can be slow for long closed-loop simulations.


## Article-alignment note

The public package keeps only the proposed QT3-FIS path. It is intended for clean public reproducibility and method demonstration, not for reproducing every comparison table, private notebook output, or exact circuit-resource table from the article.
