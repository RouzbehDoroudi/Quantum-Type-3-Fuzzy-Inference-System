# Data folder

This public code package does not include external or publisher-controlled data.

For the structural benchmark, place the El Centro earthquake record here if you want paper-level reproduction:

```text
data/elcentro.mat
```

The script accepts MATLAB files with either:

- key `e` as a 2×N or N×2 array containing time and acceleration, or
- keys `t` and `a`.

If no file is provided, the structural script runs a synthetic earthquake-like demo record.
