# Spike-native sensing-to-computing

Code and data for "Spike-native Sensing-to-Computing Integrated System Enabled by Stochastic Operation of a Silicon Nanomembrane Artificial Retina".

| Script | Reproduces |
|---|---|
| `optimization/vd_optimization.py` | Fig. 3, Supplementary Table 2 |
| `optimization/seed_validation.py` | Fig. 1g, Supplementary Fig. 7 |
| `optimization/adaptive_threshold.py` | Fig. 4, Supplementary Table 3 |
| `optimization/boundary_check.py` | Methods (search-range verification) |

## System requirements

- Python 3.12, NumPy 1.26.4, pandas 2.2.2 (tested versions)
- OS: macOS / Windows / Linux
- No non-standard hardware required

## Installation

```bash
pip install numpy==1.26.4 pandas==2.2.2
```

Install time: about 1 minute on a normal desktop computer.

## Demo

From the repository root:

```bash
python optimization/vd_optimization.py      # 2-5 min
```

Expected output:

```
Distribution | Final R2 |    theta |      p1 |      p2
--------------------------------------------------
        Beta |  0.96310 |   4925.0 |  1.0000 |  3.2105
    Logistic |  0.96141 |   4574.4 |  0.1400 |  0.1400
     Uniform |  0.94655 |   4854.1 |  0.1737 |  0.2395
```

```bash
python optimization/seed_validation.py      # 1-2 min
```

Expected output:

```
R2 (mean +/- sd) : 0.9629 +/- 0.0011
         seed 42 : 0.96141
```

```bash
python optimization/adaptive_threshold.py   # 5-10 min
```

Expected output (Supplementary Table 3):

```
   dim | Full       | 0.21 0.14 | 2816.3 | 0.9567
   dim | theta-only | 0.14 0.14 | 2750.6 | 0.9371
   dim | Fixed      | 0.14 0.14 | 4574.4 | 0.8465
normal | Full       | 0.14 0.14 | 4574.4 | 0.9619
bright | Full       | 0.15 0.11 | 5452.7 | 0.9751
bright | theta-only | 0.14 0.14 | 5459.4 | 0.9717
bright | Fixed      | 0.14 0.14 | 4574.4 | 0.9208
```

```bash
python optimization/boundary_check.py       # 15-30 min
```

Verifies that the bounded search grids do not limit the optimization result; results are written to `results/boundary_check.csv`.

## Instructions for use

The scripts read the lookup tables in `data/`, measured under dim, normal and bright illumination: the first column `intensity` lists the input levels, and the remaining columns (named by bias voltage: `0.00V`, `0.01V`, ...) contain the array-averaged photocurrent. Any lookup table in this format can be used. Raw per-pixel measurement data are available from the corresponding authors upon reasonable request.

Scripts for SNN training and the CPU/GPU/chip inference benchmark are described in Methods; chip interface code requires custom hardware and is available upon reasonable request.

## License

MIT License. See `LICENSE`.
