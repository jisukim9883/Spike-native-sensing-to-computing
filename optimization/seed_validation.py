"""
Seed robustness of the optimized stochastic Vd operation.

Evaluates the frozen operating point (mu* = sigma* = 0.14 V, theta* = 4574.37)
on independently generated Vd sequences, as described in Methods:
the value in Supplementary Table 2 (R2 = 0.961) corresponds to the single
sequence used during optimization (seed 42); across 100 independent sequences
the fit yields R2 = 0.963 +/- 0.001 (Fig. 1g, Supplementary Fig. 7).

Input : data/mean_current_summary.csv
Output: results/seed_validation.csv   (per-seed R2, slope, firing-rate span)

Tested with Python 3.12, NumPy 1.26.4, pandas 2.2.2.
Runtime: ~1-2 min on a normal desktop computer.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════
SUMMARY_CSV = Path("data/mean_current_summary.csv")
OUT_ROOT = Path("results")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MU, SIGMA, THETA   = 0.14, 0.14, 4574.37   # optimized operating point (Supp. Table 2)
BIAS_MIN, BIAS_MAX = 0.0, 0.99
N_SAMPLES = 5000
SEEDS     = range(100)
OPT_SEED  = 42                              # sequence used during optimization

# ══════════════════════════════════════════
# Data
# ══════════════════════════════════════════
df          = pd.read_csv(SUMMARY_CSV)
intensities = df["intensity"].to_numpy(float)
bias_cols   = [c for c in df.columns if c != "intensity"]
bias_grid   = np.array([float(c.replace("V", "")) for c in bias_cols])
M           = df[bias_cols].to_numpy(dtype=float)
print(f"Data: {len(intensities)} intensities x {len(bias_grid)} bias steps")
print(f"Operating point: mu={MU}, sigma={SIGMA}, theta={THETA}\n")

# ══════════════════════════════════════════
# Common functions (identical to vd_optimization.py)
# ══════════════════════════════════════════
def linear_r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res = np.sum((y - A @ coef) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def sample_vd(seed):
    return np.clip(np.random.default_rng(seed).logistic(MU, SIGMA, N_SAMPLES),
                   BIAS_MIN, BIAS_MAX)

def get_currents(v):
    return np.vstack([np.interp(v, bias_grid, row) for row in M])

# ══════════════════════════════════════════
# Evaluate the frozen operating point on independent Vd sequences
# ══════════════════════════════════════════
rows = []
for sd in SEEDS:
    rates = (get_currents(sample_vd(sd)) > THETA).mean(axis=1)
    rows.append(dict(seed=sd,
                     R2=linear_r2(intensities, rates),
                     slope=np.polyfit(intensities, rates, 1)[0],
                     rate_min=rates.min(), rate_max=rates.max(),
                     span=rates.max() - rates.min()))

d = pd.DataFrame(rows)
d.to_csv(OUT_ROOT / "seed_validation.csv", index=False)

# ══════════════════════════════════════════
# Summary
# ══════════════════════════════════════════
r2 = d.R2.to_numpy()
print(f"{'n seeds':>18s} : {len(d)}")
print(f"{'R2 (mean +/- sd)':>18s} : {r2.mean():.4f} +/- {r2.std(ddof=1):.4f}"
      f"   [{r2.min():.4f}, {r2.max():.4f}]")
s42 = d[d.seed == OPT_SEED]
if len(s42):
    print(f"{'seed 42':>18s} : {s42.R2.values[0]:.5f}"
          f"   (single sequence used during optimization; Supp. Table 2)")
print(f"{'slope':>18s} : {d.slope.mean():.5f} +/- {d.slope.std(ddof=1):.5f}")
print(f"{'firing-rate span':>18s} : {d.span.mean():.4f} +/- {d.span.std(ddof=1):.4f}")
print(f"\nSaved: {OUT_ROOT / 'seed_validation.csv'}")