"""
Environment-adaptive threshold control under varying illumination
(cf. Fig. 4, Supplementary Table 3).

For each illumination condition (dim / normal / bright), three settings are
evaluated with the deployed logistic Vd distribution as reference:

  Full    : joint grid sweep of (mu, sigma) with a theta scan per grid point
  theta-only : theta re-scan with (mu, sigma) fixed at the deployed values
  Fixed   : deployed operating point (mu*, sigma*, theta*_normal) unchanged

For the normal condition, the deployed operating point itself
(mu* = sigma* = 0.14 V, theta* = 4574.37) is evaluated as the reference row.
Each quantity is aggregated over six independent Vd sequences
(median theta, mean R2).

Input : data/mean_current_summary_dim.csv
        data/mean_current_summary.csv          (normal)
        data/mean_current_summary_bright.csv
Output: results/adaptive_threshold.csv

Expected output (Supplementary Table 3):
    normal  Full        0.14  0.14  4574.4  0.9619
    dim     Full        0.21  0.14  2816.3  0.9567
    dim     theta-only  0.14  0.14  2750.6  0.9371
    dim     Fixed       0.14  0.14  4574.4  0.8465
    bright  Full        0.15  0.11  5452.7  0.9751
    bright  theta-only  0.14  0.14  5459.4  0.9717
    bright  Fixed       0.14  0.14  4574.4  0.9208

Tested with Python 3.12, NumPy 1.26.4, pandas 2.2.2.
Runtime: ~5-10 min on a normal desktop computer (23 x 23 grid sweeps).
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════
LOOKUPS = {   # condition -> (path, environment index for seed offset)
    "dim":    (Path("data/mean_current_summary_dim.csv"),    0),
    "normal": (Path("data/mean_current_summary.csv"),        1),
    "bright": (Path("data/mean_current_summary_bright.csv"), 2),
}
OUT_ROOT   = Path("results")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MU_STAR, SIGMA_STAR, THETA_STAR = 0.14, 0.14, 4574.37   # deployed operating point
BIAS_MIN, BIAS_MAX = 0.0, 0.99
N_SAMPLES = 4000
N_THRESH  = 512
BASE_SEED = 123
N_SEEDS   = 6
THETA_PCT = (1.0, 99.0)                                  # theta scan range (percentile)
FULL_GRID = np.round(np.linspace(0.04, 0.26, 23), 3)     # mu, sigma grid for Full

# ══════════════════════════════════════════
# Common functions
# ══════════════════════════════════════════
def sample_vd(mu, sig, seed):
    return np.clip(np.random.default_rng(seed).logistic(mu, sig, N_SAMPLES),
                   BIAS_MIN, BIAS_MAX)

def r2_of_rates(ints, rates):
    xc = ints - ints.mean()
    yc = rates - rates.mean()
    den = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float(((xc * yc).sum() / den) ** 2) if den > 0 else 0.0

def r2_over_thetas(ints, cur, ths):
    """R2 of the linear fit for every candidate theta (vectorized)."""
    cs = np.sort(cur, axis=1)
    n  = cur.shape[1]
    idx = np.vstack([np.searchsorted(cs[i], ths, side="right")
                     for i in range(cur.shape[0])])
    rates = (n - idx) / n
    xc = ints - ints.mean()
    yc = rates - rates.mean(axis=0, keepdims=True)
    den = np.sqrt((xc**2).sum() * (yc**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, (xc[:, None] * yc).sum(0) / den, 0.0)
    return r**2

def scan_theta_multiseed(ints, grid, M, mu, sig, seeds):
    """Per-seed theta scan; aggregate as median theta / mean R2."""
    ths_l, r2_l = [], []
    for s in seeds:
        cur = np.vstack([np.interp(sample_vd(mu, sig, s), grid, row) for row in M])
        lo, hi = np.percentile(cur, THETA_PCT)
        ths = np.linspace(lo, hi, N_THRESH)
        r2  = r2_over_thetas(ints, cur, ths)
        k   = int(np.nanargmax(r2))
        ths_l.append(float(ths[k]))
        r2_l.append(float(r2[k]))
    return float(np.median(ths_l)), float(np.mean(r2_l))

def eval_fixed_multiseed(ints, grid, M, mu, sig, theta, seeds):
    r2_l = []
    for s in seeds:
        cur = np.vstack([np.interp(sample_vd(mu, sig, s), grid, row) for row in M])
        r2_l.append(r2_of_rates(ints, (cur > theta).mean(axis=1)))
    return float(np.mean(r2_l))

# ══════════════════════════════════════════
# Run
# ══════════════════════════════════════════
rows = []
print(f"{'condition':>9s} | {'mode':>10s} | {'mu':>5s} {'sigma':>5s} | "
      f"{'theta':>8s} | {'R2':>7s}")
print("-" * 58)

for cond, (path, env_idx) in LOOKUPS.items():
    if not path.exists():
        print(f"{cond:>9s} | file not found: {path} -- skipped")
        continue

    df    = pd.read_csv(path)
    ints  = df["intensity"].to_numpy(float)
    cols  = [c for c in df.columns if c != "intensity"]
    grid  = np.array([float(c.replace("V", "")) for c in cols])
    M     = df[cols].to_numpy(float)
    seeds = [BASE_SEED + env_idx * 100 + s for s in range(N_SEEDS)]

    if cond == "normal":
        # Reference row: deployed operating point evaluated under this convention
        r2 = eval_fixed_multiseed(ints, grid, M, MU_STAR, SIGMA_STAR, THETA_STAR, seeds)
        rows.append(dict(condition=cond, mode="Full", mu=MU_STAR, sigma=SIGMA_STAR,
                         theta=THETA_STAR, R2=r2))
        print(f"{cond:>9s} | {'Full':>10s} | {MU_STAR:5.2f} {SIGMA_STAR:5.2f} | "
              f"{THETA_STAR:8.1f} | {r2:7.4f}")
        continue

    # ---- Full: joint (mu, sigma) grid sweep with per-point theta scan
    best = (-np.inf, None, None, None)
    for mu in FULL_GRID:
        for sig in FULL_GRID:
            th, r2 = scan_theta_multiseed(ints, grid, M, mu, sig, seeds)
            if r2 > best[0]:
                best = (r2, float(mu), float(sig), th)
    r2, mu, sig, th = best
    rows.append(dict(condition=cond, mode="Full", mu=mu, sigma=sig, theta=th, R2=r2))
    print(f"{cond:>9s} | {'Full':>10s} | {mu:5.2f} {sig:5.2f} | {th:8.1f} | {r2:7.4f}")

    # ---- theta-only: (mu, sigma) fixed at deployed values
    th, r2 = scan_theta_multiseed(ints, grid, M, MU_STAR, SIGMA_STAR, seeds)
    rows.append(dict(condition=cond, mode="theta-only", mu=MU_STAR, sigma=SIGMA_STAR,
                     theta=th, R2=r2))
    print(f"{cond:>9s} | {'theta-only':>10s} | {MU_STAR:5.2f} {SIGMA_STAR:5.2f} | "
          f"{th:8.1f} | {r2:7.4f}")

    # ---- Fixed: deployed operating point unchanged
    r2 = eval_fixed_multiseed(ints, grid, M, MU_STAR, SIGMA_STAR, THETA_STAR, seeds)
    rows.append(dict(condition=cond, mode="Fixed", mu=MU_STAR, sigma=SIGMA_STAR,
                     theta=THETA_STAR, R2=r2))
    print(f"{cond:>9s} | {'Fixed':>10s} | {MU_STAR:5.2f} {SIGMA_STAR:5.2f} | "
          f"{THETA_STAR:8.1f} | {r2:7.4f}")

pd.DataFrame(rows).to_csv(OUT_ROOT / "adaptive_threshold.csv", index=False)
print(f"\nSaved: {OUT_ROOT / 'adaptive_threshold.csv'}")