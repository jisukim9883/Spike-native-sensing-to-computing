"""
Two-stage optimization of the stochastic Vd distribution
(Logistic / Beta / Uniform), as described in Methods.

Input : data/mean_current_summary.csv
        array-averaged Iph-Vd lookup table
        (26 grayscale levels x 100 bias values; first column "intensity",
         remaining columns named by bias voltage, e.g. "0.00V")
Output: results/history_<distribution>.csv
        per-iteration optimization log (cf. Supplementary Table 2)

Final values reported in Supplementary Table 2:
    Logistic : R2 = 0.9614   mu* = 0.1400    sigma* = 0.1400   theta* = 4574.37
    Beta     : R2 = 0.9631   alpha* = 1.0000 beta* = 3.2105    theta* = 4925.01
    Uniform  : R2 = 0.9466   c* = 0.1737     w* = 0.2395       theta* = 4854.09

Tested with Python 3.12, NumPy 1.26.4, pandas 2.2.2.
Runtime: ~2-5 min on a normal desktop computer.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════
SUMMARY_CSV = Path("data/mean_current_summary.csv")
OUT_ROOT    = Path("results")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

BIAS_MIN, BIAS_MAX = 0.0, 0.99
N_SAMPLES   = 5000
N_THRESH    = 512
BASE_SEED   = 42
MAX_ITER    = 20
TOL         = 1e-5
INIT_THETA  = 2500.0

# Logistic grid: [0.02, 0.14] V in steps of 0.005 V (25 x 25 points)
CENTER_GRID = np.linspace(0.02, 0.14, 25)
SCALE_GRID  = np.linspace(0.02, 0.14, 25)

DIST_CONFIGS = {
    "Logistic": {
        "p1_init":  0.05, "p2_init": 0.06,
        "p1_label": "mu", "p2_label": "sigma",
        "p1_grid":  CENTER_GRID,
        "p2_grid":  SCALE_GRID,
    },
    "Beta": {
        # alpha >= 1: the beta density diverges at Vd = 0 for alpha < 1
        "p1_init":  2.0, "p2_init": 2.0,
        "p1_label": "alpha", "p2_label": "beta",
        "p1_grid":  np.linspace(1.0, 8.0, 20),
        "p2_grid":  np.linspace(1.0, 8.0, 20),
    },
    "Uniform": {
        # c: center, w: half-width; sampled interval is [c - w, c + w]
        # intersected with the accessible bias range
        "p1_init":  0.40, "p2_init": 0.20,
        "p1_label": "c",  "p2_label": "w",
        "p1_grid":  np.linspace(0.10, 0.80, 20),
        "p2_grid":  np.linspace(0.05, 0.45, 20),
    },
}

# ══════════════════════════════════════════
# Data
# ══════════════════════════════════════════
df          = pd.read_csv(SUMMARY_CSV)
intensities = df["intensity"].to_numpy()
bias_cols   = [c for c in df.columns if c != "intensity"]
bias_grid   = np.array([float(c.replace("V", "")) for c in bias_cols])
M           = df[bias_cols].to_numpy(dtype=float)
print(f"Data: {len(intensities)} intensities x {len(bias_grid)} bias steps")
print(f"N_SAMPLES={N_SAMPLES}, BASE_SEED={BASE_SEED}\n")

# ══════════════════════════════════════════
# Common functions
# ══════════════════════════════════════════
def linear_r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat   = A @ coef
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return float(1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0

def get_currents(samples):
    return np.vstack([np.interp(samples, bias_grid, row) for row in M])

def scan_theta(cur):
    ths = np.linspace(cur.min(), cur.max(), N_THRESH)
    best_r2, best_th = -np.inf, ths[0]
    for th in ths:
        r2 = linear_r2(intensities, (cur > th).mean(axis=1))
        if r2 > best_r2:
            best_r2, best_th = r2, th
    return best_th, best_r2

# ══════════════════════════════════════════
# Samplers (fixed seed: a given parameter set always
# generates the same Vd sequence)
# ══════════════════════════════════════════
def make_sampler(dist_name):
    def _logistic(p1, p2):
        return np.clip(np.random.default_rng(BASE_SEED).logistic(p1, p2, N_SAMPLES),
                       BIAS_MIN, BIAS_MAX)
    def _beta(p1, p2):
        return np.random.default_rng(BASE_SEED).beta(p1, p2, N_SAMPLES) * BIAS_MAX
    def _uniform(p1, p2):
        lo = np.clip(p1 - p2, BIAS_MIN, BIAS_MAX)
        hi = np.clip(p1 + p2, BIAS_MIN, BIAS_MAX)
        if hi <= lo:
            return np.full(N_SAMPLES, (lo + hi) / 2)
        return np.random.default_rng(BASE_SEED).uniform(lo, hi, N_SAMPLES)
    return {"Logistic": _logistic, "Beta": _beta, "Uniform": _uniform}[dist_name]

# ══════════════════════════════════════════
# Two-stage coordinate ascent
# ══════════════════════════════════════════
def run_coord_ascent(dist_name, cfg):
    sampler  = make_sampler(dist_name)
    p1_grid  = cfg["p1_grid"]
    p2_grid  = cfg["p2_grid"]
    p1, p2   = cfg["p1_init"], cfg["p2_init"]
    l1, l2   = cfg["p1_label"], cfg["p2_label"]

    # Initial state: theta fixed at INIT_THETA
    th  = INIT_THETA
    cur = get_currents(sampler(p1, p2))
    r2  = linear_r2(intensities, (cur > th).mean(axis=1))

    print(f"\n{'='*60}")
    print(f"  {dist_name}  (init: {l1}={p1}, {l2}={p2}, theta={th:.0f})")
    print(f"{'Iter':>4} | {'Stage':>7} | {'R2':>8} | {l1:>8} | {l2:>7} | {'theta':>10}")
    print(f"{'-'*55}")
    print(f"{'0':>4} | {'Initial':>7} | {r2:.5f} | {p1:>8.4f} | {p2:>7.4f} | {th:>10.2f}")

    history = [{"iter": 0, "stage": "initial",
                "p1": p1, "p2": p2, "theta": th, "R2": r2}]
    prev_r2 = r2

    for it in range(1, MAX_ITER + 1):
        # Stage 1: theta scan at fixed (p1, p2)
        cur_s1        = get_currents(sampler(p1, p2))
        th_new, r2_s1 = scan_theta(cur_s1)
        if r2_s1 >= prev_r2 - TOL:
            th = th_new
        history.append({"iter": it, "stage": "stage1",
                        "p1": p1, "p2": p2, "theta": th, "R2": r2_s1})
        print(f"{it:>4} | {'Stage1':>7} | {r2_s1:.5f} | {p1:>8.4f} | {p2:>7.4f} | {th:>10.2f}")

        # Stage 2: (p1, p2) grid search at fixed theta
        best_r2_s2, best_p1, best_p2 = -np.inf, p1, p2
        for q1 in p1_grid:
            for q2 in p2_grid:
                cur_q = get_currents(sampler(q1, q2))
                rates = (cur_q > th).mean(axis=1)
                r2_q  = linear_r2(intensities, rates)
                if r2_q > best_r2_s2:
                    best_r2_s2, best_p1, best_p2 = r2_q, q1, q2

        history.append({"iter": it, "stage": "stage2",
                        "p1": best_p1, "p2": best_p2, "theta": th, "R2": best_r2_s2})
        print(f"{it:>4} | {'Stage2':>7} | {best_r2_s2:.5f} | {best_p1:>8.4f} | {best_p2:>7.4f} | {th:>10.2f}")

        # Convergence
        delta = best_r2_s2 - prev_r2
        if abs(delta) < TOL and it >= 3:
            print(f"\n  Converged (dR2={delta:.2e}) at iter {it}")
            break

        # Monotone ascent: update only on improvement
        if best_r2_s2 > prev_r2:
            p1, p2, prev_r2 = best_p1, best_p2, best_r2_s2
        else:
            print(f"       -> no improvement ({best_r2_s2:.5f} <= {prev_r2:.5f}), kept")

    final = history[-1]
    print(f"\n  Final: R2={final['R2']:.5f} | theta={final['theta']:.2f} | "
          f"{l1}={final['p1']:.4f} | {l2}={final['p2']:.4f}")

    df_h = pd.DataFrame(history)
    df_h.to_csv(OUT_ROOT / f"history_{dist_name.lower()}.csv", index=False)
    return df_h

# ══════════════════════════════════════════
# Run
# ══════════════════════════════════════════
all_hist = {}
for dist_name, cfg in DIST_CONFIGS.items():
    all_hist[dist_name] = run_coord_ascent(dist_name, cfg)

# ══════════════════════════════════════════
# Summary
# ══════════════════════════════════════════
final_r2s = {n: h[h["stage"] == "stage2"]["R2"].iloc[-1] for n, h in all_hist.items()}
names_sorted = sorted(final_r2s, key=final_r2s.get, reverse=True)

print(f"\n{'='*55}")
print(f"{'Distribution':>12} | {'Final R2':>8} | {'theta':>8} | {'p1':>7} | {'p2':>7}")
print("-" * 50)
for n in names_sorted:
    last = all_hist[n][all_hist[n]["stage"] == "stage2"].iloc[-1]
    print(f"{n:>12} | {last['R2']:>8.5f} | {last['theta']:>8.1f} | "
          f"{last['p1']:>7.4f} | {last['p2']:>7.4f}")