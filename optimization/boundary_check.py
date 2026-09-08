"""
Grid-boundary check for the two-stage stochastic Vd optimization.

Repeats the identical optimization with extended parameter grids
(same procedure and step size as vd_optimization.py; only the grids differ).
The bounded results correspond to Supplementary Table 2; the extended run
verifies the effect of the search range. For the beta distribution, the
firing-rate response of the bounded and extended optima is additionally
compared to characterize the solution obtained when the constraint
alpha >= 1 is relaxed. All array-level measurements in the manuscript use
the bounded operating point.

Input : data/mean_current_summary.csv
Output: results/boundary_check.csv       (bounded vs extended, side by side)
        results/rate_curves_beta.csv     (firing rate vs intensity, beta optima)

Tested with Python 3.12, NumPy 1.26.4, pandas 2.2.2.
Runtime: ~15-30 min on a normal desktop computer (77x77 logistic grid).
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════
# Configuration (identical to vd_optimization.py except the grids)
# ══════════════════════════════════════════
SUMMARY_CSV = Path("data/mean_current_summary.csv")
OUT_ROOT = Path("results")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

BIAS_MIN, BIAS_MAX = 0.0, 0.99
N_SAMPLES   = 5000
N_THRESH    = 512
BASE_SEED   = 42
MAX_ITER    = 20
TOL         = 1e-5
INIT_THETA  = 2500.0
VAL_SEEDS   = range(100)   # out-of-sample re-evaluation at fixed parameters
DEAD_LEVEL  = 0.02         # firing rate below this counts as no response

GRID_SETS = {
    "bounded": {   # grids used in the manuscript
        "Logistic": dict(p1_grid=np.linspace(0.02, 0.14, 25),
                         p2_grid=np.linspace(0.02, 0.14, 25),
                         p1_init=0.05, p2_init=0.06,
                         p1_label="mu", p2_label="sigma"),
        "Beta":     dict(p1_grid=np.linspace(1.0, 8.0, 20),
                         p2_grid=np.linspace(1.0, 8.0, 20),
                         p1_init=2.0, p2_init=2.0,
                         p1_label="alpha", p2_label="beta"),
        "Uniform":  dict(p1_grid=np.linspace(0.10, 0.80, 20),
                         p2_grid=np.linspace(0.05, 0.45, 20),
                         p1_init=0.40, p2_init=0.20,
                         p1_label="c", p2_label="w"),
    },
    "extended": {  # same step size, wider range
        "Logistic": dict(p1_grid=np.linspace(0.02, 0.40, 77),   # step 0.005 kept
                         p2_grid=np.linspace(0.02, 0.40, 77),
                         p1_init=0.05, p2_init=0.06,
                         p1_label="mu", p2_label="sigma"),
        "Beta":     dict(p1_grid=np.linspace(0.2, 8.0, 40),     # alpha < 1 reachable
                         p2_grid=np.linspace(0.2, 8.0, 40),
                         p1_init=2.0, p2_init=2.0,
                         p1_label="alpha", p2_label="beta"),
        "Uniform":  dict(p1_grid=np.linspace(0.00, 0.95, 39),
                         p2_grid=np.linspace(0.02, 0.60, 30),
                         p1_init=0.40, p2_init=0.20,
                         p1_label="c", p2_label="w"),
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
# Common functions (identical to vd_optimization.py)
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

def make_sampler(dist_name, seed=BASE_SEED):
    def _logistic(p1, p2):
        return np.clip(np.random.default_rng(seed).logistic(p1, p2, N_SAMPLES),
                       BIAS_MIN, BIAS_MAX)
    def _beta(p1, p2):
        return np.random.default_rng(seed).beta(p1, p2, N_SAMPLES) * BIAS_MAX
    def _uniform(p1, p2):
        lo = np.clip(p1 - p2, BIAS_MIN, BIAS_MAX)
        hi = np.clip(p1 + p2, BIAS_MIN, BIAS_MAX)
        if hi <= lo:
            return np.full(N_SAMPLES, (lo + hi) / 2)
        return np.random.default_rng(seed).uniform(lo, hi, N_SAMPLES)
    return {"Logistic": _logistic, "Beta": _beta, "Uniform": _uniform}[dist_name]

# ══════════════════════════════════════════
# Two-stage coordinate ascent (identical procedure, condensed logging)
# ══════════════════════════════════════════
def run_coord_ascent(dist_name, cfg):
    sampler = make_sampler(dist_name)
    p1_grid, p2_grid = cfg["p1_grid"], cfg["p2_grid"]
    p1, p2 = cfg["p1_init"], cfg["p2_init"]

    th  = INIT_THETA
    cur = get_currents(sampler(p1, p2))
    prev_r2 = linear_r2(intensities, (cur > th).mean(axis=1))

    for it in range(1, MAX_ITER + 1):
        cur_s1        = get_currents(sampler(p1, p2))
        th_new, r2_s1 = scan_theta(cur_s1)
        if r2_s1 >= prev_r2 - TOL:
            th = th_new

        best_r2_s2, best_p1, best_p2 = -np.inf, p1, p2
        for q1 in p1_grid:
            for q2 in p2_grid:
                cur_q = get_currents(sampler(q1, q2))
                r2_q  = linear_r2(intensities, (cur_q > th).mean(axis=1))
                if r2_q > best_r2_s2:
                    best_r2_s2, best_p1, best_p2 = r2_q, q1, q2

        delta = best_r2_s2 - prev_r2
        if abs(delta) < TOL and it >= 3:
            break
        if best_r2_s2 > prev_r2:
            p1, p2, prev_r2 = best_p1, best_p2, best_r2_s2

    return p1, p2, th, prev_r2

def out_of_sample(dist_name, p1, p2, theta):
    """Re-evaluate the frozen (p1, p2, theta) on independent Vd realizations."""
    r2s = []
    for sd in VAL_SEEDS:
        cur = get_currents(make_sampler(dist_name, seed=sd)(p1, p2))
        r2s.append(linear_r2(intensities, (cur > theta).mean(axis=1)))
    a = np.array(r2s)
    return a.mean(), a.std(ddof=1)

def rate_stats(dist_name, p1, p2, theta):
    """Firing-rate response of a frozen solution."""
    rates = (get_currents(make_sampler(dist_name)(p1, p2)) > theta).mean(axis=1)
    dead  = intensities[rates < DEAD_LEVEL]
    return dict(rates=rates,
                span=float(rates.max() - rates.min()),
                rate_min=float(rates.min()),
                rate_max=float(rates.max()),
                dead_hi=float(dead.max()) if len(dead) else None)

def edge_position(val, grid):
    i = int(np.abs(grid - val).argmin())
    if i == 0:
        return "lower bound"
    if i == len(grid) - 1:
        return "upper bound"
    return f"interior ({i}/{len(grid)-1})"

# ══════════════════════════════════════════
# Run: bounded vs extended
# ══════════════════════════════════════════
rows, stats = [], {}
for grid_name, dists in GRID_SETS.items():
    for dist_name, cfg in dists.items():
        p1, p2, th, r2_in = run_coord_ascent(dist_name, cfg)
        oos_m, oos_s = out_of_sample(dist_name, p1, p2, th)
        st = rate_stats(dist_name, p1, p2, th)
        stats[(grid_name, dist_name)] = st
        rows.append(dict(grid=grid_name, distribution=dist_name,
                         p1=p1, p2=p2, theta=th,
                         R2_insample=r2_in, R2_oos_mean=oos_m, R2_oos_sd=oos_s,
                         span=st["span"], rate_max=st["rate_max"],
                         p1_position=edge_position(p1, cfg["p1_grid"]),
                         p2_position=edge_position(p2, cfg["p2_grid"])))
        l1, l2 = cfg["p1_label"], cfg["p2_label"]
        print(f"[{grid_name:8s}][{dist_name:8s}] "
              f"{l1}={p1:.4f} ({rows[-1]['p1_position']})  "
              f"{l2}={p2:.4f} ({rows[-1]['p2_position']})  "
              f"theta={th:.2f}  R2={r2_in:.5f}  "
              f"oos={oos_m:.5f}+/-{oos_s:.5f}  span={st['span']:.3f}")

res = pd.DataFrame(rows)
res.to_csv(OUT_ROOT / "boundary_check.csv", index=False)

# ══════════════════════════════════════════
# Summary 1: effect of extending the search range
# ══════════════════════════════════════════
print(f"\n{'='*70}")
print("Effect of extending the search range (bounded -> extended)")
print("-" * 70)
for dist_name in GRID_SETS["bounded"]:
    b = res[(res.grid == "bounded")  & (res.distribution == dist_name)].iloc[0]
    e = res[(res.grid == "extended") & (res.distribution == dist_name)].iloc[0]
    d_r2 = e.R2_insample - b.R2_insample
    d_th = (e.theta / b.theta - 1) * 100

    if abs(d_r2) < 1e-3:
        verdict = "identical effective solution"
    elif "bound" in e.p1_position or "bound" in e.p2_position:
        verdict = (f"optimum at grid edge; max firing rate capped at "
                   f"{e.rate_max:.2f} (span {e.span:.2f} vs {b.span:.2f} bounded)")
    elif abs(d_th) < 1.0:
        verdict = "theta effectively unchanged -> bounded operating point retained"
    else:
        verdict = "operating point shifts -> see manuscript Methods"

    print(f"{dist_name:>9s} | R2 {b.R2_insample:.4f} -> {e.R2_insample:.4f} "
          f"({d_r2/b.R2_insample*100:+.2f}%) | theta {d_th:+.2f}%")
    print(f"{'':>9s} | -> {verdict}")

# ══════════════════════════════════════════
# Summary 2: beta constraint check (alpha >= 1)
# Compares the firing-rate response of the beta optima obtained with and
# without the constraint. Both curves are computed at runtime.
# ══════════════════════════════════════════
bb, be = stats[("bounded", "Beta")], stats[("extended", "Beta")]
b_row = res[(res.grid == "bounded")  & (res.distribution == "Beta")].iloc[0]
e_row = res[(res.grid == "extended") & (res.distribution == "Beta")].iloc[0]

print(f"\n{'='*70}")
print("Beta constraint check (alpha >= 1): firing-rate response")
print("-" * 70)
for lab, row, st in [("bounded  (alpha >= 1)", b_row, bb),
                     ("extended (alpha < 1) ", e_row, be)]:
    dead = f"rate < {DEAD_LEVEL} up to intensity {st['dead_hi']:.0f}" \
           if st["dead_hi"] is not None else "no dead zone"
    print(f"{lab} | alpha={row.p1:.2f} beta={row.p2:.4f} theta={row.theta:.1f} | "
          f"rate {st['rate_min']:.4f} - {st['rate_max']:.4f} "
          f"(span {st['span']:.3f}) | {dead}")

if be["rate_max"] < 0.98:
    lost = (1 - be["rate_max"]) * 100
    print(f"\n-> Relaxing alpha >= 1 caps the maximum firing rate at "
          f"{be['rate_max']:.2f}: about {lost:.0f}% of the Vd samples remain "
          f"pinned at the low-bias rail and never cross theta at any tested "
          f"intensity, compressing the response range by "
          f"{(bb['span']-be['span'])/bb['span']*100:.0f}%.")
else:
    print("\n-> Extended beta optimum reaches the full firing-rate range; "
          "re-examine the constraint rationale.")

pd.DataFrame({"intensity": intensities,
              "rate_beta_bounded":  bb["rates"],
              "rate_beta_extended": be["rates"]}
             ).to_csv(OUT_ROOT / "rate_curves_beta.csv", index=False)
print(f"\nSaved: {OUT_ROOT / 'boundary_check.csv'}, {OUT_ROOT / 'rate_curves_beta.csv'}")