"""Check complete optimization CSV results against the README's displayed values.

Run the four optimization scripts first, then run this file from the repository
root. Comparisons use the published decimal precision, not bitwise equality
across NumPy/BLAS versions. No optimizer settings or result files are changed.
Boundary checks validate completion and agreement with the bounded search;
the README supplies no numerical reference for the extended search.
"""

import argparse
import csv
import importlib.metadata
import json
import math
import statistics
import sys
from pathlib import Path


def read_rows(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty result file: {path}")
    return rows


def verify_results(results_dir):
    """Return named checks; a false `passed` value means a reproduction mismatch."""
    checks = []

    def check(name, actual, expected):
        checks.append(dict(check=name, actual=actual, expected=expected,
                           passed=actual == expected))

    expected_distributions = {
        "Beta": "0.96310 4925.0 1.0000 3.2105",
        "Logistic": "0.96141 4574.4 0.1400 0.1400",
        "Uniform": "0.94655 4854.1 0.1737 0.2395",
    }
    final = {}
    for name, expected in expected_distributions.items():
        rows = read_rows(results_dir / f"history_{name.lower()}.csv")
        final[name] = row = [r for r in rows if r["stage"] == "stage2"][-1]
        actual = (f"{float(row['R2']):.5f} {float(row['theta']):.1f} "
                  f"{float(row['p1']):.4f} {float(row['p2']):.4f}")
        check(f"vd_optimization/{name}: R2 theta p1 p2", actual, expected)

    seeds = read_rows(results_dir / "seed_validation.csv")
    check("seed_validation/seeds", sorted(int(row["seed"]) for row in seeds), list(range(100)))
    r2 = [float(row["R2"]) for row in seeds]
    check("seed_validation/mean +/- sample sd",
          f"{statistics.mean(r2):.4f} +/- {statistics.stdev(r2):.4f}", "0.9629 +/- 0.0011")
    seed42 = next(row for row in seeds if int(row["seed"]) == 42)
    check("seed_validation/seed 42", f"{float(seed42['R2']):.5f}", "0.96141")

    expected_adaptive = {
        ("dim", "Full"): "0.21 0.14 2816.3 0.9567",
        ("dim", "theta-only"): "0.14 0.14 2750.6 0.9371",
        ("dim", "Fixed"): "0.14 0.14 4574.4 0.8465",
        ("normal", "Full"): "0.14 0.14 4574.4 0.9619",
        ("bright", "Full"): "0.15 0.11 5452.7 0.9751",
        ("bright", "theta-only"): "0.14 0.14 5459.4 0.9717",
        ("bright", "Fixed"): "0.14 0.14 4574.4 0.9208",
    }
    adaptive = read_rows(results_dir / "adaptive_threshold.csv")
    keys = [(row["condition"], row["mode"]) for row in adaptive]
    check("adaptive_threshold/complete conditions", sorted(keys), sorted(expected_adaptive))
    by_key = dict(zip(keys, adaptive))
    for key, expected in expected_adaptive.items():
        row = by_key[key]
        actual = (f"{float(row['mu']):.2f} {float(row['sigma']):.2f} "
                  f"{float(row['theta']):.1f} {float(row['R2']):.4f}")
        check(f"adaptive_threshold/{key[0]}/{key[1]}: mu sigma theta R2", actual, expected)

    boundary = read_rows(results_dir / "boundary_check.csv")
    keys = [(row["grid"], row["distribution"]) for row in boundary]
    expected_keys = [(grid, name) for grid in ("bounded", "extended") for name in final]
    check("boundary_check/complete grids", sorted(keys), sorted(expected_keys))
    by_key = dict(zip(keys, boundary))
    for name in final:
        row = by_key[("bounded", name)]
        agrees = all(math.isclose(float(row[boundary_key]), float(final[name][vd_key]),
                                 rel_tol=1e-12, abs_tol=1e-12)
                     for boundary_key, vd_key in (("p1", "p1"), ("p2", "p2"),
                                                  ("theta", "theta"), ("R2_insample", "R2")))
        check(f"boundary_check/{name}/bounded agrees with vd_optimization", agrees, True)

    curves = read_rows(results_dir / "rate_curves_beta.csv")
    check("boundary_check/beta curve rows", len(curves), 26)
    check("boundary_check/beta rates in [0, 1]",
          all(0 <= float(row[key]) <= 1 for row in curves
              for key in ("rate_beta_bounded", "rate_beta_extended")), True)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    try:
        checks = verify_results(args.results_dir)
    except (OSError, ValueError, KeyError, IndexError, StopIteration) as error:
        parser.exit(1, f"Incomplete or invalid optimization results: {error}\n")
    report = {
        "passed": all(item["passed"] for item in checks),
        "verification_environment": {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
        },
        "comparison": "README display precision; bounded-search consistency at 1e-12",
        "checks": checks,
    }
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'FAIL'}: {item['check']}")
        if not item["passed"]:
            print(f"  expected: {item['expected']}\n  actual:   {item['actual']}")
    path = args.results_dir / "verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{sum(item['passed'] for item in checks)}/{len(checks)} checks passed. Report: {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
