"""
Experiment 1: Warmup Stabilization

Shows that running without JMH warmup (-wi 0) and discarding the first
measurement iteration produces results similar to a proper warmup run.
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results", "exp1")


def load_raw_data(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data[0]["primaryMetric"]["rawData"]


def flatten(nested):
    return [v for fork in nested for v in fork]


def stats(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    cv = (std / mean) * 100 if mean != 0 else 0
    return {"mean": mean, "std": std, "cv": cv, "n": n}


def print_stats(label, s):
    print(f"  {label}:")
    print(f"    N = {s['n']}, Mean = {s['mean']:.2f} ns/op, "
          f"Std = {s['std']:.2f}, CV = {s['cv']:.2f}%")


def main():
    warmup_file = os.path.join(RESULTS_DIR, "with_warmup.json")
    no_warmup_file = os.path.join(RESULTS_DIR, "no_warmup.json")

    for f in [warmup_file, no_warmup_file]:
        if not os.path.exists(f):
            print(
                f"ERROR: {f} not found. Run the experiment first.", file=sys.stderr)
            sys.exit(1)

    warmup_raw = load_raw_data(warmup_file)
    no_warmup_raw = load_raw_data(no_warmup_file)

    warmup_all = flatten(warmup_raw)
    no_warmup_all = flatten(no_warmup_raw)
    no_warmup_trimmed = flatten([fork[1:] for fork in no_warmup_raw])

    s_warmup = stats(warmup_all)
    s_no_warmup = stats(no_warmup_all)
    s_trimmed = stats(no_warmup_trimmed)

    print("=" * 65)
    print("Experiment 1: Warmup Stabilization")
    print("=" * 65)
    print()

    print_stats("With warmup (3 wi, all 5 iterations)", s_warmup)
    print()
    print_stats("No warmup (all 5 iterations, raw)", s_no_warmup)
    print()
    print_stats("No warmup (first iteration removed, 4 remaining)", s_trimmed)
    print()

    pct_diff_raw = abs(s_no_warmup["mean"] -
                       s_warmup["mean"]) / s_warmup["mean"] * 100
    pct_diff_trimmed = abs(
        s_trimmed["mean"] - s_warmup["mean"]) / s_warmup["mean"] * 100

    print("-" * 65)
    print("Comparison to warmup baseline:")
    print(
        f"  Raw no-warmup vs warmup:     {pct_diff_raw:.2f}% difference in mean")
    print(
        f"  Trimmed no-warmup vs warmup: {pct_diff_trimmed:.2f}% difference in mean")
    print()
    print(f"  Raw no-warmup CV:     {s_no_warmup['cv']:.2f}%")
    print(f"  Trimmed no-warmup CV: {s_trimmed['cv']:.2f}%")
    print(f"  Warmup CV:            {s_warmup['cv']:.2f}%")
    print()


if __name__ == "__main__":
    main()
