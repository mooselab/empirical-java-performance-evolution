"""
Experiment 3: String Literal Length

Shows that string literal length does not meaningfully affect
benchmark performance -- literals are interned and their size
has no runtime cost at the point of return.
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results", "exp3")

DISPLAY_NAMES = {
    "benchmarkMethodWithShortLiteral": "Short",
    "benchmarkMethodWithLongLiteral": "Long",
    "benchmarkMethodWithExtraLongLiteral": "ExtraLong",
}

ORDER = [
    "benchmarkMethodWithShortLiteral",
    "benchmarkMethodWithLongLiteral",
    "benchmarkMethodWithExtraLongLiteral",
]


def load_benchmarks(filepath):
    with open(filepath) as f:
        data = json.load(f)
    results = {}
    for entry in data:
        name = entry["benchmark"].split(".")[-1]
        raw = [v for fork in entry["primaryMetric"]["rawData"] for v in fork]
        mean = sum(raw) / len(raw)
        variance = sum((x - mean) ** 2 for x in raw) / (len(raw) - 1)
        std = math.sqrt(variance)
        results[name] = {
            "mean": mean,
            "std": std,
            "score": entry["primaryMetric"]["score"],
            "error": entry["primaryMetric"]["scoreError"],
        }
    return results


def main():
    literals_file = os.path.join(RESULTS_DIR, "literals.json")

    if not os.path.exists(literals_file):
        print(
            f"ERROR: {literals_file} not found. Run the experiment first.", file=sys.stderr)
        sys.exit(1)

    benchmarks = load_benchmarks(literals_file)

    for key in ORDER:
        if key not in benchmarks:
            print(
                f"ERROR: Missing benchmark '{key}' in results.", file=sys.stderr)
            sys.exit(1)

    print("=" * 65)
    print("Experiment 3: String Literal Length")
    print("=" * 65)
    print()

    print(f"  {'Literal':12s} {'Mean (ns/op)':>14s} {'Std':>10s} {'Error':>10s}")
    print(f"  {'-'*12} {'-'*14} {'-'*10} {'-'*10}")
    for key in ORDER:
        b = benchmarks[key]
        label = DISPLAY_NAMES[key]
        print(
            f"  {label:12s} {b['mean']:14.2f} {b['std']:10.2f} {b['error']:10.2f}")

    print()
    print("Pairwise mean differences:")

    for i in range(len(ORDER)):
        for j in range(i + 1, len(ORDER)):
            a_key, b_key = ORDER[i], ORDER[j]
            a_mean = benchmarks[a_key]["mean"]
            b_mean = benchmarks[b_key]["mean"]
            pct = abs(b_mean - a_mean) / a_mean * 100
            print(f"  {DISPLAY_NAMES[a_key]:10s} vs {DISPLAY_NAMES[b_key]:10s}: "
                  f"{pct:.2f}% difference")

    print()
    print("-" * 65)

    means = [benchmarks[k]["mean"] for k in ORDER]
    overall_mean = sum(means) / len(means)
    max_dev = max(abs(m - overall_mean) / overall_mean * 100 for m in means)

    print(f"Max deviation from overall mean: {max_dev:.2f}%")
    print()


if __name__ == "__main__":
    main()
