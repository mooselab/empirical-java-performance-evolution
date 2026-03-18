"""
Experiment 2: Instrumentation Overhead

Shows that JIB instrumentation preserves the performance change ratio
between optimized and regressed methods.
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results", "exp2")


def load_benchmarks(filepath):
    with open(filepath) as f:
        data = json.load(f)
    results = {}
    for entry in data:
        name = entry["benchmark"].split(".")[-1]
        results[name] = entry["primaryMetric"]["score"]
    return results


def main():
    no_agent_file = os.path.join(RESULTS_DIR, "no_agent.json")
    with_agent_file = os.path.join(RESULTS_DIR, "with_agent.json")

    for f in [no_agent_file, with_agent_file]:
        if not os.path.exists(f):
            print(
                f"ERROR: {f} not found. Run the experiment first.", file=sys.stderr)
            sys.exit(1)

    no_agent = load_benchmarks(no_agent_file)
    with_agent = load_benchmarks(with_agent_file)

    opt_key = "benchmarkOptimizedMethod"
    reg_key = "benchmarkRegressedMethod"

    for key in [opt_key, reg_key]:
        if key not in no_agent or key not in with_agent:
            print(
                f"ERROR: Missing benchmark '{key}' in results.", file=sys.stderr)
            sys.exit(1)

    ratio_no_agent = no_agent[reg_key] / no_agent[opt_key]
    ratio_with_agent = with_agent[reg_key] / with_agent[opt_key]
    ratio_diff_pct = abs(ratio_with_agent -
                         ratio_no_agent) / ratio_no_agent * 100

    print("=" * 65)
    print("Experiment 2: Instrumentation Overhead")
    print("=" * 65)
    print()

    print("Individual scores (ns/op):")
    print(f"  {'':30s} {'No Agent':>12s}  {'With Agent':>12s}")
    print(
        f"  {'Optimized':30s} {no_agent[opt_key]:12.2f}  {with_agent[opt_key]:12.2f}")
    print(
        f"  {'Regressed':30s} {no_agent[reg_key]:12.2f}  {with_agent[reg_key]:12.2f}")
    print()

    print("Performance change ratio (Regressed / Optimized):")
    print(f"  Without agent: {ratio_no_agent:.2f}x")
    print(f"  With agent:    {ratio_with_agent:.2f}x")
    print()

    print("-" * 65)
    print(f"Ratio difference: {ratio_diff_pct:.2f}%")
    print()


if __name__ == "__main__":
    main()
