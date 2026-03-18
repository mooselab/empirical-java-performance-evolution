#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/exp1"
BENCHMARKS_JAR="$PROJECT_ROOT/target/benchmarks.jar"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$BENCHMARKS_JAR" ]; then
    echo "benchmarks.jar not found. Building project..."
    mvn -f "$PROJECT_ROOT/pom.xml" clean package -q
fi

echo "=== Experiment 1: Warmup Stabilization ==="
echo ""

echo "[Run A] With warmup (wi=3, i=5, f=3)..."
java -jar "$BENCHMARKS_JAR" \
    "benchmarkOptimizedMethod" \
    -f 3 -wi 3 -i 5 \
    -rf json -rff "$RESULTS_DIR/with_warmup.json"

echo ""
echo "[Run B] Without warmup (wi=0, i=5, f=3)..."
java -jar "$BENCHMARKS_JAR" \
    "benchmarkOptimizedMethod" \
    -f 3 -wi 0 -i 5 \
    -rf json -rff "$RESULTS_DIR/no_warmup.json"

echo ""
echo "Results saved to $RESULTS_DIR/"
echo "Run 'python3 $SCRIPT_DIR/analyze.py' to analyze."
