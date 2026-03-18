#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/exp3"
BENCHMARKS_JAR="$PROJECT_ROOT/target/benchmarks.jar"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$BENCHMARKS_JAR" ]; then
    echo "benchmarks.jar not found. Building project..."
    mvn -f "$PROJECT_ROOT/pom.xml" clean package -q
fi

echo "=== Experiment 3: String Literal Length ==="
echo ""

echo "[Run] Short, Long, and ExtraLong literal benchmarks..."
java -jar "$BENCHMARKS_JAR" \
    "benchmarkMethodWith" \
    -f 3 -wi 3 -i 5 \
    -rf json -rff "$RESULTS_DIR/literals.json"

echo ""
echo "Results saved to $RESULTS_DIR/"
echo "Run 'python3 $SCRIPT_DIR/analyze.py' to analyze."
