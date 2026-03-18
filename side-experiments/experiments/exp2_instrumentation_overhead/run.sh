#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/exp2"
BENCHMARKS_JAR="$PROJECT_ROOT/target/benchmarks.jar"
JIB_JAR="$PROJECT_ROOT/jib.jar"
CONFIG_YAML="$PROJECT_ROOT/config.yaml"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$BENCHMARKS_JAR" ]; then
    echo "benchmarks.jar not found. Building project..."
    mvn -f "$PROJECT_ROOT/pom.xml" clean package -q
fi

if [ ! -f "$JIB_JAR" ]; then
    echo "ERROR: jib.jar not found at $JIB_JAR"
    exit 1
fi

echo "=== Experiment 2: Instrumentation Overhead ==="
echo ""

BH_OPT="-Djmh.blackhole.autoDetect=false"

echo "[Run A] Without instrumentation agent..."
java "$BH_OPT" -jar "$BENCHMARKS_JAR" \
    "benchmarkOptimized|benchmarkRegressed" \
    -f 3 -wi 3 -i 5 \
    -rf json -rff "$RESULTS_DIR/no_agent.json"

echo ""
echo "[Run B] With JIB instrumentation agent..."
java "$BH_OPT" -javaagent:"$JIB_JAR=config=$CONFIG_YAML" -jar "$BENCHMARKS_JAR" \
    "benchmarkOptimized|benchmarkRegressed" \
    -f 3 -wi 3 -i 5 \
    -rf json -rff "$RESULTS_DIR/with_agent.json"

echo ""
echo "Results saved to $RESULTS_DIR/"
echo "Run 'python3 $SCRIPT_DIR/analyze.py' to analyze."
