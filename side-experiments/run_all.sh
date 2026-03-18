#!/usr/bin/env bash
set -euo pipefail

VERBOSE=false

for arg in "$@"; do
  case "$arg" in
    -v|--verbose)
      VERBOSE=true
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "JIB Performance Experiments"
echo "================================"
echo ""

echo "Building project..."
mvn -f "$PROJECT_ROOT/pom.xml" clean package -q
echo "Build complete."
echo ""

mkdir -p "$PROJECT_ROOT/results/exp1" \
         "$PROJECT_ROOT/results/exp2" \
         "$PROJECT_ROOT/results/exp3"

echo "Experiment 1: Warmup Stabilization"
echo "--------------------------------"
if [ "$VERBOSE" = true ]; then
  bash "$PROJECT_ROOT/experiments/exp1_warmup_stabilization/run.sh"
else
  bash "$PROJECT_ROOT/experiments/exp1_warmup_stabilization/run.sh" >/dev/null 2>&1
fi
echo ""

echo "Experiment 2: Instrumentation Overhead"
echo "--------------------------------"
if [ "$VERBOSE" = true ]; then
  bash "$PROJECT_ROOT/experiments/exp2_instrumentation_overhead/run.sh"
else
  bash "$PROJECT_ROOT/experiments/exp2_instrumentation_overhead/run.sh" >/dev/null 2>&1
fi
echo ""

echo "Experiment 3: String Literal Length"
echo "--------------------------------"
if [ "$VERBOSE" = true ]; then
  bash "$PROJECT_ROOT/experiments/exp3_string_literal_length/run.sh"
else
  bash "$PROJECT_ROOT/experiments/exp3_string_literal_length/run.sh" >/dev/null 2>&1
fi
echo ""

echo "Analyzing results..."
python3 "$PROJECT_ROOT/experiments/exp1_warmup_stabilization/analyze.py"
python3 "$PROJECT_ROOT/experiments/exp2_instrumentation_overhead/analyze.py"
python3 "$PROJECT_ROOT/experiments/exp3_string_literal_length/analyze.py"

echo "Done."
