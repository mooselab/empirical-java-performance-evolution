# Performance Analysis Side Experiments

## Running

Run everything (build → run experiments → analyze):

```bash
./run_all.sh
```

Raw outputs are written to:

- `results/exp1/with_warmup.json`, `results/exp1/no_warmup.json`
- `results/exp2/no_agent.json`, `results/exp2/with_agent.json`
- `results/exp3/literals.json`

## Experiments

### Experiment 1: Warmup Stabilization

- **What**: Compare a normal JMH run (with warmup) vs a run with `-wi 0` (no warmup), and also a “trimmed” version of the no-warmup run with the first measurement iteration removed.
- **How**: Runs `benchmarkOptimizedMethod` twice:
  - With warmup: `-wi 3 -i 5 -f 3` → `results/exp1/with_warmup.json`
  - No warmup: `-wi 0 -i 5 -f 3` → `results/exp1/no_warmup.json`
  - Analysis flattens raw per-iteration data and computes mean/std/CV; also recomputes stats after dropping iteration 1 from each fork.
- **Result (ns/op)**:
  - With warmup: **Mean 4465.98**, **CV 0.45%** (N=15)
  - No warmup (raw): **Mean 4083.53**, **CV 2.67%** (N=15)
  - No warmup (drop first iteration): **Mean 4032.65**, **CV 0.79%** (N=12)
  - Mean difference vs warmup baseline: **8.56% (raw)** / **9.70% (trimmed)**

### Experiment 2: Instrumentation Overhead

- **What**: Check whether JIB instrumentation changes the *ratio* between an “optimized” and “regressed” benchmark (i.e., does it preserve relative differences?).
- **How**: Runs two benchmarks (`benchmarkOptimizedMethod` and `benchmarkRegressedMethod`) twice:
  - Without agent: plain `java -jar ...` → `results/exp2/no_agent.json`
  - With agent: `java -javaagent:jib.jar=config=config.yaml -jar ...` → `results/exp2/with_agent.json`
  - Analysis computes \(\text{ratio} = \text{regressed}/\text{optimized}\) for both runs and compares.
- **Result**:
  - Scores (ns/op):
    - Optimized: **4497.13 (no agent)** vs **4722.65 (with agent)**
    - Regressed: **109246.36 (no agent)** vs **113578.88 (with agent)**
  - Ratio (Regressed / Optimized): **24.29x (no agent)** vs **24.05x (with agent)**
  - Ratio difference: **~1.00%**

### Experiment 3: String Literal Length

- **What**: Check whether returning a short vs long vs extra-long string literal measurably affects benchmark performance.
- **How**: Runs three benchmarks matching `benchmarkMethodWith*` with standard JMH warmup/measurement (`-wi 3 -i 5 -f 3`) and compares mean raw times and pairwise differences.
- **Result**:
  - Means (ns/op): **Short 0.39**, **Long 0.38**, **ExtraLong 0.38**
  - Pairwise mean differences: **0.92%**, **2.14%**, **1.23%**
  - Max deviation from overall mean: **1.13%**
