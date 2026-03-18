package com.example;

import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(3)
public class PerformanceBenchmark {

    private PerformanceComparison performanceComparison;

    @Param({"1000"})
    private int iterations;

    @Setup
    public void setup() {
        performanceComparison = new PerformanceComparison();
    }

    @Benchmark
    public String benchmarkOptimizedMethod() {
        return performanceComparison.optimizedMethod(iterations);
    }

    @Benchmark
    public String benchmarkRegressedMethod() {
        return performanceComparison.regressedMethod(iterations);
    }

    @Benchmark
    public String benchmarkMethodWithShortLiteral() {
        return performanceComparison.methodWithShortLiteral(iterations);
    }

    @Benchmark
    public String benchmarkMethodWithLongLiteral() {
        return performanceComparison.methodWithLongLiteral(iterations);
    }

    @Benchmark
    public String benchmarkMethodWithExtraLongLiteral() {
        return performanceComparison.methodWithExtraLongLiteral(iterations);
    }

}
