// graph_overhead.cu — CUDA Graph submission-overhead microbenchmark (§8.11).
//
// Measures the host-side submission cost that a CUDA Graph can remove: the time
// to enqueue K kernels on a stream vs the time to launch one graph containing K
// kernels. It uses empty kernels, which have zero GPU work, so the measured
// savings are an UPPER BOUND — real kernels overlap their submission with GPU
// execution and hide part of it. The one-time capture + instantiate cost is
// reported separately (the blueprint §8.1 field: "graph instantiation, update,
// and rebuild cost at every multigrid stage").
//
// The solver enqueues roughly 24 submissions per evaluated pass (extrapolateAxis,
// inverseDFTFused + cuFFT, computeGeometry, computeJacobianStats, computeForces,
// constraintCompute, forwardDFT + cuFFT, scalxc, m1Constraint, two residual
// reductions, m1PreconScale, preconApply, the control copy, …), so K=24 is the
// representative count. Combine the numbers here with cumes_benchmark_fixed_
// iteration's µs/iter to decide "select by measured shape".
//
// Usage: cumes_benchmark_graph_overhead [--kernels 24] [--reps 1000]
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "cumes/runtime/cuda_graph.hpp"
#include "cumes/runtime/stream.hpp"
#include "cumes_test_support.cuh"

__global__ void emptyKernel() {}

using Clock = std::chrono::steady_clock;

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    int k = 24;     // kernels per "pass"
    int reps = 200; // timed launches
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (a == std::string("--") + name) {
                if (i + 1 >= argc) { fprintf(stderr, "graph_overhead: --%s needs a value\n", name); exit(2); }
                return argv[++i];
            }
            std::string pfx = std::string("--") + name + "=";
            if (a.compare(0, pfx.size(), pfx) == 0) return argv[i] + pfx.size();
            return nullptr;
        };
        if (const char* v = need("kernels")) k = atoi(v);
        else if (const char* v = need("reps")) reps = atoi(v);
        else { fprintf(stderr, "graph_overhead: unknown option '%s'\n", a.c_str()); return 2; }
    }

    cumes::Stream stream;

    // ---- one-time capture + instantiate cost ----
    double capture_us = 0.0;
    {
        auto t0 = Clock::now();
        auto g = cumes::CudaGraph::capture(stream.get(), [&]() {
            for (int i = 0; i < k; ++i) emptyKernel<<<1, 32, 0, stream.get()>>>();
        });
        auto t1 = Clock::now();
        capture_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    // ---- enqueue K kernels per pass, host wall time (no sync) ----
    std::vector<double> enqueue_us;
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        for (int i = 0; i < k; ++i) emptyKernel<<<1, 32, 0, stream.get()>>>();
        auto t1 = Clock::now();
        enqueue_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    // ---- launch one graph per pass, host wall time (no sync) ----
    auto g = cumes::CudaGraph::capture(stream.get(), [&]() {
        for (int i = 0; i < k; ++i) emptyKernel<<<1, 32, 0, stream.get()>>>();
    });
    std::vector<double> launch_us;
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        g.launch(stream.get());
        auto t1 = Clock::now();
        launch_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    stream.synchronize();

    const double enc = median(enqueue_us), lch = median(launch_us);
    const double per_kernel = enc / (k > 0 ? k : 1);
    printf("kernels_per_pass        : %d\n", k);
    printf("capture+instantiate_us  : %.2f (one-time per graph variant)\n", capture_us);
    printf("enqueue_us_per_pass     : %.2f (median, %d reps)\n", enc, reps);
    printf("graph_launch_us_per_pass: %.2f (median, %d reps)\n", lch, reps);
    printf("submission_savings_us   : %.2f (upper bound, empty kernels)\n", enc - lch);
    printf("enqueue_us_per_kernel   : %.3f\n", per_kernel);
    return 0;
}
