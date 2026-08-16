// test_cuda_graph.cu — CUDA Graph capture/launch correctness (blueprint §8.11).
//
// Two gates for the Phase-9 graph primitive:
//   1. A three-kernel DAG captured into a graph replays the same output as
//      direct stream execution (the basic capture/instantiate/launch contract).
//   2. A cuFFT inverse transform captured into a graph replays the same real
//      geometry as direct stream execution — this is the empirical check that
//      cuFFT-in-graph works on the current stack (CUDA 12.1, sm_61) with the
//      Phase-6B explicit work area. If cuFFT cannot be captured, gate 2 reports
//      the CUDA error rather than silently skipping, so the ADR records it.
#include <cstdio>
#include <cmath>
#include <vector>

#include "vmec_types.h"
#include "fourier.cuh"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/runtime/cuda_graph.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/stream.hpp"
#include "cumes_test_support.cuh"

static int failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (cond) {                                                          \
            printf("PASS %s\n", msg);                                        \
        } else {                                                             \
            printf("FAIL %s\n", msg);                                        \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

__global__ void fillKernel(float* x, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ void addKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

template <typename T>
static double maxDiff(const T* x, const T* y, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
    return m;
}

int main() {
    printf("=== CUDA Graph capture/launch correctness ===\n");

    // ---- gate 1: three-kernel DAG, graph == stream ----
    {
        const int n = 1024;
        float *a = nullptr, *b = nullptr, *c = nullptr, *c_stream = nullptr;
        checkCuda(cudaMalloc(&a, n * 4), "a"); checkCuda(cudaMalloc(&b, n * 4), "b");
        checkCuda(cudaMalloc(&c, n * 4), "c"); checkCuda(cudaMalloc(&c_stream, n * 4), "c_stream");
        cumes::Stream stream;

        fillKernel<<<8, 128, 0, stream.get()>>>(a, n, 1.0f);
        fillKernel<<<8, 128, 0, stream.get()>>>(b, n, 2.0f);
        addKernel<<<8, 128, 0, stream.get()>>>(a, b, c_stream, n);
        stream.synchronize();

        auto g = cumes::CudaGraph::capture(stream.get(), [&]() {
            fillKernel<<<8, 128, 0, stream.get()>>>(a, n, 1.0f);
            fillKernel<<<8, 128, 0, stream.get()>>>(b, n, 2.0f);
            addKernel<<<8, 128, 0, stream.get()>>>(a, b, c, n);
        });
        g.launch(stream.get());
        stream.synchronize();

        std::vector<float> hc(n), hc_stream(n);
        checkCuda(cudaMemcpy(hc.data(), c, n * 4, cudaMemcpyDeviceToHost), "read c");
        checkCuda(cudaMemcpy(hc_stream.data(), c_stream, n * 4, cudaMemcpyDeviceToHost), "read c_stream");
        CHECK(maxDiff(hc.data(), hc_stream.data(), n) == 0.0,
              "three-kernel graph == stream (bitwise)");

        cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(c_stream);
    }

    // ---- gate 2: cuFFT inverse transform, graph == stream ----
    {
        DeviceParams<double> p;
        p.ns = 5; p.mnmax = 4; p.mpol = 4; p.ntor = 0;
        p.ntheta = 18; p.nzeta = 1; p.nfp = 1; p.nZnT = 18;
        p.ncurr = 0; p.delt = 0.9; p.ftol = 1e-14; p.max_iter = 5;
        p.tcon0 = 1.0; p.lamscale = 0.0;

        cumes::SpectralStorage<double> storage(p.ns, p.mnmax);
        SpectralState<double> st = storage.legacy_view();
        const size_t nS = (size_t)p.ns * p.mnmax;
        auto* h_cc = new double[nS]();
        auto* h_ss = new double[nS]();
        for (int j = 0; j < p.ns; ++j) {
            double s = double(j) / double(p.ns - 1);
            for (int mode = 0; mode < p.mnmax; ++mode) {
                int m = mode;
                if (m == 0) h_cc[j + mode * p.ns] = 4.0;
                else if (m == 1) h_cc[j + mode * p.ns] = 0.3 * s;
                else h_cc[j + mode * p.ns] = 0.1 * s * s;
                h_ss[j + mode * p.ns] = h_cc[j + mode * p.ns];
            }
        }
        checkCuda(cudaMemcpy(st.d_rmncc, h_cc, nS * 8, cudaMemcpyHostToDevice), "cc");
        checkCuda(cudaMemcpy(st.d_rmnss, h_ss, nS * 8, cudaMemcpyHostToDevice), "ss");
        delete[] h_cc; delete[] h_ss;

        FourierPlan<double> fp = fourierCreate(p);
    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<double> rs = realSpaceCreate(p);
        cumes::Stream stream;
        cumes::check_cufft(cufftSetStream(fp.plan_z2d, stream.get()), "cufft set z2d stream");
        cumes::check_cufft(cufftSetStream(fp.plan_d2z, stream.get()), "cufft set d2z stream");

        // stream reference
        inverseDFT(fp, rs, storage.physical_const(), p, mt.d_xm, mt.d_xn, false, stream.get());
        stream.synchronize();
        const size_t nF = (size_t)p.ns * p.nZnT;
        std::vector<double> r_e_stream(nF);
        checkCuda(cudaMemcpy(r_e_stream.data(), rs.d_r_e, nF * 8, cudaMemcpyDeviceToHost), "read stream r_e");

        // graph capture + replay of the same inverse transform
        bool cufft_graph_ok = false;
        std::string err;
        try {
            auto g = cumes::CudaGraph::capture(stream.get(), [&]() {
                inverseDFT(fp, rs, storage.physical_const(), p, mt.d_xm, mt.d_xn, false, stream.get());
            });
            g.launch(stream.get());
            stream.synchronize();
            std::vector<double> r_e_graph(nF);
            checkCuda(cudaMemcpy(r_e_graph.data(), rs.d_r_e, nF * 8, cudaMemcpyDeviceToHost), "read graph r_e");
            const double md = maxDiff(r_e_graph.data(), r_e_stream.data(), (int)nF);
            cufft_graph_ok = (md == 0.0);
            printf("  cuFFT-in-graph: max |diff| = %.3e\n", md);
        } catch (const std::exception& e) {
            err = e.what();
        }
        CHECK(cufft_graph_ok, "cuFFT inverse transform graph == stream (bitwise)");
        if (!cufft_graph_ok) printf("  cuFFT capture error: %s\n", err.c_str());

        realSpaceFree(rs);
    cumes::modeTableFree(mt);
    fourierFree(fp);
    }

    if (failures == 0) {
        printf("\ntest_cuda_graph: ALL PASS\n");
        return 0;
    }
    printf("\ntest_cuda_graph: %d FAILURES\n", failures);
    return 1;
}
