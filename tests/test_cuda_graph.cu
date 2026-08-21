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
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/runtime/cuda_graph.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/stream.hpp"
#include "cumes_test_cuda_helper.cuh"
using namespace cumes::test;


__global__ void fillKernel(float* x, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ void addKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}


int main() {
    printf("=== CUDA Graph capture/launch correctness ===\n");

    // ---- gate 1: three-kernel DAG, graph == stream ----
    {
        const int n = 1024;
        float *a = nullptr, *b = nullptr, *c = nullptr, *c_stream = nullptr;
        check_cuda(cudaMalloc(&a, n * 4), "a"); check_cuda(cudaMalloc(&b, n * 4), "b");
        check_cuda(cudaMalloc(&c, n * 4), "c"); check_cuda(cudaMalloc(&c_stream, n * 4), "c_stream");
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
        check_cuda(cudaMemcpy(hc.data(), c, n * 4, cudaMemcpyDeviceToHost), "read c");
        check_cuda(cudaMemcpy(hc_stream.data(), c_stream, n * 4, cudaMemcpyDeviceToHost), "read c_stream");
        check(max_diff(hc.data(), hc_stream.data(), n) == 0.0,
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
        // Shared manufactured state (kGraphQuad in cumes_test_cuda_helper.cuh):
        // R_00=4.0, R_10=0.3s, m>=2: 0.1s^2, Rss=Rcc. The gate compares r_e
        // only, which is built from the R slots, so uploading the remaining
        // four families as deterministic zeros (instead of leaving the device
        // slab uninitialized) is an equivalent but strictly more reproducible
        // input.
        std::vector<double> h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs;
        manufactured_state<double>(ManufacturedShape::kGraphQuad, p.ns, p.mnmax,
                                  p.ntor, h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs);
        upload_state(storage, h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs, p.ns, p.mnmax);

    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<double> rs = realSpaceCreate(p);
        cumes::ToroidalFftOperator<double> op(p, rs, mt);
        cumes::Stream stream;
        op.bind_stream(stream.get());

        // stream reference
        op.inverse(storage.physical_const(), false, stream.get());
        stream.synchronize();
        const size_t nF = (size_t)p.ns * p.nZnT;
        std::vector<double> r_e_stream(nF);
        check_cuda(cudaMemcpy(r_e_stream.data(), rs.d_r_e, nF * 8, cudaMemcpyDeviceToHost), "read stream r_e");

        // graph capture + replay of the same inverse transform
        bool cufft_graph_ok = false;
        std::string err;
        try {
            auto g = cumes::CudaGraph::capture(stream.get(), [&]() {
                op.inverse(storage.physical_const(), false, stream.get());
            });
            g.launch(stream.get());
            stream.synchronize();
            std::vector<double> r_e_graph(nF);
            check_cuda(cudaMemcpy(r_e_graph.data(), rs.d_r_e, nF * 8, cudaMemcpyDeviceToHost), "read graph r_e");
            const double md = max_diff(r_e_graph.data(), r_e_stream.data(), (int)nF);
            cufft_graph_ok = (md == 0.0);
            printf("  cuFFT-in-graph: max |diff| = %.3e\n", md);
        } catch (const std::exception& e) {
            err = e.what();
        }
        check(cufft_graph_ok, "cuFFT inverse transform graph == stream (bitwise)");
        if (!cufft_graph_ok) printf("  cuFFT capture error: %s\n", err.c_str());

        realSpaceFree(rs);
    cumes::modeTableFree(mt);
    }

    return summary();
}
