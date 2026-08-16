// test_runtime.cu — Phase 3 runtime RAII, typed views, and contiguous slabs.
//
// Exercises the new host-only CUDA runtime layer (DeviceBuffer/PinnedBuffer/
// Stream/Event/DeviceContext, centralized check_cuda/check_cufft), the typed
// SpectralView (device round-trip at the component-major layout), and the
// SpectralStorage contiguous state/velocity slabs — in particular that
// family_ptr() reproduces the exact six-family layout the legacy code read.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "vmec_types.h"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes/runtime/pinned_buffer.hpp"
#include "cumes/runtime/stream.hpp"
#include "cumes/runtime/event.hpp"
#include "cumes/runtime/device_context.hpp"
#include "cumes/core/tensor_view.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes_test_support.cuh"

// The two component enums must agree so the slab, the host snapshot, and the
// forward-DFT residual layout all share one component order.
static_assert(static_cast<int>(cumes::SpectralComponent::Rcc) ==
                  cumes::EquilibriumSnapshot::kRmncc,
              "SpectralComponent/EquilibriumSnapshot component order");
static_assert(static_cast<int>(cumes::SpectralComponent::Zsc) ==
                  cumes::EquilibriumSnapshot::kZmnsc, "order");
static_assert(static_cast<int>(cumes::SpectralComponent::Lsc) ==
                  cumes::EquilibriumSnapshot::kLmnsc, "order");
static_assert(static_cast<int>(cumes::SpectralComponent::Rss) ==
                  cumes::EquilibriumSnapshot::kRmnss, "order");
static_assert(static_cast<int>(cumes::SpectralComponent::Zcs) ==
                  cumes::EquilibriumSnapshot::kZmncs, "order");
static_assert(static_cast<int>(cumes::SpectralComponent::Lcs) ==
                  cumes::EquilibriumSnapshot::kLmncs, "order");

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

// Write through a SpectralView on device; host verifies the component-major
// [component][mode][surface] layout.
__global__ void writeSpectral(
    cumes::SpectralView<double, cumes::PhysicalStateDomain> v, int mnmax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = mnmax * v.ns();
    if (i >= total) return;
    int mode = i / v.ns();
    int surf = i % v.ns();
    for (int c = 0; c < cumes::kSpectralComponentCount; ++c) {
        v(static_cast<cumes::SpectralComponent>(c), mode, surf) =
            (double)(c * total + i);
    }
}

int main() {
    printf("=== runtime RAII / views / slabs ===\n");

    // ---- DeviceBuffer: alloc, zero, copy, move ----
    {
        cumes::DeviceBuffer<int> a(16);
        CHECK(a.size() == 16 && a.data() != nullptr, "DeviceBuffer allocates");
        a.zero();
        int h[16];
        for (int i = 0; i < 16; ++i) h[i] = 1;
        cc(cudaMemcpy(a.data(), h, 16 * sizeof(int), cudaMemcpyHostToDevice),
           "seed");
        cumes::DeviceBuffer<int> b(16);
        b.copy_from(a);
        int hb[16];
        cc(cudaMemcpy(hb, b.data(), 16 * sizeof(int), cudaMemcpyDeviceToHost),
           "read b");
        bool ok = true;
        for (int i = 0; i < 16; ++i) ok = ok && hb[i] == 1;
        CHECK(ok, "DeviceBuffer copy_from");
        cumes::DeviceBuffer<int> c(std::move(a));
        CHECK(a.data() == nullptr && c.size() == 16,
              "DeviceBuffer move nulls source");
    }

    // ---- PinnedBuffer ----
    {
        cumes::PinnedBuffer<double> p(4);
        CHECK(p.size() == 4 && p.data() != nullptr, "PinnedBuffer allocates");
        p.data()[0] = 1.0;
        p.data()[1] = 2.0;
        p.data()[2] = 3.0;
        p.data()[3] = 4.0;
        CHECK(p.data()[3] == 4.0, "PinnedBuffer writable from host");
    }

    // ---- Stream / Event / DeviceContext ----
    {
        cumes::DeviceContext ctx;
        CHECK(ctx.compute_stream() != nullptr &&
                  ctx.auxiliary_stream() != nullptr,
              "DeviceContext creates compute+aux streams");
        CHECK(ctx.capabilities().device >= 0, "DeviceContext reports a device");
        cumes::Stream s;
        CHECK(s.get() != nullptr, "Stream creates a cudaStream_t");
        cumes::Event e;
        CHECK(e.get() != nullptr, "Event creates a cudaEvent_t");
        e.record(s.get());
        e.synchronize();
    }

    // ---- SpectralStorage layout: slab offsets match the legacy 6-family order
    {
        const int ns = 7, mnmax = 4;
        cumes::SpectralStorage<double> st(ns, mnmax);
        const size_t one = (size_t)ns * mnmax;
        double* slab = st.state_slab();
        double* vslab = st.velocity_slab();
        CHECK(st.family_ptr(cumes::SpectralComponent::Rcc) == slab + 0 * one, "family_ptr rmncc at slab+0");
        CHECK(st.family_ptr(cumes::SpectralComponent::Zsc) == slab + 1 * one, "family_ptr zmnsc at slab+1");
        CHECK(st.family_ptr(cumes::SpectralComponent::Lsc) == slab + 2 * one, "family_ptr lmnsc at slab+2");
        CHECK(st.family_ptr(cumes::SpectralComponent::Rss) == slab + 3 * one, "family_ptr rmnss at slab+3");
        CHECK(st.family_ptr(cumes::SpectralComponent::Zcs) == slab + 4 * one, "family_ptr zmncs at slab+4");
        CHECK(st.family_ptr(cumes::SpectralComponent::Lcs) == slab + 5 * one, "family_ptr lmncs at slab+5");
        CHECK(st.velocity_family_ptr(cumes::SpectralComponent::Rcc) == vslab + 0 * one, "velocity_family_ptr v_rmncc at vslab+0");
        CHECK(st.velocity_family_ptr(cumes::SpectralComponent::Lcs) == vslab + 5 * one, "velocity_family_ptr v_lmncs at vslab+5");
        CHECK(st.state_buffer().size() == 6 * one &&
                  st.velocity_buffer().size() == 6 * one,
              "slab holds 6 components");
    }

    // ---- SpectralView device round-trip ----
    {
        const int ns = 5, mnmax = 3;
        cumes::SpectralStorage<double> st(ns, mnmax);
        auto view = st.physical();
        int total = mnmax * ns;
        writeSpectral<<<(total + 127) / 128, 128>>>(view, mnmax);
        cc(cudaDeviceSynchronize(), "sync view");
        std::vector<double> h(6 * (size_t)total);
        cc(cudaMemcpy(h.data(), st.state_slab(), h.size() * sizeof(double),
                      cudaMemcpyDeviceToHost),
           "read slab");
        bool ok = true;
        for (int c = 0; c < 6; ++c)
            for (int i = 0; i < total; ++i)
                ok = ok && (h[c * (size_t)total + i] == (double)(c * total + i));
        CHECK(ok, "SpectralView device writes land at [component][mode][surface]");
    }

    // ---- cuda_status: error injection throws CumesError ----
    {
        bool threw = false;
        try {
            cumes::check_cuda(cudaErrorInvalidValue, "inject");
        } catch (const cumes::CumesError&) {
            threw = true;
        }
        CHECK(threw, "check_cuda throws CumesError on error");

        threw = false;
        try {
            cumes::check_cufft(CUFFT_INVALID_PLAN, "inject");
        } catch (const cumes::CumesError&) {
            threw = true;
        }
        CHECK(threw, "check_cufft throws CumesError on error");
    }

    if (failures == 0) {
        printf("test_runtime: ALL PASS\n");
        return 0;
    }
    printf("test_runtime: %d FAILURES\n", failures);
    return 1;
}
