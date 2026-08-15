// test_arena.cu — Phase 5 DeviceArena: aligned subspans, named reporting,
// peak/liveness accounting, overflow and move semantics.
//
// The arena is a host-side primitive: it allocates one backing store and carves
// typed spans out of it. This test verifies the carve/align/zero/report/overflow
// contract, plus a device round-trip through a carved span to prove the returned
// pointers are ordinary usable device pointers.
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "cumes/runtime/device_arena.cuh"
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

__global__ void fillSpan(double* d, int n, double seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = seed + (double)i;
}

int main() {
    printf("=== DeviceArena ===\n");

    // ---- carve + alignment + reporting ----
    {
        cumes::DeviceArena arena;
        arena.allocate(4096);

        double* a = arena.alloc_span<double>("metric/tau", 100);
        // The next span starts 256-aligned from a's end; force a misalignment
        // by carving a 3-byte-then-double sequence.
        char* pad = arena.alloc_span<char>("pad", 3);
        double* b = arena.alloc_span<double>("metric/gsqrt", 100);
        (void)pad;

        CHECK(a != nullptr && b != nullptr, "alloc_span returns non-null");
        CHECK((reinterpret_cast<std::uintptr_t>(a) % alignof(double)) == 0,
              "double span aligned");
        CHECK((reinterpret_cast<std::uintptr_t>(b) % alignof(double)) == 0,
              "span after 3-byte pad realigned");
        CHECK(b >= a + 100, "spans do not overlap");
        CHECK(arena.span_count() == 3, "three spans recorded");
        CHECK(arena.total_bytes() == 4096, "total_bytes");
        CHECK(arena.peak_bytes() == arena.used_bytes(),
              "linear arena peak == used");

        const auto& spans = arena.spans();
        CHECK(spans[0].name == "metric/tau" && spans[0].bytes == 800,
              "span name + bytes reported");
    }

    // ---- zero_span + device round-trip ----
    {
        cumes::DeviceArena arena;
        arena.allocate(4096);
        double* d = arena.alloc_span<double>("state/slab", 64);
        arena.zero_span(d, 64);
        fillSpan<<<1, 64>>>(d, 64, 10.0);
        cc(cudaDeviceSynchronize(), "sync fillSpan");
        double h[64];
        cc(cudaMemcpy(h, d, 64 * sizeof(double), cudaMemcpyDeviceToHost), "read");
        bool ok = true;
        for (int i = 0; i < 64; ++i) ok = ok && h[i] == 10.0 + (double)i;
        CHECK(ok, "carved span usable by device kernels");
    }

    // ---- overflow throws CumesError ----
    {
        cumes::DeviceArena arena;
        arena.allocate(128);
        arena.alloc_span<double>("first", 8);  // 64 bytes
        bool threw = false;
        try {
            arena.alloc_span<double>("overflow", 100);  // 800 > 64 remaining
        } catch (const cumes::CumesError&) {
            threw = true;
        }
        CHECK(threw, "overflowing span throws CumesError");
        // Exact fit is fine.
        arena.alloc_span<double>("fits", 8);  // 64 bytes exactly
        CHECK(arena.used_bytes() == 128, "exact-fit span fills the arena");
    }

    // ---- bad alignment rejected ----
    {
        cumes::DeviceArena arena;
        arena.allocate(64);
        bool threw = false;
        try {
            arena.alloc_span<double>("bad-align", 4, /*align=*/3);
        } catch (const cumes::CumesError&) {
            threw = true;
        }
        CHECK(threw, "non-power-of-two alignment rejected");
    }

    // ---- move semantics ----
    {
        cumes::DeviceArena arena;
        arena.allocate(256);
        double* p = arena.alloc_span<double>("move", 8);
        cumes::DeviceArena moved(std::move(arena));
        CHECK(moved.data() != nullptr && moved.used_bytes() == 64,
              "moved arena owns the store");
        CHECK(arena.empty(), "moved-from arena is empty");
        (void)p;
    }

    // ---- empty count returns null without consuming ----
    {
        cumes::DeviceArena arena;
        arena.allocate(64);
        double* p = arena.alloc_span<double>("empty", 0);
        CHECK(p == nullptr && arena.used_bytes() == 0,
              "zero-count span consumes nothing");
    }

    if (failures == 0) {
        printf("test_arena: ALL PASS\n");
        return 0;
    }
    printf("test_arena: %d FAILURES\n", failures);
    return 1;
}
