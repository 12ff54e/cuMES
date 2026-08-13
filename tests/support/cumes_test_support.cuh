#pragma once
// cumes_test_support.cuh — shared test helpers (CUDA error checking).
//
// Phase 1 extracts only the genuinely-shared, bit-for-bit-identical helper
// (the `checkCuda`/`cc` CUDA error check that every kernel-driving test
// duplicated). The CPU scalar references (cpuInvDFT, thomasSolve,
// cpuDealiasBandpass, …) are deliberately left in their owning tests: each is
// a per-operator reference used by exactly one test, so there is nothing to
// share yet. They move into cumes_test_support when a second consumer appears.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA error check: print and exit on failure. `cc` is the short form used
// across the tests; `checkCuda` is the verbose alias.
inline void checkCuda(cudaError_t e, const char* tag) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA[%s]: %s\n", tag, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}
inline void cc(cudaError_t e, const char* tag) { checkCuda(e, tag); }
