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

#include "cumes/config/json_reader.hpp"
#include "cumes/config/validated_problem.hpp"

// CUDA error check: print and exit on failure. `cc` is the short form used
// across the tests; `checkCuda` is the verbose alias.
inline void checkCuda(cudaError_t e, const char* tag) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA[%s]: %s\n", tag, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}
inline void cc(cudaError_t e, const char* tag) { checkCuda(e, tag); }

// Load a validated problem from a JSON fixture (tests run from the source dir).
// The legacy parser used to produce an InputParams for the same fixture; this
// returns the equivalent immutable model. Tests that call this must link
// cumes_config_json (for cumes::read_and_validate).
inline cumes::ValidatedProblem loadValidated(
    const char* path = "inputs/solovev.json") {
    cumes::SolverOptions opts;
    auto vr = cumes::read_and_validate(path, opts);
    if (!vr.has_value()) {
        fprintf(stderr, "loadValidated: %s failed validation\n", path);
        exit(EXIT_FAILURE);
    }
    return std::move(vr.value());
}

// Validate a hand-built ProblemSpec into a ValidatedProblem (for tests that
// tweak ncurr/curtor/profiles without a JSON fixture). Exits on validation
// failure.
inline cumes::ValidatedProblem validateSpec(cumes::ProblemSpec spec) {
    cumes::SolverOptions opts;
    auto vr = cumes::validate(std::move(spec), opts);
    if (!vr.has_value()) {
        fprintf(stderr, "validateSpec: validation failed\n");
        exit(EXIT_FAILURE);
    }
    return std::move(vr.value());
}
