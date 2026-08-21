#pragma once
// cumes_test_cuda_helper.cuh — CUDA-side shared test helpers (error checking +
// manufactured spectral states + config loading), on top of the CUDA-free
// harness (cumes_test.h).
//
// Phase 1 extracted the genuinely-shared, bit-for-bit-identical helper (the
// `check_cuda`/`cc` CUDA error check that every kernel-driving test
// duplicated). The per-operator CPU scalar references (cpuInvDFT, thomasSolve,
// cpuDealiasBandpass, …) remain in their owning tests while they have a single
// consumer; they move here when a second consumer appears. The six-family
// manufactured state moved here in the 2026-08-16 review pass, when its
// consumer count reached four.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <vector>

#include "cumes_test.h"
#include "cumes/config/json_reader.hpp"
#include "cumes/config/validated_problem.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/state/spectral_storage.hpp"

namespace cumes::test {

// CUDA error check: print and exit on failure. `cc` is the short form used
// across the tests; `check_cuda` is the verbose alias. The status conversion
// itself is delegated to cumes::check_cuda (the production throwing boundary);
// the tests keep the exit-based UX — an uncaught CumesError would fall through
// to std::terminate and lose the tag message — so the helper catches, prints
// the tag, and exits.
inline void check_cuda(cudaError_t e, const char* tag) {
    try {
        cumes::check_cuda(e, tag);
    } catch (const cumes::CumesError&) {
        fprintf(stderr, "CUDA[%s]: %s\n", tag, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}
inline void cc(cudaError_t e, const char* tag) { check_cuda(e, tag); }

// ---------------------------------------------------------------------------
// Manufactured six-family spectral states (host vectors, column-major
// [mode*ns + j]). Frozen fixtures: the tests that consume them rely on the
// exact non-degenerate content — do not "harmlessly" tweak an envelope.
// ---------------------------------------------------------------------------
enum class ManufacturedShape {
    // Solovev-like, linear radial envelopes: R_00=4.0, R_10=0.3s, R_20=0.2s,
    // Rss=Rcc, Z_10 (sc+cs)=-0.5s, lambda=0. (test_geometry_ncurr)
    kSolovevLinear,
    // As kSolovevLinear but R_20 uses a QUADRATIC s^2 envelope — deliberate:
    // with a linear profile rCon(s) == s*rCon_LCFS exactly, so rCon - rCon0 is
    // identically zero and the constraint force vanishes no matter what tcon0
    // is (test_constraint_tcon would pass vacuously). The quadratic envelope
    // puts the interior off the LCFS-extrapolated reference, making the
    // constraint-force contribution genuinely nonzero. (test_constraint_tcon)
    kSolovevQuadM2,
    // Reduced graph fixture: R_00=4.0, R_10=0.3s, m>=2: 0.1s^2, Rss=Rcc,
    // no Z/lambda content. (test_cuda_graph)
    kGraphQuad,
    // Full-mode generic W7-X-shaped content: R has a strong m=0/n=0 DC plus
    // mild m>0 modes, Z and lambda get m=1..3 content with s envelopes — no
    // interior surface collapses to zero geometry. (test_geometry_iso)
    kW7XGeneric,
};

template <typename T>
inline void manufactured_state(ManufacturedShape shape, int ns, int mnmax,
                               int ntor, std::vector<T>& cc, std::vector<T>& ss,
                               std::vector<T>& zsc, std::vector<T>& zcs,
                               std::vector<T>& lsc, std::vector<T>& lcs) {
    const size_t n = (size_t)ns * mnmax;
    cc.assign(n, T(0)); ss.assign(n, T(0)); zsc.assign(n, T(0));
    zcs.assign(n, T(0)); lsc.assign(n, T(0)); lcs.assign(n, T(0));
    for (int j = 0; j < ns; ++j) {
        double s = (double)j / (ns - 1.0);
        double s2 = s * s;
        for (int mode = 0; mode < mnmax; ++mode) {
            int m = mode / (ntor + 1), n = mode % (ntor + 1);
            switch (shape) {
            case ManufacturedShape::kSolovevLinear:
                if (m == 0) cc[j + mode * ns] = T(4.0);
                else if (m == 1) cc[j + mode * ns] = T(0.3 * s);
                else if (m == 2) cc[j + mode * ns] = T(0.2 * s);
                ss[j + mode * ns] = cc[j + mode * ns];
                if (m == 1) { zsc[j + mode * ns] = T(-0.5 * s); zcs[j + mode * ns] = T(-0.5 * s); }
                break;
            case ManufacturedShape::kSolovevQuadM2:
                if (m == 0) cc[j + mode * ns] = T(4.0);
                else if (m == 1) cc[j + mode * ns] = T(0.3 * s);
                else if (m == 2) cc[j + mode * ns] = T(0.2 * s2);   // quadratic, deliberate
                ss[j + mode * ns] = cc[j + mode * ns];
                if (m == 1) { zsc[j + mode * ns] = T(-0.5 * s); zcs[j + mode * ns] = T(-0.5 * s); }
                break;
            case ManufacturedShape::kGraphQuad:
                if (m == 0) cc[j + mode * ns] = T(4.0);
                else if (m == 1) cc[j + mode * ns] = T(0.3 * s);
                else cc[j + mode * ns] = T(0.1 * s2);
                ss[j + mode * ns] = cc[j + mode * ns];
                break;
            case ManufacturedShape::kW7XGeneric:
                if (m == 0 && n == 0) { cc[j + mode * ns] = T(5.6); zcs[j + mode * ns] = T(0.0); }
                else if (m == 0)      { cc[j + mode * ns] = T(0.02 * s2); zcs[j + mode * ns] = T(0.01 * s2); }
                else if (m == 1)      { cc[j + mode * ns] = T(0.3 * s); ss[j + mode * ns] = T(0.1 * s);
                                        zsc[j + mode * ns] = T(0.2 * s); zcs[j + mode * ns] = T(-0.1 * s); }
                else if (m == 2)      { cc[j + mode * ns] = T(0.04 * s2); ss[j + mode * ns] = T(0.02 * s2);
                                        zsc[j + mode * ns] = T(0.03 * s2); zcs[j + mode * ns] = T(0.01 * s2); }
                else if (m <= 6)      { cc[j + mode * ns] = T(0.01 * s2); zsc[j + mode * ns] = T(0.008 * s2); }
                lsc[j + mode * ns] = T(0.02 * (m + 1) * s2);
                lcs[j + mode * ns] = T(0.01 * (m + 1) * s2);
                break;
            }
        }
    }
}

// Upload all six manufactured families into a SpectralStorage (the standard
// six-family H2D upload used by every manufactured-state test).
template <typename T>
inline void upload_state(cumes::SpectralStorage<T>& storage, const std::vector<T>& h_cc,
                         const std::vector<T>& h_ss, const std::vector<T>& h_zsc,
                         const std::vector<T>& h_zcs, const std::vector<T>& h_lsc,
                         const std::vector<T>& h_lcs, int ns, int mnmax) {
    size_t nb = (size_t)ns * mnmax * sizeof(T);
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rcc), h_cc.data(), nb, cudaMemcpyHostToDevice), "up cc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rss), h_ss.data(), nb, cudaMemcpyHostToDevice), "up ss");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zsc), h_zsc.data(), nb, cudaMemcpyHostToDevice), "up zsc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zcs), h_zcs.data(), nb, cudaMemcpyHostToDevice), "up zcs");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lsc), h_lsc.data(), nb, cudaMemcpyHostToDevice), "up lsc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lcs), h_lcs.data(), nb, cudaMemcpyHostToDevice), "up lcs");
}

// Load a validated problem from a JSON fixture (tests run from the source dir).
// Returns the immutable ValidatedProblem model. Tests that call this must link
// cumes_config_json (for cumes::read_and_validate).
inline cumes::ValidatedProblem load_validated(
    const char* path = "inputs/solovev.json") {
    cumes::SolverOptions opts;
    auto vr = cumes::read_and_validate(path, opts);
    if (!vr.has_value()) {
        fprintf(stderr, "load_validated: %s failed validation\n", path);
        exit(EXIT_FAILURE);
    }
    return std::move(vr.value());
}

// Validate a hand-built ProblemSpec into a ValidatedProblem (for tests that
// tweak ncurr/curtor/profiles without a JSON fixture). Exits on validation
// failure.
inline cumes::ValidatedProblem validate_spec(cumes::ProblemSpec spec) {
    cumes::SolverOptions opts;
    auto vr = cumes::validate(std::move(spec), opts);
    if (!vr.has_value()) {
        fprintf(stderr, "validate_spec: validation failed\n");
        exit(EXIT_FAILURE);
    }
    return std::move(vr.value());
}

}  // namespace cumes::test
