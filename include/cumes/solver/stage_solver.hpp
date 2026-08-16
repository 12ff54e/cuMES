// stage_solver.hpp — one radial stage's resource lifecycle + solve
// (blueprint §6.11). The stage owns its ns-dependent profile/Fourier/metric
// workspaces (and, transitively, the preconditioner/constraint workspaces the
// solver builds) and invokes the solver; the spectral state is owned by the
// multigrid driver (it persists across stages via prolongation).
//
// Phase 5: all five workspaces are backed by a single DeviceArena — one
// cudaMalloc per stage instead of ~110 per-array allocations. stage_arena_bytes
// is the host-side "plan" (blueprint §6.5 StageWorkspace::plan) that sums each
// module's named subspans before the arena is allocated; the arena's
// liveness/peak report is emitted after the solve. alloc_span throws on a too-
// small plan, so a miscounted size is a loud setup error, not silent aliasing.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "cumes/runtime/device_arena.cuh"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/solver/solver_bench.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/axisymmetric_operator.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "fft_traits.h"
#include "fourier.cuh"
#include "geometry.cuh"
#include "input.h"
#include "profiles.cuh"
#include "solver.cuh"

namespace cumes {

// Exact byte total of one stage's workspaces (profiles + Fourier + metric +
// preconditioner + constraint), including a small alignment slack. The per-span
// sizes mirror the *Create functions' alloc_span calls; keep the two in sync.
template <typename T>
std::size_t stage_arena_bytes(const GridParams<T>& p) {
    using Complex = typename FftTraits<T>::Complex;
    const std::size_t ns = p.ns, nH = ns - 1, nZnT = p.nZnT, mnmax = p.mnmax;
    const std::size_t szT = sizeof(T), szI = sizeof(int), szC = sizeof(Complex);
    const int nz2 = p.nzeta / 2 + 1;
    const std::size_t batchDa = (std::size_t)2 * (p.mpol - 2) * (ns - 1);

    std::size_t bytes = 0;
    // profiles: 4 full-grid + 7 half-grid radial arrays.
    bytes += (4 * ns + 7 * nH) * szT;
    // metric: 15 half-grid (ns-1, nZnT) arrays.
    bytes += 15 * nH * nZnT * szT;
    // fourier: 2 int mode tables, 43 real arrays, zeta scratch (main + compact
    // de-alias), 5 poloidal tables.
    bytes += 2 * mnmax * szI;
    bytes += 43 * ns * nZnT * szT;
    bytes += 12 * p.mpol * ns * nz2 * szC;      // d_zeta_spectra
    bytes += 12 * p.mpol * ns * p.nzeta * szT;  // d_zeta_real
    bytes += 4 * p.mpol * p.ntheta * szT;       // cos/sin/mcos/msin_th
    bytes += (p.ntheta / 2 + 1) * szT;          // fwd_w
    bytes += batchDa * p.nzeta * szT;           // d_zeta_real_c (de-alias)
    bytes += batchDa * nz2 * szC;               // d_zeta_spectra_c (de-alias)
    // preconditioner: 25*nH + 9*ns + 7*mnmax*ns + 3*(ns+1) + 1 T-elements,
    // plus the mnmax int jMin table.
    bytes += (25 * nH + 9 * ns + 7 * mnmax * ns + 3 * (ns + 1) + 1) * szT;
    bytes += mnmax * szI;
    // constraint: 10 full-grid arrays + tcon + faccon (the de-alias compact
    // scratch now lives in the fourier section above).
    bytes += (10 * ns * nZnT + ns + mnmax) * szT;
    // Alignment slack for the ~110 subspans (each padded to alignof <= 16).
    bytes += 64 * 1024;
    return bytes;
}

// Runs a single radial stage on `p` (already carrying this stage's
// ns/max_iter/ftol). Creates the stage's workspaces from one arena, runs the
// fixed-point solver on `state`, reports the arena's liveness/peak, and frees
// the non-arena resources (cuFFT plans, pinned host faccon) before returning.
// `state` stays owned by the caller; profilesCreate sets p.lamscale in place.
template <typename T>
class StageSolver {
  public:
    static SolverResult<T> run(GridParams<T>& p, const InputParams& ip,
                               SpectralStorage<T>& state,
                               cudaStream_t stream = 0,
                               SolverBench* bench = nullptr) {
        DeviceArena arena;
        arena.allocate(stage_arena_bytes<T>(p));
        // Setup (profiles/Fourier/metric) is synchronous on the default
        // stream, so it completes before the solve; the hot loop runs on the
        // explicit nonblocking compute stream (Phase 6A).
        Profiles<T> profiles(p, ip, &arena);
        RealSpaceStorage<T> rs = realSpaceCreate<T>(p, &arena);
        ToroidalFftOperator<T> transform(p, rs, &arena);
        GeometryOperator<T> geometry(p, &arena);

        // Transform backend selection (blueprint §8.5): for ntor=0/nzeta=1
        // the toroidal direction is a single point, so the length-one cuFFT
        // round trips are replaced by direct-poloidal synthesis/projection. The
        // operator holds only ns-independent poloidal tables, but its kernels
        // launch on `p.ns`, so one is built per stage (re-uploading the tiny
        // tables is negligible). CUMES_FORCE_GENERIC=1 restores the generic
        // backend for A/B comparison against the frozen trajectory. The solver
        // drives a single `SpectralOperator<T>*` (no axisym_active branch);
        // nullptr selects the generic ToroidalFft operator.
        bool use_axisym = (p.ntor == 0 && p.nzeta == 1);
        if (const char* e = std::getenv("CUMES_FORCE_GENERIC"))
            if (std::atoi(e) != 0) use_axisym = false;
        std::unique_ptr<AxisymmetricOperator<T>> axisym;
        if (use_axisym) axisym = std::make_unique<AxisymmetricOperator<T>>(p);

        SolverResult<T> result = solverRun<T>(state, p, profiles.workspace(),
                                              transform, rs, geometry, &arena, stream,
                                              bench, axisym.get());
        realSpaceFree(rs);
        // profiles/transform/geometry are RAII (operator destructors).

        std::printf("  stage arena: %zu spans, peak %zu bytes (%.2f MiB), "
                    "reserved %zu bytes\n",
                    arena.span_count(), arena.peak_bytes(),
                    arena.peak_bytes() / (1024.0 * 1024.0),
                    arena.total_bytes());
        return result;
    }
};

}  // namespace cumes
