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

#include "cumes/runtime/device_arena.cuh"
#include "cumes/state/spectral_storage.hpp"
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
    const std::size_t batchRz = (std::size_t)4 * p.mpol * ns;

    std::size_t bytes = 0;
    // profiles: 4 full-grid + 7 half-grid radial arrays.
    bytes += (4 * ns + 7 * nH) * szT;
    // metric: 15 half-grid (ns-1, nZnT) arrays.
    bytes += 15 * nH * nZnT * szT;
    // fourier: 2 int mode tables, 43 real arrays, zeta scratch, 5 poloidal
    // tables.
    bytes += 2 * mnmax * szI;
    bytes += 43 * ns * nZnT * szT;
    bytes += 12 * p.mpol * ns * nz2 * szC;      // d_zeta_spectra
    bytes += 12 * p.mpol * ns * p.nzeta * szT;  // d_zeta_real
    bytes += 4 * p.mpol * p.ntheta * szT;       // cos/sin/mcos/msin_th
    bytes += (p.ntheta / 2 + 1) * szT;          // fwd_w
    // preconditioner: 25*nH + 9*ns + 7*mnmax*ns + 3*(ns+1) + 1 T-elements,
    // plus the mnmax int jMin table.
    bytes += (25 * nH + 9 * ns + 7 * mnmax * ns + 3 * (ns + 1) + 1) * szT;
    bytes += mnmax * szI;
    // constraint: 10 full-grid arrays + tcon + faccon + the compact zeta
    // scratch (real + spectra) for deAlias and rCon/zCon round trips.
    bytes += (10 * ns * nZnT + ns + mnmax) * szT;
    bytes += (batchDa + batchRz) * p.nzeta * szT;
    bytes += (batchDa + batchRz) * nz2 * szC;
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
                               SpectralStorage<T>& state) {
        DeviceArena arena;
        arena.allocate(stage_arena_bytes<T>(p));
        RadialProfiles<T> rp = profilesCreate<T>(p, ip, &arena);
        FourierPlan<T> fp = fourierCreate<T>(p, &arena);
        MetricWorkspace<T> mw = metricCreate<T>(p, &arena);
        SolverResult<T> result = solverRun<T>(state, p, rp, fp, mw, &arena);
        fourierFree(fp);
        metricFree(mw);
        profilesFree(rp);

        std::printf("  stage arena: %zu spans, peak %zu bytes (%.2f MiB), "
                    "reserved %zu bytes\n",
                    arena.span_count(), arena.peak_bytes(),
                    arena.peak_bytes() / (1024.0 * 1024.0),
                    arena.total_bytes());
        return result;
    }
};

}  // namespace cumes
