// stage_solver.hpp — one radial stage's resource lifecycle + solve
// (blueprint §6.11). The stage owns its ns-dependent profile/Fourier/metric
// workspaces (and, transitively, the preconditioner/constraint/solver
// workspaces the solver builds) and invokes the solver; the spectral state is
// owned by the multigrid driver (it persists across stages via prolongation).
//
// Phase 5: all five workspaces are backed by a single DeviceArena — one
// cudaMalloc per stage instead of ~110 per-array allocations. The arena plan
// is MEASURED (7.2): stage_arena_bytes runs the real module constructors
// against a measuring arena (growth-retry on overflow) and returns the exact
// carved byte total, so a buffer added/removed in any module updates the plan
// automatically — the old hand-summed arithmetic is gone except as a
// non-authoritative seed for the measuring pass's first budget guess. The
// arena's liveness/peak report is emitted after the solve.
#ifndef CUMES_INCLUDE_CUMES_SOLVER_STAGE_SOLVER_HPP_
#define CUMES_INCLUDE_CUMES_SOLVER_STAGE_SOLVER_HPP_

#include "cumes/numerics/preconditioner.hpp"
#include "cumes/physics/constraint_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/runtime/device_arena.cuh"
#include "cumes/solver/solver_bench.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/real_space_storage.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/axisymmetric_operator.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "fft_traits.h"
#include "solver.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

namespace cumes {
namespace stage_detail {

// ---- RAII wrappers for the two free-function stage resources (8.3) -------
// realSpaceCreate/Free and modeTableCreate/Free are plain function pairs; the
// wrappers restore the "xCreate/xFree replaced by RAII classes" convention
// without restructuring the (sibling-owned) state headers. The wrapped frees
// keep their arena-backed no-op semantics: on the always-arena stage path the
// arena owns the memory, and on a nullptr-arena path the destructor frees the
// per-array cudaMallocs even when a later constructor throws.
template <typename T>
class ScopedRealSpace {
   public:
    using val_type = T;

    ScopedRealSpace(const DeviceParams<T>& p, DeviceArena* arena)
        : rs_(realSpaceCreate<T>(p, arena)) {}
    ~ScopedRealSpace() { realSpaceFree(rs_); }

    // Non-copyable, non-movable: RealSpaceStorage is a raw owning aggregate
    // with no move semantics, so a move would leave two owners of the same
    // pointers (double cudaFree on the non-arena path).
    ScopedRealSpace(const ScopedRealSpace&) = delete;
    ScopedRealSpace& operator=(const ScopedRealSpace&) = delete;
    ScopedRealSpace(ScopedRealSpace&&) = delete;
    ScopedRealSpace& operator=(ScopedRealSpace&&) = delete;

    RealSpaceStorage<T>& get() { return rs_; }
    RealSpaceStorage<T>* operator->() { return &rs_; }
    RealSpaceStorage<T>& operator*() { return rs_; }

   private:
    RealSpaceStorage<T> rs_;
};

template <typename T>
class ScopedModeTable {
   public:
    using val_type = T;

    ScopedModeTable(const DeviceParams<T>& p, DeviceArena* arena)
        : mt_(modeTableCreate<T>(p, arena)) {}
    ~ScopedModeTable() { modeTableFree(mt_); }

    ScopedModeTable(const ScopedModeTable&) = delete;
    ScopedModeTable& operator=(const ScopedModeTable&) = delete;
    ScopedModeTable(ScopedModeTable&&) = delete;
    ScopedModeTable& operator=(ScopedModeTable&&) = delete;

    DeviceModeTable& get() { return mt_; }

   private:
    DeviceModeTable mt_;
};

// Construct the exact arena-allocation sequence of one stage against `arena`
// — no longer a SEPARATE measuring pass: the real stage builds inside the
// retry helper below (completion plan step 3.2), so the constructors' own
// alloc_span calls ARE the plan and nothing is constructed twice.
}  // namespace stage_detail

// Non-authoritative sizing hint for the measuring pass's first budget guess
// (the pre-7.2 hand-sum, retained only so the measurement succeeds on the
// first attempt for the current shapes). An underestimate merely costs one
// more measuring attempt; the value returned by stage_arena_bytes is always
// the MEASURED total, never this arithmetic.
template <typename T>
std::size_t stage_arena_seed_bytes(const DeviceParams<T>& p) {
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
    // mode table: 2 int arrays (d_xm/d_xn, resolution-scoped — blueprint §6.2).
    bytes += 2 * mnmax * szI;
    // fourier: 43 real arrays, zeta scratch (main + compact de-alias), 5
    // poloidal tables.
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
    // solver: f_spec + control + psum (6.4 — carved from the stage arena;
    // the control span is the typed ControlRecord).
    bytes +=
        (6 * mnmax * ns + 4 * (ns - 1)) * szT + sizeof(cumes::ControlRecord);
    // Alignment slack for the ~110 subspans (each padded to alignof <= 16).
    bytes += 64 * 1024;
    return bytes;
}

// Growth-retry single-construction stage helper (completion plan step 3.2).
// One DeviceArena, one construction of every module, one solve: the retry
// loop exists only so an underestimating seed budget still succeeds — on the
// normal path the constructors run exactly once against exactly one
// allocation, and no temporary measuring arena exists. `fn(arena)` performs
// the full stage body (module construction + solverRun + arena reporting);
// an ArenaOverflow aborts the attempt, the scope exits destroy the partial
// modules, and the budget grows to the reported requirement.
template <typename T, typename F>
auto run_in_stage_arena(const DeviceParams<T>& p, F&& fn)
    -> decltype(fn(std::declval<DeviceArena&>())) {
    std::size_t budget = stage_arena_seed_bytes<T>(p);
    for (int attempt = 0; attempt < 16; ++attempt) {
        DeviceArena arena;
        arena.allocate(budget);
        try {
            return fn(arena);
        } catch (const ArenaOverflow& over) {
            budget = 2 * over.required_bytes + (1u << 20);
        }
    }
    throw CumesError("stage arena: sizing did not converge");
}

// Runs a single radial stage on `p` (already carrying this stage's
// ns/max_iter/ftol). Creates the stage's workspaces from one arena, runs the
// fixed-point solver on `state`, reports the arena's liveness/peak, and frees
// the non-arena resources (cuFFT plans, pinned host faccon) before returning.
// `state` stays owned by the caller; profilesCreate sets p.lamscale in place.
template <typename T>
class StageSolver {
   public:
    using val_type = T;

    static SolverResult<T> run(DeviceParams<T>& p,
                               const ValidatedProblem& vp,
                               SpectralStorage<T>& state,
                               cudaStream_t stream = 0,
                               SolverBench* bench = nullptr) {
        // One arena allocation, one construction of every module, one solve
        // (completion plan step 3.2): the modules' alloc_span calls ARE the
        // plan — there is no temporary measuring arena and nothing is
        // constructed twice. The retry loop below only grows the budget when
        // the seed underestimates (an ArenaOverflow aborts the attempt and
        // the scope exit destroys the partial modules).
        return run_in_stage_arena<T>(p, [&](DeviceArena& arena) {
            // Setup (profiles/Fourier/metric) is synchronous on the default
            // stream, so it completes before the solve; the hot loop runs on
            // the explicit nonblocking compute stream (Phase 6A). The
            // real-space storage and the mode table are scoped RAII (their
            // frees no-op on the arena path and cover the nullptr-arena
            // path).
            Profiles<T> profiles(p, vp, &arena);
            stage_detail::ScopedRealSpace<T> rs(p, &arena);
            stage_detail::ScopedModeTable<T> mt(p, &arena);
            ToroidalFftOperator<T> transform(p, *rs, mt.get(), &arena);
            GeometryOperator<T> geometry(p, &arena);

            // Transform backend selection (blueprint §8.5): for ntor=0/nzeta=1
            // the toroidal direction is a single point, so the length-one
            // cuFFT round trips are replaced by direct-poloidal
            // synthesis/projection. The operator holds only ns-independent
            // poloidal tables, but its kernels launch on `p.ns`, so one is
            // built per stage (re-uploading the tiny tables is negligible).
            // CUMES_FORCE_GENERIC=1 restores the generic backend for A/B
            // comparison against the frozen trajectory. The solver drives a
            // single `SpectralOperator<T>*` (no axisym_active branch);
            // nullptr selects the generic ToroidalFft operator.
            bool use_axisym = (p.ntor == 0 && p.nzeta == 1);
            if (const char* e = std::getenv("CUMES_FORCE_GENERIC"))
                if (std::atoi(e) != 0) use_axisym = false;
            std::unique_ptr<AxisymmetricOperator<T>> axisym;
            if (use_axisym)
                axisym = std::make_unique<AxisymmetricOperator<T>>(p);

            SolverResult<T> result =
                solverRun<T>(state, p, profiles, transform, *rs, geometry,
                             &arena, stream, bench, axisym.get());
            // profiles/transform/geometry/rs/mt are RAII (scoped wrappers +
            // the operator destructors); nothing to free manually.

            std::printf(
                "  stage arena: %zu spans, peak %zu bytes (%.2f MiB), "
                "reserved %zu bytes\n",
                arena.span_count(), arena.peak_bytes(),
                arena.peak_bytes() / (1024.0 * 1024.0), arena.total_bytes());
            return result;
        });
    }
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_SOLVER_STAGE_SOLVER_HPP_
