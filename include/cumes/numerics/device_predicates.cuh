// device_predicates.cuh — the per-pass device status/predicate kernels
// (blueprint §6.9/§7; completion plan step 1.4).
//
// These kernels implement the device-side safety decisions of the DAG:
//
//   jacobian_finalize_kernel             reset->reduce->FINALIZE the oriented
//                                      Jacobian validity (the rule is shared
//                                      with IterationController::jacobian_
//                                      invalid via cumes::JACOBIAN_EPS);
//   force_norm_finalize_kernel             finalize the force-norm factors ON
//                                      DEVICE from the refresh-pass force
//                                      norms, before the terminal predicate
//                                      (completion-plan follow-up §2.3);
//   invariant_predicate_kernel           classify the invariant residual ON
//                                      DEVICE before in-place preconditioning
//                                      (nonfinite always; converged on every
//                                      pass — refresh passes use the record's
//                                      device-finalized factors);
//   compute_residuals_preconditioned_kernel  the terminal-guarded
//   preconditioned
//                                      reduction (zero sentinel +
//                                      not_evaluated on terminal passes).
//
// They live in a PUBLIC header so tests can drive them directly with
// manufactured ControlRecords (tests/test_safety_predicates.cu); the
// production DAG (src/kernels/solver_impl.cuh) includes the same definitions,
// so the tested kernels are bit-for-bit the shipped ones. All readers/writers
// are ordered on the single compute stream — no atomics.
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_DEVICE_PREDICATES_CUH_
#define CUMES_INCLUDE_CUMES_NUMERICS_DEVICE_PREDICATES_CUH_

#include "cumes/core/tensor_view.cuh"
#include "cumes/numerics/accumulation.hpp"
#include "cumes/solver/control_record.hpp"

// Finalize the global Jacobian status (blueprint §7 "JStat"): read the just-
// reduced oriented stats and decide validity with the IDENTICAL rule the host
// controller applies (IterationController::jacobian_invalid + JACOBIAN_EPS).
// The bit gates every downstream 1/√g consumer, cache mutation, and force.
static __global__ void jacobian_finalize_kernel(
    cumes::ControlRecord* __restrict__ rec,
    int nZnT) {
    const double eps = cumes::JACOBIAN_EPS;
    const bool invalid =
        rec->jacobian_nonfinite_count > 0.0 || rec->jacobian_max_abs <= 0.0 ||
        rec->jacobian_min_oriented <= 0.0 ||
        (rec->jacobian_min_oriented < eps * rec->jacobian_max_abs &&
         rec->jacobian_min_index >= (double)nZnT);
    rec->status.jacobian_valid = invalid ? 0u : 1u;
}

// Finalize the force-norm factors ON DEVICE from the just-reduced partials
// (completion-plan follow-up §2.3). Runs on preconditioner-refresh passes,
// ordered after force_norm_reduce_kernel/rz_norm_kernel on the compute stream,
// so the required normalization IS available before the device terminal
// predicate — convergence classification is no longer structurally disabled
// on refresh passes. The expressions below are EXACTLY the host-side
// finalizeForceNorms rules (src/kernels/solver_impl.cuh), operator for
// operator: fabs, *, /, max and the ternary are all correctly-rounded IEEE
// double operations and contain no multiply-add pattern, so the device-computed
// factors are BIT-IDENTICAL to the previous host-side computation in every
// build. max is written as the std::max comparison form (a < b ? b : a), NOT
// fmax — fmax returns the non-NaN operand for (NaN, finite), which would
// diverge from the host on a pathological pass. Gated on force_norms_
// evaluated: on an invalid-Jacobian refresh pass the fields stay at the
// deterministic zero sentinel and the predicate skips convergence
// classification (see below).
static __global__ void force_norm_finalize_kernel(
    cumes::ControlRecord* __restrict__ rec,
    double delta_s,
    double lamscale) {
    if (!rec->status.force_norms_evaluated) return;
    const double sRZ = rec->force_norms[0];
    const double sL = rec->force_norms[1];
    const double sMag = rec->force_norms[2];
    double eTherm = rec->force_norms[3];
    double vol = rec->force_norms[4];
    const double h_rz = rec->force_norms[5];
    const double eMag = fabs(sMag) * delta_s;
    eTherm *= delta_s;
    vol *= delta_s;
    const double energyDensity = ((eMag < eTherm) ? eTherm : eMag) / vol;
    // Scale-free-division guards (identical to the host): degenerate
    // denominators produce the 1.0 fallback instead of inf/NaN factors.
    const double denomRZ = sRZ * energyDensity * energyDensity;
    rec->final_f_norm_rz = denomRZ > 0.0 ? (1.0 / denomRZ) : 1.0;
    const double denomL = sL * lamscale * lamscale;
    rec->final_f_norm_l = denomL > 0.0 ? (1.0 / denomL) : 1.0;
    rec->final_f_norm1 = h_rz > 0.0 ? (1.0 / h_rz) : 1.0;
}

// Classify the invariant (unpreconditioned) residual ON DEVICE before the
// in-place preconditioner (blueprint §6.9/§7 "Terminal"). The normalized
// triples are formed with the exact host expressions, so the bits agree with
// IterationController::classify_invariant bit-for-bit. Factor source:
//   use_record_factors == 0: the host's cached f_norm_rz/f_norm_l (non-refresh
//     passes — passed by value, identical to what the host will use);
//   use_record_factors != 0: the record's final_f_norm_* fields, finalized on
//     device from THIS pass's force norms on refresh passes (completion-plan
//     follow-up §2.3). The host consumes the same record fields at the fence,
//     so device and host classification share bit-identical inputs and a
//     converged refresh pass no-ops preconditioning like any terminal pass.
//     When the factors were not evaluated (invalid-Jacobian refresh pass, zero
//     sentinel) convergence classification is skipped: the host's Jacobian
//     gate restores before reading these bits anyway, and the guarded
//     preconditioner would no-op regardless.
// Nonfinite classification is factor-independent and always runs.
static __global__ void invariant_predicate_kernel(
    cumes::ControlRecord* __restrict__ rec,
    double f_norm_rz,
    double f_norm_l,
    double plain_per_el,
    double ftol,
    int use_record_factors) {
    const double f_rz = use_record_factors ? rec->final_f_norm_rz : f_norm_rz;
    const double f_l = use_record_factors ? rec->final_f_norm_l : f_norm_l;
    const double fsqr_i = rec->invariant_raw[0] * plain_per_el * f_rz * 0.25;
    const double fsqz_i = rec->invariant_raw[1] * plain_per_el * f_rz * 0.25;
    const double fsql_i = rec->invariant_raw[2] * plain_per_el * f_l;
    const bool nonfinite =
        !(isfinite(fsqr_i) && isfinite(fsqz_i) && isfinite(fsql_i));
    const bool can_classify =
        !use_record_factors || rec->status.force_norms_evaluated != 0;
    rec->status.invariant_nonfinite = nonfinite ? 1u : 0u;
    rec->status.invariant_converged =
        (!nonfinite && can_classify && fsqr_i <= ftol && fsqz_i <= ftol &&
         fsql_i <= ftol)
            ? 1u
            : 0u;
}

// Preconditioned residual reduction with the terminal gate: on a nonfinite or
// converged pass the in-place preconditioner no-op'd, so this reduction also
// no-ops — it stores the deterministic zero sentinel and leaves
// preconditioned_evaluated clear (blueprint §6.9 "not_evaluated"). The frozen
// telemetry already records zero preconditioned residuals on terminal passes
// (recordPass), so the on-disk contract is unchanged.
template <typename T>
__global__ void compute_residuals_preconditioned_kernel(
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain> f_spec,
    int ns,
    int mnmax,
    cumes::ControlRecord* __restrict__ rec) {
    using A = typename cumes::NormAccum<T>::type;  // double for mixed-float
    int comp = blockIdx.x;
    if (comp >= 3) return;
    const bool terminal = rec->status.invariant_nonfinite != 0 ||
                          rec->status.invariant_converged != 0;
    if (terminal) {
        if (threadIdx.x == 0) rec->preconditioned_raw[comp] = 0.0;
        return;
    }
    A sum = A(0);
    int total = mnmax * ns;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int mode = i / ns, j = i % ns;
        T a = f_spec(static_cast<cumes::SpectralComponent>(comp), mode, j);
        T b = f_spec(static_cast<cumes::SpectralComponent>(comp + 3), mode, j);
        sum += a * a + b * b;
    }
    __shared__ A s_sum[256];
    int tid = threadIdx.x;
    s_sum[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        rec->preconditioned_raw[comp] = s_sum[0] / (mnmax * ns);
        rec->status.preconditioned_evaluated = 1;  // idempotent (3 blocks)
    }
}

#endif  // CUMES_INCLUDE_CUMES_NUMERICS_DEVICE_PREDICATES_CUH_
