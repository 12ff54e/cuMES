// precon.cu — radial tridiagonal preconditioner for MHD force balance.
// Computes flux-surface-averaged Hessian approximation separately for R and Z,
// assembles tridiagonal systems per (m,n) mode, and solves via parallel cyclic
// reduction (tridiagSolveKernel).
// Reference: vmecpp computePreconditioningMatrix / assembleRZPreconditioner /
// TridiagonalSolveSerial.
//
// All computation is templated on the scalar type T (double or float).
#include "precon.cuh"
#include <cstdio>
#include <cmath>

// nvcc rejects `extern __shared__` arrays in a function template that is
// instantiated with different element types in one TU ("declaration is
// incompatible with previous" — both the double and float instantiations
// collide on the variable name). Route the dynamic shared-memory base through
// one non-templated device function instead: the base address is the same for
// every function in a block, so the kernels reinterpret it as T*.
namespace {
__device__ void* dynSharedBase() {
    extern __shared__ unsigned char smem_base[];
    return smem_base;
}
}


static void cc(cudaError_t e, const char* t) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA[%s]: %s\n", t, cudaGetErrorString(e)); exit(1); }
}

// ---------------------------------------------------------------------------
// Allocate
// ---------------------------------------------------------------------------
template <typename T>
PreconWorkspace<T> preconCreate(const GridParams<T>& p) {
    PreconWorkspace<T> pw{};
    int nH = p.ns - 1, nF = p.ns;
    size_t szH  = nH * sizeof(T);
    size_t szF  = nF * sizeof(T);
    size_t szH4 = 4 * nH * sizeof(T);
    size_t szH3 = 3 * nH * sizeof(T);
    size_t sz2H = 2 * nH * sizeof(T);
    size_t sz2F = 2 * nF * sizeof(T);
    size_t szMN = p.mnmax * nF * sizeof(T);

    cc(cudaMalloc(&pw.d_ax_R, szH4), "ax_R");  cc(cudaMalloc(&pw.d_ax_Z, szH4), "ax_Z");
    cc(cudaMalloc(&pw.d_bx_R, szH3), "bx_R");  cc(cudaMalloc(&pw.d_bx_Z, szH3), "bx_Z");
    cc(cudaMalloc(&pw.d_cx,  szH),  "cx");
    cc(cudaMalloc(&pw.d_arm, sz2H), "arm");    cc(cudaMalloc(&pw.d_brm, sz2H), "brm");
    cc(cudaMalloc(&pw.d_azm, sz2H), "azm");    cc(cudaMalloc(&pw.d_bzm, sz2H), "bzm");
    cc(cudaMalloc(&pw.d_ard, sz2F), "ard");    cc(cudaMalloc(&pw.d_brd, sz2F), "brd");
    cc(cudaMalloc(&pw.d_azd, sz2F), "azd");    cc(cudaMalloc(&pw.d_bzd, sz2F), "bzd");
    cc(cudaMalloc(&pw.d_cxd, szF),  "cxd");
    cc(cudaMalloc(&pw.d_sm,  szH),  "sm");     cc(cudaMalloc(&pw.d_sp,  szH),  "sp");
    cc(cudaMalloc(&pw.d_ar, szMN),  "ar");     cc(cudaMalloc(&pw.d_dr, szMN),  "dr");
    cc(cudaMalloc(&pw.d_br, szMN),  "br");
    cc(cudaMalloc(&pw.d_az, szMN),  "az");     cc(cudaMalloc(&pw.d_dz, szMN),  "dz");
    cc(cudaMalloc(&pw.d_bz, szMN),  "bz");
    cc(cudaMalloc(&pw.d_jMin, p.mnmax * sizeof(int)), "jMin");
    cc(cudaMalloc(&pw.d_lambdaPrec, szMN), "lambdaPrec");
    cc(cudaMalloc(&pw.d_bLambda, (p.ns + 1) * sizeof(T)), "bLambda");
    cc(cudaMalloc(&pw.d_dLambda, (p.ns + 1) * sizeof(T)), "dLambda");
    cc(cudaMalloc(&pw.d_cLambda, (p.ns + 1) * sizeof(T)), "cLambda");
    cc(cudaMalloc(&pw.d_rmsPhiP, sizeof(T)), "rmsPhiP");
    // Index ns of bLambda/cLambda must stay zero: the LCFS full-grid average
    // reads it (vmecpp: array sized ns+1, last entry never written).
    cc(cudaMemset(pw.d_bLambda, 0, (p.ns + 1) * sizeof(T)), "bLambda zero");
    cc(cudaMemset(pw.d_dLambda, 0, (p.ns + 1) * sizeof(T)), "dLambda zero");
    cc(cudaMemset(pw.d_cLambda, 0, (p.ns + 1) * sizeof(T)), "cLambda zero");
    return pw;
}

template <typename T>
void preconFree(PreconWorkspace<T>& pw) {
    cudaFree(pw.d_ax_R); cudaFree(pw.d_ax_Z);
    cudaFree(pw.d_bx_R); cudaFree(pw.d_bx_Z); cudaFree(pw.d_cx);
    cudaFree(pw.d_arm);  cudaFree(pw.d_brm);
    cudaFree(pw.d_azm);  cudaFree(pw.d_bzm);
    cudaFree(pw.d_ard);  cudaFree(pw.d_brd);
    cudaFree(pw.d_azd);  cudaFree(pw.d_bzd);  cudaFree(pw.d_cxd);
    cudaFree(pw.d_sm);   cudaFree(pw.d_sp);
    cudaFree(pw.d_ar);   cudaFree(pw.d_dr);   cudaFree(pw.d_br);
    cudaFree(pw.d_az);   cudaFree(pw.d_dz);   cudaFree(pw.d_bz);
    cudaFree(pw.d_jMin);
    cudaFree(pw.d_lambdaPrec);
    cudaFree(pw.d_bLambda); cudaFree(pw.d_dLambda); cudaFree(pw.d_cLambda); cudaFree(pw.d_rmsPhiP);
}

// ---------------------------------------------------------------------------
// Step 1: Compute ax[4], bx[3], cx[1] on half-grid via surface integrals.
// One block per half-grid surface; parallel reduction over nZnT points.
// Processes both R and Z preconditioners in one pass.
//
// For R: xs=zs, xu12=zu12, xu_e/o=zu_e/o
// For Z: xs=rs, xu12=ru12, xu_e/o=ru_e/o
// ---------------------------------------------------------------------------
template <typename T>
__global__ void preconComputeKernel(
    // Geometry shared by both R and Z precons
    const T* __restrict__ r12, const T* __restrict__ tau,
    const T* __restrict__ totalP, const T* __restrict__ bsupv,
    const T* __restrict__ gsqrt, const T* __restrict__ sqrtS_H,
    // For R precon: Z geometry (derivatives + odd-m coordinate value)
    const T* __restrict__ zs,  const T* __restrict__ zu12,
    const T* __restrict__ zu_e, const T* __restrict__ zu_o,
    const T* __restrict__ z_o,   // odd-m Z coordinate value (= vmecpp z1_o)
    // For Z precon: R geometry (derivatives + odd-m coordinate value)
    const T* __restrict__ rs,  const T* __restrict__ ru12,
    const T* __restrict__ ru_e, const T* __restrict__ ru_o,
    const T* __restrict__ r_o,   // odd-m R coordinate value (= vmecpp r1_o)
    int ns, int nZnT, T delta_s,
    // Outputs
    T* __restrict__ ax_R, T* __restrict__ ax_Z,
    T* __restrict__ bx_R, T* __restrict__ bx_Z,
    T* __restrict__ cx)
{
    T* s_buf = static_cast<T*>(dynSharedBase());
    int jH = blockIdx.x, tid = threadIdx.x;
    if (jH >= ns - 1) return;

    T sH = sqrtS_H[jH];
    T wInt = T(1.0) / T(nZnT);
    int base = jH * nZnT;

    // Accumulators: ax[4], bx[3], cx for R and Z
    T aR[4] = {T(0)}, aZ[4] = {T(0)};
    T bR[3] = {T(0)}, bZ[3] = {T(0)};
    T cx_v = T(0);

    for (int k = tid; k < nZnT; k += blockDim.x) {
        int idx = base + k;
        // Full-grid indices (same column-major layout, j+1 vs j surfaces)
        int idxF_i = idx;                     // inner full-grid (jH)
        int idxF_o = idx + nZnT;              // outer full-grid (jH+1)

        // Common pTau factor: -4 * r12 * totalP / tau * wInt
        T pTau = T(-4.0) * r12[idx] * totalP[idx] / tau[idx] * wInt;

        // sH reciprocal for odd-m corrections in bx terms
        T inv_sH = T(1.0) / sH;

        // ---- R preconditioner (uses Z geometry) ----
        {
            // t1a: radial derivative term (dominant)
            T t1a = zu12[idx] / delta_s;
            // t2a, t3a: parity-mixed corrections from full-grid
            T t2a = T(0.25) * (zu_e[idxF_o] / sH + zu_o[idxF_o]) / sH;
            T t3a = T(0.25) * (zu_e[idxF_i] / sH + zu_o[idxF_i]) / sH;

            aR[0] += pTau * t1a * t1a;
            aR[1] += pTau * (t1a + t2a) * (-t1a + t3a);
            aR[2] += pTau * (t1a + t2a) * (t1a + t2a);
            aR[3] += pTau * (-t1a + t3a) * (-t1a + t3a);

            // t1b, t2b: poloidal term, includes odd-m full-grid
            // correction matching vmecpp: 0.5*(xs + 0.5/sqrtSH * x1_o)
            // NOTE: x1_o is the coordinate VALUE (z1_o), not derivative (zu_o)
            T t1b = T(0.5) * (zs[idx] + T(0.5) * inv_sH * z_o[idxF_o]);
            T t2b = T(0.5) * (zs[idx] + T(0.5) * inv_sH * z_o[idxF_i]);
            bR[0] += pTau * t1b * t2b;
            bR[1] += pTau * t1b * t1b;
            bR[2] += pTau * t2b * t2b;
        }

        // ---- Z preconditioner (uses R geometry) ----
        {
            T t1a = ru12[idx] / delta_s;
            T t2a = T(0.25) * (ru_e[idxF_o] / sH + ru_o[idxF_o]) / sH;
            T t3a = T(0.25) * (ru_e[idxF_i] / sH + ru_o[idxF_i]) / sH;

            aZ[0] += pTau * t1a * t1a;
            aZ[1] += pTau * (t1a + t2a) * (-t1a + t3a);
            aZ[2] += pTau * (t1a + t2a) * (t1a + t2a);
            aZ[3] += pTau * (-t1a + t3a) * (-t1a + t3a);

            // t1b, t2b: radial term, includes odd-m full-grid
            // correction matching vmecpp: 0.5*(xs + 0.5/sqrtSH * x1_o)
            // NOTE: x1_o is the coordinate VALUE (r1_o), not derivative (ru_o)
            T t1b = T(0.5) * (rs[idx] + T(0.5) * inv_sH * r_o[idxF_o]);
            T t2b = T(0.5) * (rs[idx] + T(0.5) * inv_sH * r_o[idxF_i]);
            bZ[0] += pTau * t1b * t2b;
            bZ[1] += pTau * t1b * t1b;
            bZ[2] += pTau * t2b * t2b;
        }

        // cx: toroidal term (same for R and Z)
        // 0.25 * pFactor = 0.25 * (-4) = -1
        cx_v += -bsupv[idx] * bsupv[idx] * gsqrt[idx] * wInt;
    }

    // Parallel reduction for all 4+4+3+3+1 = 15 accumulators
    // Store in shared memory and reduce
    int nRed = 15;
    T* s = s_buf;
    s[tid] = aR[0]; s[tid + blockDim.x] = aR[1];
    s[tid + 2*blockDim.x] = aR[2]; s[tid + 3*blockDim.x] = aR[3];
    s[tid + 4*blockDim.x] = aZ[0]; s[tid + 5*blockDim.x] = aZ[1];
    s[tid + 6*blockDim.x] = aZ[2]; s[tid + 7*blockDim.x] = aZ[3];
    s[tid + 8*blockDim.x] = bR[0]; s[tid + 9*blockDim.x] = bR[1];
    s[tid + 10*blockDim.x] = bR[2];
    s[tid + 11*blockDim.x] = bZ[0]; s[tid + 12*blockDim.x] = bZ[1];
    s[tid + 13*blockDim.x] = bZ[2];
    s[tid + 14*blockDim.x] = cx_v;
    __syncthreads();

    for (int sVal = blockDim.x/2; sVal > 0; sVal >>= 1) {
        if (tid < sVal) {
            for (int c = 0; c < nRed; ++c)
                s[tid + c*blockDim.x] += s[tid + sVal + c*blockDim.x];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int jH4 = jH * 4, jH3 = jH * 3;
        ax_R[jH4+0]=s[0*blockDim.x]; ax_R[jH4+1]=s[1*blockDim.x];
        ax_R[jH4+2]=s[2*blockDim.x]; ax_R[jH4+3]=s[3*blockDim.x];
        ax_Z[jH4+0]=s[4*blockDim.x]; ax_Z[jH4+1]=s[5*blockDim.x];
        ax_Z[jH4+2]=s[6*blockDim.x]; ax_Z[jH4+3]=s[7*blockDim.x];
        bx_R[jH3+0]=s[8*blockDim.x]; bx_R[jH3+1]=s[9*blockDim.x];
        bx_R[jH3+2]=s[10*blockDim.x];
        bx_Z[jH3+0]=s[11*blockDim.x]; bx_Z[jH3+1]=s[12*blockDim.x];
        bx_Z[jH3+2]=s[13*blockDim.x];
        cx[jH] = s[14*blockDim.x];
    }
}

// ---------------------------------------------------------------------------
// Step 2: Assemble half-grid → full-grid with sm/sp parity factors.
// Also computes sm/sp on the fly.
// One block handles all half-grid surfaces.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void preconAssembleKernel(
    const T* __restrict__ ax_R, const T* __restrict__ ax_Z,
    const T* __restrict__ bx_R, const T* __restrict__ bx_Z,
    const T* __restrict__ cx,
    const T* __restrict__ sqrtS_H, const T* __restrict__ sqrtS_F,
    int ns,
    T* __restrict__ arm, T* __restrict__ brm,
    T* __restrict__ azm, T* __restrict__ bzm,
    T* __restrict__ ard, T* __restrict__ brd,
    T* __restrict__ azd, T* __restrict__ bzd,
    T* __restrict__ cxd,
    T* __restrict__ sm_out, T* __restrict__ sp_out)
{
    int jH = blockIdx.x * blockDim.x + threadIdx.x;
    if (jH >= ns - 1) return;

    int jH4 = jH * 4, jH3 = jH * 3;
    int jH_even = jH * 2;       // even parity index
    int jH_odd  = jH * 2 + 1;   // odd parity index

    // Compute sm/sp for this half-grid (vmecpp convention).
    // Half-grid jH sits between full-grid jH and jH+1.
    // sm[jH] = sqrtS_H[jH] / sqrtS_F[jH+1]  (ratio to OUTER full-grid)
    // sp[jH] = sqrtS_H[jH] / sqrtS_F[jH]    (ratio to INNER full-grid)
    T sh = sqrtS_H[jH];
    T sm = sh / sqrtS_F[jH + 1];  // outer — always safe (sqrtS_F[jH+1] > 0)
    T sp;
    if (jH == 0) {
        sp = sm;  // at innermost half-grid: inner full-grid is axis (s=0), so sp = sm
    } else {
        sp = sh / sqrtS_F[jH];  // inner — safe for jH > 0
    }
    sm_out[jH] = sm;
    sp_out[jH] = sp;

    T smsp = sm * sp;

    // ---- R preconditioner ----
    // Off-diagonal (half-grid)
    arm[jH_even] = -ax_R[jH4 + 0];            // even: -ax[0]
    arm[jH_odd]  = ax_R[jH4 + 1] * smsp;      // odd:  ax[1] * sm * sp
    brm[jH_even] = bx_R[jH3 + 0];             // even: bx[0]
    brm[jH_odd]  = bx_R[jH3 + 0] * smsp;      // odd:  bx[0] * sm * sp

    // ---- Z preconditioner ----
    azm[jH_even] = -ax_Z[jH4 + 0];
    azm[jH_odd]  = ax_Z[jH4 + 1] * smsp;
    bzm[jH_even] = bx_Z[jH3 + 0];
    bzm[jH_odd]  = bx_Z[jH3 + 0] * smsp;
}

// ---------------------------------------------------------------------------
// Step 2b: Average half-grid diagonals to full-grid.
// One thread per full-grid surface.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void preconDiagKernel(
    const T* __restrict__ ax_R, const T* __restrict__ ax_Z,
    const T* __restrict__ bx_R, const T* __restrict__ bx_Z,
    const T* __restrict__ cx,
    const T* __restrict__ sm, const T* __restrict__ sp,
    int ns,
    T* __restrict__ ard, T* __restrict__ brd,
    T* __restrict__ azd, T* __restrict__ bzd,
    T* __restrict__ cxd)
{
    int jF = blockIdx.x * blockDim.x + threadIdx.x;
    if (jF >= ns) return;

    int jF_even = jF * 2;
    int jF_odd  = jF * 2 + 1;

    // Inner half-grid index (jH_i = jF-1), outer (jH_o = jF)
    int jHi = jF - 1, jHo = jF;
    bool has_i = (jF > 0), has_o = (jF < ns - 1);

    // ---- R preconditioner diagonals ----
    // axd (radial d²/ds²):
    if (has_i && has_o) {
        ard[jF_even] = ax_R[jHi*4 + 0] + ax_R[jHo*4 + 0];
        brd[jF_even] = bx_R[jHi*3 + 1] + bx_R[jHo*3 + 2];
        // odd: ax[2]*sm² (inner) + ax[3]*sp² (outer)
        ard[jF_odd]  = ax_R[jHi*4 + 2] * sm[jHi] * sm[jHi]
                     + ax_R[jHo*4 + 3] * sp[jHo] * sp[jHo];
        brd[jF_odd]  = bx_R[jHi*3 + 1] * sm[jHi] * sm[jHi]
                     + bx_R[jHo*3 + 2] * sp[jHo] * sp[jHo];
        cxd[jF] = cx[jHi] + cx[jHo];
    } else if (has_o) {
        // jF == 0: only outer contribution
        ard[jF_even] = ax_R[jHo*4 + 0];
        ard[jF_odd]  = ax_R[jHo*4 + 3] * sp[jHo] * sp[jHo];
        brd[jF_even] = bx_R[jHo*3 + 2];
        brd[jF_odd]  = bx_R[jHo*3 + 2] * sp[jHo] * sp[jHo];
        cxd[jF] = cx[jHo];
    } else {
        // jF == ns-1: only inner contribution
        ard[jF_even] = ax_R[jHi*4 + 0];
        ard[jF_odd]  = ax_R[jHi*4 + 2] * sm[jHi] * sm[jHi];
        brd[jF_even] = bx_R[jHi*3 + 1];
        brd[jF_odd]  = bx_R[jHi*3 + 1] * sm[jHi] * sm[jHi];
        cxd[jF] = cx[jHi];
    }

    // ---- Z preconditioner diagonals ----
    if (has_i && has_o) {
        azd[jF_even] = ax_Z[jHi*4 + 0] + ax_Z[jHo*4 + 0];
        bzd[jF_even] = bx_Z[jHi*3 + 1] + bx_Z[jHo*3 + 2];
        azd[jF_odd]  = ax_Z[jHi*4 + 2] * sm[jHi] * sm[jHi]
                     + ax_Z[jHo*4 + 3] * sp[jHo] * sp[jHo];
        bzd[jF_odd]  = bx_Z[jHi*3 + 1] * sm[jHi] * sm[jHi]
                     + bx_Z[jHo*3 + 2] * sp[jHo] * sp[jHo];
    } else if (has_o) {
        azd[jF_even] = ax_Z[jHo*4 + 0];
        azd[jF_odd]  = ax_Z[jHo*4 + 3] * sp[jHo] * sp[jHo];
        bzd[jF_even] = bx_Z[jHo*3 + 2];
        bzd[jF_odd]  = bx_Z[jHo*3 + 2] * sp[jHo] * sp[jHo];
    } else {
        azd[jF_even] = ax_Z[jHi*4 + 0];
        azd[jF_odd]  = ax_Z[jHi*4 + 2] * sm[jHi] * sm[jHi];
        bzd[jF_even] = bx_Z[jHi*3 + 1];
        bzd[jF_odd]  = bx_Z[jHi*3 + 1] * sm[jHi] * sm[jHi];
    }

    // Edge pedestal for boundary stability
    const T edge_pedestal = T(0.05);
    if (jF == ns - 1) {
        ard[jF_even] *= T(1.0) + edge_pedestal;
        ard[jF_odd]  *= T(1.0) + edge_pedestal;
        brd[jF_even] *= T(1.0) + edge_pedestal;
        brd[jF_odd]  *= T(1.0) + edge_pedestal;
        azd[jF_even] *= T(1.0) + edge_pedestal;
        azd[jF_odd]  *= T(1.0) + edge_pedestal;
        bzd[jF_even] *= T(1.0) + edge_pedestal;
        bzd[jF_odd]  *= T(1.0) + edge_pedestal;
        cxd[jF]      *= T(1.0) + edge_pedestal;
    }
}

// ---------------------------------------------------------------------------
// Step 3: Assemble tridiagonal matrices per (m,n) mode.
// One thread per (mode, radial_surface).
//   ar[jF] : sup-diagonal on half-grid at jF (outer for forces at jF)
//   dr[jF] : diagonal on full-grid jF
//   br[jF] : sub-diagonal on half-grid at jF-1 (inner for forces at jF)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void tridiagAssemblyKernel(
    const T* __restrict__ arm, const T* __restrict__ brm,
    const T* __restrict__ azm, const T* __restrict__ bzm,
    const T* __restrict__ ard, const T* __restrict__ brd,
    const T* __restrict__ azd, const T* __restrict__ bzd,
    const T* __restrict__ cxd,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax, int nfp,
    T* __restrict__ ar, T* __restrict__ dr, T* __restrict__ br,
    T* __restrict__ az, T* __restrict__ dz, T* __restrict__ bz,
    int* __restrict__ jMin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = mnmax * ns;
    if (idx >= total) return;

    int mode = idx / ns;
    int jF   = idx - mode * ns;
    int mm = xm[mode], nn = xn[mode];
    int parity = mm % 2;  // m-parity: determines even/odd precon elements

    T m2 = T(mm * mm);
    T n2 = T(nn * nfp * nn * nfp);
    int nH = ns - 1;

    // Sup-diagonal: half-grid at jF (outer of forces surface jF)
    if (jF < nH) {
        int jH_par = jF * 2 + parity;
        ar[idx] = -(arm[jH_par] + brm[jH_par] * m2);
        az[idx] = -(azm[jH_par] + bzm[jH_par] * m2);
    } else {
        ar[idx] = T(0.0);
        az[idx] = T(0.0);
    }

    // Diagonal: full-grid at jF
    int jF_par = jF * 2 + parity;
    dr[idx] = -(ard[jF_par] + brd[jF_par] * m2 + cxd[jF] * n2);
    dz[idx] = -(azd[jF_par] + bzd[jF_par] * m2 + cxd[jF] * n2);

    // Sub-diagonal: half-grid at jF-1 (inner of forces surface jF)
    if (jF > 0) {
        int jH_par = (jF - 1) * 2 + parity;
        br[idx] = -(arm[jH_par] + brm[jH_par] * m2);
        bz[idx] = -(azm[jH_par] + bzm[jH_par] * m2);
    } else {
        br[idx] = T(0.0);
        bz[idx] = T(0.0);
    }

    // jMin: magnetic axis only gets m=0 contributions
    // (higher m modes vanish at axis).
    // Each mode has exactly one jF=0 entry — no race condition.
    if (jF == 0) {
        jMin[mode] = (mm == 0) ? 0 : 1;
    }

    // Special m=1 correction at jF=1: merge sub-diagonal into diagonal
    if (jF == 1 && mm == 1) {
        dr[idx] += br[idx];
        dz[idx] += bz[idx];
    }
}

// ---------------------------------------------------------------------------
// Step 4a: Lambda preconditioner — flux-surface averages on half grid.
// Port of vmecpp IdealMhdModel::updateLambdaPreconditioner (axisymmetric
// variant: n=0 only, so the guv/dLambda term drops out). The averages run
// over the reduced poloidal grid [0, pi] (the first ntheta/2+1 points of the
// full theta grid), matching vmecpp's nThetaEff under stellarator symmetry,
// with trapezoidal weights wInt from sizes.cc:
//   wInt[l] = 1/(nzeta*(nThetaReduced-1)), halved at l=0 and l=nThetaReduced-1
// Also accumulates rmsPhiP = sum(phipH^2) for lamscale.
// One block per half-grid surface; shifted layout: half-grid jH -> index jH+1.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void lambdaPrecAssembleKernel(
    const T* __restrict__ guu, const T* __restrict__ guv,
    const T* __restrict__ gvv,
    const T* __restrict__ gsqrt,
    const T* __restrict__ phipH,
    int ns, int nZnT, int ntheta, int nzeta,
    T* __restrict__ bLambda, T* __restrict__ dLambda,
    T* __restrict__ cLambda,
    T* __restrict__ rmsPhiP)
{
    int jH = blockIdx.x, tid = threadIdx.x;
    if (jH >= ns - 1) return;

    const int nThetaEven = 2 * (ntheta / 2);
    const int nThetaRed = nThetaEven / 2 + 1;  // reduced grid [0, pi]
    const T dnorm3 = T(1.0) / T(nzeta * (nThetaRed - 1));

    T bsum = T(0.0), dsum = T(0.0), csum = T(0.0);
    // Loop over ALL (zeta, reduced-theta) points: the layout is
    // [jH][zeta][theta], so the index must combine both. (FIXED 2026-08-02:
    // previously only theta at the first zeta plane was summed, making the
    // bLambda/dLambda/cLambda averages ~21-36x too small and the lambda
    // preconditioner ~32x too big near the LCFS.)
    for (int k = tid; k < nThetaRed * nzeta; k += blockDim.x) {
        int iz = k / nThetaRed, it = k % nThetaRed;
        T w = dnorm3;
        if (it == 0 || it == nThetaRed - 1) w *= T(0.5);
        int idx = (jH * nzeta + iz) * ntheta + it;
        bsum += guu[idx] / gsqrt[idx] * w;
        dsum += guv[idx] / gsqrt[idx] * w;   // 3D: toroidal coupling
        csum += gvv[idx] / gsqrt[idx] * w;
    }

    __shared__ T s_b[256], s_d[256], s_c[256];
    s_b[tid] = bsum; s_d[tid] = dsum; s_c[tid] = csum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s_b[tid] += s_b[tid + s]; s_d[tid] += s_d[tid + s]; s_c[tid] += s_c[tid + s]; }
        __syncthreads();
    }

    if (tid == 0) {
        bLambda[jH + 1] = s_b[0];
        dLambda[jH + 1] = s_d[0];
        cLambda[jH + 1] = s_c[0];
        // Deterministic rmsPhiP: one atomic from the last block, summing in
        // jH order (the original per-block atomicAdd's cross-block FP order
        // is scheduling-dependent, making rmsPhiP — and hence the lambda
        // preconditioner — vary by ~1 ulp run to run).
        if (jH == ns - 2) {
            double total = 0.0;
            for (int j = 0; j < ns - 1; ++j) total += phipH[j] * phipH[j];
            atomicAdd(rmsPhiP, total);
        }
    }
}

// ---------------------------------------------------------------------------
// Step 4b: Assemble the per-(mode, surface) lambda diagonal preconditioner.
// Port of vmecpp's lambdaPreconditioner assembly:
//   faclam = n^2*bLambda + 2mn*sign(b)*dLambda + m^2*cLambda
//          (n=0 here, so faclam = m^2*cLambda)
//   lambdaPrec = pFactor / faclam * sqrt(s)^pwr
// with pFactor = kLambdaPreconditionerDampingFactor / (4*lamscale^2),
// lamscale = sqrt(rmsPhiP * deltaS), and
// pwr = min(m^2 / 16^2, 8) suppressing high-m modes.
// The m=0 (n=0) mode is skipped (stays 0), matching vmecpp — lambda_00 is
// a gauge mode and is zeroed by the preconditioner.
// Full-grid averaging follows vmecpp's shifted layout:
//   full[jF] = 0.5*(b[jF+1] + b[jF]) for jF=1..ns-1 (LCFS reads b[ns] = 0).
// One block per mode.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void lambdaPrecFinalizeKernel(
    const T* __restrict__ bLambda, const T* __restrict__ dLambda,
    const T* __restrict__ cLambda,
    const T* __restrict__ sqrtS_F, const T* __restrict__ rmsPhiP,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax, T delta_s, int nfp,
    T* __restrict__ lambdaPrec)
{
    // One thread per (mode, jF) element — the original ran one thread per
    // mode with a serial ns loop (<<<mnmax, 1>>>). Per-element arithmetic
    // is unchanged (the per-mode scalars are recomputed per thread).
    int mode = blockIdx.x;
    int jF = blockIdx.y * blockDim.x + threadIdx.x;
    if (mode >= mnmax || jF >= ns) return;

    int m = xm[mode], n = xn[mode];
    // Guard: an all-zero phip profile (rmsPhiP == 0) would make lamscale == 0
    // and pFactor infinite. The tiny floor keeps the lambda preconditioner
    // finite; the degenerate geometry is failed earlier by the solver's
    // jacobian-stats check anyway.
    T lamscale = sqrt(fmax(rmsPhiP[0] * delta_s, T(1e-30)));
    const T pFactor = T(2.0) / (T(4.0) * lamscale * lamscale);
    const T pwr = fmin(T(m * m) / T(16.0 * 16.0), T(8.0));
    // faclam terms (vmecpp updateLambdaPreconditioner):
    //   tnn = (n*nfp)^2, tmn = 2*m*n*nfp  (nfp passed separately)
    T tnn = T(n * n) * T(nfp * nfp);
    T tmn = T(2.0) * T(m * n) * T(nfp);

    if (jF == 0) { lambdaPrec[mode * ns] = T(0.0); return; }
    if (m == 0 && n == 0) { lambdaPrec[mode * ns + jF] = T(0.0); return; }  // gauge mode

    T bFull = T(0.5) * (bLambda[jF + 1] + bLambda[jF]);
    T dFull = T(0.5) * (dLambda[jF + 1] + dLambda[jF]);
    T cFull = T(0.5) * (cLambda[jF + 1] + cLambda[jF]);
    T faclam = tnn * bFull + tmn * copysign(dFull, bFull) +
               T(m * m) * cFull;
    if (faclam == T(0.0)) faclam = T(-1.0e-10);  // kLambdaPreconditionerZeroGuard
    lambdaPrec[mode * ns + jF] =
        pFactor / faclam * pow(sqrtS_F[jF], pwr);
}

// ---------------------------------------------------------------------------
// Step 4: Thomas algorithm — one thread per (m,n) mode.
// Solves the tridiagonal system in-place on the 5 RHS components.
// Layout: f[component * mnmax * ns + mode * ns + jF]
//   comp 0=rmncc (R even)  → R tridiag
//   comp 1=zmnsc (Z even)  → Z tridiag
//   comp 2=lmnsc (lambda)  → lambda diagonal preconditioner (no tridiag)
//   comp 3=rmnss (R odd)   → R tridiag
//   comp 4=zmncs (Z odd)   → Z tridiag
//
// Fixed boundary: the solve covers rows jMin..ns-2 (jMax = ns-1), matching
// vmecpp's applyRZPreconditioner (jMax = fc_.ns - 1 for lfreeb=false). The
// LCFS row does not participate in the tridiagonal solve.
//
// Parallel cyclic reduction (PCR), 128 threads per block, one block per
// mode. The original kernel ran a serial Thomas with one thread per block:
// each of the ~500 dependent fp64 steps (with division) paid full latency
// with zero ILP — ~600 us/iter on W7-X, the single largest kernel. PCR
// eliminates the coupling distance doubling per round (k = 1,2,4,...) with
// all rows updated in parallel: ~7 rounds for ns ~ 100, each round one
// dependent reciprocal. Measured ~60 us/iter (~10x). The arithmetic differs
// from Thomas at the rounding level (different elimination order) — the
// solve is verified against the Thomas result at ~1e-12 in the benchmark.
//
// Naming follows the original kernel: a_in = upper coeff (x[j+1]),
// b_in = lower coeff (x[j-1]), d_in = diagonal. The R system (comps 0,3)
// uses ar/dr/br; the Z system (comps 1,4) uses az/dz/bz; the two passes
// reuse the same shared coefficient buffers.
//
// Shared layout: cL,cD,cU,nL,nD,nU (3+3 arrays of ns) + cF,nF (2+2 RHS
// rows of ns) = 10*ns doubles. ns ~ 100, so ~8KB.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void tridiagSolveKernel(
    T* __restrict__ f,
    const T* __restrict__ ar, const T* __restrict__ dr,
    const T* __restrict__ br,
    const T* __restrict__ az, const T* __restrict__ dz,
    const T* __restrict__ bz,
    const T* __restrict__ lambdaPrec,
    const int* __restrict__ jMin,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax)
{
    int mode = blockIdx.x;
    if (mode >= mnmax) return;
    int tid = threadIdx.x;
    int jMin_m = jMin[mode];
    int jMax = ns - 1;          // LCFS row excluded
    int nRow = jMax - jMin_m;   // solved rows jMin_m .. jMax-1
    int stride = mnmax * ns;

    // The 128-thread block covers the solved rows in a grid-stride loop (the
    // old kernel assumed nRow <= 128: for ns > 129 the tail rows were
    // silently left at their staged values — an incomplete solve). Each row
    // of a PCR round depends only on the PREVIOUS round's arrays, so the
    // per-row arithmetic is unchanged; a __syncthreads() between rounds
    // preserves the dependency. For the shipped grids (ns <= 99) the stride
    // loop runs exactly once per round — bit-identical to the old kernel.

    // Shared: cur L,d,U; nxt L,d,U; f rows (2 per pass) cur+nxt
    T* s_tri = static_cast<T*>(dynSharedBase());
    T* cL = s_tri;            // [ns]
    T* cD = cL + ns;
    T* cU = cD + ns;
    T* nL = cU + ns;
    T* nD = nL + ns;
    T* nU = nD + ns;
    T* cF = nU + ns;          // [2][ns]
    T* nF = cF + 2 * ns;      // [2][ns]

    // One PCR pass for the system (upper, diag, lower) applied to the two
    // RHS components cA and cB; writes the solutions back to f.
    auto pass = [&](const T* aU, const T* dD, const T* bL,
                    int cA, int cB) {
        const int comp[2] = {cA, cB};
        // stage lower=L=b_in, diag=D=d_in, upper=U=a_in + 2 RHS rows
        for (int j = tid; j < ns; j += blockDim.x) {
            cL[j] = bL[mode * ns + j];
            cD[j] = dD[mode * ns + j];
            cU[j] = aU[mode * ns + j];
            cF[0 * ns + j] = f[comp[0] * stride + mode * ns + j];
            cF[1 * ns + j] = f[comp[1] * stride + mode * ns + j];
        }
        // zero the j < jMin region of the staged RHS (mirrors Thomas zeroing)
        for (int j = tid; j < jMin_m; j += blockDim.x) {
            cF[0 * ns + j] = T(0.0);
            cF[1 * ns + j] = T(0.0);
        }
        __syncthreads();
        // PCR rounds: eliminate the two neighbors at distance k each round.
        // Each round processes ALL solved rows via a grid-stride loop; the
        // rounds are separated by __syncthreads (round r+1 reads every row
        // written in round r, so the tiles cannot overlap).
        for (int k = 1; k <= nRow; k <<= 1) {
            for (int r = tid; r < nRow; r += blockDim.x) {
                int j = jMin_m + r;
                bool hasL = (j - k >= jMin_m), hasR = (j + k < jMax);
                T dL = hasL ? cD[j - k] : T(0.0);
                T dR = hasR ? cD[j + k] : T(0.0);
                // Pivot guard with sign preservation: a tiny (or zero)
                // diagonal is replaced by a tiny value of the SAME sign —
                // the old clamp to +1e-30 flipped the pivot (and hence the
                // solve direction) for negative diagonals.
                if (fabs(dL) < T(1e-30)) dL = copysign(T(1e-30), dL);
                if (fabs(dR) < T(1e-30)) dR = copysign(T(1e-30), dR);
                T invL = hasL ? T(1.0) / dL : T(0.0);
                T invR = hasR ? T(1.0) / dR : T(0.0);
                T L = cL[j], D = cD[j], U = cU[j];
                T Ll = hasL ? cL[j - k] : T(0.0), Ul = hasL ? cU[j - k] : T(0.0);
                T Lr = hasR ? cL[j + k] : T(0.0), Ur = hasR ? cU[j + k] : T(0.0);
                nL[j] = -L * Ll * invL;
                nU[j] = -U * Ur * invR;
                T nDv = D - L * Ul * invL - U * Lr * invR;
                if (fabs(nDv) < T(1e-30)) nDv = copysign(T(1e-30), nDv);
                nD[j] = nDv;
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    T fv = cF[c * ns + j];
                    T fl = hasL ? cF[c * ns + j - k] : T(0.0);
                    T fr = hasR ? cF[c * ns + j + k] : T(0.0);
                    nF[c * ns + j] = fv - L * fl * invL - U * fr * invR;
                }
            }
            __syncthreads();
            T* t;
            t = cL; cL = nL; nL = t;
            t = cD; cD = nD; nD = t;
            t = cU; cU = nU; nU = t;
            t = cF; cF = nF; nF = t;
            __syncthreads();
        }
        // ---- Final solve: x = f'' / d'' (decoupled after the rounds) ----
        for (int r = tid; r < nRow; r += blockDim.x) {
            int j = jMin_m + r;
            T d = cD[j];
            if (fabs(d) < T(1e-30)) d = copysign(T(1e-30), d);
            T inv = T(1.0) / d;
            f[comp[0] * stride + mode * ns + j] = cF[0 * ns + j] * inv;
            f[comp[1] * stride + mode * ns + j] = cF[1 * ns + j] * inv;
        }
    };
    pass(ar, dr, br, 0, 3);
    __syncthreads();
    pass(az, dz, bz, 1, 4);

    // ---- Zero out RHS for j < jMin (identity matrix region): comps 0..4
    // (matching the original kernel's zeroing; comp 5 is not zeroed) ----
    for (int j = tid; j < jMin_m; j += blockDim.x) {
        f[0 * stride + mode * ns + j] = T(0.0);
        f[1 * stride + mode * ns + j] = T(0.0);
        f[2 * stride + mode * ns + j] = T(0.0);
        f[3 * stride + mode * ns + j] = T(0.0);
        f[4 * stride + mode * ns + j] = T(0.0);
    }

    // ---- Components 2 (lmnsc) and 5 (lmncs): lambda diagonal
    // preconditioner ----
    // vmecpp's applyLambdaPreconditioner: flsc/flcs[mode, jF] *= lambdaPrec
    // for all surfaces 0..ns-1 (including the LCFS). lambdaPrec is zero
    // for the m=0 mode, at the axis (jF=0), and below... no jMin filter —
    // the axis m>0 entries are zeroed by lambdaPrec itself (sqrt(s)^pwr=0).
    for (int j = tid; j < ns; j += blockDim.x) {
        T lp = lambdaPrec[mode * ns + j];
        f[2 * stride + mode * ns + j] *= lp;
        f[5 * stride + mode * ns + j] *= lp;
    }
}


// ---------------------------------------------------------------------------
// Host-side orchestration
// ---------------------------------------------------------------------------
template <typename T>
void preconCompute(const FourierPlan<T>& fp, const GridParams<T>& p,
                   const RadialProfiles<T>& rp, const MetricWorkspace<T>& mw,
                   PreconWorkspace<T>& pw) {
    int nH = p.ns - 1, nF = p.ns;
    int threads = 256;
    size_t smem = threads * 15 * sizeof(T);  // 15 accumulators

    // Step 1: Compute ax, bx, cx on half-grid
    preconComputeKernel<T><<<nH, threads, smem>>>(
        mw.d_r12, mw.d_tau, mw.d_totalPressure, mw.d_bsupv, mw.d_gsqrt,
        rp.d_sqrtS_H,
        mw.d_zs, mw.d_zu12, fp.d_zu_e, fp.d_zu_o, fp.d_z_o,
        mw.d_rs, mw.d_ru12, fp.d_ru_e, fp.d_ru_o, fp.d_r_o,
        p.ns, p.nZnT, rp.delta_s,
        pw.d_ax_R, pw.d_ax_Z, pw.d_bx_R, pw.d_bx_Z, pw.d_cx);
    cc(cudaGetLastError(), "preconCompute");

    // Step 2a: Assemble off-diagonal terms + sm/sp on half-grid
    int gridH = (nH + 255) / 256;
    preconAssembleKernel<T><<<gridH, 256>>>(
        pw.d_ax_R, pw.d_ax_Z, pw.d_bx_R, pw.d_bx_Z, pw.d_cx,
        rp.d_sqrtS_H, rp.d_sqrtS_F,
        p.ns,
        pw.d_arm, pw.d_brm, pw.d_azm, pw.d_bzm,
        pw.d_ard, pw.d_brd, pw.d_azd, pw.d_bzd, pw.d_cxd,
        pw.d_sm, pw.d_sp);
    cc(cudaGetLastError(), "preconAssemble");

    // Step 2b: Average half-grid diagonals to full-grid
    int gridF = (nF + 255) / 256;
    preconDiagKernel<T><<<gridF, 256>>>(
        pw.d_ax_R, pw.d_ax_Z, pw.d_bx_R, pw.d_bx_Z, pw.d_cx,
        pw.d_sm, pw.d_sp, p.ns,
        pw.d_ard, pw.d_brd, pw.d_azd, pw.d_bzd, pw.d_cxd);
    cc(cudaGetLastError(), "preconDiag");

    // Step 3: Assemble tridiagonal matrices per (m,n) mode
    int total = p.mnmax * nF;
    int gridMN = (total + 255) / 256;
    tridiagAssemblyKernel<T><<<gridMN, 256>>>(
        pw.d_arm, pw.d_brm, pw.d_azm, pw.d_bzm,
        pw.d_ard, pw.d_brd, pw.d_azd, pw.d_bzd, pw.d_cxd,
        fp.basis.d_xm, fp.basis.d_xn,
        p.ns, p.mnmax, p.nfp,
        pw.d_ar, pw.d_dr, pw.d_br,
        pw.d_az, pw.d_dz, pw.d_bz,
        pw.d_jMin);
    cc(cudaGetLastError(), "tridiagAssembly");

    // Step 4a/4b: Lambda diagonal preconditioner (components 2 and 5)
    {
        cc(cudaMemset(pw.d_rmsPhiP, 0, sizeof(T)), "rmsPhiP zero");
        lambdaPrecAssembleKernel<T><<<nH, threads>>>(
            mw.d_guu, mw.d_guv, mw.d_gvv, mw.d_gsqrt,
            rp.d_phip_H,
            p.ns, p.nZnT, p.ntheta, p.nzeta,
            pw.d_bLambda, pw.d_dLambda, pw.d_cLambda, pw.d_rmsPhiP);
        cc(cudaGetLastError(), "lambdaPrecAssemble");
        lambdaPrecFinalizeKernel<T><<<dim3(p.mnmax, (p.ns + 127) / 128), 128>>>(
            pw.d_bLambda, pw.d_dLambda, pw.d_cLambda,
            rp.d_sqrtS_F, pw.d_rmsPhiP,
            fp.basis.d_xm, fp.basis.d_xn,
            p.ns, p.mnmax, rp.delta_s, p.nfp,
            pw.d_lambdaPrec);
        cc(cudaGetLastError(), "lambdaPrecFinalize");
    }
}

template <typename T>
void preconApply(T* d_f_inout, const GridParams<T>& p,
                 const PreconWorkspace<T>& pw,
                 const int* xm, const int* xn) {
    // Step 4: PCR solve — one block per mode, 128 threads per block
    // (the threads cover up to ns-1 solved rows; PCR rounds in parallel)
    size_t smem = 10 * p.ns * sizeof(T);  // coeffs + RHS buffers
    tridiagSolveKernel<T><<<p.mnmax, 128, smem>>>(
        d_f_inout,
        pw.d_ar, pw.d_dr, pw.d_br,
        pw.d_az, pw.d_dz, pw.d_bz,
        pw.d_lambdaPrec,
        pw.d_jMin, xm, xn,
        p.ns, p.mnmax);
    cc(cudaGetLastError(), "tridiagSolve");
}

// ---- Explicit instantiation (double + float) ----------------------------
template PreconWorkspace<double> preconCreate<double>(const GridParams<double>&);
template PreconWorkspace<float>  preconCreate<float>(const GridParams<float>&);
template void preconFree<double>(PreconWorkspace<double>&);
template void preconFree<float>(PreconWorkspace<float>&);
template void preconCompute<double>(const FourierPlan<double>&, const GridParams<double>&, const RadialProfiles<double>&, const MetricWorkspace<double>&, PreconWorkspace<double>&);
template void preconCompute<float>(const FourierPlan<float>&, const GridParams<float>&, const RadialProfiles<float>&, const MetricWorkspace<float>&, PreconWorkspace<float>&);
template void preconApply<double>(double*, const GridParams<double>&, const PreconWorkspace<double>&, const int*, const int*);
template void preconApply<float>(float*, const GridParams<float>&, const PreconWorkspace<float>&, const int*, const int*);
