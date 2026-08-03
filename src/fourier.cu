// fourier.cu — DFT transforms with vmecpp m-parity convention.
//
// Internal (folded, n>=0) product basis, matching vmecpp's
// FourierToReal3DSymmFastPoloidal (dft_toroidal.cc):
//   R = rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ)
//   Z = zmnsc*sin(mθ)cos(nζ) + zmncs*cos(mθ)sin(nζ)
//   λ = lmnsc*sin(mθ)cos(nζ) + lmncs*cos(mθ)sin(nζ)
// Mode index: mode = m*(ntor+1) + n, m = 0..mpol-1, n = 0..ntor;
// xn = n*nfp (toroidal mode number per field period).
//
// The R/Z coefficients are the plain physical cos(mθ-nζ) amplitudes (unlike
// vmecpp's internal state, which stores them divided by mscale*nscale; the
// real-space reconstruction is identical because vmecpp's basis tables carry
// the mscale/nscale). Lambda is physical in both codes; its real-space
// reconstruction carries mscale*nscale (the λ basis is mscale'd in vmecpp
// and cuMES compensates with the per-mode factor in the inverse DFT below).
//
// The toroidal derivative of lambda is stored as -∂λ/∂ζ (vmecpp convention:
// lv = -(lmksc_n*sinmu + lmkcs_n*cosmu) with sinnvn = -n*nfp*sin(nζ) and
// cosnvn = +n*nfp*cos(nζ)); bsupu = (lamscale*lv + chip')/√g.

#include "fourier.cuh"
#include "constraint.cuh"
#include <cstdio>
#include <cmath>

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
static void checkCufft(cufftResult r, const char* tag) {
    if (r != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT error [%s]: %d\n", tag, (int)r);
        exit(EXIT_FAILURE);
    }
}

FourierPlan fourierCreate(const GridParams& p, cublasHandle_t handle) {
    FourierPlan fp{}; fp.handle = handle;
    int nZnT = p.nZnT, mnmax = p.mnmax;
    size_t nbytes_basis = nZnT * mnmax * sizeof(double);
    size_t nbytes_mode  = mnmax * sizeof(int);
    size_t nbytes_real  = p.ns * nZnT * sizeof(double);

    auto* h_xm  = new int[mnmax], *h_xn = new int[mnmax];

    // Folded mode table: m = 0..mpol-1, n = 0..ntor, mode = m*(ntor+1)+n.
    // xn = the RAW toroidal mode number n. The zeta grid covers one field
    // period (zeta = 2*pi*k/nzeta, physical phi = zeta/nfp), so the basis is
    // cos(m*theta - n*zeta) — matching vmecpp (fourier_basis.cc:122,137).
    // The toroidal DERIVATIVES carry the nfp factor (cosnvn = n*nfp*cosnv).
    for (int m = 0; m < p.mpol; ++m)
        for (int n = 0; n < p.ntor + 1; ++n) {
            int mode = m * (p.ntor + 1) + n;
            h_xm[mode] = m; h_xn[mode] = n;
        }
    checkCuda(cudaMalloc(&fp.basis.d_xm, nbytes_mode), "xm");
    checkCuda(cudaMalloc(&fp.basis.d_xn, nbytes_mode), "xn");
    checkCuda(cudaMemcpy(fp.basis.d_xm, h_xm, nbytes_mode, cudaMemcpyHostToDevice), "xm");
    checkCuda(cudaMemcpy(fp.basis.d_xn, h_xn, nbytes_mode, cudaMemcpyHostToDevice), "xn");
    delete[] h_xm; delete[] h_xn;

    auto am = [&](double*& p, const char* n) { checkCuda(cudaMalloc(&p, nbytes_real), n); };
    am(fp.d_r_e,"r_e"); am(fp.d_z_e,"z_e"); am(fp.d_l_e,"l_e");
    am(fp.d_ru_e,"ru_e"); am(fp.d_zu_e,"zu_e"); am(fp.d_lu_e,"lu_e");
    am(fp.d_r_o,"r_o"); am(fp.d_z_o,"z_o"); am(fp.d_l_o,"l_o");
    am(fp.d_ru_o,"ru_o"); am(fp.d_zu_o,"zu_o"); am(fp.d_lu_o,"lu_o");
    am(fp.d_r_real,"r_r"); am(fp.d_z_real,"z_r"); am(fp.d_l_real,"l_r");
    am(fp.d_ru_real,"ru_r"); am(fp.d_zu_real,"zu_r"); am(fp.d_lu_real,"lu_r");
    am(fp.d_rv_real,"rv_r"); am(fp.d_zv_real,"zv_r"); am(fp.d_lv_real,"lv_r");
    // Parity-split toroidal derivatives (3D geometry needs them separately)
    am(fp.d_rv_e,"rve"); am(fp.d_rv_o,"rvo");
    am(fp.d_zv_e,"zve"); am(fp.d_zv_o,"zvo");
    am(fp.d_lv_e,"lve"); am(fp.d_lv_o,"lvo");
    am(fp.d_armn_e,"ae"); am(fp.d_armn_o,"ao");
    am(fp.d_azmn_e,"aze"); am(fp.d_azmn_o,"azo");
    am(fp.d_brmn_e,"be"); am(fp.d_brmn_o,"bo");
    am(fp.d_bzmn_e,"bze"); am(fp.d_bzmn_o,"bzo");
    am(fp.d_blmn_e,"ble"); am(fp.d_blmn_o,"blo");
    am(fp.d_crmn_e,"ce"); am(fp.d_crmn_o,"co");
    am(fp.d_czmn_e,"cze"); am(fp.d_czmn_o,"czo");
    am(fp.d_clmn_e,"cle"); am(fp.d_clmn_o,"clo");
    am(fp.d_fr_real,"fr"); am(fp.d_fz_real,"fz"); am(fp.d_fl_real,"fl");

    // ---- cuFFT backend: poloidal tables, scratch, plans ----
    checkCuda(cudaMalloc(&fp.d_zeta_spectra,
        (size_t)12 * p.mpol * p.ns * (p.nzeta / 2 + 1) * sizeof(double2)), "spectra");
    checkCuda(cudaMalloc(&fp.d_zeta_real,
        (size_t)12 * p.mpol * p.ns * p.nzeta * sizeof(double)), "zeta");
    auto amt = [&](double*& q, const char* n, size_t cnt) {
        checkCuda(cudaMalloc(&q, cnt * sizeof(double)), n);
    };
    amt(fp.d_cos_th, "costh", p.mpol * p.ntheta);  amt(fp.d_sin_th, "sinth", p.mpol * p.ntheta);
    amt(fp.d_mcos_th, "mcosth", p.mpol * p.ntheta); amt(fp.d_msin_th, "msinth", p.mpol * p.ntheta);
    amt(fp.d_fwd_w, "fwdw", p.ntheta / 2 + 1);

    auto* h_cos_th  = new double[p.mpol * p.ntheta];
    auto* h_sin_th  = new double[p.mpol * p.ntheta];
    auto* h_mcos_th = new double[p.mpol * p.ntheta];
    auto* h_msin_th = new double[p.mpol * p.ntheta];
    for (int m = 0; m < p.mpol; ++m)
        for (int l = 0; l < p.ntheta; ++l) {
            double th = 2.0 * M_PI * l / p.ntheta;
            double c = cos(m * th), s = sin(m * th);
            h_cos_th[m * p.ntheta + l]  = c;
            h_sin_th[m * p.ntheta + l]  = s;
            h_mcos_th[m * p.ntheta + l] = m * c;
            h_msin_th[m * p.ntheta + l] = -m * s;
        }
    // Reduced-grid trapezoid weights for the forward quadrature (vmecpp
    // intNorm = 1/(nZeta*(nThetaRed-1)), endpoint-halved).
    int nThetaRed = p.ntheta / 2 + 1;
    auto* h_fwd_w = new double[nThetaRed];
    double intNorm = 1.0 / ((double)p.nzeta * (nThetaRed - 1));
    for (int l = 0; l < nThetaRed; ++l) {
        h_fwd_w[l] = intNorm;
        if (l == 0 || l == nThetaRed - 1) h_fwd_w[l] *= 0.5;
    }
    checkCuda(cudaMemcpy(fp.d_cos_th, h_cos_th, (size_t)p.mpol * p.ntheta * sizeof(double),
                         cudaMemcpyHostToDevice), "cp costh");
    checkCuda(cudaMemcpy(fp.d_sin_th, h_sin_th, (size_t)p.mpol * p.ntheta * sizeof(double),
                         cudaMemcpyHostToDevice), "cp sinth");
    checkCuda(cudaMemcpy(fp.d_mcos_th, h_mcos_th, (size_t)p.mpol * p.ntheta * sizeof(double),
                         cudaMemcpyHostToDevice), "cp mcosth");
    checkCuda(cudaMemcpy(fp.d_msin_th, h_msin_th, (size_t)p.mpol * p.ntheta * sizeof(double),
                         cudaMemcpyHostToDevice), "cp msinth");
    checkCuda(cudaMemcpy(fp.d_fwd_w, h_fwd_w, (size_t)nThetaRed * sizeof(double),
                         cudaMemcpyHostToDevice), "cp fwdw");
    delete[] h_cos_th; delete[] h_sin_th; delete[] h_mcos_th; delete[] h_msin_th;
    delete[] h_fwd_w;

    // Batched 1D real FFTs of length nzeta, one batch element per (slot, m, j).
    int n = p.nzeta, nz2 = p.nzeta / 2 + 1;
    int batch = 12 * p.mpol * p.ns;
    int inemb = nz2, outemb = n;
    checkCufft(cufftPlanMany(&fp.plan_z2d, 1, &n, &inemb, 1, nz2,
                             &outemb, 1, n, CUFFT_Z2D, batch), "plan z2d");
    checkCufft(cufftPlanMany(&fp.plan_d2z, 1, &n, &outemb, 1, n,
                             &inemb, 1, nz2, CUFFT_D2Z, batch), "plan d2z");
    return fp;
}

void fourierFree(FourierPlan& fp) {
    auto cuFree = [](double* p) { cudaFree(p); };
    cudaFree(fp.basis.d_xm); cudaFree(fp.basis.d_xn);
    cuFree(fp.d_r_e);cuFree(fp.d_z_e);cuFree(fp.d_l_e);
    cuFree(fp.d_ru_e);cuFree(fp.d_zu_e);cuFree(fp.d_lu_e);
    cuFree(fp.d_r_o);cuFree(fp.d_z_o);cuFree(fp.d_l_o);
    cuFree(fp.d_ru_o);cuFree(fp.d_zu_o);cuFree(fp.d_lu_o);
    cuFree(fp.d_r_real);cuFree(fp.d_z_real);cuFree(fp.d_l_real);
    cuFree(fp.d_ru_real);cuFree(fp.d_zu_real);cuFree(fp.d_lu_real);
    cuFree(fp.d_rv_real);cuFree(fp.d_zv_real);cuFree(fp.d_lv_real);
    cuFree(fp.d_rv_e);cuFree(fp.d_rv_o);
    cuFree(fp.d_zv_e);cuFree(fp.d_zv_o);
    cuFree(fp.d_lv_e);cuFree(fp.d_lv_o);
    cuFree(fp.d_armn_e);cuFree(fp.d_armn_o);cuFree(fp.d_azmn_e);cuFree(fp.d_azmn_o);
    cuFree(fp.d_brmn_e);cuFree(fp.d_brmn_o);cuFree(fp.d_bzmn_e);cuFree(fp.d_bzmn_o);
    cuFree(fp.d_blmn_e);cuFree(fp.d_blmn_o);
    cuFree(fp.d_crmn_e);cuFree(fp.d_crmn_o);cuFree(fp.d_czmn_e);cuFree(fp.d_czmn_o);
    cuFree(fp.d_clmn_e);cuFree(fp.d_clmn_o);
    cuFree(fp.d_fr_real);cuFree(fp.d_fz_real);cuFree(fp.d_fl_real);
    cufftDestroy(fp.plan_z2d); cufftDestroy(fp.plan_d2z);
    cudaFree(fp.d_zeta_spectra); cudaFree(fp.d_zeta_real);
    cudaFree(fp.d_cos_th); cudaFree(fp.d_sin_th);
    cudaFree(fp.d_mcos_th); cudaFree(fp.d_msin_th); cudaFree(fp.d_fwd_w);
}

// ---- cuFFT backend: inverse ---------------------------------------------
// Mirror of vmecpp's fft_toroidal.cc (FourierToReal3DSymmFastPoloidal):
// the ζ direction is a real inverse FFT (cuFFT Z2D), the θ direction stays a
// direct basis-table sum. All nine real-space outputs (r/z/l, their θ and ζ
// derivatives) come from the 12 slots; the m-parity split into e/o arrays and
// the odd-m scalxc division (maxsc) happen on the target side, as in vmecpp.
__global__ void inversePackKernel(
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const double* __restrict__ lmnsc, const double* __restrict__ lmncs,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mpol, int ntor, int nfp, int nz2,
    double2* __restrict__ spectra)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ns * mpol * (ntor + 1)) return;
    int j = t % ns, mode = t / ns;
    int m = xm[mode], n = xn[mode];
    double nf = n * nfp;
    double rc = rmncc[j + mode * ns], rs = rmnss[j + mode * ns];
    double zs = zmnsc[j + mode * ns], zc = zmncs[j + mode * ns];
    double lsc = lmnsc[j + mode * ns], lcs = lmncs[j + mode * ns];
    // cuFFT's Z2D synthesis is f[k] = X[0] + 2*Σ Re(X[n])cos - 2*Σ Im(X[n])sin,
    // so the n>=1 bins are halved (cancelling the 2× exactly) and the DST
    // slots carry a minus (vmecpp FillDct/FillDst). The ζ-derivative slots
    // carry the n*nfp factor with vmecpp's signs (FillDctDeriv: Im = +spec*n*nfp,
    // FillDstDeriv: Re = +spec*n*nfp). The n=0 bins of the DST/derivative
    // slots stay 0 (sin(nζ) = n*nfp = 0); no mscale/nscale — the cuMES state
    // is plain physical (λ carries ms*ns inside the state values).
    double half   = (n == 0) ? 1.0 : 0.5;
    double shalf  = (n == 0) ? 0.0 : 0.5;
    double dhalf  = (n == 0) ? 0.0 : 0.5 * nf;
    size_t step = (size_t)mpol * ns * nz2;
    double2* slot = spectra + ((size_t)m * ns + j) * nz2 + n;
    slot[0 * step] = make_double2(rc * half, 0.0);
    slot[1 * step] = make_double2(0.0, -rs * shalf);
    slot[2 * step] = make_double2(0.0, +rc * dhalf);
    slot[3 * step] = make_double2(+rs * dhalf, 0.0);
    slot[4 * step] = make_double2(zs * half, 0.0);
    slot[5 * step] = make_double2(0.0, -zc * shalf);
    slot[6 * step] = make_double2(0.0, +zs * dhalf);
    slot[7 * step] = make_double2(+zc * dhalf, 0.0);
    slot[8 * step] = make_double2(lsc * half, 0.0);
    slot[9 * step] = make_double2(0.0, -lcs * shalf);
    slot[10 * step] = make_double2(0.0, +lsc * dhalf);
    slot[11 * step] = make_double2(+lcs * dhalf, 0.0);
}

// Poloidal accumulation over the FULL θ grid. Grid (ns, 3): blockIdx.y
// selects the slot group (0 = R slots 0-3, 1 = Z slots 4-7, 2 = λ slots
// 8-11), blockIdx.x the surface j. Threads (l1=θ-half, k=ζ); each thread
// handles θ points l1 and l1 + ntheta/2 and sums the m-loop serially — the
// same accumulation order as inverseDFTKernel — then writes each output
// point exactly once. Outputs the e/o parity arrays: even m -> *_e, odd m
// -> *_o divided by maxsc (vmecpp's scalxc odd decomposition). Splitting
// the 12 slots into 3 groups cuts the shared memory per block 41.5 KB ->
// 13.8 KB, raising occupancy from 1 to 3 blocks/SM (was latency-bound at
// ~425 us/iter; the group selection is uniform per block, so the per-thread
// arithmetic is unchanged — bit-identical).
//
// Slot-to-output mapping (verified against inverseDFTKernel and
// fft_toroidal.cc). Per group, slots are (cos-slot, sin-slot, cosN-slot,
// sinN-slot) and the θ basis differs: R multiplies (cos, sin), Z and λ
// multiply (sin, cos), and λ's ζ-derivative is negated:
//   R: v = c0*cos + c1*sin      vu = c0*msin + c1*mcos
//      vv = c2*cos + c3*sin
//   Z: v = c0*sin + c1*cos      vu = c0*mcos + c1*msin
//      vv = c2*sin + c3*cos
//   λ: v = c0*sin + c1*cos      vu = c0*mcos + c1*msin
//      vv = -(c2*sin + c3*cos)
__global__ void inverseAccumulateKernel(
    const double* __restrict__ zeta_real,
    const double* __restrict__ cos_th, const double* __restrict__ sin_th,
    const double* __restrict__ mcos_th, const double* __restrict__ msin_th,
    int ns, int mpol, int ntheta, int nzeta, int nZnT, int slot0,
    double* __restrict__ e0, double* __restrict__ e1, double* __restrict__ e2,
    double* __restrict__ o0, double* __restrict__ o1, double* __restrict__ o2)
{
    int j = blockIdx.x;
    // slot0: 0 = R slots 0-3, 4 = Z slots 4-7, 8 = λ slots 8-11
    // Thread mapping: l1 = threadIdx.x (fastest), k = threadIdx.y — the
    // output stores at idx = j*nZnT + k*ntheta + l then vary l fastest and
    // coalesce; the m-loop shared reads (sm[.. + k]) become broadcasts.
    int k = threadIdx.y, l1 = threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    extern __shared__ double sh[];   // [4][mpol][nzeta]
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 4 * mpol * nzeta; i += nthreads) {
        int s = i / (mpol * nzeta), rem = i - s * mpol * nzeta;
        int m = rem / nzeta, kk = rem % nzeta;
        sh[i] = zeta_real[(((size_t)(slot0 + s) * mpol + m) * ns + j) * nzeta + kk];
    }
    __syncthreads();
    double maxsc = fmax(sqrt((double)j / (ns - 1.0)), sqrt(1.0 / (ns - 1.0)));
    double facO = 1.0 / maxsc;
    bool isR = (slot0 == 0);
    double signV = (slot0 == 8) ? -1.0 : 1.0;
    size_t mstride = (size_t)mpol * nzeta;
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        int l = l1 + pass * (ntheta / 2);
        double v0e = 0, v1e = 0, v2e = 0;
        double v0o = 0, v1o = 0, v2o = 0;
        for (int m = 0; m < mpol; ++m) {
            const double* sm = sh + m * nzeta;
            double c0 = sm[0 * mstride + k], c1 = sm[1 * mstride + k];
            double c2 = sm[2 * mstride + k], c3 = sm[3 * mstride + k];
            double cosm = cos_th[m * ntheta + l], sinm = sin_th[m * ntheta + l];
            double mcos = mcos_th[m * ntheta + l], msin = msin_th[m * ntheta + l];
            double fac = (m % 2 == 1) ? facO : 1.0;
            double t0 = isR ? cosm : sinm, t1 = isR ? sinm : cosm;
            double u0 = isR ? msin : mcos, u1 = isR ? mcos : msin;
            double v0 = fac * (c0 * t0 + c1 * t1);
            double v1 = fac * (c0 * u0 + c1 * u1);
            double v2 = signV * fac * (c2 * t0 + c3 * t1);
            if (m % 2 == 1) { v0o += v0; v1o += v1; v2o += v2; }
            else            { v0e += v0; v1e += v1; v2e += v2; }
        }
        int idx = j * nZnT + k * ntheta + l;
        e0[idx] = v0e; e1[idx] = v1e; e2[idx] = v2e;
        o0[idx] = v0o; o1[idx] = v1o; o2[idx] = v2o;
    }
}

// The 9 combined (e+o) real-space arrays (used by the dump machinery and the
// legacy test-only forwardDFTDirect).
__global__ void combineParityKernel(
    const double* __restrict__ r_e,  const double* __restrict__ z_e,
    const double* __restrict__ l_e,  const double* __restrict__ ru_e,
    const double* __restrict__ zu_e, const double* __restrict__ lu_e,
    const double* __restrict__ rv_e, const double* __restrict__ zv_e,
    const double* __restrict__ lv_e, const double* __restrict__ r_o,
    const double* __restrict__ z_o,  const double* __restrict__ l_o,
    const double* __restrict__ ru_o, const double* __restrict__ zu_o,
    const double* __restrict__ lu_o, const double* __restrict__ rv_o,
    const double* __restrict__ zv_o, const double* __restrict__ lv_o,
    int nZnT, int ns,
    double* __restrict__ r_real,  double* __restrict__ z_real,
    double* __restrict__ l_real,  double* __restrict__ ru_real,
    double* __restrict__ zu_real, double* __restrict__ lu_real,
    double* __restrict__ rv_real, double* __restrict__ zv_real,
    double* __restrict__ lv_real)
{
    int j = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nZnT) return;
    int idx = k + j * nZnT;
    r_real[idx] = r_e[idx] + r_o[idx];   z_real[idx] = z_e[idx] + z_o[idx];
    l_real[idx] = l_e[idx] + l_o[idx];   ru_real[idx] = ru_e[idx] + ru_o[idx];
    zu_real[idx] = zu_e[idx] + zu_o[idx]; lu_real[idx] = lu_e[idx] + lu_o[idx];
    rv_real[idx] = rv_e[idx] + rv_o[idx]; zv_real[idx] = zv_e[idx] + zv_o[idx];
    lv_real[idx] = lv_e[idx] + lv_o[idx];
}

static void inverseDFTCufft(const FourierPlan& fp, const SpectralState& st,
                            const GridParams& p) {
    // Zero the half-spectra (only bins n <= ntor are filled).
    checkCuda(cudaMemset(fp.d_zeta_spectra, 0,
        (size_t)12 * p.mpol * p.ns * (p.nzeta / 2 + 1) * sizeof(double2)), "inv zero");
    int total = p.ns * p.mnmax;
    inversePackKernel<<<(total + 255) / 256, 256>>>(
        st.d_rmncc, st.d_rmnss, st.d_zmnsc, st.d_zmncs, st.d_lmnsc, st.d_lmncs,
        fp.basis.d_xm, fp.basis.d_xn,
        p.ns, p.mpol, p.ntor, p.nfp, p.nzeta / 2 + 1, fp.d_zeta_spectra);
    checkCufft(cufftExecZ2D(fp.plan_z2d, fp.d_zeta_spectra, fp.d_zeta_real), "inv z2d");
    dim3 blk(p.ntheta / 2, p.nzeta);
    dim3 grd(p.ns);
    size_t invSmem = 4 * p.mpol * p.nzeta * sizeof(double);
    // R slots 0-3 -> r/ru/rv, Z slots 4-7 -> z/zu/zv, λ slots 8-11 -> l/lu/lv
    inverseAccumulateKernel<<<grd, blk, invSmem>>>(
        fp.d_zeta_real,
        fp.d_cos_th, fp.d_sin_th, fp.d_mcos_th, fp.d_msin_th,
        p.ns, p.mpol, p.ntheta, p.nzeta, p.nZnT, 0,
        fp.d_r_e, fp.d_ru_e, fp.d_rv_e, fp.d_r_o, fp.d_ru_o, fp.d_rv_o);
    inverseAccumulateKernel<<<grd, blk, invSmem>>>(
        fp.d_zeta_real,
        fp.d_cos_th, fp.d_sin_th, fp.d_mcos_th, fp.d_msin_th,
        p.ns, p.mpol, p.ntheta, p.nzeta, p.nZnT, 4,
        fp.d_z_e, fp.d_zu_e, fp.d_zv_e, fp.d_z_o, fp.d_zu_o, fp.d_zv_o);
    inverseAccumulateKernel<<<grd, blk, invSmem>>>(
        fp.d_zeta_real,
        fp.d_cos_th, fp.d_sin_th, fp.d_mcos_th, fp.d_msin_th,
        p.ns, p.mpol, p.ntheta, p.nzeta, p.nZnT, 8,
        fp.d_l_e, fp.d_lu_e, fp.d_lv_e, fp.d_l_o, fp.d_lu_o, fp.d_lv_o);
    dim3 cblk(32), cgrd((p.nZnT + 31) / 32, p.ns);
    combineParityKernel<<<cgrd, cblk>>>(
        fp.d_r_e, fp.d_z_e, fp.d_l_e, fp.d_ru_e, fp.d_zu_e, fp.d_lu_e,
        fp.d_rv_e, fp.d_zv_e, fp.d_lv_e,
        fp.d_r_o, fp.d_z_o, fp.d_l_o, fp.d_ru_o, fp.d_zu_o, fp.d_lu_o,
        fp.d_rv_o, fp.d_zv_o, fp.d_lv_o,
        p.nZnT, p.ns,
        fp.d_r_real, fp.d_z_real, fp.d_l_real,
        fp.d_ru_real, fp.d_zu_real, fp.d_lu_real,
        fp.d_rv_real, fp.d_zv_real, fp.d_lv_real);
    checkCuda(cudaGetLastError(), "inv cuFFT");
    checkCuda(cudaDeviceSynchronize(), "inv cuFFT sync");
}

void inverseDFT(const FourierPlan& fp, const SpectralState& st, const GridParams& p) {
    inverseDFTCufft(fp, st, p);
}

// ---- cuFFT backend: forward ----------------------------------------------
// Mirror of vmecpp's dft_ForcesToFourier_3d_symm (fft_toroidal.cc): a fused
// poloidal reduction over the REDUCED θ grid [0, π] builds 12 real ζ-signals
// per (m, j) (the m-parity selects the _e/_o force arrays, xmpq = m(m-1)
// folds the spectral-condensation constraint into tempR/tempZ), a batched
// real FFT gives their half-spectra, and the recovery recombines the bins.
// Slot definitions (signs verified against the reference kernels):
//   rmkcc  += tempR*cos + brmn*(-m*sin)     rmkss += tempR*sin + brmn*(+m*cos)
//   zmksc  += tempZ*sin + bzmn*(+m*cos)     zmkcs += tempZ*cos + bzmn*(-m*sin)
//   lmksc  += blmn*(+m*cos)                 lmkcs += blmn*(-m*sin)
//   rmkccN -= crmn*cos;  rmkssN -= crmn*sin
//   zmkscN -= czmn*sin;  zmkcsN -= czmn*cos
//   lmkscN -= clmn*sin;  lmkcsN -= clmn*cos
// with the θ tables carrying the trapezoid weight (d_fwd_w, intNorm with
// endpoint ½ — the θ > π points are not on vmecpp's reduced grid).
__global__ void forwardReduceKernel(
    const double* __restrict__ armn_e, const double* __restrict__ armn_o,
    const double* __restrict__ azmn_e, const double* __restrict__ azmn_o,
    const double* __restrict__ brmn_e, const double* __restrict__ brmn_o,
    const double* __restrict__ bzmn_e, const double* __restrict__ bzmn_o,
    const double* __restrict__ crmn_e, const double* __restrict__ crmn_o,
    const double* __restrict__ czmn_e, const double* __restrict__ czmn_o,
    const double* __restrict__ blmn_e, const double* __restrict__ blmn_o,
    const double* __restrict__ clmn_e, const double* __restrict__ clmn_o,
    const double* __restrict__ frcon_e, const double* __restrict__ frcon_o,
    const double* __restrict__ fzcon_e, const double* __restrict__ fzcon_o,
    const double* __restrict__ cos_th, const double* __restrict__ sin_th,
    const double* __restrict__ mcos_th, const double* __restrict__ msin_th,
    const double* __restrict__ fwd_w,
    int ns, int mpol, int ntheta, int nThetaRed, int nzeta, int nZnT,
    double* __restrict__ zeta_real)
{
    int j = blockIdx.y, m = blockIdx.x;
    // Thread mapping: l = threadIdx.x (fastest), k = threadIdx.y — the 14
    // force-array loads at idx = j*nZnT + k*ntheta + l then vary l fastest
    // and coalesce. blockDim.x is padded to 16 lanes; threads with
    // l >= nThetaRed contribute zero. The per-(slot, k) sum over the 16
    // l-values is a warp shuffle tree (two k groups per warp, width 16) —
    // replacing the previous shared-memory atomicAdd, which after the
    // dimension swap serialized 16-way per address (2.5 ms/iter).
    int k = threadIdx.y, l = threadIdx.x;
    bool active = (l < nThetaRed);
    double v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;
    double v6 = 0, v7 = 0, v8 = 0, v9 = 0, v10 = 0, v11 = 0;
    if (active) {
        bool m_even = (m % 2 == 0);
        const double* armn = m_even ? armn_e : armn_o;
        const double* azmn = m_even ? azmn_e : azmn_o;
        const double* brmn = m_even ? brmn_e : brmn_o;
        const double* bzmn = m_even ? bzmn_e : bzmn_o;
        const double* crmn = m_even ? crmn_e : crmn_o;
        const double* czmn = m_even ? czmn_e : czmn_o;
        const double* blmn = m_even ? blmn_e : blmn_o;
        const double* clmn = m_even ? clmn_e : clmn_o;
        const double* frcon = m_even ? frcon_e : frcon_o;
        const double* fzcon = m_even ? fzcon_e : fzcon_o;
        double xmpq = (double)m * (m - 1);
        int idx = j * nZnT + k * ntheta + l;
        double w = fwd_w[l];
        double cosm = w * cos_th[m * ntheta + l], sinm = w * sin_th[m * ntheta + l];
        double mcos = w * mcos_th[m * ntheta + l], msin = w * msin_th[m * ntheta + l];
        double tempR = armn[idx] + xmpq * frcon[idx];
        double tempZ = azmn[idx] + xmpq * fzcon[idx];
        double br = brmn[idx], bz = bzmn[idx];
        double cr = crmn[idx], cz = czmn[idx];
        double bl = blmn[idx], cl = clmn[idx];
        v0 = tempR * cosm + br * msin;
        v1 = tempR * sinm + br * mcos;
        v2 = -cr * cosm;
        v3 = -cr * sinm;
        v4 = tempZ * sinm + bz * mcos;
        v5 = tempZ * cosm + bz * msin;
        v6 = -cz * sinm;
        v7 = -cz * cosm;
        v8 = bl * mcos;
        v9 = bl * msin;
        v10 = -cl * sinm;
        v11 = -cl * cosm;
    }
    // Warp reduction over the 16 l-values (two k groups per warp, width 16).
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        v0 += __shfl_down_sync(mask, v0, off, 16);
        v1 += __shfl_down_sync(mask, v1, off, 16);
        v2 += __shfl_down_sync(mask, v2, off, 16);
        v3 += __shfl_down_sync(mask, v3, off, 16);
        v4 += __shfl_down_sync(mask, v4, off, 16);
        v5 += __shfl_down_sync(mask, v5, off, 16);
        v6 += __shfl_down_sync(mask, v6, off, 16);
        v7 += __shfl_down_sync(mask, v7, off, 16);
        v8 += __shfl_down_sync(mask, v8, off, 16);
        v9 += __shfl_down_sync(mask, v9, off, 16);
        v10 += __shfl_down_sync(mask, v10, off, 16);
        v11 += __shfl_down_sync(mask, v11, off, 16);
    }
    if (l == 0) {
        double* base = zeta_real + ((size_t)m * ns + j) * nzeta;
        size_t step = (size_t)mpol * ns * nzeta;
        #pragma unroll
        for (int s = 0; s < 12; ++s) {
            double v = (s == 0) ? v0 : (s == 1) ? v1 : (s == 2) ? v2 : (s == 3) ? v3
                     : (s == 4) ? v4 : (s == 5) ? v5 : (s == 6) ? v6 : (s == 7) ? v7
                     : (s == 8) ? v8 : (s == 9) ? v9 : (s == 10) ? v10 : v11;
            base[s * step + k] = v;
        }
    }
}

// Coefficient recovery from the 12 half-spectra (bin n of each slot).
// With nf = n*nfp and mn = mscale*nscale:
//   frcc = mn*(Re F_rmkcc + nf*Im F_rmkccN)   frss = mn*(-Im F_rmkss + nf*Re F_rmkssN)
//   fzsc = mn*(Re F_zmksc + nf*Im F_zmkscN)   fzcs = mn*(-Im F_zmkcs + nf*Re F_zmkcsN)
//   flsc = mn*(Re F_lmksc + nf*Im F_lmkscN)   flcs = mn*(-Im F_lmkcs + nf*Re F_lmkcsN)
// Surface coverage (vmecpp dft_ForcesToFourier_3d_symm): axis j=0 keeps only
// the m=0 frcc/fzcs (incl. the crmn/czmn toroidal terms); the LCFS j=ns-1
// keeps only the λ components; f_spec is pre-zeroed by the caller.
__global__ void forwardRecoverKernel(
    const double2* __restrict__ spectra,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mpol, int mnmax, int nfp, int nz2,
    double* __restrict__ f_spec)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ns * mnmax) return;
    int j = t % ns, mode = t / ns;
    int m = xm[mode], n = xn[mode];
    double nf = n * nfp;
    double ms = (m == 0) ? 1.0 : sqrt(2.0);
    double nsq = (n == 0) ? 1.0 : sqrt(2.0);
    double mn = ms * nsq;
    size_t step = (size_t)mpol * ns * nz2;
    const double2* slot = spectra + ((size_t)m * ns + j) * nz2 + n;
    double2 F0 = slot[0 * step],  F1 = slot[1 * step],  F2 = slot[2 * step],  F3 = slot[3 * step];
    double2 F4 = slot[4 * step],  F5 = slot[5 * step],  F6 = slot[6 * step],  F7 = slot[7 * step];
    double2 F8 = slot[8 * step],  F9 = slot[9 * step],  F10 = slot[10 * step], F11 = slot[11 * step];
    double* out = f_spec + j + mode * ns;
    if (j == 0) {
        if (m > 0) return;   // axis: m=0 only
        out[0 * mnmax * ns] = mn * (F0.x + nf * F2.y);
        out[4 * mnmax * ns] = mn * (-F5.y + nf * F7.x);
        return;
    }
    if (j == ns - 1) {       // LCFS: λ only
        out[2 * mnmax * ns] = mn * (F8.x + nf * F10.y);
        out[5 * mnmax * ns] = mn * (-F9.y + nf * F11.x);
        return;
    }
    out[0 * mnmax * ns] = mn * (F0.x + nf * F2.y);
    out[1 * mnmax * ns] = mn * (F4.x + nf * F6.y);
    out[2 * mnmax * ns] = mn * (F8.x + nf * F10.y);
    out[3 * mnmax * ns] = mn * (-F1.y + nf * F3.x);
    out[4 * mnmax * ns] = mn * (-F5.y + nf * F7.x);
    out[5 * mnmax * ns] = mn * (-F9.y + nf * F11.x);
}

static void forwardDFTCufft(const FourierPlan& fp, double* d_f_spectral,
                            const GridParams& p, const ConstraintWorkspace& cw) {
    // vmecpp zeroes the target before projecting (m_physical_f.setZero());
    // the kernels write only the entries vmecpp computes.
    checkCuda(cudaMemset(d_f_spectral, 0,
                         (size_t)6 * p.mnmax * p.ns * sizeof(double)), "fwd zero");
    dim3 blk(16, p.nzeta);  // x padded to 16 lanes (warp shuffle width)
    dim3 grd(p.mpol, p.ns);
    forwardReduceKernel<<<grd, blk>>>(
        fp.d_armn_e, fp.d_armn_o, fp.d_azmn_e, fp.d_azmn_o,
        fp.d_brmn_e, fp.d_brmn_o, fp.d_bzmn_e, fp.d_bzmn_o,
        fp.d_crmn_e, fp.d_crmn_o, fp.d_czmn_e, fp.d_czmn_o,
        fp.d_blmn_e, fp.d_blmn_o, fp.d_clmn_e, fp.d_clmn_o,
        cw.d_frcon_e, cw.d_frcon_o, cw.d_fzcon_e, cw.d_fzcon_o,
        fp.d_cos_th, fp.d_sin_th, fp.d_mcos_th, fp.d_msin_th, fp.d_fwd_w,
        p.ns, p.mpol, p.ntheta, p.ntheta / 2 + 1, p.nzeta, p.nZnT,
        fp.d_zeta_real);
    checkCufft(cufftExecD2Z(fp.plan_d2z, fp.d_zeta_real, fp.d_zeta_spectra), "fwd d2z");
    int total = p.ns * p.mnmax;
    forwardRecoverKernel<<<(total + 255) / 256, 256>>>(
        fp.d_zeta_spectra, fp.basis.d_xm, fp.basis.d_xn,
        p.ns, p.mpol, p.mnmax, p.nfp, p.nzeta / 2 + 1, d_f_spectral);
    checkCuda(cudaGetLastError(), "fwd cuFFT");
    checkCuda(cudaDeviceSynchronize(), "fwd cuFFT sync");
}

void forwardDFT(const FourierPlan& fp, double* d_f_spectral, const GridParams& p,
                const ConstraintWorkspace& cw) {
    forwardDFTCufft(fp, d_f_spectral, p, cw);
}
