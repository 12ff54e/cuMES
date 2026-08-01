// fourier.cu — DFT transforms with vmecpp m-parity convention.
//
// Parity convention (matches vmecpp):
//   Even m -> "e" arrays, odd m -> "o" arrays.
//   Each parity array receives the FULL contribution from its modes:
//     R: rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ)
//     Z: zmnsc*sin(mθ)cos(nζ) + zmncs*(-cos(mθ)sin(nζ))
//     λ: lmnsc*sin(mθ)cos(nζ)
//
// Spectral state (5 components per mode per surface):
//   rmncc: R cos(mθ)cos(nζ)  (even parity in the combined basis)
//   rmnss: R sin(mθ)sin(nζ)  (odd parity in the combined basis)
//   zmnsc: Z sin(mθ)cos(nζ)  (even parity in the combined basis)
//   zmncs: Z -cos(mθ)sin(nζ) (odd parity in the combined basis, note minus sign)
//   lmnsc: λ sin(mθ)cos(nζ)  (stellarator-symmetric: even parity only)

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

FourierPlan fourierCreate(const GridParams& p, cublasHandle_t handle) {
    FourierPlan fp{}; fp.handle = handle;
    int nZnT = p.nZnT, mnmax = p.mnmax;
    size_t nbytes_basis = nZnT * mnmax * sizeof(double);
    size_t nbytes_mode  = mnmax * sizeof(int);
    size_t nbytes_real  = p.ns * nZnT * sizeof(double);

    auto* h_cos = new double[nZnT*mnmax], *h_sin = new double[nZnT*mnmax];
    auto* h_cc  = new double[nZnT*mnmax], *h_ss  = new double[nZnT*mnmax];
    auto* h_sc  = new double[nZnT*mnmax], *h_cs  = new double[nZnT*mnmax];
    auto* h_xm  = new int[mnmax], *h_xn = new int[mnmax];

    for (int m = 0; m < p.mpol; ++m)
        for (int n = 0; n < p.ntor; ++n) {
            int mode = m * p.ntor + n;
            h_xm[mode] = m; h_xn[mode] = n * p.nfp;
        }

    for (int it = 0; it < p.ntheta; ++it) {
        double theta = 2.0 * M_PI * it / p.ntheta;
        for (int iz = 0; iz < p.nzeta; ++iz) {
            double zeta = 2.0 * M_PI * iz / p.nzeta;
            int idx = iz * p.ntheta + it;
            for (int mode = 0; mode < mnmax; ++mode) {
                int m = h_xm[mode], n = h_xn[mode];
                double cm = cos(m*theta), sm = sin(m*theta);
                double cn = cos(n*zeta),  sn = sin(n*zeta);
                h_cos[idx+mode*nZnT] = cm*cn + sm*sn;
                h_sin[idx+mode*nZnT] = sm*cn - cm*sn;
                h_cc[idx+mode*nZnT]  = cm*cn;
                h_ss[idx+mode*nZnT]  = sm*sn;
                h_sc[idx+mode*nZnT]  = sm*cn;
                h_cs[idx+mode*nZnT]  = cm*sn;
            }
        }
    }

    auto alloc = [&](double*& p, const char* n) { checkCuda(cudaMalloc(&p, nbytes_basis), n); };
    alloc(fp.basis.d_cos_mt_nz, "cos"); alloc(fp.basis.d_sin_mt_nz, "sin");
    alloc(fp.basis.d_cc, "cc"); alloc(fp.basis.d_ss, "ss");
    alloc(fp.basis.d_sc, "sc"); alloc(fp.basis.d_cs, "cs");
    checkCuda(cudaMalloc(&fp.basis.d_xm, nbytes_mode), "xm");
    checkCuda(cudaMalloc(&fp.basis.d_xn, nbytes_mode), "xn");

    auto cpy = [&](double* d, double* h, const char* n) { checkCuda(cudaMemcpy(d, h, nbytes_basis, cudaMemcpyHostToDevice), n); };
    cpy(fp.basis.d_cos_mt_nz, h_cos, "cos"); cpy(fp.basis.d_sin_mt_nz, h_sin, "sin");
    cpy(fp.basis.d_cc, h_cc, "cc"); cpy(fp.basis.d_ss, h_ss, "ss");
    cpy(fp.basis.d_sc, h_sc, "sc"); cpy(fp.basis.d_cs, h_cs, "cs");
    checkCuda(cudaMemcpy(fp.basis.d_xm, h_xm, nbytes_mode, cudaMemcpyHostToDevice), "xm");
    checkCuda(cudaMemcpy(fp.basis.d_xn, h_xn, nbytes_mode, cudaMemcpyHostToDevice), "xn");
    delete[] h_cos; delete[] h_sin; delete[] h_cc; delete[] h_ss; delete[] h_sc; delete[] h_cs;
    delete[] h_xm; delete[] h_xn;

    auto am = [&](double*& p, const char* n) { checkCuda(cudaMalloc(&p, nbytes_real), n); };
    am(fp.d_r_e,"r_e"); am(fp.d_z_e,"z_e"); am(fp.d_l_e,"l_e");
    am(fp.d_ru_e,"ru_e"); am(fp.d_zu_e,"zu_e"); am(fp.d_lu_e,"lu_e");
    am(fp.d_r_o,"r_o"); am(fp.d_z_o,"z_o"); am(fp.d_l_o,"l_o");
    am(fp.d_ru_o,"ru_o"); am(fp.d_zu_o,"zu_o"); am(fp.d_lu_o,"lu_o");
    am(fp.d_r_real,"r_r"); am(fp.d_z_real,"z_r"); am(fp.d_l_real,"l_r");
    am(fp.d_ru_real,"ru_r"); am(fp.d_zu_real,"zu_r"); am(fp.d_lu_real,"lu_r");
    am(fp.d_rv_real,"rv_r"); am(fp.d_zv_real,"zv_r"); am(fp.d_lv_real,"lv_r");
    am(fp.d_armn_e,"ae"); am(fp.d_armn_o,"ao");
    am(fp.d_azmn_e,"aze"); am(fp.d_azmn_o,"azo");
    am(fp.d_brmn_e,"be"); am(fp.d_brmn_o,"bo");
    am(fp.d_bzmn_e,"bze"); am(fp.d_bzmn_o,"bzo");
    am(fp.d_blmn_e,"ble"); am(fp.d_blmn_o,"blo");
    am(fp.d_fr_real,"fr"); am(fp.d_fz_real,"fz"); am(fp.d_fl_real,"fl");
    return fp;
}

void fourierFree(FourierPlan& fp) {
    auto cuFree = [](double* p) { cudaFree(p); };
    cuFree(fp.basis.d_cos_mt_nz); cuFree(fp.basis.d_sin_mt_nz);
    cuFree(fp.basis.d_cc); cuFree(fp.basis.d_ss); cuFree(fp.basis.d_sc); cuFree(fp.basis.d_cs);
    cudaFree(fp.basis.d_xm); cudaFree(fp.basis.d_xn);
    cuFree(fp.d_r_e);cuFree(fp.d_z_e);cuFree(fp.d_l_e);
    cuFree(fp.d_ru_e);cuFree(fp.d_zu_e);cuFree(fp.d_lu_e);
    cuFree(fp.d_r_o);cuFree(fp.d_z_o);cuFree(fp.d_l_o);
    cuFree(fp.d_ru_o);cuFree(fp.d_zu_o);cuFree(fp.d_lu_o);
    cuFree(fp.d_r_real);cuFree(fp.d_z_real);cuFree(fp.d_l_real);
    cuFree(fp.d_ru_real);cuFree(fp.d_zu_real);cuFree(fp.d_lu_real);
    cuFree(fp.d_rv_real);cuFree(fp.d_zv_real);cuFree(fp.d_lv_real);
    cuFree(fp.d_armn_e);cuFree(fp.d_armn_o);cuFree(fp.d_azmn_e);cuFree(fp.d_azmn_o);
    cuFree(fp.d_brmn_e);cuFree(fp.d_brmn_o);cuFree(fp.d_bzmn_e);cuFree(fp.d_bzmn_o);
    cuFree(fp.d_blmn_e);cuFree(fp.d_blmn_o);
    cuFree(fp.d_fr_real);cuFree(fp.d_fz_real);cuFree(fp.d_fl_real);
}

// ---- inverse DFT (parity coefficients → real space) -------------------

__global__ void inverseDFTKernel(
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const double* __restrict__ lmnsc,
    const double* __restrict__ cc, const double* __restrict__ ss,
    const double* __restrict__ sc, const double* __restrict__ cs,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax, int nZnT,
    double* __restrict__ r_e,  double* __restrict__ z_e,  double* __restrict__ l_e,
    double* __restrict__ ru_e, double* __restrict__ zu_e, double* __restrict__ lu_e,
    double* __restrict__ r_o,  double* __restrict__ z_o,  double* __restrict__ l_o,
    double* __restrict__ ru_o, double* __restrict__ zu_o, double* __restrict__ lu_o,
    double* __restrict__ r_real, double* __restrict__ z_real, double* __restrict__ l_real,
    double* __restrict__ ru_real, double* __restrict__ zu_real, double* __restrict__ lu_real,
    double* __restrict__ rv_real, double* __restrict__ zv_real, double* __restrict__ lv_real)
{
    int j = blockIdx.y, k = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= ns || k >= nZnT) return;

    double re=0, ze=0, le=0, rue=0, zue=0, lue=0;
    double ro=0, zo=0, lo=0, ruo=0, zuo=0, luo=0;
    double rve=0, rvo=0, zve=0, zvo=0, lve=0, lvo=0;

    for (int m = 0; m < mnmax; ++m) {
        double cc_v = cc[k+m*nZnT], ss_v = ss[k+m*nZnT];
        double sc_v = sc[k+m*nZnT], cs_v = cs[k+m*nZnT];
        int mi = xm[m], ni = xn[m];

        double rc = rmncc[j+m*ns], rs_v = rmnss[j+m*ns];
        double zs_v = zmnsc[j+m*ns], zc = zmncs[j+m*ns];
        double lc = lmnsc[j+m*ns];

        // vmecpp parity convention: split by m parity, NOT by trigonometric factor.
        // Even m -> e arrays, odd m -> o arrays.
        // Each parity array receives the FULL contribution from the mode
        // (both cos-cos and sin-sin for R, both sin-cos and cos-sin for Z).
        bool m_even = (mi % 2 == 0);

        // R: rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ)
        if (m_even) {
            re  += rc * cc_v + rs_v * ss_v;
            rue += -mi * rc * sc_v + mi * rs_v * cs_v;
            rve += -ni * rc * cs_v + ni * rs_v * sc_v;
        } else {
            ro  += rc * cc_v + rs_v * ss_v;
            ruo += -mi * rc * sc_v + mi * rs_v * cs_v;
            rvo += -ni * rc * cs_v + ni * rs_v * sc_v;
        }

        // Z: zmnsc*sin(mθ)cos(nζ) + zmncs*(-cos(mθ)sin(nζ))
        if (m_even) {
            ze  += zs_v * sc_v - zc * cs_v;
            zue += mi * zs_v * cc_v + mi * zc * ss_v;
            zve += -ni * zs_v * ss_v - ni * zc * cc_v;
        } else {
            zo  += zs_v * sc_v - zc * cs_v;
            zuo += mi * zs_v * cc_v + mi * zc * ss_v;
            zvo += -ni * zs_v * ss_v - ni * zc * cc_v;
        }

        // λ: lmnsc*sin(mθ)cos(nζ)  (stellarator-symmetric, split by m parity)
        if (m_even) {
            le  += lc * sc_v;
            lue += mi * lc * cc_v;
            lve += -ni * lc * ss_v;
        } else {
            lo  += lc * sc_v;
            luo += mi * lc * cc_v;
            lvo += -ni * lc * ss_v;
        }
    }

    // vmecpp's real-space odd-parity arrays carry the odd-m decomposition
    // (scalxc = 1/max(sqrt(s_F), sqrt(1/(ns-1)))): its kernels (jacobian,
    // metric, force, bcontra) are written for decomposed odd inputs, and its
    // real-space odd values come out as physical/max(...). cuMES stores the
    // plain physical coefficients, so the odd real-space must be divided by
    // max(...) here to feed the (vmecpp-formula) kernels the inputs they
    // expect. The lambda basis additionally carries the mscale = sqrt(2)
    // factor (lambda coefficients are NOT mscale-normalized in either code,
    // unlike R/Z), so the lambda real-space is multiplied by sqrt(2).
    // Verified: at iter 1 (initial state) these factors cancel against the
    // state's s^(m/2) profile and the outputs are unchanged; they only
    // matter once the odd-m profiles develop s-dependence.
    double maxsc = fmax(sqrt((double)j / (ns - 1.0)), sqrt(1.0 / (ns - 1.0)));
    ro /= maxsc; ruo /= maxsc; rvo /= maxsc;
    zo /= maxsc; zuo /= maxsc; zvo /= maxsc;
    le *= M_SQRT2; lue *= M_SQRT2; lve *= M_SQRT2;
    lo *= M_SQRT2 / maxsc; luo *= M_SQRT2 / maxsc; lvo *= M_SQRT2 / maxsc;

    int idx = k + j * nZnT;
    r_e[idx]=re; z_e[idx]=ze; l_e[idx]=le; ru_e[idx]=rue; zu_e[idx]=zue; lu_e[idx]=lue;
    r_o[idx]=ro; z_o[idx]=zo; l_o[idx]=lo; ru_o[idx]=ruo; zu_o[idx]=zuo; lu_o[idx]=luo;
    r_real[idx]=re+ro; z_real[idx]=ze+zo; l_real[idx]=le+lo;
    ru_real[idx]=rue+ruo; zu_real[idx]=zue+zuo; lu_real[idx]=lue+luo;
    rv_real[idx]=rve+rvo; zv_real[idx]=zve+zvo; lv_real[idx]=lve+lvo;
}

void inverseDFT(const FourierPlan& fp, const SpectralState& st, const GridParams& p) {
    dim3 block(32); dim3 grid((p.nZnT+31)/32, p.ns);
    inverseDFTKernel<<<grid, block>>>(
        st.d_rmncc, st.d_rmnss, st.d_zmnsc, st.d_zmncs, st.d_lmnsc,
        fp.basis.d_cc, fp.basis.d_ss, fp.basis.d_sc, fp.basis.d_cs,
        fp.basis.d_xm, fp.basis.d_xn,
        p.ns, p.mnmax, p.nZnT,
        fp.d_r_e, fp.d_z_e, fp.d_l_e, fp.d_ru_e, fp.d_zu_e, fp.d_lu_e,
        fp.d_r_o, fp.d_z_o, fp.d_l_o, fp.d_ru_o, fp.d_zu_o, fp.d_lu_o,
        fp.d_r_real, fp.d_z_real, fp.d_l_real,
        fp.d_ru_real, fp.d_zu_real, fp.d_lu_real,
        fp.d_rv_real, fp.d_zv_real, fp.d_lv_real);
    checkCuda(cudaGetLastError(), "invDFT");
    checkCuda(cudaDeviceSynchronize(), "invDFT sync");
}

// ---- forward DFT (parity forces → 5-component spectral forces) -------
// Matches vmecpp's ForcesToFourier integration-by-parts convention:
//   frcc = armn*cos(mθ)cos(nζ) - brmn*sin(mθ)cos(nζ)  (+ 3D terms)
//   fzsc = azmn*sin(mθ)cos(nζ) + bzmn*cos(mθ)cos(nζ)  (+ 3D terms)
//   flsc = blmn*cos(mθ)cos(nζ)                         (+ 3D terms)
// Poloidal forces (brmn/bzmn) are NOT simply added to radial forces
// before projection — they use different trigonometric weights due to
// integration by parts in the energy gradient.

__global__ void forwardDFTParityKernel(
    // Radial MHD forces
    const double* __restrict__ armn_e, const double* __restrict__ armn_o,
    const double* __restrict__ azmn_e, const double* __restrict__ azmn_o,
    // Poloidal MHD forces (projected with different weights)
    const double* __restrict__ brmn_e, const double* __restrict__ brmn_o,
    const double* __restrict__ bzmn_e, const double* __restrict__ bzmn_o,
    // Lambda force
    const double* __restrict__ blmn_e, const double* __restrict__ blmn_o,
    // Spectral-condensation constraint force (vmecpp frcon/fzcon): enters
    // frcc/fzsc as xmpq[m] * frcon with cosmui/sinmui weights.
    const double* __restrict__ frcon_e, const double* __restrict__ frcon_o,
    const double* __restrict__ fzcon_e, const double* __restrict__ fzcon_o,
    // Basis functions
    const double* __restrict__ cc, const double* __restrict__ ss,
    const double* __restrict__ sc, const double* __restrict__ cs,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax, int nZnT,
    double* __restrict__ f_spec)
{
    // The kernel writes only the entries vmecpp's dft_ForcesToFourier_2d_symm
    // computes; the caller must zero the full f_spec first (like vmecpp's
    // m_physical_f.setZero()).
    int mode = blockIdx.x, j = blockIdx.y;
    if (mode >= mnmax || j >= ns) return;
    int mm = xm[mode], nn = xn[mode];
    bool m_even = (mm % 2 == 0);

    // vmecpp normalization: mscale (√2 for m>0) × integration weight.
    // vmecpp uses trapezoidal integration over the reduced [0,π] grid with
    // intNorm = 1/(nZeta*(nThetaReduced-1)). For the full [0,2π) grid with
    // even-function symmetry, the uniform weight w = mscale/nZnT is equivalent.
    double mscale = (mm == 0) ? 1.0 : sqrt(2.0);
    double w = mscale / nZnT;
    if (mm > 0 && nn > 0) w *= 2.0;  // 3D: no symmetry, need 2× weight

    // The integration-by-parts (IBP) terms in vmecpp use cosmumi = m * cosmui
    // and sinmumi = -m * sinmui. So brmn/bzmn contributions get an extra
    // factor of m compared to the raw sin/cos projection. The m factor
    // applies to ALL modes: for m=0 the fzsc and flsc projections are
    // identically zero (vmecpp: cosmumi = 0, sinmui = 0).
    double m_ibp = (double)mm;
    // Spectral-condensation constraint weight: vmecpp's xmpq[m] = m*(m-1)
    // (zero for m=0,1 — the constraint couples m>=2 only).
    double xmpq_m = (double)mm * (mm - 1);

    // vmecpp's surface coverage in dft_ForcesToFourier_2d_symm:
    //   - R/Z: jF = 0..ns-2 (jMaxRZ = ns-1 for fixed boundary) — the LCFS
    //     R/Z forces are never computed and stay zero.
    //   - axis jF=0: only m=0 contributions (num_m = 1).
    //   - lambda: jF = 1..ns-1 — the axis lambda stays zero, the LCFS
    //     lambda force IS computed.
    if (j == 0) {
        if (mm > 0) return;  // axis m>0: nothing computed
        // axis m=0: only frcc is nonzero (fzsc/frss/fzcs vanish through the
        // sin(0) basis; lambda excluded at the axis).
        double sr_e = 0;
        for (int k = 0; k < nZnT; ++k) {
            int idx = k + j * nZnT;
            sr_e += armn_e[idx] * cc[k + mode * nZnT];  // brmn term: m_ibp=0
        }
        f_spec[j + mode*ns + 0*mnmax*ns] = sr_e * w;
        return;
    }
    if (j == ns - 1) {
        // LCFS: only the lambda force is computed.
        double sl_e = 0;
        for (int k = 0; k < nZnT; ++k) {
            int idx = k + j * nZnT;
            sl_e += (m_even ? blmn_e[idx] : blmn_o[idx]) * m_ibp *
                    cc[k + mode * nZnT];
        }
        f_spec[j + mode*ns + 2*mnmax*ns] = sl_e * w;
        return;
    }

    // interior surfaces: full spectrum
    double sr_e=0, sr_o=0, sz_e=0, sz_o=0, sl_e=0;
    for (int k = 0; k < nZnT; ++k) {
        double cc_v=cc[k+mode*nZnT], ss_v=ss[k+mode*nZnT];
        double sc_v=sc[k+mode*nZnT], cs_v=cs[k+mode*nZnT];
        int idx = k + j * nZnT;
        if (m_even) {
            // Radial R: (armn + xmpq·frcon) → cos(mθ)cos(nζ) for frcc
            // (vmecpp dft_ForcesToFourier_2d_symm: _rcc = rnkcc + xmpq*rcon_cc)
            // Poloidal R: brmn → -m·sin(mθ)cos(nζ) (IBP of ∂_θ, vmecpp sinmumi)
            sr_e += (armn_e[idx] + xmpq_m * frcon_e[idx]) * cc_v
                  + brmn_e[idx] * (-m_ibp * sc_v);
            // Radial R: armn → sin(mθ)sin(nζ) for frss
            // Poloidal R: brmn → m·cos(mθ)sin(nζ) (IBP of ∂_ζ)
            sr_o += armn_e[idx] * ss_v + brmn_e[idx] * m_ibp * cs_v;
            // Radial Z: (azmn + xmpq·fzcon) → sin(mθ)cos(nζ) for fzsc
            // Poloidal Z: bzmn → m·cos(mθ)cos(nζ) (IBP of ∂_θ, vmecpp cosmumi)
            sz_e += (azmn_e[idx] + xmpq_m * fzcon_e[idx]) * sc_v
                  + bzmn_e[idx] * m_ibp * cc_v;
            // Radial Z: azmn → -cos(mθ)sin(nζ) for fzcs
            // Poloidal Z: bzmn → m·sin(mθ)sin(nζ) (IBP of ∂_ζ)
            sz_o += azmn_e[idx] * (-cs_v) + bzmn_e[idx] * m_ibp * ss_v;
            // Lambda: blmn → m·cos(mθ)cos(nζ) for flsc
            // (matches vmecpp dft_ForcesToFourier_2d_symm: lnksc_m * cosmumi)
            // cosmumi = m * cosmui, so the m factor applies to ALL modes.
            sl_e += blmn_e[idx] * m_ibp * cc_v;
        } else {
            sr_e += (armn_o[idx] + xmpq_m * frcon_o[idx]) * cc_v
                  + brmn_o[idx] * (-m_ibp * sc_v);
            sr_o += armn_o[idx] * ss_v + brmn_o[idx] * m_ibp * cs_v;
            sz_e += (azmn_o[idx] + xmpq_m * fzcon_o[idx]) * sc_v
                  + bzmn_o[idx] * m_ibp * cc_v;
            sz_o += azmn_o[idx] * (-cs_v) + bzmn_o[idx] * m_ibp * ss_v;
            // Lambda (odd m): blmn_o → m·cos(mθ)cos(nζ) for flsc
            // (matches vmecpp dft_ForcesToFourier_2d_symm: cosmumi = m * cosmui)
            sl_e += blmn_o[idx] * m_ibp * cc_v;
        }
    }
    f_spec[j + mode*ns + 0*mnmax*ns] = sr_e * w;
    f_spec[j + mode*ns + 1*mnmax*ns] = sz_e * w;
    f_spec[j + mode*ns + 2*mnmax*ns] = sl_e * w;
    f_spec[j + mode*ns + 3*mnmax*ns] = sr_o * w;
    f_spec[j + mode*ns + 4*mnmax*ns] = sz_o * w;
}

void forwardDFT(const FourierPlan& fp, double* d_f_spectral, const GridParams& p,
                const ConstraintWorkspace& cw) {
    // vmecpp zeroes the target before projecting (m_physical_f.setZero());
    // the kernel writes only the entries vmecpp computes.
    checkCuda(cudaMemset(d_f_spectral, 0,
                         (size_t)5 * p.mnmax * p.ns * sizeof(double)), "fwd zero");
    {   dim3 b(1), g(p.mnmax, p.ns);
        forwardDFTParityKernel<<<g, b>>>(
            fp.d_armn_e, fp.d_armn_o, fp.d_azmn_e, fp.d_azmn_o,
            fp.d_brmn_e, fp.d_brmn_o, fp.d_bzmn_e, fp.d_bzmn_o,
            fp.d_blmn_e, fp.d_blmn_o,
            cw.d_frcon_e, cw.d_frcon_o, cw.d_fzcon_e, cw.d_fzcon_o,
            fp.basis.d_cc, fp.basis.d_ss, fp.basis.d_sc, fp.basis.d_cs,
            fp.basis.d_xm, fp.basis.d_xn,
            p.ns, p.mnmax, p.nZnT, d_f_spectral);
        checkCuda(cudaGetLastError(), "fwdDFT"); }
    checkCuda(cudaDeviceSynchronize(), "fwdDFT sync");
}

// Backward-compatible forward DFT for tests
__global__ void forwardDFTKernel(
    const double* __restrict__ fr, const double* __restrict__ fz,
    const double* __restrict__ fl,
    const double* __restrict__ cos_mt_nz, const double* __restrict__ sin_mt_nz,
    int ns, int mnmax, int nZnT, double* __restrict__ f_spec)
{
    int m=blockIdx.x, j=blockIdx.y;
    if(m>=mnmax||j>=ns) return;
    double w=(m==0)?(1.0/nZnT):(2.0/nZnT), sr=0,sz=0,sl=0;
    for(int k=0;k<nZnT;++k){
        double c=cos_mt_nz[k+m*nZnT], s=sin_mt_nz[k+m*nZnT];
        sr+=fr[k+j*nZnT]*c; sz+=fz[k+j*nZnT]*s; sl+=fl[k+j*nZnT]*c;
    }
    f_spec[j+m*ns+0*mnmax*ns]=sr*w; f_spec[j+m*ns+1*mnmax*ns]=sz*w;
    f_spec[j+m*ns+2*mnmax*ns]=sl*w;
}

void forwardDFTDirect(const FourierPlan& fp, double* d_f_spec, const GridParams& p) {
    dim3 b(1), g(p.mnmax, p.ns);
    forwardDFTKernel<<<g, b>>>(fp.d_fr_real, fp.d_fz_real, fp.d_fl_real,
        fp.basis.d_cos_mt_nz, fp.basis.d_sin_mt_nz, p.ns, p.mnmax, p.nZnT, d_f_spec);
    checkCuda(cudaGetLastError(), "fwdDFTDirect");
    checkCuda(cudaDeviceSynchronize(), "fwdDFTDirect sync");
}
