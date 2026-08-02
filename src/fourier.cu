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

    // Basis tables on the full [0,2π)x[0,2π) grid. zeta covers the full
    // torus; the nfp enters through xn (n*nfp), same as vmecpp.
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
    cuFree(fp.d_rv_e);cuFree(fp.d_rv_o);
    cuFree(fp.d_zv_e);cuFree(fp.d_zv_o);
    cuFree(fp.d_lv_e);cuFree(fp.d_lv_o);
    cuFree(fp.d_armn_e);cuFree(fp.d_armn_o);cuFree(fp.d_azmn_e);cuFree(fp.d_azmn_o);
    cuFree(fp.d_brmn_e);cuFree(fp.d_brmn_o);cuFree(fp.d_bzmn_e);cuFree(fp.d_bzmn_o);
    cuFree(fp.d_blmn_e);cuFree(fp.d_blmn_o);
    cuFree(fp.d_crmn_e);cuFree(fp.d_crmn_o);cuFree(fp.d_czmn_e);cuFree(fp.d_czmn_o);
    cuFree(fp.d_clmn_e);cuFree(fp.d_clmn_o);
    cuFree(fp.d_fr_real);cuFree(fp.d_fz_real);cuFree(fp.d_fl_real);
}

// ---- inverse DFT (parity coefficients → real space) -------------------
// Basis (plain, per mode with cc=cos(mθ)cos(nζ), ss=sin(mθ)sin(nζ),
// sc=sin(mθ)cos(nζ), cs=cos(mθ)sin(nζ)):
//   R    = rc*cc + rs*ss              R_θ = -m*rc*sc + m*rs*cs
//   Z    = zs*sc + zc*cs              Z_θ =  m*zs*cc - m*zc*ss
//   λ    = lsc*sc + lcs*cs            λ_θ =  m*lsc*cc - m*lcs*ss
//   R_ζ  = -n*rc*cs + n*rs*sc         Z_ζ = -n*zs*ss + n*zc*cc
//   λ_ζ stored as -∂λ/∂ζ (vmecpp):    lv  =  n*lsc*ss - n*lcs*cc
// (n here is xn = n*nfp.)
//
// vmecpp's real-space odd-parity arrays carry the odd-m decomposition
// (scalxc = 1/max(sqrt(s_F), sqrt(1/(ns-1)))): its kernels (jacobian,
// metric, force, bcontra) are written for decomposed odd inputs, and its
// real-space odd values come out as physical/max(...). cuMES stores the
// plain physical coefficients, so the odd real-space must be divided by
// max(...) here to feed the (vmecpp-formula) kernels the inputs they
// expect. The lambda basis additionally carries the mscale*nscale factor
// (lambda coefficients are NOT mscale-normalized in either code, unlike
// R/Z), so each lambda mode's real-space contribution is multiplied by
// mscale[m]*nscale[n] (sqrt2 each for m>0 / n>0).
__global__ void inverseDFTKernel(
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const double* __restrict__ lmnsc, const double* __restrict__ lmncs,
    const double* __restrict__ cc, const double* __restrict__ ss,
    const double* __restrict__ sc, const double* __restrict__ cs,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax, int nZnT, int nfp,
    double* __restrict__ r_e,  double* __restrict__ z_e,  double* __restrict__ l_e,
    double* __restrict__ ru_e, double* __restrict__ zu_e, double* __restrict__ lu_e,
    double* __restrict__ r_o,  double* __restrict__ z_o,  double* __restrict__ l_o,
    double* __restrict__ ru_o, double* __restrict__ zu_o, double* __restrict__ lu_o,
    double* __restrict__ rv_e, double* __restrict__ zv_e, double* __restrict__ lv_e,
    double* __restrict__ rv_o, double* __restrict__ zv_o, double* __restrict__ lv_o,
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
        int mi = xm[m], ni = xn[m] * nfp;  // physical toroidal derivative factor

        double rc = rmncc[j+m*ns], rs_v = rmnss[j+m*ns];
        double zs_v = zmnsc[j+m*ns], zc = zmncs[j+m*ns];
        double lsc = lmnsc[j+m*ns], lcs = lmncs[j+m*ns];

        // vmecpp parity convention: split by m parity, NOT by trigonometric
        // factor. Even m -> e arrays, odd m -> o arrays. Each parity array
        // receives the FULL contribution from the mode.
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

        // Z: zmnsc*sin(mθ)cos(nζ) + zmncs*cos(mθ)sin(nζ)
        if (m_even) {
            ze  += zs_v * sc_v + zc * cs_v;
            zue += mi * zs_v * cc_v - mi * zc * ss_v;
            zve += -ni * zs_v * ss_v + ni * zc * cc_v;
        } else {
            zo  += zs_v * sc_v + zc * cs_v;
            zuo += mi * zs_v * cc_v - mi * zc * ss_v;
            zvo += -ni * zs_v * ss_v + ni * zc * cc_v;
        }

        // λ: lmnsc*sin(mθ)cos(nζ) + lmncs*cos(mθ)sin(nζ); lu = λ_θ,
        // lv = -λ_ζ (vmecpp convention). NOTE: the λ STATE already carries
        // the mscale*nscale factors (state = vmecpp-decomposed * ms*ns,
        // verified against the vmecpp state dumps at 1e-13), so the
        // reconstruction uses the RAW basis -- an additional lfac here
        // would double-count (FIXED 2026-08-02: the old lfac made the
        // iter-2+ lambda real-space (ms*ns)x too big, e.g. 2x for m=2n=1).
        if (m_even) {
            le  += lsc * sc_v + lcs * cs_v;
            lue += mi * lsc * cc_v - mi * lcs * ss_v;
            lve += ni * lsc * ss_v - ni * lcs * cc_v;
        } else {
            lo  += lsc * sc_v + lcs * cs_v;
            luo += mi * lsc * cc_v - mi * lcs * ss_v;
            lvo += ni * lsc * ss_v - ni * lcs * cc_v;
        }
    }

    // Odd-m decomposition (scalxc) for the odd real-space arrays, matching
    // vmecpp's decomposed real space (see header comment).
    double maxsc = fmax(sqrt((double)j / (ns - 1.0)), sqrt(1.0 / (ns - 1.0)));
    ro /= maxsc; ruo /= maxsc; rvo /= maxsc;
    zo /= maxsc; zuo /= maxsc; zvo /= maxsc;
    lo /= maxsc; luo /= maxsc; lvo /= maxsc;

    int idx = k + j * nZnT;
    r_e[idx]=re; z_e[idx]=ze; l_e[idx]=le; ru_e[idx]=rue; zu_e[idx]=zue; lu_e[idx]=lue;
    r_o[idx]=ro; z_o[idx]=zo; l_o[idx]=lo; ru_o[idx]=ruo; zu_o[idx]=zuo; lu_o[idx]=luo;
    rv_e[idx]=rve; zv_e[idx]=zve; lv_e[idx]=lve;
    rv_o[idx]=rvo; zv_o[idx]=zvo; lv_o[idx]=lvo;
    r_real[idx]=re+ro; z_real[idx]=ze+zo; l_real[idx]=le+lo;
    ru_real[idx]=rue+ruo; zu_real[idx]=zue+zuo; lu_real[idx]=lue+luo;
    rv_real[idx]=rve+rvo; zv_real[idx]=zve+zvo; lv_real[idx]=lve+lvo;
}

void inverseDFT(const FourierPlan& fp, const SpectralState& st, const GridParams& p) {
    dim3 block(32); dim3 grid((p.nZnT+31)/32, p.ns);
    inverseDFTKernel<<<grid, block>>>(
        st.d_rmncc, st.d_rmnss, st.d_zmnsc, st.d_zmncs, st.d_lmnsc, st.d_lmncs,
        fp.basis.d_cc, fp.basis.d_ss, fp.basis.d_sc, fp.basis.d_cs,
        fp.basis.d_xm, fp.basis.d_xn,
        p.ns, p.mnmax, p.nZnT, p.nfp,
        fp.d_r_e, fp.d_z_e, fp.d_l_e, fp.d_ru_e, fp.d_zu_e, fp.d_lu_e,
        fp.d_r_o, fp.d_z_o, fp.d_l_o, fp.d_ru_o, fp.d_zu_o, fp.d_lu_o,
        fp.d_rv_e, fp.d_zv_e, fp.d_lv_e, fp.d_rv_o, fp.d_zv_o, fp.d_lv_o,
        fp.d_r_real, fp.d_z_real, fp.d_l_real,
        fp.d_ru_real, fp.d_zu_real, fp.d_lu_real,
        fp.d_rv_real, fp.d_zv_real, fp.d_lv_real);
    checkCuda(cudaGetLastError(), "invDFT");
    checkCuda(cudaDeviceSynchronize(), "invDFT sync");
}

// ---- forward DFT (parity forces → 6-component spectral forces) -------
// Matches vmecpp's dft_ForcesToFourier_3d_symm (dft_toroidal.cc), in the
// full-grid weight convention. For mode (m,n): w = mscale*nscale/nZnT with
// mscale = sqrt2 (m>0) and nscale = sqrt2 (n>0) — the full-grid equivalent
// of vmecpp's reduced-grid trapezoid with intNorm (the mscale/nscale live
// in vmecpp's basis tables cosmui/cosnv etc.).
//
// The poloidal forces (brmn/bzmn) project with the m-weighted basis
// (cosmumi = m*cosmui, sinmumi = -m*sinmui, integration by parts in θ) and
// the toroidal forces (crmn/czmn/clmn) with the n*nfp-weighted derivative
// basis (sinnvn = -n*nfp*sin(nζ), cosnvn = +n*nfp*cos(nζ)):
//   frcc +=  tempR*cc           + brmn*(-m*sc)  + crmn*(+n*cs)
//   frss +=  tempR*ss           + brmn*(+m*cs)  + crmn*(-n*sc)
//   fzsc +=  tempZ*sc           + bzmn*(+m*cc)  + czmn*(+n*ss)
//   fzcs +=  tempZ*cs           + bzmn*(+m*ss)  + czmn*(-n*cc)
//   flsc +=  blmn*(+m*cc)                                   + clmn*(+n*ss)
//   flcs +=  blmn*(-m*ss)                                   + clmn*(-n*cc)
// where tempR = armn + xmpq[m]*frcon, tempZ = azmn + xmpq[m]*fzcon.
// (Signs verified against dft_toroidal.cc; the _n projections carry the
// n*nfp factor from the ζ-derivative basis.)

__global__ void forwardDFTParityKernel(
    // Radial MHD forces
    const double* __restrict__ armn_e, const double* __restrict__ armn_o,
    const double* __restrict__ azmn_e, const double* __restrict__ azmn_o,
    // Poloidal MHD forces (projected with different weights)
    const double* __restrict__ brmn_e, const double* __restrict__ brmn_o,
    const double* __restrict__ bzmn_e, const double* __restrict__ bzmn_o,
    // Toroidal MHD forces (3D)
    const double* __restrict__ crmn_e, const double* __restrict__ crmn_o,
    const double* __restrict__ czmn_e, const double* __restrict__ czmn_o,
    // Lambda forces (poloidal blmn, toroidal clmn)
    const double* __restrict__ blmn_e, const double* __restrict__ blmn_o,
    const double* __restrict__ clmn_e, const double* __restrict__ clmn_o,
    // Spectral-condensation constraint force (vmecpp frcon/fzcon): enters
    // frcc/fzsc as xmpq[m] * frcon with cosmui/sinmui weights.
    const double* __restrict__ frcon_e, const double* __restrict__ frcon_o,
    const double* __restrict__ fzcon_e, const double* __restrict__ fzcon_o,
    // Basis functions
    const double* __restrict__ cc, const double* __restrict__ ss,
    const double* __restrict__ sc, const double* __restrict__ cs,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax, int nZnT, int ntheta, int nfp,
    double* __restrict__ f_spec)
{
    // The kernel writes only the entries vmecpp's dft_ForcesToFourier_3d_symm
    // computes; the caller must zero the full f_spec first (like vmecpp's
    // m_physical_f.setZero()).
    int mode = blockIdx.x, j = blockIdx.y;
    if (mode >= mnmax || j >= ns) return;
    int mm = xm[mode], nn = xn[mode] * nfp;  // toroidal IBP factor (n*nfp)
    bool m_even = (mm % 2 == 0);

    // vmecpp's quadrature: the forward DFT integrates over the REDUCED theta
    // grid [0, pi] (nThetaRed = ntheta/2+1 points, the first nThetaRed of the
    // full grid) with the trapezoid intNorm = 1/(nZeta*(nThetaRed-1)),
    // endpoint-halved, times mscale*nscale (fourier_basis.cc: cosmui/cosnv).
    // A uniform full-grid weight w = mscale*nscale/nZnT is NOT equivalent for
    // the odd-about-theta=pi integrand parts (m=1.. modes of the R/Z forces).
    // (FIXED 2026-08-02: the old uniform weight made the poloidal-IBP and
    // toroidal terms deviate from vmecpp, e.g. flsc by ~0.16 and frcc by ~9.)
    const int nThetaRed = ntheta / 2 + 1;
    const int nZeta = nZnT / ntheta;
    const double intNorm = 1.0 / (double)(nZeta * (nThetaRed - 1));
    double mscale = (mm == 0) ? 1.0 : sqrt(2.0);
    double nscale = (nn == 0) ? 1.0 : sqrt(2.0);
    const double w_base = mscale * nscale * intNorm;
    // Per-point weight; points with theta index >= nThetaRed are NOT part of
    // vmecpp's reduced grid and contribute nothing.
    auto w_k = [&](int k) -> double {
        int it = k % ntheta;
        if (it >= nThetaRed) return 0.0;
        double wk = w_base;
        if (it == 0 || it == nThetaRed - 1) wk *= 0.5;
        return wk;
    };

    // The integration-by-parts (IBP) terms: cosmumi = m*cosmui and
    // sinmumi = -m*sinmui, so brmn/bzmn/blmn get an extra factor of m;
    // the toroidal forces get the ζ-derivative factor n*nfp.
    double m_ibp = (double)mm;
    double n_ibp = (double)nn;
    // Spectral-condensation constraint weight: vmecpp's xmpq[m] = m*(m-1)
    // (zero for m=0,1 — the constraint couples m>=2 only).
    double xmpq_m = (double)mm * (mm - 1);

    // vmecpp's surface coverage in dft_ForcesToFourier_3d_symm:
    //   - R/Z: jF = 0..ns-2 (jMaxRZ = ns-1 for fixed boundary) — the LCFS
    //     R/Z forces are never computed and stay zero.
    //   - axis jF=0: only m=0 contributions (num_m = 1).
    //   - lambda: jF = 1..ns-1 — the axis lambda stays zero, the LCFS
    //     lambda force IS computed.
    if (j == 0) {
        if (mm > 0) return;  // axis m>0: nothing computed
        // Axis m=0: frcc gets the direct armn term AND the toroidal crmn term
        // (rmkcc_n*sinnvn, the m=0 cosmui=1 part), and fzcs is nonzero via
        // zmkcs*sinnv + zmkcs_n*cosnvn (m=0: cosmui=1, sinmumi=0).
        // (FIXED 2026-08-02: the old axis branch computed only the armn term
        // of frcc; it missed the crmn contribution and the whole fzcs.)
        double sr_e = 0, sz_o = 0;
        for (int k = 0; k < nZnT; ++k) {
            double wk = w_k(k);
            if (wk == 0.0) continue;
            int idx = k + j * nZnT;
            double cc_v = cc[k + mode * nZnT], cs_v = cs[k + mode * nZnT];
            sr_e += (armn_e[idx] * cc_v + crmn_e[idx] * (n_ibp * cs_v)) * wk;
            sz_o += (azmn_e[idx] * cs_v + czmn_e[idx] * (-n_ibp * cc_v)) * wk;
        }
        f_spec[j + mode*ns + 0*mnmax*ns] = sr_e;
        f_spec[j + mode*ns + 4*mnmax*ns] = sz_o;
        return;
    }
    if (j == ns - 1) {
        // LCFS: only the lambda forces are computed.
        double sl_e = 0, slcs = 0;
        for (int k = 0; k < nZnT; ++k) {
            double wk = w_k(k);
            if (wk == 0.0) continue;
            int idx = k + j * nZnT;
            double cc_v = cc[k+mode*nZnT], ss_v = ss[k+mode*nZnT];
            double sc_v = sc[k+mode*nZnT], cs_v = cs[k+mode*nZnT];
            double blmn = m_even ? blmn_e[idx] : blmn_o[idx];
            double clmn = m_even ? clmn_e[idx] : clmn_o[idx];
            sl_e  += (blmn * (m_ibp * cc_v) + clmn * (n_ibp * ss_v)) * wk;
            slcs  += (blmn * (-m_ibp * ss_v) + clmn * (-n_ibp * cc_v)) * wk;
        }
        f_spec[j + mode*ns + 2*mnmax*ns] = sl_e;
        f_spec[j + mode*ns + 5*mnmax*ns] = slcs;
        return;
    }

    // interior surfaces: full spectrum
    double sr_e=0, sr_o=0, sz_e=0, sz_o=0, sl_e=0, slcs=0;
    for (int k = 0; k < nZnT; ++k) {
        double wk = w_k(k);
        if (wk == 0.0) continue;
        double cc_v=cc[k+mode*nZnT], ss_v=ss[k+mode*nZnT];
        double sc_v=sc[k+mode*nZnT], cs_v=cs[k+mode*nZnT];
        int idx = k + j * nZnT;
        if (m_even) {
            double tempR = armn_e[idx] + xmpq_m * frcon_e[idx];
            double tempZ = azmn_e[idx] + xmpq_m * fzcon_e[idx];
            sr_e += (tempR * cc_v
                  + brmn_e[idx] * (-m_ibp * sc_v)
                  + crmn_e[idx] * (n_ibp * cs_v)) * wk;
            sr_o += (tempR * ss_v
                  + brmn_e[idx] * (m_ibp * cs_v)
                  + crmn_e[idx] * (-n_ibp * sc_v)) * wk;
            sz_e += (tempZ * sc_v
                  + bzmn_e[idx] * (m_ibp * cc_v)
                  + czmn_e[idx] * (n_ibp * ss_v)) * wk;
            sz_o += (tempZ * cs_v
                  + bzmn_e[idx] * (-m_ibp * ss_v)
                  + czmn_e[idx] * (-n_ibp * cc_v)) * wk;
            sl_e += (blmn_e[idx] * (m_ibp * cc_v)
                  + clmn_e[idx] * (n_ibp * ss_v)) * wk;
            slcs += (blmn_e[idx] * (-m_ibp * ss_v)
                  + clmn_e[idx] * (-n_ibp * cc_v)) * wk;
        } else {
            double tempR = armn_o[idx] + xmpq_m * frcon_o[idx];
            double tempZ = azmn_o[idx] + xmpq_m * fzcon_o[idx];
            sr_e += (tempR * cc_v
                  + brmn_o[idx] * (-m_ibp * sc_v)
                  + crmn_o[idx] * (n_ibp * cs_v)) * wk;
            sr_o += (tempR * ss_v
                  + brmn_o[idx] * (m_ibp * cs_v)
                  + crmn_o[idx] * (-n_ibp * sc_v)) * wk;
            sz_e += (tempZ * sc_v
                  + bzmn_o[idx] * (m_ibp * cc_v)
                  + czmn_o[idx] * (n_ibp * ss_v)) * wk;
            sz_o += (tempZ * cs_v
                  + bzmn_o[idx] * (-m_ibp * ss_v)
                  + czmn_o[idx] * (-n_ibp * cc_v)) * wk;
            sl_e += (blmn_o[idx] * (m_ibp * cc_v)
                  + clmn_o[idx] * (n_ibp * ss_v)) * wk;
            slcs += (blmn_o[idx] * (-m_ibp * ss_v)
                  + clmn_o[idx] * (-n_ibp * cc_v)) * wk;
        }
    }
    f_spec[j + mode*ns + 0*mnmax*ns] = sr_e;
    f_spec[j + mode*ns + 1*mnmax*ns] = sz_e;
    f_spec[j + mode*ns + 2*mnmax*ns] = sl_e;
    f_spec[j + mode*ns + 3*mnmax*ns] = sr_o;
    f_spec[j + mode*ns + 4*mnmax*ns] = sz_o;
    f_spec[j + mode*ns + 5*mnmax*ns] = slcs;
}

void forwardDFT(const FourierPlan& fp, double* d_f_spectral, const GridParams& p,
                const ConstraintWorkspace& cw) {
    // vmecpp zeroes the target before projecting (m_physical_f.setZero());
    // the kernel writes only the entries vmecpp computes.
    checkCuda(cudaMemset(d_f_spectral, 0,
                         (size_t)6 * p.mnmax * p.ns * sizeof(double)), "fwd zero");
    {   dim3 b(1), g(p.mnmax, p.ns);
        forwardDFTParityKernel<<<g, b>>>(
            fp.d_armn_e, fp.d_armn_o, fp.d_azmn_e, fp.d_azmn_o,
            fp.d_brmn_e, fp.d_brmn_o, fp.d_bzmn_e, fp.d_bzmn_o,
            fp.d_crmn_e, fp.d_crmn_o, fp.d_czmn_e, fp.d_czmn_o,
            fp.d_blmn_e, fp.d_blmn_o, fp.d_clmn_e, fp.d_clmn_o,
            cw.d_frcon_e, cw.d_frcon_o, cw.d_fzcon_e, cw.d_fzcon_o,
            fp.basis.d_cc, fp.basis.d_ss, fp.basis.d_sc, fp.basis.d_cs,
            fp.basis.d_xm, fp.basis.d_xn,
            p.ns, p.mnmax, p.nZnT, p.ntheta, p.nfp, d_f_spectral);
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
