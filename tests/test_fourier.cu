// test_fourier.cu — algorithm correctness tests for parity-split DFT transforms.
// Build: cd cuMES/build && cmake .. && make test_fourier && ./test_fourier
// Conventions (matching the solver): folded mode table mode = m*(ntor+1)+n,
// n = 0..ntor; inverse DFT with the plus-zmncs convention, lmncs coefficient,
// lv = -∂λ/∂ζ, per-mode mscale*nscale factor on lambda; forward DFT with
// w = mscale*nscale/nZnT and 6 spectral force components.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cublas_v2.h>
#include <vector>
#include "vmec_types.h"
#include "fourier.cuh"
#include "constraint.cuh"

constexpr int kNs=3, kMpol=3, kNtor=2, kNtheta=16, kNzeta=8, kNfp=1;
constexpr int kMnmax=kMpol*(kNtor+1), kNZnT=kNtheta*kNzeta;

static int g_failures=0;
static void checkNear(double g,double e,double t,const char* s,int j,int k){
    if(fabs(g-e)>t){fprintf(stderr,"FAIL [%s] j=%d k=%d got=%.15e exp=%.15e\n",s,j,k,g,e);++g_failures;}}
static void checkMode(double g,double e,double t,const char* s,int j,int m){
    if(fabs(g-e)>t){fprintf(stderr,"FAIL [%s] j=%d m=%d got=%.15e exp=%.15e\n",s,j,m,g,e);++g_failures;}}
static void cc(cudaError_t e,const char* t){if(e!=cudaSuccess){fprintf(stderr,"CUDA[%s]:%s\n",t,cudaGetErrorString(e));exit(1);}}

// Host copy of the basis tables (the test cannot read device pointers).
static void hostBasis(const GridParams& p,
    std::vector<double>& hcc, std::vector<double>& hss,
    std::vector<double>& hsc, std::vector<double>& hcs,
    std::vector<int>& hxm, std::vector<int>& hxn){
    hcc.assign(p.nZnT*p.mnmax,0); hss.assign(p.nZnT*p.mnmax,0);
    hsc.assign(p.nZnT*p.mnmax,0); hcs.assign(p.nZnT*p.mnmax,0);
    hxm.resize(p.mnmax); hxn.resize(p.mnmax);
    for(int m=0;m<p.mpol;++m) for(int n=0;n<p.ntor+1;++n){
        int mode=m*(p.ntor+1)+n; hxm[mode]=m; hxn[mode]=n*p.nfp;
    }
    for(int it=0;it<p.ntheta;++it){
        double th=2*M_PI*it/p.ntheta;
        for(int iz=0;iz<p.nzeta;++iz){
            double ze=2*M_PI*iz/p.nzeta;
            int idx=iz*p.ntheta+it;
            for(int mode=0;mode<p.mnmax;++mode){
                int m=hxm[mode], n=hxn[mode];
                double cm=cos(m*th), sm=sin(m*th), cn=cos(n*ze), sn=sin(n*ze);
                hcc[idx+mode*p.nZnT]=cm*cn; hss[idx+mode*p.nZnT]=sm*sn;
                hsc[idx+mode*p.nZnT]=sm*cn; hcs[idx+mode*p.nZnT]=cm*sn;
            }
        }
    }
}

// CPU: parity inverse DFT matching inverseDFTKernel.
static void cpuInvDFT(const double* cc_, const double* ss_, const double* zsc_, const double* zcs_,
    const double* lsc_, const double* lcs_,
    const double* pcc, const double* pss, const double* psc, const double* pcs,
    const int* xm, const int* xn, int ns, int mnmax, int nZnT,
    double* r, double* z, double* l, double* ru, double* zu, double* lu,
    double* rv, double* zv, double* lv){
    for(int j=0;j<ns;++j) for(int k=0;k<nZnT;++k){
        double re=0,ro=0,ze=0,zo=0,le=0,lo=0,rue=0,ruo=0,zue=0,zuo=0,lue=0,luo=0;
        double rve=0,rvo=0,zve=0,zvo=0,lve=0,lvo=0;
        for(int m=0;m<mnmax;++m){
            double cc_v=pcc[k+m*nZnT],ss_v=pss[k+m*nZnT],sc_v=psc[k+m*nZnT],cs_v=pcs[k+m*nZnT];
            int mi=xm[m], ni=xn[m];
            double rc=cc_[j+m*ns], rs=ss_[j+m*ns];
            double zs=zsc_[j+m*ns], zc=zcs_[j+m*ns];
            double ls=lsc_[j+m*ns], lcs=lcs_[j+m*ns];
            // No mscale*nscale on the lambda reconstruction: the lambda STATE
            // already carries ms*ns (state = vmecpp-decomposed * ms*ns), so
            // the real-space uses the raw basis (FIXED 2026-08-02 to match
            // the vmecpp-verified convention; the old lfac double-counted).
            if(mi%2==0){
                re  += rc*cc_v + rs*ss_v;         rue += -mi*rc*sc_v + mi*rs*cs_v;
                rve += -ni*rc*cs_v + ni*rs*sc_v;
                ze  += zs*sc_v + zc*cs_v;         zue += mi*zs*cc_v - mi*zc*ss_v;
                zve += -ni*zs*ss_v + ni*zc*cc_v;
                le  += ls*sc_v + lcs*cs_v;        lue += mi*ls*cc_v - mi*lcs*ss_v;
                lve += ni*ls*ss_v - ni*lcs*cc_v;
            } else {
                ro  += rc*cc_v + rs*ss_v;         ruo += -mi*rc*sc_v + mi*rs*cs_v;
                rvo += -ni*rc*cs_v + ni*rs*sc_v;
                zo  += zs*sc_v + zc*cs_v;         zuo += mi*zs*cc_v - mi*zc*ss_v;
                zvo += -ni*zs*ss_v + ni*zc*cc_v;
                lo  += ls*sc_v + lcs*cs_v;        luo += mi*ls*cc_v - mi*lcs*ss_v;
                lvo += ni*ls*ss_v - ni*lcs*cc_v;
            }
        }
        double maxsc=fmax(sqrt((double)j/(ns-1.0)),sqrt(1.0/(ns-1.0)));
        ro/=maxsc; ruo/=maxsc; rvo/=maxsc; zo/=maxsc; zuo/=maxsc; zvo/=maxsc;
        lo/=maxsc; luo/=maxsc; lvo/=maxsc;
        int idx=k+j*nZnT;
        r[idx]=re+ro; z[idx]=ze+zo; l[idx]=le+lo;
        ru[idx]=rue+ruo; zu[idx]=zue+zuo; lu[idx]=lue+luo;
        rv[idx]=rve+rvo; zv[idx]=zve+zvo; lv[idx]=lve+lvo;
    }
}

static void gpuInv(SpectralState& st, FourierPlan& fp, const GridParams& p,
    const double* cc_, const double* ss_, const double* zsc_, const double* zcs_,
    const double* lsc_, const double* lcs_){
    size_t nb=p.ns*p.mnmax*sizeof(double);
    cc(cudaMemcpy(st.d_rmncc,cc_,nb,cudaMemcpyHostToDevice),"up cc");
    cc(cudaMemcpy(st.d_rmnss,ss_,nb,cudaMemcpyHostToDevice),"up ss");
    cc(cudaMemcpy(st.d_zmnsc,zsc_,nb,cudaMemcpyHostToDevice),"up zsc");
    cc(cudaMemcpy(st.d_zmncs,zcs_,nb,cudaMemcpyHostToDevice),"up zcs");
    cc(cudaMemcpy(st.d_lmnsc,lsc_,nb,cudaMemcpyHostToDevice),"up lsc");
    cc(cudaMemcpy(st.d_lmncs,lcs_,nb,cudaMemcpyHostToDevice),"up lcs");
    inverseDFT(fp,st,p);
}

static int t_inv_constR(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_inverseDFT_constantR ... ");
    std::vector<double> cc_(p.ns*p.mnmax,0),ss_(p.ns*p.mnmax,0),zs(p.ns*p.mnmax,0),zc(p.ns*p.mnmax,0),ls_(p.ns*p.mnmax,0),lcs(p.ns*p.mnmax,0);
    for(int j=0;j<p.ns;++j) cc_[j+0*p.ns]=4.0;  // R_00
    double* h_r=new double[p.ns*p.nZnT], *h_rv=new double[p.ns*p.nZnT];
    std::vector<double> r(p.ns*p.nZnT),z(p.ns*p.nZnT),l(p.ns*p.nZnT);
    std::vector<double> ru(p.ns*p.nZnT),zu(p.ns*p.nZnT),lu(p.ns*p.nZnT);
    std::vector<double> rv(p.ns*p.nZnT),zv(p.ns*p.nZnT),lv(p.ns*p.nZnT);
    gpuInv(st,fp,p,cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),lcs.data());
    cc(cudaMemcpy(h_r,fp.d_r_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get r");
    cc(cudaMemcpy(h_rv,fp.d_rv_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get rv");
    std::vector<double> hcc,hss,hsc,hcs; std::vector<int> hxm,hxn;
    hostBasis(p,hcc,hss,hsc,hcs,hxm,hxn);
    cpuInvDFT(cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),lcs.data(),
        hcc.data(),hss.data(),hsc.data(),hcs.data(),
        hxm.data(),hxn.data(),p.ns,p.mnmax,p.nZnT,
        r.data(),z.data(),l.data(),ru.data(),zu.data(),lu.data(),rv.data(),zv.data(),lv.data());
    for(int i=0;i<p.ns*p.nZnT;++i){
        checkNear(h_r[i],r[i],1e-12,"R",i/p.nZnT,i%p.nZnT);
        checkNear(h_rv[i],rv[i],1e-12,"Rv",i/p.nZnT,i%p.nZnT);
    }
    delete[] h_r; delete[] h_rv;
    printf(g_failures==lf?"PASS\n":"FAIL\n");
    return g_failures-lf;
}

static int t_inv_theta(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_inverseDFT_thetaDerivative ... ");
    std::vector<double> cc_(p.ns*p.mnmax,0),ss_(p.ns*p.mnmax,0),zs(p.ns*p.mnmax,0),zc(p.ns*p.mnmax,0),ls_(p.ns*p.mnmax,0),lcs(p.ns*p.mnmax,0);
    int m1=1*(p.ntor+1)+0;
    for(int j=0;j<p.ns;++j) cc_[j+m1*p.ns]=0.3;  // R_10
    double* h_r=new double[p.ns*p.nZnT], *h_ru=new double[p.ns*p.nZnT];
    std::vector<double> r(p.ns*p.nZnT),z(p.ns*p.nZnT),l(p.ns*p.nZnT);
    std::vector<double> ru(p.ns*p.nZnT),zu(p.ns*p.nZnT),lu(p.ns*p.nZnT);
    std::vector<double> rv(p.ns*p.nZnT),zv(p.ns*p.nZnT),lv(p.ns*p.nZnT);
    gpuInv(st,fp,p,cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),lcs.data());
    cc(cudaMemcpy(h_r,fp.d_r_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get r");
    cc(cudaMemcpy(h_ru,fp.d_ru_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get ru");
    std::vector<double> hcc,hss,hsc,hcs; std::vector<int> hxm,hxn;
    hostBasis(p,hcc,hss,hsc,hcs,hxm,hxn);
    cpuInvDFT(cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),lcs.data(),
        hcc.data(),hss.data(),hsc.data(),hcs.data(),
        hxm.data(),hxn.data(),p.ns,p.mnmax,p.nZnT,
        r.data(),z.data(),l.data(),ru.data(),zu.data(),lu.data(),rv.data(),zv.data(),lv.data());
    for(int i=0;i<p.ns*p.nZnT;++i){
        checkNear(h_r[i],r[i],1e-12,"R",i/p.nZnT,i%p.nZnT);
        checkNear(h_ru[i],ru[i],1e-12,"Ru",i/p.nZnT,i%p.nZnT);
    }
    delete[] h_r; delete[] h_ru;
    printf(g_failures==lf?"PASS\n":"FAIL\n");
    return g_failures-lf;
}

static int t_inv_zeta(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_inverseDFT_zetaDerivative ... ");
    std::vector<double> cc_(p.ns*p.mnmax,0),ss_(p.ns*p.mnmax,0),zs(p.ns*p.mnmax,0),zc(p.ns*p.mnmax,0),ls_(p.ns*p.mnmax,0),lcs(p.ns*p.mnmax,0);
    int m1=1*(p.ntor+1)+1;  // R_11 (cos(θ-ζ)): folded rmncc=rmnss=0.2
    for(int j=0;j<p.ns;++j){ cc_[j+m1*p.ns]=0.2; ss_[j+m1*p.ns]=0.2; }
    double* h_rv=new double[p.ns*p.nZnT];
    std::vector<double> r(p.ns*p.nZnT),z(p.ns*p.nZnT),l(p.ns*p.nZnT);
    std::vector<double> ru(p.ns*p.nZnT),zu(p.ns*p.nZnT),lu(p.ns*p.nZnT);
    std::vector<double> rv(p.ns*p.nZnT),zv(p.ns*p.nZnT),lv(p.ns*p.nZnT);
    gpuInv(st,fp,p,cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),lcs.data());
    cc(cudaMemcpy(h_rv,fp.d_rv_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get rv");
    std::vector<double> hcc,hss,hsc,hcs; std::vector<int> hxm,hxn;
    hostBasis(p,hcc,hss,hsc,hcs,hxm,hxn);
    cpuInvDFT(cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),lcs.data(),
        hcc.data(),hss.data(),hsc.data(),hcs.data(),
        hxm.data(),hxn.data(),p.ns,p.mnmax,p.nZnT,
        r.data(),z.data(),l.data(),ru.data(),zu.data(),lu.data(),rv.data(),zv.data(),lv.data());
    for(int i=0;i<p.ns*p.nZnT;++i) checkNear(h_rv[i],rv[i],1e-12,"Rv",i/p.nZnT,i%p.nZnT);
    delete[] h_rv;
    printf(g_failures==lf?"PASS\n":"FAIL\n");
    return g_failures-lf;
}

static int t_fwd_const(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_forwardDFT_constant ... ");
    size_t nbr=p.ns*p.nZnT*sizeof(double);
    std::vector<double> fr(p.ns*p.nZnT,3.0);
    std::vector<double> fs(6*p.ns*p.mnmax,0);
    double* d_fs; cc(cudaMalloc(&d_fs,6*p.ns*p.mnmax*sizeof(double)),"fs");
    ConstraintWorkspace cw_zero{};
    size_t nfc = (size_t)p.ns * p.nZnT * sizeof(double);
    cudaMalloc(&cw_zero.d_frcon_e, nfc); cudaMemset(cw_zero.d_frcon_e, 0, nfc);
    cudaMalloc(&cw_zero.d_frcon_o, nfc); cudaMemset(cw_zero.d_frcon_o, 0, nfc);
    cudaMalloc(&cw_zero.d_fzcon_e, nfc); cudaMemset(cw_zero.d_fzcon_e, 0, nfc);
    cudaMalloc(&cw_zero.d_fzcon_o, nfc); cudaMemset(cw_zero.d_fzcon_o, 0, nfc);
    // armn_e = 3.0 on all surfaces; everything else zero.
    cc(cudaMemset(fp.d_armn_e, 0, nbr), "ms"); cc(cudaMemcpy(fp.d_armn_e, fr.data(), nbr, cudaMemcpyHostToDevice), "armn_e");
    cc(cudaMemset(fp.d_armn_o, 0, nbr), "ms"); cc(cudaMemset(fp.d_azmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_azmn_o, 0, nbr), "ms");
    cc(cudaMemset(fp.d_brmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_brmn_o, 0, nbr), "ms");
    cc(cudaMemset(fp.d_bzmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_bzmn_o, 0, nbr), "ms");
    cc(cudaMemset(fp.d_blmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_blmn_o, 0, nbr), "ms");
    cc(cudaMemset(fp.d_crmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_crmn_o, 0, nbr), "ms");
    cc(cudaMemset(fp.d_czmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_czmn_o, 0, nbr), "ms");
    cc(cudaMemset(fp.d_clmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_clmn_o, 0, nbr), "ms");
    forwardDFT(fp,d_fs,p,cw_zero);
    cc(cudaMemcpy(fs.data(),d_fs,6*p.ns*p.mnmax*sizeof(double),cudaMemcpyDeviceToHost),"get fs");
    for(int j=0;j<p.ns-1;++j){
        checkNear(fs[j+0*p.mnmax*p.ns],3.0,1e-13,"fR_cc",j,0);
        for(int m=1;m<p.mnmax;++m){
            checkMode(fs[j+m*p.ns+0*p.mnmax*p.ns],0.0,1e-13,"fR0",j,m);
            checkMode(fs[j+m*p.ns+3*p.mnmax*p.ns],0.0,1e-13,"fR0_ss",j,m);
        }
    }
    cudaFree(d_fs);
    cudaFree(cw_zero.d_frcon_e); cudaFree(cw_zero.d_frcon_o);
    cudaFree(cw_zero.d_fzcon_e); cudaFree(cw_zero.d_fzcon_o);
    printf(g_failures==lf?"PASS\n":"FAIL\n");
    return g_failures-lf;
}

static int t_fwd_sine(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_forwardDFT_sine ... ");
    size_t nbr=p.ns*p.nZnT*sizeof(double);
    std::vector<double> fz(p.ns*p.nZnT,0);
    std::vector<double> fs(6*p.ns*p.mnmax,0);
    double* d_fs; cc(cudaMalloc(&d_fs,6*p.ns*p.mnmax*sizeof(double)),"fs");
    ConstraintWorkspace cw_zero{};
    size_t nfc = (size_t)p.ns * p.nZnT * sizeof(double);
    cudaMalloc(&cw_zero.d_frcon_e, nfc); cudaMemset(cw_zero.d_frcon_e, 0, nfc);
    cudaMalloc(&cw_zero.d_frcon_o, nfc); cudaMemset(cw_zero.d_frcon_o, 0, nfc);
    cudaMalloc(&cw_zero.d_fzcon_e, nfc); cudaMemset(cw_zero.d_fzcon_e, 0, nfc);
    cudaMalloc(&cw_zero.d_fzcon_o, nfc); cudaMemset(cw_zero.d_fzcon_o, 0, nfc);
    // F_Z = sin(θ)cos(ζ) on interior surfaces: picked up by fzsc of (1,1).
    // m=1 is ODD parity: the input lives in azmn_o (parity-split arrays).
    for(int j=0;j<p.ns-1;++j) for(int k=0;k<p.nZnT;++k){
        int it=k%p.ntheta, iz=k/p.ntheta;
        double th=2*M_PI*it/p.ntheta, ze=2*M_PI*iz/p.nzeta;
        fz[k+j*p.nZnT]=sin(th)*cos(ze);
    }
    cc(cudaMemset(fp.d_armn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_armn_o, 0, nbr), "ms");
    cc(cudaMemset(fp.d_azmn_e, 0, nbr), "ms");
    cc(cudaMemcpy(fp.d_azmn_o, fz.data(), nbr, cudaMemcpyHostToDevice), "azmn_o");
    cc(cudaMemset(fp.d_brmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_brmn_o, 0, nbr), "ms");
    cc(cudaMemset(fp.d_bzmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_bzmn_o, 0, nbr), "ms");
    cc(cudaMemset(fp.d_blmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_blmn_o, 0, nbr), "ms");
    cc(cudaMemset(fp.d_crmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_crmn_o, 0, nbr), "ms");
    cc(cudaMemset(fp.d_czmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_czmn_o, 0, nbr), "ms");
    cc(cudaMemset(fp.d_clmn_e, 0, nbr), "ms"); cc(cudaMemset(fp.d_clmn_o, 0, nbr), "ms");
    forwardDFT(fp,d_fs,p,cw_zero);
    cc(cudaMemcpy(fs.data(),d_fs,6*p.ns*p.mnmax*sizeof(double),cudaMemcpyDeviceToHost),"get fs");
    int m11=1*(p.ntor+1)+1;
    // vmecpp convention: the reduced-grid trapezoid with mscale*nscale
    // weights gives the normalized coefficient 0.5 for a unit raw mode
    // (mscale*nscale = 2, sum sin^2*cos^2 over the reduced grid = 1/2).
    // The axis (j=0) is m=0 only (vmecpp dft_ForcesToFourier mmax=1).
    for(int j=0;j<p.ns-1;++j){
        double exp = (j == 0) ? 0.0 : 0.5;
        checkNear(fs[j+m11*p.ns+1*p.mnmax*p.ns],exp,1e-13,"fZ_sc",j,m11);
    }
    cudaFree(d_fs);
    cudaFree(cw_zero.d_frcon_e); cudaFree(cw_zero.d_frcon_o);
    cudaFree(cw_zero.d_fzcon_e); cudaFree(cw_zero.d_fzcon_o);
    printf(g_failures==lf?"PASS\n":"FAIL\n");
    return g_failures-lf;
}

static int t_gpuVcpu_inv(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_gpuVcpu_inverseDFT ... ");
    std::vector<double> cc_(p.ns*p.mnmax,0),ss_(p.ns*p.mnmax,0),zs(p.ns*p.mnmax,0),zc(p.ns*p.mnmax,0),ls_(p.ns*p.mnmax,0),lcs(p.ns*p.mnmax,0);
    for(int j=0;j<p.ns;++j) for(int m=0;m<p.mnmax;++m){
        cc_[j+m*p.ns]=0.001*(m+1)*(j+1);
        ss_[j+m*p.ns]=0.002*(m+1)*(j+1);
        zs[j+m*p.ns]=0.003*(m+1)*(j+1);
        zc[j+m*p.ns]=0.004*(m+1)*(j+1);
        ls_[j+m*p.ns]=0.005*(m+1)*(j+1);
        lcs[j+m*p.ns]=0.006*(m+1)*(j+1);
    }
    double* h_r=new double[p.ns*p.nZnT], *h_z=new double[p.ns*p.nZnT];
    double* h_l=new double[p.ns*p.nZnT], *h_ru=new double[p.ns*p.nZnT];
    double* h_zu=new double[p.ns*p.nZnT], *h_lu=new double[p.ns*p.nZnT];
    double* h_rv=new double[p.ns*p.nZnT], *h_zv=new double[p.ns*p.nZnT];
    double* h_lv=new double[p.ns*p.nZnT];
    std::vector<double> r(p.ns*p.nZnT),z(p.ns*p.nZnT),l(p.ns*p.nZnT);
    std::vector<double> ru(p.ns*p.nZnT),zu(p.ns*p.nZnT),lu(p.ns*p.nZnT);
    std::vector<double> rv(p.ns*p.nZnT),zv(p.ns*p.nZnT),lv(p.ns*p.nZnT);
    gpuInv(st,fp,p,cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),lcs.data());
    cc(cudaMemcpy(h_r,fp.d_r_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get r");
    cc(cudaMemcpy(h_z,fp.d_z_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get z");
    cc(cudaMemcpy(h_l,fp.d_l_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get l");
    cc(cudaMemcpy(h_ru,fp.d_ru_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get ru");
    cc(cudaMemcpy(h_zu,fp.d_zu_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get zu");
    cc(cudaMemcpy(h_lu,fp.d_lu_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get lu");
    cc(cudaMemcpy(h_rv,fp.d_rv_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get rv");
    cc(cudaMemcpy(h_zv,fp.d_zv_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get zv");
    cc(cudaMemcpy(h_lv,fp.d_lv_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"get lv");
    std::vector<double> hcc,hss,hsc,hcs; std::vector<int> hxm,hxn;
    hostBasis(p,hcc,hss,hsc,hcs,hxm,hxn);
    cpuInvDFT(cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),lcs.data(),
        hcc.data(),hss.data(),hsc.data(),hcs.data(),
        hxm.data(),hxn.data(),p.ns,p.mnmax,p.nZnT,
        r.data(),z.data(),l.data(),ru.data(),zu.data(),lu.data(),rv.data(),zv.data(),lv.data());
    for(int i=0;i<p.ns*p.nZnT;++i){
        checkNear(h_r[i],r[i],1e-12,"R",i/p.nZnT,i%p.nZnT);
        checkNear(h_z[i],z[i],1e-12,"Z",i/p.nZnT,i%p.nZnT);
        checkNear(h_l[i],l[i],1e-12,"L",i/p.nZnT,i%p.nZnT);
        checkNear(h_ru[i],ru[i],1e-12,"Ru",i/p.nZnT,i%p.nZnT);
        checkNear(h_zu[i],zu[i],1e-12,"Zu",i/p.nZnT,i%p.nZnT);
        checkNear(h_lu[i],lu[i],1e-12,"Lu",i/p.nZnT,i%p.nZnT);
        checkNear(h_rv[i],rv[i],1e-12,"Rv",i/p.nZnT,i%p.nZnT);
        checkNear(h_zv[i],zv[i],1e-12,"Zv",i/p.nZnT,i%p.nZnT);
        checkNear(h_lv[i],lv[i],1e-12,"Lv",i/p.nZnT,i%p.nZnT);
    }
    delete[] h_r; delete[] h_z; delete[] h_l;
    delete[] h_ru; delete[] h_zu; delete[] h_lu;
    delete[] h_rv; delete[] h_zv; delete[] h_lv;
    printf(g_failures==lf?"PASS\n":"FAIL\n");
    return g_failures-lf;
}

int main(){
    printf("=== Fourier Transform Tests (folded n>=0 basis) ===\n");
    GridParams p;
    p.ns=kNs; p.mnmax=kMnmax; p.ntheta=kNtheta; p.nzeta=kNzeta;
    p.nfp=kNfp; p.nZnT=kNZnT; p.mpol=kMpol; p.ntor=kNtor;
    p.ncurr=0; p.delt=1.0; p.ftol=1e-14; p.max_iter=10; p.lamscale=1.0;

    cublasHandle_t cb; cublasCreate(&cb);
    FourierPlan fp=fourierCreate(p,cb);
    SpectralState st{};
    size_t nb=p.ns*p.mnmax*sizeof(double);
    cc(cudaMalloc(&st.d_rmncc,nb),"cc"); cc(cudaMalloc(&st.d_rmnss,nb),"ss");
    cc(cudaMalloc(&st.d_zmnsc,nb),"zsc"); cc(cudaMalloc(&st.d_zmncs,nb),"zcs");
    cc(cudaMalloc(&st.d_lmnsc,nb),"lsc"); cc(cudaMalloc(&st.d_lmncs,nb),"lcs");
    cc(cudaMalloc(&st.d_v_rmncc,nb),"vcc"); cc(cudaMalloc(&st.d_v_rmnss,nb),"vss");
    cc(cudaMalloc(&st.d_v_zmnsc,nb),"vzsc"); cc(cudaMalloc(&st.d_v_zmncs,nb),"vzcs");
    cc(cudaMalloc(&st.d_v_lmnsc,nb),"vlsc"); cc(cudaMalloc(&st.d_v_lmncs,nb),"vlcs");

    t_inv_constR(p,cb,fp,st);
    t_inv_theta(p,cb,fp,st);
    t_inv_zeta(p,cb,fp,st);
    t_fwd_const(p,cb,fp,st);
    t_fwd_sine(p,cb,fp,st);
    t_gpuVcpu_inv(p,cb,fp,st);

    fourierFree(fp);
    cublasDestroy(cb);
    printf(g_failures==0?"ALL PASS\n":"%d FAILURES\n",g_failures);
    return g_failures==0?0:1;
}
