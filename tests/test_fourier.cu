// test_fourier.cu — algorithm correctness tests for parity-split DFT transforms.
// Build: cd cuMES/build && cmake .. && make test_fourier && ./test_fourier
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cublas_v2.h>
#include <vector>
#include "vmec_types.h"
#include "fourier.cuh"
#include "constraint.cuh"

constexpr int kNs=3, kMpol=3, kNtor=2, kNtheta=16, kNzeta=8, kNfp=1;
constexpr int kMnmax=kMpol*kNtor, kNZnT=kNtheta*kNzeta;

static int g_failures=0;
static void checkNear(double g,double e,double t,const char* s,int j,int k){
    if(fabs(g-e)>t){fprintf(stderr,"FAIL [%s] j=%d k=%d got=%.15e exp=%.15e\n",s,j,k,g,e);++g_failures;}}
static void checkMode(double g,double e,double t,const char* s,int j,int m){
    if(fabs(g-e)>t){fprintf(stderr,"FAIL [%s] j=%d m=%d got=%.15e exp=%.15e\n",s,j,m,g,e);++g_failures;}}
static void cc(cudaError_t e,const char* t){if(e!=cudaSuccess){fprintf(stderr,"CUDA[%s]:%s\n",t,cudaGetErrorString(e));exit(1);}}

// CPU: parity inverse DFT (m-parity convention matching vmecpp)
static void cpuInvDFT(const double* cc_, const double* ss_, const double* zsc_, const double* zcs_,
    const double* lsc_, const double* pcc, const double* pss, const double* psc, const double* pcs,
    const int* xm, int ns, int mnmax, int nZnT,
    double* r, double* z, double* l, double* ru, double* zu, double* lu){
    for(int j=0;j<ns;++j) for(int k=0;k<nZnT;++k){
        double re=0,ro=0,ze=0,zo=0,le=0,lo=0,rue=0,ruo=0,zue=0,zuo=0,lue=0,luo=0;
        for(int m=0;m<mnmax;++m){
            double cc_v=pcc[k+m*nZnT],ss_v=pss[k+m*nZnT],sc_v=psc[k+m*nZnT],cs_v=pcs[k+m*nZnT];
            int mi=xm[m];
            bool m_even = (mi % 2 == 0);
            // R: rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ)
            if (m_even) {
                re  += cc_[j+m*ns]*cc_v + ss_[j+m*ns]*ss_v;
                rue += -mi*cc_[j+m*ns]*sc_v + mi*ss_[j+m*ns]*cs_v;
            } else {
                ro  += cc_[j+m*ns]*cc_v + ss_[j+m*ns]*ss_v;
                ruo += -mi*cc_[j+m*ns]*sc_v + mi*ss_[j+m*ns]*cs_v;
            }
            // Z: zmnsc*sin(mθ)cos(nζ) + zmncs*(-cos(mθ)sin(nζ))
            if (m_even) {
                ze  += zsc_[j+m*ns]*sc_v - zcs_[j+m*ns]*cs_v;
                zue += mi*zsc_[j+m*ns]*cc_v + mi*zcs_[j+m*ns]*ss_v;
            } else {
                zo  += zsc_[j+m*ns]*sc_v - zcs_[j+m*ns]*cs_v;
                zuo += mi*zsc_[j+m*ns]*cc_v + mi*zcs_[j+m*ns]*ss_v;
            }
            // λ: lmnsc*sin(mθ)cos(nζ)
            if (m_even) {
                le  += lsc_[j+m*ns]*sc_v;
                lue += mi*lsc_[j+m*ns]*cc_v;
            } else {
                lo  += lsc_[j+m*ns]*sc_v;
                luo += mi*lsc_[j+m*ns]*cc_v;
            }
        }
        r[k+j*nZnT]=re+ro; z[k+j*nZnT]=ze+zo; l[k+j*nZnT]=le+lo;
        ru[k+j*nZnT]=rue+ruo; zu[k+j*nZnT]=zue+zuo; lu[k+j*nZnT]=lue+luo;
    }
}

// CPU: parity forward DFT (m-parity convention)
static void cpuFwdDFT(const double* fr, const double* fz, const double* fl,
    const double* pcc, const double* pss, const double* psc, const double* pcs,
    const int* xm, const int* xn, int ns, int mnmax, int nZnT, double* fs){
    for(int j=0;j<ns;++j) for(int m=0;m<mnmax;++m){
        int mm=xm[m], nn=xn[m];
        double w;
        if(mm==0&&nn==0)w=1.0/nZnT; else if(mm==0||nn==0)w=2.0/nZnT; else w=4.0/nZnT;
        double sre=0,sro=0,sze=0,szo=0,sle=0;
        for(int k=0;k<nZnT;++k){
            double cc_v=pcc[k+m*nZnT],ss_v=pss[k+m*nZnT],sc_v=psc[k+m*nZnT],cs_v=pcs[k+m*nZnT];
            int idx=k+j*nZnT;
            // Both parities contain the same total force (for round-trip testing).
            // For even m, the e array is projected; for odd m, the o array.
            // Total force FR = even-m + odd-m contributions, so either way
            // the full FR is used for the projection.
            sre+=fr[idx]*cc_v; sro+=fr[idx]*ss_v;
            sze+=fz[idx]*sc_v; szo+=fz[idx]*(-cs_v);
            sle+=fl[idx]*sc_v;
        }
        fs[j+m*ns+0*mnmax*ns]=sre*w; fs[j+m*ns+1*mnmax*ns]=sze*w;
        fs[j+m*ns+2*mnmax*ns]=sle*w; fs[j+m*ns+3*mnmax*ns]=sro*w;
        fs[j+m*ns+4*mnmax*ns]=szo*w;
    }
}

// GPU helpers
static void gpuInv(SpectralState& st, FourierPlan& fp, const GridParams& p,
    const double* cc_, const double* ss_, const double* zsc_, const double* zcs_, const double* lsc_,
    double* r, double* z, double* l, double* ru, double* zu, double* lu){
    size_t nb=p.ns*p.mnmax*sizeof(double), nbr=p.ns*p.nZnT*sizeof(double);
    cc(cudaMemcpy(st.d_rmncc,cc_,nb,cudaMemcpyHostToDevice),"up cc");
    cc(cudaMemcpy(st.d_rmnss,ss_,nb,cudaMemcpyHostToDevice),"up ss");
    cc(cudaMemcpy(st.d_zmnsc,zsc_,nb,cudaMemcpyHostToDevice),"up zsc");
    cc(cudaMemcpy(st.d_zmncs,zcs_,nb,cudaMemcpyHostToDevice),"up zcs");
    cc(cudaMemcpy(st.d_lmnsc,lsc_,nb,cudaMemcpyHostToDevice),"up lsc");
    inverseDFT(fp,st,p);
    cc(cudaMemcpy(r,fp.d_r_real,nbr,cudaMemcpyDeviceToHost),"dn r");
    cc(cudaMemcpy(z,fp.d_z_real,nbr,cudaMemcpyDeviceToHost),"dn z");
    cc(cudaMemcpy(l,fp.d_l_real,nbr,cudaMemcpyDeviceToHost),"dn l");
    cc(cudaMemcpy(ru,fp.d_ru_real,nbr,cudaMemcpyDeviceToHost),"dn ru");
    cc(cudaMemcpy(zu,fp.d_zu_real,nbr,cudaMemcpyDeviceToHost),"dn zu");
    cc(cudaMemcpy(lu,fp.d_lu_real,nbr,cudaMemcpyDeviceToHost),"dn lu");
}

static void gpuFwd(FourierPlan& fp, const GridParams& p,
    const double* fr, const double* fz, const double* fl, double* fs){
    size_t nbr=p.ns*p.nZnT*sizeof(double), nbs=5*p.ns*p.mnmax*sizeof(double);
    // Copy total forces to both parity arrays for round-trip testing.
    // The m-parity forward DFT projects:
    //   even m: uses armn_e/azmn_e/brmn_e/bzmn_e/blmn_e
    //   odd m:  uses armn_o/azmn_o/brmn_o/bzmn_o/blmn_o
    cc(cudaMemcpy(fp.d_armn_e,fr,nbr,cudaMemcpyHostToDevice),"up ae");
    cc(cudaMemcpy(fp.d_armn_o,fr,nbr,cudaMemcpyHostToDevice),"up ao");
    cc(cudaMemcpy(fp.d_azmn_e,fz,nbr,cudaMemcpyHostToDevice),"up aze");
    cc(cudaMemcpy(fp.d_azmn_o,fz,nbr,cudaMemcpyHostToDevice),"up azo");
    cc(cudaMemcpy(fp.d_blmn_e,fl,nbr,cudaMemcpyHostToDevice),"up ble");
    cc(cudaMemcpy(fp.d_blmn_o,fl,nbr,cudaMemcpyHostToDevice),"up blo");
    // Poloidal forces are zero for round-trip test (no IBP needed)
    cc(cudaMemset(fp.d_brmn_e,0,nbr),"zero be");
    cc(cudaMemset(fp.d_brmn_o,0,nbr),"zero bo");
    cc(cudaMemset(fp.d_bzmn_e,0,nbr),"zero bze");
    cc(cudaMemset(fp.d_bzmn_o,0,nbr),"zero bzo");
    double* d_fs; cc(cudaMalloc(&d_fs,nbs),"malloc fs");
    ConstraintWorkspace cw_zero{}; cudaMalloc(&cw_zero.d_frcon_e, (size_t)1); cudaMemset(cw_zero.d_frcon_e, 0, 8);
      cudaMalloc(&cw_zero.d_frcon_o, (size_t)1); cudaMemset(cw_zero.d_frcon_o, 0, 8);
      cudaMalloc(&cw_zero.d_fzcon_e, (size_t)1); cudaMemset(cw_zero.d_fzcon_e, 0, 8);
      cudaMalloc(&cw_zero.d_fzcon_o, (size_t)1); cudaMemset(cw_zero.d_fzcon_o, 0, 8);
      forwardDFT(fp,d_fs,p,cw_zero);
    cc(cudaMemcpy(fs,d_fs,nbs,cudaMemcpyDeviceToHost),"dn fs");
    cudaFree(d_fs);
}

// Test 1
static int t_inv_constR(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_inverseDFT_constantR ... ");
    std::vector<double> cc_(p.ns*p.mnmax,0),ss_(p.ns*p.mnmax,0),zs(p.ns*p.mnmax,0),zc(p.ns*p.mnmax,0),ls_(p.ns*p.mnmax,0);
    for(int j=0;j<p.ns;++j){cc_[j]=2.0; ss_[j]=2.0;}
    std::vector<double> r(p.ns*p.nZnT),z(p.ns*p.nZnT),l(p.ns*p.nZnT),ru(p.ns*p.nZnT),zu(p.ns*p.nZnT),lu(p.ns*p.nZnT);
    gpuInv(st,fp,p,cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),r.data(),z.data(),l.data(),ru.data(),zu.data(),lu.data());
    for(int j=0;j<p.ns;++j) for(int k=0;k<p.nZnT;++k){
        checkNear(r[k+j*p.nZnT],2.0,1e-14,"R",j,k); checkNear(z[k+j*p.nZnT],0.0,1e-14,"Z",j,k);
        checkNear(ru[k+j*p.nZnT],0.0,1e-14,"dR/dth",j,k);
    }
    printf("%s\n",g_failures==lf?"PASS":"FAIL"); return g_failures==lf?0:1;
}

// Test 2: theta derivative
static int t_inv_theta(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_inverseDFT_thetaDerivative ... ");
    int m1=1*p.ntor+0;
    std::vector<double> cc_(p.ns*p.mnmax,0),ss_(p.ns*p.mnmax,0),zs(p.ns*p.mnmax,0),zc(p.ns*p.mnmax,0),ls_(p.ns*p.mnmax,0);
    cc_[0+m1*p.ns]=1.0; ss_[0+m1*p.ns]=1.0;
    std::vector<double> r(p.ns*p.nZnT),z(p.ns*p.nZnT),l(p.ns*p.nZnT),ru(p.ns*p.nZnT),zu(p.ns*p.nZnT),lu(p.ns*p.nZnT);
    gpuInv(st,fp,p,cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),r.data(),z.data(),l.data(),ru.data(),zu.data(),lu.data());
    // Odd-m real-space carries the decomposition physical/max(sqrt(s_F),
    // sqrt(1/(ns-1))) — at the axis (j=0): maxsc = sqrt(1/(ns-1)).
    double maxsc = sqrt(1.0 / (p.ns - 1.0));
    for(int k=0;k<p.nZnT;++k){
        int it=k%p.ntheta; double th=2.0*M_PI*it/p.ntheta;
        checkNear(r[k],cos(th)/maxsc,1e-13,"R",0,k);
        checkNear(ru[k],-sin(th)/maxsc,1e-13,"dR/dth",0,k);
    }
    printf("%s\n",g_failures==lf?"PASS":"FAIL"); return g_failures==lf?0:1;
}

// Test 3: zeta derivative
static int t_inv_zeta(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_inverseDFT_zetaDerivative ... ");
    int n1=0*p.ntor+1;
    std::vector<double> cc_(p.ns*p.mnmax,0),ss_(p.ns*p.mnmax,0),zs(p.ns*p.mnmax,0),zc(p.ns*p.mnmax,0),ls_(p.ns*p.mnmax,0);
    cc_[0+n1*p.ns]=1.0; ss_[0+n1*p.ns]=1.0;
    std::vector<double> r(p.ns*p.nZnT),z(p.ns*p.nZnT),l(p.ns*p.nZnT),ru(p.ns*p.nZnT),zu(p.ns*p.nZnT),lu(p.ns*p.nZnT);
    gpuInv(st,fp,p,cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),r.data(),z.data(),l.data(),ru.data(),zu.data(),lu.data());
    std::vector<double> rv(p.ns*p.nZnT); cc(cudaMemcpy(rv.data(),fp.d_rv_real,p.ns*p.nZnT*sizeof(double),cudaMemcpyDeviceToHost),"dn rv");
    for(int k=0;k<p.nZnT;++k){
        int iz=k/p.ntheta; double ze=2.0*M_PI*iz/p.nzeta;
        checkNear(rv[k],-sin(ze),1e-13,"dR/dz",0,k);
    }
    printf("%s\n",g_failures==lf?"PASS":"FAIL"); return g_failures==lf?0:1;
}

// Test 4: forward DFT constant
static int t_fwd_const(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_forwardDFT_constant ... ");
    std::vector<double> fr(p.ns*p.nZnT,3.0),fz(p.ns*p.nZnT,0),fl(p.ns*p.nZnT,0),fs(5*p.ns*p.mnmax);
    gpuFwd(fp,p,fr.data(),fz.data(),fl.data(),fs.data());
    for(int j=0;j<p.ns;++j){
        checkMode(fs[j+0*p.mnmax*p.ns],3.0,1e-14,"fR_cc",j,0);
        for(int m=1;m<p.mnmax;++m){checkMode(fs[j+m*p.ns+0*p.mnmax*p.ns],0.0,1e-14,"fR0",j,m);}
        for(int m=1;m<p.mnmax;++m){checkMode(fs[j+m*p.ns+3*p.mnmax*p.ns],0.0,1e-14,"fR0_ss",j,m);}
    }
    printf("%s\n",g_failures==lf?"PASS":"FAIL"); return g_failures==lf?0:1;
}

// Test 5: forward DFT sine
static int t_fwd_sine(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_forwardDFT_sine ... ");
    std::vector<double> fr(p.ns*p.nZnT,0),fz(p.ns*p.nZnT,0),fl(p.ns*p.nZnT,0),fs(5*p.ns*p.mnmax);
    for(int k=0;k<p.nZnT;++k){int it=k%p.ntheta; fr[k]=sin(2.0*M_PI*it/p.ntheta);}
    gpuFwd(fp,p,fr.data(),fz.data(),fl.data(),fs.data());
    for(int m=0;m<p.mnmax;++m){
        checkMode(fs[0+m*p.ns+0*p.mnmax*p.ns],0.0,1e-13,"fR_cc",0,m);
        checkMode(fs[0+m*p.ns+3*p.mnmax*p.ns],0.0,1e-13,"fR_ss",0,m);
    }
    printf("%s\n",g_failures==lf?"PASS":"FAIL"); return g_failures==lf?0:1;
}

// Test 6: round-trip
static int t_roundTrip(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_roundTrip_identity ... ");
    // Precompute xm/xn to know which modes have non-zero parity components
    std::vector<int> hxm(p.mnmax),hxn(p.mnmax);
    for(int m=0;m<p.mpol;++m) for(int n=0;n<p.ntor;++n){int md=m*p.ntor+n; hxm[md]=m; hxn[md]=n*p.nfp;}

    std::vector<double> cc_(p.ns*p.mnmax),ss_(p.ns*p.mnmax),zs(p.ns*p.mnmax),zc(p.ns*p.mnmax),ls_(p.ns*p.mnmax);
    for(int j=0;j<p.ns;++j) for(int m=0;m<p.mnmax;++m){
        int idx=j+m*p.ns;
        int mm=hxm[m], nn=hxn[m];
        // Set parity coefficients — only non-zero where basis is non-vanishing
        cc_[idx]=0.1*(j*p.mnmax+m+1);
        ss_[idx]=(mm>0&&nn>0)?0.15*(j*p.mnmax+m+1):0.0; // sin(mθ)sin(nζ)=0 if m=0 or n=0
        zs[idx]=(mm>0)?0.1*(j*p.mnmax+m+11):0.0;  // sin(mθ)cos(nζ)=0 if m=0
        zc[idx]=(mm>0&&nn>0)?0.12*(j*p.mnmax+m+11):0.0; // cos(mθ)sin(nζ)=0 if m=0 or n=0
        ls_[idx]=(mm>0)?0.1*(j*p.mnmax+m+21):0.0;  // sin(mθ)cos(nζ)=0 if m=0
    }
    std::vector<double> r(p.ns*p.nZnT),z(p.ns*p.nZnT),l(p.ns*p.nZnT),ru(p.ns*p.nZnT),zu(p.ns*p.nZnT),lu(p.ns*p.nZnT);
    gpuInv(st,fp,p,cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),r.data(),z.data(),l.data(),ru.data(),zu.data(),lu.data());
    std::vector<double> fs(5*p.ns*p.mnmax);
    gpuFwd(fp,p,r.data(),z.data(),l.data(),fs.data());
    double tol=1e-12;
    for(int j=0;j<p.ns;++j) for(int m=0;m<p.mnmax;++m){
        checkMode(fs[j+m*p.ns+0*p.mnmax*p.ns],cc_[j+m*p.ns],tol,"RT_cc",j,m);
        checkMode(fs[j+m*p.ns+3*p.mnmax*p.ns],ss_[j+m*p.ns],tol,"RT_ss",j,m);
        checkMode(fs[j+m*p.ns+2*p.mnmax*p.ns],ls_[j+m*p.ns],tol,"RT_lsc",j,m);
        checkMode(fs[j+m*p.ns+1*p.mnmax*p.ns],zs[j+m*p.ns],tol,"RT_zsc",j,m);
        checkMode(fs[j+m*p.ns+4*p.mnmax*p.ns],zc[j+m*p.ns],tol,"RT_zcs",j,m);
    }
    printf("%s\n",g_failures==lf?"PASS":"FAIL"); return g_failures==lf?0:1;
}

// Test 7: GPU vs CPU inverse
static int t_gpuVcpu_inv(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_gpuVsCpu_inverse ... ");
    std::vector<double> cc_(p.ns*p.mnmax),ss_(p.ns*p.mnmax),zs(p.ns*p.mnmax),zc(p.ns*p.mnmax),ls_(p.ns*p.mnmax);
    for(int j=0;j<p.ns;++j) for(int m=0;m<p.mnmax;++m){
        int idx=j+m*p.ns;
        cc_[idx]=0.1*(j*p.mnmax+m+1); ss_[idx]=0.15*(j*p.mnmax+m+3);
        zs[idx]=(m==0)?0.0:0.1*(j*p.mnmax+m+11); zc[idx]=(m==0)?0.0:0.12*(j*p.mnmax+m+5);
        ls_[idx]=0.1*(j*p.mnmax+m+21);
    }
    std::vector<double> gr(p.ns*p.nZnT),gz(p.ns*p.nZnT),gl(p.ns*p.nZnT),gru(p.ns*p.nZnT),gzu(p.ns*p.nZnT),glu(p.ns*p.nZnT);
    gpuInv(st,fp,p,cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),gr.data(),gz.data(),gl.data(),gru.data(),gzu.data(),glu.data());
    // Build CPU basis
    std::vector<double> hcc(p.nZnT*p.mnmax),hss(p.nZnT*p.mnmax),hsc(p.nZnT*p.mnmax),hcs(p.nZnT*p.mnmax);
    std::vector<int> hxm(p.mnmax),hxn(p.mnmax);
    for(int m=0;m<p.mpol;++m) for(int n=0;n<p.ntor;++n){int md=m*p.ntor+n; hxm[md]=m; hxn[md]=n*p.nfp;}
    for(int it=0;it<p.ntheta;++it){double th=2.0*M_PI*it/p.ntheta;
        for(int iz=0;iz<p.nzeta;++iz){double ze=2.0*M_PI*iz/p.nzeta; int id=iz*p.ntheta+it;
            for(int m=0;m<p.mnmax;++m){int mi=hxm[m],ni=hxn[m]; double cm=cos(mi*th),sm=sin(mi*th),cn=cos(ni*ze),sn=sin(ni*ze);
                hcc[id+m*p.nZnT]=cm*cn; hss[id+m*p.nZnT]=sm*sn; hsc[id+m*p.nZnT]=sm*cn; hcs[id+m*p.nZnT]=cm*sn;}
        }
    }
    std::vector<double> cr(p.ns*p.nZnT),cz(p.ns*p.nZnT),cl(p.ns*p.nZnT),cru(p.ns*p.nZnT),czu(p.ns*p.nZnT),clu(p.ns*p.nZnT);
    cpuInvDFT(cc_.data(),ss_.data(),zs.data(),zc.data(),ls_.data(),hcc.data(),hss.data(),hsc.data(),hcs.data(),hxm.data(),p.ns,p.mnmax,p.nZnT,cr.data(),cz.data(),cl.data(),cru.data(),czu.data(),clu.data());
    double tol=1e-13;
    for(int j=0;j<p.ns;++j) for(int k=0;k<p.nZnT;++k){
        int idx=k+j*p.nZnT; checkNear(gr[idx],cr[idx],tol,"R_cpu",j,k); checkNear(gru[idx],cru[idx],tol,"dR_cpu",j,k);
    }
    printf("%s\n",g_failures==lf?"PASS":"FAIL"); return g_failures==lf?0:1;
}

// Test 8: GPU vs CPU forward
static int t_gpuVcpu_fwd(GridParams& p, cublasHandle_t cb, FourierPlan& fp, SpectralState& st){
    int lf=g_failures; printf("  test_gpuVsCpu_forward ... ");
    std::vector<double> fr(p.ns*p.nZnT),fz(p.ns*p.nZnT),fl(p.ns*p.nZnT);
    for(int j=0;j<p.ns;++j) for(int k=0;k<p.nZnT;++k){
        int idx=k+j*p.nZnT; fr[idx]=0.1*((j*p.nZnT+k)%17-8); fz[idx]=0.1*((j*p.nZnT+k+5)%13-6); fl[idx]=0.1*((j*p.nZnT+k+10)%11-5);
    }
    std::vector<double> gfs(5*p.ns*p.mnmax); gpuFwd(fp,p,fr.data(),fz.data(),fl.data(),gfs.data());
    std::vector<double> hcc(p.nZnT*p.mnmax),hss(p.nZnT*p.mnmax),hsc(p.nZnT*p.mnmax),hcs(p.nZnT*p.mnmax);
    std::vector<int> hxm(p.mnmax),hxn(p.mnmax);
    for(int m=0;m<p.mpol;++m) for(int n=0;n<p.ntor;++n){int md=m*p.ntor+n; hxm[md]=m; hxn[md]=n*p.nfp;}
    for(int it=0;it<p.ntheta;++it){double th=2.0*M_PI*it/p.ntheta;
        for(int iz=0;iz<p.nzeta;++iz){double ze=2.0*M_PI*iz/p.nzeta; int id=iz*p.ntheta+it;
            for(int m=0;m<p.mnmax;++m){int mi=hxm[m],ni=hxn[m]; double cm=cos(mi*th),sm=sin(mi*th),cn=cos(ni*ze),sn=sin(ni*ze);
                hcc[id+m*p.nZnT]=cm*cn; hss[id+m*p.nZnT]=sm*sn; hsc[id+m*p.nZnT]=sm*cn; hcs[id+m*p.nZnT]=cm*sn;}
        }
    }
    std::vector<double> cfs(5*p.ns*p.mnmax);
    cpuFwdDFT(fr.data(),fz.data(),fl.data(),hcc.data(),hss.data(),hsc.data(),hcs.data(),hxm.data(),hxn.data(),p.ns,p.mnmax,p.nZnT,cfs.data());
    double tol=1e-12;
    for(int j=0;j<p.ns;++j) for(int m=0;m<p.mnmax;++m) for(int c=0;c<5;++c)
        checkMode(gfs[j+m*p.ns+c*p.mnmax*p.ns],cfs[j+m*p.ns+c*p.mnmax*p.ns],tol,"fwd",j,m);
    printf("%s\n",g_failures==lf?"PASS":"FAIL"); return g_failures==lf?0:1;
}

int main(){
    GridParams p; p.ns=kNs; p.mnmax=kMnmax; p.ntheta=kNtheta; p.nzeta=kNzeta;
    p.nfp=kNfp; p.nZnT=kNZnT; p.mpol=kMpol; p.ntor=kNtor;
    printf("=== Fourier Transform Tests (Parity) ===\n\n");
    cublasHandle_t cb; cublasCreate(&cb);
    SpectralState st{}; size_t nb=p.ns*p.mnmax*sizeof(double);
    cc(cudaMalloc(&st.d_rmncc,nb),"rmncc"); cc(cudaMalloc(&st.d_rmnss,nb),"rmnss");
    cc(cudaMalloc(&st.d_zmnsc,nb),"zmnsc"); cc(cudaMalloc(&st.d_zmncs,nb),"zmncs");
    cc(cudaMalloc(&st.d_lmnsc,nb),"lmnsc");
    cc(cudaMalloc(&st.d_v_rmncc,nb),"vcc"); cc(cudaMalloc(&st.d_v_rmnss,nb),"vss");
    cc(cudaMalloc(&st.d_v_zmnsc,nb),"vzsc"); cc(cudaMalloc(&st.d_v_zmncs,nb),"vzcs");
    cc(cudaMalloc(&st.d_v_lmnsc,nb),"vlsc");
    FourierPlan fp=fourierCreate(p,cb);
    t_inv_constR(p,cb,fp,st); t_inv_theta(p,cb,fp,st); t_inv_zeta(p,cb,fp,st);
    t_fwd_const(p,cb,fp,st); t_fwd_sine(p,cb,fp,st);
    t_roundTrip(p,cb,fp,st); t_gpuVcpu_inv(p,cb,fp,st); t_gpuVcpu_fwd(p,cb,fp,st);
    printf("\n========================================\n  %d / 8 tests passed\n  %d assertion failures\n========================================\n",8-(g_failures>0?1:0),g_failures);
    cudaFree(st.d_rmncc); cudaFree(st.d_rmnss); cudaFree(st.d_zmnsc); cudaFree(st.d_zmncs); cudaFree(st.d_lmnsc);
    cudaFree(st.d_v_rmncc); cudaFree(st.d_v_rmnss); cudaFree(st.d_v_zmnsc); cudaFree(st.d_v_zmncs); cudaFree(st.d_v_lmnsc);
    fourierFree(fp); cublasDestroy(cb);
    return g_failures>0?1:0;
}
