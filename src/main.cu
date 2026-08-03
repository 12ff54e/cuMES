// main.cu — entry point: init → solve → output.
// Input selection: default Solovev (hardcoded); CUMES_INPUT=w7x selects the
// W7-X parameters from input_w7x.h. A JSON parser will replace both later.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

#include "input.h"
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"
#include "forces.cuh"
#include "solver.cuh"
#include "output.cuh"
#include "profiles.cuh"

static void checkCuda(cudaError_t err, const char* tag) {
    if(err!=cudaSuccess){fprintf(stderr,"CUDA error [%s]: %s\n",tag,cudaGetErrorString(err));exit(1);}
}
static void checkCublas(cublasStatus_t st, const char* tag) {
    if(st!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS error [%s]: %d\n",tag,(int)st);exit(1);}
}

static GridParams initParams(const InputParams& ip) {
    GridParams p;
    p.ns=ip.ns; p.mpol=ip.mpol; p.ntor=ip.ntor;
    p.ntheta=ip.ntheta; p.nzeta=ip.nzeta; p.nfp=ip.nfp;
    p.nZnT=p.ntheta*p.nzeta;
    p.mnmax=p.mpol*(p.ntor+1);   // folded basis: mode = m*(ntor+1)+n
    p.ncurr=ip.ncurr;
    p.delt=ip.delt; p.ftol=ip.ftol; p.max_iter=ip.max_iter;
    p.lamscale=0.0;              // set by profilesCreate
    return p;
}

// Initial state from vmecpp's interpFromBoundaryAndAxis (fourier_geometry.cc):
//   m=0: linear interpolation in s between the magnetic axis (raxis_c /
//        zaxis_s) and the boundary; zmnsc/rmnss have no m=0 content.
//   m>0: s^(m/2) radial envelope so higher modes vanish faster near the axis.
// cuMES stores the plain physical coefficients (vmecpp's internal state
// divides by mscale*nscale, but its mscale'd basis makes the real-space
// reconstruction identical).
static void initState(SpectralState& st, const GridParams& p, const InputParams& ip) {
    bool loadInit = false;
    if (const char* e = getenv("CUMES_LOAD_INIT")) loadInit = atoi(e) != 0;
    size_t nb = (size_t)p.ns * p.mnmax * sizeof(double);
    size_t nb_one = nb;
    if (loadInit) {
        FILE* fp = fopen("vmecpp_init.bin", "rb");
    if (fp) {
        int ns_file, mnmax_file;
        fread(&ns_file, sizeof(int), 1, fp);
        fread(&mnmax_file, sizeof(int), 1, fp);
        if (ns_file == p.ns && mnmax_file == p.mnmax) {
            printf("Loading initial state from vmecpp_init.bin (ns=%d, mnmax=%d)\n", ns_file, mnmax_file);
            auto* h_rmncc = new double[p.ns * p.mnmax];
            auto* h_zmnsc = new double[p.ns * p.mnmax];
            auto* h_lmnsc = new double[p.ns * p.mnmax];
            auto* h_rmnss = new double[p.ns * p.mnmax];
            auto* h_zmncs = new double[p.ns * p.mnmax];
            auto* h_lmncs = new double[p.ns * p.mnmax];
            fread(h_rmncc, sizeof(double), p.ns * p.mnmax, fp);
            fread(h_zmnsc, sizeof(double), p.ns * p.mnmax, fp);
            fread(h_lmnsc, sizeof(double), p.ns * p.mnmax, fp);
            fread(h_rmnss, sizeof(double), p.ns * p.mnmax, fp);
            fread(h_zmncs, sizeof(double), p.ns * p.mnmax, fp);
            fread(h_lmncs, sizeof(double), p.ns * p.mnmax, fp);
            fclose(fp);

            checkCuda(cudaMalloc(&st.d_rmncc,nb),"cc"); checkCuda(cudaMalloc(&st.d_rmnss,nb),"ss");
            checkCuda(cudaMalloc(&st.d_zmnsc,nb),"zsc"); checkCuda(cudaMalloc(&st.d_zmncs,nb),"zcs");
            checkCuda(cudaMalloc(&st.d_lmnsc,nb),"lsc"); checkCuda(cudaMalloc(&st.d_lmncs,nb),"lcs");
            checkCuda(cudaMalloc(&st.d_v_rmncc,nb),"vcc"); checkCuda(cudaMalloc(&st.d_v_rmnss,nb),"vss");
            checkCuda(cudaMalloc(&st.d_v_zmnsc,nb),"vzsc"); checkCuda(cudaMalloc(&st.d_v_zmncs,nb),"vzcs");
            checkCuda(cudaMalloc(&st.d_v_lmnsc,nb),"vlsc"); checkCuda(cudaMalloc(&st.d_v_lmncs,nb),"vlcs");

            checkCuda(cudaMemcpy(st.d_rmncc, h_rmncc, nb, cudaMemcpyHostToDevice), "cpy cc");
            checkCuda(cudaMemcpy(st.d_zmnsc, h_zmnsc, nb, cudaMemcpyHostToDevice), "cpy zsc");
            checkCuda(cudaMemcpy(st.d_lmnsc, h_lmnsc, nb, cudaMemcpyHostToDevice), "cpy lsc");
            checkCuda(cudaMemcpy(st.d_rmnss, h_rmnss, nb, cudaMemcpyHostToDevice), "cpy ss");
            checkCuda(cudaMemcpy(st.d_zmncs, h_zmncs, nb, cudaMemcpyHostToDevice), "cpy zcs");
            checkCuda(cudaMemcpy(st.d_lmncs, h_lmncs, nb, cudaMemcpyHostToDevice), "cpy lcs");

            // vmecpp stores boundary values separately (not in the spectral
            // state). cuMES embeds the boundary in the spectral coefficients
            // at j=ns-1. Patch the LCFS values to match the folded boundary;
            // also zero out m>0 modes at the magnetic axis (j=0) — vmecpp
            // does this via extrapolateTowardsAxis().
            {
                double *h_cc, *h_ss, *h_zsc, *h_zcs, *h_lsc, *h_lcs;
                h_cc = new double[p.ns*p.mnmax];
                h_ss = new double[p.ns*p.mnmax];
                h_zsc = new double[p.ns*p.mnmax];
                h_zcs = new double[p.ns*p.mnmax];
                h_lsc = new double[p.ns*p.mnmax];
                h_lcs = new double[p.ns*p.mnmax];
                checkCuda(cudaMemcpy(h_cc, st.d_rmncc, nb, cudaMemcpyDeviceToHost), "get cc");
                checkCuda(cudaMemcpy(h_ss, st.d_rmnss, nb, cudaMemcpyDeviceToHost), "get ss");
                checkCuda(cudaMemcpy(h_zsc, st.d_zmnsc, nb, cudaMemcpyDeviceToHost), "get zsc");
                checkCuda(cudaMemcpy(h_zcs, st.d_zmncs, nb, cudaMemcpyDeviceToHost), "get zcs");
                checkCuda(cudaMemcpy(h_lsc, st.d_lmnsc, nb, cudaMemcpyDeviceToHost), "get lsc");
                checkCuda(cudaMemcpy(h_lcs, st.d_lmncs, nb, cudaMemcpyDeviceToHost), "get lcs");
                int jB = p.ns - 1;  // LCFS index
                for (int m = 0; m < p.mpol; ++m) {
                    for (int n = 0; n < p.ntor + 1; ++n) {
                        int mn = m * (p.ntor + 1) + n;
                        h_cc[jB + mn * p.ns] = ip.rbcc[m][n];
                        h_ss[jB + mn * p.ns] = ip.rbss[m][n];
                        h_zsc[jB + mn * p.ns] = ip.zbsc[m][n];
                        h_zcs[jB + mn * p.ns] = ip.zbcs[m][n];
                        // Fix axis: zero all m>0 modes at j=0 (axis regularity)
                        if (m > 0) {
                            h_cc[0 + mn * p.ns] = 0.0;
                            h_ss[0 + mn * p.ns] = 0.0;
                            h_zsc[0 + mn * p.ns] = 0.0;
                            h_zcs[0 + mn * p.ns] = 0.0;
                            h_lsc[0 + mn * p.ns] = 0.0;
                            h_lcs[0 + mn * p.ns] = 0.0;
                        }
                    }
                }
                checkCuda(cudaMemcpy(st.d_rmncc, h_cc, nb, cudaMemcpyHostToDevice), "set cc");
                checkCuda(cudaMemcpy(st.d_rmnss, h_ss, nb, cudaMemcpyHostToDevice), "set ss");
                checkCuda(cudaMemcpy(st.d_zmnsc, h_zsc, nb, cudaMemcpyHostToDevice), "set zsc");
                checkCuda(cudaMemcpy(st.d_zmncs, h_zcs, nb, cudaMemcpyHostToDevice), "set zcs");
                checkCuda(cudaMemcpy(st.d_lmnsc, h_lsc, nb, cudaMemcpyHostToDevice), "set lsc");
                checkCuda(cudaMemcpy(st.d_lmncs, h_lcs, nb, cudaMemcpyHostToDevice), "set lcs");
                delete[] h_cc; delete[] h_ss; delete[] h_zsc; delete[] h_zcs; delete[] h_lsc; delete[] h_lcs;
            }
            printf("  Fixed LCFS boundary and axis regularity\n");
            checkCuda(cudaMemset(st.d_v_rmncc,0,nb),"vcc"); checkCuda(cudaMemset(st.d_v_rmnss,0,nb),"vss");
            checkCuda(cudaMemset(st.d_v_zmnsc,0,nb),"vzsc"); checkCuda(cudaMemset(st.d_v_zmncs,0,nb),"vzcs");
            checkCuda(cudaMemset(st.d_v_lmnsc,0,nb),"vlsc"); checkCuda(cudaMemset(st.d_v_lmncs,0,nb),"vlcs");

            delete[] h_rmncc; delete[] h_zmnsc; delete[] h_lmnsc;
            delete[] h_rmnss; delete[] h_zmncs; delete[] h_lmncs;
            return;
        }
            fclose(fp);
        }
    }

    auto* c=new double[p.ns*p.mnmax](), *s=new double[p.ns*p.mnmax]();
    auto* zsc=new double[p.ns*p.mnmax](), *zcs=new double[p.ns*p.mnmax]();
    auto* lsc=new double[p.ns*p.mnmax](), *lcs=new double[p.ns*p.mnmax]();

    for(int j=0;j<p.ns;++j){
        double sFlux = j/(p.ns-1.0);          // normalized flux s
        double sqrtS  = sqrt(sFlux);           // sqrt(s)
        for(int m=0;m<p.mpol;++m){
            for(int n=0;n<p.ntor+1;++n){
                int mn = m*(p.ntor+1)+n;
                if(m==0){
                    // m=0: linear in s between axis and boundary
                    c[j+mn*p.ns]   = sFlux*ip.rbcc[0][n] + (1.0-sFlux)*ip.raxis_c[n];
                    zcs[j+mn*p.ns] = sFlux*ip.zbcs[0][n] - (1.0-sFlux)*ip.zaxis_s[n];
                    // rmnss/zmnsc: no m=0 content; lambda: zero initially
                } else if(m==1){
                    // m=1: s^(1/2) radial envelope (s^(m/2)), matching
                    // vmecpp's physical state (interpFromBoundaryAndAxis).
                    // NOTE: the real-space odd-parity values then carry the
                    // 1/max(sqrt(s),sqrt(1/(ns-1))) decomposition factor
                    // (applied in the inverse DFT), so the real-space m=1
                    // contribution is constant across the interior — matching
                    // vmecpp's decomposed real space (its real-space odd =
                    // physical/max).
                    double w = sqrtS;  // s^(1/2)
                    c[j+mn*p.ns]   = w * ip.rbcc[m][n];
                    s[j+mn*p.ns]   = w * ip.rbss[m][n];
                    zsc[j+mn*p.ns] = w * ip.zbsc[m][n];
                    zcs[j+mn*p.ns] = w * ip.zbcs[m][n];
                } else {
                    // m>=2: s^(m/2) radial envelope, vanishing at axis
                    double w = pow(sqrtS, m);  // s^(m/2)
                    c[j+mn*p.ns]   = w * ip.rbcc[m][n];
                    s[j+mn*p.ns]   = w * ip.rbss[m][n];
                    zsc[j+mn*p.ns] = w * ip.zbsc[m][n];
                    zcs[j+mn*p.ns] = w * ip.zbcs[m][n];
                }
                // lmnsc/lmncs: zero initially (lambda is a free gauge)
            }
        }
    }
    printf("  initState: vmecpp interpFromBoundaryAndAxis (m>0 s^(m/2))\n");

    checkCuda(cudaMalloc(&st.d_rmncc,nb),"cc"); checkCuda(cudaMalloc(&st.d_rmnss,nb),"ss");
    checkCuda(cudaMalloc(&st.d_zmnsc,nb),"zsc"); checkCuda(cudaMalloc(&st.d_zmncs,nb),"zcs");
    checkCuda(cudaMalloc(&st.d_lmnsc,nb),"lsc"); checkCuda(cudaMalloc(&st.d_lmncs,nb),"lcs");
    checkCuda(cudaMalloc(&st.d_v_rmncc,nb),"vcc"); checkCuda(cudaMalloc(&st.d_v_rmnss,nb),"vss");
    checkCuda(cudaMalloc(&st.d_v_zmnsc,nb),"vzsc"); checkCuda(cudaMalloc(&st.d_v_zmncs,nb),"vzcs");
    checkCuda(cudaMalloc(&st.d_v_lmnsc,nb),"vlsc"); checkCuda(cudaMalloc(&st.d_v_lmncs,nb),"vlcs");

    checkCuda(cudaMemcpy(st.d_rmncc,c,nb,cudaMemcpyHostToDevice),"cpy cc");
    checkCuda(cudaMemcpy(st.d_rmnss,s,nb,cudaMemcpyHostToDevice),"cpy ss");
    checkCuda(cudaMemcpy(st.d_zmnsc,zsc,nb,cudaMemcpyHostToDevice),"cpy zsc");
    checkCuda(cudaMemcpy(st.d_zmncs,zcs,nb,cudaMemcpyHostToDevice),"cpy zcs");
    checkCuda(cudaMemcpy(st.d_lmnsc,lsc,nb,cudaMemcpyHostToDevice),"cpy lsc");
    checkCuda(cudaMemcpy(st.d_lmncs,lcs,nb,cudaMemcpyHostToDevice),"cpy lcs");
    checkCuda(cudaMemset(st.d_v_rmncc,0,nb),"vcc"); checkCuda(cudaMemset(st.d_v_rmnss,0,nb),"vss");
    checkCuda(cudaMemset(st.d_v_zmnsc,0,nb),"vzsc"); checkCuda(cudaMemset(st.d_v_zmncs,0,nb),"vzcs");
    checkCuda(cudaMemset(st.d_v_lmnsc,0,nb),"vlsc"); checkCuda(cudaMemset(st.d_v_lmncs,0,nb),"vlcs");

    delete[] c; delete[] s; delete[] zsc; delete[] zcs; delete[] lsc; delete[] lcs;
}

static void freeState(SpectralState& st) {
    cudaFree(st.d_rmncc); cudaFree(st.d_rmnss); cudaFree(st.d_zmnsc);
    cudaFree(st.d_zmncs); cudaFree(st.d_lmnsc); cudaFree(st.d_lmncs);
    cudaFree(st.d_v_rmncc); cudaFree(st.d_v_rmnss); cudaFree(st.d_v_zmnsc);
    cudaFree(st.d_v_zmncs); cudaFree(st.d_v_lmnsc); cudaFree(st.d_v_lmncs);
}

int main() {
    InputParams ip = initInputParams();
    GridParams p=initParams(ip);
    printf("=== cuMES — CUDA Magnetic Equilibrium Solver ===\n");
    fflush(stdout);
    printf("input: %s\n", p.ncurr == 0 ? "solovev" : "w7x");
    printf("ns=%d mnmax=%d ntheta=%d nzeta=%d nZnT=%d nfp=%d mpol=%d ntor=%d ncurr=%d\n",
           p.ns,p.mnmax,p.ntheta,p.nzeta,p.nZnT,p.nfp,p.mpol,p.ntor,p.ncurr);

    cublasHandle_t cublasHandle; checkCublas(cublasCreate(&cublasHandle),"cublas");

    SpectralState st{}; initState(st,p,ip);
    RadialProfiles rp=profilesCreate(p,ip);
    FourierPlan fp=fourierCreate(p,cublasHandle);
    MetricWorkspace mw=metricCreate(p);

    SolverResult result=solverRun(st,p,rp,fp,mw);
    outputSaveBinary(st, p, "cumes_state.bin");
    outputPrint(st,p,result.iterations,result.converged,result.fsqr,result.fsqz,result.fsql);

    fourierFree(fp); metricFree(mw); profilesFree(rp); freeState(st);
    cublasDestroy(cublasHandle);
    printf("\nDone.\n"); return result.converged?0:1;
}
