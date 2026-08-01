// main.cu — entry point: init → solve → output.
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

static GridParams initParams() {
    GridParams p; p.ns=kNsVal; p.mnmax=kMnmax; p.ntheta=kNtheta;
    p.nzeta=kNzeta; p.nfp=kNfp; p.nZnT=kNZnT; p.mpol=kMpol; p.ntor=kNtor;
    return p;
}

static void initState(SpectralState& st, const GridParams& p) {
    // Default: start from the internal initial state (vmecpp's
    // interpFromBoundaryAndAxis, below). Loading an initial state from
    // dump/vmecpp (vmecpp_init.bin, created by scripts/create_vmecpp_init.py)
    // is DISABLED by default so runs are reproducible from the true baseline;
    // opt in with CUMES_LOAD_INIT=1 for the same-state handoff protocol.
    bool loadInit = false;
    if (const char* e = getenv("CUMES_LOAD_INIT")) loadInit = atoi(e) != 0;
    if (loadInit) {
        FILE* fp = fopen("vmecpp_init.bin", "rb");
    if (fp) {
        int ns_file, mnmax_file;
        fread(&ns_file, sizeof(int), 1, fp);
        fread(&mnmax_file, sizeof(int), 1, fp);
        if (ns_file == p.ns && mnmax_file == p.mnmax) {
            printf("Loading initial state from vmecpp_init.bin (ns=%d, mnmax=%d)\n", ns_file, mnmax_file);
            size_t nb = p.ns * p.mnmax * sizeof(double);
            auto* h_rmncc = new double[p.ns * p.mnmax];
            auto* h_zmnsc = new double[p.ns * p.mnmax];
            auto* h_lmnsc = new double[p.ns * p.mnmax];
            auto* h_rmnss = new double[p.ns * p.mnmax];
            auto* h_zmncs = new double[p.ns * p.mnmax];
            fread(h_rmncc, sizeof(double), p.ns * p.mnmax, fp);
            fread(h_zmnsc, sizeof(double), p.ns * p.mnmax, fp);
            fread(h_lmnsc, sizeof(double), p.ns * p.mnmax, fp);
            fread(h_rmnss, sizeof(double), p.ns * p.mnmax, fp);
            fread(h_zmncs, sizeof(double), p.ns * p.mnmax, fp);
            fclose(fp);

            checkCuda(cudaMalloc(&st.d_rmncc,nb),"cc"); checkCuda(cudaMalloc(&st.d_rmnss,nb),"ss");
            checkCuda(cudaMalloc(&st.d_zmnsc,nb),"zsc"); checkCuda(cudaMalloc(&st.d_zmncs,nb),"zcs");
            checkCuda(cudaMalloc(&st.d_lmnsc,nb),"lsc");
            checkCuda(cudaMalloc(&st.d_v_rmncc,nb),"vcc"); checkCuda(cudaMalloc(&st.d_v_rmnss,nb),"vss");
            checkCuda(cudaMalloc(&st.d_v_zmnsc,nb),"vzsc"); checkCuda(cudaMalloc(&st.d_v_zmncs,nb),"vzcs");
            checkCuda(cudaMalloc(&st.d_v_lmnsc,nb),"vlsc");

            checkCuda(cudaMemcpy(st.d_rmncc, h_rmncc, nb, cudaMemcpyHostToDevice), "cpy cc");
            checkCuda(cudaMemcpy(st.d_zmnsc, h_zmnsc, nb, cudaMemcpyHostToDevice), "cpy zsc");
            checkCuda(cudaMemcpy(st.d_lmnsc, h_lmnsc, nb, cudaMemcpyHostToDevice), "cpy lsc");
            checkCuda(cudaMemcpy(st.d_rmnss, h_rmnss, nb, cudaMemcpyHostToDevice), "cpy ss");
            checkCuda(cudaMemcpy(st.d_zmncs, h_zmncs, nb, cudaMemcpyHostToDevice), "cpy zcs");

            // vmecpp stores boundary values separately (not in the spectral state).
            // cuMES embeds the boundary in the spectral coefficients at j=ns-1.
            // Patch the LCFS values to match the boundary from solovev.json.
            // Also zero out m>0 modes at the magnetic axis (j=0) — vmecpp does
            // this via extrapolateTowardsAxis().
            {
                auto* rbc = new double[p.mnmax], *zbs = new double[p.mnmax];
                setSolovevBoundary(rbc, zbs, p.mnmax);
                double *h_cc, *h_ss, *h_zsc, *h_zcs, *h_lsc;
                h_cc = new double[p.ns*p.mnmax];
                h_ss = new double[p.ns*p.mnmax];
                h_zsc = new double[p.ns*p.mnmax];
                h_zcs = new double[p.ns*p.mnmax];
                h_lsc = new double[p.ns*p.mnmax];
                checkCuda(cudaMemcpy(h_cc, st.d_rmncc, nb, cudaMemcpyDeviceToHost), "get cc");
                checkCuda(cudaMemcpy(h_ss, st.d_rmnss, nb, cudaMemcpyDeviceToHost), "get ss");
                checkCuda(cudaMemcpy(h_zsc, st.d_zmnsc, nb, cudaMemcpyDeviceToHost), "get zsc");
                checkCuda(cudaMemcpy(h_zcs, st.d_zmncs, nb, cudaMemcpyDeviceToHost), "get zcs");
                checkCuda(cudaMemcpy(h_lsc, st.d_lmnsc, nb, cudaMemcpyDeviceToHost), "get lsc");
                int jB = p.ns - 1;  // LCFS index
                for (int m = 0; m < p.mnmax; ++m) {
                    // Fix LCFS: set to boundary values
                    h_cc[jB + m * p.ns] = rbc[m];
                    h_ss[jB + m * p.ns] = rbc[m];
                    h_zsc[jB + m * p.ns] = zbs[m];
                    h_zcs[jB + m * p.ns] = zbs[m];
                    // Fix axis: zero all m>0 modes at j=0 (axis regularity)
                    if (m > 0) {
                        h_cc[0 + m * p.ns] = 0.0;
                        h_ss[0 + m * p.ns] = 0.0;
                        h_zsc[0 + m * p.ns] = 0.0;
                        h_zcs[0 + m * p.ns] = 0.0;
                        h_lsc[0 + m * p.ns] = 0.0;
                    }
                }
                checkCuda(cudaMemcpy(st.d_rmncc, h_cc, nb, cudaMemcpyHostToDevice), "set cc");
                checkCuda(cudaMemcpy(st.d_rmnss, h_ss, nb, cudaMemcpyHostToDevice), "set ss");
                checkCuda(cudaMemcpy(st.d_zmnsc, h_zsc, nb, cudaMemcpyHostToDevice), "set zsc");
                checkCuda(cudaMemcpy(st.d_zmncs, h_zcs, nb, cudaMemcpyHostToDevice), "set zcs");
                checkCuda(cudaMemcpy(st.d_lmnsc, h_lsc, nb, cudaMemcpyHostToDevice), "set lsc");
                delete[] h_cc; delete[] h_ss; delete[] h_zsc; delete[] h_zcs; delete[] h_lsc;
                delete[] rbc; delete[] zbs;
            }
            printf("  Fixed LCFS boundary and axis regularity\n");
            checkCuda(cudaMemset(st.d_v_rmncc,0,nb),"vcc"); checkCuda(cudaMemset(st.d_v_rmnss,0,nb),"vss");
            checkCuda(cudaMemset(st.d_v_zmnsc,0,nb),"vzsc"); checkCuda(cudaMemset(st.d_v_zmncs,0,nb),"vzcs");
            checkCuda(cudaMemset(st.d_v_lmnsc,0,nb),"vlsc");

            delete[] h_rmncc; delete[] h_zmnsc; delete[] h_lmnsc;
            delete[] h_rmnss; delete[] h_zmncs;
            return;
        }
            fclose(fp);
        }
    }

    // Fallback: vmecpp's interpFromBoundaryAndAxis algorithm.
    // m=0: linear interpolation between axis and boundary in flux s.
    // m>0: s^(m/2) scaling so higher modes vanish faster near axis.
    // For axisymmetric (n=0): rmnss and zmncs = 0 (sin(nζ)=0).
    // Note: vmecpp divides by mscale=√2 for m>0 in the coefficient,
    // but cuMES's inverse DFT does not apply mscale, so we store
    // the boundary value directly. The resulting real-space geometry
    // is identical to vmecpp's.
    auto* rbc=new double[p.mnmax], *zbs=new double[p.mnmax];
    setSolovevBoundary(rbc,zbs,p.mnmax);
    size_t nb=p.ns*p.mnmax*sizeof(double);
    double raxis = 4.0;  // raxis_c[0] from solovev.json

    auto* c=new double[p.ns*p.mnmax](), *s=new double[p.ns*p.mnmax]();
    auto* zsc=new double[p.ns*p.mnmax](), *zcs=new double[p.ns*p.mnmax]();
    auto* lsc=new double[p.ns*p.mnmax]();

    for(int j=0;j<p.ns;++j){
        double sFlux = j/(p.ns-1.0);          // normalized flux s
        double sqrtS  = sqrt(sFlux);           // sqrt(s)
        for(int m=0;m<p.mnmax;++m){
            if(m==0){
                // m=0: linear in s between axis and boundary
                c[j+m*p.ns]   = sFlux * rbc[m] + (1.0 - sFlux) * raxis;
                zsc[j+m*p.ns] = 0.0;  // Z has no m=0 component
            } else if(m==1){
                // m=1: s^(1/2) radial envelope, matching vmecpp's physical
                // state (interpFromBoundaryAndAxis: m>0 -> s^(m/2)).
                // NOTE: the real-space odd-parity values then carry the
                // 1/max(sqrt(s),sqrt(1/(ns-1))) decomposition factor (applied
                // in the inverse DFT), so the real-space m=1 contribution is
                // constant 1.026 across the interior — matching vmecpp's
                // decomposed real space (its real-space odd = physical/max).
                double w = sqrtS;  // s^(1/2)
                c[j+m*p.ns]   = w * rbc[m];
                zsc[j+m*p.ns] = w * zbs[m];
            } else {
                // m>=2: s^(m/2) radial envelope, vanishing at axis
                double w = pow(sqrtS, m);  // s^(m/2)
                c[j+m*p.ns]   = w * rbc[m];
                zsc[j+m*p.ns] = w * zbs[m];
            }
            s[j+m*p.ns]   = 0.0;  // rmnss=0 (sin(nζ)=0 for n=0)
            zcs[j+m*p.ns] = 0.0;  // zmncs=0
            lsc[j+m*p.ns] = 0.0;  // lambda=0 initially
        }
    }
    printf("  initState: vmecpp interpFromBoundaryAndAxis (m>0 s^(m/2))\n");

    checkCuda(cudaMalloc(&st.d_rmncc,nb),"cc"); checkCuda(cudaMalloc(&st.d_rmnss,nb),"ss");
    checkCuda(cudaMalloc(&st.d_zmnsc,nb),"zsc"); checkCuda(cudaMalloc(&st.d_zmncs,nb),"zcs");
    checkCuda(cudaMalloc(&st.d_lmnsc,nb),"lsc");
    checkCuda(cudaMalloc(&st.d_v_rmncc,nb),"vcc"); checkCuda(cudaMalloc(&st.d_v_rmnss,nb),"vss");
    checkCuda(cudaMalloc(&st.d_v_zmnsc,nb),"vzsc"); checkCuda(cudaMalloc(&st.d_v_zmncs,nb),"vzcs");
    checkCuda(cudaMalloc(&st.d_v_lmnsc,nb),"vlsc");

    checkCuda(cudaMemcpy(st.d_rmncc,c,nb,cudaMemcpyHostToDevice),"cpy cc");
    checkCuda(cudaMemcpy(st.d_rmnss,s,nb,cudaMemcpyHostToDevice),"cpy ss");
    checkCuda(cudaMemcpy(st.d_zmnsc,zsc,nb,cudaMemcpyHostToDevice),"cpy zsc");
    checkCuda(cudaMemcpy(st.d_zmncs,zcs,nb,cudaMemcpyHostToDevice),"cpy zcs");
    checkCuda(cudaMemcpy(st.d_lmnsc,lsc,nb,cudaMemcpyHostToDevice),"cpy lsc");
    checkCuda(cudaMemset(st.d_v_rmncc,0,nb),"vcc"); checkCuda(cudaMemset(st.d_v_rmnss,0,nb),"vss");
    checkCuda(cudaMemset(st.d_v_zmnsc,0,nb),"vzsc"); checkCuda(cudaMemset(st.d_v_zmncs,0,nb),"vzcs");
    checkCuda(cudaMemset(st.d_v_lmnsc,0,nb),"vlsc");

    delete[] rbc; delete[] zbs; delete[] c; delete[] s; delete[] zsc; delete[] zcs; delete[] lsc;
}

static void freeState(SpectralState& st) {
    cudaFree(st.d_rmncc); cudaFree(st.d_rmnss); cudaFree(st.d_zmnsc);
    cudaFree(st.d_zmncs); cudaFree(st.d_lmnsc);
    cudaFree(st.d_v_rmncc); cudaFree(st.d_v_rmnss); cudaFree(st.d_v_zmnsc);
    cudaFree(st.d_v_zmncs); cudaFree(st.d_v_lmnsc);
}

int main() {
    GridParams p=initParams();
    printf("=== cuMES — CUDA Magnetic Equilibrium Solver ===\n");
    fflush(stdout);
    printf("ns=%d mnmax=%d ntheta=%d nzeta=%d nZnT=%d nfp=%d\n",
           p.ns,p.mnmax,p.ntheta,p.nzeta,p.nZnT,p.nfp);

    cublasHandle_t cublasHandle; checkCublas(cublasCreate(&cublasHandle),"cublas");

    SpectralState st{}; initState(st,p);
    RadialProfiles rp=profilesCreate(p);
    FourierPlan fp=fourierCreate(p,cublasHandle);
    MetricWorkspace mw=metricCreate(p);

    SolverResult result=solverRun(st,p,rp,fp,mw);
    outputSaveBinary(st, p, "cumes_state.bin");
    outputPrint(st,p,result.iterations,result.converged,result.fsqr,result.fsqz,result.fsql);

    fourierFree(fp); metricFree(mw); profilesFree(rp); freeState(st);
    cublasDestroy(cublasHandle);
    printf("\nDone.\n"); return result.converged?0:1;
}
