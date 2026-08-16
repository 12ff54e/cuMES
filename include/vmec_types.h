// vmec_types.h — common data structures.
// Parity convention (matches vmecpp):
//   Even m -> "e" arrays, odd m -> "o" arrays.
//   Each parity array receives the FULL contribution from its modes.
//
// Internal (folded, n>=0) product basis, matching vmecpp's internal "fc"
// representation (FourierToReal3DSymmFastPoloidal):
//   R = rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ)
//   Z = zmnsc*sin(mθ)cos(nζ) + zmncs*cos(mθ)sin(nζ)
//   λ = lmnsc*sin(mθ)cos(nζ) + lmncs*cos(mθ)sin(nζ)
// Mode index: mode = m*(ntor+1) + n with m = 0..mpol-1, n = 0..ntor
// (mnmax = mpol*(ntor+1)); the physical cos(mθ-nζ) coefficients fold as
//   rmncc[m,n] = rmnc[n]+rmnc[-n],  rmnss[m,n] = rmnc[n]-rmnc[-n],
//   zmnsc[m,n] = zmns[n]+zmns[-n],  zmncs[m,n] = zmns[-n]-zmns[n].
// The toroidal derivative of λ is stored as -∂λ/∂ζ (vmecpp convention:
// lv = -(lmksc_n*sinmu + lmkcs_n*cosmu) with sinnvn=-n*nfp*sin(nζ),
// cosnvn=+n*nfp*cos(nζ)); bsupu = (lamscale*lv + chip')/√g.
#pragma once

#include "cumes/config/device_params.hpp"  // DeviceParams<T> (the per-stage pack)

// App-level precision switch. All modules are templated on T; this alias is
// what main.cu (and the diagnostics) build with. Configure via
//   cmake -B build-float -DCUMES_USE_FLOAT=ON
// The on-disk state files (cumes_state.bin, vmecpp_init.bin) stay double
// regardless; only the GPU computation uses T.
#ifdef CUMES_USE_FLOAT
using Real = float;
#else
using Real = double;
#endif
