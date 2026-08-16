// fft_traits.h — precision dispatch for the batched ζ-cuFFT plans.
// The transform/constraint plan and exec call sites are
// templated on the scalar type T; this trait maps T onto the cuFFT types,
// plan enums and exec functions (double -> D2Z/Z2D, float -> R2C/C2R).
// The plan strides (inembed/idist/onembed/odist, in elements) are identical
// across precisions — only the type enum and exec entry point change.
#pragma once
#include <cufft.h>

template <typename T> struct FftTraits;

template <> struct FftTraits<double> {
    using Complex = cufftDoubleComplex;                 // double2
    static constexpr cufftType kForward = CUFFT_D2Z;    // real -> half-spectrum
    static constexpr cufftType kInverse = CUFFT_Z2D;    // half-spectrum -> real
    static cufftResult execForward(cufftHandle p, double* in, Complex* out) { return cufftExecD2Z(p, in, out); }
    static cufftResult execInverse(cufftHandle p, Complex* in, double* out)  { return cufftExecZ2D(p, in, out); }
};

template <> struct FftTraits<float> {
    using Complex = cufftComplex;                       // float2
    static constexpr cufftType kForward = CUFFT_R2C;
    static constexpr cufftType kInverse = CUFFT_C2R;
    static cufftResult execForward(cufftHandle p, float* in, Complex* out) { return cufftExecR2C(p, in, out); }
    static cufftResult execInverse(cufftHandle p, Complex* in, float* out) { return cufftExecC2R(p, in, out); }
};
