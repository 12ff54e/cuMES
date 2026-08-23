// fft_traits.h — precision dispatch for the batched ζ-cuFFT plans.
// The transform/constraint plan and exec call sites are
// templated on the scalar type T; this trait maps T onto the cuFFT types,
// plan enums and exec functions (double -> D2Z/Z2D, float -> R2C/C2R).
// The plan strides (inembed/idist/onembed/odist, in elements) are identical
// across precisions — only the type enum and exec entry point change.
#ifndef CUMES_INCLUDE_FFT_TRAITS_H_
#define CUMES_INCLUDE_FFT_TRAITS_H_
#include <cufft.h>

template <typename T>
struct FftTraits;

template <>
struct FftTraits<double> {
    using val_type = double;
    using Complex = cufftDoubleComplex;              // double2
    static constexpr cufftType FORWARD = CUFFT_D2Z;  // real -> half-spectrum
    static constexpr cufftType INVERSE = CUFFT_Z2D;  // half-spectrum -> real
    static cufftResult exec_forward(cufftHandle p, double* in, Complex* out) {
        return cufftExecD2Z(p, in, out);
    }
    static cufftResult exec_inverse(cufftHandle p, Complex* in, double* out) {
        return cufftExecZ2D(p, in, out);
    }
};

template <>
struct FftTraits<float> {
    using val_type = float;
    using Complex = cufftComplex;  // float2
    static constexpr cufftType FORWARD = CUFFT_R2C;
    static constexpr cufftType INVERSE = CUFFT_C2R;
    static cufftResult exec_forward(cufftHandle p, float* in, Complex* out) {
        return cufftExecR2C(p, in, out);
    }
    static cufftResult exec_inverse(cufftHandle p, Complex* in, float* out) {
        return cufftExecC2R(p, in, out);
    }
};

#endif  // CUMES_INCLUDE_FFT_TRAITS_H_
