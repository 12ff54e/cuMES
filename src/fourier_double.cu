#include "fourier_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template cumes::DeviceModeTable cumes::modeTableCreate<double>(
    const DeviceParams<double>&,
    cumes::DeviceArena*);
template cumes::RealSpaceStorage<double> realSpaceCreate<double>(
    const DeviceParams<double>&,
    cumes::DeviceArena*);
template void realSpaceFree<double>(cumes::RealSpaceStorage<double>&);

// ToroidalFftOperator (owns the cuFFT plans + transform scratch; a
// SpectralOperator backend).
template class cumes::ToroidalFftOperator<double>;
