#include "fourier_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template FourierPlan<float>  fourierCreate<float>(const GridParams<float>&, cumes::DeviceArena*);
template void fourierFree<float>(FourierPlan<float>&);
template void inverseDFT<float>(const FourierPlan<float>&, const SpectralState<float>&, const GridParams<float>&, bool);
template void forwardDFT<float>(const FourierPlan<float>&, float*, const GridParams<float>&, const ConstraintWorkspace<float>&);
template void fourierCombineParity<float>(const FourierPlan<float>&, const GridParams<float>&);
