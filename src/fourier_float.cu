#include "fourier_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template FourierPlan<float>  fourierCreate<float>(const GridParams<float>&, cumes::DeviceArena*);
template void fourierFree<float>(FourierPlan<float>&);
template void inverseDFT<float>(const FourierPlan<float>&, cumes::SpectralView<const float, cumes::PhysicalStateDomain>, const GridParams<float>&, bool, cudaStream_t);
template void inverseDFTFused<float>(const FourierPlan<float>&, cumes::SpectralView<const float, cumes::PhysicalStateDomain>, const GridParams<float>&, bool, float*, float*, cudaStream_t);
template void forwardDFT<float>(const FourierPlan<float>&, cumes::SpectralView<float, cumes::DecomposedResidualDomain>, const GridParams<float>&, const ConstraintWorkspace<float>&, cudaStream_t);
template void fourierCombineParity<float>(const FourierPlan<float>&, const GridParams<float>&, cudaStream_t);
