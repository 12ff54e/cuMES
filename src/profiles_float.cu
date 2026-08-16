#include "profiles_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template RadialProfiles<float>  profilesCreate<float>(DeviceParams<float>&, const cumes::ValidatedProblem&, cumes::DeviceArena*);
template void profilesFree<float>(RadialProfiles<float>&);
