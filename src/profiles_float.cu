#include "profiles_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template RadialProfiles<float>  profilesCreate<float>(GridParams<float>&, const InputParams&, cumes::DeviceArena*);
template void profilesFree<float>(RadialProfiles<float>&);
