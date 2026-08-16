// profiles.cuh — allocate and initialise radial profiles on GPU.
#pragma once
#include "vmec_types.h"

namespace cumes { class DeviceArena; class ValidatedProblem; }

// Evaluate profiles on CPU from the validated problem, upload to device.
// Fills p.lamscale (sqrt(deltaS * sum phipH^2)).
// Caller owns the returned struct; call profilesFree() to release.
// The ValidatedProblem coefficients stay double (host config); T conversion
// happens here. With `arena == nullptr` each radial array is its own
// cudaMalloc (legacy); with an arena the 11 arrays are named subspans of one
// stage allocation.
template <typename T>
RadialProfiles<T> profilesCreate(DeviceParams<T>& p, const cumes::ValidatedProblem& vp,
                                 cumes::DeviceArena* arena = nullptr);
template <typename T>
void profilesFree(RadialProfiles<T>& rp);
