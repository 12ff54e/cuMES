// profiles.cuh — allocate and initialise radial profiles on GPU.
#pragma once
#include "vmec_types.h"
#include "input.h"

// Evaluate profiles on CPU from the input parameters, upload to device.
// Fills p.lamscale (sqrt(deltaS * sum phipH^2)).
// Caller owns the returned struct; call profilesFree() to release.
// InputParams stays double (host config); T conversion happens here.
template <typename T>
RadialProfiles<T> profilesCreate(GridParams<T>& p, const InputParams& ip);
template <typename T>
void profilesFree(RadialProfiles<T>& rp);
