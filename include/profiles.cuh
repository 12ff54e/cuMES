// profiles.cuh — allocate and initialise radial profiles on GPU.
#pragma once
#include "vmec_types.h"
#include "input.h"

// Evaluate profiles on CPU from the input parameters, upload to device.
// Fills p.lamscale (sqrt(deltaS * sum phipH^2)).
// Caller owns the returned struct; call profilesFree() to release.
RadialProfiles profilesCreate(GridParams& p, const InputParams& ip);
void profilesFree(RadialProfiles& rp);
