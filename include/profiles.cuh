// profiles.cuh — allocate and initialise radial profiles on GPU.
#pragma once
#include "vmec_types.h"

// Evaluate profiles on CPU, upload to device.
// Caller owns the returned struct; call profilesFree() to release.
RadialProfiles profilesCreate(const GridParams& p);
void profilesFree(RadialProfiles& rp);
