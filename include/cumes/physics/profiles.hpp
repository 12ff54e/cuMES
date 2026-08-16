// profiles.hpp — radial profile boundary (blueprint §6.4, §6.7).
//
// Profiles are immutable prescribed data evaluated on the HOST and uploaded once
// per stage (they do not change across iterations), so unlike the enqueue-only
// device operators they are a host-side build step. The legacy profilesCreate
// (src/profiles_impl.cuh) is the reference implementation; this operator owns
// the RadialProfiles workspace it builds.
#pragma once

#include "cumes/state/real_fields.cuh"
#include "input.h"
#include "profiles.cuh"

namespace cumes {

class DeviceArena;

template <class T>
class Profiles {
 public:
  Profiles(GridParams<T>& p, const InputParams& ip, DeviceArena* arena)
      : rp_(profilesCreate(p, ip, arena)) {}
  ~Profiles() { profilesFree(rp_); }

  Profiles(const Profiles&) = delete;
  Profiles& operator=(const Profiles&) = delete;
  Profiles(Profiles&&) noexcept = default;
  Profiles& operator=(Profiles&&) noexcept = default;

  const RadialProfiles<T>& workspace() const { return rp_; }

  // Typed radial-profile view bundle (the 11 half/full-grid arrays) — the
  // solver consumes this instead of naming RadialProfiles' raw `d_*` pointers.
  RadialProfileViews<T> profile_views() const {
    RadialProfileViews<T> v;
    v.iota_F = rp_.d_iota_F; v.phip_F = rp_.d_phip_F;
    v.chi_F = rp_.d_chi_F; v.sqrtS_F = rp_.d_sqrtS_F;
    v.iota_H = rp_.d_iota_H; v.pres_H = rp_.d_pres_H; v.phip_H = rp_.d_phip_H;
    v.dVds_H = rp_.d_dVds_H; v.sqrtS_H = rp_.d_sqrtS_H;
    v.curr_H = rp_.d_curr_H; v.chip_H = rp_.d_chip_H;
    return v;
  }

  // The radial grid spacing (normalised flux coordinate; = 1/(ns-1)).
  T delta_s() const { return rp_.delta_s; }

 private:
  RadialProfiles<T> rp_;
};

}  // namespace cumes
