// profiles.hpp — radial profile boundary (blueprint §6.4, §6.7).
//
// Profiles are immutable prescribed data evaluated on the HOST and uploaded
// once per stage (they do not change across iterations), so unlike the
// enqueue-only device operators they are a host-side build step. The operator
// OWNS the 11 full/half-grid radial arrays directly (carved from the stage
// arena) and exposes them as a typed RadialProfileViews bundle plus the radial
// grid spacing. (Migration step 13.3: the legacy RadialProfiles struct +
// profilesCreate/profilesFree are gone.)
#ifndef CUMES_INCLUDE_CUMES_PHYSICS_PROFILES_HPP_
#define CUMES_INCLUDE_CUMES_PHYSICS_PROFILES_HPP_

#include "cumes/config/device_params.hpp"
#include "cumes/state/real_fields.cuh"

namespace cumes {

class DeviceArena;
class ValidatedProblem;

template <class T>
class Profiles {
   public:
    Profiles(DeviceParams<T>& p,
             const ValidatedProblem& vp,
             DeviceArena* arena);
    ~Profiles();

    // Non-movable: the destructor frees the owned radial arrays without
    // nulling, so a defaulted move would double-free (review finding 3.2).
    Profiles(const Profiles&) = delete;
    Profiles& operator=(const Profiles&) = delete;
    Profiles(Profiles&&) noexcept = delete;
    Profiles& operator=(Profiles&&) noexcept = delete;

    // Typed radial-profile view bundle (the 11 half/full-grid arrays) — the
    // solver consumes this instead of naming raw `d_*` pointers.
    RadialProfileViews<T> profile_views() const {
        RadialProfileViews<T> v;
        v.iota_F = d_iota_F_;
        v.phip_F = d_phip_F_;
        v.chi_F = d_chi_F_;
        v.sqrtS_F = d_sqrtS_F_;
        v.iota_H = d_iota_H_;
        v.pres_H = d_pres_H_;
        v.phip_H = d_phip_H_;
        v.dVds_H = d_dVds_H_;
        v.sqrtS_H = d_sqrtS_H_;
        v.curr_H = d_curr_H_;
        v.chip_H = d_chip_H_;
        return v;
    }

    // The radial grid spacing (normalised flux coordinate; = 1/(ns-1)).
    T delta_s() const { return delta_s_; }

   private:
    T* d_iota_F_ = nullptr;
    T* d_phip_F_ = nullptr;
    T* d_chi_F_ = nullptr;
    T* d_sqrtS_F_ = nullptr;
    T* d_iota_H_ = nullptr;
    T* d_pres_H_ = nullptr;
    T* d_phip_H_ = nullptr;
    T* d_dVds_H_ = nullptr;
    T* d_sqrtS_H_ = nullptr;
    T* d_curr_H_ = nullptr;
    T* d_chip_H_ = nullptr;
    T delta_s_ = T(0);
    bool arena_backed_ = false;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_PHYSICS_PROFILES_HPP_
