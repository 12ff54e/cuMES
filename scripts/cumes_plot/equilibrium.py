"""Equilibrium reconstruction and geometry calculations."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


SIGN_J = -1.0
MU0 = 4.0e-7 * np.pi


def mode_tables(mnmax, ntor):
    m = np.arange(mnmax) // (ntor + 1)
    n = np.arange(mnmax) % (ntor + 1)
    return m, n


def pack_series(a, b, series, mm, nn, nth, nzt):
    """Pack one family-parity's modes into a complex 2-D spectrum C[p, q]
    over the full-period grid (theta_j = 2*pi*j/nth, zeta_k = 2*pi*k/nzt).
    series='R': a*cos(mθ)cos(nζ) + b*sin(mθ)sin(nζ) decomposes as
        1/4[(a-b)(e^{+i(mθ+nζ)} + e^{-i(mθ+nζ)}) + (a+b)(e^{+i(mθ-nζ)} + e^{-i(mθ-nζ)})]
    series='Z': a*sin(mθ)cos(nζ) + b*cos(mθ)sin(nζ) decomposes as
        1/4[(a+b)(-i e^{+i(mθ+nζ)} + i e^{-i(mθ+nζ)}) + (a-b)(-i e^{+i(mθ-nζ)} + i e^{-i(mθ-nζ)})]
    (λ shares the Z form). At m=0/n=0 the ±mirror entries collide, the sin
    terms cancel and the cos terms double — exactly the right result — so
    no special cases are needed."""
    C = np.zeros((nth, nzt), dtype=np.complex128)
    rows = np.concatenate([mm, -mm, mm, -mm]) % nth
    cols = np.concatenate([nn, -nn, -nn, nn]) % nzt
    if series == "R":
        vals = 0.25 * np.concatenate([a - b, a - b, a + b, a + b])
    else:
        vals = 0.25j * np.concatenate([-(a + b), a + b, -(a - b), a - b])
    np.add.at(C, (rows, cols), vals)
    return C


def eval_state(fams, ns, j, th, zt, ntor, nfp):
    """Stored parity arrays at full-grid surface j on a uniform full-period
    (th, zt) grid — the exact inverseDFT convention (kernels/fourier_impl.cuh),
    synthesized with a batched 2-D IFFT instead of a direct mode sum:

      R = rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ)
      Z = zmnsc*sin(mθ)cos(nζ) + zmncs*cos(mθ)sin(nζ)
      λ = lmnsc*sin(mθ)cos(nζ) + lmncs*cos(mθ)sin(nζ)

    even m -> e arrays (plain), odd m -> o arrays divided by
    maxsc = max(sqrt(s_j), sqrt(ds)) (vmecpp's scalxc decomposition).
    Derivatives are spectral differentiation of the packed spectrum:
    ×i·m for ∂/∂θ, ×i·n·nfp for ∂/∂ζ (the solver's toroidal derivative
    tables carry the nfp factor). The λ ζ-derivative slots store -dλ/dζ
    (signV = -1).
    """
    m, n = mode_tables(fams["rmncc"].shape[0] // ns, ntor)
    nth, nzt = th.size, zt.size
    assert th.ndim == 1 and zt.ndim == 1, "eval_state takes 1-D angle axes"
    assert np.allclose(th, 2.0 * np.pi * np.arange(nth) / nth) and \
        np.allclose(zt, 2.0 * np.pi * np.arange(nzt) / nzt), \
        "eval_state needs uniform full-period grids"
    par_odd = (m % 2 == 1)
    maxsc = max(np.sqrt(j / (ns - 1)), np.sqrt(1.0 / (ns - 1)))
    fac = np.where(par_odd, 1.0 / maxsc, 1.0)

    def fam(name):
        mode = m * (ntor + 1) + n     # folded mode index
        return fams[name][mode * ns + j]  # mode-major, surface j column

    rc, rs = fam("rmncc") * fac, fam("rmnss") * fac
    zs, zc = fam("zmnsc") * fac, fam("zmncs") * fac
    lsc, lcs = fam("lmnsc") * fac, fam("lmncs") * fac
    ev, od = ~par_odd, par_odd
    C = np.zeros((6, nth, nzt), dtype=np.complex128)
    C[0] = pack_series(rc[ev], rs[ev], "R", m[ev], n[ev], nth, nzt)
    C[1] = pack_series(rc[od], rs[od], "R", m[od], n[od], nth, nzt)
    C[2] = pack_series(zs[ev], zc[ev], "Z", m[ev], n[ev], nth, nzt)
    C[3] = pack_series(zs[od], zc[od], "Z", m[od], n[od], nth, nzt)
    C[4] = pack_series(lsc[ev], lcs[ev], "Z", m[ev], n[ev], nth, nzt)
    C[5] = pack_series(lsc[od], lcs[od], "Z", m[od], n[od], nth, nzt)
    # derivative spectra: ×i·m for ∂/∂θ, ×i·n·nfp for ∂/∂ζ; the λ
    # ζ-derivative slots (indices 16, 17) are negated (signV = -1).
    dth = 1j * (np.fft.fftfreq(nth) * nth)[:, None]
    dzt = 1j * (np.fft.fftfreq(nzt) * nzt * nfp)[None, :]
    S = np.concatenate([C, C * dth, C * dzt], axis=0)   # (18, nth, nzt)
    S[16:] *= -1.0
    G = np.fft.ifft2(S).real * (nth * nzt)
    keys = ("re", "ro", "ze", "zo", "le", "lo",
            "rue", "ruo", "zue", "zuo", "lue", "luo",
            "rve", "rvo", "zve", "zvo", "lve", "lvo")
    out = {"th": th, "zt": zt}
    for k, g in zip(keys, G):
        out[k] = g
    return out


def half_grid(arrays, ns, jh, chip, phip_avg, lamscale):
    """Mirror of baseGeometryKernel + magneticFieldKernel for half-grid
    surface jh (kernels/geometry_impl.cuh). arrays = the two adjacent full-grid
    surfaces (stored parity values); phip_avg = 0.5·(phipF(jh)+phipF(jh+1))
    and lamscale come from make_profiles. Returns the half-grid geometry
    (r12/z12), the covariant metric, lambda derivatives, and |B| (Tesla —
    the B^u·B_u products are invariant under the angular coordinate
    scaling, so the kernel units are physical)."""
    ds = 1.0 / (ns - 1)
    s_in = np.sqrt(jh / (ns - 1))
    s_out = np.sqrt((jh + 1) / (ns - 1))
    s_h = np.sqrt((jh + 0.5) / (ns - 1))
    r, rp = arrays[0], arrays[1]

    # ---- base geometry (stored parity values, exactly as the kernel) ----
    r12 = 0.5 * ((r["re"] + rp["re"]) + s_h * (r["ro"] + rp["ro"]))
    z12 = 0.5 * ((r["ze"] + rp["ze"]) + s_h * (r["zo"] + rp["zo"]))
    ru12 = 0.5 * ((r["rue"] + rp["rue"]) + s_h * (r["ruo"] + rp["ruo"]))
    zu12 = 0.5 * ((r["zue"] + rp["zue"]) + s_h * (r["zuo"] + rp["zuo"]))
    rs_ = ((rp["re"] - r["re"]) + s_h * (rp["ro"] - r["ro"])) / ds
    zs_ = ((rp["ze"] - r["ze"]) + s_h * (rp["zo"] - r["zo"])) / ds

    tau1 = ru12 * zs_ - rs_ * zu12
    tau2 = (rp["ruo"] * rp["zo"] + r["ruo"] * r["zo"]
            - rp["zuo"] * rp["ro"] - r["zuo"] * r["ro"]
            + (rp["rue"] * rp["zo"] + r["rue"] * r["zo"]
               - rp["zue"] * rp["ro"] - r["zue"] * r["ro"]) / s_h)
    tau = tau1 + 0.25 * tau2
    gsqrt = tau * r12

    sfi2, sfo2 = s_in * s_in, s_out * s_out
    guu = 0.5 * ((r["rue"] ** 2 + r["zue"] ** 2) + (rp["rue"] ** 2 + rp["zue"] ** 2)
                 + sfi2 * (r["ruo"] ** 2 + r["zuo"] ** 2)
                 + sfo2 * (rp["ruo"] ** 2 + rp["zuo"] ** 2)) \
        + s_h * ((r["rue"] * r["ruo"] + r["zue"] * r["zuo"])
                 + (rp["rue"] * rp["ruo"] + rp["zue"] * rp["zuo"]))
    # NOTE: the sH-weighted cross terms sit INSIDE the 0.5, matching vmecpp's
    # ComputeMetricElements exactly (metric_kernel.h) — moving them outside
    # doubles the cross-term weight (caught historically by the iter-1 dump).
    guv = 0.5 * ((r["rue"] * r["rve"] + r["zue"] * r["zve"])
                 + (rp["rue"] * rp["rve"] + rp["zue"] * rp["zve"])
                 + sfi2 * (r["ruo"] * r["rvo"] + r["zuo"] * r["zvo"])
                 + sfo2 * (rp["ruo"] * rp["rvo"] + rp["zuo"] * rp["zvo"])
                 + s_h * ((r["rue"] * rp["rvo"] + r["zue"] * rp["zvo"])
                          + (rp["rue"] * rp["rvo"] + rp["zue"] * rp["zvo"])
                          + (r["rve"] * r["ruo"] + r["zve"] * r["zuo"])
                          + (rp["rve"] * rp["ruo"] + rp["zve"] * rp["zuo"])))
    gvv = 0.5 * (r["re"] ** 2 + rp["re"] ** 2
                 + sfi2 * r["ro"] ** 2 + sfo2 * rp["ro"] ** 2) \
        + s_h * (r["re"] * r["ro"] + rp["re"] * rp["ro"]) \
        + 0.5 * ((r["rve"] ** 2 + r["zve"] ** 2) + (rp["rve"] ** 2 + rp["zve"] ** 2)
                 + sfi2 * (r["rvo"] ** 2 + r["zvo"] ** 2)
                 + sfo2 * (rp["rvo"] ** 2 + rp["zvo"] ** 2)) \
        + s_h * ((r["rve"] * r["rvo"] + r["zve"] * r["zvo"])
                 + (rp["rve"] * rp["rvo"] + rp["zve"] * rp["zvo"]))

    # ---- magnetic field (magneticFieldKernel) ----
    lu_h = 0.5 * ((r["lue"] + rp["lue"]) + s_h * (r["luo"] + rp["luo"]))
    lv_h = 0.5 * ((r["lve"] + rp["lve"]) + s_h * (r["lvo"] + rp["lvo"]))
    inv = np.zeros_like(gsqrt)
    np.divide(1.0, gsqrt, out=inv,
              where=np.isfinite(gsqrt) & (np.abs(gsqrt) > 1e-30))
    bsupv = (lamscale * lu_h + phip_avg) * inv
    bsupu = lamscale * lv_h * inv + chip * inv             # chi' from the solve
    bsubu = guu * bsupu + guv * bsupv
    bsubv = guv * bsupu + gvv * bsupv
    bsq = bsupu * bsubu + bsupv * bsubv
    return {"r12": r12, "z12": z12, "guu": guu, "guv": guv, "gvv": gvv,
            "gsqrt": gsqrt, "lu_h": lu_h, "lv_h": lv_h, "bsupu": bsupu,
            "bsupv": bsupv, "bmag": np.sqrt(np.maximum(bsq, 0.0))}


def make_profiles(cfg, ns):
    """Radial profile evaluators exactly as profile_functions.hpp +
    kernels/profiles_impl.cuh: maxToroidalFlux = signJ·phiedge/(2π·torflux(1)),
    phipF(j) = maxTF·torfluxDeriv(Δs·j), lamscale = sqrt(Δs·Σ phipH²),
    iota/curr Horner evaluations with the normX = min(|x·bloat|, 1) clamp,
    and the ncurr=1 normalization Itor = signJ·μ0·curtor/(2π·C_edge). The
    aphi=[1.0] case keeps the historic constant-φ' arithmetic bit-stable.
    Returns a dict with phip_avg(jh), lamscale, chip_iota(jh) (ncurr=0) or
    itor/curr_h(jh) (ncurr=1)."""
    aphi = cfg["aphi"]
    ds = 1.0 / (ns - 1)

    def torflux(x):
        ret = 0.0
        for c in reversed(aphi):
            ret = x * ret + c
        return x * ret

    def torflux_deriv(x):
        return sum((i + 1) * c * x ** i for i, c in enumerate(aphi))

    def iota_eval(x):
        ret = 0.0
        for c in reversed(cfg["ai"]):
            ret = x * ret + c
        return ret

    def curr_eval(x):
        norm = min(abs(x * cfg["bloat"]), 1.0)
        ret = 0.0
        for i in range(len(cfg["ac"]) - 1, -1, -1):
            ret = norm * ret + cfg["ac"][i] / (i + 1)
        return norm * ret

    maxTF = SIGN_J * cfg["phiedge"] / (2.0 * np.pi * torflux(1.0))
    if aphi == [1.0]:
        # Constant φ' (torfluxDeriv ≡ 1): the historic arithmetic, kept
        # bit-stable with the pre-generalization formula.
        phip = maxTF
        lamscale = abs(phip) * np.sqrt((ns - 1) * ds)
        prof = {
            "phip_avg": lambda jh: phip,
            "phip_full": lambda j: phip,
            "lamscale": lamscale,
        }
    else:
        phip_F = np.array([maxTF * torflux_deriv(ds * j) for j in range(ns)])
        phip_H = 0.5 * (phip_F[:-1] + phip_F[1:])
        lamscale = float(np.sqrt(ds * np.sum(phip_H * phip_H)))
        prof = {
            "phip_avg": lambda jh: 0.5 * (phip_F[jh] + phip_F[jh + 1]),
            "phip_full": lambda j: phip_F[j],
            "lamscale": lamscale,
        }
    if cfg["ncurr"] == 0:
        def chip_iota(jh):
            sh = ds * (jh + 0.5)
            tf = min(torflux(sh), 1.0)
            return maxTF * iota_eval(tf) * torflux_deriv(sh)
        prof["chip_iota"] = chip_iota
    else:
        c_edge = curr_eval(1.0)
        prof["itor"] = SIGN_J * MU0 * cfg["curtor"] / (2.0 * np.pi * c_edge)

        def curr_h(jh):
            sh = ds * (jh + 0.5)
            return prof["itor"] * curr_eval(min(torflux(sh), 1.0))
        prof["curr_h"] = curr_h
    return prof


def solve_chip(fams, ns, jh, cfg, prof):
    """chi' for half-grid surface jh. ncurr=1: ncurr1FinalizeKernel —
    chi' = (currH − Σ(guu·B^θ_λ + guv·B^ζ)·w) / Σ(guu/√g·w), summed over the
    reduced-theta trapezoid with dnorm3 = 1/(nzeta·(nThetaRed−1)), currH
    evaluated at the FLUX coordinate sh (the solver's convention). ncurr=0:
    the prescribed-iota profile χ' = maxTF·ι(tf)·torfluxDeriv(sh)
    (kernels/profiles_impl.cuh)."""
    if cfg["ncurr"] == 0:
        return prof["chip_iota"](jh)
    ntheta = cfg["ntheta"]
    nz = cfg["nzeta"]
    ntheta_red = ntheta // 2 + 1
    # eval_state needs a uniform full-period grid; the reduced-theta grid is
    # the first nThetaRed rows of the solver's ntheta grid, so evaluate on
    # the full grid and slice.
    th = 2.0 * np.pi * np.arange(ntheta) / ntheta
    zt = 2.0 * np.pi * np.arange(nz) / nz
    a = [eval_state(fams, ns, j, th, zt, cfg["ntor"], cfg["nfp"])
         for j in (jh, jh + 1)]
    a = [{k: v[:ntheta_red] for k, v in e.items() if k not in ("th", "zt")}
         for e in a]
    h = half_grid(a, ns, jh, 0.0, prof["phip_avg"](jh), prof["lamscale"])
    w = np.full(ntheta_red, 1.0 / (nz * (ntheta_red - 1)))
    w[0] *= 0.5
    w[-1] *= 0.5
    jv = np.sum(w[:, None] * (h["guu"] * h["bsupu"] + h["guv"] * h["bsupv"]))
    one_over = np.zeros_like(h["gsqrt"])
    np.divide(1.0, h["gsqrt"], out=one_over, where=np.abs(h["gsqrt"]) > 1e-30)
    avg = np.sum(w[:, None] * h["guu"] * one_over)
    curr = prof["curr_h"](jh)
    return (curr - jv) / avg if avg != 0.0 else 0.0


def boundary_from_params(params, th, zt):
    """Boundary R/Z from the embedded raw harmonics (signed-n VMEC
    convention: cos(mθ − nζ) / sin(mθ − nζ))."""
    R = np.zeros_like(th)
    Z = np.zeros_like(th)
    for m, n, value in params["rbc"]:
        R += value * np.cos(m * th - n * zt)
    for m, n, value in params["zbs"]:
        Z += value * np.sin(m * th - n * zt)
    return R, Z


def periodic_close(A):
    """Close a 2-D periodic grid: append the first row (theta = 2pi) and the
    first column (zeta = 2pi) so the rendered surface mesh has no seam."""
    A = np.concatenate([A, A[:1, :]], axis=0)
    A = np.concatenate([A, A[:, :1]], axis=1)
    return A


def surface_intensity(X, Y, Z, light_dir, lo=0.55):
    """Lambertian illumination from the GEOMETRIC surface normals (cross
    product of the grid tangents), rescaled to [lo, 1] so the side facing
    away from the light stays bright. Unlike LightSource.hillshade over the
    |B| scalar this produces real 3-D relief: the |B| heightfield is so
    smooth that its normal-intensity range collapses and (with fraction<1)
    the whole map clips to black."""
    gx1, gy1, gz1 = np.gradient(X, axis=0), np.gradient(Y, axis=0), np.gradient(Z, axis=0)
    gx2, gy2, gz2 = np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)
    nx = gy1 * gz2 - gz1 * gy2
    ny = gz1 * gx2 - gx1 * gz2
    nz = gx1 * gy2 - gy1 * gx2
    mag = np.sqrt(nx * nx + ny * ny + nz * nz)
    lam = (nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]) / mag
    lam = (lam - lam.min()) / (lam.max() - lam.min())
    return lo + (1.0 - lo) * lam


def to_cartesian(R, Z, nfp):
    """Replicate one field period into the full torus (phi = zeta/nfp),
    closing both periodic grid directions."""
    nz = R.shape[1]
    zt = np.linspace(0.0, 2.0 * np.pi, nz, endpoint=False)
    Rfull = np.concatenate([R] * nfp, axis=1)
    Zfull = np.concatenate([Z] * nfp, axis=1)
    phi = np.concatenate([(zt + 2.0 * np.pi * k) / nfp for k in range(nfp)])
    Rfull = periodic_close(Rfull)
    Zfull = periodic_close(Zfull)
    phi = np.append(phi, 2.0 * np.pi)
    return Rfull * np.cos(phi[None, :]), Rfull * np.sin(phi[None, :]), Zfull


def converged_axis(fams, ns, ntor, nfp, n=240):
    """The converged magnetic axis: the m=0 content of the innermost
    computed surface (j=1). The axis is theta-independent by construction
    (only m=0 modes enter), so the curve runs through the center of every
    cross-section and cannot leave the plasma:

        R_ax(zeta) = sum_n rmncc(0, n, j=1) cos(n zeta)
        Z_ax(zeta) = sum_n zmncs(0, n, j=1) sin(n zeta)

    with zeta the per-field-period angle, replicated nfp times around the
    full torus (phi = zeta/nfp). The axis carries genuine nfp-fold
    modulation, matching the seed convention in seed_state.hpp
    (raxis_c/zaxis_s interpolate against the same folded toroidal
    mode n)."""
    zt = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    R = np.zeros_like(zt)
    Z = np.zeros_like(zt)
    for nn in range(ntor + 1):
        R += fams["rmncc"][nn * ns + 1] * np.cos(nn * zt)
        Z += fams["zmncs"][nn * ns + 1] * np.sin(nn * zt)
    Rfull = np.concatenate([R] * nfp)
    Zfull = np.concatenate([Z] * nfp)
    phi = np.concatenate([(zt + 2.0 * np.pi * k) / nfp for k in range(nfp)])
    X = Rfull * np.cos(phi)
    Y = Rfull * np.sin(phi)
    return np.append(X, X[0]), np.append(Y, Y[0]), np.append(Zfull, Zfull[0])


def field_lines(edge, th, zt, seeds, nfp, geometry=None,
                zeta_span=6 * 2.0 * np.pi, n_steps=2400):
    """Trace d(theta)/d(zeta) = B^theta/B^zeta on the edge half-grid surface
    (RK4, periodic in both angles), lifted onto the (r12, z12) geometry.
    `edge` is the half_grid() result on the (th, zt) solver grid. When
    `geometry` is supplied, the half-grid field-line direction is lifted onto
    that full-grid surface instead of the half-grid geometry."""
    def periodic_interpolator(theta_grid, zeta_grid, values):
        """Close both uniform angular axes before linear interpolation."""
        values = np.asarray(values)
        expected = (len(theta_grid), len(zeta_grid))
        if values.shape != expected:
            raise ValueError(
                f"angular field shape {values.shape} does not match "
                f"coordinate grid {expected}")
        closed = np.concatenate([values, values[:1]], axis=0)
        closed = np.concatenate([closed, closed[:, :1]], axis=1)
        return RegularGridInterpolator(
            (np.append(theta_grid, 2.0 * np.pi),
             np.append(zeta_grid, 2.0 * np.pi)),
            closed, method="linear", bounds_error=True)

    ratio = edge["bsupu"] / edge["bsupv"]
    ri = periodic_interpolator(th, zt, ratio)
    r_surface = edge["r12"] if geometry is None else geometry["r"]
    z_surface = edge["z12"] if geometry is None else geometry["z"]
    if geometry is None:
        geometry_th, geometry_zt = th, zt
    else:
        geometry_th = np.linspace(
            0.0, 2.0 * np.pi, r_surface.shape[0], endpoint=False)
        geometry_zt = np.linspace(
            0.0, 2.0 * np.pi, r_surface.shape[1], endpoint=False)
    rgi = periodic_interpolator(geometry_th, geometry_zt, r_surface)
    zgi = periodic_interpolator(geometry_th, geometry_zt, z_surface)
    lines = []
    for seed in seeds:
        zeta = np.linspace(0.0, zeta_span, n_steps)
        theta = np.empty_like(zeta)
        theta[0] = seed
        h = zeta[1] - zeta[0]
        for i in range(n_steps - 1):
            th_i, zt_i = theta[i] % (2 * np.pi), zeta[i] % (2 * np.pi)
            k1 = ri([th_i, zt_i])[0]
            k2 = ri([(theta[i] + 0.5 * h * k1) % (2 * np.pi),
                     (zeta[i] + 0.5 * h) % (2 * np.pi)])[0]
            k3 = ri([(theta[i] + 0.5 * h * k2) % (2 * np.pi),
                     (zeta[i] + 0.5 * h) % (2 * np.pi)])[0]
            k4 = ri([(theta[i] + h * k3) % (2 * np.pi),
                     (zeta[i] + h) % (2 * np.pi)])[0]
            theta[i + 1] = theta[i] + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        pts = np.stack([theta % (2 * np.pi), zeta % (2 * np.pi)], axis=1)
        r_line = rgi(pts)
        z_line = zgi(pts)
        phi = zeta / nfp
        lines.append((r_line * np.cos(phi), r_line * np.sin(phi), z_line))
    return lines
