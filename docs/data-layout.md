# cuMES data layout

Status date: 2026-08-16 (Phase 10). These are the storage contracts the kernels
and the typed views enforce. They are tested by `test_fourier`, the operator
view tests, and the golden I/O tests, and are normative for any migration
(blueprint §4.1–§4.2).

## 1. Grid and indexing

Normalized flux coordinate on the full grid (`j = 0..ns-1`) and half grid
(`j = 0..ns-2`):

```
s_j         = j / (ns-1)
s_{j+1/2}   = (j+1/2) / (ns-1)
Δs          = 1 / (ns-1)
```

`mode = m * (ntor+1) + n`, `m = 0..mpol-1`, `n = 0..ntor` (folded `n >= 0`
basis), `mnmax = mpol * (ntor+1)`. The physical toroidal mode is `N = n * nfp`
(the ζ grid covers one field period). Mode 0 is the (0,0) DC mode.

## 2. Spectral state (component-major, surface contiguous)

Six families in a fixed order — `Rcc, Zsc, Lsc, Rss, Zcs, Lcs` — each laid out
`[mode][surface]` with surface contiguous:

```
offset(component, mode, surface) = component * (mnmax * ns) + mode * ns + surface
```

The order is **Rcc, Zsc, Lsc** (the `m`-parity-even "cos/cos, sin/cos,
sin/cos" families) then **Rss, Zcs, Lcs** (the odd "sin/sin, cos/sin,
cos/sin" families). This is the `SpectralComponent` enum order and the order of
the six pointers in the legacy `SpectralState`.

The physical amplitude convention: the state stores plain physical Fourier
amplitudes; the forward quadrature projects onto the orthonormal basis with
`mscale`/`nscale` (`√2` for `m>0` / `n>0`), and the descent step re-applies
`S_mn = mscale * nscale`.

## 3. Real-space fields (parity-split, point contiguous)

Real space is split by **m parity**, not by trig factor (blueprint §4.2):

- even `m` → `*_e` arrays, odd `m` → `*_o` arrays;
- each parity array carries the full mode contribution.

Full-grid real layout is `[surface][zeta][theta]`, theta (point) contiguous:

```
index(point, surface) = point + surface * nZnT,   nZnT = ntheta * nzeta
```

Odd-`m` work values are regularized by `1 / max(√s, √Δs)`; the odd physical
values carry `1/scalxc` so the product is the stored value.

Half-grid arrays (`√g`, metric, `B`, `P_tot`) are `[half_surface][zeta][theta]`
with `ns-1` surfaces (`half_points = (ns-1) * nZnT`).

## 4. Force residuals (parity-split, point contiguous)

The 16 force arrays `armn/azmn/brmn/bzmn/crmn/czmn/blmn/clmn` × `e/o` are full
real-space fields in the same `[surface][zeta][theta]` point-contiguous layout.
After `forwardDFT` they become six spectral families in the §2 component-major
layout. The decomposed residuals and velocities are a **different domain** from
the physical state and are typed `DecomposedResidualDomain` /
`DecomposedVelocityDomain` views so a kernel cannot mix them.

## 5. Reduced-theta quadrature (forward only)

The forward projection uses `nThetaRed = ntheta/2 + 1` points over `θ ∈ [0, π]`
with trapezoid weight

```
w = mscale * nscale / (nzeta * (nThetaRed - 1)) * e_k,
    e_k = 1/2 at a θ endpoint, 1 otherwise.
```

This is a distinct typed view (`QuadraturePlan`), never an integer
reinterpretation of the full-grid view.

## 6. Legacy binary v0 contract

The on-disk `cumes_state.bin` (little-endian) is

```
int32 ns
int32 mnmax
double rmncc[mode][surface]
double zmnsc[mode][surface]
double lmnsc[mode][surface]
double rmnss[mode][surface]
double zmncs[mode][surface]
double lmncs[mode][surface]
```

each family `mnmax * ns` doubles, surface contiguous. The axis row (`j=0`) is
the constant-extrapolated row, which the comparison scripts intentionally skip.
NetCDF/HDF5 v0 maps a logical value `family[surface, mode]` to device offset
`surface + mode * ns` — golden tests pin this against a declaration/order
transpose.
