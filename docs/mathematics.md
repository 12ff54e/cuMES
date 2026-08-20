# cuMES mathematics — normative numerical contracts

Extracted verbatim from `docs/cuda-overhaul-blueprint.md` §4 (the overhaul
plan's normative specification; the blueprint keeps a pointer back here).
Sections 1–11 below correspond one-to-one to the blueprint's §4.1–§4.11. The
layout contracts derived from §§1–2 live in `docs/data-layout.md`; the
implementation architecture in `docs/architecture.md`.

Everything here is contract: a change that alters any expression below is at
least Class B and needs a differential proof (see `docs/verification.md` §6
for the Class A/B/C gates).


The redesign may change storage, scheduling, and kernel decomposition, but these contracts must be explicit and covered by component tests.

## 1. Coordinates, grids, and indexing

Use normalized flux coordinate

\[
s_j = \frac{j}{n_s-1},\qquad
s_{j+1/2} = \frac{j+1/2}{n_s-1},\qquad
\Delta s = \frac{1}{n_s-1}.
\]

Keep these logical layouts:

- spectral: `[component][mode][surface]`, with surface contiguous;
- full-grid real: `[surface][zeta][theta]`, with theta/point contiguous;
- half-grid real: `[half_surface][zeta][theta]`;
- compact reduced theta quadrature: a distinct typed view, never an integer reinterpretation of a full-grid view.

All element-count products must use checked `size_t` arithmetic. `GridShape::validate()` must verify radial minima, even/reduced theta requirements, cuFFT-compatible zeta sizes, mode ranges, quadrature coverage, launch limits, and the selected backend's constraints before any allocation.

## 2. Folded Fourier representation

For the stored nonnegative toroidal index `n`, define the physical toroidal mode

\[
N = n\,n\_{\mathrm{fp}}.
\]

The stellarator-symmetric product basis is

\[
\begin{aligned}
R(s,\theta,\zeta) &= \sum_{m,n}
\left[R^{cc}_{mn}(s)\cos(m\theta)\cos(n\zeta)
+R^{ss}_{mn}(s)\sin(m\theta)\sin(n\zeta)\right],\\
Z(s,\theta,\zeta) &= \sum_{m,n}
\left[Z^{sc}_{mn}(s)\sin(m\theta)\cos(n\zeta)
+Z^{cs}_{mn}(s)\cos(m\theta)\sin(n\zeta)\right],\\
\lambda(s,\theta,\zeta) &= \sum_{m,n}
\left[L^{sc}_{mn}(s)\sin(m\theta)\cos(n\zeta)
+L^{cs}\_{mn}(s)\cos(m\theta)\sin(n\zeta)\right].
\end{aligned}
\]

For raw signed-`n` boundary harmonics and `n>0`, folding into this product basis is

\[
\begin{aligned}
R^{cc}_{m,n}&=rbc(m,+n)+rbc(m,-n),&
R^{ss}_{m,n}&=rbc(m,+n)-rbc(m,-n)\quad(m>0),\\
Z^{sc}_{m,n}&=zbs(m,+n)+zbs(m,-n)\quad(m>0),&
Z^{cs}_{m,n}&=zbs(m,-n)-zbs(m,+n).
\end{aligned}
\]

At `n=0`, `rbc` is accumulated once into `Rcc`, and `zbs` contributes to `Zsc` only for `m>0`; sine-in-zeta families vanish and `zbs(0,0)` has no basis function. Duplicate raw entries are summed deliberately. Folding validation precedes any fixed/device packing.

Poloidal derivatives multiply by `m`; physical toroidal derivatives multiply by `N`, not the raw stored `n`. The inverse DFT computes derivatives analytically,

\[
\partial_\theta\cos(m\theta-n\zeta)=-m\sin(m\theta-n\zeta),\qquad
\partial_\theta\sin(m\theta-n\zeta)=+m\cos(m\theta-n\zeta),
\]

\[
\partial_\zeta\cos(m\theta-n\zeta)=+n\sin(m\theta-n\zeta),\qquad
\partial_\zeta\sin(m\theta-n\zeta)=-n\cos(m\theta-n\zeta),
\]

in "index space" (not physical radians); the Jacobian and metric are consistent
with this convention, so only the absolute scaling is affected. The existing
inverse path deliberately represents

\[
l_v = -\partial_v\lambda.
\]

That sign convention must be part of the field type or name.

The state contains physical Fourier amplitudes. Forces and velocities use the VMEC-decomposed representation. A state update therefore reapplies the mode normalization

\[
S_{mn}=m_{\mathrm{scale}}n_{\mathrm{scale}},\qquad
m_{\mathrm{scale}}=\begin{cases}1&m=0\\\sqrt2&m>0\end{cases},\quad
n\_{\mathrm{scale}}=\begin{cases}1&n=0\\\sqrt2&n>0\end{cases}.
\]

The `m=1` `R^{ss}/Z^{cs}` pair uses a trajectory-sensitive mixed gauge. For decomposed forces, let the incoming unmixed pair be `(f_Rss,f_Zcs)`. The mixed pair is

\[
\tilde f_{Rss}=\frac{f_{Rss}+f_{Zcs}}{\sqrt2},\qquad
\tilde f_{Zcs}=\frac{f_{Rss}-f_{Zcs}}{\sqrt2}.
\]

The second component is instead set to zero when

\[
\texttt{iter2}<2\quad\text{or}\quad
\mathrm{FSQZ}\_{\mathrm{previous}}<10^{-6}.
\]

The same mixed representation is used by velocity. Because physical state is stored in the undone gauge, its `m=1` increments are

\[
\Delta R^{ss}=\Delta t\,S_{mn}(v_{Rss}+v_{Zcs}),\qquad
\Delta Z^{cs}=\Delta t\,S_{mn}(v_{Rss}-v_{Zcs}).
\]

Before the radial solve, the odd-parity `m=1` pair is scaled with

\[
d=a_{R,d}+b_{R,d}+a_{Z,d}+b_{Z,d},\qquad
\tilde f_{Rss}\leftarrow\frac{a_{R,d}+b_{R,d}}{d}\tilde f_{Rss},\qquad
\tilde f_{Zcs}\leftarrow\frac{a_{Z,d}+b_{Z,d}}{d}\tilde f_{Zcs},
\]

using a scale-aware check on `d`. These conversions must be named operations, not duplicated index arithmetic.

Axis and boundary rules are also part of the representation contract:

- before every inverse transform, copy all six `m=1` families from surface 1 to the axis;
- for `m=0`, copy the `Lcs` family from surface 1 to the axis (the chi-force leftover);
- descent skips every `m>0` axis coefficient;
- fixed-boundary R/Z coefficients do not move at the LCFS;
- both lambda families remain free at the LCFS.

Initially preserve the current in-place axis extrapolation for Class A compatibility. The structured target should later stage extrapolated axis values as a transform input view instead of conflating them with canonical persisted coefficients. The state file preserves the current axis row; the v1 schema records its axis convention explicitly. That separation is a Class B/format change and must retain identical real-space axis geometry and force trajectory.

For odd `m`, the real-space work representation is regularized as

\[
q_o^{\mathrm{work}}(s)=
\frac{q_o^{\mathrm{physical}}(s)}
{\max(\sqrt{s},\sqrt{\Delta s})}.
\]

Physical state, regularized odd work values, decomposed residuals, decomposed velocities, and mixed-gauge values are different domains and should have different C++ view types.

## 3. Forward quadrature

The forward transform uses the reduced theta trapezoid, not a generic FFT round-trip normalization. For `nThetaRed = ntheta/2 + 1`, its weight is

\[
w_{k,m,n}=
\frac{m_{\mathrm{scale}}n_{\mathrm{scale}}}
{n_\zeta(n\_{\theta,\mathrm{red}}-1)}\,e_k,
\qquad
e_k=\begin{cases}\tfrac12&k\text{ is a theta endpoint}\\1&\text{otherwise.}\end{cases}
\]

The exact endpoint, axis, LCFS, parity, and zero-mode rules should live in one `QuadraturePlan`/`ModeTable`, shared by CPU reference and GPU code.

## 4. Profiles

Let

\[
T(s)=s\sum_i a_{\Phi i}s^i.
\]

The normalized toroidal-flux coordinate used by the power-series profiles is

\[
t(s)=\min(T(s),1),\qquad
\widehat t(s)=\min\left(|bloat\,t(s)|,1\right).
\]

Then the toroidal-flux derivative is

\[
\Phi'(s)=
\frac{\operatorname{signJ}\,\Phi\_{\mathrm{edge}}}
{2\pi T(1)}T'(s),
\]

and for fixed-iota mode

\[
\chi'(s)=\iota(s)\Phi'(s).
\]

For fixed-iota mode,

\[
\iota(s)=\sum_i a_{\iota i}t(s)^i.
\]

For pressure, apply the pedestal clamp before the toroidal-flux mapping:

\[
s_p=\min(s,s_{\mathrm{pres,ped}}),\qquad
\widehat t_p=\min\left(|bloat\,\min(T(s_p),1)|,1\right).
\]

The current gamma-zero implementation stores pressure in magnetic units:

\[
p(s)=\mu_0\,p_{\mathrm{scale}}\sum_i a_{Mi}\widehat t_p^{i}.
\]

For prescribed-current mode, define the integrated polynomial, its bloat-clamped radial profile, and the separately normalized edge value

\[
J_C(x)=\sum_i\frac{a_{Ci}}{i+1}x^{i+1},\qquad
C_H(s)=J_C(\widehat t(s)),\qquad
C_{edge}=J_C\left(\min(|bloat|,1)\right),
\]

then normalize

\[
I_{tor}=\frac{\operatorname{signJ}\,\mu_0\,curtor}{2\pi C_{edge}},\qquad
I_H(s)=I_{tor}C_H(s).
\]

`C_edge` is intentionally independent of `T(1)`; replacing it by `C_H(1)` changes general non-unit toroidal-flux profiles.

Its lambda normalization is

\[
L=\lambda_{\mathrm{scale}}=
\sqrt{\Delta s\sum_{j+1/2}\Phi'(s\_{j+1/2})^2}.
\]

`Phi'_H` in this sum is evaluated directly at `s_H`. By contrast, magnetic-field construction currently uses the full-grid average

\[
\Phi'\_{F\to H}=\frac12\left(\Phi'\_F(j)+\Phi'\_F(j+1)\right),
\]

which is not generally equal to direct half-grid evaluation for nonlinear `T(s)`. Both arrays and their consumers must remain distinct.

Validation requires nonzero, well-scaled `T(1)`, `C_edge` when prescribed current is active, plasma-volume/norm denominators, and `lambda_scale`. The new profile module separates immutable prescribed data from geometry-dependent evolving quantities. Until a gamma-dependent pressure law and `dV/ds` are implemented, only `gamma=0` is supported and `dV/ds=1` remains an explicit compatibility model.

The current full-grid parity helper uses `sqrt(s+1e-12)` at the axis while transforms/refinement use exact `sqrt(s)`. Phase 0 preserves this as `AxisRegularizationPolicy::LegacyEpsilon`; a later exact-axis policy may remove it only as a Class B numerical change with dedicated axis and trajectory tests.

## 5. Half-grid interpolation and geometry

For an even/odd parity pair, the current staggered interpolation is

\[
q_H=\frac12\left[(q_{e,j}+q_{e,j+1})
+\sqrt{s_H}(q_{o,j}+q\_{o,j+1})\right],
\]

\[
q_{s,H}=\frac{q_{e,j+1}-q_{e,j}
+\sqrt{s_H}(q_{o,j+1}-q\_{o,j})}{\Delta s}.
\]

The code forms

\[
\tau_1=R_{u,H}Z_{s,H}-R_{s,H}Z\_{u,H},\qquad
\tau=\tau_1+\frac14\tau_2,\qquad
\sqrt g=R_H\tau,
\]

where, writing the two neighboring full surfaces as `in` and `out`,

\[
\begin{aligned}
\tau_2={}&R_{u,o}^{out}Z_o^{out}+R_{u,o}^{in}Z_o^{in}
-Z_{u,o}^{out}R_o^{out}-Z_{u,o}^{in}R_o^{in}\\
&+\frac{1}{\sqrt{s_H}}\left(
R_{u,e}^{out}Z_o^{out}+R_{u,e}^{in}Z_o^{in}
-Z_{u,e}^{out}R_o^{out}-Z_{u,e}^{in}R_o^{in}
\right).
\end{aligned}
\]

This parity correction must be transcribed into a named, unit-tested host/device function; it must not be replaced by the simplified expression in the old project notes without a separate derivation and differential proof.

The stored metric is covariant:

\[
g_{uu}=R_u^2+Z_u^2,\qquad
g_{uv}=R_uR_v+Z_uZ_v,\qquad
g_{vv}=R^2+R_v^2+Z_v^2,
\]

with the current parity-staggered averaging applied to each term.

Validity uses the orientation-adjusted statistic

\[
J_{min}=\min_{H,\theta,\zeta}\left(\operatorname{signJ}\sqrt g\right),\qquad
J_{scale}=\max_{H,\theta,\zeta}|\sqrt g|.
\]

A stage point set is valid only when the nonfinite count is zero and

\[
J_{min}>\kappa_J\,\epsilon_T\,J_{scale},
\]

where `kappa_J` is a documented policy constant and the degenerate `J_scale=0` case is invalid. This definition avoids treating the normally negative raw Jacobian as a failure merely because of coordinate orientation.

## 6. Magnetic field and pressure

Using \(l_v=-\partial_v\lambda\),

\[
B^v=\frac{L\lambda_u+\Phi'_{F\to H}}{\sqrt g},\qquad
B^u=\frac{Ll_v+\chi'}{\sqrt g}.
\]

The covariant components and total pressure are

\[
B_u=g_{uu}B^u+g_{uv}B^v,\qquad
B_v=g_{uv}B^u+g\_{vv}B^v,
\]

\[
P\_{\mathrm{tot}}=p+\frac12\left(B^uB_u+B^vB_v\right).
\]

For prescribed-current mode, the half-grid closure is

\[
\chi'_H=
\frac{I_H-\left\langle g_{uu}B^u_\lambda+g_{uv}B^v\right\rangle}
{\left\langle g\_{uu}/\sqrt g\right\rangle},
\qquad
\iota_H=\frac{\chi'\_H}{\Phi'\_H}.
\]

The `ncurr=0` and `ncurr=1` data flows should be separate policy paths so fixed profiles cannot accidentally execute or mutate the current-closure state.

## 7. Force operator

The weak-form force kernel builds the reusable half-grid fluxes

\[
Q=R_HP_{\mathrm{tot}},\qquad
G_{uu}=\sqrt g(B^u)^2,\qquad
G_{uv}=\sqrt g B^uB^v,\qquad
G\_{vv}=\sqrt g(B^v)^2,
\]

then combines radial differences with poloidal and toroidal derivative terms for the even and odd parity families. The exact current equations should be copied first into a scalar CPU reference with the same operation order. Only after differential tests pass should the GPU implementation be split, fused, or algebraically simplified.

The hybrid lambda force uses

\[
r_b=0.1(1-s),
\]

\[
F_{\lambda,u}=-L\left[(1-r_b)\langle B_v\rangle+r_bB_{v,\mathrm{alt}}\right],
\qquad
F\_{\lambda,v}=-L\langle B_u\rangle.
\]

These `-L` formulas apply for `j>0`. At the magnetic axis, the current compatibility operator deliberately leaves the blended `B_v` average and the `B_u` average positive and unscaled by `-L`; its odd output then uses the configured axis `sqrt(s)` helper. `B_v_alt` must be a named expression with its own manufactured test because it mixes `g_vv/sqrt(g)`, normalized `lambda_u`, and `g_uv B^u`. The axis exception requires a direct regression test.

## 8. Spectral-condensation constraint

Define

\[
x\_{mpq}=m(m-1),
\]

and reconstruct

\[
R_{\mathrm{con}},Z_{\mathrm{con}}
=\sum_{m,n}x_{mpq}\{R_{mn},Z_{mn}\}\,\mathrm{basis}\_{mn}.
\]

With the LCFS-extrapolated reference fields `R_con,0`, `Z_con,0`,

\[
R_{\mathrm{con},0}(s)=s\,R_{\mathrm{con},\mathrm{LCFS}},\qquad
Z_{\mathrm{con},0}(s)=s\,Z_{\mathrm{con},\mathrm{LCFS}},
\]

This reference is reset on the first pass and after every restart, i.e. when `iter2 == iter1`.

and therefore

\[
g_{\mathrm{eff}}=
(R_{\mathrm{con}}-R_{\mathrm{con},0})R_u+
(Z_{\mathrm{con}}-Z\_{\mathrm{con},0})Z_u.
\]

The bandpass covers `m=1,...,mpol-2`. The current per-mode scale is

\[
f\_{\mathrm{accon}}(m)=\frac{1}{4m^2(m+1)^2}
\]

for the actual poloidal mode `m`, combined with the user `tcon0`. Mode limits, de-alias quadrature, and reference-reset cadence must be explicit inputs to the operator.

The current radial multiplier should receive the parsed input through

\[
M\_{tcon}=tcon0\,
\frac{1+n_s\left(1/60+n_s/24000\right)}{16},
\]

followed by the existing surface normalization

\[
tcon_j=\min\left(\frac{|a_{R,d,j}^{even}|}{\langle R_u^2\rangle_j},
\frac{|a_{Z,d,j}^{even}|}{\langle Z_u^2\rangle_j}\right)
M_{tcon}(32\Delta s)^2,
\]

with `tcon[j=0]=0` on the magnetic axis and the current LCFS half-weight. The overhaul should preserve this formula first, while replacing its exact-zero fallbacks with scale-aware validation.

## 9. Preconditioner

For each `(m,n)` and parity, the R/Z tridiagonal approximation uses

\[
D\_{mn}= -\left(A_d^{(p)}+m^2B_d^{(p)}+N^2C_d\right),
\]

\[
U_{j,mn},L_{j,mn}= -\left(A_h^{(p)}+m^2B_h^{(p)}\right)
\]

It solves

\[
L_jx_{j-1}+D_jx_j+U_jx_{j+1}=f_j.
\]

In current array names, `ar[j]` is `U_j`, the outer/super-diagonal multiplying `x[j+1]`, and `br[j]` is `L_j`, the inner/sub-diagonal multiplying `x[j-1]`. The target API uses `lower/diagonal/upper` names to prevent this historical naming trap.

The operator solves this as a batch over mode/component systems. The backend contract must state its supported row range and numerical pivot policy; a backend must never silently process only a prefix of the rows.

For fixed boundary,

\[
j\_{min}(m)=\begin{cases}0&m=0\\1&m>0\end{cases},
\]

and the R/Z solve covers `j_min,...,ns-2`; the LCFS row is excluded. For `m=1`, the lower coefficient at `j=1` is folded into the first solved diagonal, `D_1 <- D_1+L_1`, enforcing the axis treatment. The lambda diagonal applies through the LCFS but is zero at the axis and for the `(m,n)=(0,0)` gauge mode.

The lambda factor is

\[
f_\lambda=N^2b_\lambda+2mN\operatorname{copysign}(d_\lambda,b_\lambda)+m^2c\_\lambda,
\]

Here `copysign(d,b)=sign(b)|d|`; it is not `sign(b)*d` when `d<0`. Any proposed change to this expression must first establish the intended VMEC++ contract with a differential test.

\[
P_\lambda=
\frac{2}{4L^2}
\frac{(\sqrt{s})^{\min(m^2/256,8)}}{f_\lambda}.
\]

Every denominator must be checked relative to a norm of its local coefficients. A breakdown returns `NumericalStatus::SingularPreconditioner`; it must not be converted silently to a positive constant.

## 10. Residual, damping, and descent

Force normalization is refreshed with the preconditioner when

\[
(\texttt{iter2}-\texttt{iter1})\bmod 25=0,
\]

where `iter1` is the latest restart anchor and `iter2` is the effective iteration. Diagnostic mode must not add a refresh and thereby change the trajectory.

With reduced-grid trapezoidal weight `w`, define

\[
\begin{aligned}
E_{mag}&=\left|\sum \sqrt g\,\frac{|B|^2}{2}w\right|\Delta s,\\
E_{therm}&=\left(\sum_H p_H\frac{dV}{ds}\_H\right)\Delta s,\\
V&=\left(\sum_H\frac{dV}{ds}\_H\right)\Delta s,\\
e&=\frac{\max(E_{mag},E_{therm})}{V},\\
S_{RZ}&=\sum g\_{uu}R_H^2w,\\
S_L&=\sum(B_u^2+B_v^2)w.
\end{aligned}
\]

`dV/ds_H` is accumulated as `signJ·Σ√g·w` (signJ = −1).

Then

\[
f_{norm,RZ}=\frac{1}{S_{RZ}e^2},\qquad
f_{norm,L}=\frac{1}{S_L\,L^2},\qquad
f_{norm,1}=\frac{1}{RZNorm}.
\]

`RZNorm` is the squared decomposed R/Z state norm: divide physical coefficients by `S_mn`, exclude the `Rcc(0,0)` offset and stored `m>0` axis values, and use a factor `1/2` for the squared `m=1` `Rss/Zcs` pair because of its mixed representation. Every denominator above must pass a scale-aware positive check.

The invariant residuals are

\[
\begin{aligned}
\mathrm{FSQR}&=\frac14 f_{\mathrm{norm,RZ}}
\sum[(F_R^{cc})^2+(F_R^{ss})^2],\\
\mathrm{FSQZ}&=\frac14 f_{\mathrm{norm,RZ}}
\sum[(F_Z^{sc})^2+(F_Z^{cs})^2],\\
\mathrm{FSQL}&=f_{\mathrm{norm,L}}
\sum[(F_\lambda^{sc})^2+(F\_\lambda^{cs})^2].
\end{aligned}
\]

Convergence requires all three invariant components to satisfy their stage tolerance, equivalently

\[
\max(\mathrm{FSQR},\mathrm{FSQZ},\mathrm{FSQL})\le f\_{tol}.
\]

After preconditioning, define the controller scalar

\[
\begin{aligned}
\mathrm{FSQR1}&=f_{norm,1}\sum[(P^{-1}F_R^{cc})^2+(P^{-1}F_R^{ss})^2],\\
\mathrm{FSQZ1}&=f_{norm,1}\sum[(P^{-1}F_Z^{sc})^2+(P^{-1}F_Z^{cs})^2],\\
\mathrm{FSQL1}&=\Delta s\sum[(P^{-1}F_\lambda^{sc})^2+(P^{-1}F_\lambda^{cs})^2],\\
f_k&=\mathrm{FSQR1}\_k+\mathrm{FSQZ1}\_k+\mathrm{FSQL1}\_k.
\end{aligned}
\]

The damping estimate is based on the residual log-ratio:

\[
\tau_k^{-1}=
\frac{\min\left(\left|\log(f_k/f_{k-1})\right|,0.15\right)}{\Delta t}.
\]

The ten-entry history has exact restart/zero rules:

- whenever `iter2 == iter1`, fill all ten entries with `0.15/Delta t`;
- shift the history once per evaluated pass;
- insert a new log-ratio sample only when `iter2 > iter1`;
- if `f_k == 0`, insert zero rather than evaluating the logarithm;
- on the restart-anchor pass, the last entry remains the initialized `0.15/Delta t` value.

After the existing ten-sample average,

\[
d_\tau=\frac{\Delta t}{2}\overline{\tau^{-1}},\qquad
b_1=1-d_\tau,\qquad
f_{\mathrm{ac}}=\frac{1}{1+d_\tau}.
\]

The accelerated descent is

\[
v^{k+1}=f_{\mathrm{ac}}\left(b_1v^k+\Delta t\,P^{-1}F^k\right),
\qquad
x^{k+1}=x^k+\Delta t\,S_{mn}v^{k+1}.
\]

For a finite pass that reaches time-step control, the exact application order is:

1. compute the refresh/restart decision from the current residuals;
2. execute descent;
3. if progress requests refresh, copy the **post-descent** physical state to the checkpoint;
4. if a restart was selected, restore the older checkpoint after descent and zero velocity, thereby discarding that descent;
5. increment the effective iteration only on a non-restart pass.

A nonfinite invariant residual follows the earlier exceptional path: restore and reduce the step without executing descent. The ordering of invariant residual, preconditioning, preconditioned residual, damping history, restart decision, descent, post-descent checkpoint refresh, and post-descent restore is part of the numerical contract. The new controller must reproduce it from a pure state machine before any ordering change is attempted.

With `f_min` the running minimum of the preconditioned sum and `age=iter2-iter1`, the current control predicates are

\[
\begin{aligned}
refresh&:\quad f_k\le f_{min}\ \land\ age>10,\\
bad_jacobian&:\quad f_k>100f_{min}\ \land\ iter2>iter1,\\
bad_progress&:\quad age>12\ \land\ iter2>50\ \land\
(\mathrm{FSQR}+\mathrm{FSQZ})>10^{-2}.
\end{aligned}
\]

`bad_jacobian` multiplies `Delta t` by `0.9`; `bad_progress` divides it by `1.03`; both reset the restart anchor. These historically named conditions are distinct from the earlier oriented-Jacobian device status and should receive clearer enum names in schema v1 while legacy telemetry retains the original labels.

There is also a top-of-pass convergence-problem branch. If the accumulated bad-Jacobian counter equals 25 or 50, before axis extrapolation or geometry the solver restores the checkpoint, increments the counter, sets

\[
\Delta t=
\begin{cases}
0.98\,\Delta t_{initial},&\text{post-increment counter}<50,\\
0.96\,\Delta t_{initial},&\text{otherwise},
\end{cases}
\]

resets the restart/log anchors, and continues without geometry, descent, or effective-iteration increment. This maintenance pass must be represented explicitly by `IterationController::next_schedule()` and included in trajectory replay.

## 11. Multigrid prolongation

Odd modes are interpolated in the scaled coordinate

\[
x_c(s)=\frac{x_{\mathrm{physical}}(s)}
{\max(\sqrt{s},\sqrt{\Delta s\_{\mathrm{old}}})}.
\]

The old-axis stencil is `2*x_c(s1)-x_c(s2)`. The result is unscaled on the new grid, new odd-mode axis entries are zero, and the LCFS is copied exactly. These four rules need direct property tests for all six coefficient families.
