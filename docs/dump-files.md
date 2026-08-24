# Dump files (diagnostics)

This document describes the `CUMES_DUMP`-gated observability output: which
files the solver writes, when, what they contain, and their on-disk layout.
The producer is the dump module (`include/cumes/solver/dump_windows.hpp`,
`src/kernels/dump_impl.cuh`); the dump windows themselves fire inside
`EquilibriumOperator::enqueue` at fixed observation points of the iteration
DAG, and the solver-run helpers cover the pre-loop and post-loop ends.

These files are development diagnostics, **not a stable format**. They are
paired with the dump producer: the consumers are
`scripts/compare_bitwise.py` / `scripts/capture_baseline.sh` (the Class A
bitwise gate) and same-build test tooling. Everything stable is the versioned
containers (`docs/output-formats.md`).

## 1. Gating and knobs

- Compile gate: the dump machinery is compiled in only when the solver TUs
  are built with `DUMP_CUMES_VERIFY` defined (`CUMES_ENABLE_VERIFY_DUMP=ON` —
  the verify/sanitizer/float presets). Production-style builds compile it out.
- Runtime gate: with the machinery compiled in, nothing is written unless
  `CUMES_DUMP=1` (read once per process).
- Window knobs: `CUMES_DUMP_ITER` (default 150) selects the handoff
  iteration, `CUMES_E2_START` (default 560) opens the invariant/
  preconditioned-force windows, `CUMES_MAX_ITER` overrides the max-iteration
  cap (and with it the final-pass window). All three read once per process
  (`CUMES_MAX_ITER` is folded per stage, since multigrid overwrites
  `p.max_iter`).
- The dump windows fire for whichever spectral backend runs
  (`CUMES_FORCE_GENERIC` selects it); the files are written either way.

All output lands in `dump/cuMES/` relative to the working directory. In a
multigrid run every stage re-enters `solver_run`, so files without an
iteration tag (the `init_*` set, `final_fspec.bin`) and the per-pass record
are overwritten by each stage — the surviving copy is the final stage's.

## 2. File formats

Dump files are **T-native**: element arrays are written as `sizeof(T)` native
elements (float builds write floats), so files are read back by same-build
tooling only. Three layouts exist:

**A. Device arrays** (`dump_device_array`, nearly all `*.bin` files):

```
uint64_t nelem                 (native endianness)
nelem × sizeof(T) elements     (device-major layout: point + surface*nZnT
                               for real space, surface + mode*ns for spectral)
```

**B. Host scalar/row arrays** (`precon_jmin_iter_1.bin`,
`precon_sizes_iter_1.bin` — written by `dump_step_precon` directly):

```
uint64_t n
n × double
```

**C. Text** (`force_norms_iter_<n>.txt`): ten `key value` lines at `%.17e`
precision — `magneticEnergy`, `thermalEnergy`, `plasmaVolume`,
`energyDensity`, `forceNormSumRZ`, `forceNormSumL`, `rzNorm`, `fNormRZ`,
`fNormL`, `fNorm1` (the vmecpp force-norm text format).

**Per-pass trajectory record** (`per_iter_residuals_cumes.bin`): layout A
with `double` elements regardless of build, serialized **column-major**:
`uint64_t n` followed by 15 blocks of `n` doubles, one block per
`PassRecord` field, in the field order
`invariant_fsqr invariant_fsqz invariant_fsql preconditioned_fsqr
preconditioned_fsqz preconditioned_fsql delta_t otav dtau b1 fac iter2
iter1 reason axis_r` (byte-identical to the legacy array-of-doubles layout;
see `include/cumes/solver/pass_record.hpp`).

## 3. Naming scheme

Every windowed file is named

```
<window>_<array>_iter_<tag>.bin
```

- `<window>` — the pipeline point that produced it (table below), replacing
  the old vmecpp step letters (A–I, GC, `step_0`, `step_half`, `step_precon`,
  `step_final`).
- `<array>` — the buffer's established identifier (`rmncc`, `lu_e`, `bsupu`,
  `jmin`, …), unchanged.
- `<tag>` — `1` for the first pass, and the effective iteration (`iter2`)
  afterwards, mirroring vmecpp's 1-based dump naming. Per-stage one-shots
  (`init_*`, `final_fspec.bin`) carry no tag.

The vmecpp step-letter mapping:

| vmecpp step | window | Files |
| --- | --- | --- |
| `step_0_` | `init_` | pre-loop state/profile snapshots |
| A | `postinverse_` | real-space geometry right after the inverse transform |
| B, C | `metric_` | √g and the metric elements g_uu/g_uv/g_vv |
| D | `bcontra_` | contravariant B^θ/B^ζ |
| E | `forceterm_` | the MHD force term arrays (A/B/C terms) |
| F | `force_` | the b-force arrays the constraint chain augments |
| G, GC | `constraint_` | augmented forces + constraint-chain intermediates (the two letter sets are array-disjoint, hence one window name) |
| H | `scaled_` | f_spec after the odd-m decomposition scaling |
| I | `preconditioned_` | f_spec after the in-place preconditioner |
| `step_half_` | `halfgrid_` | half-grid geometry + field arrays |
| `step_precon_` | `precon_` | preconditioner tridiagonal elements + intermediates |
| `step_final_` | `final_` | final-pass force slab |
| — | `fspec_invariant_`, `fspec_precon_`, `state_`, `vel_`, `force_norms_`, `per_iter_residuals_cumes` | unchanged |

## 4. The dump windows

Sizes: `n_real = ns·nZnT`, `n_half = (ns-1)·nZnT`, `n_spec = ns·mnmax`.

### 4.1 init (pre-loop, once per stage)

`dump_step_0`, before the iteration loop. Layout A.

| File | Contents |
| --- | --- |
| `init_rmncc.bin`, `init_zmnsc.bin`, `init_lmnsc.bin`, `init_rmnss.bin`, `init_zmncs.bin`, `init_lmncs.bin` | the six spectral state families (`n_spec`, mode-major) as the stage starts |
| `init_currH.bin`, `init_chipH.bin`, `init_iotaH.bin` | the initial half-grid profiles (`ns-1`) |
| `init_iotaF.bin` | the full-grid iota profile (`ns`) |

### 4.2 iter0 diagnostic (first pass, post-inverse)

`dump_iter0_loop_diag`: dumps the full even-parity real-space R array and
prints the LCFS value.

| File | Contents |
| --- | --- |
| `iter0_diag_r_e.bin` | `rs.d_r_e` (`n_real`) |

### 4.3 postinverse (post-inverse transform, pre-geometry)

`dump_step_a`. Tags 1 and 2: the λ real-space derivatives; tag 1 only: the
full R/Z/λ snapshot (the combined `*_real` arrays are materialized by
`combine_parity` immediately before the dump — the hot loop runs the inverse
transform with `do_combine=false`).

| File | Contents |
| --- | --- |
| `postinverse_lu_e.bin`, `postinverse_lu_o.bin`, `postinverse_l_real.bin`, `postinverse_lv_e.bin`, `postinverse_lv_o.bin` (tags 1, 2) | λ poloidal/toroidal derivatives (`n_real`) |
| `postinverse_r_real.bin`, `postinverse_z_real.bin` (tag 1) | combined (e+o) R, Z |
| `postinverse_r_e.bin`, `postinverse_r_o.bin`, `postinverse_z_e.bin`, `postinverse_z_o.bin`, `postinverse_l_e.bin`, `postinverse_l_o.bin` (tag 1) | even/odd parity R, Z, λ |
| `postinverse_ru_real.bin`, `postinverse_zu_real.bin`, `postinverse_lu_real.bin`, `postinverse_rv_real.bin`, `postinverse_zv_real.bin`, `postinverse_lv_real.bin` (tag 1) | combined poloidal/toroidal derivatives |
| `postinverse_ru_e.bin`, `postinverse_ru_o.bin`, `postinverse_zu_e.bin`, `postinverse_zu_o.bin`, `postinverse_lu_e.bin`, `postinverse_lu_o.bin` (tag 1) | parity-split poloidal derivatives |

### 4.4 metric + bcontra (post-field)

`dump_step_d`, after the magnetic-field operator.

| File | Contents |
| --- | --- |
| `metric_gsqrt_iter_1.bin`, `metric_guu_iter_1.bin`, `metric_guv_iter_1.bin`, `metric_gvv_iter_1.bin` | Jacobian + metric elements on the half grid (`n_half`) |
| `bcontra_bsupu_iter_<tag>.bin`, `bcontra_bsupv_iter_<tag>.bin` (tags 1, 2) | contravariant B^θ/B^ζ (`n_half`) |

### 4.5 precon (iter 0, on the preconditioner refresh pass)

`dump_step_precon`. Layout A except `jmin`/`sizes` (layout B).

| File | Contents |
| --- | --- |
| `precon_ar_iter_1.bin`, `precon_dr_iter_1.bin`, `precon_br_iter_1.bin`, `precon_az_iter_1.bin`, `precon_dz_iter_1.bin`, `precon_bz_iter_1.bin` | tridiagonal elements, mode-major (`n_spec`) |
| `precon_jmin_iter_1.bin` | the per-mode jMin as doubles (`mnmax`) |
| `precon_arm_iter_1.bin`, `precon_ard_iter_1.bin`, `precon_brm_iter_1.bin`, `precon_brd_iter_1.bin`, `precon_azm_iter_1.bin`, `precon_azd_iter_1.bin`, `precon_bzm_iter_1.bin`, `precon_bzd_iter_1.bin` | odd/even-diagonal intermediates |
| `precon_cxd_iter_1.bin` | λ-channel diagonal (`ns`) |
| `precon_sizes_iter_1.bin` | `{ns, ns-1, mpol, 1.0}` (layout B) |

### 4.6 halfgrid / forceterm / force (post-force)

`dump_step_ef`, after the MHD force operator (before the constraint force is
added).

| File | Contents |
| --- | --- |
| `halfgrid_r12_iter_1.bin`, `halfgrid_zu12_iter_1.bin`, `halfgrid_tau_iter_1.bin`, `halfgrid_gsqrt_iter_1.bin`, `halfgrid_rs_iter_1.bin`, `halfgrid_zs_iter_1.bin` | half-grid geometry (`n_half`) |
| `halfgrid_totalP_iter_1.bin`, `halfgrid_bsupu_iter_1.bin`, `halfgrid_bsupv_iter_1.bin`, `halfgrid_bsubu_iter_1.bin`, `halfgrid_bsubv_iter_1.bin` | half-grid total pressure + B (`n_half`) |
| `forceterm_armn_e.bin`, `forceterm_armn_o.bin`, `forceterm_azmn_e.bin`, `forceterm_azmn_o.bin`, `forceterm_brmn_e.bin`, `forceterm_brmn_o.bin`, `forceterm_bzmn_e.bin`, `forceterm_bzmn_o.bin`, `forceterm_crmn_e.bin`, `forceterm_crmn_o.bin`, `forceterm_czmn_e.bin`, `forceterm_czmn_o.bin`, `forceterm_blmn_e.bin`, `forceterm_blmn_o.bin`, `forceterm_clmn_e.bin`, `forceterm_clmn_o.bin` (tag 1) | the A/B/C-term force arrays (`n_real`) |
| `force_brmn_e.bin`, `force_brmn_o.bin`, `force_bzmn_e.bin`, `force_bzmn_o.bin`, `force_blmn_e.bin`, `force_blmn_o.bin` (tags 1, 2) | the b-force arrays (`n_real`) |

### 4.7 constraint (iter 0, post-constraint)

`dump_step_g`, after the constraint operator has added the spectral
condensation force.

| File | Contents |
| --- | --- |
| `constraint_brmn_e_iter_1.bin`, `constraint_brmn_o_iter_1.bin`, `constraint_bzmn_e_iter_1.bin`, `constraint_bzmn_o_iter_1.bin` | the augmented b-forces (`n_real`) |
| `constraint_rmncc_iter_1.bin`, `constraint_rmnss_iter_1.bin`, `constraint_zmnsc_iter_1.bin`, `constraint_zmncs_iter_1.bin` | the spectral state the chain consumed (`n_spec`) |
| `constraint_rCon_iter_1.bin`, `constraint_zCon_iter_1.bin`, `constraint_gConEff_iter_1.bin`, `constraint_gCon_iter_1.bin`, `constraint_frcon_e_iter_1.bin`, `constraint_frcon_o_iter_1.bin`, `constraint_fzcon_e_iter_1.bin`, `constraint_fzcon_o_iter_1.bin` | the constraint force chain (`n_real`) |
| `constraint_tcon_iter_1.bin` (`ns`), `constraint_faccon_iter_1.bin` (`mpol`) | the constraint-force multiplier profiles |

### 4.8 scaled (post-decomposition scaling)

`dump_step_h`, after the odd-m `scalxc` scaling.

| File | Contents |
| --- | --- |
| `scaled_f_spec_iter_<tag>.bin` (tags 1, `CUMES_DUMP_ITER`) | the decomposed force slab `f_spec` (`6·n_spec`) |

### 4.9 final (final pass, post-m1-gauge)

`dump_step_final`, on the last pass before the invariant-residual reduction.

| File | Contents |
| --- | --- |
| `final_fspec.bin` | the decomposed force slab (`6·n_spec`) |

### 4.10 preconditioned + handoffs (post-preconditioner)

`dump_step_i`. Tags 1, 2–4, 51, and `CUMES_DUMP_ITER`..`+2`.

| File | Contents |
| --- | --- |
| `preconditioned_f_spec_iter_<tag>.bin` | the preconditioned force slab (`6·n_spec`) |
| `state_rmncc_iter_<n>.bin`, `state_zmnsc_iter_<n>.bin`, `state_lmnsc_iter_<n>.bin`, `state_rmnss_iter_<n>.bin`, `state_zmncs_iter_<n>.bin`, `state_lmncs_iter_<n>.bin` | the spectral state at the handoff window (`n_spec`) |
| `vel_vrmncc_iter_<n>.bin`, `vel_vzmnsc_iter_<n>.bin`, `vel_vlmnsc_iter_<n>.bin`, `vel_vrmnss_iter_<n>.bin`, `vel_vzmncs_iter_<n>.bin`, `vel_vlmncs_iter_<n>.bin` | the velocities (`n_spec`) |

### 4.11 E2-start windows (invariant/preconditioned force)

Opened for `iter2` in `[CUMES_E2_START, CUMES_E2_START+40)` by `dump_step_h`
and `dump_step_i` — the same force slab at the two observation points, one
file per pass.

| File | Contents |
| --- | --- |
| `fspec_invariant_iter_<n>.bin` | f_spec after the m1 gauge, pre-preconditioner |
| `fspec_precon_iter_<n>.bin` | f_spec after the in-place preconditioner |

### 4.12 force-norm telemetry (refresh passes)

`dump_force_norms`, on the preconditioner-refresh cadence.

| File | Contents |
| --- | --- |
| `force_norms_iter_<n>.txt` | layout C (the ten force-norm keys, `%.17e`) |

## 5. Consumers and the Class A gate

- `scripts/capture_baseline.sh` regenerates a run tree on demand
  (`CUMES_DUMP=1`, fixed knobs) and lifts the essentials
  (`per_iter_residuals_cumes.bin`, the `init_*` set) to the tree root plus a
  `dump_manifest.sha256` of the full `dump/cuMES/` set.
- `scripts/compare_bitwise.py` compares two such trees: the final state
  payload, the per-pass record, the `init_*` set, the dump manifest
  checksums, and (with `--full`) every dump file.
- Renaming a dump file invalidates stored baselines' manifests — regenerate
  with `capture_baseline.sh` rather than editing `.verify-scratch/` (which
  is gitignored and never committed).
