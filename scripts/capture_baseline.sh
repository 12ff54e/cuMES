#!/usr/bin/env bash
# capture_baseline.sh — regenerate the cuMES forensic baseline run tree.
#
# The overhaul's Class A "bitwise equivalence" gate needs a byte-comparable
# reference. Rather than committing megabytes of binary dump data, the baseline
# is defined by its *recipe*: a git revision + a build + fixed inputs + fixed
# environment knobs. Running this script at that revision reproduces the tree.
#
#   baseline recipe = { revision, inputs/*.json, CUMES_DUMP=1, stage caps from
#                       the JSON, same toolchain/flags/GPU }
#
# Because the dump machinery couples to the trajectory (solver.cu adds a precon
# refresh at iter2==kDumpIter when dumpEnabled()), every run compared with
# scripts/compare_bitwise.py MUST use the identical knobs — in particular
# CUMES_DUMP=1 — so the dump-window refresh is present on both sides. This
# script fixes those knobs; do not override them when producing a candidate
# tree.
#
# Each output tree IS a solver run scratch directory (so the full dump set is
# present and byte-comparable), plus the committed essentials and provenance:
#
#   <out>/<precision>/<config>/
#     cumes_state.bin              final state (2 ints + 6 x ns*mnmax doubles)
#     per_iter_residuals_cumes.bin 15-col double trajectory record (all stages)
#     step_0_*.bin                 initial-state component snapshots
#     dump_manifest.sha256         SHA-256 of the FULL dump/cuMES set
#     PROVENANCE.txt               revision, build, knobs that produced the tree
#     run.log                      full solver stdout
#     dump/cuMES/                  the full run dump set (step_A..I, fspec,
#                                  state, vel, force_norms — ~150M for W7-X)
#
# The whole tree is regenerable and lives under a gitignored scratch dir; the
# only committed artifacts are this script and scripts/compare_bitwise.py.
#
# Usage:
#   scripts/capture_baseline.sh \
#       --build <build-dir> [--float-build <float-build-dir>] \
#       --out <baseline-tree-root> [--configs solovev,w7x] [--no-full-manifest] \
#       [--float-ftol <cfg>=<tol>]... [--schema]
#
#   --build             double-precision build dir (contains ./cuMES)
#   --float-build       optional float build dir; when given, float configs are
#                       captured too (with ftol relaxed, since float cannot
#                       meet the double-tuned tolerances).
#   --out               output tree root, e.g. .verify-scratch/baseline
#   --configs           comma-separated configs to run (default solovev,w7x)
#   --float-ftol        per-config float tolerance override, repeatable
#                       (e.g. --float-ftol w7x=1e-2). Defaults: solovev=1e-6,
#                       w7x=1e-2 (W7-X prescribed-current stalls at ~3e-3 in
#                       float, so 1e-3 never converges; 1e-2 does).
#   --no-full-manifest  skip writing dump_manifest.sha256 (still stage the
#                       essentials; the full dump set stays in the tree).
#   --schema            also emit the NetCDF/HDF5 schema dumps into each tree
#                       (state.nc / state.h5), freezing the on-disk schema so
#                       the I/O-matrix goldens can be regenerated on demand.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

BUILD=""
FLOAT_BUILD=""
OUT=""
CONFIGS="solovev,w7x"
FULL_MANIFEST=1
SCHEMA=0
REV=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
# Per-config float convergence tolerance (the float residual floor is
# problem-dependent; W7-X prescribed-current stalls at ~3e-3).
declare -A FLOAT_FTOL=([solovev]=1.0e-6 [w7x]=1.0e-2)

usage() { grep '^#' "$0" | sed -n 's/^# \{0,1\}//p' | head -45; }

while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD="$2"; shift 2;;
    --float-build) FLOAT_BUILD="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --configs) CONFIGS="$2"; shift 2;;
    --float-ftol) FLOAT_FTOL[${2%%=*}]="${2#*=}"; shift 2;;
    --no-full-manifest) FULL_MANIFEST=0; shift;;
    --schema) SCHEMA=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[ -n "$BUILD" ] || { echo "error: --build is required" >&2; exit 2; }
[ -n "$OUT" ]   || { echo "error: --out is required" >&2; exit 2; }
[ -x "$BUILD/cuMES" ] || { echo "error: $BUILD/cuMES not found" >&2; exit 2; }
mkdir -p "$OUT"

run_capture() { # $1=build-dir  $2=precision-label  $3=input.json  $4=out-subdir
  local build="$1" prec="$2" input="$3" sub="$4"
  local tree="$OUT/$prec/$sub"
  rm -rf "$tree"
  mkdir -p "$tree"
  echo "== capturing $prec/$sub from $input =="
  # inputs/*.json are read relative to the CWD; the tree IS the run scratch,
  # so dump/ and cumes_state.bin land inside it.
  ( cd "$tree" \
    && CUMES_DUMP=1 "$OLDPWD/$build/cuMES" "$OLDPWD/$input" cumes_state.bin \
       > run.log 2>&1 )
  [ -f "$tree/cumes_state.bin" ] || { echo "error: $prec/$sub produced no state" >&2; exit 1; }

  # Essentials: per-iteration trajectory record + initial-state snapshots are
  # already in dump/cuMES; copy them to the tree top level for quick access.
  cp "$tree/dump/cuMES/per_iter_residuals_cumes.bin" "$tree/per_iter_residuals_cumes.bin"
  for f in "$tree"/dump/cuMES/step_0_*.bin; do
    [ -e "$f" ] && cp "$f" "$tree/"
  done

  # NetCDF/HDF5 schema dumps: re-run the identical solve with the other output
  # suffixes so the on-disk schemas are frozen as part of the recipe. These are
  # separate files in the tree; compare_bitwise.py's --full mode includes them.
  if [ "$SCHEMA" = 1 ]; then
    ( cd "$tree" && CUMES_DUMP=1 "$OLDPWD/$build/cuMES" "$OLDPWD/$input" state.nc \
       > run-schema-nc.log 2>&1 )
    ( cd "$tree" && CUMES_DUMP=1 "$OLDPWD/$build/cuMES" "$OLDPWD/$input" state.h5 \
       > run-schema-h5.log 2>&1 )
    echo "  schema: state.nc + state.h5 captured"
  fi

  if [ "$FULL_MANIFEST" = 1 ] && [ -d "$tree/dump/cuMES" ]; then
    ( cd "$tree" && find dump/cuMES -type f | sort | while read -r f; do
        printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "${f#dump/cuMES/}"
      done ) > "$tree/dump_manifest.sha256"
    echo "  manifest: $(wc -l < "$tree/dump_manifest.sha256") files"
  fi

  cat > "$tree/PROVENANCE.txt" <<EOF
revision:      $REV
build:         $build
precision:     $prec
input:         $input
stage caps:    from $input (ns_array/niter_array/ftol_array)
knobs:         CUMES_DUMP=1 (required for byte-compare comparability)
output:        cumes_state.bin (binary, double on disk)
recorded:      $(date -u +%Y-%m-%dT%H:%M:%SZ)
generated-by:  scripts/capture_baseline.sh
EOF
  echo "  -> $tree"
}

# Double precision (the verification configuration).
for cfg in ${CONFIGS//,/ }; do
  run_capture "$BUILD" double "inputs/$cfg.json" "$cfg"
done

# Float (experimental): relax the double-tuned ftol entries to a level the
# run can actually reach (the float residual floor is problem-dependent).
if [ -n "$FLOAT_BUILD" ]; then
  [ -x "$FLOAT_BUILD/cuMES" ] || { echo "error: $FLOAT_BUILD/cuMES not found" >&2; exit 2; }
  local_scratch="$OUT/float-inputs"
  mkdir -p "$local_scratch"
  for cfg in ${CONFIGS//,/ }; do
    ft=${FLOAT_FTOL[$cfg]:-1.0e-6}
    sed -E "s/\"(ftol_array)\"[^]]*\]/\"\1\" : [${ft}, ${ft}, ${ft}]/" \
        "inputs/$cfg.json" > "$local_scratch/$cfg.json"
    run_capture "$FLOAT_BUILD" float "$local_scratch/$cfg.json" "$cfg"
  done
fi

echo
echo "baseline captured at $OUT (revision $REV)"
echo "verify a candidate tree with: scripts/compare_bitwise.py $OUT/double/<cfg> <candidate>/double/<cfg>"
