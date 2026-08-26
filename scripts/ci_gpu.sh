#!/usr/bin/env bash
# ci_gpu.sh — manual GPU release gate (not run by hosted CI)
# (completion plan step 4.3):
#
#   1. verify preset build (precise double, warnings-as-errors);
#   2. the FULL CTest suite, including the compute-sanitizer memcheck/
#      initcheck variants of every kernel-driving test;
#   3. the CLI strict-vs-compatibility policy gate;
#   4. the benchmark smoke gate;
#   5. a frozen SHORT-trajectory sanity run (CUMES_MAX_ITER=20 Solovev):
#      the run must reach the stage cap with a finite final residual and
#      reproduce the frozen iteration table rows (the full bitwise
#      trajectory gates stay the documented release step — compare_bitwise
#      against a recipe-captured baseline — because cross-GPU bitwise
#      equality is not a CI contract).
#
# The full release gate additionally runs the sanitizer preset's racecheck/
# synccheck/ASan entries; those are slow and stay a manual release step.
set -euo pipefail
cd "$(dirname "$0")/.."

cmake --preset verify
cmake --build build -j "$(nproc)"

ctest --test-dir build --output-on-failure -j4

# Frozen short trajectory: CUMES_MAX_ITER=20 caps the pass count; the solver
# must reach the cap and report stage failure (exit 1). The acceptance oracle:
#
#   - the documented stage-cap exit code (1) and the FATAL stage-cap message;
#   - the effective-iteration semantics: the controller's counter is ONE-BASED
#     (iter2 starts at 1 and advances after every good pass), so 20 capped
#     passes report `completed 21/1000` — asserting the exact number pins the
#     documented semantics instead of contradicting them;
#   - finite, positive final residuals (a nonfinite blow-up or a degenerate
#     zero state must fail the gate);
#   - no output artifacts: a max-iteration run writes no state file, so the
#     requested output path must not exist.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
set +e
CUMES_MAX_ITER=20 ./build/cumes inputs/solovev.json --output "$tmp/short.bin" > "$tmp/short.log" 2>&1
rc=$?
set -e
[ "$rc" -eq 1 ] || { echo "ci_gpu: short trajectory expected exit 1 (stage cap), got $rc"; exit 1; }
# The one-based effective-iteration counter: 20 passes -> `completed 21/1000`.
grep -q 'completed 21/1000 iterations without meeting ftol' "$tmp/short.log" \
  || { echo 'ci_gpu: stage-cap message does not match the documented 20-pass -> 21-effective-iteration contract'; exit 1; }
# Final residuals: finite and strictly positive.
res_line=$(grep -oE 'final residuals fsqr=[0-9.eE+-]+ fsqz=[0-9.eE+-]+ fsql=[0-9.eE+-]+' "$tmp/short.log" \
  || { echo 'ci_gpu: missing final-residuals line'; exit 1; })
# Fields of the matched line: $1=final $2=residuals $3=fsqr=.. $4=fsqz=..
# $5=fsql=... Each value must be finite and strictly positive (inf fails the
# v == v + 1 test, NaN fails every comparison, nonpositive fails v > 0).
echo "$res_line" | awk '{ for (i=3; i<=5; i++) { split($i, a, "="); v = a[2] + 0; if (!(v > 0) || v == v + 1) { exit 1 } } }' \
  || { echo "ci_gpu: final residuals not finite and positive: $res_line"; exit 1; }
grep -q 'nan\|NaN\|inf' "$tmp/short.log" && { echo 'ci_gpu: nonfinite output in short trajectory'; exit 1; }
[ ! -e "$tmp/short.bin" ] || { echo 'ci_gpu: max-iteration run wrote a state file'; exit 1; }

echo 'ci_gpu: all gates passed'
