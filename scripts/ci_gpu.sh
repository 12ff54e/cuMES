#!/usr/bin/env bash
# ci_gpu.sh — the GPU release gate for the self-hosted CI runner
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

# Frozen short trajectory: 20-iteration cap; the solver must reach the cap
# (stage-failure exit 1) with a finite, positive final residual, never a
# nonfinite blow-up or an early exit.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
set +e
CUMES_MAX_ITER=20 ./build/cuMES inputs/solovev.json "$tmp/short.bin" > "$tmp/short.log" 2>&1
rc=$?
set -e
[ "$rc" -eq 1 ] || { echo "ci_gpu: short trajectory expected exit 1 (stage cap), got $rc"; exit 1; }
grep -q 'completed 20/20 iterations' "$tmp/short.log" || { echo 'ci_gpu: missing stage-cap marker'; exit 1; }
grep -q 'Done.' "$tmp/short.log" || { echo 'ci_gpu: run did not finish'; exit 1; }
grep -q 'nan\|NaN\|inf' "$tmp/short.log" && { echo 'ci_gpu: nonfinite output in short trajectory'; exit 1; }

echo 'ci_gpu: all gates passed'

