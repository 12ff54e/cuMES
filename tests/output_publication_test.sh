#!/usr/bin/env bash
# output_publication_test.sh — the end-to-end durable-publication contract
# (completion-plan follow-up §3).
#
# Drives the REAL binary to a converged Solovev run and asserts:
#
#   - a publication failure (target is a directory) makes main exit nonzero
#     with the typed "FAILED to write output state" message, leaves the target
#     untouched, and drops no stray temp file;
#   - when the temp cannot even be created (read-only destination directory),
#     main still fails and a PRE-EXISTING destination keeps its exact bytes.
#
# Usage: output_publication_test.sh <path-to-cuMES> <path-to-converging-input>
set -u

BIN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
INPUT=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
fail=0
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp" || exit 2

# Precision probe: float builds cannot meet the double-tuned stage ftols, so
# the converged end-to-end cases run only in the double build. The publication
# protocol itself is precision-independent (the per-backend fault boundaries
# are covered in test_output_failure at every precision).
cat > probe.json <<'EOF'
{"mpol": 2, "ntor": 0, "nfp": 1, "am": [1.0], "aphi": [1.0], "ai": [0.5],
 "rbc": [{"n": 0, "m": 1, "value": 1.0}],
 "zbs": [{"n": 0, "m": 1, "value": 0.5}],
 "ns_array": [5], "niter_array": [1], "ftol_array": [1e-6]}
EOF
if "$BIN" probe.json probe.bin 2>&1 | grep -q 'precision: float'; then
  echo "SKIP end-to-end converged publication cases (float build)"
  exit 0
fi

check() { # name cond
  if [ "$2" -eq 0 ]; then
    echo "PASS $1"
  else
    echo "FAIL $1"
    fail=1
  fi
}

# ---- Case 1: publication failure -> main fails, target untouched ----------
# A directory with a valid .bin suffix: the writer produces the same-directory
# temp fine but rename(temp, target) fails; the run must report failure and
# leave the directory (and no stray temp) behind.
mkdir -p target.bin
set +e
"$BIN" "$INPUT" target.bin > case1.log 2>&1
rc1=$?
set -e
check "publication failure exits nonzero" "$([ "$rc1" -ne 0 ] && echo 0 || echo 1)"
check "publication failure reports typed message"       "$(grep -q 'FAILED to write output state' case1.log && echo 0 || echo 1)"
check "publication failure leaves target untouched"       "$([ -d target.bin ] && echo 0 || echo 1)"
check "publication failure leaves no stray temp"       "$(ls target.bin.tmp.* 2>/dev/null | grep -q . && echo 1 || echo 0)"
rmdir target.bin

# ---- Case 2: temp creation fails -> previous destination preserved --------
mkdir -p ro
printf 'PRECIOUS-BYTES-42
' > ro/state.bin
cp ro/state.bin ro/state.bin.expected
chmod 555 ro
set +e
"$BIN" "$INPUT" ro/state.bin > case2.log 2>&1
rc2=$?
set -e
chmod 755 ro
check "read-only destination exits nonzero"       "$([ "$rc2" -ne 0 ] && echo 0 || echo 1)"
check "read-only destination reports typed message"       "$(grep -q 'FAILED to write output state' case2.log && echo 0 || echo 1)"
check "read-only destination preserves previous bytes"       "$(cmp -s ro/state.bin ro/state.bin.expected && echo 0 || echo 1)"

if [ "$fail" -ne 0 ]; then
  echo "output_publication_test: FAILED"
  exit 1
fi
echo "output_publication_test: all checks passed"
exit 0
