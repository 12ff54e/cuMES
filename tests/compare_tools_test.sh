#!/usr/bin/env bash
# Dependency-free end-to-end smoke tests for the four comparison CLIs.

set -euo pipefail

COMPARE_STATES=$1
COMPARE_RUNS=$2
COMPARE_BITWISE=$3
COMPARE_WOUT=$4

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

make_state() { # $1=path: version=1, ns=2, mnmax=1, six zero families
  {
    printf 'CUMES001\001\000\000\000\002\000\000\000\001\000\000\000'
    dd if=/dev/zero bs=96 count=1 status=none
  } > "$1"
}

expect_failure() {
  local expected=$1
  shift
  set +e
  "$@" >"$scratch/command.out" 2>"$scratch/command.err"
  local actual=$?
  set -e
  if [ "$actual" -ne "$expected" ]; then
    echo "expected exit $expected, got $actual: $*" >&2
    cat "$scratch/command.out" >&2
    cat "$scratch/command.err" >&2
    exit 1
  fi
}

make_state "$scratch/state-a.bin"
cp "$scratch/state-a.bin" "$scratch/state-b.bin"
"$COMPARE_STATES" "$scratch/state-a.bin" "$scratch/state-b.bin" >/dev/null

# Set family rmncc, mode 0, interior j=1 to 1.0 in the second state.
printf '\000\000\000\000\000\000\360\077' | \
  dd of="$scratch/state-b.bin" bs=1 seek=28 conv=notrunc status=none
expect_failure 1 "$COMPARE_STATES" "$scratch/state-a.bin" \
  "$scratch/state-b.bin"
cp "$scratch/state-a.bin" "$scratch/state-b.bin"

cat >"$scratch/a.log" <<'EOF'
    0 |   1.000e-01   2.000e-01   3.000e-01 | 9.00e-01
    1 |   1.000e-03   2.000e-03   3.000e-03 | 8.00e-01
BAD JACOBIAN (iter2=1) delt=8.000e-01
CONVERGED at iter 2
  Status: CONVERGED
  Iterations: 2
  FSQR: 1.000e-09
  FSQZ: 2.000e-09
  FSQL: 3.000e-09
EOF
cp "$scratch/a.log" "$scratch/b.log"
"$COMPARE_RUNS" --no-state "$scratch/a.log" "$scratch/b.log" >/dev/null
"$COMPARE_RUNS" "$scratch/a.log" "$scratch/state-a.bin" \
  "$scratch/b.log" "$scratch/state-b.bin" >/dev/null
sed 's/1.000e-03/2.000e-01/' "$scratch/a.log" >"$scratch/b.log"
expect_failure 1 "$COMPARE_RUNS" --no-state "$scratch/a.log" "$scratch/b.log"

mkdir "$scratch/baseline" "$scratch/run"
for tree in baseline run; do
  cp "$scratch/state-a.bin" "$scratch/$tree/cumes_state.bin"
  printf 'trajectory\n' >"$scratch/$tree/per_iter_residuals_cumes.bin"
  printf 'initial\n' >"$scratch/$tree/init_state.bin"
done
"$COMPARE_BITWISE" "$scratch/baseline" "$scratch/run" >/dev/null
printf 'changed\n' >"$scratch/run/per_iter_residuals_cumes.bin"
expect_failure 1 "$COMPARE_BITWISE" "$scratch/baseline" "$scratch/run"

# The HDF5-backed semantic path is exercised when real wout fixtures are used;
# this smoke gate at least verifies the fourth CLI is present and parseable.
"$COMPARE_WOUT" --help >/dev/null

echo "comparison tool smoke tests passed"
