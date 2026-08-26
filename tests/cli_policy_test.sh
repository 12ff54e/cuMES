#!/usr/bin/env bash
# cli_policy_test.sh — strict-vs-compatibility CLI policy, end to end
# (completion plan step 2.1). Runs the real cuMES binary against tiny inputs:
#
#   - strict schema (default) rejects unknown input keys;
#   - --compatibility warns and continues (vmecpp-style ignore);
#   - input is a mandatory positional argument;
#   - output is an option and defaults to $PWD/cumes-output.bin;
#   - strict rejects an unknown output suffix.
#
# Usage: cli_policy_test.sh <path-to-cuMES>
set -u

# Resolve the binary to an absolute path BEFORE cd-ing into the scratch dir.
BIN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
fail=0
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp" || exit 2

# Input WITH an unknown key (n_theta): strict must reject, compat must warn.
# ftol 1e-6 sits exactly ON the float policy's documented tolerance floor
# (mixed-float rejects anything below 1e-6 at startup), so the fixtures reach
# the behavior under test in EVERY precision (completion-plan follow-up §3.2).
cat > in_unknown.json <<'EOF'
{"mpol": 2, "ntor": 0, "nfp": 1, "am": [1.0], "aphi": [1.0], "ai": [0.5],
 "n_theta": 6,
 "rbc": [{"n": 0, "m": 1, "value": 1.0}],
 "zbs": [{"n": 0, "m": 1, "value": 0.5}],
 "ns_array": [5], "niter_array": [1], "ftol_array": [1e-6]}
EOF

# Clean input: one 5-surface stage, one iteration (never converges; the run
# exits 1 with the stage-cap message — enough to prove it got past validation).
cat > in_clean.json <<'EOF'
{"mpol": 2, "ntor": 0, "nfp": 1, "am": [1.0], "aphi": [1.0], "ai": [0.5],
 "rbc": [{"n": 0, "m": 1, "value": 1.0}],
 "zbs": [{"n": 0, "m": 1, "value": 0.5}],
 "ns_array": [5], "niter_array": [1], "ftol_array": [1e-6]}
EOF

check() { # name want_exit want_grep want_absent -- cmd...
  local name=$1 want_exit=$2 want_grep=$3 want_absent=$4
  shift 4
  local out
  out=$("$@" 2>&1)
  local got=$?
  if [ "$got" -ne "$want_exit" ]; then
    echo "FAIL $name (exit $got, wanted $want_exit)"
    fail=1
  elif [ -n "$want_grep" ] && ! printf '%s' "$out" | grep -q "$want_grep"; then
    echo "FAIL $name (missing '$want_grep')"
    printf '%s
' "$out" | tail -4
    fail=1
  elif [ -n "$want_absent" ] && printf '%s' "$out" | grep -q "$want_absent"; then
    echo "FAIL $name (unexpected '$want_absent')"
    printf '%s
' "$out" | tail -4
    fail=1
  else
    echo "PASS $name"
  fi
}

check "strict default rejects unknown key" 1       "unknown input key 'n_theta'" "WARNING: unknown input key"       "$BIN" in_unknown.json --output out.bin

check "--compatibility warns and continues" 1       "WARNING: unknown input key" "input validation failed"       "$BIN" --compatibility in_unknown.json --output out.bin

check "input is mandatory" 22       "too few arguments" ""       "$BIN"

check "output defaults under PWD" 1       "cumes-output.bin" ""       "$BIN" in_clean.json
if [ ! -f cumes-output.bin ]; then
  echo "FAIL default output file exists"
  fail=1
else
  echo "PASS default output file exists"
fi

check "output is not positional" 22       "too many arguments" ""       "$BIN" in_clean.json out.bin

check "strict rejects unknown output suffix" 1       "unrecognized output suffix" ""       "$BIN" in_clean.json --output out.weird

if [ "$fail" -ne 0 ]; then
  echo "cli_policy_test: FAILED"
  exit 1
fi
echo "cli_policy_test: all checks passed"
exit 0
