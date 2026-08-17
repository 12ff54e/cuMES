#!/usr/bin/env bash
# cli_policy_test.sh — strict-vs-compatibility CLI policy, end to end
# (completion plan step 2.1). Runs the real cuMES binary against tiny inputs:
#
#   - strict schema (default) rejects unknown input keys;
#   - --compatibility warns and continues (vmecpp-style ignore);
#   - strict requires an explicit output path;
#   - --compatibility restores the legacy cumes_state.bin default;
#   - strict rejects an unknown output suffix;
#   - --compatibility restores the legacy unknown-suffix fallback.
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
cat > in_unknown.json <<'EOF'
{"mpol": 2, "ntor": 0, "nfp": 1, "am": [1.0], "aphi": [1.0], "ai": [0.5],
 "n_theta": 6,
 "rbc": [{"n": 0, "m": 1, "value": 1.0}],
 "zbs": [{"n": 0, "m": 1, "value": 0.5}],
 "ns_array": [5], "niter_array": [1], "ftol_array": [1e-14]}
EOF

# Clean input: one 5-surface stage, one iteration (never converges; the run
# exits 1 with the stage-cap message — enough to prove it got past validation).
cat > in_clean.json <<'EOF'
{"mpol": 2, "ntor": 0, "nfp": 1, "am": [1.0], "aphi": [1.0], "ai": [0.5],
 "rbc": [{"n": 0, "m": 1, "value": 1.0}],
 "zbs": [{"n": 0, "m": 1, "value": 0.5}],
 "ns_array": [5], "niter_array": [1], "ftol_array": [1e-14]}
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

check "strict default rejects unknown key" 1       "unknown input key 'n_theta'" "WARNING: unknown input key"       "$BIN" in_unknown.json out.bin

check "--compatibility warns and continues" 1       "WARNING: unknown input key" "input validation failed"       "$BIN" --compatibility in_unknown.json out.bin

check "strict requires explicit output path" 1       "no output path given" ""       "$BIN" in_clean.json

check "--compatibility restores the default output" 1       "writing binary cumes_state.bin" "cuMES: no output path given"       "$BIN" --compatibility in_clean.json
[ -f cumes_state.bin ] && echo "PASS compat default output file exists" \
    || { echo "FAIL compat default output file exists"; fail=1; }

check "strict rejects unknown output suffix" 1       "unrecognized output suffix" ""       "$BIN" in_clean.json out.weird

rm -f cumes_state.bin
check "--compatibility restores unknown-suffix fallback" 1       "writing binary cumes_state.bin" "unrecognized output suffix; pass" \
      "$BIN" --compatibility in_clean.json out.weird
[ -f cumes_state.bin ] && echo "PASS compat suffix fallback wrote cumes_state.bin" \
    || { echo "FAIL compat suffix fallback wrote cumes_state.bin"; fail=1; }

if [ "$fail" -ne 0 ]; then
  echo "cli_policy_test: FAILED"
  exit 1
fi
echo "cli_policy_test: all checks passed"
exit 0
