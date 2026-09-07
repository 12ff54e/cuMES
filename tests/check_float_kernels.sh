#!/usr/bin/env bash
# Audit every embedded architecture, including implicit promotions and
# float entry points that accidentally instantiate a double diagnostic.
set -euo pipefail
cuobjdump=$1
shift
for archive in "$@"; do
  [[ -z "$archive" ]] && continue
  "$cuobjdump" --dump-sass "$archive" | awk -v archive="$archive" '
    /Function : / { functions++ }
    /[[:space:]](DADD|DMUL|DFMA|DSETP|DSET|DMNMX|DMMA)[.[:space:]]|[[:space:]](F2F|I2F|F2I)\.[^;]*F64|MUFU\.(RCP64H|RSQ64H)/ {
      if (!bad) print "FP64 instruction in " archive ": " $0 > "/dev/stderr"
      bad = 1
    }
    END {
      if (!functions) {
        print "No compiled GPU functions found in " archive > "/dev/stderr"
        bad = 1
      }
      if (!bad) print archive ": " functions " compiled function sections, no FP64 instructions"
      exit bad
    }'
  # Real-only builds need not embed PTX. When present, inspect parameter and
  # register types too; integer/bitwise 64-bit operations remain valid.
  "$cuobjdump" --dump-ptx "$archive" | awk -v archive="$archive" '
    /\.f64/ {
      if (!bad) print "FP64 type in " archive ": " $0 > "/dev/stderr"
      bad = 1
    }
    END { exit bad }'
done
