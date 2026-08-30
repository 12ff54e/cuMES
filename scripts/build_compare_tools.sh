#!/usr/bin/env bash
# Build the four standalone C++ comparison tools without configuring cuMES.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUTPUT_DIR=${1:-"$SCRIPT_DIR/../build/compare-tools"}
CXX=${CXX:-g++}

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

flags=(-std=c++20 -O2 -Wall -Wextra -Wpedantic)
hdf5_flags=(-DCUMES_COMPARE_HAVE_HDF5=1)
for tool in compare_states compare_runs compare_bitwise; do
  "$CXX" "${flags[@]}" "$SCRIPT_DIR/$tool.cpp" -o "$OUTPUT_DIR/$tool"
done

if command -v "${H5CXX:-h5c++}" >/dev/null 2>&1; then
  # Some h5c++ wrappers retain an intermediate compare_wout.o in their CWD.
  # Keep it inside the requested output directory and remove it afterwards.
  (cd "$OUTPUT_DIR" && \
    "${H5CXX:-h5c++}" "${flags[@]}" "${hdf5_flags[@]}" \
      "$SCRIPT_DIR/compare_wout.cpp" \
      -o compare_wout && rm -f compare_wout.o)
elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists hdf5; then
  # pkg-config intentionally expands to individual compiler/linker arguments.
  # shellcheck disable=SC2046
  "$CXX" "${flags[@]}" "${hdf5_flags[@]}" $(pkg-config --cflags hdf5) \
    "$SCRIPT_DIR/compare_wout.cpp" $(pkg-config --libs hdf5) \
    -o "$OUTPUT_DIR/compare_wout"
else
  echo "warning: HDF5 compiler metadata not found; compare_wout will be a stub" \
    >&2
  "$CXX" "${flags[@]}" "$SCRIPT_DIR/compare_wout.cpp" \
    -o "$OUTPUT_DIR/compare_wout"
fi

echo "comparison tools written to $OUTPUT_DIR"
