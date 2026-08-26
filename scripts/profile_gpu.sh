#!/usr/bin/env bash

set -euo pipefail

readonly DEFAULT_NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5
readonly DEFAULT_CUDA_VERSION=12.9

usage() {
  cat <<'EOF'
Usage:
  scripts/profile_gpu.sh build
  scripts/profile_gpu.sh nsys [solovev|w7x] [gpu]
  scripts/profile_gpu.sh ncu  [solovev|w7x] [gpu] [kernel-regex]

Examples:
  scripts/profile_gpu.sh build
  scripts/profile_gpu.sh nsys w7x 2
  scripts/profile_gpu.sh ncu w7x 2 inverse_accumulate_kernel

Environment overrides:
  NVHPC_ROOT             NVIDIA HPC SDK directory
  CUDA_HOME              CUDA toolkit directory
  CUMES_PROFILE_PASSES   Timed passes (default: 50 for nsys, 10 for ncu)
  CUMES_PROFILE_WARMUP   Warm-up passes (default: 10 for nsys, 2 for ncu)
EOF
}

die() {
  echo "profile_gpu.sh: $*" >&2
  exit 1
}

readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

export NVHPC_ROOT=${NVHPC_ROOT:-$DEFAULT_NVHPC_ROOT}
export CUDA_HOME=${CUDA_HOME:-$NVHPC_ROOT/cuda/$DEFAULT_CUDA_VERSION}
export PATH="$NVHPC_ROOT/compilers/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NVHPC_ROOT/math_libs/$DEFAULT_CUDA_VERSION/targets/x86_64-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

readonly BUILD_DIR="$REPO_ROOT/build-prof"
readonly BENCHMARK="$BUILD_DIR/cumes_benchmark_fixed_iteration"
readonly PROFILE_DIR="$REPO_ROOT/profiles"

validate_config() {
  case "$1" in
    solovev | w7x) ;;
    *) die "configuration must be 'solovev' or 'w7x' (got '$1')" ;;
  esac
}

validate_gpu() {
  [[ "$1" =~ ^[0-9]+$ ]] || die "GPU index must be a non-negative integer (got '$1')"
}

require_benchmark() {
  [[ -x "$BENCHMARK" ]] || die "profiling benchmark not found; run '$0 build' first"
}

build_profile_target() {
  command -v cmake >/dev/null || die "cmake is not available"
  cd "$REPO_ROOT"
  cmake --preset profiling \
    -DCUMES_USE_VACUUM_FIELD=OFF \
    -DCUMES_USE_NETCDF=OFF \
    -DCUMES_USE_HDF5=OFF
  cmake --build build-prof --target cumes cumes_benchmark_fixed_iteration -j8
}

run_nsys() {
  local config=$1
  local gpu=$2
  local passes=${CUMES_PROFILE_PASSES:-50}
  local warmup=${CUMES_PROFILE_WARMUP:-10}
  local stamp report

  validate_config "$config"
  validate_gpu "$gpu"
  require_benchmark
  command -v nsys >/dev/null || die "nsys is not available"

  mkdir -p "$PROFILE_DIR"
  stamp=$(date +%Y%m%d-%H%M%S)
  report="$PROFILE_DIR/${config}-gpu${gpu}-${stamp}"

  cd "$REPO_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=false \
    -o "$report" \
    "$BENCHMARK" --config "$config" --passes "$passes" --warmup "$warmup"

  nsys stats --force-export=true \
    --report cuda_gpu_kern_sum,cuda_api_sum \
    --format table \
    "$report.nsys-rep" | tee "$report-summary.txt"

  echo "Nsight Systems report: $report.nsys-rep"
  echo "Text summary:          $report-summary.txt"
}

run_ncu() {
  local config=$1
  local gpu=$2
  local kernel_regex=${3:-}
  local passes=${CUMES_PROFILE_PASSES:-10}
  local warmup=${CUMES_PROFILE_WARMUP:-2}
  local stamp report ncu

  validate_config "$config"
  validate_gpu "$gpu"
  require_benchmark
  command -v sudo >/dev/null || die "sudo is not available"

  if [[ -z "$kernel_regex" ]]; then
    if [[ "$config" == w7x ]]; then
      kernel_regex=inverse_accumulate_kernel
    else
      kernel_regex=pcr_solve_kernel
    fi
  fi

  ncu="$NVHPC_ROOT/compilers/bin/ncu"
  [[ -x "$ncu" ]] || die "ncu is not available at $ncu"

  mkdir -p "$PROFILE_DIR"
  stamp=$(date +%Y%m%d-%H%M%S)
  report="$PROFILE_DIR/${config}-gpu${gpu}-${kernel_regex}-${stamp}"

  cd "$REPO_ROOT"
  sudo env CUDA_VISIBLE_DEVICES="$gpu" "$ncu" \
    --set full \
    --kernel-name "regex:$kernel_regex" \
    --launch-count 1 \
    -o "$report" \
    "$BENCHMARK" --config "$config" --passes "$passes" --warmup "$warmup"

  sudo chown "$(id -u):$(id -g)" "$report.ncu-rep"
  echo "Nsight Compute report: $report.ncu-rep"
}

main() {
  local command=${1:-help}

  case "$command" in
    build)
      [[ $# -eq 1 ]] || die "the build command takes no arguments"
      build_profile_target
      ;;
    nsys)
      [[ $# -le 3 ]] || die "too many arguments for nsys"
      run_nsys "${2:-w7x}" "${3:-2}"
      ;;
    ncu)
      [[ $# -le 4 ]] || die "too many arguments for ncu"
      run_ncu "${2:-w7x}" "${3:-2}" "${4:-}"
      ;;
    help | -h | --help)
      usage
      ;;
    *)
      usage >&2
      die "unknown command '$command'"
      ;;
  esac
}

main "$@"
