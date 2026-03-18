#!/usr/bin/env bash

set -euo pipefail

UV_BIN="/home/i-zhouyuhan/.local/bin/uv"

uv_cmd() {
  "$UV_BIN" run --active --no-project --offline "$@"
}

configure_cuda_runtime() {
  local cuda_home="$1"
  local extra_lib_dirs="$2"
  local nvrtc_lib_dir="${cuda_home}/targets/x86_64-linux/lib"
  [[ -d "$cuda_home" ]] || {
    echo "CUDA home does not exist: $cuda_home" >&2
    exit 1
  }
  [[ -d "$nvrtc_lib_dir" ]] || {
    echo "CUDA NVRTC lib dir does not exist: $nvrtc_lib_dir" >&2
    exit 1
  }

  unset LD_LIBRARY_PATH
  export CUDA_HOME="$cuda_home"
  export PATH="${CUDA_HOME}/bin:${PATH}"

  local lib_dirs=("$nvrtc_lib_dir")
  IFS=':' read -r -a raw_dirs <<<"$extra_lib_dirs"
  for lib_dir in "${raw_dirs[@]}"; do
    [[ -n "$lib_dir" ]] || continue
    [[ -d "$lib_dir" ]] || continue
    lib_dirs+=("$lib_dir")
  done
  export LD_LIBRARY_PATH="$(IFS=:; echo "${lib_dirs[*]}")"
}

subcommand="${1:-}"
shift || true

case "$subcommand" in
  start-global)
    config_path="${1:?missing config path}"
    uv_cmd tensorcast-cli global start --config="$config_path"
    ;;
  stop-global)
    uv_cmd tensorcast-cli global stop
    ;;
  status-global)
    uv_cmd tensorcast-cli global status
    ;;
  start-daemon)
    config_path="${1:?missing config path}"
    global_store_address="${2:?missing global store address}"
    cuda_home="${3:?missing cuda home}"
    nvidia_lib_dirs="${4:-}"
    configure_cuda_runtime "$cuda_home" "$nvidia_lib_dirs"
    uv_cmd tensorcast-cli daemon start \
      --config="$config_path" \
      --global-store-mode connect \
      --global-store-address "$global_store_address"
    ;;
  stop-daemon)
    uv_cmd tensorcast-cli daemon stop
    ;;
  status-daemon)
    uv_cmd tensorcast-cli daemon status
    ;;
  *)
    echo "unknown subcommand: $subcommand" >&2
    exit 1
    ;;
esac
