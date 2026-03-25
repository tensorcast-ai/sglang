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
  reset-runtime-state)
    python - <<'PY'
from __future__ import annotations

from pathlib import Path

from tensorcast.cli_utils.paths import current_global_session_path, runtime_state_path

for raw_path in (runtime_state_path(), current_global_session_path()):
    path = Path(raw_path)
    try:
        path.unlink()
        print(f"removed {path}")
    except FileNotFoundError:
        pass
print("runtime state reset")
PY
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
  wait-daemon-ready)
    daemon_address="${1:?missing daemon address}"
    timeout_s="${2:?missing timeout seconds}"
    interval_s="${3:?missing interval seconds}"
    python - "$daemon_address" "$timeout_s" "$interval_s" <<'PY'
from __future__ import annotations

import sys

from tensorcast.cli_utils.health import wait_for_daemon

address = sys.argv[1]
timeout_s = float(sys.argv[2])
interval_s = float(sys.argv[3])
if not wait_for_daemon(address, timeout=timeout_s, interval=interval_s):
    raise SystemExit(f"daemon did not reach rpc-ready state within {timeout_s}s: {address}")
print(f"daemon rpc ready: {address}")
PY
    ;;
  *)
    echo "unknown subcommand: $subcommand" >&2
    exit 1
    ;;
esac
