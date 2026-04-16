#!/usr/bin/env bash

set -euo pipefail

subcommand="${1:-}"
shift || true

case "$subcommand" in
  start)
    mooncake_bin="${1:?missing mooncake binary}"
    log_path="${2:?missing log path}"
    metadata_port="${3:?missing metadata port}"
    master_port="${4:?missing master port}"
    eviction_ratio="${5:?missing eviction ratio}"
    mkdir -p "$(dirname "$log_path")"
    nohup "$mooncake_bin" \
      --enable_http_metadata_server=true \
      --http_metadata_server_port="${metadata_port}" \
      --eviction_high_watermark_ratio="${eviction_ratio}" \
      --port="${master_port}" \
      >"${log_path}" 2>&1 < /dev/null &
    echo "$!"
    ;;
  *)
    echo "unknown subcommand: $subcommand" >&2
    exit 1
    ;;
esac
