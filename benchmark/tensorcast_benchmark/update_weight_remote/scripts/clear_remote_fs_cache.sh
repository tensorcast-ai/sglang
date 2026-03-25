#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  clear_remote_fs_cache.sh --path <remote_fs_path> [--cache-dir <dir>]...
USAGE
}

path_arg=""
declare -a cache_dirs=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path)
      path_arg="$2"
      shift 2
      ;;
    --cache-dir)
      cache_dirs+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

[[ -n "$path_arg" ]] || {
  echo "--path is required" >&2
  exit 1
}
[[ -e "$path_arg" ]] || {
  echo "path does not exist: $path_arg" >&2
  exit 1
}

mount_info="$(findmnt -n -T "$path_arg" -o TARGET,SOURCE,FSTYPE)"
read -r mount_target mount_source mount_fstype <<<"$mount_info"

echo "Target path:   $path_arg"
echo "Mount target:  $mount_target"
echo "Mount source:  $mount_source"
echo "Mount fstype:  $mount_fstype"

sync

collect_target_files() {
  if [[ -f "$path_arg" ]]; then
    printf '%s\n' "$path_arg"
    return
  fi
  find "$path_arg" -type f
}

drop_target_file_cache() {
  echo "uv is unavailable; falling back to dd iflag=nocache." >&2
  local dropped_files=0
  while IFS= read -r file_path; do
    [[ -n "$file_path" ]] || continue
    dd if="$file_path" iflag=nocache,count_bytes count=0 of=/dev/null status=none
    dropped_files=$((dropped_files + 1))
  done < <(collect_target_files)
  echo "Dropped page-cache state for $dropped_files file(s) under target path."
}

purged_cache_dirs=0
for cache_dir in "${cache_dirs[@]}"; do
  if [[ ! -d "$cache_dir" ]]; then
    echo "Skipping missing cache dir: $cache_dir"
    continue
  fi
  case "$cache_dir" in
    ""|"/"|"/var"|"/home"|"/tmp")
      echo "Refusing to purge unsafe cache dir: $cache_dir" >&2
      exit 1
      ;;
  esac
  echo "Purging client cache dir: $cache_dir"
  if [[ -w "$cache_dir" ]]; then
    find "$cache_dir" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  else
    sudo -n find "$cache_dir" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  fi
  purged_cache_dirs=1
done

if [[ $purged_cache_dirs -eq 0 ]]; then
  echo "No explicit client cache dir was purged."
fi

if [[ -w /proc/sys/vm/drop_caches ]]; then
  echo "Dropping Linux page cache without sudo."
  echo 3 > /proc/sys/vm/drop_caches
elif sudo -n true >/dev/null 2>&1; then
  echo "Dropping Linux page cache via passwordless sudo."
  if sudo -n sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'; then
    :
  else
    echo "Global drop_caches is unavailable in this environment; falling back to per-file nocache eviction."
    drop_target_file_cache
  fi
else
  echo "Global drop_caches is unavailable; falling back to per-file nocache eviction."
  drop_target_file_cache
fi

echo "Cache clear completed."
