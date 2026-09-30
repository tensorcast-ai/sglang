---
slug: sglang-tensorcast-unified-full-implementation-plan
title: TensorCast Unified FULL HiCache Backend Implementation Plan
links:
  design: ./design.md
areas: ["hicache", "storage", "tensorcast", "tests", "docs"]
related_code:
  - python/sglang/srt/mem_cache/storage/tensorcast_store/
  - python/sglang/srt/mem_cache/storage/backend_factory.py
  - python/sglang/srt/mem_cache/managers/cache_controller.py
  - python/sglang/srt/mem_cache/pool_host/common.py
  - python/sglang/srt/server_args.py
last_updated: 2026-09-05
---

# Objective

Implement the accepted [TensorCast HiCache backend design](./design.md) as a
thin SGLang adapter over TensorCast's public
`RegionBackedArtifactSession`. The initial implementation supports only the
Unified radix cache's FULL v1 page API, with:

- allocator-backed direct L2/L3 transfer as the default;
- scratch-slab transfer as an explicitly configured alternative;
- MHA K/V, asymmetric MHA K/V, and MLA KV fragment schemas;
- one process-scoped Session and one prestarted StoreDaemon attachment per
  SGLang rank; and
- rank-local permanent L3 disablement after a fatal runtime adapter or Session
  error, without terminating the SGLang process.

This plan is execution-oriented. Every phase has a unit-test gate and an E2E
gate. A phase is complete only after its implementation subtasks and both gates
pass and are checked off.

# Normative Sources And Scope

- `design.md` is the normative SGLang contract.
- TensorCast design and implementation `0122` define the public Session
  contract. SGLang must not depend on its private implementation.
- `/mnt/data/tot/todo.md` is the command-style reference for preflight, service
  startup, long-running SGLang processes, health checks, inference requests,
  log collection, and cleanup. Its deprecated TensorCast API and configuration
  fields are not implementation references.
- The deprecated SGLang PR branch is a source of operational experience and
  page-key conventions only. Do not restore its raw protobuf client, region
  bindings, special HostPool layout, allocator cleanup, or slot-generation
  state machine.

The initial implementation is limited to:

- Linux and the CUDA hardware backend;
- `--hicache-host-memory-mode=cache`;
- `page_first` or `page_first_direct`;
- synchronous `batch_exists()`, `batch_get_v1()`, and `batch_set_v1()`; and
- a TensorCast backend selected before HostPool construction.

Do not expand the PR to storage v2, Mamba/recurrent state, draft pools, DSA,
split heads, attention CP, TP-LCM, `buffer_only`, runtime recovery, daemon
supervision, or generic HostPool lifecycle changes.

# Implementation Rules

- Except for the thin integration points allowed by `design.md`, all
  TensorCast-specific implementation, tests, fixtures, and E2E helpers stay in
  `python/sglang/srt/mem_cache/storage/tensorcast_store/`.
- Keep only two substantive production modules:
  `host_allocator.py` and `tensorcast_store.py`. Small private dataclasses and
  helpers remain in those modules.
- Import TensorCast lazily and only from `tensorcast.api.store`.
- Do not construct protobufs, canonical artifact IDs, region layouts, region
  handles, storage IDs, slot indices, slot generations, or operation IDs in
  SGLang.
- Do not add TensorCast-specific fields or callbacks to HostPool classes or
  Unified tree nodes.
- Do not add a production or test-only registry reset. Any test that attaches,
  claims, fails, or terminates a process Session runs in a fresh non-forked
  Python child process.
- Unit tests use fake public Session objects and
  `TENSORCAST_CUDA_BACKEND=fake`. They do not require a daemon.
- Real-daemon tests attach only after the Global Store and StoreDaemon are
  ready. SGLang never starts or supervises those services.
- Every Python command uses `/mnt/data/tot/.venv`, activated explicitly, and
  runs through `uv run --active --no-sync`.
- Do not patch TensorCast daemon or unrelated SGLang behavior to make an E2E
  gate pass. Record an environment or pre-existing failure and stop for
  discussion.
- The known, pre-existing
  `grpc_service_impl_cpu_memfd_e2e_test` failure is not a gate for this work.

# Planned Files

```text
python/sglang/srt/mem_cache/storage/tensorcast_store/
├── design.md
├── plan.md
├── README.md
├── host_allocator.py
├── tensorcast_store.py
├── configs/
│   ├── global_store_config.yaml
│   └── store_daemon_config.yaml
├── test_tensorcast_host_allocator.py
├── test_tensorcast_store.py
└── test_tensorcast_e2e.py
```

`test_tensorcast_e2e.py` is a colocated, environment-gated test driver. Its
parent pytest process must not attach a Session. Each stateful scenario starts
a new interpreter, completes the entire attach/use/terminate lifecycle there,
and reports through exit status and captured diagnostics.

The only expected production changes outside the package are:

- `server_args.py`: add the `tensorcast` backend choice;
- `storage/backend_factory.py`: lazy construction;
- `managers/cache_controller.py`: select existing synchronous v1 zero-copy
  callbacks;
- `pool_host/common.py`: one lazy allocator-factory branch;
- `arg_groups/hicache_hook.py`: only the minimum generic selection validation
  that cannot remain in the backend factory; and
- optional dependency metadata if packaging requires it.

# Execution Order

Execute phases in order. Do not begin the next phase while the current phase's
validation gate is red.

- [x] Phase 1: Package boundary, configuration, and lazy selection
- [x] Phase 2: Early Session attachment, thin registry, and host allocator
- [x] Phase 3: FULL registration, fragments, identity inputs, and exists
- [x] Phase 4: Scratch get/put vertical slice
- [x] Phase 5: Allocator-backed direct mode and multi-region transfer
- [x] Phase 6: Failure containment, concurrency, and terminal lifecycle
- [ ] Phase 7: Integration closure, documentation, and final acceptance

# Shared Validation Contract

## Common Python environment

Run every gate from a clean shell using the repository virtual environment:

```bash
cd /mnt/data/tot
source /mnt/data/tot/.venv/bin/activate

export SGLANG_ROOT=/mnt/data/tot/thirdparty/sglang
export TENSORCAST_ROOT=/mnt/data/tot/thirdparty/tensorcast
export SGLANG_TC_TEST_DIR="${SGLANG_ROOT}/python/sglang/srt/mem_cache/storage/tensorcast_store"
export PYTHONPATH="${SGLANG_ROOT}/python:${TENSORCAST_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

test -x /mnt/data/tot/.venv/bin/python
test -d /mnt/data/models/Qwen3-14B/
uv run --active --no-sync python -c \
  "import sglang; import tensorcast; from tensorcast.api.store import RegionBackedArtifactSession"
```

Unit-test commands additionally set:

```bash
export TENSORCAST_CUDA_BACKEND=fake
```

Real-daemon and model E2E commands instead use CUDA 13 and must not inherit the
fake backend:

```bash
unset TENSORCAST_CUDA_BACKEND
export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES=0
```

## Unit-test conventions

- Pure config, fragment, identity-input, folding, and import tests may run in
  the pytest process.
- A fake Session implements only the documented public
  `tensorcast.api.store` surface.
- Stateful lifecycle cases use `subprocess` or
  `multiprocessing.get_context("spawn")`, never `fork`.
- Tests assert public specs, spans, masks, exceptions, and call counts. They do
  not inspect private TensorCast registries or generated protobufs.
- Each phase runs its narrow selector. Phase 7 runs the whole colocated suite.

## Real-daemon E2E service fixture

The adapter E2E cases in Phases 2-6 assume services started by the operator in
this order. Keep these commands in one controlling shell so all exported paths
and session names remain stable:

```bash
cd /mnt/data/tot
source /mnt/data/tot/.venv/bin/activate

unset TENSORCAST_CUDA_BACKEND
export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
ulimit -n 65535
export SGLANG_ROOT=/mnt/data/tot/thirdparty/sglang
export TENSORCAST_ROOT=/mnt/data/tot/thirdparty/tensorcast
export RUN_DIR
RUN_DIR="$(mktemp -d /tmp/sglang_tensorcast_unified_full.XXXXXX)"
export TENSORCAST_HOME="${RUN_DIR}/tensorcast_home"
export TC_CONFIG_DIR="${SGLANG_ROOT}/python/sglang/srt/mem_cache/storage/tensorcast_store/configs"
export TC_GLOBAL_CONFIG="${TC_CONFIG_DIR}/global_store_config.yaml"
export TC_DAEMON_CONFIG="${TC_CONFIG_DIR}/store_daemon_config.yaml"
export TC_GLOBAL_SESSION=sglang-tc-unified-full-global
export TC_DAEMON_SESSION=sglang-tc-unified-full-daemon
export SGLANG_TENSORCAST_E2E=1
export SGLANG_TENSORCAST_DAEMON_ADDRESS=127.0.0.1:50052

mkdir -p "${TENSORCAST_HOME}"
test -f "${TC_GLOBAL_CONFIG}"
test -f "${TC_DAEMON_CONFIG}"
test -x "${TENSORCAST_ROOT}/bazel-bin/daemon/tensorcast_daemon"
command -v tensorcast-cli

ss -ltnp \
  | rg ':30000\b|:30001\b|:62311\b|:62321\b|:50051\b|:50052\b|:18000\b|:65090\b' \
  || true

cd "${TENSORCAST_ROOT}"
uv run --active --no-sync tensorcast-cli global start \
  --config "${TC_GLOBAL_CONFIG}" \
  --gs-session "${TC_GLOBAL_SESSION}" \
  --json \
  | tee "${RUN_DIR}/tensorcast_global_start.json"

uv run --active --no-sync tensorcast-cli daemon start \
  --config "${TC_DAEMON_CONFIG}" \
  --global-store-mode connect \
  --global-store-address 127.0.0.1:50051 \
  --session "${TC_DAEMON_SESSION}" \
  --json \
  | tee "${RUN_DIR}/tensorcast_daemon_start.json"

for iteration in $(seq 1 60); do
  global_ok=0
  daemon_ok=0
  uv run --active --no-sync tensorcast-cli global status \
    --gs-session "${TC_GLOBAL_SESSION}" --json \
    > "${RUN_DIR}/tensorcast_global_status.json" && global_ok=1 || true
  uv run --active --no-sync tensorcast-cli daemon status \
    --session "${TC_DAEMON_SESSION}" --json \
    > "${RUN_DIR}/tensorcast_daemon_status.json" && daemon_ok=1 || true
  if test "${global_ok}" = 1 && test "${daemon_ok}" = 1 \
    && ss -ltnp | rg ':50051\b' >/dev/null \
    && ss -ltnp | rg ':50052\b' >/dev/null \
    && ss -ltnp | rg ':65090\b' >/dev/null; then
    echo "TENSORCAST_READY iteration=${iteration}"
    break
  fi
  sleep 2
done

uv run --active --no-sync tensorcast-cli global status \
  --gs-session "${TC_GLOBAL_SESSION}" --json
uv run --active --no-sync tensorcast-cli daemon status \
  --session "${TC_DAEMON_SESSION}" --json
ss -ltnp | rg ':50051\b' >/dev/null
ss -ltnp | rg ':50052\b' >/dev/null
ss -ltnp | rg ':65090\b' >/dev/null
```

These TensorCast service configs are derived from the last validated
configuration on the deprecated integration branch. The daemon memory budgets
are right-sized for the single-GPU local validation host while retaining the
same service and protocol settings. The capability-token settings in the
checked-in daemon config are runtime validation inputs: its fixed
non-production secret is sufficient for local payload-ref validation, but
production deployments must replace it. Do not rewrite the source config
during a test run. The daemon must inherit `RLIMIT_NOFILE >= 65535`: one TP=8
10K-token publish can keep more than 1,024 UMA artifact memfds live, so the
common shell raises the soft limit before either service is launched.

All real-daemon pytest cases use:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
unset TENSORCAST_CUDA_BACKEND
export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q -s \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_e2e.py
```

## Qwen3-14B TP=1 correctness lane

The model gate always uses:

- model: `/mnt/data/models/Qwen3-14B/`;
- tensor parallel size: `1`;
- one GPU selected with `CUDA_VISIBLE_DEVICES=0`;
- `--hicache-host-memory-mode cache`;
- a HostPool sized to the device KV pool (`--hicache-ratio 1`);
- page size 64; and
- sequential publisher and consumer rank processes so one GPU is sufficient.

The publisher and consumer use the same daemon, namespace, model identity,
model version, layout, and page size, but have different PIDs. A passing
consumer response must report a nonzero L3 storage hit. Repeating the publisher
request three times is intentional because write-through backup is normally
triggered by a cache hit.

For the no-TensorCast baseline, create a separate run directory and start one
server without `--hicache-storage-backend` or backend extra config:

```bash
cd /mnt/data/tot
source /mnt/data/tot/.venv/bin/activate
unset TENSORCAST_CUDA_BACKEND
export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export BASELINE_RUN_DIR
BASELINE_RUN_DIR="$(mktemp -d /tmp/sglang_hicache_baseline.XXXXXX)"

setsid bash -lc '
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
unset TENSORCAST_CUDA_BACKEND
export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=python \
exec uv run --active --no-sync python -m sglang.launch_server \
  --model-path /mnt/data/models/Qwen3-14B/ \
  --tp-size 1 \
  --host 127.0.0.1 \
  --port 30000 \
  --nccl-port 62311 \
  --trust-remote-code \
  --page-size 64 \
  --max-total-tokens 8192 \
  --disable-cuda-graph \
  --enable-hierarchical-cache \
  --hicache-host-memory-mode cache \
  --hicache-ratio 1
' > "${BASELINE_RUN_DIR}/server.log" 2>&1 &
echo $! > "${BASELINE_RUN_DIR}/server.pid"

for iteration in $(seq 1 180); do
  if curl -sf http://127.0.0.1:30000/health_generate >/dev/null; then
    break
  fi
  baseline_pid="$(cat "${BASELINE_RUN_DIR}/server.pid")"
  if ! kill -0 "${baseline_pid}" 2>/dev/null; then
    tail -n 200 "${BASELINE_RUN_DIR}/server.log"
    exit 1
  fi
  sleep 5
done

curl -sf http://127.0.0.1:30000/health_generate
curl -sf http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Write one short sentence about distributed inference.",
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 32}
  }' \
  | tee "${BASELINE_RUN_DIR}/response.json"

uv run --active --no-sync python - <<'PY'
import json
import os
from pathlib import Path

response = json.loads(
    Path(os.environ["BASELINE_RUN_DIR"], "response.json").read_text(
        encoding="utf-8"
    )
)
assert response.get("text"), "baseline returned empty generated text"
PY

baseline_pid="$(cat "${BASELINE_RUN_DIR}/server.pid")"
baseline_pgid="$(ps -o pgid= -p "${baseline_pid}" | tr -d ' ')"
test -n "${baseline_pgid}"
kill -TERM -- "-${baseline_pgid}"
for iteration in $(seq 1 60); do
  kill -0 "${baseline_pid}" 2>/dev/null || break
  sleep 2
done
if kill -0 "${baseline_pid}" 2>/dev/null; then
  kill -KILL -- "-${baseline_pgid}" 2>/dev/null || true
  echo "baseline server did not exit gracefully"
  exit 1
fi
```

Inspect `server.log` for unexpected failures after the process exits. A forced
KILL makes the baseline gate fail even though the cleanup command still
releases the GPU.

Generate one backend config per transfer mode. The allocator config
deliberately omits `transfer_mode` to prove that allocator is the default:

```bash
cd /mnt/data/tot
source /mnt/data/tot/.venv/bin/activate

export TC_E2E_NAMESPACE="sglang_tensorcast_${USER}_$(basename "${RUN_DIR}")"
export TC_E2E_MODEL_VERSION="qwen3_14b_$(basename "${RUN_DIR}")"

uv run --active --no-sync python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
common = {
    "prefetch_threshold": 64,
    "prefetch_timeout_base": 10.0,
    "prefetch_timeout_per_ki_token": 2.0,
    "hicache_storage_pass_prefix_keys": False,
}

for mode in ("scratch", "allocator"):
    tensorcast = {
        "daemon_address": "127.0.0.1:50052",
        "namespace": os.environ["TC_E2E_NAMESPACE"],
        "model_id": "Qwen3-14B",
        "model_version": os.environ["TC_E2E_MODEL_VERSION"],
        "session_name_prefix": "sglang-e2e",
        "region_name_prefix": "sglang-tensorcast-e2e",
        "exists_timeout_s": 30.0,
        "transfer_timeout_s": None,
        "scratch": {"capacity_bytes": 4294967296},
    }
    if mode == "scratch":
        tensorcast["transfer_mode"] = "scratch"
    payload = {**common, "tensorcast": tensorcast}
    (run_dir / f"sglang_tensorcast_{mode}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
PY
```

The literal `4294967296` configures a 4 GiB scratch arena in bytes. For the
checked-in Qwen3-14B config at TP=1, BF16 KV, page size 64, and a maximum
128-page storage batch, the expected FULL K+V bound is 2.5 GiB. The backend
must still calculate and validate the exact requirement from the constructed
HostPool and registered fragment schema; the test does not substitute this
estimate for registration-time validation.

For each server process export `TC_E2E_CONFIG`, `SGLANG_PORT`, `NCCL_PORT`, and
`SGLANG_LOG`, then launch:

```bash
export TC_E2E_CONFIG="${RUN_DIR}/sglang_tensorcast_allocator.json"
export SGLANG_PORT=30000
export NCCL_PORT=62311
export SGLANG_LOG="${RUN_DIR}/publisher.log"

setsid bash -lc '
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
unset TENSORCAST_CUDA_BACKEND
export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export UV_CACHE_DIR="${RUN_DIR}/uv_cache"
export FLASHINFER_WORKSPACE_BASE="${RUN_DIR}/flashinfer_workspace"
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
exec uv run --active --no-sync python -m sglang.launch_server \
  --model-path /mnt/data/models/Qwen3-14B/ \
  --tp-size 1 \
  --host 127.0.0.1 \
  --port "${SGLANG_PORT}" \
  --nccl-port "${NCCL_PORT}" \
  --trust-remote-code \
  --page-size 64 \
  --max-total-tokens 8192 \
  --disable-cuda-graph \
  --enable-hierarchical-cache \
  --hicache-host-memory-mode cache \
  --hicache-ratio 1 \
  --hicache-storage-backend tensorcast \
  --hicache-storage-backend-extra-config "@${TC_E2E_CONFIG}" \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy wait_complete \
  --log-requests \
  --log-requests-level 1 \
  --log-requests-format json
' > "${SGLANG_LOG}" 2>&1 &
echo $! > "${SGLANG_LOG}.pid"
```

Wait for readiness and fail early if the launcher exits:

```bash
for iteration in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${SGLANG_PORT}/health_generate" >/dev/null; then
    echo "SGLANG_READY iteration=${iteration}"
    break
  fi
  server_pid="$(cat "${SGLANG_LOG}.pid")"
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -n 200 "${SGLANG_LOG}"
    exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:${SGLANG_PORT}/health_generate"
```

Create the deterministic request and publish from the first rank:

```bash
uv run --active --no-sync python - <<'PY'
import json
import os
from pathlib import Path

prompt = (
    "TensorCast Unified FULL Qwen3-14B TP1 correctness validation. "
    "This deterministic paragraph is repeated to create complete HiCache pages. "
    * 96
)
payload = {
    "text": prompt,
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 8},
}
Path(os.environ["RUN_DIR"], "request.json").write_text(
    json.dumps(payload), encoding="utf-8"
)
PY

for request_index in 1 2 3; do
  curl -sf "http://127.0.0.1:${SGLANG_PORT}/generate" \
    -H 'Content-Type: application/json' \
    -d @"${RUN_DIR}/request.json" \
    | tee "${RUN_DIR}/publisher_request_${request_index}.json"
done
sleep 30
```

Stop the publisher and wait for the GPU/process to be released. Use TERM so
the normal SGLang shutdown path can close storage workers and the Store. KILL
is cleanup-only and makes the gate fail if graceful exit did not finish:

```bash
publisher_pid="$(cat "${SGLANG_LOG}.pid")"
publisher_pgid="$(ps -o pgid= -p "${publisher_pid}" | tr -d ' ')"
test -n "${publisher_pgid}"
kill -TERM -- "-${publisher_pgid}"

for iteration in $(seq 1 60); do
  kill -0 "${publisher_pid}" 2>/dev/null || break
  sleep 2
done
if kill -0 "${publisher_pid}" 2>/dev/null; then
  kill -KILL -- "-${publisher_pgid}" 2>/dev/null || true
  echo "publisher did not exit gracefully"
  exit 1
fi
```

Launch the consumer with the same config on port 30001/NCCL port 62321, submit
the identical request once, and save the response as
`${RUN_DIR}/consumer_shared_hit.json`. Validate the response:

```bash
export SGLANG_PORT=30001
export NCCL_PORT=62321
export SGLANG_LOG="${RUN_DIR}/consumer.log"

# Repeat the exact setsid launch block and readiness loop above with these
# exported values. Set TC_E2E_CONFIG to the consumer's selected mode first.

curl -sf "http://127.0.0.1:${SGLANG_PORT}/generate" \
  -H 'Content-Type: application/json' \
  -d @"${RUN_DIR}/request.json" \
  | tee "${RUN_DIR}/consumer_shared_hit.json"

uv run --active --no-sync python - <<'PY'
import json
import os
from pathlib import Path

response_path = Path(os.environ["RUN_DIR"], "consumer_shared_hit.json")
response = json.loads(response_path.read_text(encoding="utf-8"))
assert response.get("text"), "consumer returned empty generated text"

meta = response["meta_info"]
details = meta.get("cached_tokens_details") or {}
assert meta.get("cached_tokens", 0) > 0, "expected cached_tokens > 0"
assert details.get("storage", 0) > 0, "expected TensorCast L3 storage hit"
assert details.get("storage_backend") == "TensorcastStore"
print(json.dumps(details, indent=2))
PY
```

For scratch-only coverage, set both rank configs to scratch. For allocator-only
coverage, set both to allocator. For the final cross-mode gate, publish with
scratch and consume with allocator while keeping all artifact identity inputs
and `page_first_direct` unchanged.

## Qwen3-32B TP=2 concurrent-instance reuse lane

This lane validates that two live SGLang instances can attach four independent
rank processes to one TensorCast daemon and reuse the same rank-sharded L3
artifacts. Run it once with `TC_MULTI_MODE=scratch` for Phase 4 and once with
`TC_MULTI_MODE=allocator` for Phase 5.

The lane uses:

- model `/home/models/Qwen3-32B`;
- instance A on `CUDA_VISIBLE_DEVICES=0,1`, HTTP port `30000`, and NCCL port
  `62311`;
- instance B on `CUDA_VISIBLE_DEVICES=2,3`, HTTP port `30001`, and NCCL port
  `62321`;
- tensor parallel size 2, page size 64, and `--max-total-tokens 16384`;
- the same already-running daemon at `127.0.0.1:50052` for both instances;
- one shared namespace, model identity, model version, layout, and transfer
  mode so A TP rank 0/1 artifacts are discoverable by B TP rank 0/1; and
- different `session_name_prefix` and `region_name_prefix` values for A and B.
  The adapter appends `rank0of2` or `rank1of2`, making all four live Session and
  region resource names distinct without changing artifact identity.

The daemon may retain the single-GPU visibility used by the common service
fixture: this integration exposes CPU shared-memory regions and does not
require the daemon to mirror the four SGLang worker GPUs. Both SGLang
instances must remain alive throughout the publish/consume part of the test.

Create the mode-specific A/B configs. Across the two Phase 4/5 runs this
produces four configs. Keep identity fields deliberately short so the resulting
canonical TensorCast artifact ID remains within its 200-byte limit. The
allocator configs omit `transfer_mode` to validate the default:

```bash
cd /mnt/data/tot
source /mnt/data/tot/.venv/bin/activate

test -d /home/models/Qwen3-32B
test "${TC_MULTI_MODE}" = scratch -o "${TC_MULTI_MODE}" = allocator
export RUN_DIR TC_MULTI_MODE

uv run --active --no-sync python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
mode = os.environ["TC_MULTI_MODE"]
run_token = hashlib.sha256(str(run_dir).encode()).hexdigest()[:8]
common = {
    "prefetch_threshold": 64,
    "prefetch_timeout_base": 10.0,
    "prefetch_timeout_per_ki_token": 2.0,
    "hicache_storage_pass_prefix_keys": False,
}

for instance in ("a", "b"):
    tensorcast = {
        "daemon_address": "127.0.0.1:50052",
        "namespace": f"q32t2{mode[0]}{run_token}",
        "model_id": "Qwen3-32B",
        "model_version": f"e2e{run_token}",
        "session_name_prefix": f"q32t2-{mode[0]}-{instance}",
        "region_name_prefix": f"q32t2-{mode[0]}-{instance}",
        "exists_timeout_s": 30.0,
        "transfer_timeout_s": None,
        "scratch": {"capacity_bytes": 1073741824},
    }
    if mode == "scratch":
        tensorcast["transfer_mode"] = "scratch"
    payload = {**common, "tensorcast": tensorcast}
    path = run_dir / f"qwen32b_tp2_{mode}_{instance}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
```

The 1 GiB scratch capacity is exact for one Qwen3-32B BF16 TP=2 rank at page
size 64 and the 128-page storage batch limit: each rank-local FULL K+V page is
8 MiB. Registration must independently calculate and validate this bound.
Using 1 GiB instead of the older single-instance 4 GiB arena also leaves
daemon stable capacity for four simultaneous rank Sessions and the published
artifacts.

Start A first and wait until it is ready, then start B and wait until it is
ready. A stays live while B starts, so readiness proves both TP=2 instances and
all four rank Sessions coexist on the same daemon:

```bash
launch_qwen32_tp2_instance() {
  instance="$1"
  cuda_devices="$2"
  http_port="$3"
  nccl_port="$4"
  config="${RUN_DIR}/qwen32b_tp2_${TC_MULTI_MODE}_${instance}.json"
  log="${RUN_DIR}/qwen32b_tp2_${TC_MULTI_MODE}_${instance}.log"

  setsid env \
    CUDA_VISIBLE_DEVICES="${cuda_devices}" \
    SGLANG_INSTANCE_PORT="${http_port}" \
    SGLANG_INSTANCE_NCCL_PORT="${nccl_port}" \
    SGLANG_INSTANCE_CONFIG="${config}" \
    SGLANG_CACHE_DIR="${RUN_DIR}/sglang_cache_${instance}" \
    TENSORCAST_HOME="${TENSORCAST_HOME}" \
    bash -lc '
      cd /mnt/data/tot/thirdparty/sglang
      source /mnt/data/tot/.venv/bin/activate
      ulimit -n 65535
      unset TENSORCAST_CUDA_BACKEND
      export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      export PYTHONPATH="python:/mnt/data/tot/thirdparty/tensorcast${PYTHONPATH:+:${PYTHONPATH}}"
      exec uv run --active --no-sync python -m sglang.launch_server \
        --model-path /home/models/Qwen3-32B \
        --tp-size 2 \
        --host 127.0.0.1 \
        --port "${SGLANG_INSTANCE_PORT}" \
        --nccl-port "${SGLANG_INSTANCE_NCCL_PORT}" \
        --trust-remote-code \
        --page-size 64 \
        --max-total-tokens 16384 \
        --disable-cuda-graph \
        --enable-hierarchical-cache \
        --hicache-host-memory-mode cache \
        --hicache-ratio 1 \
        --hicache-storage-backend tensorcast \
        --hicache-storage-backend-extra-config "@${SGLANG_INSTANCE_CONFIG}" \
        --hicache-mem-layout page_first_direct \
        --hicache-io-backend direct \
        --hicache-write-policy write_through \
        --hicache-storage-prefetch-policy wait_complete \
        --log-requests \
        --log-requests-level 1 \
        --log-requests-format json
    ' > "${log}" 2>&1 &
  echo $! > "${log}.pid"
}

wait_qwen32_tp2_instance() {
  instance="$1"
  http_port="$2"
  log="${RUN_DIR}/qwen32b_tp2_${TC_MULTI_MODE}_${instance}.log"
  server_pid="$(cat "${log}.pid")"
  for iteration in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${http_port}/health_generate" >/dev/null; then
      echo "QWEN32_TP2_${instance}_READY iteration=${iteration}"
      return 0
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -n 240 "${log}"
      return 1
    fi
    sleep 5
  done
  tail -n 240 "${log}"
  return 1
}

launch_qwen32_tp2_instance a 0,1 30000 62311
wait_qwen32_tp2_instance a 30000
launch_qwen32_tp2_instance b 2,3 30001 62321
wait_qwen32_tp2_instance b 30001
```

Generate a deterministic prompt whose tokenizer length is between 10,000 and
10,240 tokens. The upper bound leaves ample room for the generated tokens
under `--max-total-tokens 16384`:

```bash
uv run --active --no-sync python - <<'PY'
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

model_path = "/home/models/Qwen3-32B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
unit = (
    "TensorCast concurrent TP2 reuse validates process-scoped regions and "
    "rank-sharded FULL KV artifacts across independent inference instances. "
)
unit_tokens = len(tokenizer.encode(unit, add_special_tokens=False))
repeat_count = (10000 + unit_tokens - 1) // unit_tokens
prompt = unit * repeat_count
prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
while prompt_tokens < 10000:
    prompt += unit
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
assert 10000 <= prompt_tokens <= 10240, prompt_tokens

payload = {
    "text": prompt,
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 8},
}
Path(os.environ["RUN_DIR"], "qwen32b_tp2_request.json").write_text(
    json.dumps(payload), encoding="utf-8"
)
print(f"qwen32b_tp2_prompt_tokens={prompt_tokens}")
PY
```

Send the request to A once, wait for its asynchronous write-through
`batch_put_from()` calls to finish, and then send the identical request to the
already-running B. Do not flush A or B and do not stop A before B consumes:

```bash
curl --max-time 600 -sf http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d @"${RUN_DIR}/qwen32b_tp2_request.json" \
  | tee "${RUN_DIR}/qwen32b_tp2_${TC_MULTI_MODE}_a_response.json"

sleep 60

curl --max-time 600 -sf http://127.0.0.1:30001/generate \
  -H 'Content-Type: application/json' \
  -d @"${RUN_DIR}/qwen32b_tp2_request.json" \
  | tee "${RUN_DIR}/qwen32b_tp2_${TC_MULTI_MODE}_b_response.json"
```

Validate deterministic correctness and a substantial page-aligned L3 hit in
B. A 10,000-token request contains at least 9,984 complete 64-token pages, so
this also proves transfer across SGLang's 128-page HiCache storage-batch
boundary:

```bash
uv run --active --no-sync python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
mode = os.environ["TC_MULTI_MODE"]
a = json.loads((run_dir / f"qwen32b_tp2_{mode}_a_response.json").read_text())
b = json.loads((run_dir / f"qwen32b_tp2_{mode}_b_response.json").read_text())

assert a.get("text"), "instance A returned empty generated text"
assert b.get("text"), "instance B returned empty generated text"
assert a["text"] == b["text"], "greedy outputs differ between A and B"
assert 10000 <= a["meta_info"]["prompt_tokens"] <= 10240
assert a["meta_info"].get("cached_tokens", 0) == 0, a["meta_info"]

meta = b["meta_info"]
details = meta.get("cached_tokens_details") or {}
assert meta.get("cached_tokens", 0) > 0, meta
assert details.get("storage", 0) >= 9984, details
assert details["storage"] % 64 == 0, details
assert details.get("storage_backend") == "TensorcastStore", details
print(json.dumps(details, indent=2))
PY
```

Inspect both instance logs for TensorCast exceptions, failed storage writes,
unexpected worker exits, or rank/session name conflicts. Cleanup must TERM and
join both recorded SGLang process groups before stopping the shared daemon and
Global Store with the common cleanup procedure.

## Qwen3-235B-A22B TP=8 single-instance L3 reload lane

This lane validates one full-node rank group against one TensorCast daemon.
Run it once with `TC_TP8_MODE=scratch` for Phase 4 and once with
`TC_TP8_MODE=allocator` for Phase 5. It uses:

- model `/mnt/data/models/Qwen3-235B-A22B` on GPUs 0-7 with TP=8;
- one SGLang instance on HTTP port `30000` and NCCL port `62311`;
- page size 64, `--max-total-tokens 16384`, `--hicache-ratio 1`, Unified
  `page_first_direct`, synchronous direct I/O, and write-through storage;
- the shared daemon at `127.0.0.1:50052`; and
- one approximately 10K-token prompt sent twice, with `/flush_cache` between
  requests. The flush removes rank-local L1/L2 entries but leaves the attached
  Sessions and TensorCast L3 artifacts alive, so the second request cannot pass
  by reusing only the local Unified radix tree or HostPool.

Create a mode-specific config. Allocator mode omits `transfer_mode` to exercise
the default. At TP=8, one Qwen3-235B-A22B BF16 rank-local FULL page contains
`64 * 94 * 1 * 128 * 2 * 2 = 3,080,192` bytes. The 512 MiB scratch arena is
larger than the exact 128-page transfer bound of 394,264,576 bytes, which the
adapter independently validates during registration:

```bash
cd /mnt/data/tot
source /mnt/data/tot/.venv/bin/activate

test -d /mnt/data/models/Qwen3-235B-A22B
test "${TC_TP8_MODE}" = scratch -o "${TC_TP8_MODE}" = allocator
export RUN_DIR TC_TP8_MODE

uv run --active --no-sync python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
mode = os.environ["TC_TP8_MODE"]
run_token = hashlib.sha256(str(run_dir).encode()).hexdigest()[:8]
tensorcast = {
    "daemon_address": "127.0.0.1:50052",
    "namespace": f"q235t8{mode[0]}{run_token}",
    "model_id": "Qwen3-235B-A22B",
    "model_version": f"e2e{run_token}",
    "session_name_prefix": f"q235t8-{mode[0]}",
    "region_name_prefix": f"q235t8-{mode[0]}",
    "exists_timeout_s": 30.0,
    "transfer_timeout_s": None,
    "scratch": {"capacity_bytes": 536870912},
}
if mode == "scratch":
    tensorcast["transfer_mode"] = "scratch"

payload = {
    "prefetch_threshold": 64,
    "prefetch_timeout_base": 10.0,
    "prefetch_timeout_per_ki_token": 2.0,
    "hicache_storage_pass_prefix_keys": False,
    "tensorcast": tensorcast,
}
(run_dir / f"qwen235b_tp8_{mode}.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
PY
```

Start the server and allow up to 30 minutes for model loading. Keep the daemon
on its single visible service GPU from the common fixture; CPU shared-memory
regions do not require its CUDA visibility to mirror the eight worker GPUs:

```bash
export TC_TP8_CONFIG="${RUN_DIR}/qwen235b_tp8_${TC_TP8_MODE}.json"
export TC_TP8_LOG="${RUN_DIR}/qwen235b_tp8_${TC_TP8_MODE}.log"

setsid env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  TC_TP8_CONFIG="${TC_TP8_CONFIG}" \
  SGLANG_CACHE_DIR="${RUN_DIR}/sglang_cache_${TC_TP8_MODE}" \
  TENSORCAST_HOME="${TENSORCAST_HOME}" \
  bash -lc '
    cd /mnt/data/tot/thirdparty/sglang
    source /mnt/data/tot/.venv/bin/activate
    ulimit -n 65535
    unset TENSORCAST_CUDA_BACKEND
    export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export PYTHONPATH="python:/mnt/data/tot/thirdparty/tensorcast${PYTHONPATH:+:${PYTHONPATH}}"
    exec uv run --active --no-sync python -m sglang.launch_server \
      --model-path /mnt/data/models/Qwen3-235B-A22B \
      --tp-size 8 \
      --host 127.0.0.1 \
      --port 30000 \
      --nccl-port 62311 \
      --trust-remote-code \
      --page-size 64 \
      --max-total-tokens 16384 \
      --disable-cuda-graph \
      --enable-hierarchical-cache \
      --hicache-host-memory-mode cache \
      --hicache-ratio 1 \
      --hicache-storage-backend tensorcast \
      --hicache-storage-backend-extra-config "@${TC_TP8_CONFIG}" \
      --hicache-mem-layout page_first_direct \
      --hicache-io-backend direct \
      --hicache-write-policy write_through \
      --hicache-storage-prefetch-policy wait_complete \
      --log-requests \
      --log-requests-level 1 \
      --log-requests-format json
  ' > "${TC_TP8_LOG}" 2>&1 &
echo $! > "${TC_TP8_LOG}.pid"

tc_tp8_pid="$(cat "${TC_TP8_LOG}.pid")"
for iteration in $(seq 1 360); do
  if curl -sf http://127.0.0.1:30000/health_generate >/dev/null; then
    echo "QWEN235_TP8_READY iteration=${iteration}"
    break
  fi
  if ! kill -0 "${tc_tp8_pid}" 2>/dev/null; then
    tail -n 300 "${TC_TP8_LOG}"
    exit 1
  fi
  sleep 5
done
curl -sf http://127.0.0.1:30000/health_generate
```

Build a deterministic 10,000-10,240-token request, execute the cold request,
wait for asynchronous write-through, flush local caches, and execute the same
request again:

```bash
uv run --active --no-sync python - <<'PY'
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

model_path = "/mnt/data/models/Qwen3-235B-A22B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
unit = (
    "Distributed inference verification record: TensorCast preserves each "
    "rank-local FULL key and value cache page exactly. "
)
suffix = "\nQuestion: What is two plus two? Answer with one digit only:"
unit_tokens = len(tokenizer.encode(unit, add_special_tokens=False))
suffix_tokens = len(tokenizer.encode(suffix, add_special_tokens=False))
repeat_count = (10000 - suffix_tokens + unit_tokens - 1) // unit_tokens
prompt = unit * repeat_count + suffix
prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
assert 10000 <= prompt_tokens <= 10240, prompt_tokens
payload = {
    "text": prompt,
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 8},
}
Path(os.environ["RUN_DIR"], "qwen235b_tp8_request.json").write_text(
    json.dumps(payload), encoding="utf-8"
)
print(f"qwen235b_tp8_prompt_tokens={prompt_tokens}")
PY

curl --max-time 900 -sf http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d @"${RUN_DIR}/qwen235b_tp8_request.json" \
  | tee "${RUN_DIR}/qwen235b_tp8_${TC_TP8_MODE}_first.json"

sleep 60
curl --max-time 180 -sf -X POST \
  'http://127.0.0.1:30000/flush_cache?timeout=120' \
  | tee "${RUN_DIR}/qwen235b_tp8_${TC_TP8_MODE}_flush.txt"

curl --max-time 900 -sf http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d @"${RUN_DIR}/qwen235b_tp8_request.json" \
  | tee "${RUN_DIR}/qwen235b_tp8_${TC_TP8_MODE}_second.json"
```

The first request must be cold. The second must reproduce its greedy output
and recover all 9,984 complete prompt tokens from TensorCast L3:

```bash
uv run --active --no-sync python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
mode = os.environ["TC_TP8_MODE"]
first = json.loads((run_dir / f"qwen235b_tp8_{mode}_first.json").read_text())
second = json.loads((run_dir / f"qwen235b_tp8_{mode}_second.json").read_text())

assert first.get("text"), "cold request returned empty generated text"
assert second.get("text"), "reload request returned empty generated text"
assert first["text"] == second["text"], "greedy outputs differ"
assert 10000 <= first["meta_info"]["prompt_tokens"] <= 10240
assert first["meta_info"].get("cached_tokens", 0) == 0, first["meta_info"]

details = second["meta_info"].get("cached_tokens_details") or {}
assert second["meta_info"].get("cached_tokens", 0) > 0, second["meta_info"]
assert details.get("storage", 0) >= 9984, details
assert details["storage"] % 64 == 0, details
assert details.get("storage_backend") == "TensorcastStore", details
assert details.get("device", 0) == 0, details
assert details.get("host", 0) == 0, details
print(json.dumps(details, indent=2))
PY
```

Inspect the server log for TensorCast failures, failed writes, unexpected
worker exits, and rank/session conflicts. TERM and join the recorded SGLang
process group between modes, but keep the common TensorCast services alive.

## Cleanup and diagnostics

Collect status before cleanup:

```bash
cd /mnt/data/tot/thirdparty/tensorcast
source /mnt/data/tot/.venv/bin/activate
uv run --active --no-sync tensorcast-cli global status \
  --gs-session "${TC_GLOBAL_SESSION}" --json \
  | tee "${RUN_DIR}/tensorcast_global_status.final.json" || true
uv run --active --no-sync tensorcast-cli daemon status \
  --session "${TC_DAEMON_SESSION}" --json \
  | tee "${RUN_DIR}/tensorcast_daemon_status.final.json" || true
find "${TENSORCAST_HOME}" -type f \
  | rg 'log|stdout|stderr|json' \
  | tee "${RUN_DIR}/tensorcast_runtime_files.txt" || true
```

Terminate only validated process groups recorded by this run, then stop daemon
before Global Store:

```bash
for pid_file in "${RUN_DIR}"/*.log.pid; do
  test -f "${pid_file}" || continue
  process_pid="$(cat "${pid_file}")"
  process_pgid="$(ps -o pgid= -p "${process_pid}" | tr -d ' ')"
  test -n "${process_pgid}" || continue
  kill -TERM -- "-${process_pgid}" 2>/dev/null || true
done

cleanup_failed=0
for pid_file in "${RUN_DIR}"/*.log.pid; do
  test -f "${pid_file}" || continue
  process_pid="$(cat "${pid_file}")"
  for iteration in $(seq 1 60); do
    kill -0 "${process_pid}" 2>/dev/null || break
    sleep 2
  done
  if kill -0 "${process_pid}" 2>/dev/null; then
    process_pgid="$(ps -o pgid= -p "${process_pid}" | tr -d ' ')"
    test -n "${process_pgid}" \
      && kill -KILL -- "-${process_pgid}" 2>/dev/null \
      || true
    echo "SGLang process did not exit gracefully: pid=${process_pid}"
    cleanup_failed=1
  fi
done

cd /mnt/data/tot/thirdparty/tensorcast
uv run --active --no-sync tensorcast-cli daemon stop \
  --session "${TC_DAEMON_SESSION}" --force || true
uv run --active --no-sync tensorcast-cli global stop \
  --gs-session "${TC_GLOBAL_SESSION}" --force || true

ps -eo pid,ppid,pgid,stat,etime,cmd \
  | rg 'sglang\.launch_server|tensorcast_daemon|global_store' \
  | rg -v rg || true
ss -ltnp \
  | rg ':30000\b|:30001\b|:62311\b|:62321\b|:50051\b|:50052\b|:18000\b|:65090\b' \
  || true
test "${cleanup_failed}" = 0
```

Keep `${RUN_DIR}` after a failure so logs remain available. Do not patch service
configuration or unrelated code in response to an environmental daemon crash.

# Detailed Plan

## Phase 1: Package Boundary, Configuration, And Lazy Selection

Purpose: establish the optional-dependency and configuration contract without
introducing data-path behavior into upper layers.

### Implementation subtasks

- [x] Use an implicit namespace package with no `__init__.py`, matching
      `mooncake_store/`; import concrete modules directly and keep TensorCast
      imports lazy.
- [x] Implement immutable, typed `TensorcastConfig` and nested scratch config
      in `host_allocator.py`.
- [x] Read only the nested `tensorcast` object, leave validation of every
      top-level sibling to its owner, and reject unknown backend-local fields.
- [x] Implement one idempotent normalizer for serialized/path input and
      already-parsed mappings without mutating caller-owned mappings.
- [x] Require `daemon_address` as a strict string and pass it unchanged to the
      public Session API, leaving endpoint validation to TensorCast.
- [x] Fill every documented default, including `transfer_mode="allocator"`,
      `model_version="unversioned"`, null transfer timeout, and the documented
      scratch capacity.
- [x] Keep `model_id=None` unresolved during early configuration and validate
      the later model-name fallback separately.
- [x] Implement the single rank-label helper and Session-options builder so
      Session and region name prefixes receive identical
      `rank{world_rank}of{world_size}` suffixes.
- [x] Pass an optional positive transfer timeout unchanged to the Session, and
      reject unsupported host-memory mode, platform, hardware backend, and
      host-layout combinations before Session attach.
- [x] Add `tensorcast` to the public storage-backend choice without importing
      the optional dependency during ordinary SGLang import or argument help.
- [x] Keep backend construction and allocator selection lazy; selecting another
      backend must not import any TensorCast module.
- [x] Define one clear selected-but-missing-dependency startup error.

### Unit tests

- [x] Add tests to `test_tensorcast_host_allocator.py` for cold optional-
      dependency behavior, selected-only import, forbidden TensorCast modules,
      and missing-dependency diagnostics.
- [x] Add config coverage to `test_tensorcast_host_allocator.py` for defaults,
      endpoint and timeout pass-through, raw/parsed/idempotent input,
      non-mutation, ignored top-level siblings, unknown backend-local fields,
      mode/platform matrix, model-ID deferral, and rank-local names.
- [x] Assert construction uses only public `tensorcast.api.store` types and
      contains no raw daemon/protobuf vocabulary.

### Validation gate

Unit gate:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_host_allocator.py \
  -k 'import or config or endpoint or defaults or normalize or rank_name'
```

E2E gate:

- [x] Run the Qwen3-14B TP=1 correctness lane without
      `--hicache-storage-backend` or backend extra config. Keep
      `--enable-hierarchical-cache --hicache-host-memory-mode cache
      --hicache-ratio 1`.
- [x] `/health_generate` returns HTTP 200.
- [x] One deterministic `/generate` request returns non-empty text.
- [x] The server log has no unexpected traceback/import error and contains no
      TensorCast backend initialization or TensorCast optional import.

Phase 1 exit criteria:

- [x] Both gates pass.
- [x] Ordinary SGLang startup remains independent of TensorCast installation.
- [x] Config and rank naming have one tested implementation.

## Phase 2: Early Session Attachment, Thin Registry, And Host Allocator

Purpose: make allocator and scratch startup converge on one process-scoped
Session before HostPool allocation, without moving ownership into HostPool.

### Implementation subtasks

- [x] Implement the backend-local thin registry in `host_allocator.py` with one
      ordinary `threading.Lock` and only the fields permitted by `design.md`.
- [x] Implement early attach for both modes during host allocator selection.
- [x] Make equal early attach idempotently return the same Session and reject
      conflicting operational options or daemon endpoints.
- [x] Keep scratch mode on the ordinary upstream `HostTensorAllocator` after
      early attach; do not allocate either scratch arena eagerly.
- [x] Implement `TensorcastHostTensorAllocator` as a thin delegate to
      `session.allocate_host_tensor()`.
- [x] Forward exact HostPool shape and dtype, require CPU device, and generate
      only non-empty diagnostic allocation names.
- [x] Keep the allocator free of region handles, resolver state, mmap/file
      descriptors, slot geometry, cleanup callbacks, finalizers, and Session
      termination.
- [x] Add the one lazy allocator-factory branch in `pool_host/common.py`; keep
      parsing, attach, and allocation policy in `host_allocator.py`.
- [x] Implement final-step Store-owner claim primitives, conflicting-owner
      rejection, and no-first-time-Store-attach rejection, but do not yet wire
      the incomplete Store into the backend factory.
- [x] Ensure an attach/allocation/startup error has no fallback to the ordinary
      allocator or scratch mode and requires rank-process exit.

### Unit tests

- [x] Test scratch selection returns the ordinary allocator only after exactly
      one early attach and performs no Session allocation.
- [x] Test allocator selection, exact tensor delegation, multiple allocations,
      non-CPU rejection, and unchanged returned tensors.
- [x] Test repeated allocator instances converge on one Session.
- [x] Test equal/conflicting attach, missing early attach, one owner claim,
      constructor-safe claim ordering, and terminal registry admission in fresh
      spawn children.
- [x] Assert registry and allocator objects expose no region, artifact, slot,
      generation, protobuf, or RPC-health state.

### Validation gate

Unit gate:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_host_allocator.py \
  -k 'allocator or early_attach or registry or store_claim or subprocess'
```

Real-daemon E2E gate:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
unset TENSORCAST_CUDA_BACKEND
export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
SGLANG_TENSORCAST_E2E=1 \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q -s \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_e2e.py \
  -k 'real_daemon_early_attach_and_allocator'
```

- [x] The E2E test runs its attach/allocate/terminate scenario in a newly
      executed child interpreter.
- [x] Scratch and allocator options attach successfully to the ready daemon in
      separate child processes.
- [x] Allocator mode returns a writable CPU tensor from at least two allocation
      calls through the same Session.
- [x] No SGLang HostPool class receives a TensorCast-specific field or callback.

Phase 2 exit criteria:

- [x] Both gates pass.
- [x] The HostPool allocation boundary is the only physical-memory seam.
- [x] Exactly one early Session is observable per test rank process.

## Phase 3: FULL Registration, Fragments, Identity Inputs, And Exists

Purpose: define SGLang-owned logical-page semantics before transferring bytes.

### Implementation subtasks

- [x] Add `TensorcastStore` in `tensorcast_store.py` with all fallible local
      constructor work before the final registry owner claim.
- [x] Implement explicit exact-type/specific-first FULL planner dispatch for
      standard MHA, asymmetric MHA, and MLA.
- [x] Implement the reusable backend-local storage-configuration compatibility
      helper; its initial checks reject unsupported layout, pool subclass,
      packed MTP draft, CP, TP-LCM, split-head, and other non-FULL-v1 semantics.
- [x] Make `register_mem_pool_host()` idempotent only for the same HostPool.
- [x] Sample `get_page_buffer_meta()` once at registration to capture immutable
      component order and byte lengths, not addresses.
- [x] Retain the actual root tensors for the Store lifetime: combined MHA
      `kv_buffer`, asymmetric `k_buffer`/`v_buffer`, or MLA `kv_buffer`.
- [x] Resolve effective model ID at registration and construct one public
      `ByteArtifactKeyspace` with
      `ARTIFACT_LAYOUT_SCHEMA_VERSION="full-fragment-v1"`.
- [x] Implement one canonical caller-input helper for compact layout tokens and
      a domain-separated fixed 32-byte digest over the TP/PP rank suffix,
      opaque logical key, and K/V/KV component tag.
- [x] Implement registered-schema expansion into ordered
      `ByteArtifactSpec` objects with no canonical artifact-ID construction.
- [x] Implement `batch_exists()` as one public Session call, logical component
      AND folding, and leading-prefix count.
- [x] Implement `exists()` through the one-item batch path.
- [x] Preserve empty-call behavior without a Session RPC.
- [x] Implement explicit `NotImplementedError` for value APIs, v2 transfer
      APIs, and `clear()`, while inheriting the base v2 pool-registration method.
- [x] Validate scratch minimum capacity from
      `min(STORAGE_BATCH_SIZE, page_num) * page_bytes` at registration and emit
      every required diagnostic field.
- [x] Validate allocator registration uses the thin TensorCast allocator and
      the identical Session object claimed by the Store.

### Unit tests

- [x] Cover the complete registration matrix for MHA, asymmetric MHA, and MLA
      in both supported layouts.
- [x] Cover exact dispatch, unknown subclasses, unsupported FULL scopes, root
      retention, schema sizes, repeated same-pool registration, and conflicting
      pool registration.
- [x] Cover all caller-owned identity spellings and assert exists/get/put
      helpers will share the same spec path.
- [x] Cover K/V distinction, MLA historical `k` component tag, TP/PP scope,
      exact fixed engine-key digests, compact layout tokens, layout schema
      compatibility marker, and transfer-mode identity independence.
- [x] Cover empty, all-hit, first-page miss, component miss, and tail-hit exists
      folding.
- [x] Cover exact and one-byte-short scratch capacity, including asymmetric
      component lengths and complete diagnostics.
- [x] Assert no test imports or derives a canonical TensorCast artifact ID.

### Validation gate

Unit gate:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_store.py \
  -k 'registration or scope or fragment or identity or exists or scratch_capacity or unsupported'
```

Real-daemon E2E gate:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
unset TENSORCAST_CUDA_BACKEND
export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
SGLANG_TENSORCAST_E2E=1 \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q -s \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_e2e.py \
  -k 'real_daemon_register_and_exists_miss'
```

- [x] The E2E child performs early scratch attach, Store claim, real supported
      FULL HostPool registration, and a real-daemon exists miss.
- [x] It observes `batch_exists(...) == 0`, makes no transfer call, terminates
      the Session once, and exits cleanly.

Phase 3 exit criteria:

- [x] Both gates pass.
- [x] A logical FULL page has one unambiguous, transfer-mode-independent public
      artifact-spec sequence.
- [x] Unsupported v2 and non-FULL paths remain explicit.

## Phase 4: Scratch Get/Put Vertical Slice

Purpose: complete the first end-to-end SGLang storage path using ordinary
HostPool backing plus Session-owned scratch arenas.

### Implementation subtasks

- [x] Implement runtime logical-key/host-index and metadata-count validation.
- [x] Validate page-major component order and runtime lengths against the
      registered schema.
- [x] Calculate each component offset against its retained root tensor.
- [x] Construct every owned span through `HostMemorySpan.from_tensor()` and
      build ordered public `RegionArtifactTransfer` values.
- [x] Implement `batch_set_v1()` through exactly one
      `session.batch_put_from()` call and fold fragment masks to logical pages.
- [x] Implement `batch_get_v1()` through exactly one
      `session.batch_get_into()` call, AND fragment masks, and force every page
      after the first failed logical page to false.
- [x] Keep all specs, spans, transfers, and result tuples immutable and local to
      one synchronous call.
- [x] Preserve Session scratch-direction synchronization; add no SGLang scratch
      buffer, copy loop, region binding, or transfer lock.
- [x] Add the lazy backend-factory registration and the minimum controller
      callback selection needed for Unified synchronous v1 zero-copy.
- [x] Ensure the complete explicitly configured scratch path can start through
      public SGLang CLI selection.

### Unit tests

- [x] Test MHA and MLA put/get transfer construction, actual addresses, byte
      lengths, owners, offsets, and caller order.
- [x] Test asymmetric K/V owners and lengths even though direct multi-region
      behavior is completed in Phase 5.
- [x] Test logical AND folding, get prefix normalization, partial K/V result,
      put non-prefix result, empty calls, and invalid runtime metadata.
- [x] Test that scratch and allocator config produce identical artifact specs.
- [x] Test the controller selects existing synchronous v1 callbacks and that
      anchor-only `HostPoolGroup` registration reaches the primary FULL pool.
- [x] Test inherited sidecar registration records the pool without advertising
      v2 transfer support.

### Validation gate

Unit gate:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_store.py \
  -k 'span or batch_get_v1 or batch_set_v1 or folding or scratch or controller or host_pool_group'
```

Real-daemon adapter E2E gate:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
unset TENSORCAST_CUDA_BACKEND
export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
SGLANG_TENSORCAST_E2E=1 \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q -s \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_e2e.py \
  -k 'real_daemon_scratch_full_round_trip'
```

Qwen correctness E2E gate:

- [x] Start one Qwen3-14B TP=1 SGLang instance with scratch config and use
      `page_first_direct` plus `direct`. Transfer mode controls only L2/L3
      behavior.
- [x] Send one deterministic prompt, wait for its asynchronous write-through
      `batch_put_from()` to complete, then call `/flush_cache` to remove only
      the rank-local device and HostPool cache while preserving TensorCast L3.
- [x] Send the identical prompt to the same SGLang instance. The storage path
      observes an exists hit and completes `batch_get_into()` successfully.
- [x] Both requests return non-empty deterministic output. The second reports
      `cached_tokens > 0`, `cached_tokens_details.storage > 0`, and
      `storage_backend == "TensorcastStore"`.
- [x] Logs contain no unexpected exception. Unit and adapter E2E tests, rather
      than extra production logging, prove the exact Session call sequence and
      that scratch mode used the ordinary HostPool allocator.
- [x] Run the Qwen3-32B TP=2 concurrent-instance reuse lane with
      `TC_MULTI_MODE=scratch`: keep instance A on GPUs 0/1 and instance B on
      GPUs 2/3 alive together, with all four rank Sessions attached to the same
      daemon.
- [x] Verify A's approximately 10K-token request is written through with
      `batch_put_from()`, then the identical request to B succeeds through
      `batch_exists()` and `batch_get_into()`. B must report at least 9,984
      page-aligned storage tokens from `TensorcastStore`, and both instance
      logs must be free of rank/session conflicts and unexpected exceptions.
- [x] Run the Qwen3-235B-A22B TP=8 single-instance L3 reload lane with
      `TC_TP8_MODE=scratch` on GPUs 0-7. After the cold request writes through,
      flush rank-local caches and verify the identical second request obtains
      at least 9,984 page-aligned storage tokens from `TensorcastStore`.

Phase 4 exit criteria:

- [x] Unit, adapter E2E, TP=1, concurrent TP=2, and full-node TP=8 model E2E
      gates pass.
- [x] Scratch transfer is a complete supported vertical slice.
- [x] Scratch registration proves its configured capacity is sufficient before
      storage workers start.

## Phase 5: Allocator-Backed Direct Mode And Multi-Region Transfer

Purpose: make the default mode use Session-allocated HostPool tensors directly
and prove SGLang does not own TensorCast region-layout logic.

### Implementation subtasks

- [x] Complete allocator-mode startup through the public SGLang path and make
      it the effective default when `transfer_mode` is omitted.
- [x] Verify every allocator-backed root returned to MHA/MLA HostPool is the
      exact Session allocation tensor later retained by Store registration.
- [x] Reuse the Phase 4 spec/span/transfer path unchanged; add no mode-specific
      packing or request compilation in `TensorcastStore`.
- [x] Submit equal MHA, asymmetric MHA, and MLA fragments in one public Session
      batch even when their spans belong to multiple allocations/regions.
- [x] Keep upstream CUDA host registration in
      `alloc_with_host_register()` and do not register the memory a second time.
- [x] Fail allocator/Session identity mismatch at registration and provide no
      automatic allocator-to-scratch fallback.
- [x] Verify direct transfer uses Session defaults: zero transparent transfer
      retries and `transfer_timeout_s=None`; expose no retry knob in SGLang.

### Unit tests

- [x] Verify allocation roots, Store roots, and span owners are object-identical
      where the public contracts require it.
- [x] Verify no SGLang copy or scratch allocation occurs on allocator get/put.
- [x] Verify asymmetric K and V are two Session allocations but one batch call.
- [x] Verify multiple allocator instances and allocations use the same Session.
- [x] Verify allocator/session mismatch and non-Session HostPool backing fail
      registration.
- [x] Verify omitted transfer mode is allocator and explicit scratch remains
      functional without changing artifact specs.

### Validation gate

Unit gate:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_host_allocator.py \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_store.py \
  -k 'allocator or direct or asymmetric or multi_region or same_session or no_copy'
```

Real-daemon adapter E2E gate:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
unset TENSORCAST_CUDA_BACKEND
export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
SGLANG_TENSORCAST_E2E=1 \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q -s \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_e2e.py \
  -k 'real_daemon_allocator_full_round_trip or real_daemon_asymmetric_multi_region'
```

Qwen correctness E2E gate:

- [x] Start one Qwen3-14B TP=1 SGLang instance with an allocator config that
      omits `transfer_mode`, proving allocator is the effective default.
- [x] Use `page_first_direct`, `--hicache-io-backend direct`, and
      `--hicache-host-memory-mode cache`.
- [x] Send one deterministic prompt, wait for asynchronous write-through, and
      call `/flush_cache` to remove only rank-local device and HostPool state
      while preserving TensorCast L3 and the process Session.
- [x] Send the identical prompt to the same instance. Both requests return
      non-empty deterministic output and the second reports `cached_tokens > 0`,
      `cached_tokens_details.storage > 0`, and
      `storage_backend == "TensorcastStore"`.
- [x] Logs contain no unexpected exception. Unit and adapter E2E tests, rather
      than extra production logging, prove that no SGLang scratch-copy path was
      used.
- [x] Run the Qwen3-32B TP=2 concurrent-instance reuse lane with
      `TC_MULTI_MODE=allocator` and omit `transfer_mode` from both A/B configs.
      Keep A on GPUs 0/1 and B on GPUs 2/3 alive together, with four distinct
      rank Sessions sharing one daemon.
- [x] Verify A's approximately 10K-token request publishes from its
      daemon-backed HostPools, then B obtains at least 9,984 page-aligned
      storage tokens through `batch_exists()` and `batch_get_into()`. Both
      outputs must match and neither instance log may contain an unexpected
      exception or failed storage write.
- [x] Run the Qwen3-235B-A22B TP=8 single-instance L3 reload lane with
      `TC_TP8_MODE=allocator`, omitting `transfer_mode`. After a successful
      local flush, verify the second identical request obtains at least 9,984
      page-aligned storage tokens from `TensorcastStore`.

Phase 5 exit criteria:

- [x] Unit, adapter E2E, TP=1, concurrent TP=2, and full-node TP=8 model E2E
      gates pass.
- [x] Allocator mode is the working default.
- [x] Real asymmetric K/V transfer proves one batch can span multiple regions
      without any SGLang region-layout code.

## Phase 6: Failure Containment, Concurrency, And Terminal Lifecycle

Purpose: make all supported FULL-v1 runtime failures degrade to a permanently
unavailable rank-local L3 path while preserving the SGLang worker and process.

### Implementation subtasks

- [x] Add one small adapter availability/publication lock, disabled flag, and
      logged flag; never hold the lock across a Session RPC.
- [x] Recheck availability before publishing successful results so no call
      publishes success after a fatal adapter transition.
- [x] Map runtime `RegionArtifactInputError`, `RegionSessionFailedError`,
      unexpected `RegionSessionTerminatedError`, and any other adapter-boundary
      `Exception` to permanent rank-local L3 unavailability.
- [x] Return 0/all-false from the current and every later supported v1 call and
      never invoke the Session data path again.
- [x] Log only the first transition, use `logger.exception()` where a traceback
      is required, and include only public failure code/operation information.
- [x] Keep ordinary false item results as item/page failures without disabling
      the adapter.
- [x] Preserve explicit `NotImplementedError` for unsupported legacy/v2/clear
      APIs; do not contain those as supported FULL-v1 failures.
- [x] Implement idempotent Store close through the valid registry owner and one
      `terminate_process_session()` call after storage workers stop.
- [x] Never terminate the Session merely because it failed and never release
      regions, unregister backing, call `release_daemon_client()`, or add
      allocator cleanup.
- [x] Reject second active Store ownership and same-process reattach/reclaim
      after terminal close.
- [x] Preserve rank-local containment only; add no cross-rank health broadcast.

### Unit tests

- [x] Inject every documented public exception into exists/get/put and verify
      current/future no-hit results, exactly one log, and no later Session call.
- [x] Inject unexpected planner, span-construction, Session, and result-folding
      exceptions and verify the worker-facing method does not raise.
- [x] Test ordinary misses do not disable the adapter.
- [x] Race a successful call against fatal disablement and prove the
      publication gate suppresses success linearized after disablement.
- [x] Exercise prefetch and backup controller callbacks with fake failures and
      prove their worker logic remains alive.
- [x] Test close, repeated close, one termination call, second owner rejection,
      terminal reclaim rejection, constructor claim safety, and retained roots
      in separate spawned processes.
- [x] Test v2 and clear remain explicit unsupported exceptions outside the
      containment boundary.

### Validation gate

Unit gate:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_host_allocator.py \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_store.py \
  -k 'failure or disabled or publication or concurrent or close or terminate or lifecycle or worker'
```

Real-daemon adapter E2E gate:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
unset TENSORCAST_CUDA_BACKEND
export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
SGLANG_TENSORCAST_E2E=1 \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q -s \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_e2e.py \
  -k 'real_daemon_terminal_lifecycle or real_daemon_failure_latches_adapter'
```

Qwen failure-containment E2E gate:

- [x] Start allocator-mode Qwen3-14B TP=1 and complete one request while the
      daemon is healthy; wait for all expected backup work to become idle.
- [x] Stop only StoreDaemon, leaving Global Store and SGLang running.
- [x] Submit a new cold prompt so the next L3 operation is metadata-only
      `batch_exists()`. Do not deliberately interrupt a direct get/put because
      the initial design accepts ambiguous in-flight completion and has no
      finite transfer timeout.
- [x] The request completes by normal recomputation, a second request also
      returns non-empty text, `/health_generate` remains HTTP 200, and the
      SGLang process remains alive.
- [x] The log contains exactly one TensorCast L3 unavailable transition and no
      uncaught storage-worker traceback or worker death.
- [x] A later request emits no second unavailable-transition log. The stronger
      no-later-Session-call invariant is proven by the fake-Session unit gate,
      without adding a production counter or inspecting private SDK state.

Phase 6 exit criteria:

- [x] Unit, adapter E2E, and model failure E2E gates pass.
- [x] Supported FULL-v1 failure does not kill the rank or its storage workers.
- [x] Session and region lifetime remain process-pinned until process exit.

## Phase 7: Integration Closure, Documentation, And Final Acceptance

Purpose: verify the complete implementation against the design boundary and
close the PR with reproducible operator documentation.

### Implementation subtasks

- [x] Add `README.md` covering prerequisites, startup order, supported matrix,
      configuration with defaults, allocator default, scratch sizing, rank
      naming, shutdown, failure behavior, and unsupported configurations.
- [x] Copy the validated Global Store and StoreDaemon YAML files from the
      deprecated integration branch into `configs/`; use these checked-in files
      as the service inputs for every E2E gate.
- [x] Revalidate the checked-in service configs against the completed adapter
      and document any required runtime secret injection without rewriting the
      YAML during a test run.
- [x] Document that services are user-prestarted, CPU shared memory must be
      enabled, the endpoint must be node-local, and process PID visibility must
      match the daemon deployment.
- [ ] Audit all imports and external diffs against the hard ownership boundary.
- [ ] Delete or avoid every deprecated raw client, protobuf, layout, region,
      slot-generation, special HostPool, and allocator-cleanup concept.
- [ ] Verify every acceptance criterion in `design.md` is represented by a
      passing test, a documented deployment precondition, or an explicit
      initial limitation.
- [ ] Update comments and user documentation if final implementation names
      differ from pseudocode, without weakening semantics.
- [ ] Run targeted ruff check and format check over every touched Python file.
- [ ] Review `git diff --check`, `git status`, and the final external diff list.

### Unit tests

- [ ] Run the complete colocated TensorCast test suite without selectors.
- [ ] Run the focused generic HiCache tests affected by the thin wiring; do not
      move or add TensorCast-specific fixtures to generic test directories.
- [ ] Repeat the optional-dependency cases in
      `test_tensorcast_host_allocator.py` in an environment where TensorCast
      import is blocked.
- [ ] Confirm all stateful lifecycle tests execute fresh interpreters and no
      registry-reset hook exists.

### Validation gate

Unit gate (full colocated suite):

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q \
  python/sglang/srt/mem_cache/storage/tensorcast_store
```

Thin-wiring regression gate; populate the exact generic test paths touched by
the implementation before checking this item:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake \
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync pytest -q \
  python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_store.py \
  -k 'controller or backend_factory or allocator_factory or host_pool_group'
```

Static gate:

```bash
cd /mnt/data/tot/thirdparty/sglang
source /mnt/data/tot/.venv/bin/activate
PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync ruff check \
  python/sglang/srt/mem_cache/storage/tensorcast_store \
  python/sglang/srt/mem_cache/storage/backend_factory.py \
  python/sglang/srt/mem_cache/managers/cache_controller.py \
  python/sglang/srt/mem_cache/pool_host/common.py \
  python/sglang/srt/arg_groups/hicache_hook.py \
  python/sglang/srt/server_args.py

PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
uv run --active --no-sync ruff format --check \
  python/sglang/srt/mem_cache/storage/tensorcast_store \
  python/sglang/srt/mem_cache/storage/backend_factory.py \
  python/sglang/srt/mem_cache/managers/cache_controller.py \
  python/sglang/srt/mem_cache/pool_host/common.py \
  python/sglang/srt/arg_groups/hicache_hook.py \
  python/sglang/srt/server_args.py

git diff --check
```

Final E2E gate:

- [ ] Start Global Store and StoreDaemon with the shared real-daemon fixture.
- [ ] Run all environment-enabled adapter E2E scenarios:

  ```bash
  cd /mnt/data/tot/thirdparty/sglang
  source /mnt/data/tot/.venv/bin/activate
  unset TENSORCAST_CUDA_BACKEND
  export LD_LIBRARY_PATH="/usr/local/cuda-13/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  SGLANG_TENSORCAST_E2E=1 \
  PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
  uv run --active --no-sync pytest -q -s \
    python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_e2e.py
  ```

- [ ] Repeat the no-TensorCast Qwen3-14B TP=1 baseline from Phase 1.
- [ ] Run the Qwen publisher/consumer lane with a scratch publisher and an
      allocator consumer using the same artifact identity and
      `page_first_direct` layout. The consumer must report a nonzero TensorCast
      storage hit.
- [ ] Run the Qwen3-32B TP=2 concurrent-instance reuse lane twice: scratch for
      Phase 4 and default allocator for Phase 5. In each run, A and B remain
      live on GPUs 0/1 and 2/3 while their four rank Sessions share one daemon;
      B must recover at least 9,984 page-aligned storage tokens published by A.
- [ ] Run allocator-mode Qwen3-14B TP=1 and execute the local serving workload:

  ```bash
  cd /mnt/data/tot/thirdparty/sglang
  source /mnt/data/tot/.venv/bin/activate
  PYTHONPATH=python:/mnt/data/tot/thirdparty/tensorcast \
  uv run --active --no-sync python -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port 30001 \
    --model /mnt/data/models/Qwen3-14B/ \
    --dataset-name random-ids \
    --num-prompts 128 \
    --random-input-len 128 \
    --random-output-len 32 \
    --random-range-ratio 1.0 \
    --max-concurrency 8
  ```

- [ ] All 128 requests complete with zero failed requests.
- [ ] `/health_generate` remains HTTP 200 before and after the workload.
- [ ] SGLang logs contain no unexpected traceback/import/worker error.
- [ ] TensorCast status remains healthy until intentional shutdown.
- [ ] Stop SGLang rank processes, then StoreDaemon, then Global Store; verify
      the recorded ports and GPUs are released.

Phase 7 exit criteria:

- [ ] Unit, static, adapter E2E, baseline model, cross-mode L3 reuse,
      concurrent TP=2 reuse, and allocator serving gates pass.
- [ ] All applicable phase and subtask checkboxes are checked.
- [ ] The implementation matches every initial-scope acceptance criterion in
      `design.md` and contains no unexplained external diff.

# Validation Failure Policy

When a gate fails:

1. Preserve `${RUN_DIR}`, SGLang logs, TensorCast CLI status JSON, and daemon
   runtime files.
2. Identify whether the failure is adapter logic, TensorCast public SDK,
   daemon/environment, or unrelated SGLang behavior.
3. Fix only an in-scope root cause. Do not weaken a correctness assertion,
   mutate daemon configs ad hoc, add a HostPool workaround, or access a private
   TensorCast API.
4. If the cause is an environment issue or an unrelated/pre-existing daemon
   crash, stop implementation work and report the exact command, failing case,
   logs, and initial diagnosis for discussion.
5. Re-run the failed phase's complete unit and E2E gates after an in-scope fix;
   do not check off individual tasks based only on a narrower reproducer.
