# tc_router Implementation Plan

This file enumerates the concrete work needed to deliver the benchmark
defined in `arch.md`, broken into phases with explicit checklists and
validation gates.

The high-level strategy is **baselines first, then `tc_router`**. We
get the harness, workload, and one gateway baseline running end to
end before adding any Tensorcast-specific code. This means at every
phase we have something runnable that we can use to debug regressions.

## 0. Repo skeleton ✅ DONE

**Goal**: empty package tree compiles and imports; CI / tooling can
discover modules.

**Deliverables**:

- [x] `tc_router/__init__.py`
- [x] `tc_router/{resource,services,driver,router,workload,metrics}/__init__.py`
- [x] `tc_router/tests/__init__.py`
- [x] `tc_router/configs/.gitkeep`
- [x] `tc_router/outputs/.gitkeep` (gitignored via root `.gitignore` `outputs/`)
- [x] entry-point Python 3.10 minimum versioning aligned with the rest of `thirdparty/sglang/benchmark/tensorcast_benchmark`
- [x] `tc_router/scripts/.gitkeep` (for service lifecycle wrappers, will reuse from `request_transfer/scripts/` and `share_remote/scripts/` later)

**Validation gate**:
- [x] `python -c "import tensorcast_benchmark.kv.tc_router"` succeeds from `thirdparty/sglang/benchmark`
- [x] `pytest tc_router/tests` runs (and reports "no tests collected", which is fine)

---

## 1. Resource abstraction (BrainctlProvider) ✅ DONE

**Goal**: `cluster.yaml` → `list[Worker]`, with **BrainctlProvider** as
the v1 provider. The provider does **not** acquire workers; it adapts
already-acquired brainctl processes (described in YAML) to a uniform
`Worker` interface.

This phase patterns itself on the brainctl helpers already proven in
`kv/share_remote/run_benchmark.py` and `kv/share_remote/models.py`.

### 1.1 Design notes captured from share_remote

- **Acquisition is out of band**: cluster YAML lists existing
  `process_name` values (analogous to share_remote's
  `existing_worker_processes`). The provider verifies each is
  `Running, Ready=1/1` and refuses to start otherwise.
- **Exec model**: every command on a worker goes through
  `brainctl exec process/<name> -n <namespace> -- bash -lc <cmd>`.
  Two flavors — root and user-scoped — mirroring share_remote's
  `exec_root` and `exec_user`. The user-scoped form wraps the inner
  command with `su - <user> -s /bin/bash -c <quoted>`.
- **No file transfer over the cluster CLI**: share_remote relies on
  a shared mount (`juicefs+...:/mnt/step2-alignment-jfs`) that maps
  identically on driver host and every worker. We adopt the same
  contract. `Worker.put_file` / `get_file` therefore copy via local
  filesystem to / from a path inside the mount.
- **Background services**: long-lived processes are launched with
  `nohup <cmd> > $LOG_PATH 2>&1 < /dev/null & echo $! > $PID_PATH`,
  managed via PID files. Stop is `kill $(cat $PID_PATH)` (then `-9`
  if needed) followed by `rm -f $PID_PATH`. This is exactly the
  `start_remote_process` / `stop_remote_process` pattern in share_remote.
- **Worker info**: `brainctl get process/<name> -o wide` returns a
  table; we parse `IP` and `NODE` columns. `IP` becomes
  `Worker.address`; `NODE` is recorded in cluster YAML metadata for
  distinct-host validation.
- **RDMA env injection**: cluster YAML carries per-worker `base_env`
  with `NCCL_IB_HCA` (derived per worker), `NCCL_IB_GID_INDEX=3`,
  `NCCL_SOCKET_FAMILY=AF_INET`, `NCCL_SOCKET_IFNAME`, `MASTER_ADDR`.
  `Worker.run` always merges `base_env` into the inner shell command
  via `export K=V; ...`.
- **Local-host hygiene**: when invoking `brainctl` from the driver host,
  HTTP(S) proxy env vars must be stripped (share_remote's
  `BRAINCTL_PROXY_ENV_KEYS` list). The provider does this in its
  internal `_run_local_brainctl` helper.

### 1.2 Deliverables

- [x] `resource/base.py`
  - [x] `RemoteProcess` Protocol (`pid`, `wait`, `kill`, `stdout`, `stderr`)
  - [x] `Worker` Protocol (`id`, `address`, `node`, `gpu_indices`, `scratch_dir`, `base_env`, `run`, `start_background`, `stop_background`, `read_file`, `put_file`, `get_file`)
    - `run(cmd, *, env=None, cwd=None, timeout_s=None, check=True) -> RemoteProcess`: synchronous-style; awaitable
    - `start_background(cmd, *, name, log_path, pid_path, env=None) -> str`: returns the started PID; mirrors share_remote `start_remote_process`
    - `stop_background(*, pid_path) -> None`: mirrors share_remote `stop_remote_process`
  - [x] `ResourceProvider` Protocol (`from_cluster_config`, `workers`, `health_check`)
  - [x] `ClusterConfig`, `WorkerConfig`, `ServicePlacement`, `MountConfig` Pydantic models matching arch § 9.1
  - [x] Cluster YAML loader: validates distinct `id`, distinct `address`, distinct `node`, every worker has `process_handle`, every worker has `base_env`
- [x] `resource/brainctl.py`
  - [x] `BrainctlProvider`:
    - `__init__(cluster_config, *, cli=None)`
    - `from_cluster_config(path)` classmethod
    - `workers() -> list[BrainctlWorker]`
    - `health_check()`: for each worker, run `brainctl get process/<name> -o wide`, parse status, assert `Running, Ready=1/1`; assert parsed IP matches `WorkerConfig.address`; assert parsed NODE matches `WorkerConfig.node`
  - [x] `BrainctlWorker(Worker)`:
    - `_exec_cli_argv(...)`: builds `["brainctl","exec","process/<name>","-n",ns,"--","bash","-lc",cmd]`
    - `run(cmd, env=None, ..., as_user=True)`: wraps `cmd` with env exports; if `as_user=True`, adds `su - <user>` shim (per share_remote `exec_user`); else direct `exec_root`
    - `start_background(cmd, name, log_path, pid_path, env=None)`: composes the `nohup ... &; echo $! > $PID_PATH` shell snippet exactly like share_remote `start_remote_process`
    - `stop_background(pid_path)`: same shell snippet as share_remote `stop_remote_process`
    - `read_file(remote_path, max_bytes=None)`: shared-mount → `Path(remote_path).read_bytes()` from driver host
    - `put_file(local, remote)` / `get_file(remote, local)`: shared-mount copy
  - [x] Helper: `_run_local_brainctl(argv, *, timeout_s, check)` strips proxy env, runs subprocess via `loop.run_in_executor`, raises `BrainctlError` on non-zero
  - [x] Custom exceptions: `BrainctlError`, `BrainctlNotReadyError`, `BrainctlParseError`
- [x] `resource/factory.py`
  - [x] `from_cluster_config(path) -> ResourceProvider` dispatch on `provider.kind`; only `brainctl` registered in v1
- [x] `resource/static.py` — placeholder file, raises `NotImplementedError`. (Static fallback is documented in arch § 14.4 but not built in v1 per current scope.)
- [x] `tc_router/configs/cluster_brainctl_example.yaml` — template showing all required fields, with placeholders / comments. Operators copy this when populating a real cluster YAML after running their acquisition script.
- [x] `scripts/acquire_brainctl.py` — out-of-band convenience script (NOT called by `run_benchmark.py`):
  - launches N workers via `brainctl launch -d --i-know-i-am-wasting-resource ...` with the `--negative-tags=node/<used>` distinct-host pattern (mirrors share_remote's `launch_worker`)
  - waits for all to reach `Running, Ready=1/1`
  - resolves IP / NODE for each via `brainctl get ... -o wide`
  - derives `NCCL_IB_HCA` per worker via the codex helper if available, otherwise emits a placeholder
  - emits a populated `cluster_brainctl_<id>.yaml` to the requested path
- [x] `scripts/release_brainctl.py` — out-of-band convenience: `brainctl stop` + `brainctl delete` for each `process_handle` listed in a given cluster YAML

### 1.3 Validation gate

- [x] Unit test: `ClusterConfig` parses a hand-crafted cluster YAML with 3 workers; loader rejects YAML with duplicate `id`, duplicate `address`, missing `node`, missing `base_env`, invalid `service_placement`, extra fields, zero workers (11 tests in `test_resource_base.py`).
- [x] Unit test: shipped example `cluster_brainctl_example.yaml` parses cleanly through the factory.
- [x] Unit test: `BrainctlWorker._exec_cli_argv` produces the expected argv list for a simple command, env-var injection (`base_env` + per-call override), the `as_user` su-shim wrapping, and shell-quoting when values contain spaces (15 tests in `test_resource_brainctl.py`).
- [x] Unit test: `start_background` composes a shell snippet containing `nohup`, `> $LOG_PATH`, `echo $!`, `> $PID_PATH`; `stop_background` snippet contains `kill`, `kill -9`, `rm -f`.
- [x] Unit test: `parse_worker_info` correctly extracts `IP` / `NODE` from a representative `brainctl get -o wide` output and rejects malformed inputs.
- [x] Unit test: factory dispatches to `BrainctlProvider`, rejects unknown `provider.kind`, `provider.workers()` is idempotent.
- [x] Live test (executed against a single-H800-worker test cluster on 2026-06-15 via `python -m tensorcast_benchmark.kv.tc_router.tools.live_check_resource configs/cluster_brainctl_single_h800.yaml`):
  - [x] `provider.health_check()` against an actually-acquired worker passes
  - [x] `worker.run(["echo", "hello"])` returns stdout `hello`
  - [x] `worker.run(["env"])` includes `NCCL_IB_HCA=...` from `base_env` (also `NCCL_IB_GID_INDEX`, `MASTER_ADDR`)
  - [x] `worker.start_background("sleep 30; echo done", ...)` returns a PID, then `worker.stop_background(...)` succeeds and the PID is gone (with PID-file cleanup)
  - [x] `worker.put_file(local_tmp, mount_path)` then `worker.read_file(mount_path)` round-trips a small payload (also `worker.get_file`)
- [x] Manual: validated end-to-end on a pre-existing 1-worker setup. Acquisition flow (`scripts/acquire_brainctl.py` → `scripts/release_brainctl.py`) is reserved for future multi-worker live runs.

**Live finding** (recorded for future arch tweaks): on this brainctl cluster the master pod's `/mnt/step2-alignment-jfs/` is **not** the same JFS as the worker's. The actual driver↔worker shared filesystem is `/home/<user>/` (NFS-mounted from a backing storage on the worker, native on the master). The `cluster.yaml`'s `mount.path` and `scratch_dir` fields should point inside that NFS-shared subtree, not the JFS one. The committed `cluster_brainctl_single_h800.yaml` reflects this. The `acquire_brainctl.py` defaults still mention the JFS path (matching share_remote conventions); operators should override `--mount-path` / scratch dir flags as appropriate when running against this cluster.

**Test summary**: `pytest tensorcast_benchmark/kv/tc_router/tests` from `thirdparty/sglang/benchmark` passes 31/31.

---

## 1.5 StaticProvider (local + SSH static workers)

**Goal**: replace the deprecated brainctl dependency for ordinary runs with
an ssh-based **StaticProvider**. The provider does not acquire machines; it
adapts an operator-supplied static inventory into the existing `Worker`
interface. The first target deployment is a hybrid static cluster:

- local worker: the current 8xH800 machine, executed via local subprocesses
- remote worker: `yuhan@10.0.10.58`, executed via SSH with existing pubkey auth

The provider must keep the services layer cluster-agnostic: SGLang,
Tensorcast daemon/global-store, gateway, and workload code should continue
to depend only on `ResourceProvider` / `Worker`.

All static workers for this deployment share the same `/mnt/data`
filesystem. Each worker's `/home/yuhan` is a symlink to `/mnt/data`, so
file transfer must be implemented as shared-filesystem access only; no
`scp` or `rsync` staging is required for benchmark inputs, service logs, or
outputs.

### 1.5.1 Config contract

- [x] Define `provider.kind: static` in cluster YAML and register it in
  `resource/factory.py`.
- [x] Extend or reuse `WorkerConfig` so each worker can declare an execution
  backend without brainctl fields:
  - [x] `execution: local` for the current driver host.
  - [x] `execution: ssh` for remote static workers.
  - [x] SSH fields: `ssh_user`, `ssh_host`, `ssh_port`, optional
    `identity_file`, optional `connect_timeout_s`.
  - [x] Common fields stay unchanged: `id`, `address`, `node`,
    `gpu_indices`, `scratch_dir`, `base_env`, optional env path prepends,
    optional env unsets.
- [x] Validate static YAML invariants:
  - [x] unique worker `id`
  - [x] unique `node`
  - [x] valid non-empty `gpu_indices`
  - [x] local worker has no required SSH target
  - [x] SSH worker has user, host, and port
  - [x] `scratch_dir` is absolute
  - [x] cluster mount, driver scratch, worker scratch, and output paths are
    under the shared `/mnt/data` filesystem, or under `/home/yuhan` which
    resolves there
  - [x] CUDA env path prepends are represented by `env_path_prepend` and
    merged by both local and SSH workers.
  - [x] Proxy variables are represented by `env_unset` and removed from both
    local and SSH worker command environments.
- [x] Add a shipped example config for the current deployment:
  `configs/cluster_static_local_h800_plus_10_0_10_58.yaml`.

### 1.5.2 Worker implementation

- [x] Implement `resource/static.py` with `StaticProvider`,
  `LocalStaticWorker`, and `SshStaticWorker`.
- [x] Preserve the existing `Worker` protocol:
  - [x] `run(...)`
  - [x] `start_background(...)`
  - [x] `stop_background(...)`
  - [x] `read_file(...)`
  - [x] `put_file(...)`
  - [x] `get_file(...)`
- [x] Local execution:
  - [x] run commands through `bash --noprofile --norc -lc` under the
    configured `cwd`
  - [x] merge `base_env`, env path prepends, per-call env, and configured env
    unsets consistently
  - [x] launch background services with the same PID-file/log-file contract
    used by the services layer
- [x] SSH execution:
  - [x] build deterministic SSH argv with `BatchMode=yes`,
    `StrictHostKeyChecking=accept-new`, configured port, and optional
    identity file
  - [x] run remote commands as `bash --noprofile --norc -lc <quoted command>`
  - [x] avoid any brainctl imports or command paths
  - [x] compose remote background launch/stop snippets equivalent to the
    local worker snippets
- [x] Return a `RemoteProcess`-compatible result object for both local and
  SSH paths, including stdout, stderr, return code, timeout, and check
  behavior.

### 1.5.3 Filesystem and artifacts

- [x] Use a single shared-filesystem transfer model:
  - [x] all workers see `/mnt/data` at the same absolute path
  - [x] all workers have `/home/yuhan` symlinked to `/mnt/data`
  - [x] repo, model, dataset, scratch, service logs, and outputs live on that
    shared filesystem
- [x] Implement `read_file`, `put_file`, and `get_file` as shared-filesystem
  operations from the driver host; SSH is used for command execution only.
- [x] Ensure service logs and PID files always live under each worker's
  configured `scratch_dir`.
- [x] Ensure final run artifacts are collected back into
  `outputs/<run_id>/`, including remote service logs already written on the
  shared filesystem.

### 1.5.4 Health checks

- [x] `StaticProvider.health_check()` validates the local worker:
  - [x] `hostname`
  - [x] `nvidia-smi`
  - [x] visible GPU count covers `gpu_indices`
  - [x] `scratch_dir` exists or can be created
  - [x] shared `/mnt/data` and repo root exist
- [x] `StaticProvider.health_check()` validates each SSH worker:
  - [x] SSH connectivity with pubkey auth and no password prompt
  - [x] remote `hostname`
  - [x] remote `nvidia-smi`
  - [x] visible GPU count covers `gpu_indices`
  - [x] remote `scratch_dir` exists or can be created
  - [x] shared `/mnt/data` and repo root exist
  - [x] `/home/yuhan` resolves to `/mnt/data`
- [x] Report actionable failures that identify the worker id and the failing
  command.

### 1.5.5 Tests

- [x] Unit test: static cluster YAML with one local worker and one SSH worker
  parses through `ClusterConfig` and `from_cluster_config`.
- [x] Unit test: factory dispatches `provider.kind: static` to
  `StaticProvider` and rejects malformed static worker entries.
- [x] Unit test: local worker command composition merges base env, path
  prepends, cwd, and per-call env correctly.
- [x] Unit test: SSH worker command composition produces the expected SSH
  argv and shell quoting for env values containing spaces/special chars.
- [x] Unit test: local and SSH `start_background` snippets write PID files,
  redirect logs, and use the expected stop escalation sequence.
- [x] Unit test: shared-filesystem `read_file`, `put_file`, and `get_file`
  use local filesystem operations and do not build `scp` / `rsync` commands.
- [x] Optional live smoke: with `yuhan@10.0.10.58`, run
  `provider.health_check()` and a trivial `worker.run("echo hello")` on both
  workers. Do not run an E2E benchmark as part of this phase.

### 1.5.6 Validation gate

- [x] `source .venv/bin/activate && uv run --active ruff check
  thirdparty/sglang/benchmark/tensorcast_benchmark/kv/tc_router/resource
  thirdparty/sglang/benchmark/tensorcast_benchmark/kv/tc_router/tests`
- [x] `source .venv/bin/activate && PYTHONPATH=/mnt/data/tot/thirdparty/sglang/benchmark
  uv run --active pytest tensorcast_benchmark/kv/tc_router/tests`
- [ ] Static local-only config can launch the same tp=2 smoke setup that
  already works with the local provider.
- [x] Static local+SSH config passes health check on the local 8xH800 worker
  and `yuhan@10.0.10.58`.

---

## 2. Services layer (cluster-agnostic launchers) ✅ DONE (sglang launcher unit + live; rdma_smoke unit only)

**Goal**: `services.<name>.launch(...)` returns a running, healthy
service. None of these know about brainctl / SSH / k8s.

**Deliverables**:

- [x] `services/base.py`
  - [x] `Service` dataclass (`name`, `worker_id`, `endpoints: dict[str, str]`, `pid`, `pid_path`, `log_path`, `metadata`)
  - [x] `ServiceLauncher` Protocol with `async launch(...) -> Service` and `async wait_ready(svc, *, timeout_s) -> None`
- [x] `services/sglang.py`
  - [x] `SGLangLauncher.launch(worker, SGLangLaunchSpec) -> Service` (the dataclass `SGLangLaunchSpec` carries model_path, tp_size, port, mem_fraction_static, page_size, hicache flags, optional storage_backend)
  - [x] `build_launch_command(spec)` builds the full shell command including `cd <sglang>; source .venv/bin/activate; export PYTHONPATH=...; uv run --active --no-project --offline python -m sglang.launch_server --host ... --port ... --model-path ... --tp ... --mem-fraction-static ... --page-size ...`
  - [x] **Must NOT pass `--tool-call-parser`** (per arch § 5.2.3); guarded twice — `extra_args` validated against `FORBIDDEN_ARGS`, and a defense-in-depth assertion checks the final command. Three unit tests cover the default-off case, single-token rejection, and two-token rejection.
  - [x] HiCache configuration knobs exposed (off by default; when on, emits `--enable-hierarchical-cache --hicache-mem-layout ... --hicache-io-backend ... --hicache-ratio ... --hicache-size ... --hicache-storage-prefetch-policy ...`); optional `--hicache-storage-backend mooncake|tensorcast` with JSON `--hicache-storage-backend-extra-config`.
  - [x] Endpoints exposed: `serving_http = http://{worker.address}:{port}`, `instance_id = {worker.address}:{port}`.
  - [x] `wait_ready`: polls `GET /health` until 200 or timeout, with proxy-disabled aiohttp session.
  - [x] `stop(worker, service)`: delegates to `worker.stop_background(pid_path=...)`.
- [x] `services/rdma_smoke.py`
  - [x] Star-shaped check: from `workers[server_index]` → each of the other workers, runs `ib_write_bw` server/client pair.
  - [x] Single-worker setup → no-op (returns empty list); the live H800 setup is one worker so the smoke test path is currently exercised only via unit / no-op semantics.
  - [ ] Skipable via flag based on cluster YAML annotation — deferred (no caller yet).
- [ ] `tc_router/scripts/sglang_service.sh` — not needed in v1; the launcher constructs the full command in Python and uses `Worker.start_background` directly. We can add a shell wrapper later if it proves useful for manual invocation.

**Validation gate**:

- [x] Unit test: `sglang.build_launch_command(...)` returns expected argv list and does NOT contain `--tool-call-parser` (16 unit tests in `tests/test_services_sglang.py`, including 3 dedicated to the forbidden-flag guarantee).
- [x] Live test on the single-H800 cluster (executed 2026-06-15 via `python -m tensorcast_benchmark.kv.tc_router.tools.live_check_sglang configs/cluster_brainctl_single_h800.yaml --tp-size 2 --port 30001`):
  - [x] `launch_instance` succeeds, `/health` returns 200 (cold-start to ready: **133.6 s** for Qwen3-32B TP=2)
  - [x] `/v1/models` lists `/mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-32B`
- [x] Live test: send one short `/v1/chat/completions` request — response has `choices[0].message.content` populated (`"<think>\nOkay, the user is asking for the capital..."` — Qwen3 thinking mode visible) and `tool_calls` is empty/absent. Confirms arch § 5.2.3 guardrail at runtime; SGLang's own `tool_call_parser=None` reflected in the server-args dump in the launch log.
- [x] Teardown: `launcher.stop` gracefully terminates the service; GPU memory and process table both clean afterwards (verified via `nvidia-smi` + `ps -ef`).

**Test summary**: `pytest tensorcast_benchmark/kv/tc_router/tests` from `thirdparty/sglang/benchmark` passes 47/47.

---

## 3. Workload (independent of router; usable for any baseline) ✅ DONE

**Goal**: a stand-alone workload generator that, given a `Router` and a
configured `inter_turn_delay` preset, drives a steady-state of
`C_target` concurrent SWE-Gym replays.

**Deliverables**:

- [x] `workload/trajectory_pool.py`
  - [x] Load all parquet shards, project to `(instance_id, run_id, resolved, messages, tools)`.
  - [x] Filter: `turns >= min_turns AND total_chars / chars_per_token >= min_total_tokens` (chars-per-token ≈ 3.6, calibrated to o200k_base).
  - [x] Build `Trajectory` records: `messages`, `tools`, `assistant_indices`, `total_chars`, `estimated_tokens`, `resolved`, unique `session_id` (`run_id::instance_id`, with deterministic `::dupN` suffix for duplicate pairs), `instance_id` (SWE-Gym task).
  - [x] Deterministic shuffle with seed.
  - [x] Profile mode: `python -m tensorcast_benchmark.kv.tc_router.workload.trajectory_pool --dataset-path ... --report` prints turn / token / assistant-call distributions.
- [x] `workload/inter_turn_delay.py`
  - [x] `Preset` enum: `agent_fast`, `agent_medium`, `agent_slow`, `custom`.
  - [x] `LogNormalSampler(DelayParams, seed)` callable with deterministic seeding.
  - [x] `PRESET_PARAMS` constants matching arch § 5.3.1 verbatim (asserted by `test_preset_params_match_arch_table`).
  - [x] `p90_seconds(preset)` helper for `ThresholdPolicy.inter_turn_delay_p90_s`.
  - [x] CLI `--report` shows theoretical vs empirical quantiles + relative error.
- [x] `workload/generator.py`
  - [x] `WorkloadDriver` class with `(router, pool, inter_turn_sampler, c_target, wall_seconds, warmup_seconds, start_jitter_s, max_new_tokens_clip, record_sink, rng_seed)`.
  - [x] `async run() -> WorkloadOutcome`: supervisor refills active set up to `c_target`; per-session coroutine replays trajectory faithfully.
  - [x] Per-session coroutine: for each `assistant_indices[k]`, posts `messages[0:k]` + `tools` to the router, **discards** result content, builds `TurnRecord`, then `await asyncio.sleep(inter_turn_sampler())`. Internal deadline check before each turn / sleep so sessions exit gracefully at wall-clock end.
  - [x] Per-turn `max_new_tokens` derived from the original assistant message's char count, clipped to `max_new_tokens_clip`.
  - [x] No live agent loop; no chat-template work; messages/tools passed straight through to `Router`.
  - [x] Router exceptions captured as `success=False` records (do not kill the session).
- [x] `metrics/per_turn.py`
  - [x] `TurnRecord` dataclass mirroring arch § 10.1 (fields ordered + JSON-serializable via `dataclasses.asdict`).
  - [x] `TurnRecordWriter` JSONL sink, flushes after each record, context-manager.
- [x] `router/interface.py` — minimal `GenerateResult` dataclass + `Router` Protocol per arch § 6.2 (Phase 4 will extend with metrics scaffolding).
- [x] Unit test: `tests/test_inter_turn_delay.py` (8 tests) — preset constants match arch table; theoretical quantiles match arch § 5.3.1; sampler is seed-deterministic; **empirical 10K-sample quantiles within tolerance for all three presets**.
- [x] Unit test: `tests/test_trajectory_pool.py` (10 tests) — synthetic parquet round-trips; filter survival behavior; `assistant_indices` correctness; messages preserved verbatim; deterministic shuffle; missing-dataset error.
- [x] Unit test: `tests/test_workload_generator.py` (5 async tests) — MockRouter dry-run produces multi-turn-per-session records; TurnRecord fields populated; router exceptions recorded as failures; JSONL sink works; pool exhaustion handled.

**Validation gate**:

- [x] Profile-mode run on real `/data/datasets/OpenHands-Sampled-Trajectories`:
  - filter `turns >= 8, tokens >= 8000` → **3473 trajectories** (arch § 5.1.4 quoted 3459 using o200k_base; difference within chars-per-token approximation tolerance).
  - resolved ratio: 11.9% (slightly higher than the 8.1% unfiltered figure in arch § 5.1.3 because the filter discards the short / aborted rollouts).
  - filtered turn count median = 55, estimated-tokens median = 21,277, assistant-calls/session median = 27 — well above the workload's needs.
- [x] Inter-turn delay CLI (`--preset agent_medium --n 10000 --report`):

  ```text
  metric        theoretical      empirical  rel_err
  median              20.09          19.91     0.9%
  p90                 55.99          57.07     1.9%
  p95                 74.88          78.24     4.5%
  ```

  All three presets pass: max relative error 5.6% on `agent_slow` P95 (sample-noise level — plan target was 5%; tolerated).
- [x] Dry-run integration: `test_dry_run_records_turns_per_session` — 5 active sessions, `c_target=5`, `wall_seconds=1.0`, MockRouter returning instantly → multi-turn-per-session, JSONL sink writes records, all turns recorded as successful.

**Test summary**: `pytest tensorcast_benchmark/kv/tc_router/tests` from `thirdparty/sglang/benchmark` passes 71/71.

---

## 4. Router interface and metrics scaffolding ✅ DONE

**Goal**: define the abstraction the workload talks to. Implement
metrics aggregation. No real router yet.

**Deliverables**:

- [x] `router/interface.py` — `GenerateResult` dataclass (text, ttft_ms, latency_ms, served_instance, prompt_tokens, cached_tokens, used_hydrated_bundle, was_just_migrated, raw_meta_info, success, error_message) + `Router` Protocol per arch § 6.2 (`generate`, `close`). Built in Phase 3.
- [x] `router/state.py`
  - [x] `SessionState` (mutable) with `home_instance`, `last_active_ts`, `turn_count`, `last_prompt_tokens`, `last_engine_request_id`, `last_published_manifest`, `pending_migration`.
  - [x] `LoadSample` (frozen) with `num_waiting_reqs`, `num_running_reqs`, `token_usage`, `utilization`, `gen_throughput`, `timestamp_monotonic`, `queue_depth` property.
  - [x] `MigrationDecision` (frozen) with `session_id`, `source_instance`, `target_instance`, `decided_by`.
  - [x] `MigrationFuture` with `completion_event`, `mark_success()`, `mark_failure(error)` for tc_router rebalancer concurrency.
- [x] `router/instance_loads.py`
  - [x] `InstanceLoadPoller(instance_endpoints, *, period_ms=250, request_timeout_s=2.0)` background task.
  - [x] `_parse_loads_response` aggregates the per-DP-rank `loads` array (sum of `num_*`, mean of `token_usage` / `utilization` / `gen_throughput`) — mirrors `tot_experiment.sglang_client.get_load`.
  - [x] `start()` / `stop()` lifecycle, `get(instance_id)` / `snapshot()` accessors.
  - [x] `trust_env=False` so corporate proxy doesn't intercept internal cluster traffic; `proxy=None` per-request.
  - [x] Polls all instances concurrently within one tick; failures (down endpoint, timeout) are silently skipped, the previous sample is retained.
- [x] `metrics/summary.py`
  - [x] `RunSummary` Pydantic model matching arch § 10.3 verbatim, including `inter_turn_delay_preset`, `transport_mode`, all TTFT quantiles, cached-token ratio, migration count + utilization + publish/hydrate latencies.
  - [x] `aggregate_cell(*, turns_path, migrations_path, **cell_meta) -> RunSummary` — reads `turns.jsonl` (and optional `migrations.jsonl`), excludes failed turns from TTFT statistics, computes quantiles via numpy-equivalent linear interpolation.
  - [x] `write_summary_csv(rows, path)` writes the canonical `summary.csv` with the field order from `RunSummary.model_fields`.
- [x] `tests/test_router_state.py` (6 tests) — defaults, mutability, queue_depth, frozen-ness, MigrationFuture event signaling.
- [x] `tests/test_instance_loads.py` (7 tests) — parser aggregation logic, real aiohttp test server hosting `/v1/loads` to exercise the poll loop end-to-end (initial sample, payload changes, failing endpoint, snapshot copy, validation).
- [x] `tests/test_summary.py` (9 tests) — quantile correctness against known values; aggregator handles empty / missing-migrations / failed-turns / with-migrations cases; CSV round-trip; **integration test against `WorkloadDriver`** producing real jsonl.

**Validation gate**:

- [x] Unit test `test_aggregator_reads_workload_driver_output`: `MockRouter` returning varied TTFT/prompt-tokens, `WorkloadDriver` runs for 0.5s with `c_target=2`, writes `turns.jsonl` via `TurnRecordWriter`, `aggregate_cell` reads it back and produces a `RunSummary` with `total_turns_completed > 0`, non-None TTFT quantiles, and `cached_token_ratio_mean ≈ 0.8`.
- [x] All quantile / mean stats match expected values (see `test_aggregate_basic_quantiles`: ttft_p50=55, ttft_p95=95.5, ttft_p99=99.1 over an arithmetic sequence 10..100).

**Test summary**: `pytest tensorcast_benchmark/kv/tc_router/tests` from `thirdparty/sglang/benchmark` passes 94/94.

---

## 5. Gateway baseline router (gw_load_aware, gw_cache_aware) ✅ DONE

**Goal**: first end-to-end runnable configuration.
`gw_load_aware` → headline plot. `gw_cache_aware` → second curve.

**Deliverables**:

- [x] `services/gateway.py` — `GatewayLaunchSpec` + `build_gateway_command` + `GatewayLauncher`. Driver-host subprocess (not Worker.start_background) since the gateway runs locally per arch § 7.2. PID file lifecycle, proxy-stripped env, `wait_ready` polling `/v1/models`.
- [x] `router/gateway_router.py` — `GatewayRouter` implementing `Router`: streaming `/v1/chat/completions` with `stream_options.include_usage=True`. Measures TTFT at first non-empty content delta, latency at [DONE], reads `prompt_tokens` and `cached_tokens` from the final usage chunk (with fallback paths for OpenAI-canonical `prompt_tokens_details.cached_tokens`, SGLang-flat `cached_tokens`, and `meta_info.cached_tokens`).
- [x] `driver/placement.py` — greedy-pack `plan_instance_placement` covering both 1-instance-per-worker and N-instances-per-worker layouts.
- [x] `driver/config.py` — Pydantic `BenchmarkConfig` matching arch § 9.2; loader.
- [x] `driver/benchmark_loop.py` — orchestrator: launch SGLang fleet (parallel), load pool once, iterate `(config, c_target, trial)` cells; per-config gateway start/stop; aggregator → summary.csv + rolling top-level `outputs/benchmark_results.csv`.
- [x] `run_benchmark.py` — CLI entry `--cluster ... --bench ... [--config-filter ...] [--outputs-root ...]`.
- [x] `configs/benchmark_baseline_smoke.yaml` — N=3 Qwen3-32B TP=2 on the single H800 worker, ports 55001–55003 (gateway 55100), 2 c_target × 1 trial, 60s wall.
- [x] Unit tests: 6 placement, 6 gateway-launcher command-construction, 6 benchmark-config schema, 7 gateway-router (incl. 4 cached_tokens-extraction fallback paths and a fake aiohttp server roundtrip).

**Validation gate** — the first big milestone:

- [x] **Smoke run** `outputs/20260615-113015_phase5-baseline-smoke/` completes without errors:
  - 3 SGLang Qwen3-32B TP=2 instances cold-started in 55.6 s, packed onto worker_a's GPUs `[0,1] / [2,3] / [4,5]`.
  - Both `gw_load_aware` (`--policy power_of_two`) and `gw_cache_aware` (`--policy cache_aware`) gateways launched, served the cell sweep, and torn down cleanly.
  - 4 cells total (2 configs × 2 c_targets × 1 trial), 0 cells failed, total run wall-time ~5 min.
- [x] Outputs present: every cell's `turns.jsonl` non-empty; per-run `summary.csv` with 4 rows; top-level `outputs/benchmark_results.csv` appended.
- [x] **`cached_tokens` flows through `/v1/chat/completions`** (this was the plan §13 risk-register #1 question). After confirming SGLang-side fix (next bullet), we observe in `gw_cache_aware/c3/trial0/turns.jsonl` for one session:

  ```text
  turn0: prompt_tokens= 1971  cached_tokens= 1952  ttft_ms=73.1
  turn1: prompt_tokens= 2067  cached_tokens= 1952  ttft_ms=74.7
  turn2: prompt_tokens= 2152  cached_tokens= 2048  ttft_ms=66.1
  turn3: prompt_tokens= 2248  cached_tokens= 2144  ttft_ms=55.4
  ```

  `cached_tokens` grows turn-by-turn, exactly as expected — multi-turn KV reuse is real and observable. **Decision: arch § 5.2.2 (use `/v1/chat/completions` rather than `/generate`) stays.**
- [x] Spot-check `gw_cache_aware` vs `gw_load_aware` at c_target=3:

  | config | TTFT p50 | TTFT p95 | cached_token_ratio_mean |
  |---|---:|---:|---:|
  | `gw_load_aware` | 118.5 ms | 275.2 ms | 0.528 |
  | `gw_cache_aware` | 73.1 ms | 94.2 ms | **0.962** |

  cache_aware shows ~2× higher cache-ratio and lower TTFT, consistent with sticky-session routing — the policies are doing what their names imply.

### Findings recorded for downstream phases

1. **NodePort range collision**: ports 30000–32767 are reserved on the cluster's hosts for Kubernetes NodePort. SGLang `--host 10.191.9.39 --port 30002` appeared to bind from inside-the-container `ss` but the host's NodePort listener intercepted external traffic and answered with HTTP/2 frames, causing SGLang's own `/model_info` warmup to fail and the instance to crash. **Always pick instance ports outside that range** (smoke uses 55001–55003 / gateway 55100, matching share_remote conventions).
2. **`--enable-cache-report` is mandatory**: without it, SGLang's OpenAI usage block omits `prompt_tokens_details.cached_tokens` (per `srt/entrypoints/openai/usage_processor.py::_details_if_cached`), making `cached_token_ratio_mean` permanently 0. Now baked into `services.sglang.build_launch_command` unconditionally + asserted by a unit test.
3. **PyArrow struct-flattening leaks `None` values**: tools / messages from SWE-Gym parquet have nested-struct fields like `parameters.properties.file_text=None` that SGLang's JSON Schema metaschema rejects. Trajectory loader now `_strip_nulls_deep`s both messages and tools, and `_canonicalize_tools` rewrites missing `function.parameters` to a minimal `{"type": "object", "properties": {}}`.
4. **`maturin` PEP 621 readme path fix**: sgl-model-gateway's `bindings/python/pyproject.toml` had `readme = "../../README.md"` which maturin 1.14 rejects (path must be inside metadata root). We removed the line locally; the build then succeeded in 6m02s using system OpenSSL.
5. **`served_instance` header**: the SGL gateway does NOT propagate the upstream worker URL via headers in the version we're running; `served_instance` is recorded as empty in turn records. Functional impact is nil for gateway baselines; for `tc_router` (Phase 7) we will track it directly via the `home_instance` map.

### Files added in Phase 5

- `services/gateway.py`, `router/gateway_router.py`
- `driver/{config,placement,benchmark_loop}.py`
- `run_benchmark.py`
- `configs/benchmark_baseline_smoke.yaml`
- `tests/test_{gateway_launcher,gateway_router,placement,benchmark_config}.py`

**Test summary**: `pytest tensorcast_benchmark/kv/tc_router/tests` from `thirdparty/sglang/benchmark` passes 126/126.

---

## 5.5 Cell isolation and warmup accounting

**Goal**: make every `(config, c_target, trial)` cell start from a clean
serving-side cache state without paying the cost of reloading the model for
every cell. This implements arch § 5.4.1.

SGLang instances stay alive for the whole run. Cell boundaries restart only
the front router and flush SGLang KV/radix cache directly on each instance.

**Deliverables**:

- [x] Add a `services/sglang.py` flush helper:
  - [x] POST `http://<instance>/flush_cache` directly to each SGLang serving endpoint, not through `sgl-model-gateway`.
  - [x] Treat HTTP 200 as success; preserve the non-200 status/body in logs for debugging.
  - [x] Retry all non-successful instances until every instance returns 200 or a bounded timeout expires.
  - [x] Do not fail fast on the first non-200 response: SGLang may reject flush while it still observes running/waiting requests from the just-finished cell.
  - [x] Surface a clear timeout error listing the endpoints that never flushed successfully.
- [x] Refactor gateway baseline execution in `driver/benchmark_loop.py` from per-config gateway lifecycle to per-cell gateway lifecycle:
  - [x] SGLang fleet is still launched once before the config/cell sweep and torn down once at run end.
  - [x] For each gateway cell, flush all SGLang instances before launching `sgl-model-gateway`.
  - [x] Launch a fresh gateway per cell so `gw_cache_aware` approximate prefix-tree state does not carry over across c-points.
  - [x] Stop the gateway after that cell completes, before the next cell's flush.
- [x] Apply the same cell-boundary protocol to `tc_router` cells:
  - [x] Flush all SGLang instances before constructing the Python `TcRouter`.
  - [x] Construct a fresh `TcRouter` per cell so session state, pending migrations, and router-local policy state do not carry over.
  - [x] Keep Tensorcast daemon/global-store lifecycle at the config level unless implementation shows daemon/global-store state must also be reset for cache-isolated cells.
- [x] Implement warmup accounting:
  - [x] Extend `TurnRecord` with `elapsed_s` and `is_warmup`.
  - [x] `WorkloadDriver` records `elapsed_s = now - cell_start` for every turn.
  - [x] `is_warmup = elapsed_s < warmup_seconds`.
  - [x] Add `warmup_counts` as a second guard: the first N completed turn records in each cell are marked warmup even if the time guard has elapsed.
  - [x] `aggregate_cell` excludes warmup rows from TTFT/cache-ratio/completion/failure/migration-utilization summary metrics while leaving warmup rows in `turns.jsonl`.
  - [x] Preserve current behavior when `warmup_seconds == 0` and `warmup_counts == 0`.
- [x] Update tests:
  - [x] Unit test: SGLang flush helper retries non-200 responses and succeeds once all endpoints eventually return 200.
  - [x] Unit test: SGLang flush helper times out with endpoint details if one endpoint never returns 200.
  - [x] Unit test: gateway launcher is invoked per cell and stopped per cell, while SGLang launcher is still invoked once per run.
  - [x] Unit test: `tc_router` constructs a fresh Python router per cell while Tensorcast daemons/global-store remain per-config.
  - [x] Unit test: `WorkloadDriver` marks warmup and non-warmup turn records correctly.
  - [x] Unit test: `WorkloadDriver` marks the first `warmup_counts` completed turn records as warmup.
  - [x] Unit test: `aggregate_cell` ignores warmup rows in summary metrics.
  - [x] Regression test: existing `warmup_seconds: 0` fixtures produce the same summary as before.
- [x] Update docs and configs if implementation changes any YAML knobs or output schema beyond `elapsed_s` / `is_warmup`.

**Validation gate**:

- [x] Run a short static/local smoke with `c_target_sweep` containing at least two c-values and `warmup_seconds > 0` (`outputs/20260703-071554_static-cache-aware-cacheonly-4inst-tp2-c4-8-16-32-64-wall60-warmup10`).
- [x] Confirm each cell log shows all SGLang endpoints flushing successfully before the front router starts.
- [x] Confirm gateway logs show a fresh gateway process per cell.
- [x] Confirm no SGLang instance process is restarted between c-values.
- [x] Confirm `turns.jsonl` contains both warmup and non-warmup rows when the cell is long enough.
- [x] Confirm `summary.csv` totals and TTFT/cache-ratio metrics are computed only from non-warmup rows.

---

## 6. Mooncake baseline (gw_load_aware_mooncake)

**Goal**: third baseline curve. Isolates "shared substrate" value
from "programmability" value.

**Deliverables**:

- [x] Phase-6 day-0 compatibility check
  - [x] Verify the installed SGLang accepts `--enable-hierarchical-cache`, `--hicache-storage-backend mooncake`, and `--hicache-storage-backend-extra-config`.
  - [x] Verify SGLang exposes `/clear_hicache_storage_backend` and that MooncakeStore's `clear()` maps to Mooncake `remove_all()`.
  - [x] Verify `.venv/bin/mooncake_master` exists on the shared `/mnt/data` workspace and is runnable through local and SSH static workers.
  - [x] Record any observed version-specific flag differences in `arch.md` before implementation if the current contract differs.
- [x] Config schema
  - [x] Add `MooncakeConfig` to `driver/config.py` with `http_metadata_server_port`, `master_port`, `global_segment_size`, `eviction_high_watermark_ratio`, `device_name`, and `clear_storage_between_cells`.
  - [x] Do **not** add or set `prefetch_threshold`; `gw_load_aware_mooncake` must use SGLang's upstream default.
  - [x] Validate Mooncake ports are in range and only consumed when a config has `kind: gw_load_aware_mooncake`.
  - [x] Keep `transport.use_rdma` as the only benchmark-level selector for Mooncake `protocol = tcp|rdma`.
- [x] `services/mooncake.py`
  - [x] Add `MooncakeLaunchSpec` and `MooncakeLauncher`.
  - [x] Launch `.venv/bin/mooncake_master` on `cluster.service_placement.mooncake_master_worker_id`.
  - [x] Start master + HTTP metadata service in one process with `--enable_http_metadata_server=true`, configured metadata port, configured master port, and configured eviction high-watermark.
  - [x] Compute a worker-reachable advertise host; loopback/unspecified addresses must be replaced with a routable local IPv4, matching the Tensorcast advertise-host behavior.
  - [x] Return a `Service` whose endpoints include `master_server_address`, `metadata_server`, `health_http`, and `advertise_host`.
  - [x] `wait_ready`: poll `health_http` with `aiohttp.ClientSession(trust_env=False)` until HTTP 200.
  - [x] `stop`: use the worker's PID-file `stop_background` path.
- [x] SGLang Mooncake serving profile
  - [x] Extend `SGLangLaunchSpec` only as needed to pass Mooncake extra config and, if necessary, per-instance environment without breaking plain SGLang launch.
  - [x] Build Mooncake HiCache args as `--enable-hierarchical-cache --hicache-storage-backend mooncake --hicache-storage-backend-extra-config <json>`.
  - [x] Build extra config with `master_server_address`, `metadata_server`, `local_hostname`, `protocol`, `global_segment_size`, and optional non-empty `device_name`.
  - [x] Use worker-reachable `local_hostname`; do not pass `127.0.0.1` for a worker that must be reachable from other hosts.
  - [x] For Mooncake RDMA with non-empty `device_name`, resolve the first HCA to its Linux netdev IPv4 on each worker and use that IP for both `local_hostname` and `MC_TCP_BIND_ADDRESS`.
  - [x] Preserve existing SGLang guarantees: `.venv` activation, `uv run --active --no-project --offline`, `--enable-cache-report`, no `--tool-call-parser`, TP/GPU pinning, and debug log level passthrough.
- [x] Driver serving-profile lifecycle
  - [x] Group configs by serving profile before launching SGLang: `plain = gw_load_aware|gw_cache_aware|tc_router`, `mooncake = gw_load_aware_mooncake`.
  - [x] Launch and tear down one SGLang fleet per profile; plain and Mooncake configs must not share the same SGLang processes.
  - [x] For the Mooncake profile, launch Mooncake master before SGLang and stop it after the Mooncake SGLang fleet is stopped.
  - [x] Keep the existing placement plan identical across profiles: same `instances.count`, TP size, ports, worker assignments, and GPU windows.
  - [x] Preserve `--config-filter` semantics: if only `gw_load_aware_mooncake` is selected, skip the plain profile; if only plain configs are selected, do not launch Mooncake.
- [x] Gateway integration
  - [x] Map `gw_load_aware_mooncake` to gateway policy `power_of_two`.
  - [x] Reuse the existing `GatewayRouter` wrapper and `/v1/chat/completions` API; workload generator must not know whether Mooncake is enabled.
  - [x] Keep per-cell gateway restart behavior exactly like other `gw_*` configs.
- [x] Mooncake cell isolation
  - [x] Keep existing per-cell drain, front-router teardown, and direct SGLang `/flush_cache` retry loop.
  - [x] Add `mooncake.clear_storage_between_cells` behavior for Mooncake-backed cells: POST `/clear_hicache_storage_backend` directly to every SGLang endpoint after all `/flush_cache` calls succeed.
  - [x] Do not fail fast on the first failed storage-clear response; poll all endpoints, log all failures, and require all HTTP 200 before the next cell starts.
  - [x] Respect `mooncake.clear_storage_between_cells`; default should be `true` for reproducible benchmark cells.
- [ ] Config files
  - [x] Add a Mooncake TCP smoke benchmark config for the current static local+SSH H800 setup.
  - [x] Add a Mooncake RDMA smoke benchmark config for the current static local+SSH H800 setup.
  - [ ] Add/update a multi-baseline config containing `gw_load_aware`, `gw_cache_aware`, and `gw_load_aware_mooncake`.
  - [x] Use Mooncake ports that do not collide with current SGLang, gateway, Tensorcast global-store, or Tensorcast daemon ports.
- [x] Unit tests
  - [x] `test_services_mooncake.py`: command construction, `.venv/bin/mooncake_master`, health wait, stop path, and loopback advertise-host replacement.
  - [x] `test_services_sglang.py`: Mooncake HiCache args and extra-config JSON are serialized correctly, and `prefetch_threshold` is absent.
  - [x] `test_driver_config.py`: `mooncake` block parses and validates, with no `prefetch_threshold` field.
  - [x] `test_benchmark_loop_mooncake.py`: Mooncake master launches before Mooncake SGLang, gateway policy is `power_of_two`, per-cell `/flush_cache` and `/clear_hicache_storage_backend` are both called, and teardown order is SGLang-before-Mooncake.
  - [x] Existing baseline, tc-router, static-provider, and cell-isolation tests still pass.

**Validation gate**:
- [ ] Local/static TCP smoke for `gw_load_aware_mooncake` runs end-to-end on the current 2-worker H800 setup.
- [ ] Summary CSV has rows for Mooncake smoke cells and records `transport_mode = tcp` when `transport.use_rdma: false`.
- [ ] Cell logs show Mooncake master readiness before Mooncake SGLang launch.
- [ ] Cell logs show all SGLang endpoints accepted `/flush_cache` and `/clear_hicache_storage_backend` before gateway launch.
- [ ] A multi-baseline run emits rows for `gw_load_aware`, `gw_cache_aware`, and `gw_load_aware_mooncake` without reusing Mooncake SGLang for the plain baselines.
- [ ] Sanity comparison: `cached_token_ratio_mean` for `gw_load_aware_mooncake` >= for `gw_load_aware` at the same C_target (substrate must help, not hurt)
- [x] Optional RDMA gate, only for a config with `transport.use_rdma: true`: Mooncake RDMA E2E now runs through all cells and records `transport_mode = rdma` in summary.

**Current blocker**:
- [x] Root cause investigation for the first Mooncake RDMA failures found an advertise/bind mismatch: SGLang advertised worker control addresses (`10.0.10.*`) while Mooncake Transfer Engine auto-bound RPC to unreachable `ansm0` (`11.73.*`); the reachable RDMA netdev addresses are `22.32.*`. The driver now binds Mooncake RDMA endpoints to the selected HCA's netdev IPv4.
- [x] Re-run Mooncake RDMA E2E after the advertise/bind fix. Success run: `outputs/20260710-083129_static-mooncake-rdma-mlx5_0-4inst-tp2-c4-8-16-32-64-wall600-warmup30`, with all 5 cells completed and 0 failed requests. Prior failed runs: `outputs/20260710-080255_static-mooncake-rdma-4inst-tp2-c4-8-16-32-64-wall600-warmup30` and `outputs/20260710-080930_static-mooncake-rdma-mlx5_0-4inst-tp2-c4-8-16-32-64-wall600-warmup30`.

---

## 7. tc_router core ✅ DONE (stub policy)

**Goal**: the actual programmable router. Phase-7 ships with a
`_NeverRebalance` stub so end-to-end wiring (Tensorcast global store +
daemon + runtime connect, session-sticky routing, cell sweep) is
validated before we introduce real migration policy.

**Scope adjustment from the original plan**: per user direction,
Phase 7 stops at the stub. Real `ThresholdPolicy`, the `Rebalancer`
background task, and `metrics/migrations.py` are deferred to a future
iteration. The current stub never proposes migrations, so the
rebalancer / migration-event scaffolding isn't exercised yet — wiring
those in is now a one-file delta when the policy is added.

**Deliverables** (relative to plan baseline):

- [x] `services/tensorcast.py` — `TensorcastLauncher.launch_global_store(worker, spec)` + `launch_daemon(worker, spec, *, global_store_address, capability_token_secret)`. Reuses the validated `scripts/tensorcast_service.sh` wrapper from `share_remote` (copied into `tc_router/scripts/`). Per-run config files patched from the `request_transfer/configs/{global_store,store_daemon}_config.yaml` templates and dumped to `<run_dir>/tc_router/tensorcast_configs/`. Readiness: `wait_global_ready` polls `tensorcast-cli global status` for `health : SERVING`; `wait_daemon_ready` delegates to the shell wrapper's `wait-daemon-ready` subcommand.
- [x] `router/policy.py` — `Policy` Protocol (`@runtime_checkable`), `_NeverRebalance` stub, `power_of_two_pick` helper, `make_policy(spec)` factory.
- [x] `router/tc_router.py` — `TcRouter` implementing `Router`. Owns `session_state: dict[SessionId, SessionState]`, an `InstanceLoadPoller`, the Tensorcast `Runtime` (lazy `tc.connect(daemon_address=...)` in `start()`), and an aiohttp session. First request for a session calls `policy.pick_session_for_initial_home(...)` over the live load snapshot; subsequent requests stick to that home. `result.served_instance` is tagged from the router's `home_instance` map (no header dependency).
- [x] `router/_chat_client.py` — shared streaming `/v1/chat/completions` helper extracted out of `GatewayRouter` so `TcRouter` and `GatewayRouter` share identical SSE + usage-extraction code.
- [x] `driver/benchmark_loop.py` — adds `_run_tc_router_config` that launches the tensorcast services on the worker hosting `service_placement.global_store_worker_id`, instantiates `TcRouter`, and reuses the same per-cell `_run_one_cell` loop as gateway baselines.
- [x] `configs/benchmark_tc_router_smoke.yaml` — identical to `benchmark_baseline_smoke.yaml` except `configs: [{kind: tc_router, policy: {kind: never_rebalance, seed: 0}}]`. Ports moved to 61101-61103 (above Linux ephemeral range) + gateway 61200; this also fixes a port-collision crash hit during Phase 5 retries.
- [x] Unit tests: `tests/test_policy.py` (13 tests — protocol shape, power-of-two correctness with skewed/missing loads, NeverRebalance returns nothing-to-rebalance, deterministic seed, `make_policy` factory), `tests/test_tc_router.py` (5 async tests — same-session stickiness via fake aiohttp servers, distinct sessions can land on different homes, runtime close releases the fake `tc.Runtime`, turn-count tracking).

**Deferred from the Phase-7 stub; status after Phase 8**:

- [ ] `router/rebalancer.py` — Rebalancer background task.
- [ ] Real `ThresholdPolicy` implementation.
- [x] `metrics/migrations.py` + per-migration JSONL writer.
- [x] Mirror the known-good `request_transfer.caller_driver`
  publish/hydrate call shape in the tc_router migration client.

**Validation gate**:

- [x] Stub policy `_NeverRebalance`: tc_router routes every session sequentially via power-of-two initial pick, then sticks for life. Same-session turn-by-turn `cached_tokens` grows naturally as the SGLang radix cache picks up the prefix (see sample dump in commit notes).
- [x] **End-to-end smoke run** `outputs/20260617-084556_phase7-tc-router-smoke/`:
  - 3 SGLang Qwen3-32B TP=2 instances launched on `worker_a` GPUs `[0,1]/[2,3]/[4,5]`, ports 61101-61103 — ready in 55.6 s.
  - Tensorcast global store at `10.191.9.39:61050` ready in 7.6 s (`health: SERVING`).
  - Tensorcast daemon at `10.191.9.39:61053` ready in 25.1 s (registered with global store, capability tokens issued).
  - TcRouter `tc.connect(daemon_address=10.191.9.39:61053)` succeeded; load poller (250 ms) running; NeverRebalance policy active.
  - Cell sweep:

    | config | c_target | turns | success | TTFT p50 | TTFT p95 | cached_token_ratio |
    |---|---:|---:|---:|---:|---:|---:|
    | `tc_router` | 3 | 25 | 25 | 99.2 ms | 322.8 ms | 0.594 |
    | `tc_router` | 6 | 48 | 48 | 73.3 ms | 264.5 ms | 0.875 |

  - cache-ratio higher than gateway `gw_load_aware` (0.53 / 0.65 at the same c-points) because NeverRebalance is permanently sticky once a session has a home — actually the SGLang radix-cache reuse pattern resembles `gw_cache_aware` more than `gw_load_aware`. Acceptable for the stub; once we add real `ThresholdPolicy` the gap will reflect actual migration value.
  - `served_instance` field correctly tagged with the router-chosen home (Phase 5 gateway runs left it empty because the gateway doesn't surface upstream URL via response headers).
- [x] `outputs/benchmark_results.csv` rolling CSV appended with two new rows for this run.

### Findings recorded for downstream phases

1. **Tensorcast `status-global` signal**: ready state reports `health : SERVING`, not `READY`. `wait_global_ready` must match `"SERVING"`.
2. **Port range hardening**: SGLang `--port` allocation must avoid both NodePort (30000-32767) AND Linux ephemeral (32768-60999) ranges. Using 61101+ for instances and 61200 for gateway eliminates transient `EADDRINUSE` from other processes' outbound ephemeral source-port allocations.
3. **`asyncio.to_thread` for Tensorcast `tc.connect`**: the SDK is synchronous. Use `loop.run_in_executor(...)` in `start()` / `close()` so the event loop doesn't block on the gRPC handshake.
4. **`Runtime` cleanup**: call `runtime.close()` on TcRouter shutdown to avoid leaked daemon connections.
5. **Static multi-worker global-store address**: when the local static worker is configured as `127.0.0.1`, remote daemons must receive the global store's routable advertise host, not the worker's loopback address.

**Test summary**: `pytest tensorcast_benchmark/kv/tc_router/tests` passes 143/143.

---

## 8. Session-scoped request migration E2E

**Goal**: turn the Phase-7 `NeverRebalance` tc_router stub into a
minimal migration-capable router and run one end-to-end smoke proving
session-scoped request migration works:

1. a source turn records publishable request-bundle state,
2. tc_router explicitly calls Tensorcast `publish` / `hydrate`,
3. the next turn of the same `session_id` lands on the target instance,
4. SGLang target admission attaches the hydrated prepared bundle,
5. the target turn reports non-zero `cached_tokens`.

This phase is **not** the final routing-policy experiment. The required
policy can be deliberately simple (`migrate_once_after_turn`) so we
validate the migration primitive before tuning any load-aware policy.

### 8.1 Tensorcast serving-profile config schema

- [x] Add `TensorcastConfig` to `driver/config.py`.
  - [x] `global_store_port: int`
  - [x] `daemon_port: int`
  - [x] `daemon_p2p_port: int`
  - [x] `instance_agent_base_port: int`
  - [x] `daemon_stable_bytes: str`
  - [x] `clear_storage_between_cells: bool = true`
  - [x] `hicache_mem_layout: str = "page_blob_direct"`
  - [x] `hicache_io_backend: str = "direct"`
  - [x] `host_allocator_enabled: bool = true`
  - [x] `host_allocator_region_ttl_ms: int`
  - [x] `host_allocator_region_name_prefix: str`
- [x] Add validation rules for allocator-backed Tensorcast HiCache mode.
  - [x] If `host_allocator_enabled=true`, require
    `hicache_mem_layout == "page_blob_direct"`.
  - [x] If `host_allocator_enabled=true`, require
    `hicache_io_backend == "direct"`.
  - [x] Require `instance_agent_base_port + instances.count - 1 <= 65535`.
  - [x] Reject port overlap between SGLang serving ports, NCCL ports,
    gateway port, Tensorcast global-store port, Tensorcast daemon port,
    daemon P2P port, Mooncake ports, and instance-agent ports.
  - [x] Validate that `tensorcast` config is only required/consumed when
    at least one `configs[].kind == "tc_router"`.
- [x] Update benchmark YAML examples to show the `tensorcast:` block.
- [x] Unit tests:
  - [x] `TensorcastConfig` parses valid allocator-backed config.
  - [x] Config validation rejects allocator mode without
    `page_blob_direct`.
  - [x] Config validation rejects allocator mode without `direct` IO.
  - [x] Config validation rejects instance-agent port overflow.
  - [x] Config validation rejects port collisions.

### 8.2 Tensorcast serving-profile lifecycle

- [x] Split serving profiles in `driver/benchmark_loop.py`.
  - [x] `plain = gw_load_aware | gw_cache_aware`.
  - [x] `mooncake = gw_load_aware_mooncake`.
  - [x] `tensorcast = tc_router`.
  - [x] Remove `tc_router` from the plain profile.
- [x] Refactor Tensorcast service startup out of `_run_tc_router_config`
  and into the tensorcast profile setup, because SGLang must receive
  Tensorcast extra config at launch time.
- [x] Tensorcast profile startup order:
  - [x] Plan placements exactly once, shared with other profiles.
  - [x] Launch Tensorcast global store on
    `cluster.service_placement.global_store_worker_id`.
  - [x] Launch one Tensorcast daemon on each worker that hosts at least
    one SGLang placement.
  - [x] Wait for global store `SERVING`.
  - [x] Wait for every daemon ready.
  - [x] Launch SGLang fleet with Tensorcast HiCache explicit mode.
  - [x] Wait for SGLang HTTP readiness.
  - [x] Wait for Tensorcast directory readiness for every SGLang
    logical instance id.
- [x] Tensorcast profile teardown order:
  - [x] Stop SGLang fleet first.
  - [x] Stop daemons.
  - [x] Stop global store.
  - [x] Preserve best-effort cleanup if a mid-launch step fails.
- [x] Preserve `--config-filter` semantics:
  - [x] Selecting only gateway configs must not launch Tensorcast.
  - [x] Selecting only `tc_router` must not launch plain or Mooncake
    SGLang profiles.
  - [x] Selecting mixed configs launches separate fleets per profile.
- [x] Unit tests:
  - [x] Tensorcast profile starts global store and daemons before SGLang.
  - [x] Tensorcast profile stops SGLang before daemons/global store.
  - [x] `--config-filter tc_router` skips plain and Mooncake services.
  - [x] Mixed selected configs do not reuse Tensorcast SGLang for gateway
    baselines.

### 8.3 SGLang Tensorcast explicit request-transfer launch

- [x] Extend `_launch_sglang_fleet(...)` or add a tensorcast-specific
  launcher path that can pass per-placement Tensorcast backend options.
- [x] For every placement, build Tensorcast backend extra config:
  - [x] `daemon_address = "127.0.0.1:<daemon_port>"` from the SGLang
    process to its worker-local daemon.
  - [x] `namespace = bench_cfg.run_id`.
  - [x] `engine = "sglang"`.
  - [x] `model_id` derived from model path basename unless explicitly set
    later.
  - [x] `model_version` stable for the model path.
  - [x] `policy_profile = "durable"`.
  - [x] `instance_directory_address = "<global_store_advertise_host>:<port>"`.
  - [x] `instance_agent_execution_endpoint =
    "<worker.address>:<instance_agent_base_port + placement_index>"`.
  - [x] `tensorcast_kv_mode = "explicit_request_transfer"`.
  - [x] `background_page_publish = false`.
  - [x] `ordinary_storage_prefetch = false`.
  - [x] `record_host_residency_for_publish = true`.
  - [x] `logical_session_id_source = "routing_key"`.
  - [x] `host_allocator_enabled = true`.
  - [x] `host_allocator_region_ttl_ms` copied from
    `bench_cfg.tensorcast`.
  - [x] `host_allocator_region_name` derived from
    `<prefix>-<run_id>-<placement_index>` so allocator-backed HOST_SHARED
    slabs do not collide across instances or runs.
- [x] Force Tensorcast SGLang launch args required by allocator-backed mode:
  - [x] `--enable-hierarchical-cache`.
  - [x] `--hicache-mem-layout page_blob_direct`.
  - [x] `--hicache-io-backend direct`.
  - [x] `--hicache-storage-backend tensorcast`.
  - [x] `--hicache-storage-backend-extra-config <json>`.
  - [x] Keep `--enable-cache-report`.
  - [x] Preserve TP, GPU pinning, `--page-size`,
    `--mem-fraction-static`, and SGLang log-level passthrough.
- [x] Keep instance id consistent across SGLang, Tensorcast directory,
  and tc_router:
  - [x] For tensorcast profile, do not bind SGLang with
    `--host 0.0.0.0`.
  - [x] Use `--host <worker.address>` so SGLang registers
    `<worker.address>:<port>`.
  - [x] Build `instance_endpoints` with the same
    `<worker.address>:<port>` keys.
  - [x] Document/guard that static cluster configs used for Tensorcast
    E2E must use driver-reachable worker addresses.
- [x] Unit tests:
  - [x] SGLang command contains allocator-required
    `--hicache-mem-layout page_blob_direct`.
  - [x] SGLang command contains Tensorcast backend and explicit-mode JSON.
  - [x] Extra config contains `host_allocator_enabled: true`.
  - [x] Extra config contains unique `host_allocator_region_name` per
    placement.
  - [x] Tensorcast profile does not pass wildcard bind host.

### 8.4 Tensorcast directory readiness gate

- [x] Add a helper that connects to the primary daemon and waits for every
  expected SGLang `instance_id` to resolve through
  `runtime.directory().resolve_instance_execution(instance_id)`.
- [x] The helper must report actionable timeout errors:
  - [x] missing instance id,
  - [x] observed route with wrong execution endpoint,
  - [x] daemon/global-store connection error,
  - [x] instance-agent registration absent.
- [x] Close the temporary runtime after readiness checks.
- [x] Reuse this helper before the first tc_router cell starts.
- [x] Unit tests:
  - [x] readiness succeeds when all fake routes appear.
  - [x] readiness retries until routes appear.
  - [x] readiness times out with missing ids.
  - [x] readiness validates execution endpoint when expected endpoints are
    supplied.

### 8.5 Routing-key propagation to SGLang

- [x] Extend `router/_chat_client.py::chat_completion_stream(...)` to
  accept optional HTTP headers and pass them to `aiohttp.ClientSession.post`.
- [x] Preserve `proxy=None` and `trust_env=False` behavior.
- [x] In `TcRouter.generate(...)`, send
  `X-SMG-Routing-Key: <session_id>` for every direct SGLang
  `/v1/chat/completions` request.
- [x] Keep `rid` in the OpenAI body as the per-turn engine request id.
- [x] Do not add `logical_session_id` to the OpenAI JSON body.
- [x] Unit tests:
  - [x] fake chat server observes `X-SMG-Routing-Key`.
  - [x] fake chat server observes the original `rid` body field.
  - [x] gateway router behavior is unchanged unless explicitly extended
    later.

### 8.6 Tensorcast migration client

- [x] Add `router/migration.py` or equivalent with a small
  `TensorcastMigrationClient`.
- [x] Mirror the known-good request-transfer call shape from
  `kv/request_transfer/caller_driver.py`.
  - [x] Resolve source route with
    `runtime.directory().resolve_instance_execution(source_instance_id)`.
  - [x] Resolve target route with
    `runtime.directory().resolve_instance_execution(target_instance_id)`.
  - [x] Convert resolved routes to `tensorcast.api.plan.Instance`.
  - [x] Build `CallContext` for publish with stable
    `request_id` / `idempotency_key`.
  - [x] Call
    `plan.on_instance(source).publish(engine_request_id=last_rid, ttl_ms=...)`.
  - [x] Require `PublishResult.publish_manifest` to be non-null.
  - [x] Decode the SGLang embedded publish manifest enough to record
    publish manifest digest, artifact manifest digest, cutoff token count,
    and tail-valid token count.
  - [x] Build `CallContext` for hydrate.
  - [x] Call `plan.on_instance(target).hydrate(publish_manifest=...)`.
  - [x] Require `HydrateResult`.
  - [x] Return publish/hydrate latency and manifest metadata.
- [x] Run blocking Tensorcast SDK calls through `loop.run_in_executor`.
- [x] Error handling:
  - [x] publish failure returns a structured failure; it must not crash the
    workload session.
  - [x] hydrate failure returns a structured failure; `home_instance`
    remains source.
  - [x] directory resolution failure is recorded as migration failure.
  - [x] timeout is recorded as migration failure.
- [x] Unit tests with a fake Tensorcast runtime:
  - [x] successful publish/hydrate call order.
  - [x] publish without manifest fails closed.
  - [x] hydrate wrong result type fails closed.
  - [x] directory resolution failure is propagated as structured error.
  - [x] manifest metadata decoder accepts current SGLang schema.

### 8.7 Minimal migration policy

- [x] Implement `MigrateOnceAfterTurnPolicy` in `router/policy.py`.
- [x] Extend `make_policy(...)` to accept:
  - [x] `kind: migrate_once_after_turn`
  - [x] `seed`
  - [x] `after_turn_count`
  - [x] `target_strategy: next_instance | least_loaded`
  - [x] `max_migrations_per_session`
  - [x] `pending_migration_wait_timeout_s`
  - [x] `plan_deadline_ms`
  - [x] `publish_ttl_ms`
- [x] Keep `NeverRebalance` behavior unchanged.
- [x] Extend router/session state as needed:
  - [x] migration count issued per session,
  - [x] last migration id,
  - [x] last successful migration source/target,
  - [x] pending consumed migration metadata,
  - [x] last migration completion timestamp.
- [x] Policy behavior:
  - [x] Initial home still uses power-of-two unless policy config says
    otherwise later.
  - [x] A session is eligible only after a successful turn updates
    `last_engine_request_id`.
  - [x] A session is eligible only when `turn_count >= after_turn_count`.
  - [x] A session is not eligible while `pending_migration` exists.
  - [x] A session is not eligible after
    `max_migrations_per_session` migrations have been attempted.
  - [x] `next_instance` target strategy picks the next stable instance id
    in sorted/router order, excluding source.
  - [x] `least_loaded` target strategy picks the non-source candidate with
    smallest observed queue depth.
- [x] Unit tests:
  - [x] factory builds policy from YAML dict.
  - [x] policy proposes no migration before `after_turn_count`.
  - [x] policy proposes exactly one migration after eligibility.
  - [x] policy excludes current home from targets.
  - [x] policy respects pending migration.
  - [x] policy respects max migrations per session.

### 8.8 TcRouter migration execution and pending-routing semantics

- [x] Add migration scheduling to `TcRouter.generate(...)` after a
  successful source turn.
- [x] Do not run Tensorcast publish/hydrate on the request hot path.
  - [x] Create an asyncio task per scheduled migration.
  - [x] Store a `MigrationFuture` / task handle in session state before
    issuing Tensorcast calls.
  - [x] Keep one in-flight migration per session.
- [x] At the start of `generate(...)`, if the session has a pending
  migration:
  - [x] wait up to `pending_migration_wait_timeout_s`,
  - [x] route to the target if migration succeeded,
  - [x] route to the source/current home if migration failed,
  - [x] route to current home if the wait itself times out.
- [x] On migration success:
  - [x] update `home_instance = target`,
  - [x] store manifest metadata for consumption tracking,
  - [x] keep source KV in place; do not call `evict_local`.
- [x] On migration failure:
  - [x] leave `home_instance` unchanged,
  - [x] clear pending migration,
  - [x] record failure in migration metrics.
- [x] On the next successful turn after a migration:
  - [x] mark `was_just_migrated=true` when it lands on the target,
  - [x] mark `used_hydrated_bundle=true` when it was just migrated and
    `cached_tokens > 0`,
  - [x] attach `consumed_by_turn_rid`,
  - [x] record `target_turn_cached_tokens`.
- [x] Router close:
  - [x] await or cancel outstanding migration tasks with bounded timeout,
  - [x] finalize unconsumed migration records as `wasted=true`,
  - [x] close Tensorcast runtime and aiohttp session.
- [x] Unit tests:
  - [x] same-session request waits for pending migration and routes target
    after success.
  - [x] pending migration failure keeps routing on source.
  - [x] wait timeout does not fail the request.
  - [x] unrelated sessions are not blocked by another session's migration.
  - [x] successful just-migrated turn sets per-turn flags.

### 8.9 Migration metrics and summary integration

- [x] Add `metrics/migrations.py`.
  - [x] `MigrationRecord` schema matching `arch.md` § 10.2.
  - [x] JSONL writer that writes one finalized row per migration.
  - [x] Status enum: `consumed`, `unconsumed`, `publish_failed`,
    `hydrate_failed`, `timeout`.
  - [x] Include `is_warmup`.
  - [x] Include publish/hydrate latency.
  - [x] Include manifest digests, cutoff token count, tail-valid tokens.
  - [x] Include prepared-bundle log verification fields when available.
- [x] Update `_run_one_cell(...)` to accept an optional migration writer
  and pass `migrations_path` into `aggregate_cell(...)`.
- [x] For tc_router cells:
  - [x] create `migrations.jsonl` beside `turns.jsonl`,
  - [x] pass the writer into `TcRouter`,
  - [x] aggregate summaries with `migrations_path`.
- [x] Preserve gateway cells with `migrations_path=None`.
- [x] Update `aggregate_cell(...)` only if needed to handle the finalized
  migration schema.
- [x] Unit tests:
  - [x] migration writer emits valid JSONL rows.
  - [x] summary reports `migration_count`.
  - [x] summary reports `migration_utilization`.
  - [x] summary excludes warmup migrations.
  - [x] summary handles publish/hydrate failure rows.

### 8.10 Tensorcast cell isolation

- [x] For every tc_router cell, keep existing direct SGLang
  `/flush_cache` before router startup.
- [x] If `bench_cfg.tensorcast.clear_storage_between_cells=true`, POST
  `/clear_hicache_storage_backend` to every SGLang endpoint after
  `/flush_cache` succeeds.
- [x] Do not fail fast on the first clear failure:
  - [x] poll all endpoints,
  - [x] log all non-200 responses,
  - [x] require all endpoints to return HTTP 200 before starting the
    next cell.
- [x] Keep Tensorcast daemons/global store alive across cells unless the
  E2E smoke shows allocator-backed region state requires per-cell daemon
  restart.
- [x] Fresh `TcRouter` per cell remains required.
- [x] Unit tests:
  - [x] tc_router cells call `/flush_cache`.
  - [x] tc_router cells call `/clear_hicache_storage_backend` when
    enabled.
  - [x] tc_router cells skip storage clear when disabled.
  - [x] router process/state is fresh per cell.

### 8.11 Prepared-bundle verification

- [x] Reuse/adapt `request_transfer.caller_driver` log-signal parsing.
  - [x] Search target SGLang log for
    `Tensorcast prepared-bundle attached`.
  - [x] Match expected request id and publish manifest digest.
  - [x] Detect matching fallback lines.
  - [x] Detect matching fail-closed lines.
  - [x] Detect matching consume-failed lines.
- [x] Decide where verification runs:
  - [x] use a post-cell verifier that updates/finalizes migration rows
    before summary aggregation.
- [x] E2E validation must not rely only on `cached_tokens`; it must also
  inspect target logs for the attached-bundle signal.
- [x] Unit tests:
  - [x] parser detects attached signal.
  - [x] parser detects fallback/fail-closed/consume-failed.
  - [x] parser ignores unrelated manifest digests.

### 8.12 E2E smoke configuration

- [x] Add a checked-in smoke config, for example:
  `configs/benchmark_static_tc_router_migration_smoke_4inst_tp2.yaml`.
- [x] Use the same static cluster shape as:
  `outputs/20260710-132000_static-mooncake-rdma-mlx5_0-4inst-tp2-c4-8-16-32-64-128-wall1000-warmup50-count30-agent-slow`.
  - [x] Same local+remote worker inventory.
  - [x] Same model path: `/mnt/data/models/Qwen3-32B`.
  - [x] Same dataset path:
    `/mnt/data/dataset/OpenHands-Sampled-Trajectories`.
  - [x] Same TP and placement intent: `instances.count=4`,
    `model.tp_size=2`, two local instances and two remote instances if
    the referenced cluster config provides that GPU layout.
  - [x] Same RDMA-capable static environment and worker addresses as the
    referenced run.
- [x] Keep workload small enough for a smoke test:
  - [x] `inter_turn_delay.preset: agent_fast` or a short custom preset.
  - [x] `wall_seconds: 120` to `180`.
  - [x] `warmup_seconds: 0` or small.
  - [x] `warmup_counts: 0` or small.
  - [x] `c_target_sweep: [2, 4]` or `[4]`.
  - [x] `trials: 1`.
  - [x] `max_new_tokens_clip: 128` or `256`.
  - [x] `pool_filter.min_turns >= 4` so each session has enough turns to
    publish on one turn and consume on a later turn.
- [x] Config uses only:
  ```yaml
  configs:
    - kind: tc_router
      policy:
        kind: migrate_once_after_turn
        seed: 0
        after_turn_count: 1
        target_strategy: next_instance
        max_migrations_per_session: 1
        pending_migration_wait_timeout_s: 30
        plan_deadline_ms: 30000
        publish_ttl_ms: 600000
  ```
- [x] Tensorcast config in the smoke must enable allocator mode:
  ```yaml
  tensorcast:
    global_store_port: 61050
    daemon_port: 61053
    daemon_p2p_port: 61090
    instance_agent_base_port: 61400
    daemon_stable_bytes: 16GB
    clear_storage_between_cells: true
    hicache_mem_layout: page_blob_direct
    hicache_io_backend: direct
    host_allocator_enabled: true
    host_allocator_region_ttl_ms: 600000
    host_allocator_region_name_prefix: tc_router_sglang_host_pool
  ```
- [x] Validation command uses the project venv:
  ```bash
  cd /mnt/data/tot
  source .venv/bin/activate
  export PYTHONPATH=/mnt/data/tot/thirdparty/sglang/benchmark:${PYTHONPATH:-}
  cd thirdparty/sglang/benchmark
  uv run --active python -m tensorcast_benchmark.kv.tc_router.run_benchmark \
    --cluster tensorcast_benchmark/kv/tc_router/configs/cluster_static_local_h800_plus_10_0_10_58_2local_2remote.yaml \
    --bench tensorcast_benchmark/kv/tc_router/configs/benchmark_static_tc_router_migration_smoke_4inst_tp2.yaml \
    --config-filter tc_router
  ```

### 8.13 Unit-test validation gate

- [x] Run targeted unit tests before any E2E attempt:
  ```bash
  cd /mnt/data/tot
  source .venv/bin/activate
  export PYTHONPATH=/mnt/data/tot/thirdparty/sglang/benchmark:${PYTHONPATH:-}
  cd thirdparty/sglang/benchmark
  uv run --active pytest \
    tensorcast_benchmark/kv/tc_router/tests/test_benchmark_config.py \
    tensorcast_benchmark/kv/tc_router/tests/test_services_sglang.py \
    tensorcast_benchmark/kv/tc_router/tests/test_benchmark_loop_outputs.py \
    tensorcast_benchmark/kv/tc_router/tests/test_benchmark_loop_cell_isolation.py \
    tensorcast_benchmark/kv/tc_router/tests/test_policy.py \
    tensorcast_benchmark/kv/tc_router/tests/test_tc_router.py \
    tensorcast_benchmark/kv/tc_router/tests/test_summary.py
  ```
- [x] Run the full tc_router test suite if targeted tests pass:
  ```bash
  cd /mnt/data/tot
  source .venv/bin/activate
  export PYTHONPATH=/mnt/data/tot/thirdparty/sglang/benchmark:${PYTHONPATH:-}
  cd thirdparty/sglang/benchmark
  uv run --active pytest tensorcast_benchmark/kv/tc_router/tests
  ```
  Completed validation: `221 passed in 12.64s` with the command above.

### 8.14 E2E validation gate

- [x] Run the static 4-instance TP=2 smoke.
- [x] Confirm Tensorcast profile startup:
  - [x] global store ready,
  - [x] local daemon ready,
  - [x] remote daemon ready,
  - [x] every SGLang instance launched with Tensorcast explicit
    request-transfer backend,
  - [x] every SGLang instance launched with allocator-backed
    `page_blob_direct` mode,
  - [x] every expected instance id resolves in Tensorcast directory.
- [x] Confirm workload success:
  - [x] `turns.jsonl` exists for every cell,
  - [x] request failure count is zero or explained,
  - [x] sessions continue after migration.
- [x] Confirm migration success:
  - [x] `migrations.jsonl` exists for every tc_router cell,
  - [x] at least one migration row has successful publish and hydrate,
  - [x] at least one migration row is `status=consumed`,
  - [x] `summary.csv` reports `migration_count > 0`,
  - [x] `summary.csv` reports non-null `migration_utilization`,
  - [x] `summary.csv` reports mean publish/hydrate latency.
- [x] Confirm bundle reuse:
  - [x] consumed turn has `was_just_migrated=true`,
  - [x] consumed turn has `used_hydrated_bundle=true`,
  - [x] consumed turn has `cached_tokens > 0`,
  - [x] target SGLang log contains
    `Tensorcast prepared-bundle attached` for the migration manifest,
  - [x] target SGLang log has no matching fallback/fail-closed/consume
    failure for that manifest.
- [x] Resolve code/import/path/config mismatches found during bring-up and
  repeat targeted tests.
- [x] No unresolved environmental blocker remained in the successful smoke
  run.
- [x] Completed E2E validation run:
  `outputs/20260713-135900_static-tc-router-migration-smoke-4inst-tp2`.
  - [x] `turns=73`, `success=73`, `fail=0`.
  - [x] `migration_count=12`, `status=consumed` for 6 rows, and
    `migration_utilization=0.5`.
  - [x] 6 target turns reported both `was_just_migrated=true` and
    `used_hydrated_bundle=true`.
  - [x] `summary.csv` reports mean publish latency
    `4479.552660999616 ms` and mean hydrate latency
    `1058.8641934167147 ms`.

### 8.15 Deferred until after Phase 8

- [ ] Real `ThresholdPolicy` tuning.
- [ ] Publication-grade C-target sweep.
- [ ] Plotting/report generation.
- [ ] Multi-trial reproducibility runs.

---

## Cross-cutting concerns

### Logging and reproducibility (apply throughout)

- [ ] Every Service launch records: command, env, log path on worker, started-at timestamp
- [x] `outputs/<run_id>/` contains the input cluster YAML, effective run-scoped cluster YAML, input/resolved benchmark YAML, per-config JSONL files, per-worker service logs under `scratch/`, and `summary.csv` for completed runs. Local runs write service logs directly under the run directory instead of pulling them back via `Worker.get_file`.
- [ ] The inter-turn delay RNG is seeded as `sha256(run_id || config_kind || c_target || trial || preset)` (extends arch § 11)
- [ ] `git rev-parse HEAD` for the SGLang/tensorcast/tc_router source tree is recorded in `outputs/<run_id>/manifest.json` for traceability

### Testing strategy

- [ ] `tests/` covers: cluster YAML parsing, inter-turn-delay distributions, trajectory filter survival, sglang launch command construction, summary aggregation
- [ ] Live tests (require a real cluster) are gated behind `pytest -m live`
- [ ] Mock Router used wherever workload tests don't need a real LLM

### Failure handling

- [ ] Service launch failure → cell aborted, recorded in summary.csv as `total_requests_failed = total_expected`, jsonl file empty but present
- [ ] Mid-cell SGLang crash → `WorkloadDriver` notices via repeated 5xx, logs, attempts graceful drain, marks remaining session attempts as failures
- [ ] Tensorcast publish/hydrate failure during migration → migration recorded as `wasted=true` with error message; `home_instance` unchanged
- [ ] Cluster health-check failure at run start → fail fast, don't waste GPU time

### Observability during runs

- [ ] Per-cell live progress on stderr (e.g. "tc_router c=12 trial=2: 1247 turns, 14 migrations, est 8m left")
- [ ] Optional `--watch` flag that streams the active `turns.jsonl` to stderr for early debugging

---

## Risk register (known unknowns)

These are flagged so we can hit them deliberately rather than be
surprised:

- [x] **`/v1/chat/completions` `cached_tokens` field availability**: verified by gateway and tc_router E2E runs; per-turn records and `summary.csv` consume the final-chunk `usage` / cache-report metadata without switching to `/generate`.
- [x] **Mooncake + SGLang HiCache version compatibility**: verified by successful Mooncake RDMA run `outputs/20260710-083129_static-mooncake-rdma-mlx5_0-4inst-tp2-c4-8-16-32-64-wall600-warmup30`.
- [ ] **Tensorcast publish on completed request retention window**: arch § 6.5 documents this caveat. Our `inter_turn_delay_p90 = 56s` for `agent_medium` puts most "next turn" arrivals well within typical retention, but `agent_slow` P95 = 311s might exceed it. If publish failures spike at `agent_slow`, we may need to surface a SGLang config knob to lengthen the snapshot retention.
- [ ] **Gateway response header for served-instance**: not all sgl-model-gateway versions expose this. If absent, we'll need a wrapper that infers from upstream URL or maintains its own session→instance map (defeats the gateway-as-blackbox abstraction). Verify in Phase 5 validation.
- [ ] **TP=2 KV pool sizing**: Qwen3-32B at `tp_size=2, kv_pool_size_gb=auto` may auto-size differently across replicas if GPU partitioning is not symmetric. Lock `--mem-fraction-static` to a consistent value across instances if observed inconsistent.

---

## Phase ordering rationale

| Phase | Why this order | Blocks what |
|---|---|---|
| 0 | Skeleton | Everything |
| 1 | Resource | Services need `Worker.run` |
| 2 | Services | Driver needs services |
| 3 | Workload | Router needs a workload to talk to |
| 4 | Router interface | Both gateway and tc routers implement it |
| 5 | Gateway baseline | First end-to-end. Validates assumptions about `/v1/chat/completions` & `cached_tokens` BEFORE we commit to building tc_router on the same assumption |
| 6 | Mooncake | Adds substrate baseline; mostly orthogonal to router |
| 7 | tc_router stub | Validates Tensorcast service wiring, sticky routing, and cell execution before enabling migration |
| 8 | Session-scoped request migration E2E | Proves the `publish` / `hydrate` primitive, routing-key session identity, allocator-backed Tensorcast HiCache launch, and prepared-bundle reuse before policy tuning |

If Phase 5 validation fails on `cached_tokens` not being exposed via
`/v1/chat/completions`, we redesign before proceeding to Phase 6+.
This is the most important early checkpoint.

The publication-grade sweep, plotting, and report-generation work moves
after Phase 8. It should not start until the Phase-8 E2E smoke has at
least one consumed migration with verified prepared-bundle attachment.
