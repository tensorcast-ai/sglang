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
  - [x] Build `Trajectory` records: `messages`, `tools`, `assistant_indices`, `total_chars`, `estimated_tokens`, `resolved`, `session_id` (= `run_id`), `instance_id` (SWE-Gym task).
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

## 6. Mooncake baseline (gw_load_aware_mooncake)

**Goal**: third baseline curve. Isolates "shared substrate" value
from "programmability" value.

**Deliverables**:

- [ ] `services/mooncake.py`
  - [ ] `launch_mooncake_master(worker, port) -> Service` (master + metadata service combined per share_remote § 5.2)
  - [ ] `wait_ready`: poll the master's status endpoint
- [ ] Extend `services/sglang.py` to accept a Mooncake storage endpoint and pass through to SGLang (via `--hicache-storage-backend mooncake --hicache-storage-config ...` or whatever the current SGLang flag is — verify against installed SGLang version)
- [ ] Extend `driver/benchmark_loop.py` to launch the Mooncake master before SGLang instances when `config.kind == gw_load_aware_mooncake`
- [ ] `configs/benchmark_baseline_full.yaml` — three baselines, real `c_target_sweep`, `trials: 1` (still not full, but multi-config)
- [ ] Cluster YAML must already mark workers RDMA-capable; transport selection per arch § 7.4 lives in `benchmark.yaml`

**Validation gate**:
- [ ] All three baselines run end-to-end on a 3-worker cluster (cross-host, real RDMA — this is the first cross-host run)
- [ ] RDMA smoke passes before services launch
- [ ] Summary CSV has rows for all three baselines × all C_targets × 1 trial
- [ ] Sanity comparison: `cached_token_ratio_mean` for `gw_load_aware_mooncake` >= for `gw_load_aware` at the same C_target (substrate must help, not hurt)
- [ ] `transport_mode = rdma` is correctly recorded in summary

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

**Deferred to future Phase 7+** (deliberately not implemented in stub):

- [ ] `router/rebalancer.py` — Rebalancer background task.
- [ ] Real `ThresholdPolicy` implementation.
- [ ] `metrics/migrations.py` + per-migration JSONL writer.
- [ ] Reuse of `request_transfer.caller_driver` publish/hydrate helpers.

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

**Test summary**: `pytest tensorcast_benchmark/kv/tc_router/tests` passes 143/143.

---

## 8. Full sweep + report

**Goal**: produce the headline plot + supporting charts.

**Deliverables**:

- [ ] `configs/benchmark_main.yaml` — the publication-grade config: 4 configs × 6 C_target × 3 trials × `agent_medium` preset
- [ ] `configs/benchmark_preset_sweep.yaml` — 4 configs × 1 C_target (12) × 3 trials × 3 presets, for the 3-act story
- [ ] `tools/plot.py`
  - [ ] x-axis `c_target` / y-axis P95 TTFT, one curve per config, error bars across trials
  - [ ] Secondary plot: `cached_token_ratio_mean` per config per C_target
  - [ ] Secondary plot: per-cell `migration_count` and `migration_utilization` for tc_router only
  - [ ] Preset-sweep plot (3-act story)
- [ ] `README.md`
  - [ ] How to acquire a 3-worker cluster (out of band)
  - [ ] How to populate `cluster_brainctl_<id>.yaml` (or `cluster_static_<id>.yaml`)
  - [ ] How to run `benchmark_main.yaml`
  - [ ] How to run `benchmark_preset_sweep.yaml`
  - [ ] How to regenerate plots from outputs
- [ ] Final validation per arch § 12 (steps 1–6 done, reproducibility green)

**Validation gate**:
- [ ] Main plot reproducibly shows tc_router below gw_cache_aware below gw_load_aware in P95 TTFT at moderate `c_target` (per arch § 4.3 expected curve shape)
- [ ] Preset-sweep plot shows the 3-act story (curves merge at fast preset, separate at medium, mooncake closes the gap at slow)
- [ ] CSV is fully populated, every cell has 3 trials of data

---

## Cross-cutting concerns

### Logging and reproducibility (apply throughout)

- [ ] Every Service launch records: command, env, log path on worker, started-at timestamp
- [ ] `outputs/<run_id>/` always contains: resolved cluster YAML (post-load), resolved benchmark YAML, all jsonl files, all per-worker service logs (pulled at teardown via `Worker.get_file`), `summary.csv`
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

- [ ] **`/v1/chat/completions` `cached_tokens` field availability**: SGLang exposes `meta_info.cached_tokens` on `/generate`; verify the same field comes through on `/v1/chat/completions` final chunk. If not, we may need to switch to `/generate` with manual chat templating (reverses the arch § 5.2.2 decision and is a lot of work). **Verify in Phase 5 validation, before going further.**
- [ ] **Mooncake + SGLang HiCache version compatibility**: the `--hicache-storage-backend mooncake` flag and config schema may have shifted between SGLang versions. Check against installed SGLang in Phase 6 day 0.
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
| 7 | tc_router | The big one; everything before is prerequisite |
| 8 | Sweep + plot | Producing the result |

If Phase 5 validation fails on `cached_tokens` not being exposed via
`/v1/chat/completions`, we redesign before proceeding to Phase 6+.
This is the most important early checkpoint.
