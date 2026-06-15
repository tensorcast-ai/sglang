# tc_router Implementation Plan

This file enumerates the concrete work needed to deliver the benchmark
defined in `arch.md`, broken into phases with explicit checklists and
validation gates.

The high-level strategy is **baselines first, then `tc_router`**. We
get the harness, workload, and one gateway baseline running end to
end before adding any Tensorcast-specific code. This means at every
phase we have something runnable that we can use to debug regressions.

## 0. Repo skeleton

**Goal**: empty package tree compiles and imports; CI / tooling can
discover modules.

**Deliverables**:

- [ ] `tc_router/__init__.py`
- [ ] `tc_router/{resource,services,driver,router,workload,metrics}/__init__.py`
- [ ] `tc_router/tests/__init__.py`
- [ ] `tc_router/configs/.gitkeep`
- [ ] `tc_router/outputs/.gitkeep` (gitignored)
- [ ] entry-point Python 3.10 minimum versioning aligned with the rest of `thirdparty/sglang/benchmark/tensorcast_benchmark`
- [ ] `tc_router/scripts/.gitkeep` (for service lifecycle wrappers, will reuse from `request_transfer/scripts/` and `share_remote/scripts/` later)

**Validation gate**:
- [ ] `python -c "import tensorcast_benchmark.kv.tc_router"` succeeds from `thirdparty/sglang/benchmark`
- [ ] `pytest tc_router/tests` runs (and reports "no tests collected", which is fine)

---

## 1. Resource abstraction (cluster-portable)

**Goal**: `cluster.yaml` → `list[Worker]`, with one provider that we
can use to drive everything else. No cluster-specific logic leaks
outside `resource/`.

**Deliverables**:

- [ ] `resource/base.py`
  - [ ] `RemoteProcess` Protocol (`pid`, `wait`, `kill`, `stdout`, `stderr`)
  - [ ] `Worker` Protocol (`id`, `address`, `gpu_indices`, `scratch_dir`, `base_env`, `run`, `put_file`, `get_file`, `read_file`)
  - [ ] `ResourceProvider` Protocol (`from_cluster_config`, `workers`, `health_check`)
  - [ ] `ClusterConfig`, `WorkerConfig`, `ServicePlacement` Pydantic models matching § 9.1 schema
  - [ ] Cluster YAML loader: parses, validates `len(workers) >= 1`, distinct `id`, distinct `address`, every worker has `base_env`
- [ ] `resource/static.py`
  - [ ] `StaticProvider` + `StaticWorker` using `asyncssh` (or `paramiko` async wrapper) for `Worker.run` over plain SSH
  - [ ] Background command support via `nohup ... &` + PID-file pattern
  - [ ] Env-var injection: `Worker.run` always merges `base_env` first, then per-call `env`
  - [ ] `put_file` / `get_file` via SFTP
- [ ] `resource/factory.py`
  - [ ] `from_cluster_config(path) -> ResourceProvider` dispatch on `provider.kind`
  - [ ] Currently dispatches to `static`; brainctl reserved
- [ ] `resource/brainctl.py` — placeholder file with `BrainctlProvider` skeleton (raises `NotImplementedError`); to be filled in Phase 6+ if/when we run on the brainctl cluster
- [ ] `tc_router/configs/cluster_static_local.yaml` — a tiny mock pointing at `localhost` for unit tests

**Validation gate**:
- [ ] Unit test: load `cluster_static_local.yaml`, call `provider.workers()`, assert 1 worker with correct attributes
- [ ] Integration test: against localhost SSH, `Worker.run(["echo","hello"])` returns stdout="hello", env injection works (echo a `base_env` variable)
- [ ] Integration test: `Worker.put_file` then `Worker.get_file` round-trips a small text file

---

## 2. Services layer (cluster-agnostic launchers)

**Goal**: `services.<name>.launch(...)` returns a running, healthy
service. None of these know about brainctl / SSH / k8s.

**Deliverables**:

- [ ] `services/base.py`
  - [ ] `Service` dataclass (`name`, `worker`, `endpoints: dict[str, str]`, `process: RemoteProcess`, `log_remote_path`)
  - [ ] `ServiceLauncher` Protocol with `async launch(...) -> Service` and `async wait_ready(svc, timeout_s) -> None`
- [ ] `services/sglang.py`
  - [ ] `launch_instance(worker, model_path, tp_size, port, kv_pool_size_gb, mooncake_storage_endpoint=None) -> Service`
  - [ ] Build the `python -m sglang.launch_server --host {worker.address} --port {port} --model-path ... --tp-size ...` command
  - [ ] **Must NOT pass `--tool-call-parser`** (per arch § 5.2.3); add a unit test asserting the constructed command does not contain that flag
  - [ ] HiCache configuration knobs exposed (so Mooncake-backed mode can pass `--hicache-storage-backend mooncake` etc.)
  - [ ] Endpoints exposed: `serving_http = http://{worker.address}:{port}`, `instance_id = {worker.address}:{port}`
  - [ ] `wait_ready`: poll `GET /health` until 200 or timeout
- [ ] `services/rdma_smoke.py`
  - [ ] Star-shaped check: from `worker_0`, ping/test RDMA reachability to each of `worker_1..N-1`
  - [ ] Reuse the smoke command shape from `share_remote/scripts/` (delegate to a shell script under `tc_router/scripts/rdma_smoke.sh` if helpful)
  - [ ] Skipable via flag if the cluster YAML carries an `rdma_smoke_passed_at: <timestamp>` annotation
- [ ] `tc_router/scripts/sglang_service.sh` (optional convenience wrapper that mirrors the pattern in `share_remote/scripts/`, callable via `Worker.run`)

**Validation gate**:
- [ ] Unit test: `sglang.build_launch_command(...)` returns expected argv list and does NOT contain `--tool-call-parser`
- [ ] Live test on a 1-worker cluster: `launch_instance` succeeds, `/health` returns 200, `/v1/models` lists the model
- [ ] Live test: send one short `/v1/chat/completions` request, assert response has `choices[0].message.content` populated and `tool_calls` is empty/absent (guards against accidental parser enable, per arch § 5.2.3)

---

## 3. Workload (independent of router; usable for any baseline)

**Goal**: a stand-alone workload generator that, given a `Router` and
a configured `inter_turn_delay` preset, drives a steady-state of
`C_target` concurrent SWE-Gym replays.

**Deliverables**:

- [ ] `workload/trajectory_pool.py`
  - [ ] Load all 3 parquet shards from `dataset_path`, project to `(instance_id, run_id, resolved, messages, tools)` columns
  - [ ] Apply filter: `turns >= min_turns AND total_chars/3.6 >= min_total_tokens` (pre-Qwen-tokenizer approximation; we re-validate with target tokenizer when running)
  - [ ] Build `Trajectory` records: ordered list of `messages`, list of `tools`, derived `assistant_indices` (positions where we issue an LLM call), `instance_id` (SWE-Gym task), `run_id` (used as `session_id`)
  - [ ] Deterministic shuffle with seed
  - [ ] Profile mode: `python -m tc_router.workload.trajectory_pool --dataset-path ... --report` prints filter survival counts, distribution stats (turns, tokens, final-prompt-tokens) — replaces the one-off probe we ran in arch validation
- [ ] `workload/inter_turn_delay.py`
  - [ ] `Preset` enum: `agent_fast`, `agent_medium`, `agent_slow`, `custom`
  - [ ] `LogNormalSampler(mu, sigma, seed)` returning a callable
  - [ ] Preset-name-to-(mu, sigma) registry exactly matching arch § 5.3.1
  - [ ] Helper: `p90_seconds(preset)` for use in policy config
- [ ] `workload/generator.py`
  - [ ] `WorkloadDriver` class with `(router, pool, inter_turn_sampler, c_target, wall_seconds, warmup_seconds, start_jitter_s, max_new_tokens_clip)`
  - [ ] `async run() -> WorkloadOutcome`: spawns supervisor + per-session coroutines; handles graceful shutdown after `wall_seconds`
  - [ ] Per-session coroutine: faithful replay loop (arch § 5.2.1); awaits `router.generate(session_id, messages, tools, sampling_params)` per `assistant` boundary; discards response content; records `TurnRecord` to a queue
  - [ ] Logical clock: per-session `last_active_ts` updated for every router call
  - [ ] No live agent loop; no chat-template work; pass `messages` and `tools` straight through to `Router`
- [ ] `metrics/per_turn.py`
  - [ ] `TurnRecord` Pydantic model matching arch § 10.1
  - [ ] JSONL writer with explicit flush
- [ ] `tests/test_inter_turn_delay.py` — sanity: 10000 samples from `agent_medium`, assert P50 in [18, 22]s, P95 in [70, 80]s
- [ ] `tests/test_trajectory_pool.py` — load a tiny synthetic parquet, assert filter survival + replay positions are correct

**Validation gate**:
- [ ] Profile-mode run on real `/data/datasets/OpenHands-Sampled-Trajectories`: prints distributions matching what arch § 5.1.3 reports (median 31 turns, 11K total tokens)
- [ ] Inter-turn delay test: `python -m tc_router.workload.inter_turn_delay --preset agent_medium --n 10000 --report` prints quantiles within 5% of the arch table
- [ ] Dry-run integration: replace `Router` with a `MockRouter` that returns instantly; drive 5 sessions for 30s; assert per-session turn count > 1, jsonl file written

---

## 4. Router interface and metrics scaffolding

**Goal**: define the abstraction the workload talks to. Implement
metrics aggregation. No real router yet.

**Deliverables**:

- [ ] `router/interface.py`
  - [ ] `GenerateResult` dataclass (text, ttft_ms, latency_ms, served_instance, prompt_tokens, cached_tokens, raw_meta_info)
  - [ ] `Router` Protocol per arch § 6.2 (`generate`, `close`)
- [ ] `router/state.py`
  - [ ] `SessionState`, `LoadSample`, `MigrationDecision`, `MigrationFuture` dataclasses (per arch § 6.3 / § 6.4)
- [ ] `router/instance_loads.py`
  - [ ] `InstanceLoadPoller` background task: HTTP GET to each instance's `/get_server_info`, refresh `instance_loads` map every `period_ms`
  - [ ] Reuses or mirrors the `get_load` helper in `tot_experiment/src/tot_experiment/sglang_client.py`
- [ ] `metrics/summary.py`
  - [ ] `RunSummary` Pydantic model matching arch § 10.3
  - [ ] Aggregator: read `turns.jsonl` + `migrations.jsonl` per cell; emit one `summary.csv` row per cell with all fields including `inter_turn_delay_preset` and `transport_mode`
- [ ] `tests/test_summary.py` — feed synthetic per-turn jsonl, assert P50/P95/P99 numerics

**Validation gate**:
- [ ] Unit test: `MockRouter` implementing `Router` Protocol, `WorkloadDriver` runs against it for 30s, `summary.py` aggregates a non-trivial summary row

---

## 5. Gateway baseline router (gw_load_aware, gw_cache_aware)

**Goal**: first end-to-end runnable configuration.
`gw_load_aware` → headline plot. `gw_cache_aware` → second curve.

**Deliverables**:

- [ ] `services/gateway.py`
  - [ ] `launch_gateway(driver_host, instances: list[Service], policy: Literal["power_of_two","cache_aware"], port) -> Service`
  - [ ] Constructs the `python3 -m sglang_router.launch_router --worker-urls ... --policy ...` command
  - [ ] Runs on the **driver host** (not on a worker), since the gateway is itself a router process and the driver is the natural place
  - [ ] `wait_ready`: poll `/v1/models` until 200
- [ ] `router/gateway_router.py`
  - [ ] `GatewayRouter` implementing `Router`: forwards `generate(session_id, messages, tools, sampling_params)` to the gateway via `/v1/chat/completions` (streaming, OpenAI client)
  - [ ] Records `served_instance` from response headers (gateway should expose `x-served-by` or similar; fall back to the upstream URL in metadata)
  - [ ] Streams the response, measures TTFT at first token, latency at completion; discards content; reads `prompt_tokens` and `cached_tokens` from final `meta_info` (if exposed by SGLang's chat-completions response — verify in Phase 5 validation)
- [ ] `driver/benchmark_loop.py` (skeletal v1)
  - [ ] Iterates `(config, c_target, trial, preset)` cells (preset list is just `[agent_medium]` for now)
  - [ ] Per cell: launch services, run `WorkloadDriver`, collect logs, tear down, write summary row
- [ ] `run_benchmark.py`
  - [ ] CLI: `--cluster <yaml>` `--bench <yaml>` `--config-filter gw_load_aware,gw_cache_aware` (optional)
  - [ ] Wires `resource.factory`, `services`, `driver`, `metrics` together
- [ ] `configs/benchmark_baseline_smoke.yaml` — minimal smoke config: `c_target_sweep: [3]`, `trials: 1`, `wall_seconds: 120`, `warmup_seconds: 30`, `configs: [gw_load_aware]`

**Validation gate (the first big milestone)**:
- [ ] Smoke run: `python run_benchmark.py --cluster cluster_static_local.yaml --bench benchmark_baseline_smoke.yaml` completes without errors
- [ ] Outputs present: `outputs/<run_id>/gw_load_aware/c3/trial0/turns.jsonl` non-empty, `summary.csv` has one row
- [ ] Spot-check: every turn record has `cached_tokens > 0` for turn_index >= 1 of the same session (proves prefix cache is working at all)
- [ ] Repeat with `configs: [gw_cache_aware]`; both runs produce comparable structure
- [ ] Manually inspect first 3 turn records: `prompt_tokens` matches what we'd expect from the trajectory length

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

## 7. tc_router core

**Goal**: the actual programmable router. Reactive in v1 (per
discussion: deferred-eager features become later policies).

**Deliverables**:

- [ ] `services/tensorcast.py`
  - [ ] `launch_global_store(worker, port) -> Service`
  - [ ] `launch_daemon(worker, global_store_endpoint, port) -> Service`
  - [ ] `wait_ready` for both
  - [ ] Reuses the existing `tensorcast_service.sh` from `request_transfer/scripts/` if compatible
- [ ] `router/policy.py`
  - [ ] `Policy` Protocol (4 hooks per arch § 6.4)
  - [ ] `ThresholdPolicy` concrete implementation
  - [ ] `policy_factory(config: dict) -> Policy` for YAML dispatch
- [ ] `router/rebalancer.py`
  - [ ] `Rebalancer` background task: tick every `period_ms`, call policy, execute migrations
  - [ ] `_publish_then_hydrate(session_state, source, target)` — uses the request_transfer caller_driver migration mechanic verbatim (publish → hydrate on instance B)
  - [ ] Cooldown enforcement (`migration_cooldown_s`)
  - [ ] Concurrency cap (`max_migrations_per_tick`)
  - [ ] Graceful failure: publish or hydrate failure → no-op, log, do NOT change `home_instance`
- [ ] `router/tc_router.py`
  - [ ] `TcRouter` implementing `Router`
  - [ ] Owns: `session_state`, `instance_loads_poller`, `rebalancer`, Tensorcast `Runtime`
  - [ ] `generate`: route to `home_instance`, generate `rid`, store as `last_engine_request_id`, post via `/v1/chat/completions`
  - [ ] On startup: connect to global store via Tensorcast `tc.connect(daemon_address=...)` (we connect to one daemon, by protocol § 4.1 of caller_driver pattern)
  - [ ] On shutdown: cancel rebalancer + load poller, close runtime
- [ ] `metrics/migrations.py`
  - [ ] `MigrationEvent` Pydantic model matching arch § 10.2
  - [ ] JSONL writer
  - [ ] Lazy "consumed_by" backfill: when a session's next turn lands on the migration target with `cached_tokens` > the published cutoff threshold, mark the migration as consumed
  - [ ] Wasted-marker on TTL expiry
- [ ] Driver-level wiring: `driver/benchmark_loop.py` recognizes `tc_router` config, launches Tensorcast services, instantiates `TcRouter` with the policy from YAML
- [ ] Reuse from `request_transfer/`: import `_decode_instance_publish_manifest`, `_publish_plan_result`, `_hydrate_plan_result`, `_PreparedBundleSignals` from `caller_driver.py` (or refactor them into a shared `tensorcast_benchmark.kv.tensorcast_helpers` module if that already exists; otherwise copy with attribution)

**Validation gate**:
- [ ] Stub policy `_NeverRebalance` configured: `tc_router` config behaves identically to `gw_load_aware` (same per-turn metrics, zero migrations). This test must pass before moving on.
- [ ] Stub policy `_AlwaysRebalanceFirstSession`: drive 5 sessions; first session migrates after turn 2; verify:
  - [ ] migration record written with publish/hydrate latencies
  - [ ] target instance log contains `Tensorcast prepared-bundle attached` (reuse the verification logic from `request_transfer`)
  - [ ] next turn's `cached_tokens > 0` and is served by the new home instance
  - [ ] `migration_utilization` for that session = 1.0
- [ ] Full smoke run with `ThresholdPolicy`, `c_target_sweep: [3, 12]`: produces non-zero migrations at C_target=12, mostly-zero at C_target=3

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
