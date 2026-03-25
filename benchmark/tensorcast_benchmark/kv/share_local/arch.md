# Share-Local KV Benchmark Architecture

## 1. Scope

This benchmark targets a single-node, local prefix-share scenario.

The benchmark does **not** model PD disaggregation, request-level KV transfer,
or programmable Tensorcast plans. It only measures the effect of a shared KV
substrate on the time-to-first-token (TTFT) of a repeated prompt served by a
different SGLang instance on the same node.

The core request pattern is:

1. Send prompt `P` to instance A.
2. Wait until the response from instance A fully completes.
3. Optionally wait for a small configurable settle interval.
4. Send the exact same prompt `P` to instance B.
5. Compare TTFT between the first request and the second request.

This benchmark is therefore a **prefix share** benchmark, not a PD handoff
benchmark.

## 2. Non-Goals

The following are explicitly out of scope for `share_local`:

- PD-prefill/decode separation.
- Request continuation on another instance.
- using `rid` or `engine_request_id` as request-transfer handles or cache-hit
  keys
- Tensorcast programmable instance-step APIs.
- Tensorcast `publish()` / `hydrate()` request-level transfer.
- External controller/router logic.
- Throughput benchmarking under mixed concurrent traffic.

Those belong to future benchmarks such as `transfer_local`, `transfer_remote`,
or router-driven integration tests.

## 3. Benchmark Semantics

The semantic contract of this benchmark is:

- Both SGLang instances are ordinary serving instances.
- Both instances run on the same 8xH800 worker.
- Both instances use the same model and the same HiCache configuration class,
  except for instance-local ports, GPU affinity, and backend-specific service
  addresses.
- The second request uses the same prompt text as the first request.
- The benchmark measures whether serving the same prompt on a different
  instance benefits from shared-prefix reuse through the configured storage
  backend.

The expectation is:

- Request 1 populates local cache state on instance A and exports reusable
  prefix pages into the shared KV substrate.
- Request 2 on instance B finds the same prefix in the shared substrate and
  reduces TTFT relative to request 1.

The benchmark may attach an explicit `rid` to both requests so logs and result
rows can be correlated across instances, but that `rid` is not the cache-hit
key. Shared-prefix reuse is determined by the token-prefix/page-hash chain that
SGLang HiCache derives from the prompt.

The Tensorcast path is still in correctness-first bring-up. The benchmark is
therefore useful both as a functional prefix-share test and as a topology /
service-lifecycle harness while the backend is being optimized.

## 4. Backend Modes

The benchmark supports two backend families:

### 4.1 `hicache-storage-backend = mooncake`

This is the existing functional prefix-share baseline.

- Start one Mooncake metadata/master service on the same worker.
- Launch two standard SGLang instances with Mooncake HiCache enabled.
- Run the request-pair TTFT experiment.

### 4.2 `hicache-storage-backend = tensorcast`

This is the Tensorcast-backed SGLang HiCache integration path under bring-up.

- Start one Tensorcast global store on the same worker.
- Start Tensorcast daemon(s) on the same worker.
- Launch two standard SGLang instances.
- The same benchmark structure is used both for service bring-up and for
  functional prefix-share validation as the integration matures.

### 4.3 Tensorcast daemon topology

When `hicache-storage-backend = tensorcast`, the benchmark supports:

- `tensorcast-daemon-mode = share`
  - one daemon shared by both SGLang instances
- `tensorcast-daemon-mode = separate`
  - two daemons on the same worker
  - one daemon is logically associated with instance A
  - one daemon is logically associated with instance B

This distinction matters because:

- `share` models one shared local daemon / storage entrypoint.
- `separate` models cross-daemon prefix sharing on the same node.

## 5. Physical Topology

The benchmark always runs on a single remote GPU worker allocated through
`brainctl`.

### 5.1 Worker

- one worker
- GPU shape: `8xH800`
- all services and commands run on this worker via `brainctl exec`
- all durable logs are written to shared `/data`

### 5.2 GPU partition

The worker is split into two fixed 4-GPU groups:

- instance A: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- instance B: `CUDA_VISIBLE_DEVICES=4,5,6,7`

Default TP is expected to be `4`, but TP size should remain configurable.

### 5.3 Service placement

- Mooncake service, when enabled, runs on the same worker.
- Tensorcast global store, when enabled, runs on the same worker.
- Tensorcast daemon(s), when enabled, run on the same worker.
- No cross-node traffic is involved in `share_local`.

## 6. SGLang Launch Contract

Both instances in `share_local` are **normal serving instances**.

They must **not** use PD-disaggregation launch flags. In particular:

- no `--disaggregation-mode`
- no PD router
- no prefill-only / decode-only split
- no request handoff protocol

The only benchmark-specific differences between the two instances should be:

- port
- GPU affinity
- backend-specific storage configuration
- instance-local log path

The benchmark should keep the request path as close as possible to an ordinary
SGLang `/generate` call so that it tests passive prefix sharing, not an
artificial benchmark-only request API.

## 7. Request Model

### 7.1 Input dataset

The benchmark uses JSONL prompts from `data/test.jsonl`.

For each selected row:

- extract `question`
- use `question` as the full prompt text

The number of prompts to run should be configurable, for example by taking the
first `N` rows.

### 7.2 Pair execution

Each prompt is executed as one ordered pair:

1. send prompt to instance A
2. wait for full completion
3. optional `settle_ms`
4. send the same prompt to instance B
5. wait for full completion

Default order is `A -> B`.

For functional prefix-share validation, the exact same prompt text must be sent
to both instances. That is what gives both sides the same tokenized prefix and
the same page-hash chain.

A future extension may support:

- `B -> A`
- alternating order per pair

That can help detect asymmetry between the two GPU groups or daemon topologies,
but it is not required for the initial version.

### 7.3 Request correlation vs reuse key

The benchmark is allowed to set an explicit request id (`rid`) on the HTTP
request, but only for correlation:

- to match instance-A and instance-B log lines for the same prompt pair
- to simplify debugging of one measured request through the benchmark outputs

The benchmark must not claim that a repeated `rid` causes reuse. Reuse only
happens when the prompt leads to the same token-prefix/page-hash identity in
HiCache.

This distinction matters because:

- `share_local` is a prefix-share benchmark, not a request-transfer benchmark
- request-transfer semantics belong to future programmable-controller tests
- `meta_info.cached_tokens` is not a reliable proof signal for shared-substrate
  prefetch in this benchmark

### 7.4 Sampling settings

The benchmark should default to deterministic generation to reduce noise:

- `temperature = 0`
- configurable `max_new_tokens`

The benchmark should measure TTFT, but also keep full end-to-end latency for
both requests in each pair.

### 7.5 Success semantics

A pair is successful if:

- request A returns HTTP 200 and yields at least one generated token
- request B returns HTTP 200 and yields at least one generated token

If request A fails, the pair fails immediately.

If request B fails, the pair is recorded as failed, but request A metrics should
still be preserved in the result row for debugging.

## 8. TTFT Measurement

This benchmark exists to compare TTFT across the two requests in a pair, so the
request client must support streaming measurement.

The preferred measurement strategy is:

- call `/generate` with `stream=true`
- record `request_start_ts`
- record `first_token_ts` when the first generated token chunk arrives
- compute `ttft_ms = first_token_ts - request_start_ts`
- continue consuming the stream to completion and record total latency

This implies that the shared SGLang client utility under
`benchmark/tensorcast_benchmark/kv/` must be extended beyond a simple non-streaming
wrapper.

Minimum client features needed by `share_local`:

- `wait_ready()`
- `generate_stream(...)`
- TTFT extraction
- full-latency extraction
- structured error reporting

## 9. Metrics

For each request pair, the benchmark should record at least:

- `prompt_id`
- `prompt_chars`
- `backend`
- `tensorcast_daemon_mode` when applicable
- `instance_a_ttft_ms`
- `instance_a_latency_ms`
- `instance_b_ttft_ms`
- `instance_b_latency_ms`
- `ttft_improvement_ms = instance_a_ttft_ms - instance_b_ttft_ms`
- `ttft_speedup_ratio = instance_a_ttft_ms / instance_b_ttft_ms`
- `status`
- `error_message`

Run-level summary should include:

- number of successful pairs
- number of failed pairs
- mean / median / p95 TTFT for instance A
- mean / median / p95 TTFT for instance B
- mean / median / p95 TTFT improvement

## 10. Logging and Output Layout

The output layout should follow the established style of
`benchmark/tensorcast_benchmark/load_weight_remote`.

Each invocation creates:

- `outputs/<run_id>/generated_configs/`
- `outputs/<run_id>/logs/`

A global append-only CSV should be kept at:

- `outputs/benchmark_results.csv`

Remote long-lived process logs should first be written to shared `/data`, for
example:

- `/data/<run_id>_worker_keepalive.log`
- `/data/<run_id>_gpu_snapshot.log`
- `/data/<run_id>_mooncake_master.log`
- `/data/<run_id>_tensorcast_global_store.log`
- `/data/<run_id>_tensorcast_daemon_shared.log`
- `/data/<run_id>_tensorcast_daemon_a.log`
- `/data/<run_id>_tensorcast_daemon_b.log`
- `/data/<run_id>_sglang_instance_a.log`
- `/data/<run_id>_sglang_instance_b.log`
- `/data/<run_id>_request_driver.log`

At the end of the run, those logs should be copied into
`outputs/<run_id>/logs/`.

The benchmark should also emit:

- one JSON or JSONL file with per-pair raw measurements
- one orchestrator log
- one summary artifact for quick inspection

## 11. Self-Contained Directory Layout

The `share_local` benchmark should be self-contained in the same style as other
Tensorcast benchmarks:

- `benchmark/tensorcast_benchmark/kv/share_local/run_benchmark.py`
- `benchmark/tensorcast_benchmark/kv/share_local/README.md`
- `benchmark/tensorcast_benchmark/kv/share_local/arch.md`
- `benchmark/tensorcast_benchmark/kv/share_local/scripts/`
- `benchmark/tensorcast_benchmark/kv/share_local/outputs/`

The benchmark-specific directory should only contain logic unique to
`share_local`.

## 12. Shared Utilities Under `benchmark/tensorcast_benchmark/kv/`

This folder will later host multiple KV benchmarks:

- `share_local`
- `share_remote`
- `transfer_local`
- `transfer_remote`

Common reusable utilities should therefore live at the `kv/` level rather than
inside one benchmark directory.

The intended shared utility split is:

- `benchmark/tensorcast_benchmark/kv/sgl_client.py`
  - async HTTP client
  - readiness checks
  - streaming `/generate`
  - TTFT measurement helpers
- `benchmark/tensorcast_benchmark/kv/dataset.py`
  - load prompts from JSONL
  - select first `N` questions
- `benchmark/tensorcast_benchmark/kv/remote.py`
  - remote worker lifecycle helpers
  - `brainctl launch`, `exec`, `cleanup`, log collection
- `benchmark/tensorcast_benchmark/kv/outputs.py`
  - run-id creation
  - output directory creation
  - CSV append helpers
- `benchmark/tensorcast_benchmark/kv/models.py`
  - Pydantic config and result models

The goal is:

- keep each benchmark self-contained at the workflow level
- avoid duplicating low-level request, dataset, remote-worker, and output logic

## 13. Execution Phases

The implementation should be staged.

### Phase 1: harness

Implement:

- remote worker allocation
- service lifecycle
- two-instance launch
- TTFT request-pair driver
- logging
- result collection

In this phase:

- Mooncake path is expected to be the first functional prefix-share baseline.
- Tensorcast path may still be harness-only until SGLang HiCache integration
  exists.

### Phase 2: functional Tensorcast share

Once SGLang exposes a functional Tensorcast HiCache backend:

- keep the same benchmark structure
- keep the same request pair semantics
- make `hicache-storage-backend=tensorcast` measure real prefix-share TTFT

The benchmark architecture should therefore be designed now so that backend
behavior can evolve without rewriting the benchmark workflow.

## 14. Key Design Decisions

The benchmark is intentionally defined by the following decisions:

- measure prefix share, not PD handoff
- use normal SGLang servers, not disaggregation servers
- send the same prompt twice on different instances
- measure TTFT difference, not just total latency
- avoid using `rid` or request-transfer semantics as the cache-hit key
- keep the topology single-node and local
- keep future benchmark families aligned through shared utilities under
  `benchmark/tensorcast_benchmark/kv/`

These decisions should remain stable unless the benchmark goal itself changes.

## 15. TODO Checklist

- [x] Create the shared `benchmark/tensorcast_benchmark/kv/` utility layout.
- [x] Add `benchmark/tensorcast_benchmark/kv/models.py` with Pydantic config and result models.
- [x] Add `benchmark/tensorcast_benchmark/kv/outputs.py` for run-id generation, output paths, and CSV helpers.
- [x] Add `benchmark/tensorcast_benchmark/kv/dataset.py` to load `data/test.jsonl` and select the first `N` prompts.
- [x] Extend dataset loading to support schema-aware LongBench inputs plus coarse min/max prompt-length filtering.
- [x] Add `benchmark/tensorcast_benchmark/kv/remote.py` for `brainctl` worker launch, exec, cleanup, and remote log collection.
- [x] Add `benchmark/tensorcast_benchmark/kv/sgl_client.py` with readiness checks and streaming `/generate` support.
- [x] Implement TTFT extraction in the shared SGLang client from streamed first-token arrival.
- [x] Create the `benchmark/tensorcast_benchmark/kv/share_local/` benchmark skeleton with `run_benchmark.py`, `README.md`, `scripts/`, and `outputs/`.
- [x] Implement single-worker allocation for one `8xH800` remote worker with shared-FS logging.
- [x] Implement remote environment smoke checks on the worker, including GPU visibility, shared repo visibility, and `/data` writability.
- [x] Implement remote GPU and port snapshot logging at run start for debugging.
- [x] Implement Mooncake service startup and shutdown for the `hicache-storage-backend=mooncake` path.
- [x] Implement Tensorcast global store startup and shutdown for the `hicache-storage-backend=tensorcast` path.
- [x] Implement Tensorcast daemon startup and shutdown for `tensorcast-daemon-mode=share`.
- [x] Implement Tensorcast daemon startup and shutdown for `tensorcast-daemon-mode=separate`.
- [x] Implement SGLang instance A launch with configurable `CUDA_VISIBLE_DEVICES`.
- [x] Implement SGLang instance B launch with configurable `CUDA_VISIBLE_DEVICES`.
- [x] Implement backend-specific SGLang launch arguments and environment wiring.
- [x] Implement readiness waiting for both SGLang instances before traffic starts.
- [x] Implement the ordered request-pair driver: request A completes before request B starts.
- [x] Add configurable prompt count, settle interval, sampling params, and request rate control.
- [x] Record per-request TTFT and end-to-end latency for both sides of each pair.
- [x] Emit per-pair raw results as JSON or JSONL under the run directory.
- [x] Compute run-level summary statistics for TTFT and TTFT improvement.
- [x] Append one summary row per run to `outputs/benchmark_results.csv`.
- [x] Validate the first functional Mooncake baseline with `Qwen3-14B`, `tp=2`, and 10 prompt pairs on a remote `8xH800` worker.
- [x] Copy remote `/data` logs back into `outputs/<run_id>/logs/`.
- [x] Make cleanup behavior configurable so the worker can be preserved for debugging.
- [x] Validate the full `mooncake` path as the first functional baseline.
- [x] Validate long-prompt Mooncake prefix sharing on `Qwen3-32B`, `tp=1`, using `page_first_direct`.
- [x] Add a reusable README example and preset wrapper for the validated `Qwen3-32B` / `hotpotqa` / `page_first_direct` Mooncake configuration.
- [x] Validate the `tensorcast share` path as a service-lifecycle harness.
- [x] Validate long-prompt Tensorcast prefix sharing on `Qwen3-32B`, `tp=2`, using `LongBench/hotpotqa.jsonl` with debug HiCache logs.
- [ ] Validate the `tensorcast separate` path as a two-daemon local topology harness.
- [x] Write `README.md` usage examples for Mooncake and Tensorcast modes.
- [x] Keep the benchmark-specific workflow self-contained while avoiding duplication with `kv/` shared utilities.
