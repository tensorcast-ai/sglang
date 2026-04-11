# Request-Transfer Benchmark Architecture

## 1. Goal

This benchmark validates phase-3 controller-driven request transfer in the
SGLang Tensorcast integration.

The tested chain is:

1. source ordinary `/generate`
2. source `publish(engine_request_id=logical_request_id)`
3. optional target worker `prefetch_manifest_result(...)`
4. target `hydrate(publish_manifest=...)`
5. target ordinary `/generate(rid=logical_request_id)`

The final target request must resume through normal SGLang ingress. This is not
the PD prebuilt/decode path.

## 2. Implemented Components

### `run_benchmark.py`

Responsibilities:

- choose local or remote topology
- launch or reuse workers
- generate Tensorcast configs
- start global store, daemons, and SGLang instances
- run the caller driver on worker A
- collect runtime stdio/session logs
- write run summary CSV

### `caller_driver.py`

Responsibilities:

- connect one Tensorcast runtime to daemon A
- resolve source/target instance routes from the instance directory
- execute `publish`, optional warmup, and `hydrate`
- drive source and target ordinary `/generate`
- verify target-side prepared-bundle attach/fallback signals
- write `prompt_results.jsonl` and `summary.json`

### `configs/`

The benchmark reuses the validated Tensorcast service templates from
`share_local` and mutates only the per-run values:

- advertise/listen addresses
- ports
- log file paths
- byte-artifact routing and payload transport knobs

### `scripts/tensorcast_service.sh`

This is the same service lifecycle wrapper used by `share_local`:

- `start-global`
- `stop-global`
- `status-global`
- `reset-runtime-state`
- `start-daemon`
- `stop-daemon`
- `status-daemon`
- `wait-daemon-ready`

## 3. Topology

### Local

One worker hosts:

- global store
- daemon A
- daemon B
- instance A
- instance B
- caller driver

### Remote

Worker A hosts:

- global store
- daemon A
- instance A
- caller driver

Worker B hosts:

- daemon B
- instance B

For both modes:

- instance A binds to `<worker_ip_a>:<port_a>`
- instance B binds to `<worker_ip_b>:<port_b>`
- `instance_id` is therefore exactly `<worker_ip>:<port>`
- backend extra config sets
  `instance_agent_execution_endpoint=<worker_ip>:<instance_agent_port>`

## 4. Control-Plane Contract

The benchmark intentionally follows the current phase-3 contract:

- one caller program
- one Tensorcast runtime
- one daemon connection for that runtime
- source and target plans emitted separately
- stable `logical_request_id`

For the v1 SGLang profile, the benchmark keeps these three identifiers equal:

- Tensorcast `logical_request_id`
- Tensorcast `engine_request_id`
- SGLang caller `rid`

The runtime connects only to daemon A, but may still execute routed source and
target plans because the directory resolves instance execution through the
global store.

## 5. Dataflow

Per prompt:

1. Load one prompt and build one stable `logical_request_id`.
2. Send ordinary `/generate` to instance A.
3. Execute `publish(engine_request_id=logical_request_id)`.
4. Decode the returned wire `PublishManifest` back into SGLang's local
   `PublishManifestRecord` so the benchmark can inspect:
   - `publish_manifest_digest`
   - `artifact_manifest_digest`
   - `cutoff_token_count`
   - `tail_valid_tokens`
5. Optionally warm the target worker with
   `prefetch_manifest_result(publish_manifest.artifact_manifest, device="cpu")`.
6. Execute `hydrate(publish_manifest=...)` on instance B.
7. Send ordinary `/generate(rid=logical_request_id)` to instance B.
8. Poll the target instance log for prepared-bundle signals.
9. Optionally evict local prepared state on both source and target.

## 6. Verification Model

The benchmark does not rely on `cached_tokens > 0` alone.

Prompt-level success requires:

- source ordinary generate succeeded
- publish succeeded
- hydrate succeeded
- target ordinary generate succeeded
- target log shows:
  `Tensorcast prepared-bundle attached logical_request_id=... manifest=...`
- target log does not show:
  - fallback
  - fail-closed
  - consume-failed
- target `meta_info.cached_tokens` equals the page-granular closed prompt
  prefix, i.e. `cutoff_token_count - tail_valid_tokens`

## 7. Logging Strategy

This benchmark intentionally writes service logs into the shared benchmark output
directory instead of per-worker `/data`.

Reason:

- in remote topology, the caller driver runs on worker A
- target instance B may run on worker B
- the caller still needs direct access to the target log to verify attach /
  fallback / fail-closed markers

So:

- SGLang stdio logs are written directly to `outputs/<run_id>/logs/*.log`
- Tensorcast daemon/global-store log files also point there
- runtime stdio/session-state files are copied there during teardown

## 8. Passive Reuse Suppression

The benchmark is correctness-first and wants attribution to prepared-bundle
consume, not passive storage prefetch.

So the default config uses:

- normal source-side Tensorcast backend
- explicit target `hydrate(...)`
- target `prefetch_threshold` set far above the prompt range

This keeps phase-2 passive prefix reuse effectively off on instance B.

## 9. Current Non-Goals

This first version does not yet cover:

- passive-share baseline
- cold-target baseline
- concurrency/throughput sweep
- multi-controller coordination
- DP routing
- decode-only target instances
