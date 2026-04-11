# Request-Transfer KV Benchmark

This benchmark validates Tensorcast phase-3 caller-driven KV handoff between
two normal SGLang serving instances.

Unlike [`share_local`](../share_local/README.md), it does not measure passive
prefix reuse. The exercised path is:

1. ordinary source `/generate`
2. caller-visible `publish(engine_request_id=...)`
3. optional worker warmup via `prefetch_manifest_result(...)`
4. caller-visible `hydrate(publish_manifest=...)`
5. ordinary target `/generate(rid=...)`

The benchmark is correctness-first. It uses one Tensorcast runtime connected to
daemon A and drives routed source/target plans through that single control-plane
entrypoint.

## Topology

### `--topology-mode local`

One GPU worker runs:

- global store
- daemon A
- daemon B
- instance A
- instance B
- caller driver

### `--topology-mode remote`

Worker A runs:

- global store
- daemon A
- instance A
- caller driver

Worker B runs:

- daemon B
- instance B

In both modes, SGLang instances bind to the worker-reachable IP instead of
`127.0.0.1`, so `instance_id == "<worker_ip>:<port>"` matches what Tensorcast
registers in the instance directory.

## Usage

Environment:

```bash
cd /home/i-zhouyuhan/tot/thirdparty/sglang
source /home/i-zhouyuhan/tot/.venv/bin/activate
export PYTHONPATH="$PWD/benchmark:$PWD/python:/home/i-zhouyuhan/tot/thirdparty/tensorcast:${PYTHONPATH:-}"
```

Local topology on one existing 4xH800 worker:

```bash
/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline \
  python benchmark/tensorcast_benchmark/kv/request_transfer/run_benchmark.py \
  --topology-mode local \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --existing-worker-process-a <worker_a> \
  --prompt-count 10 \
  --min-prompt-chars 10034 \
  --max-prompt-chars 12697
```

Remote topology with one reused worker A and one launched worker B:

```bash
/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline \
  python benchmark/tensorcast_benchmark/kv/request_transfer/run_benchmark.py \
  --topology-mode remote \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --existing-worker-process-a <worker_a> \
  --brainctl-charged-group codesign \
  --prompt-count 10 \
  --min-prompt-chars 10034 \
  --max-prompt-chars 12697
```

Remote topology with both workers reused:

```bash
/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline \
  python benchmark/tensorcast_benchmark/kv/request_transfer/run_benchmark.py \
  --topology-mode remote \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --existing-worker-process-a <worker_a> \
  --existing-worker-process-b <worker_b>
```

Useful knobs:

- `--enable-target-worker-warmup`
  Enables `prefetch_manifest_result(...)` before target hydrate.
- `--tensorcast-source-prefetch-threshold`
  Source-side passive prefetch threshold.
- `--tensorcast-target-prefetch-threshold`
  Target-side passive prefetch threshold. Default is intentionally very large so
  success is attributable to prepared-bundle consume instead of phase-2 passive
  prefix reuse.
- `--keep-worker/--no-keep-worker`
  Keep or clean up launched workers.

## Outputs

Each run creates:

- `outputs/<run_id>/prompt_results.jsonl`
  Per-prompt structured results.
- `outputs/<run_id>/summary.json`
  Run summary.
- `outputs/benchmark_results.csv`
  Appended run-level CSV.
- `outputs/<run_id>/logs/`
  Orchestrator log, caller log, SGLang logs, Tensorcast service logs, and
  runtime stdio snapshots.

Important log files:

- `logs/sglang_instance_a.log`
- `logs/sglang_instance_b.log`
- `logs/caller_driver.log`
- `logs/tensorcast_global_store.log`
- `logs/tensorcast_daemon_a.log`
- `logs/tensorcast_daemon_b.log`

This benchmark writes service logs directly into the shared output directory
instead of per-worker `/data`, so the caller driver can verify target-side
prepared-bundle signals in both local and remote topologies.

## What Counts As Success

A prompt is marked successful only when all of the following hold:

- source ordinary `/generate` succeeds
- `publish(...)` succeeds
- `hydrate(...)` succeeds
- target ordinary `/generate` succeeds
- target log contains `Tensorcast prepared-bundle attached`
- target log does not contain:
  - `Tensorcast prepared-bundle falling back to normal generate path`
  - `Tensorcast prepared-bundle fail-closed during ordinary generate admission`
  - `Tensorcast prepared-bundle consume failed`
- `meta_info.cached_tokens == published_cutoff_token_count - tail_valid_tokens`
  for the page-granular closed prompt prefix

The benchmark does not treat `cached_tokens > 0` alone as proof.

## Current Scope

Implemented in this first version:

- one caller program
- one Tensorcast runtime connected to daemon A
- routed source `publish` and target `hydrate`
- optional target worker warmup
- local and remote topology support
- structured per-prompt and per-run outputs

Not included yet:

- passive prefix-share baseline
- cold-target baseline
- throughput/concurrency sweeps
- DP-aware routing
- decode-only target instances
