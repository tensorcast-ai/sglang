# Share-Remote KV Benchmark

This benchmark is the planned multi-worker counterpart of
[`share_local`](../share_local/README.md).

It is intended to measure passive KV reuse across `N` different GPU workers for
two HiCache backends:

- `tensorcast`
- `mooncake`

Unlike [`request_transfer`](../request_transfer/README.md), this benchmark does
not exercise caller-driven `publish()` / `hydrate()` request transfer. The goal
is ordinary repeated-prompt reuse across multiple normal SGLang serving
instances.

Current status:

- architecture and configuration contract are defined here
- the benchmark scaffolding is implemented under this directory
- end-to-end validation has not been run from this README yet

## Intended Request Pattern

Each selected prompt forms one prompt group.

For a prompt group with `worker_count = N`:

1. send the prompt to instance `0`
2. wait `settle_ms`
3. send the same prompt to instance `1`
4. wait `settle_ms`
5. continue until instance `N - 1`

Prompt groups themselves are launched at `rps`, where `rps` means prompt-group
start rate, not total per-request QPS.

So:

- `rps` controls when prompt group `i` starts
- `settle_ms` controls the input gap between neighboring instances inside the
  same group
- different prompt groups may overlap in time

The benchmark does not wait for explicit source-publication drain in the first
version. Reuse attribution is based on the configured backend and the staged
prompt order, not a barriered per-prompt handoff.

## Topology

The benchmark always targets cross-worker placement with RDMA-capable workers.
Workers should land on different physical hosts.

One worker hosts exactly one SGLang instance. That instance uses the first
`tp_size` visible GPUs on that worker.

### Tensorcast

For `backend = tensorcast`:

- each worker runs one Tensorcast daemon
- each worker runs one SGLang instance
- one designated service host worker, default `worker0`, also runs the
  Tensorcast global store
- each SGLang instance connects to its local daemon
- all daemons connect to the same global store

### Mooncake

For `backend = mooncake`:

- each worker runs one SGLang instance
- one designated service host worker, default `worker0`, also runs the
  Mooncake master and metadata service
- all SGLang instances connect to that shared Mooncake control-plane service

## RDMA Contract

The benchmark requires RDMA-capable workers even when backend transport is
configured to use TCP.

The intended behavior is:

- newly launched workers:
  - run RDMA smoke checks before the real benchmark
  - first version only requires `worker0 -> worker1..N-1` star smoke, not full
    mesh
- reused workers:
  - default is to skip smoke checks

The benchmark should inject the correct RDMA environment into every launched
remote process that participates in cross-worker data transfer, following the
`brainctl-launch-remote-gpu` skill:

- `NCCL_SOCKET_IFNAME`
- `NCCL_IB_HCA`
- `NCCL_IB_GID_INDEX=3`
- `NCCL_SOCKET_FAMILY=AF_INET`
- `MASTER_ADDR`

This applies to:

- Tensorcast daemon processes when Tensorcast RDMA transport is enabled
- Mooncake service / SGLang processes when Mooncake RDMA transport is enabled
- SGLang instance processes when the backend path depends on NCCL-over-RDMA

## Intended Configuration Model

`share_remote` is planned to be YAML-driven.

The benchmark runner is expected to consume one config file and copy both the
input config and the fully resolved config into `outputs/<run_id>/`.

Example shape:

```yaml
backend: tensorcast

workers:
  count: 4
  existing_worker_processes: []
  charged_group: codesign
  keep_workers: true
  require_distinct_nodes: true
  rdma_required: true
  verify_rdma_on_reuse: false
  service_host_worker_index: 0
  default_shape:
    gpu: 4
    cpu: 64
    memory: 500000
    positive_tags: H800
    negative_tags: ""

transport:
  use_rdma: true
  rdma_smoke_mode: star

workload:
  model_path: /mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-32B
  model_name: ""
  page_size: 32
  tp_size: 2
  prompt_count: 10
  rps: 0.5
  settle_ms: 20000
  max_new_tokens: 32
  temperature: 0.0
  data_path: benchmark/tensorcast_benchmark/kv/dataset/LongBench/hotpotqa.jsonl
  min_prompt_chars: 10034
  max_prompt_chars: 12697

  hicache_storage_prefetch_policy: wait_complete

backend_config:
  tensorcast:
    global_store_port: 50051
    daemon_port_base: 50052
    daemon_p2p_port_base: 65090
    byte_artifact_shard_count: 8
  mooncake:
    master_port: 60051
    http_metadata_server_port: 8080
    protocol: rdma
    device_name: ""
    global_segment_size: 64gb
```

## Planned Layout

- `run_benchmark.py`
  - local orchestrator
  - worker acquisition / reuse
  - RDMA smoke
  - service launch
  - config generation
  - result collection
- `request_driver.py`
  - prompt scheduling
  - multi-instance request fanout
  - TTFT collection
  - structured output writing
- `models.py`
  - `share_remote`-specific Pydantic config and result models
- `outputs.py`
  - run-local output helpers and summary/CSV helpers
- `configs/`
  - Tensorcast and Mooncake service templates
- `scripts/`
  - service wrappers reused across remote runs

## Planned Outputs

Each run should write:

- `outputs/<run_id>/config.yaml`
  - input YAML copied for reproducibility
- `outputs/<run_id>/resolved_config.yaml`
  - normalized config after defaults and worker resolution
- `outputs/<run_id>/worker_inventory.json`
  - worker process names, hostnames, IPs, nodes, and RDMA-related discovery
- `outputs/<run_id>/generated_configs/`
  - per-run service configs
- `outputs/<run_id>/prompt_results.jsonl`
  - one prompt-group result per line
- `outputs/<run_id>/summary.json`
  - run-level aggregation
- `outputs/benchmark_results.csv`
  - append-only summary table
- `outputs/<run_id>/logs/worker_<idx>/`
  - per-worker service and instance logs

The important metric is TTFT, but the benchmark should also record:

- per-instance `cached_tokens`
- request success/failure
- per-position TTFT aggregates
- TTFT improvement relative to position `0`

## Success Criteria

For a run to be considered healthy:

- all requested workers are up and on distinct nodes
- required services are ready
- all instances accept requests
- prompt groups complete without systemic request failure
- later positions show the expected reuse effect for the chosen backend and
  configuration

This benchmark is intended to compare Tensorcast and Mooncake under the same
multi-worker reuse pattern. It is therefore important that topology, prompt
schedule, and TTFT aggregation semantics stay backend-neutral.

## Usage

Environment:

```bash
cd /home/i-zhouyuhan/tot/thirdparty/sglang
source /home/i-zhouyuhan/tot/.venv/bin/activate
export PYTHONPATH="$PWD/benchmark:$PWD/python:/home/i-zhouyuhan/tot/thirdparty/tensorcast:${PYTHONPATH:-}"

/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline \
  python -m tensorcast_benchmark.kv.share_remote.run_benchmark \
  --config benchmark/tensorcast_benchmark/kv/share_remote/configs/example_tensorcast.yaml
```

Mooncake example:

```bash
/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline \
  python -m tensorcast_benchmark.kv.share_remote.run_benchmark \
  --config benchmark/tensorcast_benchmark/kv/share_remote/configs/example_mooncake.yaml
```

The provided example YAMLs show the two intended configuration styles:

- `example_tensorcast.yaml`
  - `count + default_shape`
- `example_mooncake.yaml`
  - explicit `workers.entries[]`

Important default serving knobs in this benchmark:

- `workload.page_size = 32`
- `workload.hicache_storage_prefetch_policy = wait_complete`
- `prefetch_threshold = 1` for both Tensorcast and Mooncake backends
- `backend_config.mooncake.global_segment_size = 64gb`

## Implementation Notes

The current implementation provides:

- YAML-driven multi-worker orchestration
- worker reuse or allocation through `brainctl`
- distinct-node validation
- per-worker RDMA environment derivation
- `worker0 -> others` `ib_write_bw` smoke for newly launched workers
- backend-specific service bring-up for Tensorcast and Mooncake
- one SGLang instance per worker
- two-phase SGLang startup: launch all instances first, then wait for all
  health endpoints
- remote request driver execution with per-prompt and run-level outputs

What still needs real-cluster validation:

- Tensorcast remote passive reuse functionality and TTFT deltas
- Mooncake remote passive reuse functionality and TTFT deltas
- RDMA vs TCP comparison runs
- larger-`N` worker sweeps
