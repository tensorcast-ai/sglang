# Share-Remote Benchmark Architecture

## 1. Goal

`share_remote` is a planned multi-worker benchmark for passive KV reuse.

It extends the same core semantic question asked by
[`share_local`](../share_local/arch.md):

- if prompt `P` is served first by one ordinary SGLang instance,
- and the same prompt `P` is then served by other ordinary SGLang instances,
- how much TTFT benefit do later instances get from the shared KV substrate?

The difference is physical topology:

- `share_local` stays on one worker
- `share_remote` spans `N` different RDMA-capable workers

This benchmark is specifically for backend comparison between:

- Tensorcast-backed HiCache
- Mooncake-backed HiCache

It is not a controller-driven request-transfer benchmark.

## 2. Non-Goals

The following are explicitly out of scope:

- Tensorcast `publish()` / `hydrate()` caller programs
- request-level handoff semantics
- PD prebuilt resume / decode-only routing
- throughput-optimized mixed traffic benchmarking
- arbitrary multi-instance placement on one worker
- one worker hosting multiple SGLang instances in the first version
- full-mesh RDMA qualification in the first version

Those belong to other benchmarks or later expansions.

## 3. Core Semantics

### 3.1 What is being measured

The measured quantity is TTFT for the same prompt replayed across multiple
workers.

For each prompt group:

- position `0` is the first worker to see the prompt
- positions `1..N-1` are later workers receiving the same prompt after fixed
  delay

The main comparison is:

- absolute TTFT by position
- TTFT improvement of each later position relative to position `0`

### 3.2 What causes reuse

Reuse must come from the normal backend-specific passive KV sharing path.

This benchmark must not depend on:

- controller-visible request ids
- programmable Tensorcast instance operations
- benchmark-only handoff APIs

The causal mechanism is that every instance receives the exact same prompt text
and therefore derives the same token-prefix and storage identity chain expected
by the backend.

### 3.3 Scheduling contract

Let:

- `rps` = prompt-group start rate
- `settle_ms` = input delay between adjacent instances inside a prompt group

Then prompt group `g` starts at:

- `t_group(g) = g / rps`

and within that group instance position `k` starts at:

- `t_request(g, k) = t_group(g) + k * settle_ms`

So:

- `rps` is not total request QPS
- `settle_ms` is not a publication-drain barrier
- prompt groups may overlap

The first version intentionally does not wait for source-publication drain.
This keeps the benchmark close to a natural repeated-query workload rather than
an explicitly serialized handoff protocol.

## 4. Physical Topology

### 4.1 Worker placement

The benchmark requires `worker_count = N` RDMA-capable workers.

Each worker should satisfy:

- correct GPU shape
- sufficient CPU and DRAM
- shared workspace mount
- RDMA capability

Workers should land on different physical hosts. The intended acquisition flow
follows the `brainctl-launch-remote-gpu` skill:

1. launch or inspect worker `0`
2. record its `NODE`
3. launch later workers with `--negative-tags=node/<used-node>`
4. verify distinct `NODE` values before the benchmark

### 4.2 Per-worker serving model

Each worker runs exactly one SGLang instance in v1.

That instance uses the first `tp_size` visible GPUs on the worker. This keeps
the topology simple and makes worker count equal to instance count.

The benchmark should also keep a stable page configuration across all workers.
In the first version, `page_size` is an explicit workload parameter and should
default to `32`. The default storage prefetch policy should be
`wait_complete`. For passive reuse bring-up, `prefetch_threshold` should be
forced to `1` for both Tensorcast and Mooncake so storage-backed set/put is not
suppressed by a larger threshold.
For Mooncake, the default `global_segment_size` should be `64gb` so a
multi-prompt run at `tp_size=2` does not churn early prompt groups out of the
shared segment before later workers query them.

Future expansions may support:

- one worker with multiple instances
- heterogeneous GPU partitioning
- per-worker custom `CUDA_VISIBLE_DEVICES`

Those are out of scope for the initial version.

## 5. Backend-Specific Service Topology

### 5.1 Tensorcast mode

For `backend = tensorcast`:

- one Tensorcast daemon per worker
- one SGLang instance per worker
- one Tensorcast global store on `service_host_worker_index`, default `0`

Each instance uses its worker-local daemon as the storage endpoint.

All daemons use the same global store so cross-worker byte-artifact discovery
and routing are globally visible.

### 5.2 Mooncake mode

For `backend = mooncake`:

- one SGLang instance per worker
- one Mooncake master and metadata service on
  `service_host_worker_index`, default `0`

All instances connect to the same Mooncake control plane.

### 5.3 Shared placement rule

`service_host_worker_index` should default to `0` and remain configurable.

This makes:

- launch order deterministic
- service discovery easy to reconstruct from outputs
- reruns easier to compare

## 6. RDMA Behavior

### 6.1 Worker capability vs transport mode

The benchmark always requires RDMA-capable workers, even if the chosen backend
transport is configured to use TCP.

This allows apples-to-apples comparison:

- same worker class
- same network-capable hardware
- only backend transport mode changes

### 6.2 Smoke-test policy

For newly launched workers, the benchmark should perform RDMA smoke checks
before real service launch.

The first version only requires star-shaped smoke:

- `worker0 -> worker1`
- `worker0 -> worker2`
- ...
- `worker0 -> workerN-1`

Not required in v1:

- full mesh
- NCCL multi-rank collectives as part of the benchmark harness itself

For reused workers, smoke is skipped by default.

### 6.3 Per-process RDMA environment

The benchmark must inject the validated RDMA environment into every remote
process launch command that may use cross-worker RDMA.

The required variables come from the `brainctl-launch-remote-gpu` skill:

- `NCCL_SOCKET_IFNAME`
- `NCCL_IB_HCA`
- `NCCL_IB_GID_INDEX=3`
- `NCCL_SOCKET_FAMILY=AF_INET`
- `MASTER_ADDR`

This must be done in the actual process-launch command. The benchmark must not
assume worker-global shell state is sufficient.

## 7. Backend Transport Mapping

### 7.1 Tensorcast

The benchmark should expose a backend-neutral `transport.use_rdma` knob and map
it into Tensorcast daemon config:

- `communicator.enable_rdma = true|false`

Other Tensorcast settings, such as shard count and payload transport options,
remain benchmark parameters but are not part of the benchmark's core semantic
contract.

### 7.2 Mooncake

The same benchmark knob should map to Mooncake config via:

- `protocol = rdma|tcp`
- `device_name = <worker-local HCA selection>`

The benchmark should allow either:

- explicit `device_name`
- auto-derived device name per worker

The first version can keep this simple and select one worker-local HCA per
worker.

## 8. Control and Driver Structure

### 8.1 `run_benchmark.py`

Responsibilities:

- parse YAML config
- allocate or reuse workers
- validate distinct-node placement
- perform RDMA smoke on newly launched workers
- generate per-run backend configs
- start services
- start SGLang instances
- run the remote request driver
- collect logs and summaries

### 8.2 `request_driver.py`

Responsibilities:

- load prompts
- build prompt groups
- schedule requests according to `rps` and `settle_ms`
- talk to all instance URLs
- collect TTFT and other per-request metrics
- write prompt-level results and run summary

The driver should stay backend-neutral as much as possible. Backend-specific
logic should live in service launch/config generation rather than in request
scheduling.

## 9. Configuration Model

The benchmark should use a `share_remote`-specific Pydantic config rather than
reuse the existing 2-instance `kv.models.BenchmarkConfig`.

The minimal top-level config groups are:

- `backend`
- `workers`
- `transport`
- `workload`
- `backend_config`

This is necessary because `share_remote` must express:

- `N` workers instead of 1 or 2 fixed instances
- worker reuse vs allocation
- distinct-node placement policy
- RDMA smoke behavior
- backend-specific multi-node control-plane placement

## 10. Output Model

### 10.1 Per-prompt result

Each prompt-group result should include:

- prompt identity
- prompt length / chars
- one `instance_result` per position

Each `instance_result` should contain at least:

- `worker_index`
- `worker_process`
- `instance_url`
- `rid`
- `ttft_ms`
- `latency_ms`
- `cached_tokens`
- `success`
- `error_message`

### 10.2 Run summary

The run summary should aggregate:

- mean / median / p95 TTFT by position
- mean cached tokens by position
- mean TTFT improvement vs position `0`
- worker inventory
- chosen backend and transport mode

This is more informative than flattening the run into only two roles such as
instance A / instance B.

## 11. Logging and Reproducibility

The benchmark should copy all relevant configuration and discovery state into
the run directory:

- input YAML
- resolved config
- worker inventory
- generated service configs
- orchestrator log
- per-worker service logs
- per-worker SGLang logs

This is necessary because a multi-worker RDMA run is otherwise hard to
reconstruct after the fact.

## 12. Validation Strategy

The benchmark is meant to support these validation flows:

1. compare Tensorcast vs Mooncake under the same worker count, prompt schedule,
   and model
2. compare TCP vs RDMA transport under the same worker inventory
3. scale worker count from 2 to larger `N`

The first implementation target should focus on correctness and observability:

- worker lifecycle is robust
- topology is explicit
- outputs are reproducible
- TTFT by position is trustworthy

Only after that should the benchmark be extended toward larger sweeps or
backend-specific tuning studies.
