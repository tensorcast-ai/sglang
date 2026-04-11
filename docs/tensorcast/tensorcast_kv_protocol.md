# Tensorcast KV Protocol v1 (SGLang Prefix Share + Request-level Transfer)

This document defines the **standard contract** for SGLang KV handling with
Tensorcast in two scenarios:

1. **Prefix share**
   - many requests share reusable prefix KV through a distributed KV pool.
2. **Request-level transfer**
   - one request's prompt-only KV state is published from a source serving
     instance with full-prompt, page-granular closure semantics and hydrated
     into a target serving instance so the target can continue through
     ordinary `/generate` without rerunning prefill.

This document freezes the semantic contract that the SGLang Tensorcast KV
integration must satisfy:

- prefix share is primarily an internal engine/storage contract,
- request-level transfer is primarily an external caller + instance-step
  contract.

Implementation details and internal design choices live in:
- `sglang/docs/tensorcast/tensorcast_kv_integration.md`

Status:
- The prefix-share half of this protocol is already implemented in the current
  SGLang repo as an internal HiCache backend over Tensorcast byte-artifact batch
  APIs, `HOST_SHARED` region-backed transfer, and optional allocator-backed
  direct host residency.
- Tensorcast core already provides the public plan surface needed for the
  request-level transfer control plane:
  `connect`, `directory`, `plan`, `publish`, `hydrate`,
  `prefetch_manifest_result`, `evict_local`.
- Tensorcast core now also provides the explicit-handle request-transfer
  surfaces required by the SGLang KV protocol:
  - `PublishResult.publish_manifest`, a controller-visible transfer handle that
    combines generic artifact manifest data with opaque engine-owned resume
    metadata.
  - `hydrate(publish_manifest=...)`, so target-side hydrate consumes the
    explicit source-produced snapshot handle instead of re-resolving by
    `engine_request_id`.
- Prefix share is expected to use a Tensorcast-backed internal SGLang data
  plane, not per-request external plan orchestration.
- The SGLang KV-specific instance-agent sidecar / `EngineAdapter` integration
  is substantially implemented in-repo today:
  - local-rank request-transfer state machines exist in the Tensorcast HiCache
    backend,
  - ordinary `/generate` lifecycle hooks already track live requests and
    prepared-bundle claim / cleanup,
  - rank-0 `publish` / `hydrate` / `evict_local` fanout now runs through the
    real scheduler control path,
  - an instance-scoped `launch_server()`-managed sidecar `NodeAgent` ingress
    plus directory registration / heartbeat are wired for the execution
    endpoint,
  - ordinary `/generate(rid)` can now claim and consume a prepared bundle after
    successful hydrate,
  - local external-caller end-to-end validation is green,
  - and remote external-controller end-to-end validation is still pending.
- Therefore this document mixes:
  - the already-implemented prefix-share contract, and
  - the request-transfer contract whose remote end-to-end validation is not
    complete yet.

Installation prerequisite:
- Inference servers that will execute Tensorcast plans must have Tensorcast
  installed. Recommended:
  - `uv pip install "sglang[tensorcast]"`
  - or `pip install "sglang[tensorcast]"`

---

## 0) Roles and Terms

- **Caller / Controller / Router**: an external control-plane program that
  chooses source and target instances, issues HTTP requests to SGLang, and
  submits Tensorcast plans.
- **P instance (source instance)**: the SGLang instance that owns the source
  prompt-only KV state for the request and any retained publishable prompt
  snapshot.
- **D instance (target instance)**: the SGLang instance that will hydrate that
  prompt-only snapshot and later continue through ordinary `/generate`.
- **Tensorcast daemon / worker**: the node-local Tensorcast runtime that serves
  worker actions such as prefetch and pinning.
- **NodeAgent / InstanceAgent**: the instance-scoped execution host boundary for
  Tensorcast instance steps. The current Tensorcast repository provides a
  standalone `NodeAgent` reference host. In the SGLang v1 integration, this
  boundary is implemented as an instance-scoped sidecar service launched by
  `launch_server()` for the logical SGLang instance rather than as a
  daemon-internal component.
- **SGLang EngineAdapter**: the SGLang-side integration layer, co-located with
  the instance-agent sidecar, that translates Tensorcast instance actions into
  SGLang KV operations.
- **Logical SGLang instance**: one serving instance as seen by the external
  caller. It MAY internally consist of multiple TP-rank processes that jointly
  own one request's KV state.
- **TP-group coordinator**: the SGLang-side coordinator for one logical
  instance. It receives Tensorcast instance-step calls for that logical
  `instance_id` and orchestrates all required ranks. In the recommended v1
  design this is the SGLang rank-0 control-plane ingress.

Terms:

- **KV live state**: the engine-local, mutable KV cache state currently owned by
  an SGLang instance for one request.
- **Published KV snapshot**: an immutable Tensorcast-published projection of the
  KV state required to resume decode on another instance.
- **Prefix bundle**: a reusable ordered set of KV pages representing a
  shareable prefix in the distributed KV pool.
- **Request bundle**: an ordered set of KV pages plus any request-scoped resume
  metadata sufficient to continue decode for one request on another instance.
- **SGLang request-bundle metadata**: SGLang-owned local metadata for one live
  logical request. It defines the authoritative publish cutoff and ordered page
  membership for one published snapshot generation. It is not a Tensorcast
  dataplane artifact.
- **Prepared local request bundle**: target-local SGLang integration state
  created by successful hydrate and later consumed by ordinary decode-request
  admission.
- **`engine_request_id`**: an adapter-local live-request handle used to locate
  source-side KV live state and local request state. It is not an immutable
  transfer handle.
- **`ManifestResult`**: the generic Tensorcast artifact-manifest description of
  a published KV snapshot, including stable artifact-set identity and an
  optional `ManifestArtifactSetBridge`.
- **`EngineOwnedManifest`**: an opaque engine-owned control-plane payload
  returned by publish and later consumed by hydrate. Tensorcast core should
  treat it as opaque. It is not a Tensorcast dataplane byte artifact.
- **`PublishManifest`**: the controller-visible immutable transfer handle for
  one published request snapshot. It contains the generic `ManifestResult` plus
  the opaque `EngineOwnedManifest`.
- **Instance route**: the current Tensorcast execution mapping from an
  `instance_id` to exactly one host daemon and one execution endpoint at a
  time.
- **Host daemon**: the Tensorcast daemon associated with an instance route. In
  a mixed target-side plan, worker steps must target this same daemon.
- **Daemon/instance relationship**: this protocol does not assume one daemon per
  instance. A daemon MAY host multiple instances. An instance resolves to one
  host daemon at a time for routed plan execution.
- **Required ranks**: the TP/PP ranks whose participation is necessary to
  publish or hydrate a complete decode-usable request bundle. For standard MHA
  models this is typically all TP ranks. Adapter-specific optimizations MAY
  reduce the physical page-publication owner set, but not the external
  group-scoped success semantics.

Normative language:
- **MUST**, **MUST NOT**, **SHOULD**, **MAY** are used as in RFC 2119.

---

## 1) Scope and Target Outcome

### 1.1 Two scenario classes

The v1 integration covers two distinct scenario classes:

1. **Prefix share**
   - a serving instance finds that part of a request can reuse prefix KV pages
     from the distributed KV pool.
2. **Request-level transfer**
   - a controller explicitly hands off a request's KV state from one instance to
     another.

### 1.2 Prefix-share target outcome

The v1 prefix-share path targets this operational outcome:

1. An SGLang instance serves a request and performs normal in-memory
   prefix-matching.
2. When useful, SGLang checks the distributed KV pool for additional prefix
   pages.
3. Matching pages are fetched back into SGLang host/device cache structures.
4. The request continues using the shared prefix without requiring an external
   caller to orchestrate the hot path.

### 1.3 Request-transfer target outcome

The v1 request-transfer path targets this operational outcome:

1. The controller chooses a source ordinary serving instance and a target
   ordinary serving instance.
2. The source instance serves a request and builds prompt-prefill KV for that
   request. It MAY already have emitted decode tokens by the time the
   controller decides to transfer.
3. The controller asks the source instance, through Tensorcast, to publish the
   request's prompt-only KV snapshot for that request. This published snapshot
   excludes source-emitted decode tokens and decode-only KV, closes all
   page-granular prompt pages for the full prompt boundary, and carries any
   non-page-aligned tail only through `tail_valid_tokens`.
4. Optionally, the controller asks the target host daemon to prefetch the
   published artifacts for performance.
5. The controller asks the target instance, through Tensorcast, to hydrate that
   prompt-only snapshot into its engine-local prepared runtime state.
6. The controller sends the same prompt to the target instance through the
   ordinary SGLang serving ingress, using the same logical request id / `rid`
   that the target hydrate phase prepared, so the target can reuse the hydrated
   prompt prefix and execute the first decode step locally.

### 1.4 Out of scope for v1

The following are out of scope for this protocol version:

- Partial-prefix transfer semantics.
- Multi-hop migration semantics.
- External-caller orchestration on every synchronous prefix-hit path.
- A single Tensorcast plan that spans a source instance on one daemon and a
  target instance on another daemon.
- Tensorcast-native load-aware scheduling or instance queue-length signals.
- A public generic `hydrate(artifact_set_ref=...)` API that bypasses
  engine-owned resume metadata. v1 instead standardizes an explicit
  `PublishManifest` transfer handle whose generic artifact portion can still be
  used for worker-side orchestration.
- Requiring direct L1-GPU <-> shared-pool zero-copy put/get for the prefix-share
  path. v1 assumes SGLang may keep its existing L1/L2 hierarchy and integrate
  Tensorcast at the host/L2-facing boundary.
- Storing request-resume metadata as a standalone Tensorcast dataplane byte
  artifact. v1 carries that information as `EngineOwnedManifest` in the control
  plane.
- Session-append or branch-lineage handoff where one logical request id may
  refer to multiple live conversational branches.
- Batch-request or parallel-sampling handoff where one external caller action
  expands into multiple scheduler `rid` values.
- Full visible-transcript handoff that includes source-emitted decode tokens or
  decode-only KV. In the baseline v1 SGLang profile, `publish()` MAY be
  invoked after decode has already begun, but the published snapshot is still
  limited to the request's prompt-only boundary and excludes emitted decode
  tokens.
- A publish call that keeps chasing future decode continuation or any prompt
  mutation beyond the request's fixed prompt boundary. The baseline v1 profile
  targets the full prompt for the request, MAY wait up to its deadline for
  prompt-page publication to close that boundary, and MUST NOT treat decode
  continuation as part of the publish scope.
- Solving DP-replica routing for the post-hydrate ordinary `/generate` request.
  The initial v1 implementation assumes the deployment or controller already
  routes both `hydrate()` and the subsequent ordinary request to the same
  concrete target serving instance.
- Targeting SGLang's current decode-only disaggregation instances as the
  post-hydrate runtime. The initial v1 implementation targets ordinary serving
  instances that accept normal `/generate` ingress and run the first decode
  step locally.

---

## 2) Prefix Share Contract

### 2.1 Ownership and execution path (MUST)

Prefix share MUST be treated as an engine-owned internal data path.

In particular:

- the normal synchronous prefix-share path MUST NOT require an external caller,
- the normal synchronous prefix-share path SHOULD NOT be expressed as a
  Tensorcast `Plan`,
- the serving instance itself SHOULD decide when to query and materialize
  storage-backed prefix pages.

### 2.2 Integration boundary (SHOULD)

Prefix share SHOULD be integrated at the SGLang:

- `HiRadixCache` + `HiCacheController`

boundary, using a Tensorcast-backed internal storage/runtime component.

It MAY present a Mooncake-like backend surface internally, such as:

- `batch_exists(...)`
- `batch_get_v1(...)`
- `batch_set_v1(...)`

For v1, this contract assumes a host/L2-facing integration boundary. The
protocol does not require bypassing SGLang's existing L1-to-L2 offload path for
prefix share.

### 2.3 Shared distributed substrate (MUST)

Prefix share MUST use the same distributed KV substrate as request-level
transfer.

That means:

- prefix pages stored for share MUST be the same underlying page artifacts that
  request-level transfer can later reuse,
- prefix bundle metadata and request bundle metadata MAY differ,
- but they MUST be able to reference the same page artifacts.

### 2.4 Identity rules for prefix share (MUST)

Prefix share MUST use stable page- and prefix-derived identity, such as:

- page hash values,
- prefix hash chains,
- prefix bundle identity,
- or equivalent content-derived keys.

Prefix share MUST NOT use `engine_request_id` as its distributed object
identity.

### 2.5 Use of `prefix_keys` and ordering hints (SHOULD)

If SGLang provides prefix-chain hints such as `prefix_keys`, the Tensorcast
backed prefix-share path SHOULD preserve and use them as lookup and batching
hints.

### 2.6 Optional programmability for prefix prewarm (MAY)

External programmability MAY still be used for coarse-grained prefix operations,
for example:

- prewarming a known prefix bundle on selected nodes,
- inspecting prefix bundle availability,
- background repair or rollout.

But this is optional and MUST be treated as a background/control-plane
operation, not the synchronous prefix-hit path.

### 2.7 Recommended internal prefix-share flow (SHOULD)

The normal prefix-share flow SHOULD look like:

1. `HiRadixCache.match_prefix(...)` checks in-memory prefix state.
2. If additional storage-backed prefix pages may exist, SGLang computes page
   hashes and optional `prefix_keys`.
3. The Tensorcast-backed internal backend performs:
   - `batch_exists(...)` to find the consecutive hit span,
   - `batch_get_v1(...)` to fetch matching pages into host pages.
4. SGLang inserts the fetched pages into host/radix structures.
5. SGLang continues normal host-to-device load and decode flow.

This is the canonical v1 usage contract for prefix share.

### 2.8 TP-group execution model for prefix share (SHOULD)

When an SGLang serving instance uses `TP > 1`, prefix share SHOULD remain a
per-rank engine hot path.

That means:

- each rank MAY maintain its own radix / host-cache structures,
- each rank MAY publish or fetch only its own physical KV shard pages into or
  from the shared substrate,
- rank-qualified storage identity MAY be used where the physical shard identity
  must be distinct,
- and the synchronous prefix-share path SHOULD NOT funnel ordinary page-level
  get/set through a central programmable coordinator.

Cross-rank consistency for prefix share SHOULD continue to follow SGLang's
existing TP synchronization rules, such as a minimum common hit length and a
shared insertion boundary, rather than introducing a new Tensorcast control-hop
for every page access.

### 2.9 Prefix bundle metadata schema (SHOULD)

The integration SHOULD define one canonical SGLang-owned prefix bundle metadata
schema.

This schema is an integration contract, not a Tensorcast-core type. The
protocol constrains the required fields and semantics, but does not require a
particular storage backend or public API.

Recommended minimal schema:

```yaml
schema: sglang.tensorcast.prefix_bundle.v1
prefix_bundle_id: pfxb:v1:sha256(...)
created_at_ms: 1774223000123
expires_at_ms: 1774223600123
state: ready

identity:
  model_fingerprint: llama3_70b_fp16
  kv_layout_id: sglang.kv.mha.page16k.v1
  tp_size: 8
  pp_size: 1
  attention_arch: mha
  prefix_keys_digest: sha256(...)
  terminal_prefix_hash: "..."
  matched_tokens: 1024
  logical_page_count: 64

rank_shards:
  - tp_rank: 0
    pp_rank: 0
    ordered_pages:
      - logical_page_index: 0
        artifact_id: cgid:byte_artifact~...
        page_hash: "..."
        layout_id: sglang.kv.page_shard.v1
      - logical_page_index: 1
        artifact_id: cgid:byte_artifact~...
        page_hash: "..."
        layout_id: sglang.kv.page_shard.v1
  - tp_rank: 1
    pp_rank: 0
    ordered_pages: [...]

constituent_digest:
  alg: sha256
  hex: "..."
```

Normative rules:

- `prefix_bundle_id` SHOULD be derived deterministically from the logical prefix
  identity and layout/topology inputs.
- Each `ordered_pages` list MUST be strictly ordered by
  `logical_page_index` for that rank shard.
- Each referenced `artifact_id` MUST point to the same underlying page byte
  artifacts used by the shared substrate.
- `state=ready` MUST mean all referenced pages are readable and layout/topology
  compatible for that bundle identity.
- A prefix bundle MUST become stale if any referenced page is missing,
  unreadable, layout-incompatible, topology-incompatible, or inconsistent with
  `constituent_digest`.

---

## 3) External Caller Contract for Request-level Transfer

This section applies to request-level transfer only. It does not apply to the
normal prefix-share hot path.

### 3.1 Discovery and routing inputs (MUST)

The controller MUST have, for each candidate SGLang serving instance:

- an SGLang serving endpoint for inference HTTP requests,
- a Tensorcast `instance_id`,
- the corresponding Tensorcast `daemon_id` or enough information to resolve it,
- an external load signal used for routing.

Important:

- The current Tensorcast directory can resolve instance execution routes, but it
  does not expose serving HTTP address or queue-length metrics such as
  `num_waiting_reqs`.
- Therefore load-aware routing in v1 MUST rely on SGLang-side telemetry,
  service discovery, or another control-plane registry in addition to Tensorcast
  directory APIs.

### 3.1.1 Coordinator-owned instance route registration (MUST)

For a logical SGLang instance with `TP > 1`, the rank-0 coordinator MUST be the
entity that owns Tensorcast-directory registration and heartbeat for that
logical `instance_id`.

That means:

- exactly one rank-0 coordinator registers the routable Tensorcast
  `instance_id`,
- non-coordinator TP ranks remain internal to the SGLang instance and MUST NOT
  register independent routable Tensorcast instances for request transfer,
- the coordinator's lifecycle defines the logical instance lifecycle for routed
  Tensorcast execution,
- and if the coordinator stops heartbeating, the logical instance MUST be
  treated as unavailable for routed Tensorcast instance steps.

The coordinator is expected to have the same lifecycle boundary as the logical
SGLang serving instance. If rank 0 dies, the logical instance should be
considered down.

### 3.1.2 Recommended route registration schema (SHOULD)

Recommended minimal registration contract:

```yaml
tensorcast_directory:
  instance_id: sgl-inst-17
  daemon_id: daemon-a
  engine: sglang
  execution_host_kind: node_agent_grpc
  execution_endpoint: 10.0.0.5:7310
  capability_flags:
    - instance_publish
    - instance_hydrate
    - instance_evict_local
    - execution_signals

sglang_side_metadata:
  serving_http_endpoint: http://10.0.0.5:30000
  tp_size: 8
  pp_size: 1
  coordinator_rank: 0
  coordinator_epoch: 0195c9d4-6e7a-7b91-b3b5-8f91f9d90d62
  lifecycle_state: ready
```

Normative rules:

- `execution_endpoint` MUST identify the coordinator-hosted instance-agent
  sidecar execution ingress for routed Tensorcast instance steps.
- `execution_host_kind` MUST match the actual transport host shape exposed by
  the implementation; the current SGLang integration uses the generic
  `node_agent_grpc` value and relies on `engine=sglang` plus capability flags
  to identify the SGLang profile.
- `serving_http_endpoint` is not currently a standard Tensorcast directory
  field; callers SHOULD obtain it from SGLang-side discovery/telemetry rather
  than from Tensorcast directory APIs.
- `coordinator_epoch` SHOULD change on coordinator restart so stale observers
  can distinguish a restarted logical instance from an old one.
- The coordinator SHOULD heartbeat this route for as long as the logical
  instance is routable for Tensorcast instance-step execution.

### 3.2 TP-group mapping for request transfer (MUST)

When an SGLang serving instance uses `TP > 1`, one Tensorcast `instance_id`
MUST represent the whole logical SGLang instance / TP group, not an individual
TP rank.

Therefore:

- the external caller MUST target one logical `instance_id` for request-level
  `publish`, `hydrate`, and `evict_local`,
- the external caller MUST NOT be required to enumerate per-rank `instance_id`
  values,
- and the public Tensorcast programmable surface for request transfer remains
  group-scoped even if the SGLang integration internally fans out work to
  multiple ranks.

For the baseline v1 contract, source and target instances SHOULD use
layout-compatible TP/PP topology for request transfer. Cross-topology reshape
semantics MAY be added later only if the SGLang adapter explicitly documents the
request-bundle compatibility rules.

### 3.3 Recommended controller program shape (v1)

The v1 controller SHOULD look like this:

```python
import tensorcast as tc


def to_tc_instance(route):
    return tc.Instance(
        instance_id=route.instance_id,
        daemon_id=route.daemon_id,
        engine=route.engine or "",
        execution_endpoint=route.execution_endpoint,
    )


def find_worker_for_daemon(rt: tc.Runtime, daemon_id: str) -> tc.Worker:
    for route in rt.directory().list_workers().value:
        if route.daemon_id == daemon_id and route.daemon_address:
            return tc.Worker(
                worker_id=route.worker_id or daemon_id,
                daemon_address=route.daemon_address,
                daemon_id=route.daemon_id,
            )
    raise RuntimeError(f"no worker route for daemon_id={daemon_id}")


rt = tc.connect(daemon_address=gateway_addr)

# These come from SGLang-side discovery + load metrics, not Tensorcast alone.
p_meta = choose_prefill_instance_from_sglang_metrics(...)
d_meta = choose_decode_instance_from_sglang_metrics(...)

p_inst = to_tc_instance(
    rt.directory().resolve_instance_execution(p_meta.instance_id).value
)
d_inst = to_tc_instance(
    rt.directory().resolve_instance_execution(d_meta.instance_id).value
)
d_worker = find_worker_for_daemon(rt, d_inst.daemon_id or "")

logical_request_id = get_engine_request_id_for_transfer(...)

await SglClient(p_meta.http_addr).completion(
    prompt,
    max_tokens=1,
    req_id=logical_request_id,
)

# Phase 1: source publish
# This call may happen while the source request is still active or after the
# source request has already produced output. The published snapshot still
# covers only the request's prompt-only boundary.
ctx1 = tc.context(
    request_id=f"kv-publish:{logical_request_id}",
    deadline_ms=15_000,
    idempotency_key=f"kv-publish:{logical_request_id}",
)
plan1 = rt.plan(ctx1)
pub = plan1.on_instance(p_inst).publish(
    engine_request_id=logical_request_id,
    ttl_ms=60_000,
)
res1 = plan1.run()
publish_result = res1.step(pub).artifact_result
publish_manifest = publish_result.publish_manifest
artifact_manifest = publish_manifest.artifact_manifest

# Phase 2: target warmup + hydrate
ctx2 = tc.context(
    request_id=(
        "kv-load:"
        f"{artifact_manifest.key_set_digest_hex}"
    ),
    deadline_ms=15_000,
    idempotency_key=(
        "kv-load:"
        f"{artifact_manifest.key_set_digest_hex}"
    ),
)
plan2 = rt.plan(ctx2)
warm = plan2.on_worker(d_worker).prefetch_manifest_result(
    artifact_manifest,
    device="cpu",
)
hyd = plan2.on_instance(d_inst).hydrate(
    publish_manifest=publish_manifest,
    depends_on=[warm],
)
res2 = plan2.run()

await SglClient(d_meta.http_addr).completion(
    prompt,
    req_id=logical_request_id,
)
```

Notes:

- `PublishResult.publish_manifest` and `hydrate(publish_manifest=...)` are the
  preferred target surfaces for the SGLang KV integration and are implemented
  in current Tensorcast core.
- Tensorcast core also still exposes the legacy
  `publish(...).artifact_result.manifest` alias and
  `hydrate(engine_request_id=...)` compatibility surface. The protocol below
  treats that legacy hydrate form as compatibility mode only.
- The final decode continuation uses the ordinary SGLang request path. The
  target-side integration is expected to bind the caller-supplied
  `req_id=logical_request_id` to the prepared local request bundle created by
  successful hydrate.

### 3.4 Multi-plan requirement (MUST)

The controller MUST treat source publish and target hydrate as separate control
phases.

Rationale:

- Current Tensorcast daemon ingress supports at most one routed `instance_id`
  per request.
- Therefore a source `publish` on `P` and a target `hydrate` on `D` cannot be
  combined into one routed plan when `P` and `D` are different instances.

v1 controller rules:

- Source-side `publish` MUST be its own plan.
- Target-side `hydrate` MUST be a separate later plan.
- Optional target-side worker warmup MAY be:
  - a separate worker-only plan, or
  - combined with target-side `hydrate` in the same plan, provided that:
    - the plan contains only the target instance `D` as its instance target,
    - all worker steps in that plan target `D`'s host daemon.

---

## 4) Identity and Bundle Contract

### 4.1 Live-request handle vs transfer handle (MUST)

Request-level transfer MUST distinguish between:

- a **live-request handle**, used to identify mutable request state inside one
  logical SGLang instance, and
- an **immutable transfer handle**, used to identify one published request
  snapshot across instances.

For v1:

- `engine_request_id` is the live-request handle,
- `PublishManifest` is the immutable transfer handle.

There is no separate controller-facing string `transfer_handle` in v1. The
entire `PublishManifest` object is the authoritative transfer handle.

Implementations MAY embed an engine-local `publish_id`, `snapshot_id`, or
equivalent debug handle inside the opaque `EngineOwnedManifest` payload, but
callers MUST treat the full `PublishManifest` as authoritative.

### 4.2 `engine_request_id` semantics for request transfer (MUST)

For v1, `engine_request_id` is an adapter-local lookup handle for source-side
request state while that state is still publishable.

It MUST be valid for:

- source-side `publish(engine_request_id=...)`,
- source- or target-side `evict_local(engine_request_id=...)`,
- and any adapter-local live-request lookup the engine integration documents.

For the SGLang v1 profile, this means `engine_request_id` MAY resolve either:

- the active source request state while the request is still running, or
- a retained prompt-snapshot record after the ordinary source request has
  completed but before that retained publish window expires.

It MUST NOT be treated as:

- Tensorcast artifact identity,
- Tensorcast artifact-set identity,
- immutable request-snapshot identity,
- Tensorcast workflow identity,
- Tensorcast truth for distributed snapshot currentness.

Stable published request-bundle identity belongs to:

- `PublishManifest`,
- `ManifestResult.key_set_digest_hex`,
- `ManifestArtifactSetBridge`,
- `ArtifactSetRef`,
- and the artifact identities inside the generic artifact manifest.

### 4.3 `PublishManifest` and `EngineOwnedManifest` contract (MUST)

`publish()` MUST return a controller-visible `PublishManifest` for the exact
snapshot it created.

The recommended shape is:

```yaml
publish_manifest:
  schema: tensorcast.publish_manifest.v1
  artifact_manifest: <ManifestResult>
  engine_owned_manifest:
    engine: sglang
    schema: sglang.engine_owned_manifest.v1
    version: 1
    encoding: json
    created_at_ms: 1774223000123
    expires_at_ms: 1774223060123
    artifact_manifest_digest: "<artifact_manifest.key_set_digest_hex>"
    payload_sha256: "..."
    payload: "<opaque engine-owned resume payload>"
```

For the initial SGLang v1 profile, the opaque payload SHOULD minimally carry a
compatibility envelope like:

```yaml
payload:
  schema: sglang.request_bundle_payload.v1
  transfer_mode: prefill_closed_prompt_reuse
  logical_request_id: "req-123"
  cutoff_token_count: 11712
  frozen_last_page_index: 365
  tail_valid_tokens: 0
  prompt_token_digest:
    alg: sha256
    hex: "..."
  compatibility:
    model_fingerprint: llama3_70b_fp16_ckpt42
    kv_layout_id: sglang_kv_page_v2_page_blob_direct_torch.float16_ps32_mha
    dtype: float16
    page_size: 32
    tp_size: 2
    pp_size: 1
    attention_arch: mha
    required_ranks:
      - tp_rank: 0
        pp_rank: 0
      - tp_rank: 1
        pp_rank: 0
```

The current SGLang payload keeps the compatibility name
`prefill_closed_prompt_reuse`, but the normative v1 meaning is:

- publish the request's full prompt-only boundary,
- exclude emitted decode tokens and decode-only KV,
- and let the target ordinary `/generate` run the first decode step locally.

Normative rules:

- `artifact_manifest` is the generic Tensorcast-visible artifact description
  used for worker warmup and artifact-set orchestration.
- `engine_owned_manifest` is an opaque engine-owned payload used only by the
  engine integration to reconstruct decode-usable runtime state.
- Tensorcast core MUST NOT need to interpret
  `engine_owned_manifest.payload`.
- `engine_owned_manifest` MUST NOT be represented as a standalone Tensorcast
  dataplane byte artifact.
- `engine_owned_manifest.artifact_manifest_digest` MUST bind the opaque resume
  payload to the exact generic artifact manifest it was produced with.
- `payload_sha256`, when present, SHOULD cover only the control-plane payload
  bytes inside `engine_owned_manifest.payload`; it is not a second KV-page data
  digest and it MUST NOT require hashing the underlying page payloads again.
- The initial SGLang v1 profile SHOULD validate at least:
  - `model_fingerprint`
  - `kv_layout_id`
  - `dtype`
  - `page_size`
  - `tp_size`
  - `pp_size`
  - `attention_arch`
  - `required_ranks`
  - `cutoff_token_count`
  - `prompt_token_digest`
- The initial SGLang v1 profile SHOULD treat `cutoff_token_count` as the full
  prompt token count for the request rather than an aligned prefix cutoff.
- The closed byte-artifact set still remains page-granular in v1.
- If the full prompt ends inside a partially filled page, the source SHOULD
  record the trailing prompt remainder in `tail_valid_tokens` rather than
  failing the whole request-bundle publication.
- The initial SGLang v1 profile SHOULD NOT directly transfer the partial tail
  page bytes; `tail_valid_tokens` is the control-plane expression of that
  remainder while the artifact closure stays page-granular.
- If any required compatibility field mismatches on the target side,
  `hydrate(publish_manifest=...)` MUST fail closed rather than attempting a
  best-effort reinterpretation.

`PublishResult.publish_manifest` and `hydrate(publish_manifest=...)` are target
protocol surfaces for the SGLang KV integration and are implemented in current
Tensorcast core.

### 4.4 Prefix-share identity semantics (MUST)

Stable prefix-share identity belongs to:

- page hashes,
- prefix hash chains,
- prefix bundle identity,
- and the artifact identities inside the corresponding bundle manifest.

### 4.5 Relationship to SGLang request identity (SHOULD/MAY)

The v1 controller SHOULD obtain `engine_request_id` from the SGLang-side
integration layer rather than synthesizing a random string at the Tensorcast
boundary.

For the initial integration, the SGLang implementation MAY choose:

- `engine_request_id == sglang_request_id`

provided that the implementation can guarantee the required uniqueness,
lifetime, and lookup semantics for source-side publish and local request-state
operations.

For the initial SGLang request-transfer profile, the recommended concrete
mapping is:

- `logical_request_id` is the controller-visible workflow identity for one
  request handoff attempt,
- SGLang `/generate` uses an explicit caller-provided `rid=logical_request_id`,
- Tensorcast `publish(engine_request_id=...)` uses
  `engine_request_id=logical_request_id`,
- the target-side ordinary `/generate` after `hydrate()` also uses the same
  caller-provided `rid=logical_request_id`,
- and the target runtime is an ordinary serving instance that accepts normal
  `/generate` ingress and executes the first decode step locally.

This equality is a v1 integration profile, not a permanent Tensorcast protocol
requirement. Future versions MAY decouple these identifiers once SGLang adds a
more explicit request-transfer admission object.

However, the protocol does not require these two identifiers to be equal.
Future versions MAY refine source-side live-request lookup independently of the
published transfer-handle format.

The SGLang integration SHOULD reject the v1 request-transfer path when the
serving ingress would auto-generate, rewrite, or fan out `rid`, including:

- batch requests that produce multiple scheduler requests,
- parallel sampling that regenerates per-sample `rid` values,
- session append/replace flows that reinterpret one caller request against prior
  lineage,
- deployments where the ordinary post-hydrate `/generate` may be routed to a
  different DP replica than the one that executed `hydrate()`,
- target decode-only disaggregation instances that do not accept the same
  ordinary `/generate` continuation shape,
- or any other mode where one controller-visible logical request cannot map to
  exactly one stable serving `rid`.

### 4.6 Multiple publish generations (MUST/SHOULD)

Repeated `publish()` operations for the same `engine_request_id` MAY occur.
Each successful publish MUST mint a distinct immutable `PublishManifest` for
that snapshot generation.

Such a generation MAY be produced either from an active source request or from
a retained prompt-only snapshot after the source ordinary request completes.
Regardless of invocation timing, the published generation still covers only the
request's prompt-only KV state. For v1, the closed byte-artifact set remains
page-granular; if the prompt ends inside a partially filled page,
`tail_valid_tokens` records that trailing prompt tail while the partial tail
page itself is not directly transferred.

Controllers SHOULD pass the exact `PublishManifest` returned by the intended
publish phase into the corresponding hydrate phase.

An older `PublishManifest` MUST NOT be silently re-bound to a newer snapshot for
the same live request.

### 4.7 TP-group scope of live-request and transfer handles (MUST)

For request-level transfer on a logical instance with `TP > 1`,
both `engine_request_id` and `PublishManifest` MUST denote the logical request
across the whole instance / TP group.

It MUST NOT be interpreted at the public caller surface as:

- a per-rank shard identifier,
- a per-rank publication handle,
- or a public signal that the caller should issue one instance-step call per
  TP rank.

The same `engine_request_id` SHOULD be passed to source-side `publish` and to
cleanup actions that operate on the same logical live request.

The `PublishManifest` returned from that publish SHOULD be passed to the target
hydrate of the same handoff.

The implementation MAY keep additional rank-local handles or metadata
internally, but such details are adapter-owned and MUST NOT leak into the
public controller contract.

---

## 5) Source-side Publish Contract

### 5.1 Publish meaning (MUST)

`plan.on_instance(P).publish(engine_request_id=E, ...)` MUST mean:

- locate the source instance's active request for `E`, or a retained
  source-side prompt snapshot for `E`, together with the SGLang-owned
  request-bundle metadata for that publishable source state,
- freeze one fixed snapshot cutoff for this publish generation,
- construct an immutable published snapshot containing only the prompt-visible
  prefix KV needed for the target ordinary `/generate` path to reuse that
  prompt prefix and execute the first decode step locally for exactly that
  cutoff,
- establish closure for that snapshot over the shared Tensorcast substrate,
- return a `PublishResult` whose `publish_manifest` describes that published
  snapshot.

For v1, this MUST NOT be interpreted as:

- "dump whatever live device state still happens to exist after the request is
  done"

It MUST be interpreted as:

- finalize one substrate-backed request bundle whose exact cutoff and page
  membership are defined by SGLang-owned request-bundle metadata.

Even if `publish()` is invoked after the source has already emitted one or more
decode tokens, or after the source ordinary request has completed, those decode
tokens and any decode-only KV MUST NOT become part of the published snapshot.

That request-bundle metadata:

- MUST remain an SGLang integration concern,
- MUST NOT be stored as a standalone Tensorcast dataplane byte artifact,
- and MUST be authoritative for "which pages belong to this snapshot
  generation".

### 5.2 TP-group scope and coordinator behavior (MUST)

If the source logical instance `P` uses `TP > 1`, `publish()` MUST be defined
over the complete logical instance, not over a single TP-rank shard.

The SGLang-side coordinator for `P` MUST:

- receive the single Tensorcast instance-step call for the logical
  `instance_id`,
- fan the publish operation out to all required ranks,
- wait for all required ranks to either succeed or fail,
- aggregate the per-rank page publication results into one group-level request
  bundle manifest,
- and return one external `PublishResult` for the whole logical instance.

For standard MHA layouts, the required-rank set is typically all TP ranks. For
layouts such as MLA, the adapter MAY optimize which ranks physically publish
pages, but such optimizations MUST preserve the same external group-scoped
semantics and success boundary.

A source adapter MUST NOT publish only one required rank's shard and report
success for the whole request bundle.

### 5.3 Mixed-state compatibility with prefix-share writes (MUST)

It is normal and expected that, at publish time, the request snapshot spans a
mixed state such as:

- some required KV pages are already present in the shared Tensorcast KV pool
  because of prior prefix-share or write-through activity,
- some required pages are currently being published by the internal prefix-share
  data path,
- some required pages are not yet present in the shared KV pool.

This mixed state MUST NOT be treated as a protocol conflict.

Instead, source-side `publish()` MUST be defined as a **closure-establishing**
operation over the request bundle:

- reuse already-present pages,
- wait for or adopt compatible in-flight page publication when possible,
- publish missing pages,
- and only succeed once all pages required by the request bundle satisfy the
  publish contract.

This closure MUST be evaluated against one fixed publish cutoff:

- `publish()` MUST choose a cutoff before claiming success,
- it MUST close all pages up to that cutoff,
- it MAY wait for or force missing tail pages that belong to that cutoff to
  reach the shared substrate,
- and it MUST NOT keep chasing newer tokens produced after that cutoff.

For the initial SGLang v1 profile:

- the chosen cutoff MUST be the request's full prompt token count; emitted
  decode tokens MUST NOT extend the cutoff or page membership,
- the closed artifact set remains page-granular in v1,
- if the full prompt ends inside a partially filled page, that trailing prompt
  remainder SHOULD be expressed through `tail_valid_tokens` rather than causing
  the whole request-bundle publication to fail,
- `publish()` MAY be invoked while the source request is still active, after it
  has already emitted decode tokens, or after the ordinary source request has
  completed, provided the adapter still retains a publishable prompt snapshot,
- the baseline v1 profile MAY wait, up to the publish deadline, for prompt
  prefill / page publication work needed to close that full prompt boundary,
  but MUST NOT wait for decode continuation or any newer prompt mutation,
- and `tail_valid_tokens` is the external encoding for a non-page-aligned
  prompt tail while the v1 byte transfer itself remains page-granular.

### 5.4 Completeness (MUST)

Publish MUST fail if the adapter cannot produce a complete prompt-prefix
snapshot sufficient for target-side ordinary `/generate` continuation.

The source adapter MUST NOT silently publish a partial request state and claim
success.

### 5.5 Retention and adoption semantics for already-present pages (MUST)

For request-level transfer, page existence alone is not sufficient.

If a required page already exists in the shared KV pool, `publish()` MUST still
ensure that the page satisfies request-transfer requirements such as:

- readability,
- compatible content identity,
- and sufficient retention/lifetime for the transfer window.

Therefore source-side `publish()` MUST conceptually perform:

- `reuse + retain`

rather than merely:

- `reuse if present`

### 5.6 Returned transfer-handle contract (MUST)

On success, `PublishResult.publish_manifest` MUST carry the authoritative
transfer handle for that published snapshot.

The returned `PublishManifest` MUST:

- contain the generic artifact-manifest component needed for worker-side
  orchestration,
- contain the opaque `EngineOwnedManifest` needed for target-side hydrate,
- be self-contained enough that target-side hydrate does not need to re-query
  the source instance for "the latest" request bundle metadata,
- and remain immutable once returned to the caller.

The generic artifact-manifest component SHOULD include a valid
`ManifestArtifactSetBridge` so the controller can call:

```python
plan.on_worker(worker).prefetch_manifest_result(
    publish_manifest.artifact_manifest,
    device=...,
)
```

without having to reconstruct the underlying artifact set itself.

For compatibility, Tensorcast MAY continue exposing the generic artifact portion
through a legacy `PublishResult.manifest` alias. However, the controller SHOULD
treat `PublishResult.publish_manifest` as the authoritative request-transfer
output.

### 5.7 Atomic success boundary (MUST)

`publish()` MUST only report success after:

1. all required ranks have either contributed their required shards or have been
   validly accounted for by an adapter-documented optimization policy,
2. all required pages of the request bundle have been made available in the
   shared KV substrate with sufficient retention for transfer, and
3. the request bundle manifest has been committed as the authoritative bundle
   description for subsequent hydrate.

`publish()` MUST NOT report success merely because some subset of the required
pages or ranks is already satisfied.

### 5.8 TTL and lifetime (SHOULD)

The controller SHOULD provide a `ttl_ms` long enough to cover:

- publish completion,
- optional target-side prefetch,
- target-side hydrate,
- controller retries within the same transfer window.

For the current Tensorcast caller surface, this `ttl_ms` is provided on the
request-level publish action itself, for example:

```python
plan.on_instance(p_inst).publish(
    engine_request_id=logical_request_id,
    ttl_ms=60_000,
)
```

Normative retention rules:

- this `ttl_ms` is a minimum transfer-retention intent for all request-bundle
  page artifacts required by that publish,
- if a required page already exists in the shared substrate with insufficient
  remaining lifetime, `publish()` MUST upgrade retention before reporting
  success,
- the validity window advertised by `PublishManifest.engine_owned_manifest`
  SHOULD be consistent with the retained artifact lifetime for that snapshot.

The adapter MAY retain source-local prompt-snapshot state independently of
Tensorcast artifact TTL so that `publish()` can still succeed for a short
window after the ordinary source request completes.

If that retained prompt snapshot expires or is explicitly cleaned, subsequent
`publish(engine_request_id=...)` for the same source request MUST fail closed
rather than attempting to reconstruct a snapshot from decode-only state.

If the deployment later wants long-lived or durable request snapshots, the
integration MAY use stronger Tensorcast retention/policy mechanisms behind the
same publish contract. But the v1 request-transfer baseline is TTL-scoped
retention, not an implicitly durable snapshot.

---

## 6) Optional Target-side Worker Warmup Contract

### 6.1 Warmup is an optimization, not a correctness requirement

The controller MAY insert target-side worker steps before hydrate:

```python
plan.on_worker(
    d_worker
).prefetch_manifest_result(
    publish_manifest.artifact_manifest,
    device=...,
)
```

This warmup is optional and exists to reduce hydrate latency for high-cardinality
KV byte artifacts.

Hydrate correctness MUST NOT depend on the caller having performed this worker
warmup.

### 6.2 Warmup target (MUST)

If warmup is used, it MUST target the Tensorcast worker/daemon associated with
the target instance `D`.

### 6.3 Readiness floor (MUST)

The controller MUST treat `prefetch_set` / `prefetch_manifest_result` as
guaranteeing only the Tensorcast readiness floor `local_replica_ready`.

It MUST NOT assume stronger placement guarantees than that.

---

## 7) Target-side Hydrate Contract

### 7.1 Hydrate meaning (MUST)

`plan.on_instance(D).hydrate(publish_manifest=M)` MUST mean:

- validate that `M` is well-formed and compatible with the target engine /
  layout,
- use `M.artifact_manifest` as the generic artifact description for
  materialization and worker warmup,
- use `M.engine_owned_manifest` as the opaque engine-owned resume payload,
- attempt to materialize the required artifacts if they are not already locally
  ready,
- allow artifact fetch/materialization to be internally best-effort at the
  per-artifact level,
- let the EngineAdapter decide which successfully materialized pages actually
  form a decode-usable prefix / runnable state,
- reconstruct decode-usable KV live state inside the target SGLang instance
  only from the artifact subset that is both available and engine-compatible,
- prepare target-local reusable KV/cache state only; it MUST NOT inject a
  prebuilt decode request, fabricate first-token output, or reuse SGLang's
  existing PD-disaggregation prebuilt decode semantics,
- return only when that local engine state is ready for decode continuation or
  the operation has failed.

### 7.2 TP-group scope and coordinator behavior (MUST)

If the target logical instance `D` uses `TP > 1`, `hydrate()` MUST be defined
over the complete logical instance, not over a single TP-rank shard.

The SGLang-side coordinator for `D` MUST:

- receive the single Tensorcast instance-step call for the logical
  `instance_id`,
- resolve the authoritative request bundle from the provided
  `PublishManifest`,
- determine the per-rank shard assignment needed for decode continuation,
- fan the hydrate operation out to all required ranks,
- wait for all required ranks to either succeed or fail,
- and return one external hydrate result for the whole logical instance.

Hydrate MUST fail if any required rank fails, times out, or cannot reconstruct
its portion of the runnable decode state.

### 7.3 Preferred resolution source (MUST)

The preferred request-transfer contract is explicit-handle hydrate:

- publish returns `PublishManifest`,
- optional worker warmup uses `PublishManifest.artifact_manifest`,
- hydrate consumes that same `PublishManifest` by value.

The target-side hydrate path MUST NOT require a source-instance round-trip to
reconstruct or rediscover "the latest" request bundle.

### 7.4 Legacy `hydrate(engine_request_id=...)` compatibility mode (MAY/DEPRECATED)

Current Tensorcast core still exposes `hydrate(engine_request_id=...)`. In the
target SGLang KV design, this should be treated as deprecated compatibility
mode rather than the primary request-transfer protocol path.

If retained, it SHOULD be implemented as a **controller-side convenience shim**
instead of a target-instance-side resolution mechanism.

That means:

- the controller keeps a controller-owned cache or control-plane registry that
  maps a logical request id to one or more published `PublishManifest`
  generations,
- a convenience call shaped like `hydrate(engine_request_id=E)` first resolves
  exactly one cached `PublishManifest` for `E`,
- and the actual routed Tensorcast instance step sent to the target instance is
  still `hydrate(publish_manifest=M)`.

Controller-side compatibility resolution MUST:

- fail closed if zero or multiple candidate `PublishManifest` objects are
  associated with `E`,
- avoid any implicit source-instance round-trip,
- and avoid any "latest publish wins" guess based only on wall-clock time.

This compatibility mode SHOULD be limited to single-controller deployments. If
multiple independent controllers may publish or hydrate the same logical
request id, the controller-side shim is unsafe unless the deployment adds a
stronger controller-owned generation discipline.

Target-side SGLang integration MUST NOT be required to resolve transfer
snapshots by `engine_request_id`.

The canonical request-transfer protocol remains:

- `publish(engine_request_id=...) -> PublishManifest`
- `hydrate(publish_manifest=...)`

### 7.5 Success criteria (MUST)

Hydrate MUST fail if the target instance cannot reconstruct runnable decode state
for the request.

In particular:

- missing required artifacts,
- layout mismatches,
- decode-engine insertion failures,
- or incomplete KV reconstruction

MUST surface as hydrate failure.

Therefore:

- per-artifact fetch / install work MAY be internally best-effort,
- `HydrateResult` MAY report partial artifact transport outcomes for diagnosis,
- but the externally visible `hydrate()` success boundary remains fail-closed:
  it succeeds only if the EngineAdapter has produced one runnable prepared local
  request bundle for decode continuation.

For `TP > 1`, this success criterion applies to the whole logical instance:
one required rank failing means the group hydrate fails.

### 7.6 Partial-install handling (SHOULD)

If some target ranks have already materialized local state but the group hydrate
later fails, the integration SHOULD treat the target local state as tainted or
incomplete.

The coordinator SHOULD attempt best-effort group cleanup through the same
internal fan-out path used for request-level `evict_local`.

Such cleanup does not change the externally visible result: the Tensorcast
`hydrate()` call still fails.

### 7.7 Decode continuation boundary (MUST)

After successful hydrate, the target instance MUST be able to accept the decode
continuation request for the same logical request without requiring the caller to
re-run prefill.

For the SGLang v1 integration, the preferred continuation shape is:

- `hydrate(publish_manifest=...)` prepares target-local request state,
- the controller then sends the ordinary SGLang generate/decode request through
  the normal serving ingress,
- and that ordinary request carries the same logical request id in the
  caller-visible `rid` field.

Before claiming a prepared bundle for that ordinary request, the target-side
admission path MUST:

- run the same incoming prompt tokenization / normalization that ordinary
  SGLang admission would otherwise use,
- verify that the resulting prompt tokenization is compatible with the prepared
  bundle envelope,
- and at minimum re-check `cutoff_token_count` together with
  `prompt_token_digest`.

The target-side integration MUST therefore:

- install one prepared local request bundle keyed by that logical request id on
  successful hydrate,
- bind ordinary request admission for the same `rid` to one clean prepared
  local request bundle when such a bundle exists,
- claim that prepared bundle only after the incoming request passes the
  tokenization revalidation described above,
- fall back to the normal SGLang admission path if no usable prepared bundle
  exists,
- and never silently fabricate a successful resume from a stale, tainted, or
  incompatible prepared record.

Recommended conflict rules:

- if the target already has a live request for the same logical request id,
  hydrate or resume admission SHOULD fail closed rather than overwrite it,
- if multiple clean conflicting prepared-bundle generations exist for the same
  logical request id, admission SHOULD fail closed rather than guess,
- if only stale, tainted, failed, evicted, or compatibility-mismatched
  prepared records exist for that logical request id, ordinary `/generate`
  SHOULD log a warning, ignore those records for admission, and continue with
  the normal SGLang path,
- if the incoming ordinary request fails prompt revalidation against
  `prompt_token_digest` / `cutoff_token_count`, ordinary `/generate` SHOULD log
  a warning, skip prepared-bundle claim, and continue with the normal SGLang
  path,
- and one prepared bundle generation SHOULD be consumed by at most one ordinary
  decode admission in the v1 design.

The protocol does not require a new public "resume by manifest" HTTP endpoint
for v1.

---

## 8) Cleanup Contract

### 8.1 Source cleanup (MAY/SHOULD)

After target-side hydrate succeeds and the controller has committed to decode on
the target instance, the controller MAY ask the source instance to free local KV
state:

```python
plan.on_instance(p_inst).evict_local(engine_request_id=logical_request_id)
```

For `TP > 1`, this cleanup is a group-scoped operation over the logical source
instance.

For memory-sensitive deployments, source cleanup after successful handoff SHOULD
be the default policy.

### 8.2 Failure cleanup (SHOULD)

If publish succeeds but hydrate fails, the controller SHOULD apply a deployment
policy for cleanup and retry, for example:

- keep source local state and retry target hydrate elsewhere,
- or evict partially created target local state across the target logical
  instance before retry.

This retry policy is controller-owned and not automatically provided by
Tensorcast plan rollback.

---

## 9) Idempotency, Errors, and Retry

### 9.1 Per-phase idempotency keys (MUST)

The controller MUST use distinct base `idempotency_key` values for distinct
transfer phases, for example:

- `kv-publish:{engine_request_id}`
- `kv-load:{publish_manifest.artifact_manifest.key_set_digest_hex}`
- `kv-evict:{engine_request_id}`

The controller MUST NOT reuse one constant idempotency key across unrelated
transfers.

### 9.2 Retry boundary (MUST)

The controller MUST treat publish, warmup, hydrate, and cleanup as separate
retry domains.

Tensorcast does not provide transactional rollback across these phases.
Successful earlier side effects are not automatically undone when a later phase
fails.

### 9.3 Deadlines (SHOULD)

The controller SHOULD set explicit per-plan `deadline_ms` values appropriate for
the transfer phase and deployment SLA.

---

## 10) Current Tensorcast Execution Constraints

The v1 protocol is intentionally shaped around current Tensorcast execution
constraints:

- Local caller-side `Plan.run()` does not execute instance steps directly.
- Runtime/daemon-ingress execution supports only terminal-only plan execution.
- A routed plan supports at most one instance target.
- In a routed mixed plan, worker steps must target the host daemon of that same
  routed instance.
- For a TP-group-backed SGLang instance, rank fan-out behind that one
  `instance_id` is owned by the SGLang integration rather than Tensorcast core.

These constraints are why the protocol standardizes:

- a source-side publish phase,
- a separate target-side hydrate phase,
- and optional target-side warmup attached to the target phase.

---

## 11) Summary

The v1 Tensorcast KV protocol for SGLang has two coordinated parts:

1. **Prefix share**
   - SGLang internally uses a Tensorcast-backed distributed KV substrate for
     prefix reuse.
   - This path is engine-owned, page-oriented, and not externally orchestrated
     per synchronous hit.
2. **Request-level transfer**
   - Controller selects `P` and `D` using external SGLang-aware telemetry.
   - Controller issues prefill to `P`.
   - Controller runs a source logical-instance
     `publish(engine_request_id=E)` plan.
   - Controller obtains `PublishResult.publish_manifest`.
   - Controller optionally warms the target host daemon using
     `prefetch_manifest_result(publish_manifest.artifact_manifest, ...)`.
   - Controller runs a target logical-instance
     `hydrate(publish_manifest=M)` plan.
   - Controller issues decode continuation to `D`.
   - Controller optionally evicts source local KV state.

This is the canonical v1 protocol baseline that the SGLang Tensorcast KV
integration must implement.
