# Tensorcast KV Protocol v1 (SGLang Prefix Share + Request-level Transfer)

This document defines the **standard contract** for SGLang KV handling with
Tensorcast in two scenarios:

1. **Prefix share**
   - many requests share reusable prefix KV through a distributed KV pool.
2. **Request-level transfer**
   - one request's KV state is handed off from a source prefill instance to a
     target decode instance for PD-disaggregated inference.

This document freezes the semantic contract that the SGLang Tensorcast KV
integration must satisfy:

- prefix share is primarily an internal engine/storage contract,
- request-level transfer is primarily an external caller + instance-step
  contract.

Implementation details and internal design choices live in:
- `sglang/docs/tensorcast/tensorcast_kv_integration.md`

Status:
- Tensorcast core already provides the public plan surface needed for the
  request-level transfer control plane:
  `connect`, `directory`, `plan`, `publish`, `hydrate`,
  `prefetch_manifest_result`, `evict_local`.
- Prefix share is expected to use a Tensorcast-backed internal SGLang data
  plane, not per-request external plan orchestration.
- The SGLang KV-specific in-process instance-agent / `EngineAdapter`
  integration is not implemented yet.
- Therefore this document is a **v1 target protocol** for the upcoming KV
  integration, not a statement that end-to-end prefix share and request
  transfer are already working in SGLang today.

Installation prerequisite:
- Inference servers that will execute Tensorcast plans must have Tensorcast
  installed. Recommended:
  - `uv pip install "sglang[tensorcast]"`
  - or `pip install "sglang[tensorcast]"`

---

## 0) Roles and Terms

- **Caller / Controller / Router**: an external control-plane program that
  chooses prefill and decode instances, issues HTTP requests to SGLang, and
  submits Tensorcast plans.
- **P instance (source / prefill instance)**: the SGLang instance that performs
  prefill and owns the source KV live state.
- **D instance (target / decode instance)**: the SGLang instance that will
  resume decode after KV transfer.
- **Tensorcast daemon / worker**: the node-local Tensorcast runtime that serves
  worker actions such as prefetch and pinning.
- **NodeAgent / InstanceAgent**: the instance-scoped execution host boundary for
  Tensorcast instance steps. The current Tensorcast repository provides a
  standalone `NodeAgent` reference host. In the SGLang v1 integration, this
  boundary is implemented as an in-process instance-agent inside the logical
  SGLang instance rather than as a daemon-internal component.
- **SGLang EngineAdapter**: the SGLang-side integration layer, co-located with
  the in-process instance-agent, that translates Tensorcast instance actions
  into SGLang KV operations.
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
- **`engine_request_id`**: an adapter-local correlation handle passed through
  Tensorcast request-level instance actions. It is not Tensorcast artifact
  identity or artifact-set identity.
- **`ManifestResult`**: the structured result describing the published KV
  snapshot or bundle exported through Tensorcast, including stable artifact-set
  identity and an optional
  `ManifestArtifactSetBridge`.
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

1. The controller chooses a source prefill instance and a target decode
   instance.
2. The source instance performs prefill for a request.
3. The controller asks the source instance, through Tensorcast, to publish the
   full KV state needed to resume decode elsewhere.
4. Optionally, the controller asks the target host daemon to prefetch the
   published artifacts for performance.
5. The controller asks the target instance, through Tensorcast, to hydrate that
   KV state into its engine-local runtime.
6. The controller resumes decode on the target instance.

### 1.4 Out of scope for v1

The following are out of scope for this protocol version:

- Partial-prefix transfer semantics.
- Multi-hop migration semantics.
- External-caller orchestration on every synchronous prefix-hit path.
- A single Tensorcast plan that spans a source instance on one daemon and a
  target instance on another daemon.
- Tensorcast-native load-aware scheduling or instance queue-length signals.
- A public `hydrate(artifact_set_ref=...)` API. v1 uses `engine_request_id` for
  the instance-step surface and `ManifestResult` for worker-side orchestration.
- Requiring direct L1-GPU <-> shared-pool zero-copy put/get for the prefix-share
  path. v1 assumes SGLang may keep its existing L1/L2 hierarchy and integrate
  Tensorcast at the host/L2-facing boundary.

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

engine_request_id = get_engine_request_id_for_transfer(...)

await SglClient(p_meta.http_addr).completion(
    prompt,
    max_tokens=1,
    req_id=p_meta.request_id,
)

# Phase 1: source publish
ctx1 = tc.context(
    request_id=f"kv-publish:{engine_request_id}",
    deadline_ms=15_000,
    idempotency_key=f"kv-publish:{engine_request_id}",
)
plan1 = rt.plan(ctx1)
pub = plan1.on_instance(p_inst).publish(
    engine_request_id=engine_request_id,
    ttl_ms=60_000,
)
res1 = plan1.run()
publish_result = res1.step(pub).artifact_result
manifest = publish_result.manifest

# Phase 2: target warmup + hydrate
ctx2 = tc.context(
    request_id=f"kv-load:{engine_request_id}",
    deadline_ms=15_000,
    idempotency_key=f"kv-load:{engine_request_id}",
)
plan2 = rt.plan(ctx2)
warm = plan2.on_worker(d_worker).prefetch_manifest_result(
    manifest,
    device="cpu",
)
hyd = plan2.on_instance(d_inst).hydrate(
    engine_request_id=engine_request_id,
    depends_on=[warm],
)
res2 = plan2.run()

await SglClient(d_meta.http_addr).completion(
    prompt,
    req_id=p_meta.request_id,
)
```

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

### 4.1 `engine_request_id` semantics for request transfer (MUST)

For v1, `engine_request_id` is an adapter-local lookup handle.

It MUST NOT be treated as:

- Tensorcast artifact identity,
- Tensorcast artifact-set identity,
- Tensorcast workflow identity,
- Tensorcast truth for distributed snapshot currentness.

Stable published request-bundle identity belongs to:

- `ManifestResult.key_set_digest_hex`,
- `ManifestArtifactSetBridge`,
- `ArtifactSetRef`,
- and the artifact identities inside the manifest.

### 4.2 Prefix-share identity semantics (MUST)

Stable prefix-share identity belongs to:

- page hashes,
- prefix hash chains,
- prefix bundle identity,
- and the artifact identities inside the corresponding bundle manifest.

### 4.3 Relationship to SGLang request identity (SHOULD/MAY)

The v1 controller SHOULD obtain `engine_request_id` from the SGLang-side
integration layer rather than synthesizing a random string at the Tensorcast
boundary.

For the initial integration, the SGLang implementation MAY choose:

- `engine_request_id == sglang_request_id`

provided that the implementation can guarantee the required uniqueness,
lifetime, and lookup semantics for publish and hydrate.

However, the protocol does not require these two identifiers to be equal.
Future versions MAY introduce a distinct transfer or snapshot handle while
keeping the public controller shape unchanged.

### 4.4 Single-snapshot simplification for v1 (SHOULD)

To keep v1 semantics unambiguous, the controller SHOULD treat one
`engine_request_id` as having at most one active publish -> hydrate handoff at a
time.

If the implementation wants to support repeated publish operations for the same
`engine_request_id`, the SGLang adapter MUST define and document how later
hydrates disambiguate which published snapshot is authoritative.

### 4.5 TP-group scope of `engine_request_id` (MUST)

For request-level transfer on a logical instance with `TP > 1`,
`engine_request_id` MUST denote the logical request across the whole instance /
TP group.

It MUST NOT be interpreted at the public caller surface as:

- a per-rank shard identifier,
- a per-rank publication handle,
- or a public signal that the caller should issue one instance-step call per
  TP rank.

The same `engine_request_id` SHOULD be passed to source-side `publish`,
target-side `hydrate`, and cleanup actions for the same handoff.

The implementation MAY keep additional rank-local handles or metadata
internally, but such details are adapter-owned and MUST NOT leak into the
public controller contract.

---

## 5) Source-side Publish Contract

### 5.1 Publish meaning (MUST)

`plan.on_instance(P).publish(engine_request_id=E, ...)` MUST mean:

- locate the source instance's current KV live state for `E`,
- construct an immutable published snapshot containing all KV data required to
  resume decode on another instance,
- publish that snapshot into Tensorcast as artifacts,
- return a `PublishResult` whose `manifest` describes that published snapshot.

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

### 5.4 Completeness (MUST)

Publish MUST fail if the adapter cannot produce a complete snapshot sufficient
for target-side decode continuation.

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

### 5.6 Returned manifest contract (MUST)

On success, `PublishResult.manifest` MUST carry stable snapshot identity that the
controller can use for worker-side orchestration.

For v1, the source adapter SHOULD include a valid
`ManifestArtifactSetBridge` so the controller can call:

```python
plan.on_worker(worker).prefetch_manifest_result(manifest, device=...)
```

without having to reconstruct the underlying artifact set itself.

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
    engine_request_id=engine_request_id,
    ttl_ms=60_000,
)
```

The adapter MAY choose to retain source-local live state independently of
Tensorcast artifact TTL.

---

## 6) Optional Target-side Worker Warmup Contract

### 6.1 Warmup is an optimization, not a correctness requirement

The controller MAY insert target-side worker steps before hydrate:

```python
plan.on_worker(d_worker).prefetch_manifest_result(manifest, device=...)
```

This warmup is optional and exists to reduce hydrate latency for high-cardinality
KV byte artifacts.

Hydrate correctness MUST NOT depend on the caller having performed this worker
warmup.

### 6.2 Warmup target (MUST)

If warmup is used, it MUST target the Tensorcast worker/daemon associated with
the target decode instance `D`.

### 6.3 Readiness floor (MUST)

The controller MUST treat `prefetch_set` / `prefetch_manifest_result` as
guaranteeing only the Tensorcast readiness floor `local_replica_ready`.

It MUST NOT assume stronger placement guarantees than that.

---

## 7) Target-side Hydrate Contract

### 7.1 Hydrate meaning (MUST)

`plan.on_instance(D).hydrate(engine_request_id=E)` MUST mean:

- resolve the authoritative published KV snapshot associated with `E`,
- materialize the required artifacts if they are not already locally ready,
- reconstruct decode-usable KV live state inside the target SGLang instance,
- return only when that local engine state is ready for decode continuation or
  the operation has failed.

### 7.2 TP-group scope and coordinator behavior (MUST)

If the target logical instance `D` uses `TP > 1`, `hydrate()` MUST be defined
over the complete logical instance, not over a single TP-rank shard.

The SGLang-side coordinator for `D` MUST:

- receive the single Tensorcast instance-step call for the logical
  `instance_id`,
- resolve the authoritative request bundle for `engine_request_id`,
- determine the per-rank shard assignment needed for decode continuation,
- fan the hydrate operation out to all required ranks,
- wait for all required ranks to either succeed or fail,
- and return one external hydrate result for the whole logical instance.

Hydrate MUST fail if any required rank fails, times out, or cannot reconstruct
its portion of the runnable decode state.

### 7.3 Resolution source (MUST)

Because the public Tensorcast `hydrate` instance step currently accepts only
`engine_request_id`, the SGLang integration MUST provide an internal resolution
mechanism from `E` to the published snapshot to hydrate.

This resolution mechanism MAY be implemented via:

- adapter-local metadata,
- a transfer registry,
- manifest-backed state remembered by the source/target integration,
- or another equivalent mechanism.

But the public controller contract remains:

- publish returns manifest for controller-side worker orchestration,
- hydrate is invoked with `engine_request_id`.

### 7.4 Success criteria (MUST)

Hydrate MUST fail if the target instance cannot reconstruct runnable decode state
for the request.

In particular:

- missing required artifacts,
- layout mismatches,
- decode-engine insertion failures,
- or incomplete KV reconstruction

MUST surface as hydrate failure.

For `TP > 1`, this success criterion applies to the whole logical instance:
one required rank failing means the group hydrate fails.

### 7.5 Partial-install handling (SHOULD)

If some target ranks have already materialized local state but the group hydrate
later fails, the integration SHOULD treat the target local state as tainted or
incomplete.

The coordinator SHOULD attempt best-effort group cleanup through the same
internal fan-out path used for request-level `evict_local`.

Such cleanup does not change the externally visible result: the Tensorcast
`hydrate()` call still fails.

### 7.6 Decode continuation boundary (MUST)

After successful hydrate, the target instance MUST be able to accept the decode
continuation request for the same logical request without requiring the caller to
re-run prefill.

---

## 8) Cleanup Contract

### 8.1 Source cleanup (MAY/SHOULD)

After target-side hydrate succeeds and the controller has committed to decode on
the target instance, the controller MAY ask the source instance to free local KV
state:

```python
plan.on_instance(p_inst).evict_local(engine_request_id=engine_request_id)
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
- `kv-load:{engine_request_id}`
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
   - Controller obtains `PublishResult.manifest`.
   - Controller optionally warms the target host daemon using
     `prefetch_manifest_result(manifest, ...)`.
   - Controller runs a target logical-instance
     `hydrate(engine_request_id=E)` plan.
   - Controller issues decode continuation to `D`.
   - Controller optionally evicts source local KV state.

This is the canonical v1 protocol baseline that the SGLang Tensorcast KV
integration must implement.
