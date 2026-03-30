# Tensorcast KV Integration Design (SGLang)

This document describes the **recommended integration design** for SGLang KV
cache with Tensorcast.

It complements the protocol document:
- `sglang/docs/tensorcast/tensorcast_kv_protocol.md`

Protocol vs design split:
- The protocol document freezes the external contract and target semantics.
- This design document explains how SGLang and Tensorcast should be connected
  internally to realize those semantics.

Status:
- This document now mixes two states:
  - the currently implemented prefix-share substrate in this repo,
  - and the still-target request-transfer design that is not implemented yet.
- Implemented today in the SGLang repo:
  - `python/sglang/srt/mem_cache/storage/tensorcast_store/`
  - byte-artifact-native `batch_exists(...)`
  - byte-artifact-native `batch_set_v1(...)`
  - byte-artifact-native `batch_get_v1(...)`
  - `benchmark/tensorcast_benchmark/kv/share_local` automation for
    `tensorcast-daemon-mode=share|separate`, explicit benchmark `rid`, and
    source-publication-drain tracking
- Current Tensorcast core already exposes the legacy request-transfer surface
  `publish(engine_request_id=...)`, `hydrate(engine_request_id=...)`,
  `evict_local(...)`, and generic `ManifestResult`.
- Not implemented yet, and still required for request transfer:
  `PublishResult.publish_manifest`, opaque `EngineOwnedManifest`, and
  `hydrate(publish_manifest=...)`. Those surfaces are not implemented in
  Tensorcast core today and must be added as part of the KV integration work.

---

## 1) Executive Summary

The recommended design is:

- **one shared Tensorcast-backed KV data plane**
- with **two upper interfaces**

The two upper interfaces are:

1. **Prefix share data plane**
   - used on the hot path when SGLang serves requests and finds storage-backed
     prefix hits,
   - integrated into `HiRadixCache` + `HiCacheController`,
   - shaped like a Mooncake-style storage backend,
   - not driven by an external Tensorcast caller program per request.
2. **Request-level transfer control plane**
   - used for PD-disaggregated handoff from prefill instance to decode instance,
   - driven by an external caller/controller/router,
   - expressed through Tensorcast instance steps such as `publish`,
     `hydrate`, and `evict_local`,
   - optionally augmented with worker-side `prefetch_manifest_result(...)`.

The key design rule is:

- these two upper interfaces MUST share the same underlying distributed KV pool,
  the same page identity, and the same bundle/manifest semantics.

Tensorcast must therefore not be reduced to only:

- a programmable controller surface, or
- a Mooncake-like storage adapter.

It must serve both roles on top of one shared KV substrate.

---

## 2) Why Two Upper Interfaces Are Necessary

### 2.1 Prefix share and request transfer are different workloads

Although both operate on KV cache, they have very different execution patterns:

- **Prefix share**
  - high-frequency,
  - latency-sensitive,
  - executed inside the serving engine hot path,
  - naturally batched at page granularity,
  - tightly coupled to host-pool allocation, prefetch throttling, and radix-tree
    insertion.
- **Request-level transfer**
  - lower-frequency,
  - control-plane orchestrated,
  - spans multiple serving instances,
  - needs explicit publish / hydrate lifecycle,
  - benefits from external caller logic and programmability.

Trying to force both onto one interface leads to a bad outcome in both
directions:

- using external `Plan` orchestration for every prefix share would be too heavy,
- using only a storage backend API would hide Tensorcast's instance-step
  programmability and make PD transfer awkward.

### 2.2 Why the SGLang hot path should stay internal

Current SGLang prefix-share behavior already lives in:

- `HiRadixCache` for prefix/radix/node ownership,
- `HiCacheController` for host/device/storage movement,
- `HiCacheStorage` backends for page-store operations.

This path includes engine-local policies such as:

- prefetch thresholds,
- prefetch cancellation,
- host memory quota,
- TP synchronization,
- partial progress and insertion into the radix tree.

This is the natural place for Tensorcast-backed prefix share.

### 2.3 Why request transfer still wants programmability

For PD-disaggregated inference, an external caller must be able to:

- choose source and target instances,
- sequence prefill and decode,
- publish request-scoped KV state,
- optionally prewarm the target host daemon,
- hydrate the target instance,
- decide cleanup and retry behavior.

This is precisely the kind of control-plane orchestration that Tensorcast
programmability is good at.

---

## 3) Architecture Overview

### 3.1 Layering

The integration should be organized as:

1. **Shared KV substrate**
   - Tensorcast-backed distributed storage pool for KV pages and bundles.
2. **Prefix share interface**
   - internal SGLang data-plane integration for page-level share.
3. **Request transfer interface**
   - external Tensorcast programmability integration for request-level transfer.

### 3.2 Shared substrate responsibilities

The shared substrate owns:

- page-level identity,
- publication and retrieval of KV page artifacts,
- prefix-bundle metadata,
- generic artifact-manifest data for request-level transfer,
- consistency between storage-backed prefix share and request-level transfer.

The substrate MUST be the single source of truth for distributed KV objects.

Engine-owned request-resume metadata is intentionally not modeled as a separate
Tensorcast dataplane artifact. It is carried in the request-transfer control
plane as `EngineOwnedManifest`.

### 3.3 Upper interface responsibilities

The prefix share interface owns:

- hot-path existence queries,
- page-level get/set,
- host-pool materialization,
- insertion back into SGLang's in-memory radix structures.

The request transfer interface owns:

- instance-scoped publish/hydrate semantics,
- external control-plane sequencing,
- optional worker warmup,
- explicit request handoff lifecycle.

### 3.4 Component graph

```mermaid
flowchart TD
    subgraph Caller["External controller / router"]
        C1["Choose P / D instances"]
        C2["Run Tensorcast plans"]
    end

    subgraph Src["Source logical SGLang instance P"]
        PIA["In-process InstanceAgent + EngineAdapter"]
        P0["Coordinator (rank-0 ingress)"]
        PR["TP ranks"]
        PH["HiRadixCache + HiCacheController"]
    end

    subgraph Dst["Target logical SGLang instance D"]
        DIA["In-process InstanceAgent + EngineAdapter"]
        D0["Coordinator (rank-0 ingress)"]
        DR["TP ranks"]
        DH["HiRadixCache + HiCacheController"]
    end

    subgraph TC["Tensorcast"]
        WD["Host daemon worker"]
        KV["Shared KV substrate<br/>page byte artifacts + bundle metadata"]
    end

    C1 --> C2
    C2 -->|"source-side routed instance steps"| PIA
    C2 -->|"target-side routed instance steps"| DIA
    C2 -->|"prefetch_manifest_result"| WD
    PIA -->|"local coordinator call"| P0
    DIA -->|"local coordinator call"| D0
    P0 --> PR
    D0 --> DR
    PR --> PH
    DR --> DH
    PH <--> KV
    DH <--> KV
    WD <--> KV
```

For SGLang v1, the instance-scoped execution host is intentionally modeled
inside the logical SGLang instance boundary:

- the Tensorcast `NodeAgent` semantics are realized as an in-process
  instance-agent,
- the `EngineAdapter` is SGLang-side integration code in that same boundary,
- and the Tensorcast daemon remains the worker/data-plane host rather than the
  owner of instance-step execution.

The standalone Tensorcast `NodeAgent` in the Tensorcast repository should be
treated as a reference implementation of the execution-host contract, not as
the required deployment topology for SGLang.

### 3.5 End-to-end split between hot path and programmable path

```mermaid
flowchart LR
    A["Prefix-share hot path"] --> A1["Per-rank batch_exists / batch_get_v1 / batch_set_v1"]
    A1 --> A2["HiRadixCache + HiCacheController keep TP-consistent view"]
    A2 --> A3["No external caller and no coordinator hop per page"]

    B["Request-level transfer"] --> B1["Caller runs publish / warmup / hydrate plans"]
    B1 --> B2["Coordinator fans out to required ranks"]
    B2 --> B3["Shared substrate reused for the same page artifacts"]
```

### 3.6 Coordinator-owned directory registration and liveness

For request-level transfer, the rank-0 coordinator SHOULD own the Tensorcast
directory-facing lifecycle of the logical SGLang instance.

Why this ownership is recommended:

- the coordinator is the public control-plane ingress for the logical instance,
- the coordinator also owns the in-process instance-agent execution endpoint,
- the coordinator and the logical SGLang serving instance are intended to share
  one lifecycle boundary,
- and if rank 0 dies, the logical instance should stop being routable for
  Tensorcast instance-step execution.

Recommended route-registration schema:

```yaml
tensorcast_directory:
  instance_id: sgl-inst-17
  daemon_id: daemon-a
  engine: sglang
  execution_host_kind: sglang_inproc_instance_agent.v1
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

Recommended ownership rules:

- only rank 0 registers and heartbeats the logical Tensorcast `instance_id`,
- other TP ranks remain internal workers behind that one logical instance,
- `execution_endpoint` points at the coordinator-hosted in-process
  instance-agent ingress,
- and `serving_http_endpoint` remains SGLang-owned discovery metadata rather
  than a required Tensorcast directory field.

Recommended lifecycle state machine:

```mermaid
stateDiagram-v2
    [*] --> Booting
    Booting --> Registering: coordinator bound local services
    Registering --> Ready: Tensorcast route registered and heartbeats active
    Ready --> Draining: stop accepting new transfer work
    Draining --> Stopped: heartbeat stops and route expires
    Ready --> Stopped: coordinator crash / rank-0 failure
    Stopped --> Registering: coordinator restart with new epoch
```

Recommended registration flow:

```mermaid
flowchart TD
    A["Rank-0 coordinator starts"] --> B["Bind HTTP serving endpoint"]
    B --> C["Bind in-process InstanceAgent endpoint"]
    C --> D["Register instance route into Tensorcast directory"]
    D --> E["Send periodic instance heartbeats"]
    E --> F["Caller resolves instance_id -> execution_endpoint"]
    E --> G["If rank-0 exits, heartbeats stop"]
    G --> H["Directory marks instance inactive / unroutable"]
```

---

## 4) Shared KV Substrate

### 4.1 Shared identity model

The shared substrate SHOULD use:

- **page-level identity** as the stable storage identity,
- **bundle-level identity** as the orchestration identity.

Recommended identity layers:

- **KV page artifact**
  - unit: one page of KV data,
  - stable identity based on SGLang's page hash chain / page hash value.
- **Prefix bundle**
  - unit: an ordered set of KV pages representing a reusable prefix.
- **Request bundle**
  - unit: an ordered set of KV pages sufficient to resume one request on another
    instance.

This lets prefix share and request transfer reuse the same published page
artifacts while exposing different upper-level semantics.

#### 4.1.1 One SGLang page shard maps to one high-cardinality Tensorcast byte artifact

The concrete storage unit inside Tensorcast SHOULD be:

- one rank-local SGLang KV page shard

represented as:

- one Tensorcast byte artifact.

This is intentionally high-cardinality:

- one request bundle may contain many pages,
- one prefix bundle may contain many pages,
- and each page is stored and deduplicated independently.

For `TP > 1`, the unit is still one page shard, not a logical whole-request
tensor and not an all-rank aggregate object.

Therefore:

- the shared substrate stores many page-sized byte artifacts,
- prefix bundles and request bundles are metadata layers over those artifacts,
- and deduplication happens at page-artifact granularity.

#### 4.1.2 Recommended byte-artifact identity contract

Each page artifact SHOULD use:

- a valid Tensorcast byte-artifact `artifact_id`,
- and a `layout_id` that identifies the page serialization contract.

The exact naming scheme is integration-owned, but the artifact identity input
SHOULD be derived from:

- page hash or equivalent content-derived identity,
- model / KV layout family,
- page size,
- dtype / encoding contract,
- and rank-local shard qualifiers such as TP / PP ownership when required.

The `layout_id` SHOULD version the byte-level page format so that:

- incompatible page encodings never silently alias,
- request bundles can reject incompatible pages early,
- and future serialization changes can coexist safely.

### 4.2 Byte-artifact boundary and data ownership

For v1, the ownership boundary is the existing SGLang host/L2 page boundary.

That means:

- SGLang owns the `L1(device) -> L2(host)` movement,
- Tensorcast begins at a frozen host-page view,
- Tensorcast does not own live mutable device KV pages in v1.

The publication boundary for one page shard is:

1. the page is resident in an SGLang host buffer,
2. the integration freezes or snapshots that host-page contents for one
   publication attempt,
3. the shared runtime wraps those bytes as a Tensorcast byte artifact candidate,
4. the shared runtime publishes that candidate into the distributed pool.

The retrieval boundary is the reverse:

1. the shared runtime resolves a page byte artifact from Tensorcast,
2. the bytes are materialized into an SGLang host page buffer,
3. SGLang later decides whether and when to load the page back into device KV
   memory.

#### 4.2.1 Page-to-artifact conversion contract

The SGLang-side shared runtime SHOULD conceptually perform:

1. `host page buffer -> bytes payload`
2. `payload + artifact_id + layout_id -> OpenByteArtifact`
3. `OpenByteArtifact -> SealedByteArtifact`
4. `SealedByteArtifact -> put-if-absent / retain into Tensorcast`

The Tensorcast artifact API already exposes this shape explicitly:

- `OpenByteArtifact`
- `SealedByteArtifact`
- `seal_byte_artifact(...)`

So the design target is not a vague "store some page bytes somewhere". It is:

- convert one frozen SGLang host page shard into one sealed Tensorcast byte
  artifact candidate with explicit invariants,
- then publish or adopt it through the shared runtime.

This is a semantic contract, not a requirement that the hottest path must
literally instantiate Python objects page-by-page.

The implementation MAY use lower-level Tensorcast region-based fast paths such
as batch region `put-if-absent` and region `get-into` operations, provided the
result still obeys the same contract:

- one page shard maps to one byte-artifact identity,
- publication is content-checked and deduplicated at that artifact identity,
- bundle metadata refers to those page artifact identities rather than to opaque
  engine-private buffers.

For the current Tensorcast core, the high-throughput batch region path is
VRAM-region-backed; host-region support is a medium-term extension rather than a
current assumption.

#### 4.2.2 Data-path graph for one page shard

```mermaid
flowchart LR
    A["SGLang device page shard<br/>(mutable, engine-owned)"]
    B["SGLang host page shard<br/>(L2, publish boundary)"]
    C["Frozen page snapshot<br/>bytes payload"]
    D["OpenByteArtifact<br/>(artifact_id, layout_id, payload)"]
    E["SealedByteArtifact<br/>(payload digest + invariants)"]
    F["Tensorcast shared pool<br/>byte artifact"]
    G["Prefix bundle metadata / PublishManifest"]

    A -->|"backup / offload"| B
    B -->|"freeze for publication attempt"| C
    C --> D
    D -->|"seal"| E
    E -->|"put-if-absent / adopt / retain"| F
    F --> G
```

#### 4.2.3 Reverse data path for retrieval

```mermaid
flowchart RL
    A["Tensorcast shared pool<br/>byte artifact"]
    B["Resolved page payload<br/>bytes"]
    C["SGLang host page shard<br/>(L2)"]
    D["SGLang device page shard<br/>(L1)"]

    A -->|"batch_get / prefetch / hydrate materialization"| B
    B -->|"fill host page buffer"| C
    C -->|"load_back when SGLang decides"| D
```

#### 4.2.4 v1 short-term implementation: byte-artifact-native substrate over GPU staging

This is the current implementation strategy in the SGLang repo:

- `batch_exists(...)` uses Tensorcast `BatchExists(...)`,
- `batch_set_v1(...)` uses `BatchPutIfAbsentFromRegion(...)`,
- `batch_get_v1(...)` uses `BatchGetIntoRegion(...)`,
- reusable per-rank VRAM staging regions are managed inside
  `tensorcast_store/client.py`,
- and the steady-state prefix-share path no longer goes through generic
  `tc.Store.put(..., key=...)` / tensor fetch APIs.

For the immediate implementation, the shared substrate SHOULD be
byte-artifact-native end-to-end:

- page existence is checked by byte-artifact identity,
- page publication is performed by byte-artifact put-if-absent,
- page retrieval is performed by byte-artifact get-into,
- and prefix/request metadata refers to page artifact identities rather than to
  Tensor-valued `key=` artifacts.

However, the current Tensorcast batch region APIs impose an important
implementation constraint:

- `BatchPutIfAbsentFromRegion(...)` and `BatchGetIntoRegion(...)` are the
  intended high-throughput batch byte-artifact data path,
- but today they accept only VRAM-backed registered regions,
- not host/L2 buffers directly.

Therefore the first production-oriented SGLang integration SHOULD use:

- **host/L2 pages as the semantic ownership boundary**, and
- **reusable GPU staging buffers as the transport scratch boundary**.

This means:

- SGLang still decides when a page becomes backup-eligible and when a host page
  is admissible,
- Tensorcast still stores one page shard as one byte artifact,
- but the hottest batch ingress/egress path temporarily uses:
  - `host page -> H2D into coalesced GPU staging region -> Tensorcast batch RPC`
  - `Tensorcast batch RPC -> GPU staging region -> D2H into host page`

The GPU staging buffer is therefore a transport adapter, not a semantic change
to the substrate boundary.

The short-term implementation SHOULD NOT treat the following as the steady-state
hot path for prefix share:

- generic `tc.Store.put(..., key=...)` / `artifact(key=...).tensor(...)`,
- `HomeBatchPutIfAbsent(...)`,
- `HomeBatchGet(...)`.

Those paths remain useful for fallback, bring-up, and debugging, but they are
not the intended high-cardinality page IO path because they operate through
payload bodies rather than through client-owned region-backed batch transfer.

#### 4.2.5 v1 short-term write path

For one publication batch, the recommended write-side flow is:

1. SGLang offloads or backs up mutable L1 device KV into stable L2 host pages.
2. The Tensorcast-backed backend selects one ordered batch of rank-local page
   shards to publish.
3. For each page shard, the backend derives:
   - byte-artifact `artifact_id`,
   - `layout_id`,
   - byte length,
   - payload digest / invariant contract.
4. The backend packs those pages into one coalesced GPU staging buffer with one
   contiguous slice per page shard.
5. The backend reuses or recreates a registered VRAM region for that staging
   buffer.
6. The backend issues one `BatchPutIfAbsentFromRegion(...)` call with:
   - one item per page shard,
   - one coalesced `source_layout`,
   - explicit per-item invariants.
7. Per-item outcomes update the SGLang-side batch publication result:
   - `ready` for newly published or already-existing pages,
   - `failed` for publication errors.

In the current prefix-share implementation, this is realized as the
`batch_set_v1(...)` success mask plus duplicate/failure accounting rather than
as the full Phase-3 request-bundle publication registry.

Duplicate publication SHOULD be handled at the byte-artifact level:

- the backend does not need a per-page generic `exists()+put()` sequence,
- the batch put-if-absent response itself is the authority on
  absent/already-present/conflict outcomes.

```mermaid
flowchart LR
    A["SGLang host pages<br/>(rank-local L2 shards)"]
    B["Derive artifact_id / layout_id / invariant"]
    C["Pack coalesced GPU staging buffer"]
    D["Register or reuse VRAM region"]
    E["BatchPutIfAbsentFromRegion<br/>one item per page shard"]
    F["Tensorcast byte-artifact pool"]
    G["SGLang batch publication state"]

    A --> B
    B --> C
    C -->|"H2D"| D
    D --> E
    E -->|"put-if-absent outcomes"| F
    E --> G
```

#### 4.2.6 v1 short-term read path

For one retrieval batch, the recommended read-side flow is:

1. SGLang computes the ordered candidate page identities for the request or
   prefix continuation.
2. The backend uses `BatchExists(...)` to determine the consecutive hit span.
3. The backend allocates or reuses a coalesced GPU staging buffer large enough
   for the hit span.
4. The backend issues `BatchGetIntoRegion(...)` so Tensorcast materializes the
   hit pages directly into the staging region.
5. The backend copies each resolved page slice from GPU staging back into the
   target L2 host pages.
6. SGLang inserts those host pages into its radix-tree / host-cache structures
   and later decides whether to load them back to L1 device memory.

This preserves the v1 ownership rule:

- Tensorcast does not mutate live L1 KV state directly for prefix share,
- the retrieval result becomes visible to SGLang first as host/L2 pages.

```mermaid
flowchart LR
    A["Ordered page artifact_ids"]
    B["BatchExists<br/>find consecutive hit span"]
    C["Allocate / reuse GPU staging buffer"]
    D["BatchGetIntoRegion<br/>materialize hit pages"]
    E["D2H copy into host/L2 page buffers"]
    F["Insert into HiRadixCache / host cache"]
    G["Optional later load_back host->device"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
```

#### 4.2.7 Medium-term plan: host-native region registration and allocator

The short-term GPU staging path is a practical consequence of the current
Tensorcast region API, not the desired final substrate shape.

The medium-term direction SHOULD be:

- Tensorcast adds a generalized region-registration surface such as
  `RegisterRegion(memory_kind=VRAM|HOST_SHARED)` or equivalent,
- Tensorcast batch byte-artifact region RPCs accept host-shared regions in
  addition to VRAM regions,
- SGLang adds a Tensorcast-aware host allocator for `HostKVCache`, analogous in
  spirit to the existing Mooncake host allocator integration,
- L2 pages are allocated directly from Tensorcast-shareable host memory rather
  than from ordinary pinned host memory,
- the byte-artifact identity, layout, invariant, and metadata layers remain
  unchanged.

With that medium-term shape, the data path becomes:

- write side:
  - `host/L2 page -> host shared region -> BatchPutIfAbsentFromRegion`
- read side:
  - `BatchGetIntoRegion -> host shared region / host page`

So the extra `H2D` and `D2H` transport copies disappear without moving the
semantic publication boundary away from SGLang's existing host/L2 layer.

This medium-term plan is explicitly future work:

- it requires Tensorcast core changes,
- it requires an SGLang allocator integration change,
- and it is not a blocker for the immediate byte-artifact-native benchmark
  bring-up.

### 4.3 Relationship to current SGLang hashing

The shared substrate SHOULD preserve SGLang's existing page-hash semantics and
prefix-chain hints:

- page hash values derived from token sequences,
- optional `prefix_keys` chain passed to storage backends,
- existing host-page indexing and page-aligned IO behavior.

This avoids inventing a second identity system for the same KV content.

### 4.4 Bundle metadata and transfer-handle metadata

The integration SHOULD support two distinct metadata families:

- **Prefix bundle metadata**
  - maps a prefix identity to an ordered set of page artifacts.
- **Request-transfer metadata**
  - consists of:
    - a generic Tensorcast artifact manifest over page artifacts, and
    - an opaque engine-owned resume manifest needed for decode continuation.

Prefix-share bundles and request-level transfer snapshots MAY share the same
page artifacts.

The engine-owned request-resume payload MUST NOT be represented as a separate
Tensorcast byte artifact in the dataplane.

#### 4.4.1 Prefix bundle metadata record schema

The integration SHOULD define one canonical prefix bundle metadata schema so
prefix prewarm, inspection, and repair logic do not invent incompatible record
shapes.

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

Recommended validation rules:

- `prefix_bundle_id` is deterministic for the same logical prefix identity and
  layout/topology contract,
- `ordered_pages` is strictly ordered by `logical_page_index` within each rank
  shard,
- all referenced `artifact_id` values point to the same page byte artifacts
  used by the shared substrate,
- and `state=ready` means the referenced page closure is readable and
  layout/topology compatible for that bundle identity.

#### 4.4.2 Prefix bundle invalidation rules

A prefix bundle SHOULD be treated as stale if any of the following becomes
true:

- a referenced page artifact is missing or unreadable,
- a referenced page's layout is incompatible with `identity.kv_layout_id`,
- the observed TP/PP topology is incompatible with the bundle identity,
- the referenced page set no longer matches `constituent_digest`,
- or the bundle has passed `expires_at_ms`.

#### 4.4.3 PublishManifest and EngineOwnedManifest schema

For request-level transfer, the exported metadata shape SHOULD be:

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

Recommended interpretation:

- `artifact_manifest` is the generic Tensorcast-visible description of the page
  artifact closure for the published snapshot.
- `engine_owned_manifest` is the engine-owned resume payload. Tensorcast core
  should transport it but not interpret its payload.
- the entire `publish_manifest` object is the external transfer handle; there
  is no second controller-facing string `transfer_handle`.
- `payload` is control-plane data, not a Tensorcast dataplane object.

Recommended rules:

- `engine_owned_manifest.artifact_manifest_digest` MUST bind the opaque resume
  payload to exactly one generic artifact manifest.
- if the engine wants a string `publish_id` / `snapshot_id` for observability
  or debugging, it SHOULD place that identifier inside the opaque
  `engine_owned_manifest` payload or envelope rather than inventing a second
  authoritative transfer handle.
- `PublishResult.publish_manifest` and `hydrate(publish_manifest=...)` are
  target integration surfaces and are not implemented in Tensorcast core today.

### 4.5 Manifest alignment

Tensorcast `ManifestResult` and `ManifestArtifactSetBridge` should be treated as
the generic/exported artifact-manifest representation when leaving the engine
boundary.

That means:

- the EngineAdapter should not invent an incompatible second artifact-manifest
  format,
- request-level `publish()` should return an `artifact_manifest` that points
  back to the same underlying page artifacts used by prefix share,
- and `EngineOwnedManifest` should bind to that same artifact manifest rather
  than duplicating the page-artifact truth independently.

### 4.6 TP>1 shard ownership and rank-qualified storage identity

When `TP > 1`, the shared substrate SHOULD treat per-rank KV shards as distinct
physical objects even when they correspond to the same logical token prefix.

In practice, this means:

- each TP rank continues to own its local device / host page buffers,
- each TP rank continues to own its local radix / host-cache structures,
- physical storage identity MAY include TP/PP rank qualification or an
  equivalent namespace,
- and bundle metadata is responsible for reassembling one logical prefix bundle
  or request bundle from those per-rank shards.

This matches the current SGLang mental model:

- token-prefix identity is shared logically across ranks,
- physical page placement and bytes remain rank-local.

### 4.7 No coordinator on the prefix-share hot path

The shared substrate SHOULD NOT require a central coordinator for ordinary
page-level prefix-share `exists/get/set`.

Instead:

- each rank publishes or fetches its own shard pages,
- SGLang's existing TP synchronization remains authoritative for a common hit
  length, insertion boundary, and completion progress,
- and the coordinator is reserved for higher-level request-bundle operations
  such as programmable `publish` / `hydrate` / `evict_local`.

Coordinator-free hot path does not mean "no synchronization". It means:

- page-level IO stays rank-local,
- while cross-rank consistency continues to use SGLang's existing TP collectives
  and HiCache rules rather than adding a new Tensorcast control hop per page.

### 4.8 Page state models owned by the shared runtime

The shared runtime SHOULD define at least two distinct state models for pages:

- a local placement-oriented state used at the engine / host-buffer boundary,
- a publication-registry state used for deduplication and closure.

These are related but not identical.

#### 4.8.1 Publication-relevant local page placement state

This state compresses the placement information that matters to Tensorcast
integration. It is not intended to mirror every internal SGLang cache detail.

```mermaid
stateDiagram-v2
    [*] --> DeviceResident
    DeviceResident --> HostResident: backup / offload
    HostResident --> FrozenForPublish: snapshot boundary established
    FrozenForPublish --> PublishInFlight: ensure_page_published()
    PublishInFlight --> PoolPublished: put/adopt success
    PublishInFlight --> PublishError: publish failed
    PoolPublished --> HostResident: batch_get / prefetch into host
    HostResident --> DeviceResident: load_back
    HostResident --> [*]: local evict
    PoolPublished --> [*]: artifact expires or becomes unavailable
    PublishError --> FrozenForPublish: retry publish
```

Recommended interpretation:

- `DeviceResident`: page exists only as live engine-side device KV state for the
  purposes of Tensorcast integration.
- `HostResident`: page bytes are present in SGLang host memory and can be used
  for prefix-share read or publication preparation.
- `FrozenForPublish`: the page has an immutable snapshot boundary for one
  publication attempt.
- `PoolPublished`: the shared runtime can treat the page artifact contract as
  satisfied in Tensorcast.

#### 4.8.2 Page publication registry state

This is the per-page coordination state owned by the SGLang-side shared runtime.

```mermaid
stateDiagram-v2
    [*] --> Absent
    Absent --> InFlight: first publisher registers
    InFlight --> Ready: publish or adopt succeeded
    InFlight --> Failed: terminal publish error
    Failed --> InFlight: retry
    Ready --> Ready: retain / touch ttl / reuse
    Ready --> Absent: expire / invalidate
```

Recommended semantics:

- `Absent`: no usable distributed publication is currently known.
- `InFlight`: some rank or bundle-level workflow is actively trying to satisfy
  the page publication contract.
- `Ready`: the page is published with usable identity and retention.
- `Failed`: the most recent publication attempt failed and should not be treated
  as ready.

---

## 5) Upper Interface A: Prefix Share Data Plane

### 5.1 Design choice

Prefix share SHOULD be integrated as an internal Tensorcast-backed HiCache data
plane, not as an external caller-driven programmable workflow.

Operationally, this means Tensorcast should look more like:

- a Mooncake-style storage backend

than like:

- a per-request external controller action.

### 5.2 Recommended integration point

The recommended SGLang integration boundary is:

- `HiRadixCache` + `HiCacheController`

not just:

- `HiCacheStorage` in isolation.

Reason:

- `HiCacheStorage` only exposes page-store verbs,
- while the real prefix-share semantics also depend on radix-tree ownership,
  prefetch thresholds, cancellation, and insertion semantics controlled by
  `HiRadixCache` and `HiCacheController`.

In practice, the first shippable implementation can still expose a
`TensorCastHiCacheStorage`-like surface, but it should be backed by a shared
runtime that understands the broader KV substrate.

### 5.3 Required backend shape

The prefix-share backend SHOULD provide a Mooncake-like internal API:

- `batch_exists(keys, extra_info)`
- `batch_get_v1(keys, host_indices, extra_info)`
- `batch_set_v1(keys, host_indices, extra_info)`

Key properties:

- batch-oriented,
- page-oriented,
- compatible with host-pool zero-copy or near-zero-copy paths,
- able to honor `prefix_keys` as an additional lookup hint.

### 5.4 v1 memory-hierarchy constraint

For v1, the shared KV substrate SHOULD be integrated at the existing
SGLang HiCache host/L2 boundary.

Concretely, the recommended first implementation is:

- SGLang remains responsible for L1(device) <-> L2(host) movement,
- the Tensorcast-backed shared substrate handles L2(host) <-> distributed pool
  publication and retrieval.

This means v1 does not require:

- direct L1-GPU -> shared-pool zero-copy publish for the prefix-share path,
- direct shared-pool -> L1-GPU zero-copy fetch for the prefix-share path,
- or bypassing SGLang's current host-pool-based HiCache flow.

The reason is pragmatic:

- this keeps the integration minimally invasive to the current SGLang HiCache
  architecture,
- preserves the existing `HiCacheController` / host-pool scheduling model,
- and matches the existing Mooncake-like storage backend contract.

Under the current Tensorcast byte-artifact fast-path constraints, the first
implementation MAY still use reusable GPU staging buffers internally for batch
publication and retrieval. That does not change the memory-hierarchy contract:

- SGLang still owns `L1 <-> L2`,
- Tensorcast still owns `L2 <-> distributed pool`,
- GPU staging is only a temporary transport scratch path until host-native
  region registration exists.

Tensorcast GPU fast paths such as CUDA-IPC-backed mapped-target materialization
remain valuable, but they should be treated as later optimizations or
request-transfer-specific accelerations, not as the baseline assumption for the
v1 prefix-share substrate.

### 5.5 Prefix-share read path

The intended flow is:

1. SGLang matches in-memory prefix using `HiRadixCache`.
2. On storage-backed continuation, SGLang computes page hashes and optional
   prefix chain.
3. The Tensorcast-backed backend checks existence for consecutive pages.
4. Matching pages are fetched into host memory.
5. SGLang inserts the returned pages into its host/radix structures.
6. Later, SGLang loads them from host to device as usual.

This path MUST remain engine-owned and local to SGLang's hot path.

```mermaid
flowchart LR
    A["Request arrives on one logical instance"] --> B["Each TP rank checks in-memory radix state"]
    B --> C["Need more prefix pages from substrate?"]
    C -->|No| D["Continue normal execution"]
    C -->|Yes| E["Per-rank compute page hashes + prefix_keys"]
    E --> F["Tensorcast-backed backend batch_exists(...)"]
    F --> G["SGLang TP sync chooses common hit span"]
    G --> H["Tensorcast-backed backend batch_get_v1(...) into host pages"]
    H --> I["Each rank inserts fetched host pages into radix / host cache"]
    I --> J["SGLang later load_back host->device if needed"]
    J --> D
```

### 5.6 Prefix-share write path

The intended flow is:

1. SGLang backs up KV pages from device to host as it already does.
2. On write-through or write-back-to-storage, the Tensorcast-backed backend
   publishes those pages into the shared KV substrate.
3. The storage publication records page identity and, when available, prefix
   chain hints.

```mermaid
flowchart LR
    A["Device KV page shard becomes backup-eligible"] --> B["SGLang backs it up to host page buffer"]
    B --> C["Shared runtime derives page artifact identity"]
    C --> D["Page publication registry: absent / inflight / ready"]
    D --> E["Publish or adopt one byte artifact per page shard"]
    E --> F["Update prefix-bundle metadata when applicable"]
```

### 5.7 Why this should not use external plans

Using an external caller and `Plan` for synchronous prefix share would be a bad
fit because:

- prefix share is too frequent,
- the hot path needs internal partial-progress handling,
- SGLang already owns scheduling and memory admission,
- `Plan` is a control-plane abstraction, not a per-page low-latency data-plane
  primitive.

### 5.8 Optional programmability for prefix bundles

Programmability can still help for prefix share, but only at a coarse
granularity such as:

- prewarming a known popular prefix bundle to selected nodes,
- debugging or inspecting prefix-bundle availability,
- background repair or rollout.

This is optional and is not the main synchronous prefix-hit path.

---

## 6) Upper Interface B: Request-level Transfer Control Plane

### 6.1 Design choice

Request-level transfer SHOULD use Tensorcast programmability plus an in-process
SGLang instance-agent / EngineAdapter integration.

This is the right place for:

- `publish(engine_request_id=...)`
- preferred `hydrate(publish_manifest=...)`
- compatibility `hydrate(engine_request_id=...)`
- `evict_local(engine_request_id=...)`
- optional `prefetch_manifest_result(...)`

For the current Tensorcast caller surface, request-bundle retention intent is
carried by the `ttl_ms` argument on `publish(...)`, not by `CallContext`.

The explicit-handle rule is:

- `engine_request_id` identifies source-side live request state,
- successful publish returns `PublishManifest`,
- worker warmup uses `PublishManifest.artifact_manifest`,
- target hydrate SHOULD consume that same `PublishManifest`,
- `evict_local(engine_request_id=...)` continues to operate on live local
  request state rather than on immutable transfer handles.

```mermaid
flowchart LR
    A["publish(engine_request_id, ttl_ms)"] --> B["PublishManifest"]
    B --> C["artifact_manifest<br/>for prefetch_manifest_result(...)"]
    B --> D["engine_owned_manifest<br/>opaque resume payload"]
    B --> E["hydrate(publish_manifest=...)"]
    F["evict_local(engine_request_id)"] --> G["local live-request cleanup"]
```

Compatibility note:

- current Tensorcast core exposes only the legacy
  `hydrate(engine_request_id=...)` surface,
- the SGLang KV integration should add `PublishResult.publish_manifest` and
  `hydrate(publish_manifest=...)`,
- and any legacy `hydrate(engine_request_id=...)` convenience should be
  implemented at the controller/helper layer rather than by making the target
  instance resolve transfer snapshots by request id.

### 6.2 Logical instance mapping for TP>1

For request-level transfer, a Tensorcast `instance_id` SHOULD represent one
logical SGLang serving instance / TP group, not one TP-rank process.

Therefore:

- the external caller sees one `instance_id` for one logical instance,
- the external caller issues one `publish` / `hydrate` / `evict_local` call per
  logical instance,
- per-rank fan-out and aggregation stay inside the SGLang integration.

This mapping keeps the controller contract stable even when one SGLang serving
instance internally contains multiple TP ranks.

### 6.3 Coordinator role and external semantics

The request-transfer path SHOULD introduce one SGLang-side coordinator per
logical instance.

The recommended coordinator is the rank-0 control-plane ingress for that TP
group.

Its responsibilities are:

- accept one group-scoped Tensorcast instance-step call for the logical
  `instance_id`,
- validate request identity and layout compatibility,
- fan the operation out to all required ranks,
- gather per-rank results,
- build one group-level `PublishResult`, `HydrateResult`, or `BatchResult`,
- define one success/fail outcome for the whole logical instance.

For standard MHA layouts, the required-rank set is usually all TP ranks. For
layouts such as MLA, the integration MAY optimize which ranks physically publish
pages, but the public programmable semantics remain group-scoped.

### 6.4 Coordinator-to-rank communication path

The preferred implementation is to reuse SGLang's native control-plane path
from the in-process instance-agent boundary rather than letting the Tensorcast
daemon side or a separate helper independently contact each TP-rank process.

Concretely, the recommended flow is:

1. The in-process instance-agent / EngineAdapter receives `publish`,
   `manifest`, `hydrate`, or `evict_local` for one logical Tensorcast
   `instance_id`.
2. The adapter calls a local SGLang coordinator endpoint for that logical
   instance.
3. The coordinator uses SGLang's existing rank-0 control ingress and internal
   broadcast path to fan the request out to other ranks.
4. Each required rank performs its local shard operation.
5. The coordinator gathers the per-rank results and returns one group-level
   result back through the adapter.

The integration SHOULD avoid ad-hoc direct networking from the instance-agent
boundary to every rank process. The fan-out contract should stay inside
SGLang's native coordinator path.

For quick experiments, a generic collective RPC path is acceptable. For the
real integration, SGLang SHOULD define typed internal request/output objects for
KV publish / manifest / hydrate / evict so that structured results, not just
boolean success, can be returned.

```mermaid
flowchart TD
    A["In-process InstanceAgent / EngineAdapter<br/>inside logical SGLang instance"] --> B["Local SGLang coordinator endpoint"]
    B --> C["Rank-0 control ingress"]
    C --> D["Broadcast typed internal request to required ranks"]
    D --> E["Each rank executes local shard op"]
    E --> F["Coordinator gathers per-rank outputs"]
    F --> G["Return one group-level result to EngineAdapter"]
```

Recommended typed internal schema:

```yaml
KvTransferControlRequest:
  schema: sglang.tensorcast.kv_transfer_control_request.v1
  op_type: publish | hydrate | evict_local
  op_id: uuid
  logical_request_id: string
  instance_id: string

  tp_world:
    tp_size: int
    pp_size: int
    required_ranks:
      - tp_rank: int
        pp_rank: int

  snapshot:
    cutoff_token_count: int
    last_page_index: int
    page_size: int
    radix_last_hash: string | null
    bundle_digest: string

  publish_args:
    publication_mode: substrate_finalize
    allow_force_flush_missing_tail: true
    ttl_ms: int | null

  hydrate_args:
    publish_manifest_digest: string
    artifact_manifest_digest: string
    engine_owned_manifest_sha256: string
    install_mode: prepare_only

  evict_args:
    scope: prepared_bundle | live_request
```

```yaml
KvTransferRankResult:
  schema: sglang.tensorcast.kv_transfer_rank_result.v1
  op_id: uuid
  logical_request_id: string
  tp_rank: int
  pp_rank: int
  status: success | failed | partial

  publish_result:
    snapshot_cutoff_token_count: int
    longest_present_prefix_pages: int
    forced_flush_pages: int
    published_page_count: int
    missing_page_count: int

  hydrate_result:
    hydrated_page_count: int
    install_ready: bool
    prepared_bundle_key: string | null

  evict_result:
    evicted_live_pages: int
    evicted_prepared_pages: int

  error:
    code: string | null
    message: string | null
```

Recommended coordinator semantics:

- `publish` succeeds only when all required ranks have closed the same fixed
  snapshot cutoff and the coordinator can commit one immutable
  `PublishManifest`.
- `hydrate` succeeds only when all required ranks report `install_ready=true`
  for the same `PublishManifest` generation.
- `evict_local` is group-scoped across the logical instance even when physical
  cleanup work happens per rank.

### 6.5 Transfer-handle and data reuse rule

Request-level transfer MUST reuse the shared KV substrate, not create a second
separate storage universe.

That means:

- source-side `publish()` should describe a request bundle over already-stored or
  newly-stored page artifacts,
- target-side `hydrate()` should reconstruct engine-local runtime state from
  those same page artifacts,
- optional worker warmup should prefetch those same artifacts.

It MUST also preserve the handle split:

- `engine_request_id` locates mutable live request state on the source or
  during local cleanup,
- `PublishManifest` identifies one immutable published snapshot for transfer.

The controller SHOULD pass the entire `PublishManifest` by value to the target
hydrate phase.

The integration SHOULD NOT invent:

- a second controller-visible string transfer handle,
- or a standalone request-metadata byte artifact in Tensorcast dataplane.

It SHOULD instead rely on one SGLang-owned request-bundle metadata record per
live logical request. That metadata stays inside the SGLang integration and is
used to:

- identify the fixed publish cutoff,
- enumerate the authoritative ordered page set for that cutoff,
- bind one `PublishManifest` generation to one logical request generation,
- and later prepare target-side decode admission.

Retention implementation rule:

- the `ttl_ms` carried on `publish(engine_request_id=..., ttl_ms=...)` SHOULD
  be translated into the minimum retention lease for all required page
  artifacts of that published snapshot,
- if a page is already present with shorter remaining lifetime, publish should
  retain / touch / re-publish as needed before declaring closure satisfied,
- stronger durable/policy-backed retention is an optional later extension, not
  the baseline v1 request-transfer assumption.

### 6.6 Mixed-state handling between passive page writes and active publish

The integration MUST explicitly support a mixed state where:

- some KV pages have already been passively written into the shared substrate by
  prefix-share or write-through activity,
- some pages are currently in-flight through that passive path,
- and some pages are still absent.

This mixed state is normal and SHOULD be the expected steady-state behavior of a
shared substrate.

Therefore request-level `publish()` MUST NOT mean:

- "upload every page from scratch"

It MUST mean:

- establish a complete request bundle closure over the shared substrate.

Concretely, source-side `publish()` should:

1. resolve the live request plus its SGLang-owned request-bundle metadata,
2. freeze one fixed snapshot boundary expressed at least as token/page cutoff,
3. enumerate the full required page set for exactly that cutoff,
4. for each page:
   - reuse it if already ready,
   - join or adopt compatible in-flight publication when possible,
   - publish it if missing,
   - if it belongs to the chosen cutoff but has not yet reached the shared
     substrate, force or wait for the missing tail flush without chasing newer
     tokens beyond the chosen cutoff,
   - and upgrade retention when mere existence is not sufficient,
5. commit the `PublishManifest` only after closure is satisfied.

This gives the two upper interfaces different semantic strength:

- passive prefix-share writes:
  - opportunistic,
  - page-level,
  - best-effort population of the shared substrate.
- active request-level `publish()`:
  - bundle-level,
  - closure-establishing,
  - responsible for completeness and transfer retention.

The request-bundle metadata used in step 1 is an SGLang integration concern:

- it is not a Tensorcast-core registry,
- it is not a standalone Tensorcast dataplane artifact,
- and it is the source of truth for "what exactly belongs to this published
  snapshot generation".

### 6.7 Request-level publish flow

The recommended source-side publish flow is:

```mermaid
flowchart TD
    A["Tensorcast publish(engine_request_id, ttl_ms)"] --> B["Coordinator resolves logical request<br/>and SGLang-owned request-bundle metadata"]
    B --> C["Freeze one fixed snapshot cutoff<br/>(token/page frontier)"]
    C --> D["Enumerate required ranks and required page shards<br/>for exactly that cutoff"]
    D --> E["For each required page shard: ready / inflight / missing?"]
    E --> F["Reuse ready pages"]
    E --> G["Adopt or wait on compatible inflight publication"]
    E --> H["Force or wait for missing tail flush<br/>then publish missing host-page shards"]
    F --> I["All required page shards for cutoff satisfy closure"]
    G --> I
    H --> I
    I --> J["Assemble artifact_manifest + EngineOwnedManifest<br/>bound to cutoff digest"]
    J --> K["Commit immutable PublishManifest"]
    K --> L["Return PublishResult(publish_manifest, put_outcomes)"]
```

Failure boundary:

- the flow fails before `K` if any required rank or required page shard cannot
  satisfy closure,
- partial page success does not imply request-bundle success.

### 6.8 Request-level hydrate flow

The recommended target-side hydrate flow is:

```mermaid
flowchart TD
    A["Tensorcast hydrate(publish_manifest)"] --> B["Coordinator validates PublishManifest"]
    B --> C["Bind engine_owned_manifest to artifact_manifest_digest"]
    C --> D["Compute per-rank hydrate work items"]
    D --> E["Optional worker warmup already made some artifacts local_replica_ready"]
    E --> F["Each required rank materializes missing page artifacts into host pages"]
    F --> G["Each required rank reconstructs local decode-usable state"]
    G --> H["Coordinator gathers per-rank success / failure"]
    H -->|All required ranks succeeded| I["Install prepared local request bundle<br/>keyed by logical_request_id"]
    H -->|Any required rank failed| J["Best-effort group cleanup / mark tainted"]
    I --> K["Return group-level HydrateResult"]
    J --> L["Return hydrate failure"]
```

Recommended hydrate semantics:

- artifact fetch/materialization may be internally best-effort at the
  per-artifact level,
- `HydrateResult` may therefore carry partial `get_outcomes` /
  `missing_artifact_ids`-style diagnostic information,
- but the external group-level `hydrate()` call remains fail-closed,
- and success is reported only when the EngineAdapter has installed one
  runnable prepared local request bundle that ordinary decode admission can
  consume.

In other words:

- partial artifact transport is a diagnostic fact,
- runnable decode preparation is the success criterion.

#### 6.8.1 Target-side ingress after hydrate

The preferred target-side ingress for decode continuation is:

- `hydrate(publish_manifest=...)` prepares target-local request state,
- the controller then sends the ordinary SGLang generate/decode request through
  the normal serving endpoint,
- and that ordinary request carries the same logical request id in the
  caller-visible SGLang `rid` field.

Recommended target-side contract:

1. successful `hydrate()` stores one prepared local request bundle keyed by the
   logical request id carried through the publish/hydrate flow,
2. the prepared bundle is local SGLang integration state rather than a
   Tensorcast directory object,
3. when the controller later sends the ordinary `/generate` request with the
   same `rid`, the target coordinator binds that request admission to the
   prepared bundle instead of re-running prefill,
4. if no prepared bundle exists for that `rid`, the normal SGLang path applies
   and no hidden transfer recovery occurs.

This keeps the external serving ingress minimally invasive:

- no new public "resume by manifest" HTTP endpoint is required for v1,
- the existing ordinary SGLang request path remains the decode-entry surface,
- and Tensorcast-driven `hydrate()` remains a separate control-plane
  preparation step.

```mermaid
flowchart TD
    A["Controller runs hydrate(publish_manifest=...)"] --> B["Target coordinator installs prepared bundle"]
    B --> C["Controller sends ordinary /generate with same rid"]
    C --> D["SGLang request admission resolves rid -> prepared bundle"]
    D -->|found| E["Resume decode without re-running prefill"]
    D -->|not found| F["Fall back to ordinary SGLang admission semantics"]
```

#### 6.8.2 Legacy compatibility hydrate resolution

If we keep `hydrate(engine_request_id=...)` for compatibility, it SHOULD be a
controller-side shim rather than a target-instance-side resolution mechanism.

The controller-side cache / registry:

- belongs to the controller or router control plane,
- stores previously returned `PublishManifest` objects keyed by logical request
  id,
- is not a Tensorcast-core registry feature,
- and is not an SGLang target-instance runtime responsibility.

This compatibility mode SHOULD be explicitly limited to single-controller
deployments. If multiple independent controllers can publish or hydrate the
same logical request id, the compatibility shim should be considered unsafe and
disabled.

Recommended compatibility-resolution flow:

```mermaid
flowchart TD
    A["controller helper hydrate(engine_request_id)"] --> B["Lookup cached PublishManifest by logical_request_id"]
    B -->|exactly one match| C["Emit remote hydrate(publish_manifest) plan"]
    B -->|zero or many matches| D["Fail closed"]
```

The target SGLang instance SHOULD receive only the explicit-handle form
`hydrate(publish_manifest=...)`.

### 6.9 Instance-agent and EngineAdapter role

In the SGLang v1 design, the Tensorcast instance-step execution boundary is an
in-process instance-agent hosted inside the logical SGLang instance. That
instance-agent receives the routed Tensorcast instance step and delegates the
actual SGLang-specific translation work to the EngineAdapter.

The SGLang EngineAdapter is responsible for translating Tensorcast instance
steps into SGLang operations:

- `publish`
  - identify the group-scoped request plus its request-bundle metadata,
  - invoke coordinator fan-out when the logical instance has multiple ranks,
  - ensure relevant pages exist in the shared KV substrate,
  - produce one group-level `PublishManifest` output.
- `hydrate`
  - resolve the authoritative `PublishManifest`,
  - invoke coordinator fan-out when the logical instance has multiple ranks,
  - fetch or locate required pages,
  - reconstruct target-side decode-usable KV state across all required ranks,
  - install the prepared local request bundle that ordinary `/generate(rid)`
    admission will consume.
- `evict_local`
  - remove local engine-side request state after handoff or cleanup across the
    whole logical instance.

The EngineAdapter is therefore the bridge from:

- Tensorcast's one-instance-step / one-`instance_id` public surface

to:

- SGLang's potentially multi-rank internal execution model.

The Tensorcast daemon is not the owner of this translation logic. It may remain
the worker-step and shared-substrate host on the same node, but the
instance-step execution host belongs to the SGLang instance boundary.

### 6.10 Request bundle vs prefix bundle

A request bundle is not identical to a prefix bundle.

A request bundle MAY contain:

- a suffix beyond a reusable shared prefix,
- request-scoped decode state,
- engine-specific metadata needed for resume.

But request bundles SHOULD reuse underlying prefix/shared pages whenever
possible.

---

## 7) Shared Runtime Component

### 7.1 Why a shared runtime is useful

To avoid duplicating logic, both upper interfaces should call into one shared
SGLang-side Tensorcast KV runtime component.

This component would own:

- page publication and retrieval,
- bundle metadata assembly and resolution,
- SGLang-owned request-bundle metadata for live logical requests,
- prepared-bundle admission state for hydrated target requests,
- Tensorcast-specific keying and artifact operations,
- translation between SGLang page hashes and Tensorcast artifact identity.

This shared runtime component MUST live on the SGLang integration side. It is
not a Tensorcast-core responsibility.

### 7.2 Suggested responsibilities

The shared runtime should expose internal operations such as:

- ensure page artifacts are published,
- fetch page artifacts into host pages,
- build prefix bundle metadata,
- build `PublishManifest` from generic artifact-manifest data plus opaque
  `EngineOwnedManifest`,
- resolve request transfer by `PublishManifest`,
- resolve prefix bundle by prefix identity.

For request-level transfer, this shared runtime SHOULD also provide a
coordinator-facing API that can:

- freeze a request snapshot boundary,
- enumerate required ranks and shard contributions,
- assemble one group-level `PublishManifest`,
- resolve one group-level `PublishManifest` back into per-rank hydrate work
  items.

It SHOULD also own a **page publication registry** or equivalent deduplication
mechanism for SGLang-side publication work.

This registry is an SGLang-integration concern, not a Tensorcast-core feature.

Its purpose is to coordinate SGLang-side page publication attempts so that:

- passive prefix-share writes and active request-level publish can deduplicate by
  page identity,
- active publish can observe compatible in-flight passive publication,
- retention upgrades can be applied without pretending that "already exists"
  always means "publish contract already satisfied".

The exact implementation is flexible, but it SHOULD behave like a per-page
publication coordination table with states such as:

- absent,
- in-flight,
- ready,
- failed.

The exact Python module layout is flexible, but the logic should not be split
independently across:

- a storage backend implementation,
- an EngineAdapter implementation,
- and controller-specific helpers

without a shared source of truth.

#### 7.2.1 Source-side `RequestBundleState` record

To make source-side `publish()` deterministic, the shared runtime SHOULD keep
one coordinator-owned record per live logical request. This record is the
authoritative local source of truth for:

- the current live request frontier,
- the fixed cutoff chosen for one publish generation,
- the ordered page membership of that generation,
- and the publish manifest generation last committed for that request.

Conceptual Pydantic-style schema:

```python
from pydantic import BaseModel
from typing import Literal


class RankCoord(BaseModel):
    tp_rank: int
    pp_rank: int


class PageClosureEntry(BaseModel):
    logical_page_index: int
    page_hash: str
    publication_state: Literal["absent", "inflight", "ready", "failed"]
    artifact_id: str | None = None
    host_resident: bool
    last_error: str | None = None


class RankSnapshotCursor(BaseModel):
    tp_rank: int
    pp_rank: int
    latest_token_count: int
    latest_last_page_index: int
    frozen_cutoff_token_count: int | None = None
    frozen_last_page_index: int | None = None
    force_flush_cursor: int | None = None
    ordered_pages: list[PageClosureEntry]


class RequestBundleState(BaseModel):
    logical_request_id: str
    instance_id: str
    engine_request_id: str
    model_fingerprint: str
    kv_layout_id: str
    tp_size: int
    pp_size: int
    state: Literal[
        "live_tracking",
        "snapshot_closing",
        "closing_tail_flush",
        "closure_ready",
        "published",
        "publish_failed",
        "cleaned",
    ]
    snapshot_seq: int
    publish_op_id: str | None = None
    frozen_cutoff_token_count: int | None = None
    frozen_last_page_index: int | None = None
    bundle_digest: str | None = None
    latest_publish_manifest_digest: str | None = None
    required_ranks: list[RankCoord]
    rank_snapshots: list[RankSnapshotCursor]
    created_at_ms: int
    updated_at_ms: int
```

Interpretation notes:

- `ordered_pages` is a conceptual schema. Real implementations MAY store this
  in a more compact array-oriented form because page cardinality can be high.
- `publication_state` should be derived from or reconciled with the shared
  page publication registry; it does not replace that registry.
- `snapshot_seq` is monotonically increasing per logical request and gives the
  coordinator a local generation number even before a `PublishManifest` exists.
- `latest_publish_manifest_digest` records the last committed external transfer
  handle generation for observability and idempotent retry.

Recommended ownership:

- rank 0 owns mutation of `RequestBundleState`,
- non-coordinator ranks contribute per-rank cursors and page membership data,
- and only the coordinator may transition the record into `published`.

#### 7.2.2 Source-side `RequestBundleState` state machine

```mermaid
stateDiagram-v2
    [*] --> LiveTracking
    LiveTracking --> SnapshotClosing: publish(op_id) accepted
    SnapshotClosing --> ClosingTailFlush: fixed cutoff sealed on all required ranks
    ClosingTailFlush --> ClosureReady: all required page shards up to cutoff are ready
    ClosureReady --> Published: PublishManifest committed
    ClosingTailFlush --> PublishFailed: any required rank/page terminal failure
    PublishFailed --> SnapshotClosing: retry publish with new snapshot_seq
    Published --> LiveTracking: newer live progress continues beyond published cutoff
    Published --> Cleaned: source local request cleanup
    LiveTracking --> Cleaned: request released without further publish
```

Recommended semantics:

- `LiveTracking`: request is still advancing and the coordinator records the
  latest visible frontier.
- `SnapshotClosing`: one publish generation has claimed a fixed cutoff and
  ranks are sealing page membership for exactly that cutoff.
- `ClosingTailFlush`: the system is forcing or waiting for missing tail pages
  up to the chosen cutoff, but it is not chasing newer tokens.
- `ClosureReady`: all pages required for the chosen cutoff satisfy the publish
  contract and the coordinator can assemble `PublishManifest`.
- `Published`: one immutable generation exists; the live request may still
  continue and later publish a newer generation.
- `PublishFailed`: the chosen cutoff could not be closed and no external
  success was reported.

#### 7.2.3 Source-side snapshot closure algorithm

Recommended coordinator algorithm:

1. Resolve `RequestBundleState` by `logical_request_id`.
2. Increment `snapshot_seq` and assign `publish_op_id`.
3. Freeze one cutoff:
   - `frozen_cutoff_token_count`
   - `frozen_last_page_index`
4. Broadcast that cutoff to all required ranks.
5. Each rank seals its `RankSnapshotCursor` to that cutoff and returns the
   ordered page list for exactly that cutoff.
6. For each page shard in cutoff order:
   - adopt if `publication_state=ready`,
   - join/wait if `publication_state=inflight`,
   - if `absent`, call a rank-local `force_flush_to_cutoff()` path that pushes
     the missing host-page shard into the shared substrate,
   - fail if `failed` or if host/L2 materialization for the cutoff cannot be
     recovered.
7. Compute `bundle_digest` from the ordered per-rank page identities bound to
   that cutoff.
8. Only after all required ranks report closure satisfied, assemble and commit
   `PublishManifest`.

Recommended implementation rules:

- `force_flush_to_cutoff()` MUST be bounded by the chosen cutoff.
- A page for token frontier `N+1` MUST NOT be pulled into the current publish
  generation if the frozen cutoff is `N`.
- Source request execution MAY continue after the cutoff is frozen; newer tokens
  simply belong to a future `snapshot_seq`.
- If the live request is released before the chosen cutoff can be closed,
  `publish()` should fail rather than invent a partial bundle.

#### 7.2.4 Target-side `PreparedBundleRecord`

To make ordinary `/generate(rid)` admission deterministic after `hydrate()`,
the shared runtime SHOULD keep one target-local record per hydrated transfer
generation.

Conceptual Pydantic-style schema:

```python
from pydantic import BaseModel
from typing import Literal


class RankInstallRecord(BaseModel):
    tp_rank: int
    pp_rank: int
    state: Literal["preparing", "ready", "failed", "cleaned"]
    hydrated_page_count: int
    runnable_prefix_tokens: int
    local_install_handle: str | None = None
    last_error: str | None = None


class PreparedBundleRecord(BaseModel):
    logical_request_id: str
    target_instance_id: str
    publish_manifest_digest: str
    artifact_manifest_digest: str
    engine_owned_manifest_sha256: str
    state: Literal[
        "preparing",
        "prepared",
        "claimed",
        "attached",
        "consumed",
        "failed",
        "evicted",
    ]
    required_ranks: list[RankCoord]
    rank_installs: list[RankInstallRecord]
    claim_token: str | None = None
    active_scheduler_rid: str | None = None
    prepared_bundle_key: str | None = None
    created_at_ms: int
    prepared_at_ms: int | None = None
    claimed_at_ms: int | None = None
    cleaned_at_ms: int | None = None
```

Interpretation notes:

- `PreparedBundleRecord` is local SGLang integration state, not a Tensorcast
  directory object and not a distributed registry entry.
- `publish_manifest_digest` is the primary generation key. The same
  `logical_request_id` may see more than one generation over time.
- `claim_token` is an admission-time compare-and-swap token used to prevent two
  concurrent `/generate(rid)` requests from consuming the same prepared bundle.

Recommended table key:

- primary key: `(logical_request_id, publish_manifest_digest)`
- secondary admission index: `logical_request_id -> latest prepared generation`

#### 7.2.5 Target-side `PreparedBundleRecord` state machine

```mermaid
stateDiagram-v2
    [*] --> Preparing
    Preparing --> Prepared: all required ranks install runnable state
    Preparing --> Failed: hydrate group failure
    Prepared --> Claimed: ordinary /generate(rid) atomically claims bundle
    Claimed --> Attached: scheduler creates live request from prepared state
    Claimed --> Prepared: admission rollback before live request is created
    Attached --> Consumed: live request owns the state
    Prepared --> Evicted: cleanup without use
    Failed --> Evicted: cleanup
    Consumed --> Evicted: request finished / local cleanup
```

Recommended semantics:

- `Preparing`: hydrate is still materializing/installing state.
- `Prepared`: one runnable bundle exists and is available for exactly one decode
  admission claim.
- `Claimed`: admission has won the race for this bundle but has not yet
  attached it to an SGLang live request object.
- `Attached`: scheduler/live-request construction is in progress using this
  prepared bundle.
- `Consumed`: ownership has moved to the live request; the prepared-bundle
  record is no longer reusable for another decode admission.

#### 7.2.6 Target-side admission binding algorithm

Recommended coordinator algorithm for ordinary `/generate(rid)` after hydrate:

1. Request admission receives caller-provided `rid=logical_request_id`.
2. Coordinator checks for an existing live request with the same logical id.
3. Coordinator checks the prepared-bundle table for that logical id.
4. If no prepared bundle exists:
   - follow the normal SGLang path.
5. If exactly one `PreparedBundleRecord(state="prepared")` exists:
   - atomically transition it to `claimed`,
   - mint `claim_token`,
   - construct the live request from the prepared bundle,
   - transition to `attached` and then `consumed`.
6. If the prepared bundle is present but in `claimed`, `attached`, `consumed`,
   or `failed`:
   - fail closed rather than silently falling back to a second hidden resume.

Recommended conflict rules:

- If a live request already exists for `logical_request_id`, `hydrate()` SHOULD
  reject installing a new prepared bundle for that id.
- If a prepared bundle already exists with a different
  `publish_manifest_digest`, a second hydrate SHOULD fail closed unless the
  existing record has already been cleaned.
- Repeated hydrate with the same `publish_manifest_digest` MAY be treated as an
  idempotent retry.
- Prepared-bundle consumption SHOULD be one-shot. Reusing the same prepared
  bundle for multiple decode admissions should not be the default v1 behavior.

### 7.3 Bundle state models

The shared runtime SHOULD define explicit lifecycle states for both prefix
bundles and request bundles. These state machines are not Tensorcast-core
objects; they are SGLang integration state.

These lifecycle models are intentionally higher-level than the concrete
coordinator-owned records in:

- `7.2.1 Source-side RequestBundleState`
- `7.2.4 Target-side PreparedBundleRecord`

The concrete records describe implementation state for one coordinator/runtime.
The lifecycle models below describe the broader semantic phases that the
integration exposes and reasons about.

#### 7.3.1 Prefix bundle state

Prefix bundles are background or hot-path share metadata over already-published
or newly-published page artifacts.

```mermaid
stateDiagram-v2
    [*] --> Unknown
    Unknown --> Building: page sequence observed and metadata being assembled
    Building --> Ready: ordered page set is complete enough for prefix-share use
    Building --> Failed: metadata assembly failed
    Ready --> Ready: reused by more requests
    Ready --> Stale: constituent pages expired or metadata invalidated
    Stale --> Building: rebuild from current page set
    Failed --> Building: retry assembly
```

Recommended semantics:

- `Ready` means the bundle can answer prefix-share existence / fetch queries.
- `Stale` means the bundle name may still exist logically, but its page closure
  is no longer trusted.
- `Stale` should be entered whenever the validation rules in
  `4.4.2 Prefix bundle invalidation rules` are violated.

#### 7.3.2 Request bundle state

Request bundles are stronger than prefix bundles because they must satisfy
decode-resume closure for one logical request.

```mermaid
stateDiagram-v2
    [*] --> LiveOnly
    LiveOnly --> SnapshotClosing: publish begins and snapshot boundary freezes
    SnapshotClosing --> PublishingPages: required page closure is being satisfied
    PublishingPages --> Published: immutable PublishManifest committed
    PublishingPages --> PublishFailed: closure failed
    Published --> Hydrating: target hydrate begins
    Hydrating --> Hydrated: all required ranks installed runnable state
    Hydrating --> HydrateFailed: one or more required ranks failed
    Hydrated --> CleanupPending: decode ownership moved, local cleanup optional
    CleanupPending --> [*]: cleanup complete or TTL expiry
    PublishFailed --> SnapshotClosing: retry publish
    HydrateFailed --> Hydrating: retry hydrate
```

Recommended interpretation:

- `Published` is the first state in which a request bundle is externally
  authoritative for request-level transfer.
- every successful transition into `Published` SHOULD mint one immutable
  `PublishManifest` generation.
- `Hydrated` is target-local success, not global workflow completion.
- `CleanupPending` keeps request-level transfer semantics separate from source
  or target local eviction policy.

---

## 8) Fit With Current Tensorcast Capabilities

### 8.1 What Tensorcast already provides

Current Tensorcast already provides the right building blocks for the
request-level control plane:

- runtime-bound `Plan`,
- worker `prefetch_set` / `prefetch_manifest_result`,
- instance `publish(engine_request_id=...)`,
- legacy `hydrate(engine_request_id=...)`,
- `evict_local(engine_request_id=...)`,
- `ManifestResult` and `ManifestArtifactSetBridge`.

These are necessary building blocks, but not yet the complete target
request-transfer surface.

### 8.2 What Tensorcast does not yet directly provide

Current Tensorcast does not yet directly provide all the primitives needed for
either:

- the prefix-share hot path, or
- the final explicit-handle request-transfer surface.

The main gaps are:

- a stable public page-store-style batch API tailored for high-cardinality KV
  page IO, rather than today’s lower-level byte-artifact batch-region surfaces,
- host-native batch region registration for byte-artifact ingress/egress,
- a Tensorcast-aware SGLang host allocator for L2 page residency,
- explicit public prefix-bundle programmability,
- `PublishResult.publish_manifest`,
- opaque `EngineOwnedManifest` carriage in the instance-step result path,
- `hydrate(publish_manifest=...)`,
- and a public hydrate-by-manifest or hydrate-by-artifact-set instance-step
  surface.

### 8.3 Design consequence

Because of these gaps, the recommended implementation strategy is:

- build the prefix-share path as an internal SGLang integration over Tensorcast
  storage/artifact primitives,
- while using current Tensorcast programmability for request transfer with as
  little Tensorcast-core extension as possible.

---

## 9) Recommended Implementation Order

### 9.1 Phase 1: shared substrate

Implemented in the current repo:

Implement:

- the shared SGLang-side Tensorcast KV runtime,
- page publication/retrieval.

Goal:

- make Tensorcast usable as the distributed KV pool for prefix share.

### 9.2 Phase 1.5 + Phase 2: byte-artifact-native prefix share and benchmark bring-up

Implemented in the current repo:

Implement:

- the byte-artifact-native batch `exists/get/set` path over VRAM staging,
- the `share_local` Tensorcast benchmark harness,
- `tensorcast-daemon-mode=share`,
- `tensorcast-daemon-mode=separate`,
- explicit benchmark `rid`,
- source-publication-drain measurement,
- and log-based validation of real HiCache prefix reuse.

Goal:

- make the Tensorcast backend a real SGLang prefix-share backend rather than a
  placeholder topology harness.

### 9.3 Phase 3: request-level EngineAdapter

Still target design; not implemented yet:

Implement:

- SGLang in-process instance-agent / EngineAdapter `publish`,
- SGLang in-process instance-agent / EngineAdapter `hydrate`,
- SGLang in-process instance-agent / EngineAdapter `evict_local`,
- `PublishManifest` / `EngineOwnedManifest` plumbing across the Tensorcast
  result and instance-step boundary,
- explicit-handle `hydrate(publish_manifest=...)` support,
- coordinator-owned Tensorcast directory registration / heartbeat for the
  in-process instance-agent execution endpoint,
- coordinator-side typed internal control requests for KV publish / manifest /
  hydrate / evict,
- `PublishManifest` construction on top of the same page artifacts.
- target-side prepared-bundle admission binding from ordinary `/generate(rid)`
  into hydrated decode state.

Goal:

- enable PD transfer using the protocol in
  `tensorcast_kv_protocol.md`.

Optional controller-side helper work, if we decide to keep a convenience
`hydrate(engine_request_id=...)` entrypoint, should stay outside the SGLang
target-instance runtime and simply resolve a cached `PublishManifest` before
emitting the canonical remote `hydrate(publish_manifest=...)` plan.

### 9.4 Future optional programmable prefix bundle operations

If useful later, add coarse-grained programmability for prefix bundles, such as:

- prefix-bundle prewarm,
- rollout to selected nodes,
- debugging and inspection.

This phase is optional and should not block the main two-path integration.

---

## 10) Non-goals and Guardrails

The integration SHOULD avoid these failure modes:

- building one Tensorcast path for prefix share and a separate incompatible path
  for request transfer,
- treating `engine_request_id` as the stable distributed identity for KV pages,
- exposing one Tensorcast `instance_id` per TP rank to the external caller for
  request transfer,
- routing ordinary prefix-share page IO through the request-transfer
  coordinator,
- routing synchronous prefix-hit traffic through external callers and `Plan`,
- reducing Tensorcast to only a thin `HiCacheStorage` replacement without
  reusing it in request-level transfer.

The core guardrail is simple:

- one distributed KV substrate,
- two upper interfaces,
- one consistent identity model.

---

## 11) Summary

The recommended Tensorcast + SGLang KV integration is:

- **shared bottom layer**
  - page artifacts, bundle metadata, shared distributed KV pool.
- **upper interface A: prefix share**
  - internal SGLang hot path,
  - Mooncake-like batch exists/get/set,
  - no per-request external programmability.
- **upper interface B: request transfer**
  - external caller-driven Tensorcast programmability,
  - coordinator-backed logical-instance `publish` / `hydrate` / `evict_local`,
  - optional worker warmup via `prefetch_manifest_result(...)`.

This is the intended design baseline for the implementation work that follows.
