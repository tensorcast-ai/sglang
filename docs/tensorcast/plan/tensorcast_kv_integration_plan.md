# Tensorcast KV Integration Plan (SGLang)

This is the executable TODO list for implementing Tensorcast-backed KV cache
integration in SGLang.

It follows the target semantics frozen in:

- `sglang/docs/tensorcast/tensorcast_kv_protocol.md`
- `sglang/docs/tensorcast/tensorcast_kv_integration.md`

The implementation is intentionally split into:

1. `Phase 1`: shared KV substrate
2. `Phase 1.5`: byte-artifact-native substrate refactor
3. `Phase 2`: prefix share interface
4. `Phase 2.5`: host-native CPU-region path
5. `Phase 3`: request transfer interface

This split is the right one for the current codebase because:

- `Phase 1` is the smallest useful data-plane unit,
- `Phase 1.5` is the first performance-critical substrate rewrite, but still
  stays below request-level programmability,
- `Phase 2` can still be validated entirely inside ordinary SGLang serving with
  no external controller,
- `Phase 2.5` extends the same prefix-share path with `HOST_SHARED` local
  regions and allocator-backed host residency, but still requires no external
  controller,
- `Phase 3` is the first phase that requires Tensorcast programmability-facing
  API additions and instance-agent / coordinator work.

Current repo status:

- `Phase 0` through `Phase 2.5` are implemented in the current codebase.
- `Phase 3A` through `Phase 3C` and the local `M6` external-caller validation
  portion of `Phase 3D` are implemented in the current codebase.
- Remaining planned work is:
  - the remote `M6` validation in `Phase 3D`,
  - and the hardening / observability / performance follow-up in `Phase 4`.

## Target Outcome

- Add a Tensorcast-backed shared KV substrate for SGLang HiCache pages, with KV
  pages represented in Tensorcast as high-cardinality byte artifacts.
- Make `--hicache-storage-backend tensorcast` a real SGLang HiCache backend
  rather than a benchmark placeholder.
- Make `benchmark/tensorcast_benchmark/kv/share_local` run end-to-end with
  `hicache-storage-backend=tensorcast` on `charged-group=codesign`.
- Preserve the architectural split from the KV docs:
  - prefix share stays an internal SGLang hot path,
  - request-level transfer uses an external controller plus Tensorcast plans.
- Add the Phase-3 request-transfer surface centered on explicit
  `PublishManifest` / `EngineOwnedManifest`, not on implicit reuse of
  `engine_request_id`.

## Phase 0 - Foundation and Scope Freeze

- [x] Keep documentation aligned before code lands:
  - [x] Treat `tensorcast_kv_protocol.md` as the external contract.
  - [x] Treat `tensorcast_kv_integration.md` as the internal design source of truth.
  - [x] Keep this plan file updated as implementation decisions change.
- [x] Freeze phase boundaries:
  - [x] `Phase 1` must not require Tensorcast plan / instance-step changes.
  - [x] `Phase 1.5` must preserve the Phase-1 external substrate contract while replacing the hot-path data plane.
  - [x] `Phase 2` must not require an external controller or request handoff.
  - [x] `Phase 2.5` must stay within the existing prefix-share hot path while replacing the local SGLang <-> daemon memory boundary.
  - [x] `Phase 3` is the only phase allowed to extend Tensorcast request-transfer semantics.
- [x] Freeze the Phase-2 success criterion:
  - [x] `benchmark/tensorcast_benchmark/kv/share_local` must run with
    `hicache-storage-backend=tensorcast`.
  - [x] The benchmark must start Tensorcast services automatically.
  - [x] The benchmark must preserve the same output contract as the Mooncake path.
  - [x] The target first validation environment is `charged-group=codesign`.
- [x] Preserve non-goals for early phases:
  - [x] No PD-prefill/decode split in `share_local`.
  - [x] No request-level `publish()` / `hydrate()` in `share_local`.
  - [x] No Tensorcast-core semantic changes for prefix-share hot path behavior.

## Phase 1 - Shared KV Substrate

This phase creates the smallest useful unit: a Tensorcast-backed HiCache
storage backend that can publish and retrieve page data correctly.

- [x] Add a built-in SGLang HiCache backend named `tensorcast`:
  - [x] Extend `--hicache-storage-backend` validation to include `tensorcast`.
  - [x] Register a Tensorcast backend in `StorageBackendFactory`.
  - [x] Add a dedicated backend package under
    `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/`.
- [x] Define the SGLang-side Tensorcast backend config surface:
  - [x] Parse backend extra config from `--hicache-storage-backend-extra-config`.
  - [x] Include daemon/global-store addresses, local daemon association, durable policy,
    and any benchmark-only overrides.
  - [x] Validate missing or inconsistent fields with clear errors.
- [x] Implement page identity and namespacing rules:
  - [x] Use SGLang page hash / prefix-chain-derived identity as the logical page key.
  - [x] Add TP-rank namespacing for MHA so different TP shards do not collide.
  - [x] Preserve MLA ownership rules instead of blindly duplicating all ranks.
  - [x] Keep page identity shared between prefix share and future request transfer.
- [x] Implement Tensorcast-backed page operations:
  - [x] `batch_exists(...)`
  - [x] `batch_get_v1(...)`
  - [x] `batch_set_v1(...)`
  - [x] Map each SGLang page to a Tensorcast byte artifact payload boundary.
  - [x] Keep v1 at the host/L2-facing boundary rather than requiring direct L1 GPU zero-copy semantics.
- [x] Define substrate publication semantics:
  - [x] Default KV-page publication policy should be durable-but-evictable.
  - [x] Duplicate page publication must be idempotent at the SGLang integration layer.
  - [x] Partial backend visibility must be tolerated so later publish/snapshot code can reason over mixed already-published and newly-published pages.
- [x] Integrate with current HiCache control flow without changing hot-path ownership:
  - [x] Keep per-rank radix tree ownership in SGLang.
  - [x] Keep TP synchronization in `HiCacheController` / `HiRadixCache`.
  - [x] Do not introduce a request-level coordinator hop for ordinary prefix share.
- [x] Add observability:
  - [x] Backend init logs for daemon/store endpoints and rank suffix rules.
  - [x] Counters or debug logs for page `exists/get/set`, duplicate put skip, and backend failures.
- [x] Add unit-level verification:
  - [x] Key/schema tests for TP/MLA namespacing.
  - [x] Storage round-trip tests for page put/get.
  - [x] Duplicate publication tests.
  - [x] Partial batch hit tests for prefix existence queries.

### Phase 1 Exit Criteria

- [x] One SGLang process can store and retrieve HiCache pages through Tensorcast.
- [x] TP>1 ranks do not key-collide in the shared substrate.
- [x] No external controller or Tensorcast instance-step logic is required yet.

## Phase 1.5 - Byte-Artifact-Native Substrate Refactor

This phase replaces the current generic artifact `key=` hot path with the real
byte-artifact-native batch path needed for scalable high-cardinality KV page IO.

It has two parts:

- **Phase 1.5A**
  - immediate refactor to the current best available Tensorcast path:
    byte-artifact-native batch IO over reusable GPU staging buffers.
- **Phase 1.5B**
  - medium-term follow-up to remove the temporary H2D / D2H staging copies by
    adding host-native region support and allocator integration.

### Phase 1.5A - Immediate byte-artifact-native batch refactor

- [x] Freeze the refactor contract:
  - [x] Preserve the existing external SGLang HiCache backend surface:
    - `batch_exists(...)`
    - `batch_get_v1(...)`
    - `batch_set_v1(...)`
  - [x] Preserve the existing semantic ownership boundary:
    - SGLang still owns `L1(device) <-> L2(host)`,
    - Tensorcast still begins at frozen L2 pages.
  - [x] Preserve page identity compatibility with future request transfer:
    - one rank-local page shard maps to one byte-artifact identity,
    - future bundle metadata can continue to point to those page artifact identities.
- [x] Finalize the byte-artifact identity and invariant schema in code:
  - [x] Replace the current `key=`-oriented page publication contract with a
    byte-artifact-native contract centered on:
    - `artifact_id`
    - `layout_id`
    - `byte_length`
    - payload digest / `PutIfAbsentInvariant`
  - [x] Make the identity builder explicit and shared by the current hot path so
    later request-transfer manifest construction can reuse it:
    - publication,
    - existence checks,
    - retrieval.
  - [x] Keep TP/PP/MLA ownership rules encoded in the same identity scheme.
  - [x] Strengthen the canonical logical-page identity recipe before Phase 3:
    - add `model_version` or served checkpoint revision,
    - require `layout_id` to encode dtype and page size explicitly,
    - keep TP / PP shard ownership in `engine_key`,
    - and forbid run-local values such as `run_id` or `daemon_id`.
  - [x] Add explicit byte-artifact verification-mode support for engine-owned
    logical KV pages:
    - keep strict payload-digest mode as the generic default,
    - add `LAYOUT_AND_SIZE_ONLY` for SGLang logical page identities,
    - and document that repeated logical-page `put` is first-writer-wins rather
      than upsert.
- [x] Add a dedicated SGLang-side Tensorcast byte-artifact client/runtime layer:
  - [x] Separate the hot path from the current generic `tc.Store.put(...)` /
    tensor fetch implementation.
  - [x] Reuse persistent daemon client state across batches instead of
    reconstructing page-local control objects.
  - [x] Encapsulate byte-artifact RPCs and response decoding behind one backend
    interface so `TensorcastStore` does not own all protocol details directly.
- [x] Rebuild the existence path on batch byte-artifact APIs:
  - [x] Replace per-page generic existence probing with `BatchExists(...)`.
  - [x] Build canonical `ArtifactSelection` values from page artifact identity.
  - [x] Preserve the current prefix semantics:
    - return the longest consecutive hit prefix,
    - fail closed on malformed or inconsistent batch outcomes.
- [x] Rebuild the write path on `BatchPutIfAbsentFromRegion(...)`:
  - [x] Introduce a reusable GPU staging-buffer manager per rank / process.
  - [x] Define coalesced staging layout rules:
    - one contiguous staging slice per page shard,
    - one batch `TargetLayout` covering the whole packed region.
  - [x] Implement host-page to staging-buffer packing:
    - collect L2 page shards,
    - copy or pack them into the coalesced GPU staging region,
    - preserve deterministic per-item slice mapping.
  - [x] Implement VRAM-region registration lifecycle:
    - register once and reuse while the existing capacity is sufficient,
    - clean up safely on process shutdown or backend reset,
    - leave explicit host-region support and richer region-lifecycle policy to later Tensorcast work.
  - [x] Build one `BatchPutIfAbsentFromRegionItem` per page shard with explicit
    invariant metadata.
  - [x] Interpret per-item batch outcomes into SGLang batch publication state:
    - new success,
    - already exists / adopt duplicate,
    - hard failure.
  - [x] Remove the generic outer `exists()+put()` logic from the steady-state
    publication path.
- [x] Rebuild the read path on `BatchGetIntoRegion(...)`:
  - [x] Introduce a reusable GPU staging-buffer manager for retrieval batches.
  - [x] Pack the target layout for the requested hit span.
  - [x] Materialize hit pages into GPU staging with one batch get.
  - [x] Copy or unpack staging slices back into SGLang L2 host-page buffers.
  - [x] Preserve partial-hit and partial-failure semantics compatible with the
    current HiCache controller expectations.
- [x] Clarify daemon-mode behavior:
  - [x] Support `tensorcast-daemon-mode=share`.
  - [x] Support `tensorcast-daemon-mode=separate`.
  - [x] Keep the same byte-artifact identity and publication semantics across
    both modes even if the internal transport setup differs.
  - [x] Keep the steady-state Tensorcast backend on the byte-artifact-native path
    rather than silently mixing generic per-page get/put operations into the hot path.
- [ ] Strengthen observability for the refactor:
  - [x] Add timing breakdowns for:
    - batch exists RPC,
    - batch put pack / stage-copy / RPC,
    - batch get pack / RPC / host fill.
  - [ ] Add timing breakdowns for:
    - region registration / reuse.
  - [x] Add counters or cumulative debug stats for:
    - pages published,
    - pages adopted as duplicates,
    - publication failures by outcome class.
  - [ ] Add counters for:
    - pages fetched.
  - [x] Surface enough logging or metrics to explain publication drain time in
    the benchmark without relying only on indirect log timing.
- [ ] Add focused correctness tests for the new substrate path:
  - [x] identity / invariant generation tests,
  - [ ] coalesced layout packing tests,
  - [x] storage outcome / duplicate publication tests at the backend layer,
  - [x] partial retrieval / partial hit tests,
  - [ ] region lifecycle tests where feasible.
- [x] Add benchmark-driven validation for the refactor:
  - [x] rerun `share_local` in `share` mode,
  - [x] rerun `share_local` in `separate` mode,
  - [x] compare source publication drain and TTFT deltas before and after the
    refactor,
  - [x] confirm prefix reuse signals still come from real page-hash-based hits.

### Phase 1.5B - Medium-term host-native substrate path

- [x] Freeze the medium-term direction:
  - [x] remove GPU staging via `HOST_SHARED` region-backed batch IO,
  - [x] preserve the byte-artifact identity and invariant contract,
  - [x] reuse the same daemon-exported slab mechanism for both Phase A and Phase
    B.
- [x] Move the executable implementation plan into `Phase 2.5`:
  - [x] `Phase 2.5A` now tracks Phase-A host staging on `HOST_SHARED`,
  - [x] `Phase 2.5B` now tracks Phase-B allocator-backed zero-copy host
    residency.

### Phase 1.5 Exit Criteria

- [x] The prefix-share hot path no longer relies on generic per-page
  `tc.Store.put(...)` / CPU tensor fetch for steady-state batch IO.
- [x] `batch_exists(...)`, `batch_get_v1(...)`, and `batch_set_v1(...)` all use
  byte-artifact-native batch semantics.
- [x] The benchmark remains functional in both `share` and `separate` daemon
  modes after the refactor.
- [x] The refactor produces enough observability to explain substrate
  publication and retrieval cost at the batch level.
- [x] The medium-term host-native direction is captured as an explicit follow-up
  under the same phase, not as an undocumented idea.

## Phase 2 - Prefix Share Interface and Benchmark Bring-up

This phase turns the shared substrate into a real prefix-share path and makes
the `share_local` benchmark the primary validation harness.

### Phase 2A - Benchmark Harness Completion

- [x] Finish the `share_local` Tensorcast backend path in
  `benchmark/tensorcast_benchmark/kv/share_local/run_benchmark.py`:
  - [x] Replace the current hard failure for `backend=tensorcast`.
  - [x] Generate Tensorcast config files into `outputs/<run_id>/generated_configs/`
    from `share_local/configs/`.
  - [x] Support `tensorcast-daemon-mode=share`.
  - [x] Support `tensorcast-daemon-mode=separate`.
  - [x] Start and stop Tensorcast global store and daemon(s) on the remote worker.
  - [x] Wait for service readiness and fail early on readiness timeout.
  - [x] Copy Tensorcast logs from `/data` into the benchmark output directory.
- [x] Reuse the proven benchmark service lifecycle pattern:
  - [x] Follow the service lifecycle model already used by
    `benchmark/tensorcast_benchmark/load_weight_remote/scripts/tensorcast_service.sh`.
  - [x] Decide whether to reuse that helper directly or extract a shared KV benchmark helper.
- [x] Add benchmark request-id control:
  - [x] Extend `benchmark/tensorcast_benchmark/kv/sgl_client.py` so `/generate` can send an explicit `rid`.
  - [x] Make `request_driver.py` construct deterministic request ids for each request pair.
  - [x] Keep the benchmark able to compare the same logical prompt across instances.
- [x] Keep benchmark result compatibility:
  - [x] Preserve `pair_results.jsonl`, `summary.json`, `logs/`, and append-only CSV behavior.
  - [x] Preserve the Mooncake path as the baseline.
- [x] Keep the target test environment explicit:
  - [x] Use `--brainctl-charged-group codesign` in documentation/examples for Tensorcast validation.
  - [x] Keep `--existing-worker-process` supported so an already-running 8xH800 worker can be reused.

### Phase 2B - SGLang Prefix-Share Functional Wiring

- [x] Make SGLang serve with `--hicache-storage-backend tensorcast`:
  - [x] Pass Tensorcast backend config into the SGLang server launch path.
  - [x] Ensure both instances in `share_local` can point at one shared daemon or two separate daemons.
  - [x] Keep the instances otherwise identical to the Mooncake benchmark topology.
- [x] Preserve SGLang-native prefix-share behavior:
  - [x] Storage queries must happen from the existing HiCache hot path.
  - [x] Prefix-tree insertion remains owned by `HiRadixCache`.
  - [x] No controller program is involved.
- [x] Verify functional prefix reuse:
  - [x] Confirm instance A writes reusable pages into the shared substrate while serving.
  - [x] Confirm instance B performs storage-backed prefix hits.
  - [x] Confirm the validation flow exposes actual cache reuse signals for the measured prompt pair, not only startup/service bring-up.
- [x] Validate on the target benchmark:
  - [x] Run `share_local` on `Qwen3-32B`, `tp=2`, same-node two-instance topology.
  - [x] Inspect logs and CSV output to confirm the Tensorcast path behaves like a functional prefix-share backend, not just a topology harness.

### Phase 2 Exit Criteria

- [x] `benchmark/tensorcast_benchmark/kv/share_local` runs with `hicache-storage-backend=tensorcast`.
- [x] The benchmark starts Tensorcast services automatically and cleans them up reliably.
- [x] Prefix-share requests on instance B can reuse pages produced by instance A.
- [x] The benchmark remains runnable on `charged-group=codesign`.

Current validation note:

- Validated runs:
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260330-114615_tensorcast_tp2_pairs1`
    - `tensorcast-daemon-mode=share`
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260330-184418_tensorcast_tp2_pairs1`
    - `tensorcast-daemon-mode=separate`
- Common validation shape:
  - `Qwen3-32B`
  - `tp=2`
  - same-node two-instance topology
  - dataset `LongBench/hotpotqa.jsonl`
  - `max_prompt_chars=35000`
  - `--extra-server-args "--page-size 32 --log-level debug"`
- Source-side publication for the measured prompt is confirmed by repeated
  `Tensorcast batch_set_v1 ... cumulative_pages=365` lines in instance-A logs.
- Target-side reuse for the same measured prompt is confirmed in instance-B
  logs by:
  - `HiCache storage hit query ... hit_tokens=11680 queried_pages=365`
  - `Prefetching 365 pages for request ...`
  - repeated `Tensorcast batch_get_v1 pages=128 succeeded=128`
- TTFT remains workload- and topology-sensitive in these runs, so the primary
  proof of prefix reuse is the request-scoped HiCache/storage log sequence
  above rather than any single positive TTFT delta.
- `meta_info.cached_tokens` remains `0` in this flow and must not be used as
  the proof signal for shared-substrate reuse.
- The explicit benchmark `rid` is only a correlation label for logs and result
  rows. The actual HiCache reuse decision is driven by token-prefix/page-hash
  identity, which is why the benchmark must send the exact same prompt text to
  both instances.

## Phase 2.5 - Host-Native CPU-Region Path

This phase removes the temporary GPU staging dependency from the Tensorcast
HiCache backend by extending the same region-backed batch API family to
`HOST_SHARED` and then binding SGLang L2 host residency directly onto that
exported slab model.

It has two parts:

- **Phase 2.5A**
  - bring up the shared `HOST_SHARED` slab mechanism and use it as a persistent
    host staging surface.
- **Phase 2.5B**
  - bind `HostKVCache` directly to allocator-managed exported slabs so L2 pages
    live in Tensorcast-shareable host memory with slot-generation protection.

Validation cadence for this phase is milestone-gated:

- the first layer is Tensorcast unit/integration testing tied to internal
  milestones `M0` through `M4`,
- the second layer is SGLang end-to-end validation via
  `benchmark/tensorcast_benchmark/kv/share_local/run_benchmark.py`,
- do not defer all validation to the end of Phase 2.5; each milestone should be
  exercised as soon as the corresponding code path is wired.

Milestone definitions used below:

- `M0`
  - **What it is**:
    the generic local-region substrate needed for host slabs.
  - **What must be true when it is complete**:
    a local client can register or attach a long-lived `HOST_SHARED` slab,
    obtain a lease-scoped region handle, keep it alive, and release it cleanly.
  - **What it does not include yet**:
    batch byte-artifact put/get on `HOST_SHARED`.
- `M1`
  - **What it is**:
    `BatchPutIfAbsentFromRegion(HOST_SHARED source)` correctness.
  - **What must be true when it is complete**:
    the local daemon can read bytes from a validated `HOST_SHARED` source layout
    and publish byte artifacts with the same per-item semantics as the existing
    `VRAM` path.
  - **What it does not include yet**:
    `HOST_SHARED` target materialization or allocator-backed direct residency.
- `M2`
  - **What it is**:
    `BatchGetIntoRegion(HOST_SHARED target)` correctness.
  - **What must be true when it is complete**:
    the local daemon can materialize hits directly into a validated
    `HOST_SHARED` target region, which is enough to switch SGLang from GPU
    staging to Phase-A host staging.
  - **What it does not include yet**:
    allocator slot generation, direct L2 residency, or CPU-source direct RDMA
    on the put path.
- `M3`
  - **What it is**:
    allocator-backed slot lifetime correctness for Phase B.
  - **What must be true when it is complete**:
    SGLang can reserve, pin, validate, retire, and recycle exported slab slots
    with `generation` protection, and stale completions cannot resurrect or
    corrupt reused slots.
  - **What it does not include yet**:
    optional host pinning via `cudaHostRegister(...)`.
- `M4`
  - **What it is**:
    host registration attach or detach plus fallback policy.
  - **What must be true when it is complete**:
    the SGLang-local mapping can optionally be host-registered for faster
    `L2(host) -> L1(device)` copies, and attach or detach remains correct even
    when host registration is unavailable.
  - **What it does not change**:
    the underlying byte-artifact semantics or slot-generation rules from `M3`.

### Phase 2.5A - Phase A host staging on `HOST_SHARED`

- [x] Freeze the Phase-2.5A contract:
  - [x] Keep the same prefix-share hot path ownership:
    - no external controller,
    - no request-transfer semantics,
    - no change to page identity or routed byte-artifact truth.
  - [x] Replace only the local SGLang <-> daemon memory boundary:
    - remove the mandatory GPU staging hop,
    - allow one host-side memcpy into or out of a persistent scratch slab,
    - keep daemon-side routed byte-artifact transport semantics unchanged.
  - [x] Require Phase A scratch slabs to reuse the same export / attach / lease
    model that Phase B allocator slabs will later use.
- [x] Implement `M0` Tensorcast host-slab export / attach / lease / keepalive:
  - [x] Tensorcast proto and client surface:
    - [x] extend `tensorcast/proto/tensorcast/daemon/v2/store_daemon.proto`
      toward the unified region model:
      - `RegionMemoryKind`,
      - `RegionRef`,
      - generic region registration messages or an equivalent compatibility
        wrapper shape,
      - room for `HOST_SHARED` storage entries in `TargetLayout`.
    - [x] keep `RegisterVramRegion` compatibility as a wrapper or legacy alias
      rather than breaking existing GPU users.
    - [x] update `tensorcast/tensorcast/daemon_ctl.py` and
      `tensorcast/tensorcast/api/store/__init__.py` so Python callers can use
      the new local-region surface.
  - [x] Tensorcast daemon region registry and lifecycle:
    - [x] refactor `tensorcast/daemon/state/ipc_region_registry.*` into a
      generic local-region registry or add an equivalent generic layer above it,
      so `VRAM` and `HOST_SHARED` share:
      - region ids,
      - owner_pid/session metadata,
      - lease refcounting,
      - TTL refresh,
      - poison state,
      - explicit unregister.
    - [x] add host-region descriptors for daemon-managed slabs:
      - slab size,
      - attach token or metadata,
      - whether the region is daemon-managed scratch or allocator-backed.
    - [x] implement daemon-side slab allocation and cleanup for `HOST_SHARED`.
  - [x] Tensorcast local attach mechanism:
    - [x] define how the local client receives a usable memfd for one exported
      slab.
    - [x] because plain gRPC does not transfer file descriptors, add a local-only
      attach path that can hand the memfd to the client process, for example via
      UDS FD passing or another explicitly local mechanism.
    - [x] keep this attach path local-only and lease-scoped; it must never be
      usable cross-host.
  - [x] Tensorcast daemon validation and ownership boundary:
    - [x] extend `daemon/service/controllers/external_target_access_service.*`
      so local source or target validation can reason about `HOST_SHARED`
      regions, not only CUDA IPC regions.
    - [x] extend
      `daemon/service/controllers/materialization_target_storage_utils.*` and
      `daemon/service/byte_artifact_region_layout.*` so region-backed accesses
      can acquire host-shared layouts in addition to VRAM layouts.
    - [x] keep trust-boundary rules unchanged:
      - batch region RPCs stay loopback or UDS-only,
      - remote daemons still never write directly into caller-visible memory.
  - [x] Tensorcast tests for `M0`:
    - [x] region registration tests for `HOST_SHARED`,
    - [x] lease keepalive and expiry tests,
    - [x] local attach or detach tests,
    - [x] slab cleanup on explicit release and owner exit.
- [x] Implement `M1` `BatchPutIfAbsentFromRegion(HOST_SHARED source)` correctness:
  - [x] Tensorcast request validation and lowering:
    - [x] extend
      `ExternalTargetAccessService::validate_local_source_layout(...)` so
      `BatchPutIfAbsentFromRegion(...)` accepts `HOST_SHARED` source entries in
      addition to `VRAM`.
    - [x] teach `byte_artifact_region_layout.*` to build region slices backed by
      daemon-managed host slabs.
    - [x] make `byte_artifact_controller.cc` open source bytes from
      `HOST_SHARED` region slices without changing byte-artifact identity,
      invariant, or verification-mode semantics.
  - [x] Tensorcast data-path scope:
    - [x] keep CPU-source direct RDMA explicitly out of scope for this
      milestone.
    - [x] allow the daemon-side put transport to continue using the current
      communicator/export realization after the local source bytes have been
      validated and opened.
  - [x] Tensorcast tests for `M1`:
    - [x] correctness tests for `BatchPutIfAbsentFromRegion(HOST_SHARED source)`
      with:
      - [x] success,
      - [x] duplicate adoption,
      - [x] partial failure,
      - [x] invalid bounds or poisoned region rejection.
  - [x] After `M1`, optionally do a narrow SGLang-side local integration check
    without yet switching the main backend path.
    Later `share_local` active-path validation exceeded this narrower check, so
    this item is treated as covered by stronger end-to-end evidence.
- [x] Implement `M2` `BatchGetIntoRegion(HOST_SHARED target)` correctness:
  - [x] Tensorcast request validation and target access:
    - [x] extend
      `ExternalTargetAccessService::validate_local_target_layout(...)` so
      `BatchGetIntoRegion(...)` accepts `HOST_SHARED` target entries in addition
      to `VRAM`.
    - [x] generalize
      `materialization_target_storage_utils.*` so target storage leases can
      represent host-shared targets, not only CUDA IPC mappings.
    - [x] keep device checks fail-closed:
      `device_uuid` remains meaningful for `VRAM` and not applicable for pure
      `HOST_SHARED`.
  - [x] Tensorcast target execution path:
    - [x] extend `byte_artifact_controller.cc` and any shared lowering helpers
      so `BatchGetIntoRegion(...)` can materialize directly into `HOST_SHARED`
      target slices.
    - [x] reuse or adapt the existing CPU direct-write sink path where possible
      instead of inventing a second host-target data plane.
      `HOST_SHARED` target materialization now goes through
      `materialize_mapped_loader_into_target(...)` with a dedicated
      `TargetLayoutHostSink`, so the controller no longer carries a bespoke
      `read_at(...)`-into-host helper path.
    - [x] preserve existing batch-get semantics:
      - [x] partial hit behavior,
      - [x] per-item vs pack-scoped failure rules,
      - [x] layout and verification-mode checks.
  - [x] Tensorcast tests for `M2`:
    - [x] correctness tests for `BatchGetIntoRegion(HOST_SHARED target)` with:
      - [x] full hit,
      - [x] partial hit,
      - [x] invalid target bounds,
      - [x] failure propagation without whole-slab poison.
- [x] Switch SGLang Tensorcast backend to Phase-A host staging after `M2`:
  - [x] Add a per-rank host-slab attachment manager in the Tensorcast backend.
  - [x] Replace `_StagingRegionManager`-style VRAM staging with persistent
    `HOST_SHARED` scratch slabs.
  - [x] Make `batch_set_v1(...)` copy ordinary L2 pages into the scratch slab
    before issuing `BatchPutIfAbsentFromRegion(...)`.
  - [x] Make `batch_get_v1(...)` issue `BatchGetIntoRegion(...)` into the
    scratch slab, then copy the filled bytes back into ordinary L2 pages.
  - [x] Remove GPU staging from the active Tensorcast backend path once the
    `HOST_SHARED` path is enabled; do not keep a runtime GPU-staging fallback in
    the same backend implementation.
- [x] Validate Phase A in SGLang end to end immediately after the integration
  switch:
  - [x] Run `share_local` in `share` mode.
  - [x] Run `share_local` in `separate` mode.
  - [x] Verify:
    - [x] functional `batch_exists / batch_get / batch_set`,
    - [x] real prefix reuse hits,
    - [x] no Tensorcast GPU staging allocation or H2D/D2H bounce remains on the
      active path.

### Phase 2.5B - Phase B allocator-backed zero-copy host residency

- [x] Freeze the Phase-2.5B contract:
  - [x] One allocator slot equals one HiCache KV page.
  - [x] One slot lifetime is guarded by a monotonically increasing
    `generation`.
  - [x] Phase B must reuse the same daemon-exported `HOST_SHARED` slab model as
    Phase A rather than introducing a second host-memory API family.
  - [x] Once Phase B becomes the active backend path, no GPU staging fallback is
    retained.
- [x] Implement the Phase-B slot-token and slot-lifecycle contract:
  - [x] Extend the region-backed request model so the caller can identify one
    slot lifetime, not only a byte window:
    - `region_id`
    - `memory_kind=HOST_SHARED`
    - `slot_index`
    - `slot_generation`
    - `offset_bytes`
    - `length_bytes`
  - [x] Update Tensorcast proto, layout validation, and response-mapping code so
    these slot-lifetime fields survive request decoding, target/source
    validation, and per-item result mapping.
  - [x] Keep the first safe rollout at one logical slot token per KV page even
    if adjacent slots are later coalesced for execution efficiency.
  - [x] Define and wire the caller-owned slot state machine:
    - `SlotFree`
    - `SlotReserved`
    - `GetInFlight`
    - `SlotResident`
    - `PutInFlight`
    - `SlotInvalid`
    - `SlotRetiring`
  - [x] Define pin or refcount rules so eviction cannot recycle a slot while it
    is reserved or in flight.
  - [x] Define `SlotInvalid` and retirement semantics for partial failure:
    - get-side fill failure invalidates the target slot,
    - ordinary put failure does not invalidate the local source slot,
    - generation bumps only after retirement and ref-drain.
- [x] Implement `M3` allocator-slab page-slot state / recycle / partial-failure
  correctness:
  - [x] Tensorcast-side plumbing:
    - [x] ensure host-region validation remains slot-token aware but does not
      create a daemon-owned allocator table,
    - [x] reject allocator-backed `HOST_SHARED` generic target validation so
      allocator slabs only flow through the byte-artifact region-backed path,
    - [x] preserve slot token metadata through `BatchGetIntoRegion` and
      `BatchPutIfAbsentFromRegion` result construction so SGLang can revalidate
      generation before making a page visible.
  - [x] Add Tensorcast unit/integration tests for:
    - [x] allocator-backed `HOST_SHARED` validation requires explicit offsets
      with complete `slot_index` and `slot_generation`,
    - [x] allocator-backed `HOST_SHARED` generic target validation fails closed,
    - [x] request-scoped `slot_generation` metadata is preserved across repeated
      slot reuse requests so Tensorcast does not echo a stale cached generation,
    - [x] partial batch failure,
    - [x] caller-owned generation lifetime remains external to Tensorcast; the
      daemon preserves request-scoped slot tokens but does not own allocator
      recycle state.
  - [x] Add SGLang-side groundwork tests for slot-tracker correctness where
    feasible:
    - pin or refcount state,
    - stale completion rejection,
    - retirement and reuse guards,
    - store-layer seam coverage that rejects stale `batch_get_v1(...)`
      completion before the fetched page becomes radix-visible.
- [x] Implement `M4` host-registration attach / detach / fallback:
  - [x] Add optional one-time `cudaHostRegister(...)` on the SGLang-local slab
    mapping.
  - [x] Keep Tensorcast host-region semantics independent from host pinning:
    the daemon-side `HOST_SHARED` region model must remain correct whether or
    not the local client successfully host-registers the mapping.
  - [x] Add the matching detach path:
    - drain in-flight slot refs,
    - retire resident or invalid slots,
    - `cudaHostUnregister(...)`,
    - unmap,
    - release slab lease.
  - [x] Define the fallback behavior when host registration is unavailable:
    correctness must remain intact and only the performance policy changes.
  - [x] Add `M4` test coverage:
    - allocator-side register/unregister lifecycle,
    - register failure fallback without breaking cleanup,
    - store-layer end-to-end regression with `HOST_SHARED` allocator path.
- [x] Add the SGLang Phase-B allocator integration:
  - [x] Implement a Tensorcast-aware `HostKVCache` allocator backed by one
    daemon-exported slab per rank.
  - [x] Make L2 host pages live directly in exported slab slots rather than in a
    separate ordinary host pool.
  - [x] Fail closed unless allocator-backed TensorCast host pools use the
    page-contiguous `page_blob_direct` layout and a live exported region
    binding.
  - [x] Make `batch_set_v1(...)` publish directly from resident slot offsets
    without an extra copy into a scratch slab.
  - [x] Make `batch_get_v1(...)` reserve destination slots and materialize
    directly into those slots.
  - [x] Insert fetched pages into radix-visible HiCache state only after get
    success and generation revalidation.
  - [x] Retire invalid slots correctly on failure.
- [x] Validate Phase B in SGLang end to end immediately after allocator
  integration:
  - [x] Rerun the same `share_local` benchmark shape used for Phase A.
  - [x] Compare against the last GPU-staging baseline and the Phase-A host
    staging path.
  - [x] Verify:
    - functionality remains correct,
    - prefix reuse still hits,
    - `TTFT`, `batch_get`, and `batch_set` metrics improve or at least behave as
      expected,
    - GPU staging is completely absent from the active Tensorcast backend path.

### Phase 2.5 Exit Criteria

- [x] Tensorcast has a daemon-managed `HOST_SHARED` slab lifecycle validated by
  milestone tests `M0` through `M4`.
- [x] `BatchPutIfAbsentFromRegion(...)` supports `HOST_SHARED` source layouts.
- [x] `BatchGetIntoRegion(...)` supports `HOST_SHARED` target layouts.
- [x] Phase A host staging runs end to end in `share_local` without GPU staging.
- [x] Phase B allocator-backed host residency runs end to end in `share_local`
  with slot-generation protection and no GPU staging fallback.

Current Phase-2.5A validation note:

- Tensorcast milestone coverage currently confirmed locally by:
  - `//core/store/materialization/dataplane:target_layout_host_sink_test`
  - `//daemon:byte_artifact_region_layout_host_shared_test`
  - `//daemon:grpc_service_impl_batch_runtime_test --test_arg=[host_shared]`
- SGLang end-to-end host-staging bring-up is confirmed in:
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260401-224749_tensorcast_tp2_pairs10`
    - `tensorcast-daemon-mode=separate`
    - `--wait-for-source-publication-drain`
    - `--source-publication-drain-idle-s 10`
    - `prompt_count=10`
    - `--hicache-storage-prefetch-policy wait_complete`
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260401-230828_tensorcast_tp2_pairs10`
    - `tensorcast-daemon-mode=share`
    - `--wait-for-source-publication-drain`
    - `--source-publication-drain-idle-s 10`
    - `prompt_count=10`
    - `--hicache-storage-prefetch-policy wait_complete`
- That run shows:
  - instance A successfully publishing through `batch_set_v1(...)`,
  - instance B reaching `batch_exists(...)` hits and successful
    `batch_get_v1(...)` on the `HOST_SHARED` path,
  - all 10 prompt pairs reaching storage-backed prefix hits on instance B
    without falling back to `batch_set_v1(...)` in both `share` and `separate`
    daemon modes,
  - the active SGLang Tensorcast backend using `region_ref.memory_kind=HOST_SHARED`
    with `device_uuid=""`, not the old `vram_region_id` staging path.
- Performance is not yet an exit criterion for Phase 2.5A bring-up:
  - mean TTFT was still worse than the GPU-staging baseline,
  - `host_fill_ms` remained significant because Phase A still performs one host
    copy between the Tensorcast scratch slab and ordinary HiCache L2 pages.

Current Phase-2.5B validation note:

- SGLang local allocator plumbing currently confirmed by:
  - `python/sglang/test/tensorcast/test_memory_pool_host_page_blob_direct.py`
  - `python/sglang/test/tensorcast/test_tensorcast_host_allocator.py`
  - `python/sglang/test/tensorcast/test_tensorcast_store.py`
  - `python/sglang/test/tensorcast/test_host_shared_slot_state.py`
- That coverage confirms:
  - the `page_blob_direct` host layout keeps one TensorCast-facing KV page in a
    page-contiguous blob shape,
  - TensorCast extra config is parsed before host-pool construction,
  - TensorCast allocator-backed `HOST_SHARED` slabs can back a `HostKVCache`
    allocation directly,
  - allocator-backed host-pool construction is wired without regressing the
    existing slot-token plumbing,
  - allocator-backed TensorCast registration now fails closed unless the host
    pool exposes a live binding on the page-contiguous `page_blob_direct`
    layout,
  - the active TensorCast page client now publishes `batch_set_v1(...)` directly
    from resident allocator slot offsets on the exported `HOST_SHARED` slab,
  - `batch_get_v1(...)` now reserves destination slots and materializes
    directly into those allocator-backed `HOST_SHARED` slots, and
  - releasing resident or invalid fetched pages now retires the slot token so
    later reuse bumps generation instead of re-entering with a stale slot
    lifetime.
- Partial Phase-2.5B end-to-end validation is confirmed in:
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260402-152750_tensorcast_tp2_pairs10`
    - `tensorcast-daemon-mode=separate`
    - `--wait-for-source-publication-drain`
    - `--source-publication-drain-idle-s 10`
    - `prompt_count=10`
    - `--hicache-storage-prefetch-policy wait_complete`
    - `--hicache-mem-layout page_blob_direct`
    - `--hicache-io-backend direct`
    - allocator-backed TensorCast host residency enabled
- That run shows:
  - instance A `batch_set_v1(...)` remains functionally correct with no failed
    pages,
  - source-side publication now runs with `stage_copy_ms=0.00`, confirming the
    scratch host-staging copy is removed from the active put path,
  - instance B still reaches `batch_exists(...)` hits and successful
    `batch_get_v1(...)` with positive cached-token reuse on the same long-prompt
    benchmark shape used for the Phase-A separate-mode baseline,
  - overall TTFT still regresses versus the best Phase-A host-staging baseline,
    which is consistent with `batch_get_v1(...)` still paying the ordinary
    scratch-region `host_fill_ms` copy on the fetch path.
- Updated Phase-2.5B direct-get validation is confirmed in:
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260402-160726_tensorcast_tp2_pairs10`
    - `tensorcast-daemon-mode=separate`
    - `--wait-for-source-publication-drain`
    - `--source-publication-drain-idle-s 10`
    - `prompt_count=10`
    - `--hicache-storage-prefetch-policy wait_complete`
    - `--hicache-mem-layout page_blob_direct`
    - `--hicache-io-backend direct`
    - allocator-backed TensorCast host residency enabled
- That run shows:
  - instance A `batch_set_v1(...)` remains functionally correct with no failed
    pages and still keeps `stage_copy_ms=0.00`,
  - instance B reaches full `batch_exists(...)` prefix hits and successful
    `batch_get_v1(...)` on the allocator-backed direct-slot path,
  - `batch_get_v1(...)` now reports `host_fill_ms=0.00`, confirming the old
    scratch host-staging copy is gone from the active fetch path,
  - mean instance-B TTFT improves by about `483.61 ms` relative to
    `20260402-152750_tensorcast_tp2_pairs10`,
  - mean TTFT improvement flips from `-363.26 ms` in
    `20260402-152750_tensorcast_tp2_pairs10` to `+82.69 ms`, and
  - mean TTFT improvement is about `1058.35 ms` better than the earlier
    Phase-A separate-mode host-staging baseline
    `20260401-224749_tensorcast_tp2_pairs10`.
- Updated Phase-2.5B share-mode validation is confirmed in:
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260402-203549_tensorcast_tp2_pairs10`
    - `tensorcast-daemon-mode=share`
    - `--wait-for-source-publication-drain`
    - `--source-publication-drain-idle-s 10`
    - `prompt_count=10`
    - `--hicache-storage-prefetch-policy wait_complete`
    - `--hicache-mem-layout page_blob_direct`
    - `--hicache-io-backend direct`
    - allocator-backed TensorCast host residency enabled
- That run shows:
  - all `10/10` prompt pairs succeed with real TensorCast-backed prefix reuse
    on instance B,
  - instance A `batch_set_v1(...)` remains successful with
    `stage_copy_ms=0.00`,
  - instance B `batch_get_v1(...)` stays on the allocator-backed direct-slot
    path with `host_fill_ms=0.00`,
  - instance B average cached tokens are about `16.4k`, confirming reuse on the
    same long-prompt benchmark shape, and
  - mean TTFT improvement is `+337.96 ms` with mean speedup ratio `1.24x` in
    the shared-daemon topology.
- Updated `M4` host-registration validation is confirmed in:
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260403-170738_tensorcast_tp2_pairs10`
    - `tensorcast-daemon-mode=separate`
    - `--wait-for-source-publication-drain`
    - `prompt_count=10`
    - `--hicache-storage-prefetch-policy wait_complete`
    - `--hicache-mem-layout page_blob_direct`
    - `--hicache-io-backend direct`
    - allocator-backed TensorCast host residency enabled
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260403-171931_tensorcast_tp2_pairs10`
    - `tensorcast-daemon-mode=share`
    - `--wait-for-source-publication-drain`
    - `prompt_count=10`
    - `--hicache-storage-prefetch-policy wait_complete`
    - `--hicache-mem-layout page_blob_direct`
    - `--hicache-io-backend direct`
    - allocator-backed TensorCast host residency enabled
- Those runs show:
  - each rank successfully performs one-time `cudaHostRegister(...)` on the
    local allocator-backed `HOST_SHARED` slab mapping,
  - `batch_get_v1(...)` remains on the allocator-backed direct-slot path with
    `host_fill_ms=0.00`, so `M4` does not reintroduce the old host copy-back
    path,
  - `share` mode remains healthy and shows `10/10` successful prefix-reuse
    pairs with mean TTFT improvement `+410.99 ms`, and
  - `separate` mode remains functionally correct on the same path.
- Post-Phase-2.5 publish-slowdown follow-up is confirmed in:
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260405-125504_tensorcast_tp2_pairs10`
    - `tensorcast-daemon-mode=separate`
    - `--wait-for-source-publication-drain`
    - `prompt_count=10`
    - `--hicache-storage-prefetch-policy wait_complete`
    - `--hicache-mem-layout page_blob_direct`
    - `--hicache-io-backend direct`
    - allocator-backed TensorCast host residency enabled
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260405-173854_tensorcast_tp2_pairs10`
    - `tensorcast-daemon-mode=share`
    - `--wait-for-source-publication-drain`
    - `prompt_count=10`
    - `--hicache-storage-prefetch-policy wait_complete`
    - `--hicache-mem-layout page_blob_direct`
    - `--hicache-io-backend direct`
    - allocator-backed TensorCast host residency enabled
- That follow-up confirms:
  - the `ReplicaRegistry::erase()` O(1) fix removes the source-side
    publication-time growth that previously appeared as prompt count increased,
  - in `separate` mode, instance-A `batch_set_v1(...)` no longer trends upward:
    mean batch time is about `363.90 ms`, median `364.65 ms`, max `942.01 ms`,
  - in `share` mode, instance-A `batch_set_v1(...)` also remains stable:
    mean batch time is about `190.68 ms`, median `196.36 ms`, max `569.92 ms`,
  - instance-B still reaches full `batch_exists(...)` hits and successful
    `batch_get_v1(...)` with positive cached-token reuse in both daemon modes,
    and
  - the store-layer regression tests
    `//core/store:replica_registry_test`,
    `//core/store:stable_dram_cache_manager_test`, and
    `//core/store:eviction_service_test`
    all pass after the registry fix.
- Post-refactor modularity validation is confirmed in:
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260407-152641_tensorcast_tp2_pairs10`
    - `tensorcast-daemon-mode=share`
    - `--pair-rps 0.5`
    - `--settle-ms 10000`
    - `--hicache-mem-layout page_blob_direct`
    - `--hicache-io-backend direct`
    - allocator-backed TensorCast host residency enabled
  - `benchmark/tensorcast_benchmark/kv/share_local/outputs/20260407-153305_tensorcast_tp2_pairs10`
    - `tensorcast-daemon-mode=separate`
    - `--pair-rps 0.5`
    - `--settle-ms 10000`
    - `--hicache-mem-layout page_blob_direct`
    - `--hicache-io-backend direct`
    - allocator-backed TensorCast host residency enabled
- Those runs show:
  - the post-sweep SGLang-side refactor keeps `10/10` successful prompt pairs
    in both daemon modes,
  - instance B still reaches real prefix reuse with about `16.4k` average
    cached tokens in both runs,
  - `share` mode keeps a positive mean TTFT improvement (`+552.78 ms`), and
  - `separate` mode remains functionally correct with batch `exists/get/set`
    behavior in the same order of magnitude as the April 6 baselines.

## Phase 3 - Request Transfer Interface

This phase adds the programmable request-level handoff path for PD-disaggregated
inference.

Validation cadence for this phase should also be milestone-gated rather than
left to one final bring-up:

- the first layer is Tensorcast API / proto / node-agent unit testing for the
  request-transfer surface,
- the second layer is SGLang pure-Python state-machine and coordinator testing
  for publish / hydrate / claim / fallback semantics,
- the third layer is controller-driven end-to-end validation on real SGLang
  instances after the individual milestones are already green.

Recommended Phase-3 milestones:

- `M0`
  - **What it is**:
    Tensorcast request-transfer control-plane surface roundtrip.
  - **What must be true when it is complete**:
    `PublishManifest`, `EngineOwnedManifest`, `PublishResult.publish_manifest`,
    and `hydrate(publish_manifest=...)` serialize, deserialize, and execute
    through plan / node-agent plumbing without ambiguity.
  - **What it does not include yet**:
    real SGLang publish / hydrate semantics.
- `M1`
  - **What it is**:
    SGLang-local source and target state models.
  - **What must be true when it is complete**:
    `RequestBundleState`, page publication registry, prepared-bundle table, and
    prepared-hold bookkeeping all have explicit state machines with pure-Python
    tests.
  - **What it does not include yet**:
    real Tensorcast publish / hydrate calls.
- `M2`
  - **What it is**:
    rank-0 coordinator fanout and aggregation semantics.
  - **What must be true when it is complete**:
    one logical instance op fans out to required TP ranks, aggregates results,
    and fails closed on missing or failed required ranks.
  - **What it does not include yet**:
    end-to-end artifact materialization.
- `M3`
  - **What it is**:
    source-side `publish(...)` closure correctness.
  - **What must be true when it is complete**:
    publish can resolve active or retained source request state, freeze a
    full-prompt page-granular closure target, adopt/wait/force missing pages
    up to that boundary, and assemble one immutable `PublishManifest` while
    surfacing any non-page-aligned prompt tail via `tail_valid_tokens`.
  - **What it does not include yet**:
    target-side prepared-bundle admission.
- `M4`
  - **What it is**:
    target-side `hydrate(...)` install correctness.
  - **What must be true when it is complete**:
    hydrate validates compatibility, materializes required pages, installs one
    prepared bundle plus holds, and cleans up correctly on required-rank
    failure.
  - **What it does not include yet**:
    ordinary `/generate` claim / revalidation.
- `M5`
  - **What it is**:
    ordinary `/generate(rid)` admission binding correctness.
  - **What must be true when it is complete**:
    incoming prompt tokenization is revalidated against
    `prompt_token_digest` / `cutoff_token_count`; clean prepared bundles are
    claimed exactly once; stale or incompatible records warn and fall back.
  - **What it does not include yet**:
    remote end-to-end controller validation on real instances.
- `M6`
  - **What it is**:
    single-controller end-to-end prompt-only request-bundle handoff.
  - **What must be true when it is complete**:
    a controller can serve on source, publish a full-prompt page-granular
    snapshot, optionally prefetch, hydrate on target, and resume through
    ordinary SGLang ingress.

### Phase 3A - SGLang Instance-Agent and Coordinator

Current status:

- `M1` state models and request-bundle bookkeeping now live under
  `sglang/python/sglang/srt/tensorcast/request_bundle/`, with the store-facing
  bridge remaining in
  `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/`.
- `M2` runtime coordinator, directory registration, rank-0 fanout /
  aggregation, and the instance-agent runtime now live under
  `sglang/python/sglang/srt/tensorcast/instance_ops/`.
- `M3` has now been split into:
  - local-rank publish closure on each scheduler process,
  - plus a separate group-level publish-manifest aggregation step that combines
    one local result per required rank.
- `M4` has now been split into:
  - local-rank hydrate install and prepared-bundle state on each scheduler
    process,
  - plus a separate group-level aggregation step that combines one local
    hydrate result per required rank.
- `M3`, `M4`, `M5`, and `evict_local(...)` runtime semantics now live under
  `sglang/python/sglang/srt/tensorcast/request_bundle/` and are wired into the
  real backend through `RequestBundleManager`.
- Live ordinary text `/generate` prepared-bundle claim / cleanup wiring is
  connected to the real `Scheduler` lifecycle for the Tensorcast HiCache
  backend.
- Local-rank live source-request tracking / progress observation / cleanup is
  connected to the real `Scheduler` lifecycle for ordinary text `/generate`.
- Local-rank page publication state transitions are now driven by real
  `batch_set_v1(...)` outcomes on the active Tensorcast store path, so the page
  publication registry reflects background publication progress instead of only
  pure-Python test fixtures.
- Local-rank store state intentionally remains per-scheduler-process, but rank
  0 now performs real cross-rank `publish` / `hydrate` / `evict_local`
  fanout-and-aggregation over the live scheduler control path.
- The current coordinator runtime reuses SGLang's existing collective RPC /
  object-group control path and is explicitly limited to `dp_size == 1`.
- Local `hydrate(...)` now materializes host-resident prepared pages plus
  prepared-hold state on each scheduler process, and ordinary `/generate(rid)`
  consumes that prepared state by installing a host prefix into the normal
  HiRadix path before decode.
- A rank-0 `launch_server()`-managed Tensorcast instance-agent sidecar gRPC
  ingress is now wired for the Tensorcast HiCache backend.
- That ingress reuses a dedicated `instance_agent_execution_endpoint`,
  forwards only
  `publish` / `hydrate` / `evict_local` through the existing
  scheduler `rpc_ipc_name`, and preserves the scheduler-owned coordinator
  semantics.
- Remaining Phase 3 gaps are now remote validation and true multi-node
  external-caller execution hardening, not the instance-agent ingress boundary
  itself.

- [x] Add the SGLang-side instance-agent boundary:
  - [x] Implement Tensorcast `NodeAgent` semantics as a `launch_server()`-managed
    instance-agent sidecar for the logical SGLang instance.
  - [x] Co-locate the SGLang `EngineAdapter` with that instance-agent service.
  - [x] Keep the Tensorcast daemon as worker/data-plane host, not as owner of instance-step execution.
  - [x] Add typed SGLang-internal instance-op request / result objects for
    `publish`, `hydrate`, and `evict_local`.
  - [x] Add pure-Python tests for those typed request / result objects and
    per-rank result aggregation.
- [x] Add the rank-0 coordinator role:
  - [x] Rank 0 owns the logical `instance_id` / coordinator-epoch runtime lifecycle.
  - [x] Rank 0 owns Tensorcast directory registration for the instance-agent sidecar execution endpoint.
  - [x] Rank 0 receives `publish`, `hydrate`, and `evict_local` calls for the logical instance.
  - [x] Rank 0 fans out to the required TP ranks and aggregates success/failure.
  - [x] Group-scoped success requires all required ranks to succeed.
  - [x] Make required-rank timeouts and missing-rank handling explicit and fail-closed.
  - [x] Reuse the real scheduler collective RPC / object-group path rather than a test-only coordinator fanout shim.
  - [x] Add coordinator unit tests for all-success, one-rank-failure,
    timeout, and idempotent retry cases.
- [x] Add SGLang-owned request bundle bookkeeping:
  - [x] Keep request-bundle metadata inside SGLang, not in a Tensorcast registry.
  - [x] Add a page publication registry inside SGLang integration code.
  - [x] Add request-bundle snapshot state needed to decide publish cutoff and ordered page membership.
  - [x] Keep the snapshot logic non-blocking to engine execution except where an explicit publish must wait for pending page publication to reach the requested cutoff.
  - [x] Explicitly model page publication states needed by publish closure:
    `ready`, `inflight`, `absent`, `failed`.
  - [x] Explicitly model page-aligned cutoff clipping and `snapshot_seq`
    advancement.
  - [x] Connect local-rank live source request tracking / progress / cleanup to
    the real `Scheduler` lifecycle for ordinary text `/generate`.
  - [x] Reflect passive background `batch_set_v1(...)` publication outcomes
    into the local-rank page publication registry so `ready`, `inflight`, and
    retryable `absent` transitions come from the real store path.
  - [x] Add pure-Python tests for page publication registry transitions,
    request-bundle state transitions, page-aligned cutoff clipping, and publish
    retry generation advancement.
  - [x] Add store-wiring tests for background `batch_set_v1(...)` success,
    per-item failure rollback, and publish-refresh preservation of inflight
    page state.
- [x] Define decode-side prepared state:
  - [x] Hydrate prepares a target-local request bundle.
  - [x] Ordinary decode ingress later consumes that prepared state.
  - [x] Hydrate itself must not silently start decode.
  - [x] Store prepared-bundle state separately from live request state.
  - [x] Store prepared-hold references separately from HiCache prefix-match hot
    state.
  - [x] Add pure-Python tests for one-shot claim, stale prepared record
    fallback, and live-request conflict handling.

### Phase 3B - Tensorcast Request-Transfer Surface Extensions

- [x] Extend Tensorcast result/control types:
  - [x] Add `EngineOwnedManifest` as an opaque engine-owned payload.
  - [x] Add `PublishManifest` as the controller-visible immutable transfer handle.
  - [x] Extend `PublishResult` to carry `publish_manifest`.
  - [x] Make manifest digest binding and legacy alias behavior explicit.
- [x] Extend plan and node-agent APIs:
  - [x] Add `hydrate(publish_manifest=...)`.
  - [x] Preserve legacy `hydrate(engine_request_id=...)` only as a compatibility surface for controller-side cached manifests.
  - [x] Ensure the controller, not the target instance, owns any compatibility cache from `engine_request_id` to the last known `PublishManifest`.
  - [x] Define fail-closed behavior for zero or multiple cached manifest matches
    in the compatibility shim.
- [x] Extend proto serialization and Python APIs:
  - [x] Update plan proto.
  - [x] Update node-agent proto.
  - [x] Update Python plan builders and result deserialization.
  - [x] Update executor/server serialization paths.
  - [x] Update or add Tensorcast tests for:
    - [x] plan-spec roundtrip of `PublishManifest`,
    - [x] node-agent execution of `hydrate(publish_manifest=...)`,
    - [x] compatibility-shim failure semantics for zero / one / many cached
      manifest matches,
    - [x] opaque `EngineOwnedManifest` payload passthrough.
- [x] Add Tensorcast tests for the new request-transfer handle semantics.

### Phase 3C - SGLang EngineAdapter Publish/Hydrate/Evict

- [x] Implement source-side `publish(...)`:
  - [x] Resolve source request state from SGLang-owned request metadata.
  - [x] Resolve retained prompt-snapshot state after the ordinary source
    request completes, so post-completion `publish()` stays possible within a
    bounded retention window.
  - [x] Reject unsupported v1 shapes early:
    - [x] no batch request transfer,
    - [x] no parallel-sampling transfer,
    - [x] no session-lineage transfer.
  - [x] Allow `publish()` to be invoked after the source has already emitted
    decode tokens while keeping snapshot membership prompt-only.
  - [x] Ensure `publish()` targets the request's full prompt token count and
    only reports success after page-granular closure for that boundary.
  - [x] Allow bounded waiting for prompt prefill / page publication needed to
    close that full prompt boundary, while still refusing to chase decode
    continuation or any newer prompt mutation.
  - [x] For the initial SGLang v1 profile, keep the byte-artifact closure
    page-granular while carrying any non-page-aligned prompt tail through
    `tail_valid_tokens` instead of directly transferring a partial tail page.
  - [x] Keep the published generation prompt-only:
    - [x] emitted decode tokens do not extend cutoff/page membership,
    - [x] decode-only KV is excluded from the published generation.
  - [x] Gather required-rank ordered page membership for exactly that cutoff.
  - [x] Close the cutoff over the page publication registry:
    - [x] reuse `ready` pages,
    - [x] join or wait `inflight` pages,
    - [x] force flush `absent` pages if local bytes still exist,
    - [x] fail closed on `failed` or unrecoverable missing pages.
  - [x] Reuse already-published substrate pages when possible rather than re-uploading every page.
  - [x] Assemble one `PublishManifest` containing:
    - [x] generic artifact-manifest data,
    - [x] compatibility envelope,
    - [x] `prompt_token_digest`,
    - [x] `cutoff_token_count`,
    - [x] `tail_valid_tokens` for non-page-aligned prompt tails while aligned
      prompts still emit `0`.
  - [x] Return a `PublishManifest` containing:
    - generic Tensorcast manifest data,
    - opaque `EngineOwnedManifest` needed by SGLang to resume decode.
  - [x] Keep publish execution split into:
    - one local-rank closure result per scheduler process,
    - one later group-level aggregation step that combines all required ranks
      into the controller-visible `PublishManifest`.
  - [x] Add unit tests for:
    - [x] ready-only publish,
    - [x] wait-on-inflight publish,
    - [x] force-flush missing tail publish,
    - [x] full-prompt closure waits for prompt-page publication rather than
      committing a smaller visible-prefix generation,
    - [x] non-page-aligned prompt tail is surfaced via `tail_valid_tokens`
      without direct tail-page transfer,
    - [x] live-request released before closure failure,
    - [x] group-level aggregation across one local publish result per rank,
    - [x] mid-decode prompt-only publish,
    - [x] post-completion retained prompt-snapshot publish.
- [x] Implement target-side `hydrate(...)`:
  - [x] Accept `publish_manifest=...` as the canonical v1 path.
  - [x] Validate manifest compatibility before any destructive local install.
  - [x] Materialize the request bundle into target-local prepared state.
  - [x] Keep v1 hydrate distinct from SGLang PD-disaggregation prebuilt decode semantics; hydrate prepares state but does not directly start decode.
  - [x] Separate transport success from decode-usability filtering.
  - [x] Only after successful prepare should the ordinary decode request be admitted on the target instance.
  - [x] Ordinary `/generate(rid)` must re-run incoming prompt tokenization / normalization and verify `prompt_token_digest` plus `cutoff_token_count` before claiming a prepared bundle.
  - [x] Install prepared-hold references keyed by slot token / generation rather
    than by page start alone.
  - [x] Make repeated `hydrate()` with the same manifest an explicit idempotent
    retry policy.
  - [x] Keep hydrate execution split into:
    - one local-rank install result plus local prepared-bundle state per
      scheduler process,
    - one later group-level aggregation step that combines all required ranks
      into the fail-closed controller-visible hydrate result.
  - [x] Add unit tests for:
    - [x] compatibility mismatch fail-closed,
    - [x] local-rank hydrate failure,
    - [x] successful local prepared-bundle install,
    - [x] idempotent retry with the same manifest,
    - [x] group-level aggregation across one local hydrate result per rank.
- [x] Implement ordinary `/generate(rid)` prepared-bundle binding:
  - [x] Re-run the same prompt tokenization / normalization as ordinary SGLang
    admission.
  - [x] Validate `prompt_token_digest` and `cutoff_token_count` before claim.
  - [x] CAS-claim exactly one clean prepared bundle generation.
  - [x] Warning + fall back to the normal SGLang path for stale, tainted, or
    incompatible prepared records.
  - [x] Fail closed for active claim conflicts or multiple clean prepared
    generations.
  - [x] Add unit tests for:
    - [x] exact-match successful claim,
    - [x] prompt mismatch fallback,
    - [x] stale-record fallback,
    - [x] claimed-conflict fail-closed,
    - [x] multiple-clean-generation fail-closed.
- [x] Implement `evict_local(...)`:
  - [x] Remove target-local prepared/live request bundle state.
  - [x] Keep local evict semantics distinct from global page deletion.
  - [x] Release prepared holds and clean any target-local bookkeeping without
    deleting shared page artifacts.
  - [x] Add unit tests for prepared-only cleanup and post-consume cleanup.
- [x] Add optional worker warmup:
  - [x] Use `prefetch_manifest_result(...)` only as an optimization.
  - [x] Keep `prefetch` orthogonal to correctness of `hydrate`.
  - [x] Add a test showing that `hydrate()` still succeeds without worker
    prefetch.

### Phase 3D - Caller / Controller Validation

Current status:

- local controller/runtime validation in Tensorcast Python tests is green for:
  - manifest-cache happy path,
  - compatibility-shim zero / one / many match behavior,
  - publish failure surfacing,
  - hydrate failure surfacing.
- a realistic external caller benchmark scaffold now exists under
  `benchmark/tensorcast_benchmark/kv/request_transfer/` with:
  - one-runtime controller flow,
  - local and remote topology orchestration,
  - source ordinary `/generate`,
  - controller `publish`,
  - optional worker warmup,
  - controller `hydrate`,
  - and target ordinary `/generate` verification hooks.
- the latest local one-prompt correctness run now reaches:
  - source ordinary `/generate` success,
  - controller-side `publish` success for the full prompt page-granular
    closure,
  - controller-side `hydrate` success on the target instance,
  - target ordinary `/generate(rid)` success,
  - prepared-bundle attach in target logs on all required ranks,
  - and `prepared_bundle_verified` green with
    `cached_tokens == cutoff_token_count - tail_valid_tokens`.
- the latest remote one-prompt correctness run
  (`20260412-144250_request_transfer_tp2_pairs1`) now reaches:
  - `topology_mode=remote` with source and target on distinct workers,
  - source ordinary `/generate` success,
  - controller-side `publish` success for the full prompt page-granular
    closure,
  - controller-side `hydrate` success on the target instance,
  - target ordinary `/generate(rid)` success,
  - prepared-bundle attach in target logs on all required ranks,
  - no prepared-bundle fallback / fail-closed / consume-failed markers in the
    target log, and
  - `prepared_bundle_verified` green with
    `cached_tokens == cutoff_token_count - tail_valid_tokens`.

- [x] Add a realistic external caller example for controller-driven
  prompt-only request-bundle reuse:
  - [x] choose source and target instances,
  - [x] serve on source,
  - [x] publish the full prompt page-granular closure on source,
  - [x] optionally prefetch on target host daemon,
  - [x] hydrate on target,
  - [x] resume decode on target.
- [x] Add local integration tests for single-controller request transfer:
  - [x] controller-side manifest cache happy path,
  - [x] controller-side compatibility shim zero / one / many match behavior,
  - [x] controller-visible source publish failure and target hydrate failure
    surfacing.
- [x] Add remote end-to-end validation for `M6`:
  - [x] source ordinary serving instance prefill,
  - [x] controller-side publish,
  - [x] target ordinary serving instance hydrate,
  - [x] target ordinary `/generate` claim via the same stable `rid`,
  - [x] log-based verification that ordinary `/generate` used the prepared
    bundle rather than ordinary prefix prefetch.
- [x] Keep the v1 restriction explicit:
  - [x] single controller,
  - [x] explicit `PublishManifest`,
  - [x] no generic cross-daemon multi-instance super-plan beyond current Tensorcast routing limits.
  - [x] require one explicit stable caller `rid` per transferred request.
  - [x] reject batch-request, parallel-sampling, and session-lineage handoff in the initial profile.
  - [x] keep the published generation at the request's prompt-only boundary rather than decode-token continuation.
  - [x] allow `publish()` invocation after source-side decode has started or after the ordinary source request completes, as long as only prompt-only KV is published.
  - [x] leave DP-replica routing for the post-hydrate ordinary `/generate` out of scope for the initial implementation.
  - [x] leave decode-only target instances out of scope for the initial implementation; v1 first targets ordinary serving instances that run the first decode step locally.
  - [x] ordinary `/generate(rid)` should claim a clean prepared bundle when available, but warning + fall back to the normal SGLang path if only stale/tainted/incompatible prepared records exist.

### Phase 3 Exit Criteria

- [x] Milestones `M0` through `M5` each have self-contained unit or local
  integration coverage and are green before `M6` remote validation.
- [x] A controller can publish one request's full prompt, page-granular,
  prompt-only snapshot from a source instance.
- [x] A controller can hydrate that snapshot on a target instance using
  `PublishManifest`.
- [x] The target instance can reuse that published prompt-only snapshot through
  ordinary SGLang ingress without rerunning prefill for the closed pages.
- [x] Prepared-bundle claim revalidates incoming prompt tokenization against
  `prompt_token_digest` and `cutoff_token_count`.
- [x] The integration keeps request-bundle metadata owned by SGLang, not Tensorcast core.

## Phase 4 - Hardening, Failure Semantics, and Performance

- [ ] Failure semantics:
  - [ ] define retryable vs terminal publish/hydrate failures,
  - [ ] define partial page availability behavior clearly,
  - [ ] keep controller-visible errors actionable.
- [ ] Lease and retention tuning:
  - [ ] validate durable-but-evictable page retention policy under storage pressure,
  - [ ] revisit explicit TTL/retain semantics only after v1 correctness is stable.
- [ ] Observability:
  - [ ] add end-to-end publish/hydrate metrics,
  - [ ] add benchmark-visible counters for prefix-share hit depth and request-transfer page counts,
  - [ ] add coordinator lifecycle visibility.
- [ ] Performance work:
  - [ ] measure whether explicit worker prefetch helps hydrate latency,
  - [ ] decide whether deeper CUDA IPC / zero-copy fast paths are needed after the host/L2-facing baseline works,
  - [ ] benchmark `share` vs `separate` daemon topology on the same node.

## Code Map

The most important implementation boundary is:

- `Phase 1` stayed primarily in the SGLang repo.
- `Phase 1.5` required Tensorcast daemon/runtime work for byte-artifact batch
  transport and performance, in addition to the SGLang-side backend.
- `Phase 2.5` extends that same integration boundary into Tensorcast
  region-backed host memory, slab lifecycle, and SGLang host-allocator wiring.
- `Phase 3` is where Tensorcast programmability-facing request-transfer API
  changes still become necessary.

### SGLang files to modify

- `sglang/docs/tensorcast/tensorcast_kv_protocol.md`
- `sglang/docs/tensorcast/tensorcast_kv_integration.md`
- `sglang/docs/tensorcast/plan/tensorcast_kv_integration_plan.md`
- `sglang/python/sglang/srt/server_args.py`
- `sglang/python/sglang/srt/mem_cache/storage/backend_factory.py`
- `sglang/python/sglang/srt/mem_cache/storage/__init__.py`
- `sglang/python/sglang/srt/mem_cache/hicache_storage.py`
- `sglang/python/sglang/srt/mem_cache/hiradix_cache.py`
- `sglang/python/sglang/srt/managers/cache_controller.py`
- `sglang/python/sglang/srt/managers/io_struct.py`
- `sglang/python/sglang/srt/entrypoints/http_server.py`
- `sglang/python/sglang/srt/managers/scheduler.py`
- `sglang/python/sglang/launch_server.py`
- `sglang/benchmark/tensorcast_benchmark/kv/models.py`
- `sglang/benchmark/tensorcast_benchmark/kv/outputs.py`
- `sglang/benchmark/tensorcast_benchmark/kv/remote.py`
- `sglang/benchmark/tensorcast_benchmark/kv/sgl_client.py`
- `sglang/benchmark/tensorcast_benchmark/kv/share_local/run_benchmark.py`
- `sglang/benchmark/tensorcast_benchmark/kv/share_local/request_driver.py`
- `sglang/benchmark/tensorcast_benchmark/kv/share_local/README.md`
- `sglang/benchmark/tensorcast_benchmark/kv/share_local/arch.md`
- `sglang/benchmark/tensorcast_benchmark/kv/share_local/configs/global_store_config.yaml`
- `sglang/benchmark/tensorcast_benchmark/kv/share_local/configs/store_daemon_config.yaml`

### SGLang files already added for Phase 1 / Phase 1.5 / Phase 2 / Phase 2.5

- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/__init__.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/tensorcast_store.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/config.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/client.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/host_allocator.py`
- `sglang/python/sglang/srt/mem_cache/host_shared_slot_state.py`
- `sglang/python/sglang/test/tensorcast/test_tensorcast_store.py`
- `sglang/python/sglang/test/tensorcast/test_tensorcast_host_allocator.py`
- `sglang/python/sglang/test/tensorcast/test_memory_pool_host_page_blob_direct.py`
- `sglang/python/sglang/test/tensorcast/test_host_shared_slot_state.py`
- `sglang/benchmark/tensorcast_benchmark/kv/share_local/scripts/tensorcast_service.sh`

### Phase 3 SGLang file map

Already added for `M1` groundwork:

- `sglang/python/sglang/srt/tensorcast/request_bundle/request_bundle_types.py`
- `sglang/python/sglang/srt/tensorcast/request_bundle/request_bundle_state.py`
- `sglang/python/sglang/srt/tensorcast/request_bundle/bundle_manager.py`
- `sglang/python/sglang/srt/tensorcast/instance_ops/instance_ops_types.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/tensorcast_store.py`

Already added for `M2`:

- `sglang/python/sglang/srt/tensorcast/instance_ops/instance_ops_coordinator.py`
- `sglang/python/sglang/srt/tensorcast/instance_ops/instance_directory.py`
- `sglang/python/sglang/srt/tensorcast/instance_ops/instance_ops_runtime.py`
- `sglang/python/sglang/srt/tensorcast/instance_ops/instance_agent.py`
- `sglang/python/sglang/srt/tensorcast/instance_ops/instance_agent_service.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/config.py`
- `sglang/python/sglang/srt/managers/scheduler.py`
- `sglang/python/sglang/srt/managers/scheduler_tensorcast_instance_ops_mixin.py`
- `sglang/python/sglang/srt/managers/io_struct.py`
- `sglang/python/sglang/srt/entrypoints/engine.py`
- `sglang/python/sglang/srt/entrypoints/http_server.py`

Already added for `M3`:

- `sglang/python/sglang/srt/tensorcast/request_bundle/request_bundle_publish.py`

Already added for `M4`:

- `sglang/python/sglang/srt/tensorcast/request_bundle/request_bundle_hydrate.py`

Already added for `M5`:

- `sglang/python/sglang/srt/tensorcast/request_bundle/prepared_bundle_admission.py`

Already added for `evict_local(...)`:

- `sglang/python/sglang/srt/tensorcast/request_bundle/prepared_bundle_evict.py`

Remaining SGLang-side Phase-3 work is expected to land mainly in:

- any narrow runtime glue discovered during remote `M6` validation.

### Phase 3 SGLang test map

Already added for `M1`:

- `sglang/python/sglang/test/tensorcast/test_request_bundle_state.py`
- `sglang/python/sglang/test/tensorcast/test_tensorcast_store.py`
- `sglang/python/sglang/test/tensorcast/test_request_bundle_manager.py`

Already added for `M2`:

- `sglang/python/sglang/test/tensorcast/test_instance_ops_coordinator.py`
- `sglang/python/sglang/test/tensorcast/test_instance_directory.py`
- `sglang/python/sglang/test/tensorcast/test_instance_ops_runtime.py`
- `sglang/python/sglang/test/tensorcast/test_instance_ops_scheduler_mixin.py`
- `sglang/python/sglang/test/tensorcast/test_instance_agent.py`

Already added for `M3`:

- `sglang/python/sglang/test/tensorcast/test_request_bundle_publish.py`

Already added for `M4`:

- `sglang/python/sglang/test/tensorcast/test_request_bundle_hydrate.py`

Already added for `M5`:

- `sglang/python/sglang/test/tensorcast/test_prepared_bundle_admission.py`

Already added for `evict_local(...)`:

- `sglang/python/sglang/test/tensorcast/test_prepared_bundle_evict.py`

Additional local integration coverage already added:

- the existing files above now also cover local manager, scheduler-mixin, and
  instance-agent integration paths rather than only isolated state machines.

Remaining Phase-3 validation is remote `M6` end-to-end verification plus any
runtime hardening it uncovers; there is no known missing standalone unit-test
module at this point.

### Tensorcast files to modify

- `Phase 2.5` is expected to touch Tensorcast daemon/runtime code for
  region-backed `HOST_SHARED` placement in addition to the SGLang integration.

These are split into two categories:

- Phase 1.5 / Phase 2 already required Tensorcast daemon/runtime changes for
  byte-artifact batch transport and performance.
- Phase 3 is where request-transfer programmability APIs still need to change.

Representative Tensorcast Phase-1.5 / Phase-2 areas are:

- `tensorcast/proto/daemon/v2/store_daemon.proto`
- `tensorcast/daemon/service/controllers/byte_artifact_controller.cc`
- `tensorcast/daemon/service/payload_transport/payload_transport_broker.cc`
- `tensorcast/core/store/runtime/ingestion/`
- `tensorcast/core/store/materialization/dataplane/sources/`
- `tensorcast/daemon/service/routing/`

Representative Tensorcast Phase-2.5 areas are:

- Proto and Python client surface:
  - `tensorcast/proto/tensorcast/daemon/v2/store_daemon.proto`
  - `tensorcast/tensorcast/daemon_ctl.py`
  - `tensorcast/tensorcast/api/store/__init__.py`
- Region registry and lifecycle:
  - `tensorcast/daemon/state/ipc_region_registry.h`
  - `tensorcast/daemon/state/ipc_region_registry.cc`
  - `tensorcast/daemon/service/grpc_service_impl.cc`
  - `tensorcast/daemon/service/grpc_service_impl.h`
- Local region validation and lowering:
  - `tensorcast/daemon/service/controllers/external_target_access_service.h`
  - `tensorcast/daemon/service/controllers/external_target_access_service.cc`
  - `tensorcast/daemon/service/controllers/materialization_target_storage_utils.h`
  - `tensorcast/daemon/service/controllers/materialization_target_storage_utils.cc`
  - `tensorcast/daemon/service/byte_artifact_region_layout.h`
  - `tensorcast/daemon/service/byte_artifact_region_layout.cc`
- Byte-artifact region ingress or egress:
  - `tensorcast/daemon/service/controllers/byte_artifact_controller.h`
  - `tensorcast/daemon/service/controllers/byte_artifact_controller.cc`
- Existing CPU-target dataplane pieces likely to be reused or extended:
  - `tensorcast/core/store/materialization/dataplane/sinks/cpu_va_sink.h`
  - `tensorcast/core/store/materialization/dataplane/sinks/cpu_va_sink.cc`
  - `tensorcast/core/store/materialization/dataplane/sources/remote_key_source.h`
  - `tensorcast/core/store/materialization/dataplane/sources/remote_key_source.cc`
- Tensorcast tests that should gain Phase-2.5 coverage:
  - `tensorcast/daemon/state/ipc_region_registry_test.cc`
  - `tensorcast/daemon/service/grpc_service_impl_batch_runtime_test.cc`
  - `tensorcast/daemon/service/grpc_service_impl_batch_redirect_e2e_test.cc`
  - `tensorcast/tests/python/test_store_region_registration.py`

Phase-3 request-transfer areas remain:

- `tensorcast/proto/tensorcast/plan/v1/plan.proto`
- `tensorcast/proto/tensorcast/node_agent/v1/node_agent.proto`
- `tensorcast/tensorcast/api/plan/plan.py`
- `tensorcast/tensorcast/engine_adapter/artifact_api.py`
- `tensorcast/tensorcast/engine_adapter/adapter.py`
- `tensorcast/tensorcast/node_agent/executor.py`
- `tensorcast/tensorcast/node_agent/server.py`
- `tensorcast/tests/python/test_kvcache_adapter.py`
- `tensorcast/tests/python/node_agent/test_plan_execution.py`
- `tensorcast/tests/python/api/test_plan_spec.py`
- `tensorcast/tools/build_proto_python.sh`

Tensorcast tests that should gain Phase-3 coverage:

- `tensorcast/tests/python/api/test_plan_spec.py`
  - `PublishManifest` / `hydrate(publish_manifest=...)` plan roundtrip
- `tensorcast/tests/python/node_agent/test_plan_execution.py`
  - explicit-handle hydrate execution and compatibility-shim failures
- `tensorcast/tests/python/test_kvcache_adapter.py`
  - opaque `EngineOwnedManifest` passthrough at the adapter boundary

### Optional Tensorcast files that may need targeted fixes

- `tensorcast/tensorcast/api/runtime.py`
- `tensorcast/tensorcast/api/directory.py`
- `tensorcast/tensorcast/runtime.py`
- `tensorcast/tensorcast/client_runtime.py`

These should only be touched if the SGLang request-transfer implementation
exposes a concrete routing or runtime gap that cannot be solved in SGLang's
integration layer.
