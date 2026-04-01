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
  API additions and in-process instance-agent work.

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

- [ ] Keep documentation aligned before code lands:
  - [ ] Treat `tensorcast_kv_protocol.md` as the external contract.
  - [ ] Treat `tensorcast_kv_integration.md` as the internal design source of truth.
  - [ ] Keep this plan file updated as implementation decisions change.
- [ ] Freeze phase boundaries:
  - [ ] `Phase 1` must not require Tensorcast plan / instance-step changes.
  - [ ] `Phase 1.5` must preserve the Phase-1 external substrate contract while replacing the hot-path data plane.
  - [ ] `Phase 2` must not require an external controller or request handoff.
  - [ ] `Phase 2.5` must stay within the existing prefix-share hot path while replacing the local SGLang <-> daemon memory boundary.
  - [ ] `Phase 3` is the only phase allowed to extend Tensorcast request-transfer semantics.
- [ ] Freeze the Phase-2 success criterion:
  - [ ] `benchmark/tensorcast_benchmark/kv/share_local` must run with
    `hicache-storage-backend=tensorcast`.
  - [ ] The benchmark must start Tensorcast services automatically.
  - [ ] The benchmark must preserve the same output contract as the Mooncake path.
  - [ ] The target first validation environment is `charged-group=codesign`.
- [ ] Preserve non-goals for early phases:
  - [ ] No PD-prefill/decode split in `share_local`.
  - [ ] No request-level `publish()` / `hydrate()` in `share_local`.
  - [ ] No Tensorcast-core semantic changes for prefix-share hot path behavior.

## Phase 1 - Shared KV Substrate

This phase creates the smallest useful unit: a Tensorcast-backed HiCache
storage backend that can publish and retrieve page data correctly.

- [ ] Add a built-in SGLang HiCache backend named `tensorcast`:
  - [x] Extend `--hicache-storage-backend` validation to include `tensorcast`.
  - [x] Register a Tensorcast backend in `StorageBackendFactory`.
  - [x] Add a dedicated backend package under
    `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/`.
- [ ] Define the SGLang-side Tensorcast backend config surface:
  - [x] Parse backend extra config from `--hicache-storage-backend-extra-config`.
  - [x] Include daemon/global-store addresses, local daemon association, durable policy,
    and any benchmark-only overrides.
  - [x] Validate missing or inconsistent fields with clear errors.
- [ ] Implement page identity and namespacing rules:
  - [x] Use SGLang page hash / prefix-chain-derived identity as the logical page key.
  - [x] Add TP-rank namespacing for MHA so different TP shards do not collide.
  - [x] Preserve MLA ownership rules instead of blindly duplicating all ranks.
  - [x] Keep page identity shared between prefix share and future request transfer.
- [ ] Implement Tensorcast-backed page operations:
  - [x] `batch_exists(...)`
  - [x] `batch_get_v1(...)`
  - [x] `batch_set_v1(...)`
  - [x] Map each SGLang page to a Tensorcast byte artifact payload boundary.
  - [x] Keep v1 at the host/L2-facing boundary rather than requiring direct L1 GPU zero-copy semantics.
- [ ] Define substrate publication semantics:
  - [x] Default KV-page publication policy should be durable-but-evictable.
  - [x] Duplicate page publication must be idempotent at the SGLang integration layer.
  - [x] Partial backend visibility must be tolerated so later publish/snapshot code can reason over mixed already-published and newly-published pages.
- [ ] Integrate with current HiCache control flow without changing hot-path ownership:
  - [x] Keep per-rank radix tree ownership in SGLang.
  - [x] Keep TP synchronization in `HiCacheController` / `HiRadixCache`.
  - [x] Do not introduce a request-level coordinator hop for ordinary prefix share.
- [ ] Add observability:
  - [x] Backend init logs for daemon/store endpoints and rank suffix rules.
  - [x] Counters or debug logs for page `exists/get/set`, duplicate put skip, and backend failures.
- [ ] Add unit-level verification:
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

- [ ] Finish the `share_local` Tensorcast backend path in
  `benchmark/tensorcast_benchmark/kv/share_local/run_benchmark.py`:
  - [x] Replace the current hard failure for `backend=tensorcast`.
  - [x] Generate Tensorcast config files into `outputs/<run_id>/generated_configs/`
    from `share_local/configs/`.
  - [x] Support `tensorcast-daemon-mode=share`.
  - [x] Support `tensorcast-daemon-mode=separate`.
  - [x] Start and stop Tensorcast global store and daemon(s) on the remote worker.
  - [x] Wait for service readiness and fail early on readiness timeout.
  - [x] Copy Tensorcast logs from `/data` into the benchmark output directory.
- [ ] Reuse the proven benchmark service lifecycle pattern:
  - [x] Follow the service lifecycle model already used by
    `benchmark/tensorcast_benchmark/load_weight_remote/scripts/tensorcast_service.sh`.
  - [x] Decide whether to reuse that helper directly or extract a shared KV benchmark helper.
- [ ] Add benchmark request-id control:
  - [x] Extend `benchmark/tensorcast_benchmark/kv/sgl_client.py` so `/generate` can send an explicit `rid`.
  - [x] Make `request_driver.py` construct deterministic request ids for each request pair.
  - [x] Keep the benchmark able to compare the same logical prompt across instances.
- [ ] Keep benchmark result compatibility:
  - [x] Preserve `pair_results.jsonl`, `summary.json`, `logs/`, and append-only CSV behavior.
  - [x] Preserve the Mooncake path as the baseline.
- [ ] Keep the target test environment explicit:
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
- [ ] Implement `M0` Tensorcast host-slab export / attach / lease / keepalive:
  - [ ] Tensorcast proto and client surface:
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
  - [ ] Tensorcast daemon region registry and lifecycle:
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
  - [ ] Tensorcast daemon validation and ownership boundary:
    - [x] extend `daemon/service/controllers/external_target_access_service.*`
      so local source or target validation can reason about `HOST_SHARED`
      regions, not only CUDA IPC regions.
    - [x] extend
      `daemon/service/controllers/materialization_target_storage_utils.*` and
      `daemon/service/byte_artifact_region_layout.*` so region-backed accesses
      can acquire host-shared layouts in addition to VRAM layouts.
    - [ ] keep trust-boundary rules unchanged:
      - batch region RPCs stay loopback or UDS-only,
      - remote daemons still never write directly into caller-visible memory.
  - [ ] Tensorcast tests for `M0`:
    - [x] region registration tests for `HOST_SHARED`,
    - [x] lease keepalive and expiry tests,
    - [x] local attach or detach tests,
    - [x] slab cleanup on explicit release and owner exit.
- [ ] Implement `M1` `BatchPutIfAbsentFromRegion(HOST_SHARED source)` correctness:
  - [ ] Tensorcast request validation and lowering:
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
  - [ ] Tensorcast tests for `M1`:
    - [x] correctness tests for `BatchPutIfAbsentFromRegion(HOST_SHARED source)`
      with:
      - [x] success,
      - [x] duplicate adoption,
      - [x] partial failure,
      - [x] invalid bounds or poisoned region rejection.
  - [ ] After `M1`, optionally do a narrow SGLang-side local integration check
    without yet switching the main backend path.
- [ ] Implement `M2` `BatchGetIntoRegion(HOST_SHARED target)` correctness:
  - [ ] Tensorcast request validation and target access:
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
  - [ ] Tensorcast target execution path:
    - [x] extend `byte_artifact_controller.cc` and any shared lowering helpers
      so `BatchGetIntoRegion(...)` can materialize directly into `HOST_SHARED`
      target slices.
    - [ ] reuse or adapt the existing CPU direct-write sink path where possible
      instead of inventing a second host-target data plane.
    - [x] preserve existing batch-get semantics:
      - [x] partial hit behavior,
      - [x] per-item vs pack-scoped failure rules,
      - [x] layout and verification-mode checks.
  - [ ] Tensorcast tests for `M2`:
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

- [ ] Freeze the Phase-2.5B contract:
  - [ ] One allocator slot equals one HiCache KV page.
  - [ ] One slot lifetime is guarded by a monotonically increasing
    `generation`.
  - [ ] Phase B must reuse the same daemon-exported `HOST_SHARED` slab model as
    Phase A rather than introducing a second host-memory API family.
  - [ ] Once Phase B becomes the active backend path, no GPU staging fallback is
    retained.
- [ ] Implement the Phase-B slot-token and slot-lifecycle contract:
  - [ ] Extend the region-backed request model so the caller can identify one
    slot lifetime, not only a byte window:
    - `region_id`
    - `memory_kind=HOST_SHARED`
    - `slot_index`
    - `slot_generation`
    - `offset_bytes`
    - `length_bytes`
  - [ ] Update Tensorcast proto, layout validation, and response-mapping code so
    these slot-lifetime fields survive request decoding, target/source
    validation, and per-item result mapping.
  - [ ] Keep the first safe rollout at one logical slot token per KV page even
    if adjacent slots are later coalesced for execution efficiency.
  - [ ] Define and wire the caller-owned slot state machine:
    - `SlotFree`
    - `SlotReserved`
    - `GetInFlight`
    - `SlotResident`
    - `PutInFlight`
    - `SlotInvalid`
    - `SlotRetiring`
  - [ ] Define pin or refcount rules so eviction cannot recycle a slot while it
    is reserved or in flight.
  - [ ] Define `SlotInvalid` and retirement semantics for partial failure:
    - get-side fill failure invalidates the target slot,
    - ordinary put failure does not invalidate the local source slot,
    - generation bumps only after retirement and ref-drain.
- [ ] Implement `M3` allocator-slab page-slot state / recycle / partial-failure
  correctness:
  - [ ] Tensorcast-side plumbing:
    - [ ] ensure host-region validation remains slot-token aware but does not
      create a daemon-owned allocator table,
    - [ ] preserve slot token metadata through `BatchGetIntoRegion` and
      `BatchPutIfAbsentFromRegion` result construction so SGLang can revalidate
      generation before making a page visible.
  - [ ] Add Tensorcast unit/integration tests for:
    - slot validation by generation,
    - stale completion rejection,
    - partial batch failure,
    - slot retirement and recycle.
  - [ ] Add SGLang-side correctness checks where feasible for pinning,
    provisional visibility, and eviction interaction.
- [ ] Implement `M4` host-registration attach / detach / fallback:
  - [ ] Add optional one-time `cudaHostRegister(...)` on the SGLang-local slab
    mapping.
  - [ ] Keep Tensorcast host-region semantics independent from host pinning:
    the daemon-side `HOST_SHARED` region model must remain correct whether or
    not the local client successfully host-registers the mapping.
  - [ ] Add the matching detach path:
    - drain in-flight slot refs,
    - retire resident or invalid slots,
    - `cudaHostUnregister(...)`,
    - unmap,
    - release slab lease.
  - [ ] Define the fallback behavior when host registration is unavailable:
    correctness must remain intact and only the performance policy changes.
  - [ ] Add Tensorcast unit/integration tests for `M4`.
- [ ] Add the SGLang Phase-B allocator integration:
  - [ ] Implement a Tensorcast-aware `HostKVCache` allocator backed by one
    daemon-exported slab per rank.
  - [ ] Make L2 host pages live directly in exported slab slots rather than in a
    separate ordinary host pool.
  - [ ] Make `batch_set_v1(...)` publish directly from resident slot offsets
    without an extra copy into a scratch slab.
  - [ ] Make `batch_get_v1(...)` reserve destination slots and materialize
    directly into those slots.
  - [ ] Insert fetched pages into radix-visible HiCache state only after get
    success and generation revalidation.
  - [ ] Retire invalid slots correctly on failure.
- [ ] Validate Phase B in SGLang end to end immediately after allocator
  integration:
  - [ ] Rerun the same `share_local` benchmark shape used for Phase A.
  - [ ] Compare against the last GPU-staging baseline and the Phase-A host
    staging path.
  - [ ] Verify:
    - functionality remains correct,
    - prefix reuse still hits,
    - `TTFT`, `batch_get`, and `batch_set` metrics improve or at least behave as
      expected,
    - GPU staging is completely absent from the active Tensorcast backend path.

### Phase 2.5 Exit Criteria

- [ ] Tensorcast has a daemon-managed `HOST_SHARED` slab lifecycle validated by
  milestone tests `M0` through `M4`.
- [x] `BatchPutIfAbsentFromRegion(...)` supports `HOST_SHARED` source layouts.
- [x] `BatchGetIntoRegion(...)` supports `HOST_SHARED` target layouts.
- [x] Phase A host staging runs end to end in `share_local` without GPU staging.
- [ ] Phase B allocator-backed host residency runs end to end in `share_local`
  with slot-generation protection and no GPU staging fallback.

Current Phase-2.5A validation note:

- Tensorcast milestone coverage currently confirmed locally by:
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

## Phase 3 - Request Transfer Interface

This phase adds the programmable request-level handoff path for PD-disaggregated
inference.

### Phase 3A - SGLang In-Process Instance-Agent and Coordinator

- [ ] Add the SGLang-side in-process instance-agent boundary:
  - [ ] Implement Tensorcast `NodeAgent` semantics inside the logical SGLang instance.
  - [ ] Co-locate the SGLang `EngineAdapter` with that instance-agent.
  - [ ] Keep the Tensorcast daemon as worker/data-plane host, not as owner of instance-step execution.
- [ ] Add the rank-0 coordinator role:
  - [ ] Rank 0 owns the logical `instance_id` lifecycle and Tensorcast directory registration.
  - [ ] Rank 0 receives `publish`, `hydrate`, and `evict_local` calls for the logical instance.
  - [ ] Rank 0 fans out to the required TP ranks and aggregates success/failure.
  - [ ] Group-scoped success requires all required ranks to succeed.
- [ ] Add SGLang-owned request bundle bookkeeping:
  - [ ] Keep request-bundle metadata inside SGLang, not in a Tensorcast registry.
  - [ ] Add a page publication registry inside SGLang integration code.
  - [ ] Add request-bundle snapshot state needed to decide publish cutoff and ordered page membership.
  - [ ] Keep the snapshot logic non-blocking to engine execution except where an explicit publish must wait for pending page publication to reach the requested cutoff.
- [ ] Define decode-side prepared state:
  - [ ] Hydrate prepares a target-local request bundle.
  - [ ] Ordinary decode ingress later consumes that prepared state.
  - [ ] Hydrate itself must not silently start decode.

### Phase 3B - Tensorcast Request-Transfer Surface Extensions

- [ ] Extend Tensorcast result/control types:
  - [ ] Add `EngineOwnedManifest` as an opaque engine-owned payload.
  - [ ] Add `PublishManifest` as the controller-visible immutable transfer handle.
  - [ ] Extend `PublishResult` to carry `publish_manifest`.
- [ ] Extend plan and node-agent APIs:
  - [ ] Add `hydrate(publish_manifest=...)`.
  - [ ] Preserve legacy `hydrate(engine_request_id=...)` only as a compatibility surface for controller-side cached manifests.
  - [ ] Ensure the controller, not the target instance, owns any compatibility cache from `engine_request_id` to the last known `PublishManifest`.
- [ ] Extend proto serialization and Python APIs:
  - [ ] Update plan proto.
  - [ ] Update node-agent proto.
  - [ ] Update Python plan builders and result deserialization.
  - [ ] Update executor/server serialization paths.
- [ ] Add Tensorcast tests for the new request-transfer handle semantics.

### Phase 3C - SGLang EngineAdapter Publish/Hydrate/Evict

- [ ] Implement source-side `publish(...)`:
  - [ ] Resolve source request state from SGLang-owned request metadata.
  - [ ] Ensure the publish snapshot covers the full intended request cutoff.
  - [ ] Reuse already-published substrate pages when possible rather than re-uploading every page.
  - [ ] Return a `PublishManifest` containing:
    - generic Tensorcast manifest data,
    - opaque `EngineOwnedManifest` needed by SGLang to resume decode.
- [ ] Implement target-side `hydrate(...)`:
  - [ ] Accept `publish_manifest=...` as the canonical v1 path.
  - [ ] Materialize the request bundle into target-local prepared state.
  - [ ] Separate transport success from decode-usability filtering.
  - [ ] Only after successful prepare should the ordinary decode request be admitted on the target instance.
- [ ] Implement `evict_local(...)`:
  - [ ] Remove target-local prepared/live request bundle state.
  - [ ] Keep local evict semantics distinct from global page deletion.
- [ ] Add optional worker warmup:
  - [ ] Use `prefetch_manifest_result(...)` only as an optimization.
  - [ ] Keep `prefetch` orthogonal to correctness of `hydrate`.

### Phase 3D - Caller / Controller Validation

- [ ] Add a realistic external caller example for PD handoff:
  - [ ] choose source and target instances,
  - [ ] prefill on source,
  - [ ] publish on source,
  - [ ] optionally prefetch on target host daemon,
  - [ ] hydrate on target,
  - [ ] resume decode on target.
- [ ] Add local integration tests for single-controller request transfer.
- [ ] Keep the v1 restriction explicit:
  - [ ] single controller,
  - [ ] explicit `PublishManifest`,
  - [ ] no generic cross-daemon multi-instance super-plan beyond current Tensorcast routing limits.

### Phase 3 Exit Criteria

- [ ] A controller can publish one request's KV snapshot from a source instance.
- [ ] A controller can hydrate that snapshot on a target instance using `PublishManifest`.
- [ ] The target instance can resume decode using ordinary SGLang ingress.
- [ ] The integration keeps request-bundle metadata owned by SGLang, not Tensorcast core.

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

### SGLang files already added for Phase 1 / Phase 1.5 / Phase 2

- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/__init__.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/tensorcast_store.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/config.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/client.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/test_tensorcast_store.py`
- `sglang/benchmark/tensorcast_benchmark/kv/share_local/scripts/tensorcast_service.sh`

### New SGLang files still likely needed for Phase 3

- `sglang/python/sglang/srt/tensorcast/instance_agent.py`
- `sglang/python/sglang/srt/tensorcast/coordinator.py`
- `sglang/python/sglang/srt/tensorcast/engine_adapter.py`
- `sglang/python/sglang/srt/tensorcast/page_publication_registry.py`
- `sglang/python/sglang/srt/tensorcast/request_bundle_state.py`

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
- `tensorcast/tools/build_proto_python.sh`

### Optional Tensorcast files that may need targeted fixes

- `tensorcast/tensorcast/api/runtime.py`
- `tensorcast/tensorcast/api/directory.py`
- `tensorcast/tensorcast/runtime.py`
- `tensorcast/tensorcast/client_runtime.py`

These should only be touched if the SGLang request-transfer implementation
exposes a concrete routing or runtime gap that cannot be solved in SGLang's
integration layer.
