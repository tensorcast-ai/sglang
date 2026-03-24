# Tensorcast KV Integration Plan (SGLang)

This is the executable TODO list for implementing Tensorcast-backed KV cache
integration in SGLang.

It follows the target semantics frozen in:

- `sglang/docs/tensorcast/tensorcast_kv_protocol.md`
- `sglang/docs/tensorcast/tensorcast_kv_integration.md`

The implementation is intentionally split into:

1. `Phase 1`: shared KV substrate
2. `Phase 2`: prefix share interface
3. `Phase 3`: request transfer interface

This split is the right one for the current codebase because:

- `Phase 1` is the smallest useful data-plane unit,
- `Phase 2` can be validated entirely inside ordinary SGLang serving with no
  external controller,
- `Phase 3` is the first phase that requires Tensorcast programmability-facing
  API additions and in-process instance-agent work.

## Target Outcome

- Add a Tensorcast-backed shared KV substrate for SGLang HiCache pages, with KV
  pages represented in Tensorcast as high-cardinality byte artifacts.
- Make `--hicache-storage-backend tensorcast` a real SGLang HiCache backend
  rather than a benchmark placeholder.
- Make `benchmark/tensorcast/kv/share_local` run end-to-end with
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
  - [ ] `Phase 2` must not require an external controller or request handoff.
  - [ ] `Phase 3` is the only phase allowed to extend Tensorcast request-transfer semantics.
- [ ] Freeze the Phase-2 success criterion:
  - [ ] `benchmark/tensorcast/kv/share_local` must run with
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
  - [ ] Extend `--hicache-storage-backend` validation to include `tensorcast`.
  - [ ] Register a Tensorcast backend in `StorageBackendFactory`.
  - [ ] Add a dedicated backend package under
    `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/`.
- [ ] Define the SGLang-side Tensorcast backend config surface:
  - [ ] Parse backend extra config from `--hicache-storage-backend-extra-config`.
  - [ ] Include daemon/global-store addresses, local daemon association, durable policy,
    and any benchmark-only overrides.
  - [ ] Validate missing or inconsistent fields with clear errors.
- [ ] Implement page identity and namespacing rules:
  - [ ] Use SGLang page hash / prefix-chain-derived identity as the logical page key.
  - [ ] Add TP-rank namespacing for MHA so different TP shards do not collide.
  - [ ] Preserve MLA ownership rules instead of blindly duplicating all ranks.
  - [ ] Keep page identity shared between prefix share and future request transfer.
- [ ] Implement Tensorcast-backed page operations:
  - [ ] `batch_exists(...)`
  - [ ] `batch_get_v1(...)`
  - [ ] `batch_set_v1(...)`
  - [ ] Map each SGLang page to a Tensorcast byte artifact payload boundary.
  - [ ] Keep v1 at the host/L2-facing boundary rather than requiring direct L1 GPU zero-copy semantics.
- [ ] Define substrate publication semantics:
  - [ ] Default KV-page publication policy should be durable-but-evictable.
  - [ ] Duplicate page publication must be idempotent at the SGLang integration layer.
  - [ ] Partial backend visibility must be tolerated so later publish/snapshot code can reason over mixed already-published and newly-published pages.
- [ ] Integrate with current HiCache control flow without changing hot-path ownership:
  - [ ] Keep per-rank radix tree ownership in SGLang.
  - [ ] Keep TP synchronization in `HiCacheController` / `HiRadixCache`.
  - [ ] Do not introduce a request-level coordinator hop for ordinary prefix share.
- [ ] Add observability:
  - [ ] Backend init logs for daemon/store endpoints and rank suffix rules.
  - [ ] Counters or debug logs for page `exists/get/set`, duplicate put skip, and backend failures.
- [ ] Add unit-level verification:
  - [ ] Key/schema tests for TP/MLA namespacing.
  - [ ] Storage round-trip tests for page put/get.
  - [ ] Duplicate publication tests.
  - [ ] Partial batch hit tests for prefix existence queries.

### Phase 1 Exit Criteria

- [ ] One SGLang process can store and retrieve HiCache pages through Tensorcast.
- [ ] TP>1 ranks do not key-collide in the shared substrate.
- [ ] No external controller or Tensorcast instance-step logic is required yet.

## Phase 2 - Prefix Share Interface and Benchmark Bring-up

This phase turns the shared substrate into a real prefix-share path and makes
the `share_local` benchmark the primary validation harness.

### Phase 2A - Benchmark Harness Completion

- [ ] Finish the `share_local` Tensorcast backend path in
  `benchmark/tensorcast/kv/share_local/run_benchmark.py`:
  - [ ] Replace the current hard failure for `backend=tensorcast`.
  - [ ] Generate Tensorcast config files into `outputs/<run_id>/generated_configs/`
    from `share_local/configs/`.
  - [ ] Support `tensorcast-daemon-mode=share`.
  - [ ] Support `tensorcast-daemon-mode=separate`.
  - [ ] Start and stop Tensorcast global store and daemon(s) on the remote worker.
  - [ ] Wait for service readiness and fail early on readiness timeout.
  - [ ] Copy Tensorcast logs from `/data` into the benchmark output directory.
- [ ] Reuse the proven benchmark service lifecycle pattern:
  - [ ] Follow the service lifecycle model already used by
    `benchmark/tensorcast/load_weight_remote/scripts/tensorcast_service.sh`.
  - [ ] Decide whether to reuse that helper directly or extract a shared KV benchmark helper.
- [ ] Add benchmark request-id control:
  - [ ] Extend `benchmark/tensorcast/kv/sgl_client.py` so `/generate` can send an explicit `rid`.
  - [ ] Make `request_driver.py` construct deterministic request ids for each request pair.
  - [ ] Keep the benchmark able to compare the same logical prompt across instances.
- [ ] Keep benchmark result compatibility:
  - [ ] Preserve `pair_results.jsonl`, `summary.json`, `logs/`, and append-only CSV behavior.
  - [ ] Preserve the Mooncake path as the baseline.
- [ ] Keep the target test environment explicit:
  - [ ] Use `--brainctl-charged-group codesign` in documentation/examples for Tensorcast validation.
  - [ ] Keep `--existing-worker-process` supported so an already-running 8xH800 worker can be reused.

### Phase 2B - SGLang Prefix-Share Functional Wiring

- [ ] Make SGLang serve with `--hicache-storage-backend tensorcast`:
  - [ ] Pass Tensorcast backend config into the SGLang server launch path.
  - [ ] Ensure both instances in `share_local` can point at one shared daemon or two separate daemons.
  - [ ] Keep the instances otherwise identical to the Mooncake benchmark topology.
- [ ] Preserve SGLang-native prefix-share behavior:
  - [ ] Storage queries must happen from the existing HiCache hot path.
  - [ ] Prefix-tree insertion remains owned by `HiRadixCache`.
  - [ ] No controller program is involved.
- [ ] Verify functional prefix reuse:
  - [ ] Confirm instance A writes reusable pages into the shared substrate while serving.
  - [ ] Confirm instance B performs storage-backed prefix hits.
  - [ ] Confirm the returned metrics show actual cache reuse signals, not only service bring-up.
- [ ] Validate on the target benchmark:
  - [ ] Run `share_local` on `Qwen3-32B`, `tp=2`, same-node two-instance topology.
  - [ ] Inspect logs and CSV output to confirm the Tensorcast path behaves like a functional prefix-share backend, not just a topology harness.

### Phase 2 Exit Criteria

- [ ] `benchmark/tensorcast/kv/share_local` runs with `hicache-storage-backend=tensorcast`.
- [ ] The benchmark starts Tensorcast services automatically and cleans them up reliably.
- [ ] Prefix-share requests on instance B can reuse pages produced by instance A.
- [ ] The benchmark remains runnable on `charged-group=codesign`.

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

- `Phase 1` and `Phase 2` should be primarily SGLang changes.
- `Phase 3` is where Tensorcast core API changes become necessary.

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
- `sglang/benchmark/tensorcast/kv/models.py`
- `sglang/benchmark/tensorcast/kv/outputs.py`
- `sglang/benchmark/tensorcast/kv/remote.py`
- `sglang/benchmark/tensorcast/kv/sgl_client.py`
- `sglang/benchmark/tensorcast/kv/share_local/run_benchmark.py`
- `sglang/benchmark/tensorcast/kv/share_local/request_driver.py`
- `sglang/benchmark/tensorcast/kv/share_local/README.md`
- `sglang/benchmark/tensorcast/kv/share_local/arch.md`
- `sglang/benchmark/tensorcast/kv/share_local/configs/global_store_config.yaml`
- `sglang/benchmark/tensorcast/kv/share_local/configs/store_daemon_config.yaml`

### New SGLang files likely needed

- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/__init__.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/tensorcast_store.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/config.py`
- `sglang/python/sglang/srt/mem_cache/storage/tensorcast_store/client.py`
- `sglang/python/sglang/srt/tensorcast/instance_agent.py`
- `sglang/python/sglang/srt/tensorcast/coordinator.py`
- `sglang/python/sglang/srt/tensorcast/engine_adapter.py`
- `sglang/python/sglang/srt/tensorcast/page_publication_registry.py`
- `sglang/python/sglang/srt/tensorcast/request_bundle_state.py`
- `sglang/benchmark/tensorcast/kv/share_local/scripts/tensorcast_service.sh`

### Tensorcast files to modify

These are primarily Phase-3 changes. Phase 1 and Phase 2 should try to reuse
existing Tensorcast data-plane/runtime surfaces and only touch core code for
real bugs or missing primitives.

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
