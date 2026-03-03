# Tensorcast Loader and Weight Updates Plan (SGLang)

This is the executable TODO list for implementing `load_format=tensorcast` + pull-by-key online updates.

## Target Outcome

- Bootstrap/load from Tensorcast artifact by key with `tc.artifact(key=...)`.
- Generate and reuse TP/PP slice plans via symbolic trace (`TraceMode`) so only needed shard tensors are materialized.
- Provide a dedicated admin API `POST /update_weights_from_tensorcast` for versioned key-based updates.
- Enforce strict safety for cache/graph/runtime semantics and monotonic numeric versions.

## Phase 0 — Foundation

- [ ] Add/confirm documentation baseline:
  - [ ] Keep `sglang/docs/tensorcast/tensorcast_loader_and_weight_updates.md` and `.../tensorcast_weight_update_protocol.md` aligned with implementation decisions.
  - [ ] Add this plan file under `sglang/docs/tensorcast/plan/`.
- [ ] Define v1 scope:
  - [ ] `Tensorcast` mode is canonical-checkpoint + pull-by-key.
  - [ ] Local `from_disk` is bootstrap fallback only, never online update source.
  - [ ] `--weight-version` in Tensorcast mode is numeric and monotonic.

## Phase 1 — Tensorcast Runtime Connection (Smallest dependency unit)

- [ ] Configuration ingestion:
  - [ ] Extend/validate `model_loader_extra_config` keys used by Tensorcast mode.
  - [ ] Add explicit required keys (daemon addr, model name, key template, daemon options).
  - [ ] Parse/validate with clear errors if absent/misconfigured.
- [ ] Runtime client lifecycle:
  - [ ] Create lightweight client init helper under Tensorcast loader module.
  - [ ] Add connect/retry behavior with bounded timeout.
  - [ ] Cache a per-process client handle and avoid repeated daemon connects.
- [ ] Artifact resolution utilities:
  - [ ] Resolve artifact key from `weight_version` using template.
  - [ ] Add helper that converts local fallback config to Tensorcast `FallbackOptions`.
  - [ ] Add artifact describe wrapper (metadata probe) used by both bootstrap and update.
- [ ] Observability:
  - [ ] Add clear startup logs for daemon endpoint, model name, template, and connection status.
  - [ ] Add failure paths for connection errors, describe failures, and key/template mismatch.

## Phase 2 — TraceMode/Data-Plan Infrastructure (Subtasks, incremental)

- [ ] Establish trace plan core:
  - [ ] Add internal `TensorcastTracePlan` schema (source name + source slice + destination name + destination slice + op transforms).
  - [ ] Make planner rank-aware (TP/PP aware) and deterministic across ranks.
- [ ] Implement symbolic op coverage:
  - [ ] Add/verify support for `copy_`, `fill_`, `view`, `narrow`, `slice`.
  - [ ] Add support for `cat/stack`, `transpose/permute`, `to`, `contiguous`, `clone`.
  - [ ] Track affine index transforms so resulting plan is still shard-selectable.
- [ ] Add unsupported-op fallback policy:
  - [ ] For ops that cannot be symbolically reduced to 1-D sliceable metadata, mark source tensors as unsliceable.
  - [ ] Keep `subset(name)` and fall back to materializing full source tensor only for the affected name.
  - [ ] Emit structured warning when fallback is used.
- [ ] Add trace cache:
  - [ ] Deterministic cache key includes model family, revision/arch, quant/config, world size, tp/pp rank.
- [ ] Verify correctness:
  - [ ] Golden path: cat+transpose models produce slice plans.
  - [ ] Fallback path: no unsupported-op crash; logs + full-materialize fallback.

## Phase 3 — TensorcastModelLoader (Bootstrap/Reload Core)

- [ ] Add `TensorcastModelLoader` entrypoint in model-loader registry:
  - [ ] Integrate in `get_model_loader(...)` when `load_format == "tensorcast"`.
- [ ] Bootstrap implementation:
  - [ ] Build no-init/meta model for tracing.
  - [ ] Run TraceMode against canonical checkpoint key (or explicit bootstrap key).
  - [ ] Materialize only required source tensors via:
    - `artifact.subset(materialize_names).view(slices=...)`
    - `artifact.tensor_dict(...)`
  - [ ] Apply copy/transform plan into actual initialized model parameters/buffers.
- [ ] Post-load semantics alignment:
  - [ ] Call `quant_method.process_weights_after_loading(module)` on final host/device map.
  - [ ] Call `post_load_weights(model, model_config)` after all copies.
- [ ] Plan reuse:
  - [ ] Use same trace result across bootstrap and later updates when model layout unchanged.
- [ ] Error handling:
  - [ ] Ensure model object stays at previous state on failure.
  - [ ] Return typed, actionable loader errors (plan miss, slice mismatch, artifact miss).

## Phase 4 — Pull-by-key Online Update API

- [ ] Add request schema:
  - [ ] `weight_version` (required, int).
  - [ ] `artifact_key` (optional, integrity check).
  - [ ] `flush_cache` (default true for Tensorcast mode).
  - [ ] `abort_all_requests` / `recapture_cuda_graph` flags as needed.
- [ ] Add `POST /update_weights_from_tensorcast`:
  - [ ] Add admin-only guard.
  - [ ] Add synchronous response contract (success only when reload applied).
  - [ ] Return 200/400/409/500 as defined in protocol.
- [ ] Route to model-runner update handler:
  - [ ] Serialize update attempts.
  - [ ] Resolve key from template using `weight_version`.
- [ ] Reject malformed calls:
  - [ ] Missing/invalid version rejected immediately.
  - [ ] `artifact_key` mismatch against template rejected.

## Phase 5 — Update Execution Semantics and Safety (Critical)

- [ ] Runtime version policy:
  - [ ] Parse `current_version` and incoming version as base-10 integers.
  - [ ] Enforce monotonicity: reject rollback, idempotent no-op allowed.
  - [ ] Update in-memory/version state only after successful apply.
- [ ] Update protocol:
  - [ ] Gate inference with existing model-update lock mechanism.
  - [ ] Open artifact by key and call `describe()` before loading.
- [ ] Memory/cuda safety:
  - [ ] `flush_cache` true by default and effectively required for Tensorcast mode.
  - [ ] Cross-rank TP/PP barrier after apply.
  - [ ] `torch.cuda.synchronize()` before and after materialization/apply.
- [ ] CUDA graph policy:
  - [ ] If graphs enabled:
    - [ ] require `recapture_cuda_graph=true`, or
    - [ ] reject update in v1.
  - [ ] Add explicit log branch for each outcome.
- [ ] Failure atomicity:
  - [ ] On apply failure, keep serving previous weights.
  - [ ] Preserve previous `weight_version`.

## Phase 6 — Integration & Compatibility

- [ ] Ensure existing endpoints still behave:
  - [ ] Keep `/update_weights_from_disk` for filesystem path semantics.
  - [ ] Document it is not the Tensorcast update primary path.
- [ ] Ensure no model-path mutation occurs when Tensorcast update path is used.
- [ ] Confirm request routing and locking behavior for both PP and TP ranks.
- [ ] Verify admin and non-admin response behavior in all new branches.

## Phase 7 — Validation and Documentation Closeout

- [ ] Dry run checklist: **skip** (no runnable SGLang test environment available in this repo context).
  - [ ] Bootstrap with qwen3 and tensorcast key.
  - [ ] Trigger new weight version from publisher simulation.
  - [ ] Confirm `model_info.weight_version` increments on success.
- [ ] Runtime safety checklist: **skip** (no runnable SGLang test environment available in this repo context).
  - [ ] Validate cache flush and graph recapture settings.
  - [ ] Validate rollback is rejected and older versions retained.
- [ ] Failure-case checklist: **skip** (no runnable SGLang test environment available in this repo context).
  - [ ] Artifact key missing.
  - [ ] Version rollback.
  - [ ] Trace unsupported-op fallback.
  - [ ] Daemon unreachable.
- [ ] Update docs:
  - [ ] Keep protocol + loader docs in sync with any API or validation changes.
  - [ ] Add explicit “known limits / fallback mode” section.

## Code Map

### Primary vLLM references (known-good design)

- `vllm/docs/design/tensorcast_loader_and_weight_updates.md`
- `vllm/docs/design/tensorcast_weight_update_protocol.md`
- `vllm/vllm/model_executor/model_loader/tensorcast_loader.py`
- `vllm/vllm/model_executor/model_loader/tensorcast_utils.py`
- `vllm/vllm/model_executor/model_loader/tensorcast_runtime.py`
- `vllm/vllm/model_executor/model_loader/registry.py`
- `vllm/vllm/entrypoints/openai/serving_chat.py` (HTTP control flow for analogies)

### SGLang files to modify

- `sglang/python/sglang/srt/configs/load_config.py` (load format + tensorcast config exposure)
- `sglang/python/sglang/srt/server_args.py` (admin/API/version defaults and validation notes)
- `sglang/python/sglang/srt/model_executor/model_runner.py` (new Tensorcast update execution path)
- `sglang/python/sglang/srt/model_loader/loader.py` (Tensorcast loader + TraceMode + post-load execution)
- `sglang/python/sglang/srt/model_loader/utils.py` (post-load hooks reuse)
- `sglang/python/sglang/srt/entrypoints/http_server.py` (new route registration and response)
- `sglang/python/sglang/srt/io_struct.py` (payload schema for new request)
- `sglang/python/sglang/srt/managers/tokenizer_communicator_mixin.py` (writer lock and synchronization behavior if reused)
- `sglang/python/sglang/srt/managers/tokenizer_manager.py` (model update control semantics if needed)
- `sglang/docs/tensorcast/tensorcast_loader_and_weight_updates.md` (design sync)
- `sglang/docs/tensorcast/tensorcast_weight_update_protocol.md` (protocol sync)

### New files to add (likely)

- `sglang/python/sglang/srt/model_loader/tensorcast_loader.py` (loader + trace/apply plan executor)
- `sglang/python/sglang/srt/model_loader/tensorcast_trace.py` (trace collector + op handlers)
- `sglang/python/sglang/srt/model_loader/tensorcast_runtime.py` (Tensorcast artifact/client helper, if split)

### Optional future extension (not v1)

- [ ] Generalized `/update_weights_from_disk` loader-agnostic behavior.
- [ ] Full benchmark pass on cold/hot swap latency and cache footprint.
