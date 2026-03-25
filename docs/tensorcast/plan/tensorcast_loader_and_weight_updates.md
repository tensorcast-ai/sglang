# Tensorcast Loader and Weight Updates Plan (SGLang)

This is the executable TODO list for implementing `load_format=tensorcast` + pull-by-key online updates.

## Target Outcome

- Bootstrap/load from Tensorcast artifact by key with `tc.artifact(key=...)`.
- Generate and reuse TP/PP slice plans via symbolic trace (`TraceMode`) so only needed shard tensors are materialized.
- Provide a dedicated admin API `POST /update_weights_from_tensorcast` for versioned key-based updates.
- Enforce strict safety for cache/graph/runtime semantics and monotonic numeric versions.

## Phase 0 - Foundation

- [x] Add/confirm documentation baseline:
  - [x] Keep `sglang/docs/tensorcast/tensorcast_loader_and_weight_updates.md` and `.../tensorcast_weight_update_protocol.md` aligned with implementation decisions.
  - [x] Add this plan file under `sglang/docs/tensorcast/plan/`.
- [x] Define v1 scope:
  - [x] `Tensorcast` mode is canonical-checkpoint + pull-by-key.
  - [x] Local `from_disk` is bootstrap fallback only, never online update source.
  - [x] `--weight-version` in Tensorcast mode is numeric and monotonic.

## Phase 1 - Tensorcast Runtime Connection (Smallest dependency unit)

- [x] Configuration ingestion:
  - [x] Extend/validate `model_loader_extra_config` keys used by Tensorcast mode.
  - [x] Add explicit required keys (daemon addr, model name, key template, daemon options).
  - [x] Parse/validate with clear errors if absent/misconfigured.
- [x] Runtime client lifecycle:
  - [x] Create lightweight client init helper under Tensorcast loader module.
  - [x] Add connect/retry behavior with bounded timeout.
  - [x] Cache a per-process client handle and avoid repeated daemon connects.
- [x] Artifact resolution utilities:
  - [x] Resolve artifact key from `weight_version` using template.
  - [x] Add helper that converts local fallback config to Tensorcast `FallbackOptions`.
  - [x] Add artifact describe wrapper (metadata probe) used by both bootstrap and update.
- [x] Observability:
  - [x] Add clear startup logs for daemon endpoint, model name, template, and connection status.
  - [x] Add failure paths for connection errors, describe failures, and key/template mismatch.

## Phase 2 - TraceMode/Data-Plan Infrastructure (Subtasks, incremental)

- [x] Establish trace plan core:
  - [x] Add internal `TensorcastTracePlan` schema (source name + source slice + destination name + destination slice + op transforms).
  - [x] Make planner rank-aware (TP/PP aware) and deterministic across ranks.
- [x] Implement symbolic op coverage:
  - [x] Add/verify support for `copy_`, `fill_`, `view`, `narrow`, `slice`.
  - [x] Add support for `cat`, `transpose/permute`, `to`, `contiguous`, `clone`.
  - [x] Add support for `stack`.
  - [x] Track affine index transforms so resulting plan is still shard-selectable.
- [x] Add unsupported-op fallback policy:
  - [x] For ops that cannot be symbolically reduced to 1-D sliceable metadata, mark source tensors as unsliceable.
  - [x] Keep `subset(name)` and fall back to materializing full source tensor only for the affected name.
  - [x] Emit structured warning when fallback is used.
- [x] Add trace cache:
  - [x] Add TracePlan JSON (de)serialization helpers.
  - [x] Deterministic cache key includes model family, revision/arch, quant/config, world size, tp/pp rank.
- [ ] Verify correctness: **skip** (no runnable SGLang test environment available in this repo context).
  - [ ] Golden path: cat+transpose models produce slice plans.
  - [ ] Fallback path: no unsupported-op crash; logs + full-materialize fallback.

## Phase 3 - TensorcastModelLoader (Bootstrap/Reload Core)

- [x] Add `TensorcastModelLoader` entrypoint in model-loader registry:
  - [x] Integrate in `get_model_loader(...)` when `load_format == "tensorcast"`.
- [x] Bootstrap implementation:
  - [x] Build no-init/meta model for tracing.
  - [x] Prevent RoPE meta cache pollution (required for TraceMode):
    - [x] Note: `get_rope(...)` caches rotary modules globally in `sglang.srt.layers.rotary_embedding._ROPE_DICT` and trace-time meta init can leak into real model init.
    - [x] Implement `_evict_meta_rotary_cache()`-equivalent for SGLang (ref: `vllm/vllm/model_executor/model_loader/tensorcast_loader.py:_evict_meta_rotary_cache`).
    - [x] Call eviction after meta-model trace (or before real model init) to remove cached RoPE modules whose `cos_sin_cache.is_meta == True`.
    - [x] Log `evicted=<n>` on rank0 (best-effort; no hard failure if cache module layout changes).
    - [x] Add a guardrail: if a real model’s rotary module still has `cos_sin_cache.is_meta`, fail fast with a clear error suggesting cache eviction (avoid silent correctness bugs).
  - [x] Key-miss disk fallback (bootstrap-only convenience):
    - [x] Attempt `tc.artifact(key=artifact_key)` + `artifact.describe()` first.
    - [x] If key is NOT_FOUND and `tensorcast_allow_disk_fallback==true` and `--model-path` is provided, import via `tc.from_disk(model_path)` and continue using the returned `artifact_id` (do not re-open by key).
    - [x] After successful materialize/apply, best-effort publish canonical key mapping via `WeightPublisher.publish_from_disk(model_path, version)`; this avoids runtime `state_dict` name drift and keeps Tensorcast artifact names HF-canonical.
    - [x] Keep online update path key-only: missing key must return an error (never read from disk during online update).
  - [x] Run TraceMode against canonical checkpoint key (or explicit bootstrap key).
  - [x] Materialize only required source tensors via:
    - `artifact.subset(materialize_names).view(slices=...)`
    - `artifact.tensor_dict(...)`
  - [x] Apply copy/transform plan into actual initialized model parameters/buffers.
- [x] Post-load semantics alignment:
  - [x] Call `quant_method.process_weights_after_loading(module)` on final host/device map.
  - [x] Call `post_load_weights(model, model_config)` after all copies.
- [x] Plan reuse:
  - [x] Use same trace result across bootstrap and later updates when model layout unchanged.
- [x] Error handling:
  - [x] Ensure model object stays at previous state on failure.
  - [x] Return typed, actionable loader errors (plan miss, slice mismatch, artifact miss).

## Phase 4 - Pull-by-key Online Update API

- [x] Add request schema:
  - [x] `weight_version` (required, int).
  - [x] `artifact_key` (optional, integrity check).
  - [x] `flush_cache` (default true for Tensorcast mode).
  - [x] `abort_all_requests` / `recapture_cuda_graph` flags as needed.
- [x] Add `POST /update_weights_from_tensorcast`:
  - [x] Add admin-only guard.
  - [x] Add synchronous response contract (success only when reload applied).
  - [x] Return 200/400/409/500 as defined in protocol.
- [x] Route to model-runner update handler:
  - [x] Serialize update attempts.
  - [x] Resolve key from template using `weight_version`.
- [x] Reject malformed calls:
  - [x] Missing/invalid version rejected immediately.
  - [x] `artifact_key` mismatch against template rejected.

## Phase 5 - Update Execution Semantics and Safety (Critical)

- [x] Runtime version policy:
  - [x] Parse `current_version` and incoming version as base-10 integers.
  - [x] Enforce monotonicity: reject rollback, idempotent no-op allowed.
  - [x] Update in-memory/version state only after successful apply.
- [x] Update protocol:
  - [x] Gate inference with existing model-update lock mechanism.
  - [x] Open artifact by key and call `describe()` before loading.
- [x] Memory/cuda safety:
  - [x] `flush_cache` true by default and effectively required for Tensorcast mode.
  - [x] Cross-rank TP/PP barrier after apply.
  - [x] `torch.cuda.synchronize()` before and after materialization/apply.
- [x] CUDA graph policy:
  - [x] If graphs enabled:
    - [x] require `recapture_cuda_graph=true`, or
    - [x] reject update in v1.
  - [x] Add explicit log branch for each outcome.
- [x] Failure atomicity:
  - [x] On apply failure, keep serving previous weights.
  - [x] Preserve previous `weight_version`.

## Phase 6 - Integration & Compatibility

- [x] Ensure existing endpoints still behave:
  - [x] Keep `/update_weights_from_disk` for filesystem path semantics.
  - [x] Document it is not the Tensorcast update primary path.
- [x] Ensure no model-path mutation occurs when Tensorcast update path is used.
- [x] Confirm request routing and locking behavior for both PP and TP ranks.
- [ ] Verify admin and non-admin response behavior in all new branches: **skip** (no runnable SGLang test environment available in this repo context).

## Phase 7 - Benchmark Load Weight

- [x] Create self-contained benchmark package under `sglang/benchmark/tensorcast_benchmark/load_weight/`:
  - [x] Add `README.md` with prerequisites, one-command usage, and examples.
  - [x] Add runner/config scripts only inside this folder (no dependency on `../scripts/start_tensorcast.sh` or other out-of-tree helpers).
  - [x] Add `outputs/` directory contract and file naming rules for logs/results.
- [x] Define benchmark configuration surface (single entrypoint):
  - [x] Required knobs: `model_path`, `model_name`, `tp_size`, `weight_version`, `trials`, `port`, `load_format`.
  - [x] `load_format` supports `tensorcast` and `default`; results must include `load_format` column.
  - [x] Keep runs serial with fixed single port (e.g. `30000`), no parallel trials.
- [x] Implement Tensorcast lifecycle handling in benchmark scripts:
  - [x] Start Global Store + Store Daemon for `load_format=tensorcast`.
  - [x] Stop/cleanup and verify clean state after each benchmark run (`global status` unknown, `daemon status` no local session).
  - [x] Enforce NVRTC/CUDA library env wiring before daemon/server launch to avoid runtime mismatch regressions.
- [x] Implement artifact publish policy for Tensorcast mode:
  - [x] Publish once per unique benchmark config.
  - [x] Reuse the same published artifact across all trials of that config.
  - [x] Do not unload artifact or clear caches between trials (hot-cache effect is intentional).
- [x] Implement launch + timing collection:
  - [x] Start timer at SGLang launch command execution.
  - [x] `load_time`: from launch start to weight-load completion marker.
  - [x] `ready_time`: from launch start to first successful `/health` 200 response.
  - [x] Parse markers from server logs:
    - [x] `default`: `Load weight end.` (use last TP completion as config-level load completion).
    - [x] `tensorcast`: `store.tensor_dict.materialized` (use last TP completion as config-level load completion).
- [x] Implement trial loop and failure policy:
  - [x] On trial failure, record error details and continue subsequent trials.
  - [x] Always attempt process cleanup on both success and failure paths.
  - [x] Preserve per-trial raw stdout/stderr logs under `outputs/` for postmortem.
- [x] Implement append-only CSV reporting:
  - [x] Append all trial records to `sglang/benchmark/tensorcast_benchmark/load_weight/outputs/benchmark_results.csv`.
  - [x] Include at least: timestamp, model_path, model_name, tp_size, weight_version, load_format, trial_id, load_time_s, ready_time_s, status, error_message.
  - [x] Keep schema stable across reruns; never overwrite historical rows.
- [x] Add baseline-vs-tensorcast comparability guidance in README:
  - [x] Document identical knobs for fair comparison (same model/tp/mem settings, only `load_format` differs).
  - [x] Document that `default` baseline uses normal SGLang loader path and no Tensorcast runtime.

## Phase 8 - Benchmark Update Weight

- [x] Create self-contained benchmark package under `sglang/benchmark/tensorcast_benchmark/update_weight/`:
  - [x] Add `README.md` with prerequisites, one-command usage, and examples.
  - [x] Add runner/config scripts only inside this folder (no dependency on out-of-tree helper scripts).
  - [x] Add `outputs/` directory contract and file naming rules for logs/results.
- [x] Define benchmark configuration surface (single entrypoint):
  - [x] Required knobs: `model_path`, `model_name`, `tp_size`, `weight_version_start`, `trials`, `port`, `load_format`.
  - [x] `load_format` supports `tensorcast` and `default`; results must include `load_format` column.
  - [x] Keep runs serial with fixed single port (e.g. `30000`), no parallel trials.
- [x] Implement Tensorcast lifecycle handling for update benchmark (same baseline as Phase 7):
  - [x] Start Global Store + Store Daemon for `load_format=tensorcast`.
  - [x] Stop/cleanup and verify clean state after benchmark run (`global status` unknown, `daemon status` no local session).
  - [x] Enforce NVRTC/CUDA library env wiring before daemon/server launch.
- [x] Implement Tensorcast publish policy for update trials:
  - [x] For `trials=N`, publish `N+1` versions ahead of time: `v0..vN`.
  - [x] Use `WeightPublisher.publish(...)` (tensor dict path), not `publish_from_disk(...)`.
  - [x] Publish time is setup-only and must not be counted in benchmark metrics.
- [x] Implement server bootstrap before update trials:
  - [x] Launch SGLang once per benchmark run.
  - [x] Tensorcast mode: launch with `--load-format tensorcast` and initial `--weight-version 0`.
  - [x] Default baseline mode: launch with normal loader path and initial `--weight-version 0`.
- [x] Implement per-trial update request execution:
  - [x] Trial index `k` updates to version `k` (starting from `k=1`).
  - [x] Tensorcast endpoint:
    - [x] `POST /update_weights_from_tensorcast` with `weight_version=k`, `artifact_key=model:{model_name}:v{k}`, `flush_cache=true`, `abort_all_requests=true`, `recapture_cuda_graph=true`.
  - [x] Default baseline endpoint:
    - [x] `POST /update_weights_from_disk` with same operational flags and `model_path` fixed; use `weight_version=str(k)`.
- [x] Implement update timing collection:
  - [x] `load_time`: from TP log marker `Update engine weights online from ... begin` to `Update weights end.` (use last TP completion as trial completion).
  - [x] `ready_time`: from update HTTP request send time to HTTP 200 response receive time.
- [x] Implement failure policy for update benchmark:
  - [x] Any trial failure immediately aborts the remaining trials.
  - [x] CSV only appends successful trial records.
  - [x] Preserve per-trial raw stdout/stderr logs under `outputs/` for postmortem.
- [x] Implement append-only CSV reporting:
  - [x] Append successful trial rows to `sglang/benchmark/tensorcast_benchmark/update_weight/outputs/benchmark_results.csv`.
  - [x] Include at least: timestamp, model_path, model_name, tp_size, target_weight_version, load_format, trial_id, load_time_s, ready_time_s, status, endpoint, log_path.
  - [x] Keep schema stable across reruns; never overwrite historical rows.
- [x] Add baseline-vs-tensorcast comparability guidance in README:
  - [x] Keep launch knobs identical except update endpoint/load path differences.
  - [x] Document that baseline uses `/update_weights_from_disk` and Tensorcast uses `/update_weights_from_tensorcast`.

## Code Map

### Primary vLLM references (known-good design)

- `vllm/docs/design/tensorcast_loader_and_weight_updates.md`
- `vllm/docs/design/tensorcast_weight_update_protocol.md`
- `vllm/vllm/model_executor/model_loader/tensorcast_loader.py`
- `vllm/vllm/model_executor/model_loader/tensorcast_utils.py`
- `vllm/vllm/model_executor/model_loader/tensorcast_runtime.py`
- `vllm/vllm/model_executor/model_loader/__init__.py` (loader selection / registry pattern)
- `vllm/vllm/model_executor/model_loader/base_loader.py` (loader interface)
- `vllm/vllm/model_executor/model_loader/default_loader.py` (baseline implementation patterns)
- `vllm/vllm/entrypoints/openai/api_server.py` (Tensorcast `/set_model_weight` HTTP enforcement + gating)
- `vllm/vllm/entrypoints/llm.py` (`LLM.set_model_weight` wrapper)
- `vllm/vllm/v1/engine/llm_engine.py` (engine weight reload entrypoint)
- `vllm/vllm/v1/engine/async_llm.py` (async wrapper)

### SGLang files to modify

- `sglang/python/sglang/srt/configs/load_config.py` (load format + tensorcast config exposure)
- `sglang/python/sglang/srt/server_args.py` (admin/API/version defaults and validation notes)
- `sglang/python/sglang/srt/model_executor/model_runner.py` (new Tensorcast update execution path)
- `sglang/python/sglang/srt/model_loader/loader.py` (Tensorcast loader + TraceMode + post-load execution)
- `sglang/python/sglang/srt/model_loader/tensorcast_trace_cache.py` (trace-plan disk cache)
- `sglang/python/sglang/srt/model_loader/utils.py` (post-load hooks reuse)
- `sglang/python/sglang/srt/entrypoints/http_server.py` (new route registration and response)
- `sglang/python/sglang/srt/managers/io_struct.py` (payload schema for new request)
- `sglang/python/sglang/srt/managers/tokenizer_communicator_mixin.py` (writer lock and synchronization behavior if reused)
- `sglang/python/sglang/srt/managers/tokenizer_manager.py` (model update control semantics if needed)
- `sglang/python/sglang/srt/managers/scheduler_update_weights_mixin.py` (scheduler routing + TP barrier)
- `sglang/python/sglang/srt/managers/scheduler.py` (request dispatch registration)
- `sglang/python/sglang/srt/managers/tp_worker.py` (TP worker routing)
- `sglang/docs/tensorcast/tensorcast_loader_and_weight_updates.md` (design sync)
- `sglang/docs/tensorcast/tensorcast_weight_update_protocol.md` (protocol sync)

### New Tensorcast files added (v1)

- `sglang/python/sglang/srt/model_loader/tensorcast_loader.py` (loader + trace/apply plan executor)
- `sglang/python/sglang/srt/model_loader/tensorcast_trace.py` (trace collector + op handlers)
- `sglang/python/sglang/srt/model_loader/tensorcast_runtime.py` (Tensorcast artifact/client helper, if split)
- `sglang/python/sglang/srt/model_loader/tensorcast_trace_cache.py` (trace-plan disk cache)

### Optional future extension (not v1)

- [ ] Generalized `/update_weights_from_disk` loader-agnostic behavior.
- [ ] Full benchmark pass on cold/hot swap latency and cache footprint.
