# Tensorcast Loader and Online Weight Updates (SGLang)

This document describes the intended **Tensorcast integration design** for SGLang:

- **Bootstrap / initial load (startup)** from a Tensorcast artifact representing a *canonical checkpoint*.
- **Online weight updates (post-startup)** driven by `weight_version` and immutable, versioned Tensorcast keys.
- **Trace-driven TP/PP slice planning** so each rank can `subset()`/`view()` and materialize only what it needs.

Protocol contract:
- `sglang/docs/tensorcast/tensorcast_weight_update_protocol.md`

Status:
- This is a design/proposal document. Concrete filenames/classes may differ once implemented, but the call flows and invariants here are the target behavior.
- This document intentionally **does not** cover binding-based swap APIs (e.g. `bind_into` / `binding.swap`).

---

## 1) Scope and Key Requirements

### 1.1 Two phases

1. **Bootstrap / initial load (startup)**
   - SGLang starts and loads model weights into an initialized model instance.
   - In Tensorcast mode, bootstrap may allow `tc.from_disk(hf_folder)` (import path) as a convenience, but should prefer `tc.artifact(key=...)` when a key is configured.

2. **Online updates (post-initialization)**
   - A publisher (e.g. a trainer or a serving instance acting as publisher) publishes a new canonical checkpoint under a *new immutable* Tensorcast key.
   - SGLang reloads weights **in-place** (same Python model object) to avoid disrupting runtime state (CUDA graphs, caches, parallel groups).
   - Online updates **MUST NOT** fall back to importing from an explicit local checkpoint folder (`tc.from_disk(...)`). They must open by key via `tc.artifact(key=..., fallback=...)`.

### 1.2 Canonical checkpoint artifact (what Tensorcast stores)

The Tensorcast artifact is treated as a **canonical checkpoint**:

- Tensor names match the names used by SGLang weight-loading semantics (typically HF-style state_dict keys).
- Tensor shapes/dtypes are consistent with the configured model architecture and parallelism settings.
- The artifact is immutable; each update produces a *new key*.

This matches the SGLang expectation that model-specific logic (stacked params, fused modules, TP slicing rules, PP filtering) is encoded in `model.load_weights(...)` and per-parameter `weight_loader` functions.

---

## 2) Configuration (scheme-2 / config-driven)

SGLang configuration surfaces:

- CLI:
  - `--load-format tensorcast` (new load format)
  - `--model-loader-extra-config '<json>'` (already exists; passed into `LoadConfig.model_loader_extra_config`)
- Runtime:
  - `LoadConfig.model_loader_extra_config` is parsed from JSON if a string (see `sglang/python/sglang/srt/configs/load_config.py`).

Recommended `--model-loader-extra-config` keys (Tensorcast):

```json
{
  "tensorcast_init_mode": "auto",
  "tensorcast_daemon_address": "127.0.0.1:50052",
  "tensorcast_show_daemon_logs": false,

  "tensorcast_key_template": "model:{model_name}:v{weight_version}",
  "tensorcast_model_name": "qwen3_8b",

  "tensorcast_allow_disk_fallback": true,
  "tensorcast_fallback_prefer": "auto",

  "tensorcast_get_prefer": "auto",
  "tensorcast_export_policy": "auto",

  "tensorcast_trace_tp_slices": true,
  "tensorcast_tp_slice_plan_cache_dir": "/tmp/sglang_tensorcast_trace_cache"
}
```

Resolution rules:

- **Bootstrap**:
  - If `tensorcast_artifact_key` is provided, load from that key via `tc.artifact(key=...)`.
  - Otherwise, bootstrap can fall back to `tc.from_disk(hf_folder)` if disk fallback is enabled and a local checkpoint is available.
- **Online update**:
  - The worker resolves `artifact_key` from `weight_version` using `tensorcast_key_template` and injects it as `tensorcast_artifact_key` for that reload attempt.
  - Online updates must fail if neither `weight_version` nor an explicit `tensorcast_artifact_key` is available.

---

## 3) Bootstrap / Initial Load (startup)

### 3.1 SGLang startup call flow (high level)

SGLang model startup today flows roughly as:

1. `python -m sglang.launch_server ...`
2. `ModelRunner.load_model()` builds `LoadConfig(...)` (including `model_loader_extra_config`) and selects a loader via `get_model_loader(...)`:
   - `sglang/python/sglang/srt/model_executor/model_runner.py`
   - `sglang/python/sglang/srt/model_loader/loader.py`
3. Loader constructs the model and loads weights.

Tensorcast integration targets the same shape:

- `get_model_loader(load_config, model_config)` returns `TensorcastModelLoader` when `load_format == "tensorcast"`.
- `TensorcastModelLoader.load_model(...)` initializes the model and loads weights via Tensorcast.

### 3.2 Trace as semantic authority (slice planning)

Different architectures (Qwen/Llama/MoE/etc.) implement different weight mapping rules in `model.load_weights(...)`.

To materialize only the per-rank shard from a **canonical** checkpoint, the loader must know:

- which source tensors are needed for this rank,
- which 1-D slice (if any) to request from Tensorcast via `artifact.view(slices=...)`, and
- how to copy/fill into destination parameters/buffers.

The design follows vLLM’s approach:

1. Construct a **meta/no-init model** (or a “dry-run model”) that has the correct parameter *shapes* for the current TP/PP rank.
2. Run `model.load_weights(...)` under a `TorchDispatchMode` (“TraceMode”) using **meta tensors** as sources.
3. Record a **copy plan** at `aten::copy_` / `aten::fill_` boundaries:
   - `(ckpt_name, ckpt_slice) -> (dst_name, dst_slice)` plus constant fills.
4. Derive:
   - `expected_src_names`: source tensors referenced by the plan
   - `tensorcast_slices`: the minimal 1-D hull slice per source tensor usable by `artifact.view(slices=...)`

Key invariant:
- The trace must reflect *exactly* the semantics of the real `model.load_weights(...)` path for the current model/quant/TP/PP configuration.

### 3.3 Materialize tensors and apply the copy plan

Once a trace plan exists:

1. Open the Tensorcast artifact:
   - Bootstrap: allow `tc.from_disk(hf_folder)` if configured.
   - Online update: use `tc.artifact(key=..., fallback=...)` and call `artifact.describe()`.
2. Select and slice:
   - `artifact_tp = artifact.subset(materialize_names).view(slices=tensorcast_slices)`
3. Materialize the per-rank tensor dict on the target device:
   - `tensor_dict = artifact_tp.tensor_dict(device="cuda:<id>", options=...)`
4. Apply the copy plan into the already-initialized model parameters/buffers:
   - `dst.copy_(src)` on narrowed views
   - `dst.fill_(value)` for constant/scalar fills
5. Run post-load processing:
   - per-module `quant_method.process_weights_after_loading(module)` on the correct device
   - `post_load_weights(model, model_config)` when the model defines `post_load_weights()`

### 3.4 Post-load semantics alignment (SGLang)

SGLang effectively has **two** kinds of “post” stages after weights land in
parameters/buffers:

1. **Quant method post-processing** (repack/quantize/derive aux buffers):
   - `quant_method.process_weights_after_loading(module)`
   - Reference: `DefaultModelLoader.load_weights_and_postprocess(...)` calls this
     after `model.load_weights(...)` (`sglang/python/sglang/srt/model_loader/loader.py:669`).
2. **Model post-load hook** (assign derived members / finalize layout):
   - `post_load_weights(model, model_config)` calls `model.post_load_weights()`
     when present (`sglang/python/sglang/srt/model_loader/utils.py:119`).
   - Reference: `DummyModelLoader` calls `post_load_weights(...)` after dummy init
     (`sglang/python/sglang/srt/model_loader/loader.py:1279`).

TensorcastModelLoader **MUST** preserve these semantics for both bootstrap and
online reloads, because the trace/apply path may bypass the usual
`model.load_weights(...)` call boundary.

---

## 4) Online Weight Updates (post-initialization)

### 4.1 SGLang weight update entrypoints (and why Tensorcast needs its own)

SGLang already has admin endpoints for in-place weight updates (defined in
`sglang/python/sglang/srt/entrypoints/http_server.py`):

- `POST /update_weights_from_disk` (`http_server.py:787`)
- `POST /update_weights_from_tensor` (`http_server.py:902`)
- `POST /update_weights_from_distributed` (`http_server.py:924`)
- `POST /update_weights_from_ipc` (`http_server.py:943`)

They differ primarily in **data plane** (how bytes arrive) and in whether the
update is **loader-driven pull** or **tensor push**:

| Endpoint | Data plane | Who provides bytes? | How applied |
|---|---|---|---|
| `/update_weights_from_disk` | disk / HF cache | server pulls | `DefaultModelLoader` reload (currently hard-coded) |
| `/update_weights_from_tensor` | HTTP + tensor handle (metadata over HTTP) | caller pushes tensors | `model.load_weights(named_tensors)` or custom loader |
| `/update_weights_from_distributed` | torch.distributed broadcast | caller pushes tensors | `model.load_weights(named_tensors)` after broadcast |
| `/update_weights_from_ipc` | checkpoint-engine IPC (ZMQ handles) | caller pushes tensors | `model.load_weights(...)` via IPC integration |

Key call-chain facts (current code):

- `/update_weights_from_disk` ultimately calls
  `ModelRunner.update_weights_from_disk(...)`
  (`sglang/python/sglang/srt/model_executor/model_runner.py:1019`), which is
  currently restricted to `DefaultModelLoader` only
  (`model_runner.py:1036`). It also mutates `server_args.model_path/load_format`
  on success (`sglang/python/sglang/srt/managers/tokenizer_manager.py:1375`).
- `/update_weights_from_tensor` and `/update_weights_from_distributed` do **not**
  invoke a `ModelLoader`; they ultimately call `self.model.load_weights(...)`
  (`model_runner.py:1357` / `model_runner.py:1288`).

Tensorcast online update is a distinct data plane:

- It is a **pull-by-key** reload (`tc.artifact(key=...)`, `describe()`, then
  `subset()/view()/tensor_dict()`), and the caller wants to send a small control
  message (version/key), not the full tensor payload.

So in v1 Tensorcast uses one clear pull-by-key management interface:

1. **Dedicated Tensorcast endpoint**: `POST /update_weights_from_tensorcast`.
2. Existing `/update_weights_from_disk` remains local-checkpoint oriented and is not the
   primary Tensorcast control path.

Tensorcast online updates should follow the same operational model:

- **Gate inference** (pause/drain/abort) and **serialize** reload attempts.
- Reload weights **in-place** into the existing model instance.
- Optionally flush caches and recapture CUDA graphs.

Recommended API surface for Tensorcast (new):

- `POST /update_weights_from_tensorcast`
  - payload:
    `{ "weight_version": 123, "artifact_key": "model:qwen3_8b:v123", "flush_cache": true, "abort_all_requests": false, "recapture_cuda_graph": false }`
  - `artifact_key` is optional when SGLang is configured with
    `tensorcast_key_template`; if provided it MUST match the derived key.

Status reporting:

- `GET /model_info` returns the current `weight_version` (single source of truth in `server_args.weight_version`).

Version note (Tensorcast mode):

- SGLang’s `--weight-version` is a string and defaults to `"default"`, but
  Tensorcast updates require **numeric** versions with **monotonic** semantics
  (rollback rejected). Operators should start the server with a numeric
  `--weight-version` matching the bootstrapped artifact.

### 4.2 Online update call flow (design target)

1. TE publishes a new Tensorcast artifact under key `model:{model_name}:v{weight_version}`.
2. TE triggers SGLang reload via `POST /update_weights_from_tensorcast`.
3. SGLang:
   - acquires a global “model update” writer lock (no concurrent inference kernels),
   - resolves `artifact_key` from `weight_version` using `tensorcast_key_template`,
   - opens `tc.artifact(key=artifact_key, fallback=...)` and calls `describe()`,
   - reuses or loads the cached trace plan (no re-trace per version unless layout changes),
   - materializes `artifact.subset(...).view(...).tensor_dict(...)`,
   - applies the copy plan into the live model,
   - runs post-load processing,
   - flushes caches / recaptures CUDA graphs if requested,
   - updates `server_args.weight_version` on success (and leaves it unchanged on failure).

Online-update invariant:
- After startup, Tensorcast reloads must not import from explicit local disk paths (`tc.from_disk`). If the artifact key cannot be opened, the reload fails atomically and the previous weights remain active.

---

## 5) Implementation Notes and Constraints

### 5.1 Trace plan caching

Trace can be expensive and should be cached per rank. A cache key should include at least:

- model identity (e.g. `model_path`, `revision`, architecture),
- dtype / quantization settings affecting parameter layout,
- TP/PP world sizes and rank ids,
- a trace schema version.

### 5.2 Operator coverage in TraceMode

SGLang’s weight loaders may include operations beyond simple `view/narrow/copy_`:

- padding paths may use `torch.cat` and `torch.zeros` (e.g. non-aligned MLP weights),
- some paths may use `transpose`/`permute` or create contiguous copies.

TraceMode v1 MUST assume the following may appear in real model loaders:

- `torch.cat` / `torch.stack` for padding or fusing shards
- `transpose` / `permute`
- dtype casts (`to(dtype=...)`)
- `contiguous()` / `clone()` (sometimes indirectly)

Tensorcast tracing must either:

- **prefer** symbolic tracking for all listed ops so slice/view plans remain valid and can keep
  to `artifact.subset(...).view(slices=...)`;
- **fallback** to graceful degradation only when symbolic support is incomplete:
  - keep name-level `subset()` for the affected source names,
  - disable `view(slices=...)` for those names only and materialize full tensors for them,
  - apply deferred transforms during local copy/fill.

### 5.3 CUDA graphs and runtime state

If SGLang is using CUDA graph capture, weight reload may require:

- pausing inference safely,
- invalidating/re-capturing graphs (`recapture_cuda_graph=true`),
- coordinating across TP/PP/DP workers.

**Strict safety profile (recommended for v1)**:

- **Inference gating is mandatory**:
  - Acquire the global model-update writer lock (same mechanism used by existing
    update endpoints), so no inference kernels race with reload.
- **KV cache must be invalidated**:
  - `flush_cache` MUST be treated as `true` for Tensorcast online updates, because
    old KV states are not valid under new weights.
- **CUDA graph compatibility must be explicit**:
  - If CUDA graphs are enabled (i.e. `--disable-cuda-graph` is false, or
    piecewise CUDA graphs are enabled), then online update MUST either:
    - set `recapture_cuda_graph=true`, or
    - be rejected with an error in v1.
  - Rationale: post-load quant processing may repack weights and reallocate
    buffers, invalidating captured graphs. Even copy-only updates must prove they
    do not change parameter storage addresses.
- **Cross-rank coordination**:
  - All TP ranks MUST barrier after apply and before resuming inference (similar
    to the barriers used in `/update_weights_from_tensor` and `/update_weights_from_ipc`).
- **Device synchronization**:
  - Each rank MUST `torch.cuda.synchronize()` (or equivalent device sync) before
    and after the apply phase to avoid releasing the lock while async copies are
    still in flight.

---

## 6) Examples

### 6.1 Bootstrap from Tensorcast by key (proposed)

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --tp 8 \
  --weight-version 123 \
  --load-format tensorcast \
  --model-loader-extra-config '{
    "tensorcast_init_mode": "auto",
    "tensorcast_model_name": "qwen3_8b",
    "tensorcast_artifact_key": "model:qwen3_8b:v123"
  }'
```

### 6.2 Online update to a new version (proposed)

```bash
curl -X POST http://127.0.0.1:30000/update_weights_from_tensorcast \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <admin_api_key>' \
  -d '{ "weight_version": 124, "flush_cache": true, "recapture_cuda_graph": true }'

curl -s http://127.0.0.1:30000/model_info | jq .weight_version
```
