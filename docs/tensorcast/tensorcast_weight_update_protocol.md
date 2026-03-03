# Tensorcast Online Weight Update Protocol v1 (Publisher -> SGLang, HTTP-triggered)

This document defines the **standard contract** for online weight updates into SGLang using Tensorcast:

Publisher `tc.put` (publish) -> SGLang `tc.artifact` (consume) -> SGLang reload (gate + apply).

Implementation details (trace/copy plan + `subset()`/`view()` materialization) live in:
- `sglang/docs/tensorcast/tensorcast_loader_and_weight_updates.md`

---

## 0) Roles and Terms

- **Publisher**: produces a new canonical checkpoint and publishes it to Tensorcast (e.g. a trainer, or a serving instance acting as publisher).
- **TC (Tensorcast store/daemon)**: serves artifacts (P2P + optional managed disk fallback).
- **IS (Inference Server)**: an SGLang instance serving requests.

Terms:

- **Artifact**: immutable value published by `tc.put(...)`.
- **Versioned key**: immutable, human-readable key identifying a version.
- **Meta**: artifact metadata queryable via `artifact.describe()` before materialization.

Normative language:
- **MUST**, **MUST NOT**, **SHOULD**, **MAY** are used as in RFC 2119.

---

## 1) Identity & Naming (MUST)

### 1.1 Immutable version keys (MUST)

A Tensorcast key must not be repointed to a different artifact. Therefore the Publisher **MUST**
publish under a *new* versioned key for each update.

Standard key format (v1):

- `model:{model_name}:v{weight_version}` (example: `model:qwen3_8b:v123`)

Constraints:

- `weight_version` is a non-negative integer.
- `model_name` is a stable identifier for the deployed model family (do not use mutable values like a filesystem path that may change across deployments).
- `model_name` SHOULD be ASCII and URL/key friendly (recommended pattern: `[-._A-Za-z0-9]+`).

SGLang note (Tensorcast mode):

- SGLang’s `--weight-version` CLI flag is a **string** and defaults to `"default"`
  (`sglang/python/sglang/srt/server_args.py:392`).
- In Tensorcast mode, SGLang **MUST** treat `weight_version` as an integer and
  therefore **MUST** be started with a numeric `--weight-version` (e.g. `0` or
  the bootstrapped version).

### 1.2 Model name source of truth (MUST)

SGLang resolves `model_name` from configuration, not from the update request payload:

- SGLang uses `LoadConfig.model_loader_extra_config["tensorcast_model_name"]` when present;
- otherwise it SHOULD fall back to a stable deployment-defined name (recommended: explicitly set `tensorcast_model_name`).

The Publisher MUST use the same `model_name` string that the target SGLang instance is configured to use.

---

## 2) Publish + HTTP Trigger (MUST)

### 2.1 Publish (MUST)

```python
import tensorcast as tc

artifact_key = "model:qwen3_8b:v123"
tc.put(tensors=state_dict_cuda, key=artifact_key, policy="durable")
```

Constraints:

- `tensors` MUST be either:
  - all CUDA tensors on the same GPU device (recommended), or
  - all CPU tensors (Tensorcast will stage them to CUDA during `put`).
- Publisher MUST use a persistence-enabled policy (standard: `"durable"`).
- Publisher MUST NOT attempt to reuse an existing `artifact_key` for a different artifact.

### 2.2 Persistence barrier (MUST)

After `tc.put`, the Publisher MUST wait until the artifact is durably readable via Tensorcast before triggering SGLang reload. This avoids transient “key exists but not readable” failure modes during propagation and disk persistence.

### 2.3 Trigger SGLang reload (HTTP, MUST)

Tensorcast v1 control endpoint:

```
POST /update_weights_from_tensorcast
Content-Type: application/json

{ "weight_version": 123, "artifact_key": "model:qwen3_8b:v123", "flush_cache": true, "abort_all_requests": false }
```

Auth:

- If SGLang is configured with `--admin-api-key`, this endpoint is **admin-only** and callers MUST provide `Authorization: Bearer <admin_api_key>`.

Constraints:

- `weight_version` MUST be present and MUST match the version used to publish `artifact_key`.
- `artifact_key` MAY be included as an integrity check / debugging aid. If SGLang
  is configured with `tensorcast_key_template`, it MUST reject requests where
  `artifact_key` is present but does not equal the derived key.
- The request is synchronous: SGLang MUST only return success after the reload attempt has completed (successfully or with an error).

Response semantics (standard):

- `200 OK`: reload applied, or request was idempotent (`weight_version == current_version`).
- `400 Bad Request`: invalid payload (e.g. missing `weight_version`).
- `409 Conflict`: rollback attempt (`weight_version < current_version`) rejected.
- `500 Internal Server Error`: reload failed; old weights remain active.

---

## 3) SGLang Reload Contract (MUST)

### 3.1 Gate inference before reload (MUST)

SGLang MUST prevent weight reload from racing with inference kernels. Implementations MAY use:

- a global writer lock (e.g. `model_update_lock`),
- pausing the scheduler,
- aborting in-flight requests (optional, request-driven),
- or a combination of the above.

### 3.2 Reload serialization (MUST)

SGLang MUST serialize reload attempts so at most one reload is in-flight at any time for a given model instance.

### 3.3 Monotonic version semantics (MUST)

- SGLang MUST enforce a numeric monotonic version:
  - Parse `current_version` and requested `weight_version` as base-10 integers.
  - Reject non-numeric values in Tensorcast mode (see §1.1 SGLang note).
  - Reject rollback attempts: `weight_version < current_version` (recommended: `409 Conflict`).
  - Treat idempotent updates as success/no-op: `weight_version == current_version` (recommended: `200 OK`).
- On reload failure, SGLang MUST keep serving the previous weights and MUST NOT advance `current_version`.

`current_version` is surfaced through:

- `GET /model_info` → `weight_version`

### 3.4 `weight_version` -> `artifact_key` resolution (scheme-2, MUST)

SGLang resolves the artifact key from `weight_version` using `--model-loader-extra-config`:

```json
{
  "tensorcast_key_template": "model:{model_name}:v{weight_version}",
  "tensorcast_model_name": "qwen3_8b"
}
```

Rule:

- IS resolves `artifact_key = tensorcast_key_template.format(model_name=..., weight_version=V)`.
- IS injects the resolved key as `model_loader_extra_config["tensorcast_artifact_key"]` for that reload attempt.

### 3.5 Artifact + meta availability (MUST)

IS MUST open the artifact by key and MUST be able to query metadata before loading:

```python
artifact = tc.artifact(
  key=artifact_key,
  fallback=tc.FallbackOptions(allow_disk=True, prefer="auto"),
)
artifact.describe()
```

Notes:

- fallback is intent-only; IS MUST NOT provide an explicit disk path.

### 3.6 No `from_disk` fallback in the online update path (MUST NOT)

After startup/initialization, Publisher-triggered online updates MUST NOT fall back to importing from a local folder (`tc.from_disk(...)`). If `weight_version` cannot be resolved to a key or the key cannot be opened, reload MUST fail atomically.

---

## 4) Observability (MUST/SHOULD)

Publisher MUST log (per publish):

- `model_name`, `weight_version`, `artifact_key`, `artifact_id` (if available)

SGLang MUST log (per reload attempt):

- `model_name`, `weight_version`, `artifact_key`
- reload outcome (success/failure) and duration

---

## 5) Lifecycle & Garbage Collection (MUST)

### 5.1 Apply acknowledgement (MUST)

Publisher MUST NOT garbage-collect older versions until it has positive evidence that the target SGLang instance has applied the new version.

Standard acknowledgement mechanism:

- Publisher triggers reload via `POST /update_weights_from_tensorcast`.
- Publisher polls `GET /model_info` until `weight_version == requested_version`.

### 5.2 Retention window (MUST)

Publisher MUST retain a rollback window (policy-defined), e.g. keep the last `K` versions or keep versions for at least `T` minutes/hours after they are superseded.

---

## 6) Reference Flow (Standard v1)

### 6.1 Publisher reference algorithm

1. Choose `weight_version` (monotonic for this `model_name`).
2. Compute `artifact_key = "model:{model_name}:v{weight_version}"`.
3. `tc.put(..., key=artifact_key, policy="durable")`.
4. Wait for persistence to complete successfully.
5. `POST /update_weights_from_tensorcast` with
   `{ "weight_version": weight_version, "artifact_key": artifact_key }`.
6. Poll `GET /model_info` until it reports `weight_version`.
7. After a retention window, garbage-collect versions older than the rollback window.

### 6.2 SGLang reference algorithm

On `POST /update_weights_from_tensorcast(weight_version=V)`:

1. Reject missing/invalid `weight_version` in Tensorcast mode.
2. Enforce monotonic version:
   - if `V < current_version`: reject (`409 Conflict`),
   - if `V == current_version`: return `200 OK` (no-op),
   - if `V > current_version`: proceed.
3. Serialize reload requests (single in-flight).
4. Gate inference (pause/drain/abort optional), then apply reload.
5. Resolve `artifact_key` from config template and `V`.
6. Open artifact via `tc.artifact(key=artifact_key, fallback=...)` and call `describe()`.
7. Load/apply weights; on success update `current_version`, on failure keep old weights.
