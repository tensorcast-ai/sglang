# Tensorcast Remote Update-Weight Benchmark

This benchmark is self-contained under `sglang/benchmark/tensorcast/update_weight_remote/`.
It measures online weight-update latency when SGLang runs on a remote GPU worker.

Supported modes:

- `--load-format default`
  - only worker B is used
  - worker B runs SGLang
  - before each update trial, worker B clears the target model path cache state so each version update is a cold remote-filesystem read
- `--load-format tensorcast --topology-mode relay`
  - worker A runs Global Store + daemon A + publisher
  - worker B runs daemon B + SGLang
  - SGLang connects to daemon B via `127.0.0.1:<daemon_port>`
  - worker A pre-publishes versions `v_start .. v_start+trials`
- `--load-format tensorcast --topology-mode direct`
  - worker A runs Global Store + daemon A + publisher
  - worker B runs SGLang and connects directly to daemon A
  - this mode is kept for parity with `load_weight_remote`, but is currently experimental

## Metrics

- `load_time_s`
  - from TP log marker `Update engine weights online from ... begin.`
  - to the final per-trial completion marker across TP ranks
  - `tensorcast`: `store.tensor_dict.materialized`
  - `default`: `Capture cuda graph begin.` when `recapture_cuda_graph=true`, otherwise `Update weights end.`
- `ready_time_s`
  - from update HTTP request send time
  - to update HTTP response `200` receive time

Publish/setup time is excluded from all metrics.

## Output layout

Each invocation creates:

- `outputs/<run_id>/generated_configs/`
- `outputs/<run_id>/logs/`

Global append-only CSV:

- `outputs/benchmark_results.csv`

## Usage

Run from this folder:

```bash
cd sglang/benchmark/tensorcast/update_weight_remote
uv run python run_benchmark.py --help
```

### Tensorcast relay

```bash
uv run python run_benchmark.py \
  --load-format tensorcast \
  --topology-mode relay \
  --model-path /mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-14B \
  --model-name qwen3-14b \
  --weight-version-start 0 \
  --tp-size 1 \
  --trials 3 \
  --port 30000 \
  --mem-fraction-static 0.85 \
  --log-level debug
```

### Default baseline

```bash
uv run python run_benchmark.py \
  --load-format default \
  --model-path /mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-14B \
  --weight-version-start 0 \
  --tp-size 1 \
  --trials 3 \
  --port 30000 \
  --mem-fraction-static 0.85 \
  --log-level debug
```

## Notes

- Worker A and worker B are forced onto different nodes in relay mode.
- Worker A pre-publishes distinct Tensorcast versions by stamping a small tensor with a version-specific constant before each publish.
- Default baseline keeps `model_path` fixed and clears remote filesystem cache before every update trial so consecutive versions do not reuse the previous local cache state.
- By default workers are cleaned up at the end. Use `--keep-workers` for debugging.
