# Tensorcast Remote Load-Weight Benchmark

This benchmark is self-contained under `sglang/benchmark/tensorcast/load_weight_remote/`.
It measures SGLang load latency when the model loader runs on a remote GPU worker.

Supported modes:

- `--load-format default`
  - only worker B is used
  - worker B reads model weights from the remote filesystem
  - before each trial, worker B clears the target model path cache state
- `--load-format tensorcast --topology-mode direct`
  - worker A runs Global Store + daemon A + publisher
  - worker B runs SGLang and connects directly to daemon A
  - this mode is currently experimental; on the current Tensorcast build it can fail cross-host with `cudaIpcOpenMemHandle(...): cudaErrorInvalidValue`
- `--load-format tensorcast --topology-mode relay`
  - worker A runs Global Store + daemon A + publisher
  - worker B starts daemon B for each trial and SGLang connects to daemon B
  - SGLang connects to daemon B via `127.0.0.1:<daemon_port>` on worker B so local CPU shared-memory fallback remains valid
  - daemon B is restarted every trial so each trial pulls from worker A again

## Output layout

Each invocation creates a unique run directory:

- `outputs/<run_id>/generated_configs/`
- `outputs/<run_id>/logs/`

A global append-only CSV is maintained at:

- `outputs/benchmark_results.csv`

Tensorcast service configs are generated per-run and use unique daemon/global-store log paths.
The underlying service log files are first written to `/data/<run_id>_*.log` and are copied into
`outputs/<run_id>/logs/` at the end of the run.

## Metrics

- `load_time_s`
  - from TP log marker `Load weight begin`
  - to the final weight-load completion marker across TP ranks
- `ready_time_s`
  - from `sglang.launch_server` process start
  - to first `/health` HTTP 200

Marker rules:

- `default`: `Load weight end.`
- `tensorcast`: `store.tensor_dict.materialized`

## Usage

Run from this folder:

```bash
cd sglang/benchmark/tensorcast/load_weight_remote
uv run python run_benchmark.py --help
```

### Default baseline

```bash
uv run python run_benchmark.py \
  --load-format default \
  --model-path /mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-14B \
  --tp-size 4 \
  --trials 1 \
  --port 30000 \
  --mem-fraction-static 0.85 \
  --log-level debug
```

### Tensorcast relay

```bash
uv run python run_benchmark.py \
  --load-format tensorcast \
  --topology-mode relay \
  --model-path /mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-14B \
  --model-name qwen3-14b \
  --weight-version 0 \
  --tp-size 4 \
  --trials 1 \
  --port 30000 \
  --mem-fraction-static 0.85 \
  --log-level debug
```

### Tensorcast direct

```bash
uv run python run_benchmark.py \
  --load-format tensorcast \
  --topology-mode direct \
  --model-path /mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-14B \
  --model-name qwen3-14b \
  --weight-version 0 \
  --tp-size 4 \
  --trials 1 \
  --port 30000 \
  --mem-fraction-static 0.85 \
  --log-level debug
```

## Notes

- The benchmark launches remote workers with `brainctl` and requires cluster access from the current machine.
- Worker A and worker B are forced onto different nodes. If they land on the same node, worker A is relaunched.
- The current Tensorcast daemon startup path requires a CUDA device for pinned-memory initialization. As a result, this benchmark defaults worker A to `gpu=1` even though worker A does not run SGLang.
- By default workers are cleaned up at the end. Use `--keep-workers` for debugging.
- Worker B cache clearing is best-effort and depends on host permissions and filesystem behavior.
