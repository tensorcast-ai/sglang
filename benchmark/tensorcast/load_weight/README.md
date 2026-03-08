# Tensorcast Load-Weight Benchmark

This benchmark is self-contained under `sglang/benchmark/tensorcast/load_weight/`.
It compares SGLang startup latency for:

- `load_format=tensorcast`
- `load_format=default` (baseline)

## What it measures

- `load_time_s`:
  - from TP log marker `Load weight begin`
  - to weight-load completion marker in TP logs
- `ready_time_s`:
  - from `sglang.launch_server` process start
  - to first `/health` HTTP 200

Marker rules:

- `default`: `Load weight end.`
- `tensorcast`: `store.tensor_dict.materialized`

For TP>1, completion is taken at the last TP rank marker.

## Output

- Trial logs: `outputs/logs/`
- Global append-only CSV: `outputs/benchmark_results.csv`

CSV columns:

- `timestamp`
- `run_id`
- `trial_id`
- `model_path`
- `model_name`
- `tp_size`
- `weight_version`
- `load_format`
- `port`
- `load_time_s`
- `ready_time_s`
- `status`
- `error_message`
- `artifact_id`
- `log_path`

## Usage

Run directly inside this folder:

```bash
cd sglang/benchmark/tensorcast/load_weight
uv run python run_benchmark.py --help
```

### 1) Tensorcast benchmark

```bash
uv run python run_benchmark.py \
  --load-format tensorcast \
  --model-path Qwen/Qwen3-32B \
  --model-name qwen3-32b \
  --weight-version 1 \
  --tp-size 4 \
  --trials 5 \
  --port 30000 \
  --mem-fraction-static 0.7 \
  --log-level debug
```

Notes:

- Script starts Tensorcast Global Store + Store Daemon internally.
- Script publishes weight artifact once per run/config, then reuses it for all trials.
- Script does not clear caches between trials (hot-cache effect is intentional).
- Script enforces CUDA/NVRTC runtime library env before Tensorcast startup.
- Default tensorcast configs are loaded from `./configs/`.

### 2) Default baseline benchmark

```bash
uv run python run_benchmark.py \
  --load-format default \
  --model-path Qwen/Qwen3-32B \
  --weight-version 0 \
  --tp-size 4 \
  --trials 5 \
  --port 30000 \
  --mem-fraction-static 0.7 \
  --log-level debug
```

## Fair comparison guidance

For baseline vs tensorcast comparison, keep the following identical:

- `model_path`
- `tp_size`
- `mem_fraction_static`
- `port`
- `log_level`
- any `extra_server_args`

Only change `--load-format` (and corresponding `weight_version` / `model_name` as needed).

## Environment and UV project selection

The script supports explicit runtime selection:

- `--uv-project-root /path/to/project`:
  choose which `pyproject.toml` context `uv run` uses.
- `--env-file /path/to/env.list`:
  load child-process environment overrides from `KEY=VALUE` lines.
- `--env KEY=VALUE` (repeatable):
  add one-off environment overrides from CLI.

Example:

```bash
uv run python run_benchmark.py \
  --load-format tensorcast \
  --model-path /path/to/model \
  --model-name qwen3-8b \
  --uv-project-root /path/to/your/sglang/repo \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --env-file /path/to/bench.env
```
