# Tensorcast Update-Weight Benchmark

This benchmark is self-contained under `sglang/benchmark/tensorcast/update_weight/`.
It measures online weight update latency after a server has already been launched.

## What it benchmarks

Two update paths:

- `load_format=tensorcast`: `POST /update_weights_from_tensorcast`
- `load_format=default`: `POST /update_weights_from_disk`

For `trials=N`, each trial updates to target version `k` (`k=1..N` when start version is `0`).

## Metrics

- `load_time_s`:
  - from TP log marker `Update engine weights online from ... begin.`
  - to TP log marker `Update weights end.` (last TP rank completion)
- `ready_time_s`:
  - from update HTTP request send time
  - to HTTP response `200` receive time

Notes:

- Publish/setup time is not included in any metric.
- Any trial failure aborts the benchmark immediately.
- CSV only appends successful trial rows.

## Output

- Server log: `outputs/logs/*_server.log`
- Per-trial logs: `outputs/logs/*_trialXXX.log`
- Global append-only CSV:
  `outputs/benchmark_results.csv`

CSV columns:

- `timestamp`
- `run_id`
- `trial_id`
- `target_weight_version`
- `model_path`
- `model_name`
- `tp_size`
- `load_format`
- `endpoint`
- `load_time_s`
- `ready_time_s`
- `status`
- `log_path`

## Usage

Run directly inside this folder:

```bash
cd sglang/benchmark/tensorcast/update_weight
uv run python run_benchmark.py --help
```

### 1) Tensorcast update benchmark

```bash
uv run python run_benchmark.py \
  --load-format tensorcast \
  --model-path Qwen/Qwen3-8B \
  --model-name qwen3-8b \
  --weight-version-start 0 \
  --tp-size 4 \
  --trials 3 \
  --port 30000 \
  --mem-fraction-static 0.7 \
  --log-level debug \
  --tensorcast-nvidia-lib-dirs /usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64
```

Tensorcast mode behavior:

- Starts Global Store + Store Daemon.
- Pre-publishes versions `v0..vN` (`N=trials`) with `WeightPublisher.publish(...)`.
- Launches server with `--weight-version 0`.
- Runs trial updates to `v1..vN` through `/update_weights_from_tensorcast`.
- If `--tensorcast-nvidia-lib-dirs` is provided, keep
  `/usr/local/cuda-12.9/compat` before `/usr/local/nvidia/lib64`.
- Default tensorcast configs are loaded from `./configs/`.

### 2) Baseline update benchmark

```bash
uv run python run_benchmark.py \
  --load-format default \
  --model-path Qwen/Qwen3-8B \
  --weight-version-start 0 \
  --tp-size 4 \
  --trials 3 \
  --port 30000 \
  --mem-fraction-static 0.7 \
  --log-level debug
```

Default baseline behavior:

- Launches server with initial `--weight-version 0`.
- Runs trial updates to version `1..N` via `/update_weights_from_disk`.
- Keeps `model_path` fixed across all trials and only changes `weight_version`.

## Fair comparison guidance

For baseline vs tensorcast comparison, keep identical:

- `model_path`
- `tp_size`
- `mem_fraction_static`
- `port`
- `log_level`
- any `extra_server_args`

Only change update path (`load_format` and endpoint behavior).

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
