# Share-Local KV Benchmark

This benchmark measures local prefix-share TTFT improvement across two standard
SGLang instances running on the same remote GPU worker.

The request pattern is:

1. send a prompt to instance A
2. wait for the full response
3. optionally wait for a short settle interval
4. send the exact same prompt to instance B
5. compare TTFT between A and B

This benchmark does **not** model PD disaggregation or request-level KV
transfer.

## Layout

- `run_benchmark.py`
  - local orchestrator
- `request_driver.py`
  - remote request-pair driver
- `scripts/`
  - service helper scripts
- `outputs/<run_id>/`
  - run-local outputs, logs, generated configs, and raw results
- `outputs/benchmark_results.csv`
  - append-only run summary table

Shared KV benchmark utilities live under:

- `benchmark/tensorcast/kv/`

## Environment

Run from the SGLang repo root and use `~/tot/.venv`:

```bash
cd /home/i-zhouyuhan/tot/thirdparty/sglang
source /home/i-zhouyuhan/tot/.venv/bin/activate
export PYTHONPATH="$PWD/benchmark:$PWD:${PYTHONPATH:-}"
```

## Mooncake Example

Replace `<charged-group>` with the charged group for your cluster allocation.

```bash
/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline \
python -m tensorcast.kv.share_local.run_benchmark \
  --hicache-storage-backend mooncake \
  --brainctl-charged-group <charged-group> \
  --model-path /mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-14B \
  --tp-size 2 \
  --instance-a-cuda-visible-devices 0,1 \
  --instance-b-cuda-visible-devices 4,5 \
  --prompt-count 10 \
  --pair-rps 1.0 \
  --settle-ms 1000 \
  --max-new-tokens 32 \
  --keep-worker
```

## Validated Long-Prompt Mooncake Example

This configuration was validated on March 23, 2026 with:

- model: `Qwen3-32B`
- `tp-size=1`
- dataset: `LongBench/hotpotqa.jsonl`
- prompt filter: `5000 <= length <= 35000`
- HiCache layout: `page_first_direct`
- Mooncake global segment size: `16gb`
- result: `10/10` successful pairs showed lower TTFT on instance B
- mean TTFT improvement: `1415.17 ms`

The validated summary artifact is:

- `outputs/20260323-112623_mooncake_tp1_pairs10/summary.json`

Run the full command directly:

```bash
/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline \
python -m tensorcast.kv.share_local.run_benchmark \
  --hicache-storage-backend mooncake \
  --brainctl-charged-group <charged-group> \
  --model-path /mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-32B \
  --tp-size 1 \
  --instance-a-cuda-visible-devices 0 \
  --instance-b-cuda-visible-devices 4 \
  --data-path benchmark/tensorcast/kv/dataset/LongBench/hotpotqa.jsonl \
  --min-prompt-chars 5000 \
  --max-prompt-chars 35000 \
  --prompt-count 10 \
  --pair-rps 0.5 \
  --settle-ms 10000 \
  --max-new-tokens 16 \
  --hicache-mem-layout page_first_direct \
  --hicache-ratio 2.0 \
  --hicache-size-gb 0 \
  --mem-fraction-static 0.85 \
  --request-timeout-s 1800 \
  --instance-ready-timeout-s 3600 \
  --port-a 35000 \
  --port-b 35001 \
  --mooncake-global-segment-size 16gb \
  --keep-worker
```

For this long-prompt Mooncake path, prefer `page_first_direct`. The earlier
`page_first` runs on the same benchmark harness hit repeated Mooncake
`TRANSFER_FAIL` errors and did not show stable TTFT improvement.

## Tensorcast Example

The Tensorcast backend is reserved for future SGLang HiCache integration. The
CLI shape is already defined so the benchmark topology stays stable:

```bash
/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline \
python -m tensorcast.kv.share_local.run_benchmark \
  --hicache-storage-backend tensorcast \
  --tensorcast-daemon-mode share \
  --brainctl-charged-group <charged-group> \
  --model-path /mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-14B \
  --tp-size 2 \
  --instance-a-cuda-visible-devices 0,1 \
  --instance-b-cuda-visible-devices 4,5 \
  --prompt-count 10 \
  --pair-rps 1.0 \
  --settle-ms 1000 \
  --max-new-tokens 32 \
  --keep-worker
```

At the moment, the Tensorcast path is not yet functionally wired for prefix
share and is expected to remain a service-lifecycle harness until the SGLang
integration lands.
