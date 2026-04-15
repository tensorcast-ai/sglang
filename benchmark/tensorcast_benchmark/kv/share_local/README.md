# Share-Local KV Benchmark

This benchmark measures local prefix-share TTFT improvement across two standard
SGLang instances running on the same remote GPU worker.

The request pattern depends on `--wait-for-source-publication-drain`:

1. With `--wait-for-source-publication-drain`:
   - send a prompt to instance A
   - wait for the full response
   - wait until source publication drains
   - optionally wait for a short settle interval
   - send the exact same prompt to instance B
2. Without `--wait-for-source-publication-drain`:
   - schedule instance-A requests at the configured `--pair-rps`
   - schedule the matching instance-B request `--settle-ms` later
   - allow A/B requests from different pairs to overlap in time

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

- `benchmark/tensorcast_benchmark/kv/`

## Environment

Run from the SGLang repo root and use `~/tot/.venv`:

```bash
cd /home/i-zhouyuhan/tot/thirdparty/sglang
source /home/i-zhouyuhan/tot/.venv/bin/activate
export PYTHONPATH="$PWD/benchmark:$PWD:${PYTHONPATH:-}"
```

## Mooncake Example

Use `codesign` as the default charged group for the current validation workflow.

```bash
/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline \
python -m tensorcast_benchmark.kv.share_local.run_benchmark \
  --hicache-storage-backend mooncake \
  --brainctl-charged-group codesign \
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
python -m tensorcast_benchmark.kv.share_local.run_benchmark \
  --hicache-storage-backend mooncake \
  --brainctl-charged-group codesign \
  --model-path /mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-32B \
  --tp-size 1 \
  --instance-a-cuda-visible-devices 0 \
  --instance-b-cuda-visible-devices 4 \
  --data-path benchmark/tensorcast_benchmark/kv/dataset/LongBench/hotpotqa.jsonl \
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

The Tensorcast path is now wired as a real SGLang HiCache backend bring-up
path. The benchmark will:

- generate Tensorcast global-store / daemon config under `outputs/<run_id>/generated_configs/`
- start one global store and one or two daemons on the remote worker
- launch two ordinary SGLang instances with `--hicache-storage-backend tensorcast`
- run the same ordered prompt-pair TTFT experiment as the Mooncake path

For the short prompts used by `share_local`, the Tensorcast harness now injects
`prefetch_threshold=1` into `--hicache-storage-backend-extra-config` by
default. This is benchmark-only behavior so the HiCache storage prefetch path is
actually exercised with small prompts; override it with
`--tensorcast-prefetch-threshold` if needed.

Tensorcast runs in this benchmark must use:

- `--hicache-mem-layout page_blob_direct`
- `--hicache-io-backend direct`
- `--tensorcast-host-allocator-enabled` for the allocator-backed zero-copy host slab path

Do not reuse Mooncake's `page_first_direct` setting here. `page_first_direct`
is valid for Mooncake, but Tensorcast `share_local` only supports
`page_blob_direct`.

Note that `page_blob_direct` alone is not enough to remove `host_fill_ms` from
Tensorcast batch get. The zero-copy path only activates when the Tensorcast host
allocator is enabled, so the benchmark must also pass
`--tensorcast-host-allocator-enabled` together with
`--hicache-io-backend direct`.

```bash
/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline \
python -m tensorcast_benchmark.kv.share_local.run_benchmark \
  --hicache-storage-backend tensorcast \
  --tensorcast-daemon-mode share \
  --hicache-mem-layout page_blob_direct \
  --hicache-io-backend direct \
  --tensorcast-host-allocator-enabled \
  --brainctl-charged-group codesign \
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

Use `--tensorcast-daemon-mode separate` to model one daemon per instance on the
same worker. The primary validation target remains a same-node `tp=2`
two-instance benchmark such as `Qwen3-32B`.

This path is still an integration bring-up rather than a performance-tuned
backend. Expect correctness-first behavior and heavier per-page RPC traffic than
the eventual optimized implementation.

## Validated Long-Prompt Tensorcast Example

This configuration was validated on March 24, 2026 with:

- model: `Qwen3-32B`
- `tp-size=2`
- dataset: `LongBench/hotpotqa.jsonl`
- prompt filter: `length <= 35000`
- Tensorcast daemon mode: `share`
- HiCache layout: `page_blob_direct`
- HiCache IO backend: `direct`
- Tensorcast host allocator: enabled
- result: `1/1` successful pairs showed lower TTFT on instance B
- mean TTFT improvement: `67.75 ms`

The validated summary artifact is:

- `outputs/20260324-213808_tensorcast_tp2_pairs1/summary.json`

Run the full command directly:

```bash
/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline \
python -m tensorcast_benchmark.kv.share_local.run_benchmark \
  --hicache-storage-backend tensorcast \
  --tensorcast-daemon-mode share \
  --hicache-mem-layout page_blob_direct \
  --hicache-io-backend direct \
  --tensorcast-host-allocator-enabled \
  --existing-worker-process ws-ae2b460be336450b-worker-mjqbd \
  --model-path /mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-32B \
  --tp-size 2 \
  --instance-a-cuda-visible-devices 0,1 \
  --instance-b-cuda-visible-devices 4,5 \
  --data-path benchmark/tensorcast_benchmark/kv/dataset/LongBench/hotpotqa.jsonl \
  --max-prompt-chars 35000 \
  --prompt-count 1 \
  --pair-rps 0.1 \
  --settle-ms 10000 \
  --max-new-tokens 8 \
  --request-timeout-s 1800 \
  --instance-ready-timeout-s 3600 \
  --extra-server-args "--log-level debug --log-requests" \
  --port-a 35020 \
  --port-b 35021 \
  --keep-worker
```

## How To Prove Prefix Reuse

The benchmark sends an explicit `rid` for each request pair, but that `rid` is
only used for request/result correlation. SGLang HiCache prefix reuse is driven
by token-prefix/page-hash identity, not by request id. The important invariant
is that the two instances receive the exact same prompt text so they derive the
same page-hash chain.

Do not use `meta_info.cached_tokens` as proof for this benchmark. In the
validated Tensorcast run it remains `0` even when storage-backed prefetch
happens successfully.

For the measured request pair, the proof sequence is:

1. In instance-A logs, find repeated source-publication lines during the
   formal request window. Mooncake emits `stable_dram upload`; Tensorcast emits
   `Tensorcast batch_set_v1`. This shows the source instance is publishing
   reusable pages into the storage backend.
2. In instance-B logs, for the same formal `rid`, find
   `HiCache storage hit query ... hit_tokens > 0`.
3. Confirm that the same request then logs `Prefetching ... pages for request`
   and repeated `Artifact loaded ...` lines.

In the validated run `20260324-213808_tensorcast_tp2_pairs1`, the formal
measured request on instance B shows:

- `HiCache storage hit query ... hit_tokens=65`
- `Prefetching 65 pages for request ...`
- repeated `Artifact loaded ...`

Those request-scoped logs are the primary proof that instance B reused prefix
pages from the shared Tensorcast substrate.
