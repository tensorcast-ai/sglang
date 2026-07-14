# tc_router Benchmark Architecture

## 1. Goal

`tc_router` is an end-to-end benchmark whose purpose is to **showcase
Tensorcast programmability** by building, in user-space Python, a request
router that uses Tensorcast's request-level KV transfer
(`publish` / `hydrate`) to do something the production SGLang gateway
cannot: **actively migrate a session's KV cache from a hot serving instance
to a cold one, before the next turn of that session arrives**.

The benchmark answers, on a realistic multi-turn workload:

- Does an active-migration router achieve lower TTFT and support more
  concurrent users than the strongest passive router strategies
  (`cache_aware` and `load_aware`) shipped by `sgl-model-gateway`?
- How much of the win is attributable to the Tensorcast control-plane
  programmability vs. merely having a shared KV substrate?

The benchmark is intentionally framed as a **programmability showcase**, not
as a routing-policy paper. The policy used by the Tensorcast router is one
example of what user-space code can do on top of `publish` / `hydrate`.

## 2. Non-Goals

The following are explicitly out of scope for v1:

- PD (prefill / decode) disaggregation
- Mixed-TP topologies (e.g., some instances at TP=1 alongside others at
  TP=2). The launch harness supports `tp_size` as a parameter and the
  Qwen3-32B default exercises TP=2, but every instance in a single run
  uses the same TP shape.
- Tuning Tensorcast internals (page size, prefetch policies, etc.)
- Multi-model fleets
- Beating production gateways on throughput in low-cache-reuse workloads
- Generic single-turn benchmarks (TTFT under uniform short prompts, etc.)
- Resource acquisition for remote providers. The current local provider
  describes the current host directly; future remote providers should
  consume already-acquired workers described in a cluster YAML (see
  § 9.1).
- Comparing routing-policy algorithm quality in the abstract; we compare
  concrete gateway policies vs. concrete tc_router policy

## 3. What Is Being Compared

Four routing configurations are exercised against the **same workload, the
same N serving instances, and the same model**:

| Config | Component in front | Notes |
|---|---|---|
| `gw_load_aware` | sgl-model-gateway, `--policy power_of_two` | No cache awareness |
| `gw_cache_aware` | sgl-model-gateway, `--policy cache_aware` | Prefix affinity + imbalance / low-match fallback |
| `gw_load_aware_mooncake` | sgl-model-gateway, `--policy power_of_two`, SGLang HiCache backed by Mooncake | Isolates the value of a shared KV substrate from the value of programmability |
| `tc_router` | Our user-space Python router, Tensorcast-backed | Eager session-level KV migration via `publish` / `hydrate` |

The model, TP size, instance count, placement plan, and workload are
identical across configs. The serving launch profile is identical for
the plain gateway baselines (`gw_load_aware`, `gw_cache_aware`). The
Mooncake baseline uses the same fleet shape but starts SGLang with
HiCache backed by Mooncake. The Tensorcast router uses the same placement
plan and TP shape, but it is a distinct Tensorcast serving profile:
SGLang starts with Tensorcast HiCache in explicit request-transfer mode,
one Tensorcast daemon runs on each worker, a global store runs on the
configured service worker, and the in-process router drives
`publish` / `hydrate` plans through the Tensorcast runtime.

`gw_load_aware_mooncake` is the most important baseline: it tells us
whether the win comes from "anyone can pull prefix pages out of a shared
pool" (substrate) or from "the router actively shapes where KV lives"
(programmability).

## 4. Core Semantics

### 4.1 What is being measured

Per turn (one `/v1/chat/completions` request inside a session), the
benchmark records:

- TTFT
- end-to-end latency
- `prompt_tokens`, `cached_tokens`
- which instance served the request
- whether the request consumed a `hydrate`'d prepared bundle
- whether the session was just migrated before this turn

Per migration the Tensorcast router performed, the benchmark records:

- timestamp, session_id, source instance, target instance
- `publish` latency, `hydrate` latency
- estimated KV transfer bytes
- whether the resulting bundle was eventually consumed by a subsequent
  turn (= "migration utilization")

### 4.2 Headline plot

- x-axis: target concurrent active sessions `C_target` (also annotate
  the derived steady-state RPS on a secondary axis)
- y-axis: TTFT distribution. Collect all values; pick mean / P50 / P95 /
  P99 at write-up time
- curves: one per routing config under test

### 4.3 Expected curve shape

- `C_target / N ≈ 1`: each session has effectively its own instance, no
  contention. All four configs are indistinguishable. This is the sanity
  region.
- `1 < C_target / N ≤ ~2`: hotspots become statistically common. `cache_aware`
  starts triggering its imbalance-fallback, paying long re-prefill on
  migrated requests. `tc_router` rebalances ahead of time. Curves separate.
- `2 < C_target / N ≤ ~5`: sticky-session is forced into hotspots
  frequently. This is the most informative region of the plot.
- `C_target / N ≳ 5`: system-wide saturation; we expect `tc_router` to
  either sustain higher `C_target` at a given TTFT SLO, or fail more
  gracefully.

## 5. Workload Model

### 5.1 Dataset

[`SWE-Gym/OpenHands-Sampled-Trajectories`](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories)
on Hugging Face (MIT). Real agent rollouts on the SWE-Gym task pool,
collected by running the OpenHands agent against ~2,438 real-world
Python repository tasks. Three Parquet shards, ~290 MB total compressed
(~1.4 GB decompressed), 6,055 trajectories.

The session pool for our benchmark is built from this dataset.

#### 5.1.1 Schema

Each trajectory row:

| Field | Type | Notes |
|---|---|---|
| `instance_id` | string | Links back to the SWE-Gym task |
| `run_id` | string | Identifies the agent run (model + sampling config) |
| `resolved` | bool | Whether the agent resolved the issue |
| `messages` | list of structs | OpenAI chat format: `role`, `content`, `name`, `tool_call_id`, `tool_calls` (full structured form) |
| `tools` | list of structs | OpenAI-format function specs the agent had access to (e.g. `str_replace_editor`, `bash`) |
| `test_result` | struct | Final test outcome and patch info |

`messages` covers all four roles (`system`, `user`, `assistant`, `tool`)
and preserves structured tool calls / tool responses. This is exactly
the format SGLang's `/v1/chat/completions` accepts; see § 5.2 for how
that decouples the benchmark from any specific target-model chat
template.

#### 5.1.2 Why this dataset

It is the only public dataset that simultaneously satisfies all our
workload requirements:

- multi-turn (median 31 turns per trajectory)
- real prompt text (real Python code, real GitHub issues, real test
  output)
- agentic structure (system / user / assistant+tool_calls / tool / ...,
  prefix grows turn by turn as actions and observations accumulate)
- long context (median ~11K tokens, p95 ~45K, max ~156K)
- pool size sufficient for our `C_target` sweep (thousands after
  filtering)

ShareGPT has the multi-turn structure but no agentic prefix-growth
loop and no long-context. WildChat-1M has per-turn timestamps but
trajectories average only ~2.5 turns. Mooncake / Azure / BurstGPT
traces carry timestamps and token counts but no real prompt text, so
they cannot drive realistic radix-cache reuse.

#### 5.1.3 Distribution measured by us (o200k_base tokenizer, all 6,055 trajectories)

The o200k_base tokenizer is the GPT-4o tokenizer; since 96.2% of
trajectories were generated by `gpt-4o-2024-08-06` (the rest by
`claude-3-5-sonnet-20241022`), this matches the original token
accounting at collection time.

| Metric | min | p25 | **median** | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| turn count | 7 | 9 | **31** | 61 | 101 | 101 |
| total tokens per trajectory | 493 | 1,186 | **11,003** | 22,658 | 44,688 | 156,430 |
| final-prompt tokens (largest single LLM input within session) | 491 | 1,161 | **10,911** | 22,505 | 44,514 | 155,992 |

`resolved = True` rate: 8.1% (491 / 6,055). The remainder either
exceeded the agent's 101-turn maxiter cap (visible as the saturation at
turn count = 101) or terminated unsuccessfully. We retain both: failed
rollouts are valid prompt-sequence workload for our purposes.

For our actual deployment with a different tokenizer (Qwen3 tokenizer
in the v1 plan), the same content typically produces ~10–20% different
token counts. The numbers above are a sizing reference; the exact
filter cut-points used at benchmark time should be re-validated with
the deployment-time tokenizer.

#### 5.1.4 Filter for v1

Default pool: `turns ≥ 8 AND total_tokens ≥ 8000`. This yields
**3,459 trajectories (57.1%)** — comfortably beyond what the
`C_target` sweep can recycle, while excluding short or aborted rollouts
that don't exhibit meaningful prefix growth.

Stress subset: `turns ≥ 16 AND total_tokens ≥ 32000` retains
**792 trajectories (13.1%)** with very long context (final-prompt p95
in the 60–80K range), used in a secondary long-context sensitivity
experiment.

### 5.2 Traffic generation and trajectory replay

#### 5.2.1 Faithful prompt replay

Each trajectory in the pool is replayed as a multi-turn session against
the router. We use **faithful prompt replay**:

- Within one trajectory, every `assistant` message at position `k`
  corresponds to one LLM call. The prompt for that call is
  `messages[0:k]`: everything in the trajectory before that assistant
  message.
- The router issues that LLM call and waits for the model's response.
- Our model's response is **discarded**.
- The next turn's prompt is constructed using the trajectory's
  *originally-recorded* `messages[k]` (the original assistant turn) and
  the subsequent `tool` / `user` messages, up to the next `assistant`
  boundary.
- We sleep `LogNormal(...)` (see § 5.3) and then issue the next turn.

Discarding our model's output is what makes the comparison fair:
all four routing configurations see byte-identical prompt streams, so
prefix-cache reuse is directly comparable. We are not measuring agent
task success; we are measuring router behavior on a realistic
long-context multi-turn prompt sequence.

The trajectories were originally generated by GPT-4o (96%) and
Claude-3.5-Sonnet (4%). At benchmark time, SGLang serves them with the
target deployment model (default Qwen3-32B; see § 7) by applying the
*target model's* chat template and tool-call serialization at request
time. Whose-model-generated-the-trajectory is irrelevant to the
quantity we measure.

#### 5.2.2 Why we use `/v1/chat/completions`, not `/generate`

The router posts every request via SGLang's `/v1/chat/completions`
(OpenAI-compatible) endpoint with the structured `messages` and `tools`
fields. We deliberately do **not** apply chat templates in benchmark
code.

Reasons:

- Each model family (Llama-3, Qwen3, DeepSeek, Mistral, Hermes-style,
  ...) has its own chat template AND its own tool-call serialization.
  Re-implementing them byte-for-byte against SGLang's internal formatter
  is brittle.
- A single byte mismatch with SGLang's internal formatter causes the
  radix tree to see different prefix bytes from what SGLang produces
  internally on subsequent turns, which silently breaks cache reuse and
  therefore breaks the entire comparison.
- The router code stays model-agnostic: switching between Qwen3-32B,
  Qwen3-14B, or any other supported model is purely a launch-config
  change. No router code changes.

Letting SGLang apply its own template keeps the comparison honest and
the router code clean.

#### 5.2.3 Tool calls in the trace: why they don't pollute the measurement

The trajectories carry full structured `tool_calls` (in `assistant`
messages) and `role=tool` responses. SGLang's `/v1/chat/completions`
endpoint accepts these natively and the target model's chat template
handles serialization. Two independent pieces have to be pinned down to
keep the measurement clean:

**Input side — chat-template tool serialization (we want this).**
The model's chat template (loaded from its `tokenizer_config.json`)
knows how to render `tools` schema + `tool_calls` + `role=tool`
messages into the model-specific prompt format (Qwen3 uses
`<tool_call>...</tool_call>` blocks, etc.). This is what makes the
prefix realistic and is what we want SGLang to do for us.

**Output side — `--tool-call-parser` (we deliberately disable this).**
SGLang has a startup flag `--tool-call-parser <name>` (`qwen`,
`llama3`, `json`, `xml`, ...) that, if set, parses the streamed model
output back into structured `tool_calls` in the response body. v1 of
this benchmark **does not enable any tool-call parser**:

- the model is free to emit text that looks like a tool call (it
  probably will, given the conversation pattern), but SGLang treats it
  as plain text;
- no per-token parser state machine, no buffering, no re-serialization
  on the output path → TTFT and streaming latency are not contaminated
  by parser behavior;
- the streamed text is discarded by the router (faithful-replay
  contract from § 5.2.1), so semantic correctness of the model's
  imitated tool call doesn't matter.

This separation is the answer to "won't the model try to call tools?":
yes it will try, but the attempt produces text only, the text is
discarded, and the next turn's prompt comes from the recorded
trajectory regardless. The model's output is causally disconnected
from anything the benchmark measures or feeds forward.

The SGLang launch command for every instance under test must therefore
**omit** `--tool-call-parser`. The benchmark validation step
(§ 12 step 1) will assert that the response payload's `tool_calls`
field is absent / empty for at least one sampled turn, as a guardrail
against accidentally enabling the parser via a config template.

#### 5.2.4 Concurrency model: closed-loop within session, controlled active session count

The traffic generator maintains a target steady-state of `C_target`
active sessions:

```text
async supervisor:
  every ε seconds, if |active| < C_target:
    pop next trajectory from session_pool
    spawn session_runner(trajectory), jitter start by Uniform(0, 2 s)

session_runner(traj):
  for k where traj.messages[k].role == "assistant":
    prompt_messages = traj.messages[0:k]
    rid = f"tcrouter:{traj.session_id}:turn{turn_idx:03d}"
    result = await router.generate(
        rid=rid,
        session_id=traj.session_id,
        messages=prompt_messages,
        tools=traj.tools,
        sampling_params={"max_tokens": clip(estimate_response_tokens(traj, k), 1, 512)},
    )
    # result is awaited but content is discarded; we use traj.messages[k]
    # and any subsequent tool/user messages to build the NEXT prompt.
    if k < last_assistant_index_in(traj):
      await asyncio.sleep(sample_inter_turn_delay())
  deactivate
```

- Within a session, turn `k+1` is only issued after turn `k`'s response
  completes and the inter-turn delay has elapsed. This is closed-loop
  and matches how the agent loop behaves in production deployments.
- Across sessions, new sessions are spawned only to refill `|active|`
  up to `C_target`. The benchmark does not pin the inter-arrival
  distribution; it pins **the steady-state concurrency**.
- A small uniform jitter (0..2 s) at session start avoids artificial
  bursts when many sessions are spawned at once.

### 5.3 Inter-turn delay distribution

The delay between consecutive LLM calls within a session is sampled
from a `LogNormal(μ, σ)` distribution. Three named presets are
defined; the benchmark picks one preset per run (or a custom
`(μ, σ)` pair).

#### 5.3.1 The three presets

| Preset | μ | σ | Median | Mean | P5 | P95 | Represents |
|---|---:|---:|---:|---:|---:|---:|---|
| `agent_fast` | 2.1 | 0.6 | 8.2 s | 9.8 s | 3.0 s | 22.0 s | Tight coding-agent loop (Cursor, Copilot inline, Claude Code in fast mode); short tool replies dominate |
| `agent_medium` (default) | 3.0 | 0.8 | 20.1 s | 27.7 s | 5.4 s | 75.0 s | Typical SWE agent with mixed view / edit / short-test tool exec |
| `agent_slow` | 4.1 | 1.0 | 60.3 s | 99.5 s | 11.6 s | 311 s | Long-tool-dominated workflows (full test suites, builds, container ops); research / planning agents |

`agent_slow` σ = 1.0 (intentionally wider than the other two)
because long tool runs themselves have higher variance — a build
might be 30 s or 5 min depending on what changed. Holding σ fixed
across presets would understate the realism of the slow regime.

#### 5.3.2 Why three regimes (and what they tell us)

Each preset hits a different operating regime that exercises a
different aspect of the comparison. Together they form a 3-act story:

| Regime | Expected behavior | What it shows |
|---|---|---|
| Fast | KV stays warm in HiRadix; even passive `cache_aware` routing reuses well. All four configs perform similarly. | Sanity. Verifies the setup isn't accidentally crippling baselines. |
| Medium | KV partially evicted between turns; some sessions reach hot instances. `tc_router` actively rebalances and shows a clear win. | Core experiment. The headline plot region. |
| Slow | KV largely evicted between turns; serving relies heavily on either substrate-backed retention (Mooncake / Tensorcast prefix-share) or active migration. | Separates substrate value (`gw_load_aware_mooncake` vs `gw_load_aware`) from programmability value (`tc_router` increment over `gw_load_aware_mooncake`). |

The headline `(C_target × TTFT)` plot uses `agent_medium` only.
A secondary "preset sweep" plot — fixed `C_target` (e.g., 12),
varying preset across `agent_fast` / `agent_medium` / `agent_slow` —
demonstrates robustness across the regime axis.

#### 5.3.3 Where the delay comes from in an agentic deployment

This delay does **not** represent "user typing speed". In a real
SWE-style agent deployment the inter-turn delay is dominated by:

- tool execution time. `str_replace_editor view` takes milliseconds,
  but test execution, build steps, container ops, and `bash` commands
  routinely take seconds to minutes (this is what `agent_slow`
  captures).
- IDE / harness round-trip when the human supervises (Claude Code,
  Cursor, internal agent UIs).
- LLM provider rate limiting and queueing.
- Brief human-in-the-loop checkpoints between agent steps.

These sources collectively produce a positive, right-skewed
distribution. Log-normal is a defensible parametric family for such
durations:

- Brown, Gans, Mandelbaum, Sakov, Shen, Zeltyn, Zhao,
  *"Statistical Analysis of a Telephone Call Center: A
  Queueing-Science Perspective"*, JASA, 2005 — log-normal as the
  canonical choice for positive-valued service-time durations.
- Malmgren, Stouffer, Motter, Amaral,
  *"A Poissonian explanation for the heavy tails in e-mail
  communication"*, PNAS, 2008 — log-normal for within-active-period
  inter-event times in digital communication.

Exponential places too much mass near zero (no real tool / harness
returns instantaneously); a normal allows negative values.

#### 5.3.4 Honest disclosure on calibration provenance

The three presets are **calibrated assumptions**, not refits to a
real per-step trace. There is no public dataset of production
SWE-agent per-step inter-call delays we can fit to. Adjacent
reference: ServeGen (arXiv:2505.09999, 2025) reports population-wide
inter-turn times concentrating around ~100 s with a long tail on a
DeepSeek-R1 production trace; that population mixes active and
abandoned sessions, so our presets sit on the active side of that
distribution by design. If a future trace becomes available with
explicit active-session stratification, the parameters should be
refit against it.

The presets are **fixed** within a run (no on-the-fly tuning), and
the chosen preset is recorded in `summary.csv` per row.

### 5.4 Sweep

- `C_target` ∈ a roughly geometric grid from `N_instances` to about
  `8 × N_instances`. Example for `N = 3`: `{3, 5, 8, 12, 18, 24}`.
- At each point, run `T_wall` wall-clock seconds (default `T_wall =
  600`), discard the first `T_warmup = 120` seconds.
- 3 trials per `(config, C_target)` point.

#### 5.4.1 Cell Isolation

Cells in a `C_target` sweep must not inherit serving-side cache state from
earlier cells. Otherwise a later, larger `C_target` can look better simply
because an earlier cell warmed the same trajectory prefixes. This benchmark
therefore treats each `(config, C_target, trial)` cell as a fresh cache
experiment while keeping model load cost under control.

The SGLang serving instances are **not restarted between cells**. They are
launched once for the run and torn down at the end. Restarting Qwen3-32B TP=2
for every cell would dominate experiment time with model-load overhead and
would measure fleet churn as much as routing behavior.

Instead, every cell boundary uses this protocol:

1. Stop admitting new workload sessions when the cell reaches its wall-clock
   deadline.
2. Await all active session tasks, including their in-flight streaming
   requests, so the cell's running and waiting requests drain naturally.
3. Tear down the per-cell front router. For `gw_*`, this means stopping
   `sgl-model-gateway`; for `tc_router`, this means closing the Python router
   object. This resets router-local state such as the gateway
   `cache_aware` approximate prefix tree.
4. POST `/flush_cache` directly to every SGLang instance endpoint, not through
   the gateway. The flush must succeed on every instance before the next cell
   starts.
5. Do not fail fast on the first non-200 flush response. SGLang returns failure
   when it still observes running or waiting requests, so the driver should
   poll all instances with a bounded retry loop until every instance returns
   HTTP 200. A timeout is an experiment failure because the next cell would not
   start from a clean KV state.
6. Start a fresh front router for the next cell, then run workload warmup and
   measurement for that cell.

`/flush_cache` clears SGLang's radix/KV-cache-side state; it does not reset
`sgl-model-gateway` policy memory. The per-cell gateway restart is therefore
required for `gw_cache_aware`, even when all SGLang instances have flushed
successfully.

Mooncake-backed cells add one extra isolation step after all SGLang
instances have accepted `/flush_cache`: POST
`/clear_hicache_storage_backend` directly to the SGLang instance
endpoints. Mooncake is a shared storage substrate, so clearing only the
front-router state and local SGLang cache state is not enough; otherwise
later `C_target` cells can observe prefix pages written by earlier cells.
The clear operation is backed by SGLang's HiCache storage API
(`MooncakeStore.clear()` calls the Mooncake store's `remove_all()`), so
it is lighter than restarting the Mooncake master and keeps model-load
cost out of the cell boundary. As with `/flush_cache`, the driver should
poll every endpoint, collect all non-200 responses for diagnostics, and
only proceed after all endpoints return HTTP 200. A timeout is an
experiment failure because the next cell would inherit substrate state.

`warmup_seconds` and `warmup_counts` are measurement-window controls, not a
separate cache preload phase. The workload may start immediately after the
cell's clean startup, but records produced before
`cell_start + warmup_seconds` or among the first `warmup_counts` completed
turn records are marked as warmup and excluded from `summary.csv`. The count
guard matters for sparse cells: with low concurrency and slow inter-turn
delays, the first measured request may complete after the time guard has
already elapsed, while it can still include SGLang post-flush/kernel warmup
cost. Using both guards preserves natural multi-turn session evolution while
preventing cold-start and post-flush transients from dominating the reported
TTFT and cached-token ratios.

### 5.5 What the workload deliberately does *not* include

- No artificial bursts. The narrative depends on the workload looking
  organic.
- No Zipfian session-pick. The trajectory pool is drawn in shuffled
  order. Hotspots emerge naturally from trajectory-length variance and
  inter-turn-delay variance, not from a synthetic skew.
- No mid-session abort. Every active session replays through to the
  trajectory's last assistant message.
- No live agent loop. Tool execution is not actually run; we replay the
  recorded tool outputs from the trajectory. This is the
  faithful-replay decision in § 5.2.1.

## 6. Router Contract (Tensorcast router)

### 6.1 Component graph

```text
[traffic generator]
       │  in-process call: router.generate(rid, session_id, messages, tools, ...)
       ▼
[tc_router]
       │  routes via /v1/chat/completions to chosen instance
       ▼
[SGLang instance 0]  [SGLang instance 1]  ...  [SGLang instance N-1]
       ▲                       ▲
       │  publish / hydrate via Tensorcast plans
       │
[Tensorcast runtime (in tc_router process)]
       │
[Tensorcast daemon per worker] ── [Tensorcast global store]
```

In v1 the router is **in-process** with the traffic generator. The
gateway-baseline configs are also driven through an in-process wrapper
that posts to the gateway's HTTP endpoint, so the comparison stays
apples-to-apples at the "router call" boundary.

### 6.2 Public router API

```python
class Router(Protocol):
    async def generate(
        self,
        *,
        rid: str,                    # unique per-turn request id
        session_id: str,
        messages: list[dict],          # OpenAI chat-format messages
        tools: list[dict] | None,      # OpenAI function specs, optional
        sampling_params: dict,
    ) -> GenerateResult: ...

    async def close(self) -> None: ...
```

Both the Tensorcast router and the gateway-baseline wrapper implement
this interface. Each wrapper internally posts to its serving endpoint
via `/v1/chat/completions` (OpenAI-compatible), carrying `rid` in the
SGLang chat-completions `rid` field. The traffic generator does not
know which router it is talking to and never sees a flattened prompt
string.

`GenerateResult` carries the streaming response text (discarded in
faithful-replay mode), TTFT, total latency, served-instance label, and
the SGLang `meta_info` (specifically `prompt_tokens`, `cached_tokens`).

#### 6.2.1 Gateway and Mooncake baseline contract

The gateway baselines expose the same in-process `Router.generate(...)`
contract to the workload generator. The wrapper posts each turn to
`sgl-model-gateway` at `/v1/chat/completions`; the gateway then routes
to one SGLang serving instance.

`gw_load_aware_mooncake` is not a new routing policy. Its front router is
the same gateway power-of-two load-aware policy used by
`gw_load_aware`:

```text
[traffic generator]
       │  in-process call: router.generate(...)
       ▼
[GatewayRouter wrapper]
       │  /v1/chat/completions
       ▼
[sgl-model-gateway --policy power_of_two]
       │
       ▼
[SGLang instance 0]  [SGLang instance 1]  ...  [SGLang instance N-1]
       │                       │
       └──────────────┬────────┘
                      ▼
      [SGLang HiCache storage backend = mooncake]
                      │
                      ▼
      [Mooncake master + HTTP metadata service]
```

The only semantic delta from `gw_load_aware` is the SGLang serving
profile: every SGLang instance in the Mooncake profile starts with
hierarchical cache enabled and `--hicache-storage-backend mooncake`.
This isolates the value of shared substrate-backed KV retention while
keeping the gateway's routing algorithm load-aware and cache-unaware.

The Mooncake master is a singleton service placed by
`cluster.yaml.service_placement.mooncake_master_worker_id`. It runs the
Mooncake master and HTTP metadata service in one process, following the
same contract used by `kv/share_remote`: SGLang instances connect to the
master server address for data-plane coordination and to the HTTP
metadata endpoint for metadata. The advertised host must be reachable
from every worker. If the placement worker's configured address is
loopback (`127.0.0.1`, `localhost`, or `0.0.0.0`), the launcher must
advertise a routable local IPv4 address instead; otherwise remote
workers would try to connect to themselves.

SGLang receives the Mooncake connection data through
`--hicache-storage-backend-extra-config`, not through per-instance JSON
files. The required payload fields are:

- `master_server_address`: `<mooncake_advertise_host>:<master_port>`
- `metadata_server`: `http://<mooncake_advertise_host>:<http_metadata_server_port>/metadata`
- `local_hostname`: the worker-reachable address for the SGLang process's
  worker
- `protocol`: `tcp` when `transport.use_rdma == false`, otherwise `rdma`
- `global_segment_size`: copied from `benchmark.yaml.mooncake`
- `device_name`: copied from `benchmark.yaml.mooncake` when non-empty

For Mooncake RDMA, "worker-reachable" means the IPv4 address attached to
the selected RDMA HCA's Linux netdev, not necessarily the worker's SSH or
HTTP control address. Some clusters expose a separate control network
(`10.x`) and RDMA network (`22.x`) plus internal interfaces that are not
mutually routable. If SGLang advertises the control address, or if
Mooncake Transfer Engine auto-binds its RPC listener to an unreachable
interface, `MooncakeStore.warmup()` can hang in `store.put()` until
Mooncake reports a transfer timeout. For RDMA runs with non-empty
`mooncake.device_name`, the driver resolves the first configured HCA with
`rdma link show`, maps it to its `netdev`, reads that netdev's IPv4 with
`ip -o -4 addr show`, and uses the resulting IP for both:

- `local_hostname` in SGLang's Mooncake extra config
- `MC_TCP_BIND_ADDRESS` in the SGLang process environment

`MC_TCP_BIND_ADDRESS` is a Mooncake Transfer Engine environment variable;
despite the name, it controls the RPC bind address used during RDMA
setup as well. The Mooncake master and HTTP metadata service still
advertise their normal worker-reachable control address because those
endpoints are used for metadata/control-plane traffic, not RDMA data
transfer.

The benchmark intentionally does **not** set Mooncake
`prefetch_threshold`; SGLang's default value is used so this baseline
tracks upstream SGLang behavior rather than a benchmark-specific tuning.

#### 6.2.2 Session identity for Tensorcast request migration

The Tensorcast router uses the OpenAI-compatible
`/v1/chat/completions` endpoint directly on each selected SGLang
instance. It does **not** extend the OpenAI request JSON body with a
benchmark-specific session field. Instead, every request that tc_router
sends to SGLang carries:

```http
X-SMG-Routing-Key: <session_id>
```

SGLang already extracts this header into its internal `routing_key`.
When the Tensorcast HiCache backend is configured with:

```json
{
  "tensorcast_kv_mode": "explicit_request_transfer",
  "logical_session_id_source": "routing_key"
}
```

the request-bundle manager records the same value as
`logical_session_id`. This gives the source instance enough metadata to
publish a bundle for "the latest request of session S", and gives the
target instance enough metadata to find the longest prepared-prefix
bundle for the next request of session S.

The benchmark's `rid` remains the per-turn engine request id:

```text
tcrouter:<session_id>:turn<turn_index>
```

`rid` is used for `publish(engine_request_id=...)` and for per-turn
traceability. It is intentionally **not** the primary session contract.
The routing-key header is the primary session contract because it avoids
embedding a tc_router-specific `rid` parser inside SGLang. A
deployment that cannot set `X-SMG-Routing-Key` may opt into SGLang's
configured `logical_session_id_rid_regex` fallback, but that fallback is
not used by this benchmark.

### 6.3 Internal contract of `tc_router`

The Tensorcast router maintains:

- `session_state: dict[session_id, SessionState]` where
  `SessionState` records `{home_instance, last_active_ts, turn_count,
  last_prompt_tokens, last_engine_request_id, last_published_manifest,
  pending_migration}`. `last_engine_request_id` is the `rid` we sent
  on the most recent turn of the session and is what we hand to
  `publish(engine_request_id=...)` when migrating.
- `instance_loads: dict[instance_id, LoadSample]` periodically refreshed
  from each instance's SGLang serving HTTP endpoint
  (`/get_server_info` / pending-request count; reuses the `get_load`
  pattern already implemented in
  `tot_experiment/src/tot_experiment/sglang_client.py`)
- `pending_migrations: dict[session_id, MigrationFuture]` so an arriving
  request can wait on or supersede an in-flight migration

On each `generate(rid, session_id, messages, tools, ...)` call:

1. If `session_id` has a pending migration, wait for that migration up to
   `pending_migration_wait_timeout_s`. This is required for the E2E
   migration smoke test: the first post-migration turn should not race
   ahead to the old home before `hydrate` finishes.
2. Resolve target instance:
   - if `session_id` has a `home_instance`, route there
   - else assign `home_instance` using the policy's initial-placement hook
     and route there
3. Build the OpenAI chat-completions request with the workload-provided
   unique `rid` in the SGLang `rid` field.
4. POST to the chosen instance with `X-SMG-Routing-Key: <session_id>`.
   This is the only session-scoped metadata sent to SGLang.
5. Record per-turn metrics. If this turn consumed the next request after
   a successful migration, set `was_just_migrated=true`; if the request's
   `cached_tokens > 0` and the router has a matching consumed migration,
   set `used_hydrated_bundle=true`.
6. On success, update `last_active_ts`, `turn_count`,
   `last_prompt_tokens`, and `last_engine_request_id = rid`.
7. Let the policy decide whether a migration should be scheduled. The
   initial E2E smoke policy schedules immediately after a successful
   source turn; later load-aware policies may schedule from a background
   tick.

The full rebalancer path is a background task that ticks every
`rebalance_period_ms` (default 500 ms):

1. Refresh `instance_loads`.
2. Call `policy.should_rebalance(instance_loads, session_state) ->
   list[MigrationDecision]`.
3. For each decision `(session_id, source, target)`:
   - issue a `publish` on source using `last_engine_request_id`, then
     `hydrate` on target through the Tensorcast runtime
   - on success, update `session_state[session_id].home_instance = target`
   - emit a `MigrationEvent` to the output log
4. The rebalancer respects a per-session cooldown so the same session is
   not migrated back and forth within `migration_cooldown_s` (default
   30 s).

The first runnable migration implementation is allowed to use the same
execution machinery without a sophisticated background policy. A
`migrate_once_after_turn` smoke policy may schedule exactly one migration
for a session immediately after `turn_count >= after_turn_count` and
`last_engine_request_id` is available. This separates the correctness of
session-scoped request migration from the quality of the future
load-balancing policy.

### 6.4 Policy interface

Policy is left intentionally pluggable. Four hooks, all returning pure
decisions (no I/O):

```python
class Policy(Protocol):
    def should_rebalance(
        self,
        loads: Mapping[InstanceId, LoadSample],
        sessions: Mapping[SessionId, SessionState],
        now_ts: float,
    ) -> list[MigrationDecision]: ...

    def pick_target_instance(
        self,
        session: SessionState,
        candidates: Sequence[InstanceId],
        loads: Mapping[InstanceId, LoadSample],
    ) -> InstanceId: ...

    def pick_session_for_initial_home(
        self,
        session_id: SessionId,
        loads: Mapping[InstanceId, LoadSample],
    ) -> InstanceId: ...

    def should_consider_session_for_migration(
        self,
        session: SessionState,
        now_ts: float,
    ) -> bool: ...
```

The policy module must support three layers:

- `NeverRebalance`: baseline/stub policy. It assigns initial homes using
  the same power-of-two load-aware rule as `gw_load_aware`, never
  proposes migrations, and is the regression guard that tc_router can
  behave like a normal sticky load-aware router.
- `MigrateOnceAfterTurnPolicy`: E2E correctness policy. It is not meant
  to be a good load-balancer. It picks an initial home, waits until a
  session has completed `after_turn_count` successful turns, then
  migrates that session once to a different instance. Target selection is
  deterministic (`next_instance`) or least-loaded among non-source
  candidates. This policy is the validation gate for Tensorcast
  `publish` / `hydrate` and session-scoped prepared-bundle reuse.
- `ThresholdPolicy`: headline experiment policy. It is the first
  programmable load-aware policy and can be tuned after the migration
  primitive is correct.

`ThresholdPolicy` behavior:

- `should_rebalance`: if `max_load / min_load > rebalance_ratio` and
  `max_load - min_load > rebalance_abs`, select up to
  `max_migrations_per_tick` sessions from the hottest instance, in
  decreasing-context-length order, that are eligible per
  `should_consider_session_for_migration`. Target = least-loaded instance.
- `pick_target_instance`: least-loaded instance.
- `pick_session_for_initial_home`: least-loaded instance.
- `should_consider_session_for_migration`: true iff
  `(now_ts - session.last_active_ts) < inter_turn_delay_p90`
  AND `session.turn_count >= 2`
  AND `not session.pending_migration`.

The policy is one file. Replacing it with another file is the entire
mechanism by which user-space programmability is demonstrated.

### 6.5 Migration mechanic

#### 6.5.1 Required SGLang serving profile

`tc_router` request migration requires SGLang to be launched in
Tensorcast explicit request-transfer mode. Starting Tensorcast services
beside a plain SGLang fleet is insufficient: the source instance would
not record publishable request-bundle state, the target instance would
not run prepared-bundle admission, and the Tensorcast directory would
not know how to execute `plan.on_instance(...)`.

For a tc_router config, the driver must therefore use a dedicated
`tensorcast` serving profile with this startup order:

1. Plan SGLang placements exactly as the gateway baselines do.
2. Start the Tensorcast global store on
   `cluster.yaml.service_placement.global_store_worker_id`.
3. Start one Tensorcast daemon on every worker that hosts at least one
   SGLang instance. The daemon's advertised address must be reachable
   from the driver and from other daemons.
4. Launch every SGLang instance with Tensorcast HiCache enabled and an
   instance-agent execution endpoint assigned to that logical instance.
5. Wait for both OpenAI serving readiness and Tensorcast directory
   readiness before starting workload traffic.

The SGLang CLI flags are:

```bash
--enable-hierarchical-cache
--hicache-storage-backend tensorcast
--hicache-storage-backend-extra-config '<json>'
```

The extra config is a flat JSON object:

```jsonc
{
  "daemon_address": "127.0.0.1:61053",
  "namespace": "<run_id>",
  "engine": "sglang",
  "model_id": "Qwen3-32B",
  "model_version": "<stable-model-version>",
  "policy_profile": "durable",
  "instance_directory_address": "<global_store_advertise_host>:61050",
  "instance_agent_execution_endpoint": "<worker_address>:<agent_port>",
  "tensorcast_kv_mode": "explicit_request_transfer",
  "background_page_publish": false,
  "ordinary_storage_prefetch": false,
  "record_host_residency_for_publish": true,
  "logical_session_id_source": "routing_key"
}
```

`daemon_address` is from the SGLang process to its worker-local daemon.
For the current static/local deployments, that should be
`127.0.0.1:<daemon_port>` because the daemon runs on the same worker as
the SGLang process. The router's `tc.connect(...)` target is separate:
it connects to the first daemon's driver-reachable advertised address.
That daemon's directory view resolves all workers through the global
store.

`instance_directory_address` is the global store's advertised
`<host>:<port>`, not a local loopback address, because remote workers
must register their SGLang instances into the same global directory.

`instance_agent_execution_endpoint` must be a worker-reachable endpoint
dedicated to the SGLang instance-agent sidecar. The benchmark should
derive it deterministically from a base port plus placement index, for
example `<worker.address>:<instance_agent_base_port + i>`. The endpoint
must not collide with the serving HTTP port, Tensorcast daemon port,
daemon P2P port, Mooncake ports, or NCCL ports.

The logical Tensorcast `instance_id` for request-level transfer is the
whole SGLang serving instance / TP group:

```text
<server_args.host>:<server_args.port>
```

For tc_router runs, `server_args.host` must be the same worker-reachable
host used by the benchmark's `instance_endpoints` map. If the process is
launched with `--host 0.0.0.0`, SGLang would register
`0.0.0.0:<port>` while the router tries to resolve
`<worker.address>:<port>`, and `resolve_instance_execution()` would fail.
The tensorcast profile should therefore bind SGLang to `worker.address`
unless the SGLang integration grows a separate explicit
`instance_id` override.

After all SGLang instances report HTTP readiness, the driver performs a
Tensorcast directory readiness gate:

```python
runtime = tc.connect(daemon_address=primary_daemon_address)
for instance_id in instance_endpoints:
    runtime.directory().resolve_instance_execution(instance_id)
```

All instance routes must resolve before the cell sweep begins. This gate
catches missing `instance_directory_address`, bad instance-agent
endpoints, host/port identity mismatches, and stale directory state
before workload traffic starts.

#### 6.5.2 Publish and hydrate plan shape

For session `S` with `home_instance = A` and target instance `B`:

```python
# Inside the rebalancer task; never on the request hot path.
last_rid = session_state[S].last_engine_request_id

source_route = runtime.directory().resolve_instance_execution(A).value
target_route = runtime.directory().resolve_instance_execution(B).value
source_instance = tc.Instance(
    instance_id=source_route.instance_id,
    daemon_id=source_route.daemon_id,
    engine=source_route.engine or "sglang",
    execution_endpoint=source_route.execution_endpoint,
)
target_instance = tc.Instance(
    instance_id=target_route.instance_id,
    daemon_id=target_route.daemon_id,
    engine=target_route.engine or "sglang",
    execution_endpoint=target_route.execution_endpoint,
)

ctx_pub = tc.context(
    request_id=f"migrate-pub:{S}:{epoch}",
    idempotency_key=f"migrate-pub:{S}:{epoch}",
    deadline_ms=15_000,
)
plan_pub = runtime.plan(ctx_pub)
pub_ref = plan_pub.on_instance(source_instance).publish(
    engine_request_id=last_rid,
    ttl_ms=migration_ttl_ms,  # default 120 s
)
res_pub = plan_pub.run()
publish_manifest = res_pub.step(pub_ref).artifact_result.publish_manifest

ctx_hyd = tc.context(...)
plan_hyd = runtime.plan(ctx_hyd)
hyd_ref = plan_hyd.on_instance(target_instance).hydrate(
    publish_manifest=publish_manifest,
)
plan_hyd.run()

session_state[S].home_instance = B
session_state[S].last_published_manifest = publish_manifest
```

The implementation should not invent a second Tensorcast API wrapper.
It should mirror the known-good call shape from
`kv/request_transfer/caller_driver.py`: `CallContext`,
`runtime.plan(context)`, `plan.on_instance(...).publish(...)`, then
`plan.on_instance(...).hydrate(...)`, checking that the publish step
returns a `PublishResult` with a non-null `publish_manifest` and that
the hydrate step returns a `HydrateResult`.

The plan calls are synchronous in the Tensorcast SDK. tc_router runs them
in an executor so the asyncio workload loop is not blocked.

#### 6.5.3 Host-residency invariant for explicit publish

In `explicit_request_transfer` mode, `host_resident` in SGLang's
request-bundle page state is a concrete ownership invariant, not a loose
"stable page" hint. For a non-READY page it means the bundle manager has
a recorded retained host slot for `(rank, page_hash)` and can force-flush
that page during an explicit `publish(...)` call.

The live-request sync path must therefore derive `host_resident` from the
recorded host-resident page table. It must not mark every stable live page
as host-resident. Doing so lets the publish closure attempt a force-flush
for an ABSENT page with no host backing, which fails inside Tensorcast as
a `PlanFailedError` with a missing host-resident page-byte diagnostic.

Publish closure readiness follows these rules:

- READY pages can be published directly.
- ABSENT pages with a recorded host-resident slot can be force-flushed by
  the explicit publish path.
- ABSENT pages without recorded host backing are not ready. The migration
  should wait until the publish deadline and return a structured
  `publish_failed` row instead of issuing an invalid Tensorcast plan.

`batch_set_v1()` in explicit request-transfer mode is the path that commits
or retains host slots and records host residency for publishable pages.
After a successful force-flush, the retained host slot is released so the
allocator-backed host pool does not leak across requests or cells.

#### 6.5.4 Pending migration routing semantics

Once a migration starts for session `S`, the router sets
`session_state[S].pending_migration` before issuing Tensorcast plans.
This prevents duplicate migrations of the same source request and gives
the next request a deterministic choice:

- If the next turn for `S` arrives while the migration is still running,
  `generate()` waits up to `pending_migration_wait_timeout_s`.
- If the migration succeeds within the wait, `home_instance` has already
  been updated to the target and the request routes to the target.
- If the migration fails or times out, `home_instance` remains the
  source and the request routes to the source.
- If the wait itself times out but the migration is still running, the
  request routes to the current home. The migration may still complete
  later, but it is marked unconsumed unless a later turn uses it.

This wait is intentionally session-local. It does not block unrelated
sessions and does not serialize a worker's request stream. The wait is a
correctness guard for the E2E migration smoke test; later policies can
lower the timeout once bundle reuse has been validated.

#### 6.5.5 Minimal E2E rebalance policy

The first policy used to test session-scoped request migration is
`migrate_once_after_turn`, configured for example as:

```yaml
policy:
  kind: migrate_once_after_turn
  seed: 0
  after_turn_count: 1
  target_strategy: next_instance        # or least_loaded
  max_migrations_per_session: 1
  pending_migration_wait_timeout_s: 30
  plan_deadline_ms: 30000
  publish_ttl_ms: 600000
```

This policy does not attempt to optimize load. It is successful if:

1. the first successful source turn leaves a publishable
   `last_engine_request_id`,
2. `publish(last_engine_request_id)` succeeds on the source,
3. `hydrate(publish_manifest)` succeeds on a different target instance,
4. the next turn of the same `session_id` routes to the target,
5. SGLang target admission attaches the session-scoped prepared bundle,
6. the target turn reports non-zero `cached_tokens`.

Only after this path is stable should the benchmark tune
`ThresholdPolicy` or other load-aware policies.

#### 6.5.6 Bundle reuse verification signals

The router's own state is not enough to prove that the target actually
consumed the hydrated bundle. The E2E gate verifies multiple signals:

- `migrations.jsonl` contains a successful migration with source,
  target, publish latency, hydrate latency, and manifest digest.
- The next turn for that session has `served_instance == target` and
  `was_just_migrated == true`.
- The next turn has `cached_tokens > 0`; for stricter validation, compare
  against the published cutoff minus tail-valid tokens encoded in the
  SGLang publish manifest.
- The target SGLang log contains
  `Tensorcast prepared-bundle attached` for the migrated request and the
  publish manifest digest, and does not contain a matching fallback,
  fail-closed, or consume-failed line.

`used_hydrated_bundle` in `turns.jsonl` is a router-level best-effort
field. For the smoke test it may be set when a just-migrated turn lands
on the target and reports `cached_tokens > 0`. The authoritative
prepared-bundle proof is the SGLang target log signal.

#### 6.5.7 Subtleties locked in for v1

- **Partial snapshot is accepted.** Per `tensorcast_kv_protocol.md` § 5.3,
  `publish(engine_request_id=E)` covers only the prompt-prefix up to E's
  cutoff. For multi-turn this means the published bundle ends at the last
  *user-message boundary*, **not** at the end of the assistant response.
  When the next turn arrives at B, B will re-prefill the assistant
  response part of the new prompt. The README and per-turn metrics MUST
  surface this so the partial-coverage cost is visible. We do not use the
  "placeholder-request" trick in v1.
- **Publish-after-completion retention window.** SGLang adapters retain
  a publishable prompt snapshot for `last_engine_request_id` only for as
  long as their internal request-bundle metadata is retained. If a
  session has been idle long enough that A has discarded that snapshot,
  publish will fail. v1 treats publish failure as a no-op migration and
  leaves `home_instance = A` unchanged; the policy may attempt the
  migration again on a later turn.
- **No active evict.** After successful hydrate, the source instance A
  keeps its local KV. SGLang's HiRadixCache will reclaim it under
  pressure. This matches the user's preference and avoids the
  router-misprediction failure mode of premature eviction.
- **No fallback to source.** Once `home_instance = B`, subsequent turns
  of S route to B even if A would have been fine. If B happens to be
  overloaded by the time the next turn arrives, the standard cooldown
  governs whether yet another migration is issued. There is no
  per-request "if migration not yet useful, fall back to A" logic in v1.

### 6.6 Load signal

Each instance exposes pending-request count via the SGLang serving
HTTP endpoint (`/get_server_info` or the existing `get_load` we already
wrote in `tot_experiment/src/tot_experiment/sglang_client.py`). The
router polls these endpoints in a background task at
`load_poll_period_ms` (default 250 ms) and caches the result in
`instance_loads`.

This deliberately does not go through the Tensorcast directory. Per
`tensorcast_kv_protocol.md` § 3.1, the Tensorcast directory does not
expose serving-side queue metrics; load awareness is required to live
outside it.

### 6.7 Cache signal

The router treats its own `session_state[session_id].home_instance` map
as ground truth for "where does this session's KV currently live". The
Tensorcast directory is only consulted for execution route resolution.

### 6.8 TP > 1 future-compatibility

The router code treats `instance_id` opaquely:

- `runtime.directory().resolve_instance_execution(instance_id)` produces
  the execution route regardless of TP shape
- all `plan.on_instance(...)` calls take the resolved `Instance`
  unchanged
- the SGLang serving HTTP endpoint per instance is one URL regardless of
  TP

TP > 1 is therefore a launch-harness concern, not a router concern. v1
launch harness supports `tp_size` as a parameter (default 2 for the
Qwen3-32B reference, 1 for the Qwen3-14B alternative); mixed-TP
topologies in a single run are not supported (§ 2).

## 7. Physical Topology

The current checked-in implementation supports two provider shapes:

- `resource/local.py`: a single local worker. The driver host is also the
  worker host; the placement planner can pack multiple SGLang instances
  onto disjoint GPU windows on the 8xH800 machine.
- `resource/static.py`: a static local-plus-SSH worker set. Workers are
  already acquired outside the benchmark and are described in
  `cluster.yaml`. The benchmark starts services through local subprocesses
  or SSH commands and assumes the shared `/mnt/data` filesystem is visible
  on every worker. No `scp` or `rsync` is part of the provider contract.

The publication-grade experiment uses the static provider so Tensorcast
`publish` / `hydrate` traffic can measure real cross-host transport.
The benchmark still supports local smoke runs because they are faster and
useful for debugging launch and routing logic.

The benchmark consumes workers described in a cluster YAML (see § 9.1).
For `provider.kind: local`, those workers are a declarative view of the
current host. For `provider.kind: static`, they are a declarative view of
already-reachable local/remote hosts.

### 7.1 Worker layout

Local smoke layout:

- one worker, `local_h800`, with `gpu_indices: [0, 1, 2, 3, 4, 5, 6, 7]`
- `instances.count = 3`, `tp_size = 2` packs three SGLang instances on
  GPU windows `[0,1]`, `[2,3]`, and `[4,5]`
- the same local worker hosts the Tensorcast global store and one
  Tensorcast daemon

Static local+remote layout:

- one local worker and one or more SSH workers, all sharing `/mnt/data`
- `/home/yuhan` is a symlink to `/mnt/data` on every worker, so generated
  configs, service logs, PID files, and benchmark outputs are visible
  without file transfer
- one Tensorcast daemon runs on each worker that hosts SGLang instances
- the Tensorcast global store usually runs on the local worker so the
  in-process tc_router can connect to a local or low-latency daemon while
  the directory still sees remote worker registrations

The placement planner is greedy and provider-agnostic: it walks workers
in YAML order, allocating non-overlapping `tp_size` GPU windows until
`instances.count` instances have been placed. For future cross-host
runs, a cluster YAML can instead describe several workers and the same
planner will spread or pack instances according to available GPUs.

### 7.2 Driver host

For local runs, the router and traffic generator run on the same host as
the SGLang instances. For static multi-worker runs, the router and
traffic generator run on the local driver host and send HTTP requests to
both local and remote SGLang instances. The driver host:

- has network reachability to every worker's SGLang HTTP endpoint and
  Tensorcast daemon; for local runs these are `127.0.0.1:<port>`, while
  static runs use each worker's configured routable address
- runs the `tc_router` Python process (router + workload generator +
  Tensorcast runtime client) for tc_router configs, and the
  gateway-baseline wrapper for `gw_*` configs

### 7.3 Distinct-host requirement

If a cluster YAML lists multiple workers, their `id`, `address`, and
`node` values must be distinct. The local 8xH800 setup lists a single
worker and intentionally runs multiple SGLang instances on that worker
using disjoint GPU windows.

### 7.4 RDMA capability and transport mode

For local smoke runs, `transport.use_rdma` should be `false`. The local
provider does not require RDMA environment variables, and the checked-in
`cluster_local_h800.yaml` leaves `base_env: {}`. CUDA forward-compatibility
libraries are expressed separately through `env_path_prepend`, which lets the
local provider prepend CUDA compatibility libraries to `LD_LIBRARY_PATH`
without relying on shell expansion or overwriting an existing value.
The current static H800 deployment prepends `/usr/local/cuda-13/bin` to
`PATH` and `/usr/local/cuda-13/compat` to `LD_LIBRARY_PATH`.

For future cross-host providers, the actual transport used by
Tensorcast / Mooncake is selectable per-run via `transport.use_rdma` in
`benchmark.yaml` (see § 9.2), mapping to:

- Tensorcast: `communicator.enable_rdma = true | false`
- Mooncake: `protocol = rdma | tcp`

For Mooncake TCP runs, `device_name` may be left empty and the launcher
passes `protocol: tcp` in SGLang's Mooncake extra config. For Mooncake
RDMA runs, `transport.use_rdma: true` maps to `protocol: rdma`; the
cluster YAML or benchmark config must then supply the worker-appropriate
HCA/device selection expected by Mooncake. RDMA smoke is a pre-service
gate only for runs that explicitly enable RDMA.

The chosen mode is recorded as a column in `summary.csv`.

### 7.5 RDMA smoke

`services/rdma_smoke.py` contains the star-shaped RDMA reachability
check intended for future cross-host runs. Local runs set
`transport.use_rdma: false` and skip this concern.

### 7.6 RDMA env injection

Every worker's `base_env` is merged into commands run through
`Worker.run` / `Worker.start_background`. For local smoke runs this may
be empty. `env_path_prepend` is applied after `base_env` and before any
per-call `env`, and is intended for path-like variables such as
`LD_LIBRARY_PATH`. `env_unset` is applied last and removes inherited
environment variables such as `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY`
from worker commands; this keeps local gateway/SGLang HTTP calls from being
routed through an external proxy. Future RDMA-capable providers should put per-worker
variables such as `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`,
`NCCL_SOCKET_IFNAME`, `NCCL_SOCKET_FAMILY`, and `MASTER_ADDR` in
`base_env`.

## 8. Driver Structure

```
tc_router/
  arch.md                  # this file
  README.md                # how to run (per § 9.3 invocation form)
  run_benchmark.py         # entry: combines cluster_*.yaml + benchmark.yaml
  scripts/                 # service lifecycle wrappers (reused from request_transfer / share_remote)
  configs/
    cluster_local_h800.yaml      # current local 8xH800 worker description
    cluster_local_h800_no_compat.yaml
    benchmark_local_baseline_tp1_smoke.yaml
    benchmark_local_tc_router_smoke.yaml
    benchmark_<id>.yaml          # one per experiment definition
  resource/                # cluster-portable resource abstraction (§ 7)
    __init__.py
    base.py                # Worker, ResourceProvider, RemoteProcess Protocols
    factory.py             # dispatch on cluster_yaml.provider.kind
    local.py               # LocalProvider: Worker.run uses local subprocesses
    static.py              # StaticProvider: local + SSH workers on shared /mnt/data
  services/                # service launchers, all Provider-agnostic
    __init__.py
    base.py                # Service / ServiceLauncher abstract interfaces
    sglang.py              # one SGLang instance on a Worker
    tensorcast.py          # global store + daemon
    mooncake.py            # master + metadata service
    gateway.py             # sgl-model-gateway (gw_* configs)
    rdma_smoke.py          # star-shaped reachability check
  driver/
    __init__.py
    benchmark_loop.py      # (config, c_target, trial, preset) sweep
    health.py              # generic service-ready waits
    log_collect.py         # pull logs from each worker via Worker.get_file
  router/                  # unchanged from earlier sections
    interface.py           # Router protocol
    gateway_router.py
    tc_router.py
    policy.py
    state.py
    instance_loads.py
    rebalancer.py
  workload/                # unchanged
    trajectory_pool.py
    generator.py
    inter_turn_delay.py    # log-normal sampler with named-preset support
  metrics/
    per_turn.py
    migrations.py
    summary.py
  outputs/                 # populated at runtime
  tests/
```

### 8.1 `run_benchmark.py`

Invocation:

```bash
python -m tensorcast_benchmark.kv.tc_router.run_benchmark \
  --cluster configs/cluster_local_h800.yaml \
  --bench   configs/benchmark_local_tc_router_smoke.yaml
```

Responsibilities:

- load `cluster.yaml` via `resource.factory.from_cluster_config(...)` →
  obtain a `ResourceProvider` and its `list[Worker]`
- run `health_check()` on the workers
- if a future cross-host config enables RDMA, validate transport before
  launching services
- group configs by serving profile before launching SGLang:
  - `plain`: `gw_load_aware`, `gw_cache_aware`
  - `tensorcast`: `tc_router`
  - `mooncake`: `gw_load_aware_mooncake`
- for each serving profile that appears in the filtered config set:
  - launch the service prerequisites for that profile; the Mooncake profile
    starts the Mooncake master/metadata singleton before SGLang, the
    Tensorcast profile starts the global store plus one daemon per worker
    before SGLang, and the plain profile has no storage-service
    prerequisite
  - launch the SGLang instance fleet for that profile, using the placement
    plan derived from `instances.count`, `model.tp_size`, and worker GPU
    windows
  - for the Tensorcast profile, launch SGLang with explicit
    request-transfer HiCache extra config and then wait until every logical
    SGLang `instance_id` resolves in the Tensorcast directory
  - run all configs in that profile, then tear down that profile's SGLang
    fleet and service prerequisites
- for each `(config, c_target, trial, preset)` cell of the sweep:
  - ensure the previous cell's workload has drained and no in-flight request
    remains
  - POST `/flush_cache` directly to every SGLang instance and retry until all
    instances return HTTP 200, subject to a bounded timeout
  - for Tensorcast-backed tc_router cells, POST
    `/clear_hicache_storage_backend` directly to every SGLang instance if
    the cell needs strict prepared-bundle isolation; the default migration
    smoke can rely on fresh SGLang/Tensorcast profile startup plus per-cell
    `/flush_cache`, but publication experiments should clear storage
    backends between cells when comparing cache effects
  - for Mooncake-backed cells, POST `/clear_hicache_storage_backend`
    directly to every SGLang instance and retry until all instances return
    HTTP 200, subject to a bounded timeout
  - launch a fresh front router for this cell (`sgl-model-gateway` for
    `gw_*`, Python `tc_router` for Tensorcast runs) and wait for health
  - run the traffic generator on the driver host for `T_wall` seconds
  - mark turns emitted before `T_warmup` as warmup and exclude them from
    cell-level summary metrics
  - tear down the front router for this cell, but keep SGLang instances alive
  - logs are written under the configured scratch/output paths; with the
    local provider those paths are ordinary local filesystem paths
- write a top-level `summary.csv` indexed by
  `(config, c_target, trial, preset)` with `transport_mode` recorded
- tear down any live SGLang fleet and profile-specific service prerequisites
  after the run completes or fails

The driver does **not** acquire or release workers and does not assume
which cluster CLI was used — see § 14 for the portability story.

### 8.2 `resource/` — cluster-portable abstraction

```python
class RemoteProcess(Protocol):
    pid: int | None
    returncode: int | None
    stdout: str
    stderr: str
    async def wait(self) -> int: ...
    async def kill(self) -> None: ...

class Worker(Protocol):
    id: str                       # human-readable label
    address: str                  # routable IP / host used by service endpoints
    node: str
    gpu_indices: tuple[int, ...]  # GPUs this benchmark may use on this worker
    scratch_dir: str              # writable per-worker path
    base_env: dict[str, str]      # always merged into Worker.run env

    async def run(
        self,
        cmd: list[str] | str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_s: float | None = None,
        check: bool = True,
        as_user: bool = True,
    ) -> RemoteProcess: ...

    async def start_background(
        self,
        cmd: str,
        *,
        name: str,
        log_path: str,
        pid_path: str,
        env: dict[str, str] | None = None,
    ) -> int: ...

    async def stop_background(self, *, pid_path: str) -> None: ...
    async def put_file(self, local_path: Path, remote_path: str) -> None: ...
    async def get_file(self, remote_path: str, local_path: Path) -> None: ...
    async def read_file(self, remote_path: str, *, max_bytes: int | None = None) -> bytes: ...

class ResourceProvider(Protocol):
    @classmethod
    def from_cluster_config(cls, path: str | Path) -> "ResourceProvider": ...

    def workers(self) -> list[Worker]: ...
    async def health_check(self) -> None: ...
```

The Provider adapts a cluster YAML to a uniform `Worker` interface. For
`provider.kind: local`, no acquisition happens; the provider runs
commands directly on the current host with local subprocesses.
For `provider.kind: static`, no acquisition happens either; the provider
uses the local worker directly and reaches remote workers with SSH. The
static provider assumes every worker sees the same `/mnt/data` tree, so
`put_file` / `get_file` are only compatibility hooks and are not used for
benchmark source or output transfer.

### 8.3 `services/` — Provider-agnostic service launchers

Each launcher takes a `Worker` plus configuration and emits the right
command to start its service. None of them know whether the worker is
local or remote; they only call the `Worker` protocol.

For example, `services/sglang.py::launch_instance(worker, config)`
constructs the SGLang launch command (with `--tp`, `--port`,
`--host worker.address`, model flags, **without `--tool-call-parser`**
per § 5.2.3) and calls `Worker.start_background(...)` with a log path
and PID path under `worker.scratch_dir`. The same command shape works
on the local provider and on future providers.

`services/mooncake.py` is the Mooncake-profile singleton launcher. It
starts `.venv/bin/mooncake_master` on the worker selected by
`service_placement.mooncake_master_worker_id` with
`--enable_http_metadata_server=true`, the configured HTTP metadata port,
the configured master port, and the configured eviction high-watermark.
Readiness is the master's HTTP `/health` endpoint. The launcher exposes
the master server address and metadata URL to the driver, but does not
modify SGLang commands itself; `services/sglang.py` receives those values
as HiCache Mooncake extra config when launching the Mooncake serving
profile.

### 8.4 Traffic generator and routers

Stand-alone modules in `workload/generator.py` and `router/`. They
run on the driver host (not on workers). They take a `Router`
instance via constructor and a list of (instance_address, instance_id)
pairs derived from the launched SGLang services. They are unaware of
the resource layer.

## 9. Configuration Model

The benchmark consumes **two** YAML files: a cluster description and
an experiment description. They are intentionally separate so that the
same worker description can be reused across many experiments. In the
current local setup, the cluster YAML describes the current host rather
than a pre-acquired remote allocation.

### 9.1 `cluster.yaml` — describes workers

```yaml
provider:
  kind: local

driver_host:
  scratch_dir: /home/yuhan/tot/thirdparty/sglang/benchmark/tensorcast_benchmark/kv/tc_router/outputs/local_driver

mount:
  path: /home/yuhan/tot/thirdparty/sglang/benchmark/tensorcast_benchmark/kv/tc_router/outputs
  spec: local

workers:
  - id: local_h800
    address: 127.0.0.1
    node: local_h800
    process_handle: local
    gpu_indices: [0, 1, 2, 3, 4, 5, 6, 7]
    scratch_dir: /home/yuhan/tot/thirdparty/sglang/benchmark/tensorcast_benchmark/kv/tc_router/outputs/local_worker
    base_env: {}
    env_unset: [HTTPS_PROXY, HTTP_PROXY, https_proxy, http_proxy, ALL_PROXY, all_proxy]
    env_path_prepend:
      PATH:
        - /usr/local/cuda-13/bin
      LD_LIBRARY_PATH:
        - /usr/local/cuda-13/compat

service_placement:
  global_store_worker_id: local_h800
  mooncake_master_worker_id: local_h800
```

`service_placement.global_store_worker_id` is consumed by Tensorcast
runs. `service_placement.mooncake_master_worker_id` is consumed by
`gw_load_aware_mooncake` runs and selects the worker that hosts the
Mooncake master plus HTTP metadata service. Both fields are required in
the cluster schema so a single cluster YAML can support every benchmark
config kind.

For `provider.kind: local`, the cluster YAML is the machine-local
execution contract. For `provider.kind: static`, it describes already
reachable local/remote workers and the shared mount contract; the
benchmark does not acquire or release those workers.

At runtime the driver writes an effective `cluster.yaml` into
`outputs/<timestamp>_<run_id>/`. For the local provider, it rewrites
`driver_host.scratch_dir`, `mount.path`, and every worker `scratch_dir`
under that run directory. For the static provider, it preserves the
operator-provided worker addresses and shared mount semantics while
resolving per-run output paths under the shared filesystem. The input
YAML is preserved as `cluster_input.yaml`. This keeps SGLang logs, PID
files, resolved configs, turn JSONL files, and summaries together in a
single experiment folder.

### 9.2 `benchmark.yaml` — describes an experiment

```yaml
run_id: local-tc-router-smoke
model:
  path: hf/Qwen3-32B
  tp_size: 2
instances:
  count: 3
  base_port: 61101                    # avoids NodePort and ephemeral ranges
  kv_pool_size_gb: auto
  mem_fraction_static: 0.85
  page_size: 32
  sglang_log_level: debug             # optional; emits SGLang --log-level
transport:
  use_rdma: false                     # local smoke, not a cross-host RDMA run
workload:
  dataset_path: /data/datasets/OpenHands-Sampled-Trajectories
  pool_filter:
    min_turns: 8
    min_total_tokens: 8000
  inter_turn_delay:
    preset: agent_medium              # one of agent_fast | agent_medium | agent_slow | custom
    # custom_mu / custom_sigma only honored when preset == custom
  max_new_tokens_clip: 256
  start_jitter_s: 1.0
  wall_seconds: 60
  warmup_seconds: 0
  warmup_counts: 0
  trials: 1
  c_target_sweep: [3, 6]
configs:
  - kind: tc_router
    policy:
      kind: migrate_once_after_turn     # use never_rebalance for sticky stub runs
      seed: 0
      after_turn_count: 1
      target_strategy: next_instance
      max_migrations_per_session: 1
      pending_migration_wait_timeout_s: 30
      plan_deadline_ms: 30000
      publish_ttl_ms: 600000
  - kind: gw_cache_aware
    policy:                            # optional gateway CLI knobs
      cache_threshold: 0.0             # emits --cache-threshold
      balance_abs_threshold: 1000000   # emits --balance-abs-threshold
      balance_rel_threshold: 1.5       # emits --balance-rel-threshold
  - kind: gw_load_aware_mooncake
gateway:
  host: 127.0.0.1
  port: 61200
mooncake:
  http_metadata_server_port: 62300  # only used by gw_load_aware_mooncake
  master_port: 62301
  global_segment_size: 64gb
  eviction_high_watermark_ratio: 0.9
  device_name: ""                   # optional; empty is valid for TCP
  clear_storage_between_cells: true
tensorcast:
  global_store_port: 61050             # only used by tc_router
  daemon_port: 61053
  daemon_p2p_port: 61090
  instance_agent_base_port: 61400
  daemon_stable_bytes: 16GB
  clear_storage_between_cells: true
load_polling:
  period_ms: 250
```

Validation rules at load time:

- total available GPU windows across all workers must fit
  `instances.count × model.tp_size`; the local config packs several
  instances onto one worker
- `inter_turn_delay.preset == custom` requires `custom_mu` and
  `custom_sigma`; otherwise they MUST be absent.
- `provider.kind: local` and `provider.kind: static` both allow empty
  `base_env`; the cluster YAML should define only the environment needed
  by that worker's CUDA/NCCL/RDMA/runtime setup.
- `gw_cache_aware.policy`, when present, may contain only
  `cache_threshold`, `balance_abs_threshold`, and
  `balance_rel_threshold`; the driver forwards them to
  `sgl-model-gateway`. `gw_load_aware` does not accept gateway policy
  knobs.
- `mooncake` is only consumed when at least one config has
  `kind: gw_load_aware_mooncake`. It controls the Mooncake master
  service and SGLang Mooncake connection payload. It intentionally does
  not contain a `prefetch_threshold` field; the benchmark uses SGLang's
  upstream default instead of pinning a benchmark-specific value.
- `tensorcast` is only consumed when at least one config has
  `kind: tc_router`. It controls the global-store port, per-worker daemon
  ports, instance-agent base port, daemon memory budget, and whether
  Tensorcast HiCache storage backends are cleared between cells.
- `tc_router.policy.kind=never_rebalance` is valid for sticky-router
  regression runs. `tc_router.policy.kind=migrate_once_after_turn` is the
  required first E2E migration gate. `ThresholdPolicy` and later policies
  may add their own knobs, but they must reuse the same migration
  execution path.
- `tensorcast.instance_agent_base_port + instances.count - 1` must stay
  within the valid TCP port range and must not overlap serving HTTP,
  Tensorcast daemon, Tensorcast P2P, Mooncake, gateway, or NCCL ports.

### 9.3 Invocation

```bash
python -m tensorcast_benchmark.kv.tc_router.run_benchmark \
  --cluster configs/cluster_local_h800.yaml \
  --bench   configs/benchmark_local_tc_router_smoke.yaml
```

`kv_pool_size_gb: auto` lets each instance use its standard SGLang
auto-sizing; we lock the value across configs so KV capacity is
identical.

## 10. Output Model

### 10.1 Per-turn record

Written to `outputs/<run_id>/<config>/c<C_target>/trial<i>/turns.jsonl`.
Recorded for each LLM call (one per `assistant` message in the replayed
trajectory):

```jsonc
{
  "ts": 1717920000.123,
  "elapsed_s": 142.8,
  "is_warmup": false,
  "session_id": "gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t04-run_1::getmoto__moto-5321",
  "instance_id": "getmoto__moto-5321",
  "turn_index": 5,
  "prompt_messages_count": 11,
  "prompt_tokens": 9241,
  "max_new_tokens": 312,
  "served_instance": "inst-1",
  "ttft_ms": 184.2,
  "latency_ms": 4302.7,
  "cached_tokens": 8704,
  "used_hydrated_bundle": true,
  "was_just_migrated": true,
  "rid": "tcrouter:gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t04-run_1::getmoto__moto-5321:turn005"
}
```

`session_id` is a unique replay-session key built from the dataset
`run_id` and `instance_id` as `run_id::instance_id`. If an exact pair
appears more than once in the loaded shards, later occurrences append a
deterministic `::dupN` suffix so every pool entry is a distinct
simulated user/session.
`instance_id` (column inside the per-turn record, **not** the SGLang
serving-instance label) is the SWE-Gym task ID for traceability back
to the source dataset row.

`elapsed_s` is measured from the start of the cell's workload window.
`is_warmup` is true when `elapsed_s < warmup_seconds` or when the turn is
among the first `warmup_counts` completed records in that cell. Those rows
remain in `turns.jsonl` for debugging but are excluded from `summary.csv`.

### 10.2 Per-migration record

Written to `outputs/<run_id>/<config>/c<C_target>/trial<i>/migrations.jsonl`
(only emitted by `tc_router`):

```jsonc
{
  "ts": 1717919998.901,
  "migration_id": "migrate:session-a:000001",
  "is_warmup": false,
  "session_id": "gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t04-run_1",
  "source_instance": "inst-0",
  "target_instance": "inst-1",
  "source_engine_request_id": "tcrouter:run_1:turn04",
  "status": "consumed",
  "publish_latency_ms": 41.0,
  "hydrate_latency_ms": 88.3,
  "publish_manifest_digest": "b2c9...",
  "artifact_manifest_digest": "13e4...",
  "published_cutoff_token_count": 9216,
  "tail_valid_tokens": 512,
  "transferred_bytes_estimated": 134217728,
  "decided_by": "MigrateOnceAfterTurnPolicy",
  "consumed_by_turn_rid": "tcrouter:run_1:turn05",
  "consumed_within_s": 6.4,
  "target_turn_cached_tokens": 8704,
  "prepared_bundle_attached": true,
  "prepared_bundle_fallback": false,
  "prepared_bundle_fail_closed": false,
  "prepared_bundle_consume_failed": false,
  "wasted": false
}
```

`migrations.jsonl` contains one finalized row per migration. The router
may log intermediate publish/hydrate attempts to `router.log`, but the
summary reader expects each JSONL row to represent a single migration's
final state: `status=consumed`, `status=unconsumed`,
`status=publish_failed`, `status=hydrate_failed`, or `status=timeout`.
`consumed_by_turn_rid` is populated only after a later turn of the same
session lands on the target. If no such turn occurs before the bundle's
TTL expires or before the cell ends, `wasted=true`.

### 10.3 Run summary

`outputs/<run_id>/summary.csv` columns:

- `config`, `c_target`, `trial`
- `inter_turn_delay_preset` (e.g., `agent_medium`)
- `transport_mode` (`rdma` or `tcp`)
- `ttft_p50_ms`, `ttft_p95_ms`, `ttft_p99_ms`, `ttft_mean_ms`
- `cached_token_ratio_mean` (= `cached_tokens / prompt_tokens` per turn,
  averaged)
- `total_turns_completed`
- `total_requests_failed`
- `migration_count`
- `migration_utilization` (= migrations consumed / migrations issued)
- `mean_publish_latency_ms`, `mean_hydrate_latency_ms`

All summary latency, cache-ratio, completion, failure, and migration
utilization metrics are computed over non-warmup records only
(`is_warmup == false`). Warmup rows stay in JSONL for postmortem analysis.

## 11. Logging and Reproducibility

For every `(config, C_target, trial)` cell the run directory contains:

- the resolved `BenchmarkConfig`
- service launch commands
- per-instance SGLang logs
- gateway log (if applicable)
- Tensorcast daemon + global store logs (if applicable)
- Mooncake master + metadata logs (if applicable)
- `turns.jsonl`
- `migrations.jsonl`
- `router.log` (router-internal events: load samples, rebalance decisions)

The inter-turn delay RNG is seeded as
`seed = sha256(run_id || config || c_target || trial)` so reruns of the
same cell produce the same arrival pattern.

## 12. Validation Strategy

Bring-up order (each step must pass before moving on):

1. Single config (`gw_load_aware`), `N = 1`, `C_target = 1`. Sanity:
   end-to-end loop runs, jsonl written, no errors.
2. Single config (`gw_load_aware`), `N = 3`, `C_target = 3`. Sanity:
   each session lands on its own instance, no surprises.
3. `tc_router` with `should_rebalance` stubbed to always-False. Must
   launch the Tensorcast serving profile, resolve every SGLang instance
   in the Tensorcast directory, and behave like a sticky load-aware
   router with zero migrations.
4. `tc_router` with `policy.kind=migrate_once_after_turn`, `N = 2`,
   `C_target = 1` or `2`, and short wall time. Verify:
   `publish` succeeds, `hydrate` succeeds, the next turn for that
   `session_id` waits for the pending migration if necessary, lands on
   the target instance, and reports non-zero `cached_tokens`.
5. Reuse the prepared-bundle verification logic from
   `request_transfer`: target log must contain
   `Tensorcast prepared-bundle attached` for the expected manifest digest
   and must not contain matching fallback, fail-closed, or consume-failed
   lines.
6. Confirm `migrations.jsonl` is present, contains finalized migration
   rows, and `summary.csv` reports non-zero `migration_count`,
   non-null `migration_utilization`, and mean publish/hydrate latency.
   Completed smoke baseline:
   `outputs/20260713-135900_static-tc-router-migration-smoke-4inst-tp2`
   on the static 4-instance TP=2 local+remote cluster. It completed
   73 / 73 turns with zero request failures, emitted 12 finalized
   migration rows, consumed 6 of them (`migration_utilization = 0.5`),
   and reported 6 target turns with both `was_just_migrated=true` and
   `used_hydrated_bundle=true`. No publish/hydrate/PlanFailed errors were
   observed in the successful run. The unconsumed rows are migrations that
   published and hydrated successfully but did not receive another
   same-session turn before the 180 s smoke window ended.
7. Full `C_target` sweep on the smallest valid grid
   (`C_target in {3, 8}`) for all four configs, single trial.
8. Full sweep with `trials = 3`.

Only after step 8 is reproducibly green do we treat the headline plot as
a result.

## 13. Open Questions Tracked Outside This Doc

The following are deferred and will be specified when their cost / risk
becomes clear during implementation:

- exact knobs of `ThresholdPolicy` (`rebalance_ratio`, etc.) — tuned in
  step 7 of validation.
- whether to add a "placeholder-request" mechanism for whole-session
  snapshots (would upgrade partial → full coverage). v1 is partial; this
  is a possible v2.
- bandwidth budgeting for the rebalancer under high migration rates.
- mixed-TP launch-harness support. Uniform TP > 1 is already part of the
  v1 serving contract.
- richer policy variants (length-aware target picking, predictive
  pre-migration based on inter-turn delay distribution, etc.).

## 14. Cluster Portability

The current registered provider is `local`. The benchmark remains
structured so that a new cluster can be added by implementing another
`ResourceProvider`, but no remote provider is currently shipped.

### 14.1 What is cluster-specific

Exactly one component is allowed to know about a particular execution
environment: a `ResourceProvider` implementation in
`resource/<your_provider>.py`. Everything else
(`driver/`, `services/`, `router/`, `workload/`, `metrics/`,
`run_benchmark.py`) is cluster-agnostic and interacts with workers
only through the `Worker` and `RemoteProcess` Protocols defined in
`resource/base.py`.

The current implementation is `resource/local.py`, where `Worker.run`
and `Worker.start_background` are thin wrappers around local
subprocesses.

### 14.2 Onboarding a new cluster

1. **Acquire workers** out of band if the provider is remote, using
   whatever the cluster provides (its CLI, scheduler, or web UI). The
   benchmark should not perform acquisition during `run_benchmark.py`.
   Local runs do not need this step.
2. **Write `resource/<your_provider>.py`** implementing
   `ResourceProvider`. The bulk of the work is `Worker.run`, which
   needs to:
   - run a command on a specific worker
   - inject the worker's `base_env` plus the per-call `env`
   - support `start_background(...)` / `stop_background(...)` with
     log and PID paths
   - return a `RemoteProcess` whose `wait` / `kill` / `stdout` /
     `stderr` work
   - typically this wraps a cluster-specific exec command
     (`kubectl exec`, plain `ssh`, `docker exec`, a scheduler CLI,
     ...). `resource/local.py` is the reference for protocol shape.
3. **Register the provider**: add a `kind` → class mapping in
   `resource/factory.py`.
4. **Write `configs/cluster_<your_provider>_<id>.yaml`** describing
   the workers (per § 9.1 schema). `provider.kind` matches step 3.

That is the entire delta. `benchmark.yaml`, `run_benchmark.py`, the
service launchers in `services/`, the router, the workload, and the
metrics code do not change.

### 14.3 What lives outside the benchmark

- **acquisition / release scripts** for remote clusters: not part of the
  benchmark and not invoked by `run_benchmark.py`.
- **cluster-credential management**: SSH keys, k8s contexts,
  scheduler tokens, etc. Provided through whatever channel the target
  cluster uses; the benchmark inherits the operator's environment.
- **per-cluster RDMA discovery**: which HCAs to use is recorded as
  static `base_env` in the cluster YAML. Re-deriving HCA names per
  worker is a job for acquisition / provisioning code, not the
  benchmark.

### 14.4 Static provider

`resource/static.py::StaticProvider` is the supported SSH-based provider
for the current local+remote H800 setup. It assumes:

- the local driver can run commands on the local worker directly,
- remote workers are reachable through SSH without an interactive
  password prompt,
- every worker sees the same `/mnt/data` filesystem,
- `/home/yuhan` points at `/mnt/data` on every worker,
- and generated configs/logs/outputs are written under that shared tree.

Because the filesystem is shared, the static provider does not copy the
repository, model, dataset, configs, or outputs between workers.

## Appendix A. Dataset format reference

`.parquet` is Apache's columnar binary format. A file is roughly
`[header] + [row groups (per-column compressed pages)] + [footer (schema + offsets)]`.
It is not human-readable; use `pyarrow` / `pandas` / `duckdb` to inspect.
Strong typing (lists, structs are first-class) makes it the natural
container for the OpenAI-style nested message structure used here.

### A.1 One trajectory dump (real, content truncated)

Picked from `data/train.raw-00000-of-00003.parquet` row 0. Total 13
turns; we show the first 6 + the last 1, each `content` truncated to
120 chars.

```text
instance_id : getmoto__moto-5321
run_id      : gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t04-run_1
resolved    : False

messages:
  --- turn 0  role=system
      content: 'You are a helpful assistant that can interact with a
                computer to solve tasks.\n<IMPORTANT>\n* If user provides
                a path, y ... [+167 chars]'
  --- turn 1  role=user
      content: "<uploaded_files>\n/workspace/getmoto__moto__3.1\n</uploaded_files>\n
                I've uploaded a python code repository in the direct
                ... [+2079 chars]"
  --- turn 2  role=assistant
      tool_calls[]:
          id='call_yvYHZVbXjUPsvihtKQVVOtI6'
          function.name='str_replace_editor'
          function.arguments='{"command":"view","path":"/workspace/getmoto__moto__3.1"}'
  --- turn 3  role=tool  name=str_replace_editor
                          tool_call_id=call_yvYHZVbXjUPsvihtKQVVOtI6
      content: "OBSERVATION:\nHere's the files and directories up to 2
                levels deep in /workspace/getmoto__moto__3.1, excluding
                hidden it ... [+35 chars]"
  --- turn 4  role=assistant
      tool_calls[]:
          id='call_1pJhJa9YD2hFdbWDIUhuqEIV'
          function.name='str_replace_editor'
          function.arguments='{"command":"view","path":"/workspace/getmoto__moto__3.1"}'
  --- turn 5  role=tool  name=str_replace_editor
                          tool_call_id=call_1pJhJa9YD2hFdbWDIUhuqEIV
      content: "OBSERVATION:\n... [+35 chars]"
  ... (6 more turns omitted) ...
  --- turn 12 role=assistant
      tool_calls[]:
          id='call_ex9taEHTSftS9e36qCdpxe3m'
          function.name='str_replace_editor'
          function.arguments='{"command":"view","path":"/workspace/getmoto__moto__3.1"}'

tools (first entry only):
  type                : 'function'
  function.name       : 'execute_bash'
  function.description: 'Execute a bash command in the terminal.\n
                          * Long running commands: ... [+701 chars]'
  function.parameters : <struct with fields: command, file_text,
                          insert_line, new_str, old_str, path,
                          view_range — each is OpenAI JSON-schema-style
                          {description, type, [enum, items, ...]}>

test_result:
  apply_patch_output : ''
  git_patch          : ''
  test_output        : ''
  report             : {empty_generation: True, error_eval: False,
                        failed_apply_patch: False, resolved: False,
                        test_timeout: False}
```

### A.2 What this tells us about replay shape

- The first **two** turns are always `system` + `user`, where `user`
  carries the issue description + repo summary. This is the trajectory's
  "initial context".
- After that the loop is strictly `assistant(with tool_calls) → tool →
  assistant → tool → ...`. Every `assistant` boundary corresponds to
  one LLM call we will issue at replay time.
- `tool_calls[i].function.arguments` is a JSON **string**, not a parsed
  object — SGLang's chat-completions endpoint will deserialize and
  re-serialize it according to the target model's tool format.
- A `tool` message references its triggering call via `tool_call_id`
  and (often, but not always) carries a `name` matching the called
  function. Both fields must be preserved when constructing prompts;
  SGLang relies on them.
- `tools` is a list of OpenAI function specs available to the agent
  during the original run. We pass it verbatim with each request so
  the target model sees the same tool surface.
- `test_result` and `resolved` are not used by the replay. They exist
  for filtering / analysis (e.g. "did this trajectory's agent actually
  succeed?") but are orthogonal to the prompt-stream we measure.

### A.3 Quick inspection commands

```bash
# Schema only (no row data)
python3 -c "import pyarrow.parquet as pq; \
  print(pq.read_metadata('/data/datasets/OpenHands-Sampled-Trajectories/data/train.raw-00000-of-00003.parquet').schema)"

# Row count of one shard
python3 -c "import pyarrow.parquet as pq; \
  print(pq.read_metadata('/data/datasets/OpenHands-Sampled-Trajectories/data/train.raw-00000-of-00003.parquet').num_rows)"

# Read into pandas (only the columns we need)
python3 -c "import pyarrow.parquet as pq; \
  t = pq.read_table('/data/datasets/OpenHands-Sampled-Trajectories/data/train.raw-00000-of-00003.parquet', \
                    columns=['instance_id','run_id','resolved','messages']); \
  print(t.slice(0,1).to_pylist()[0]['messages'][:2])"
```
