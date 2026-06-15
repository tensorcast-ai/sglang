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
- Resource acquisition. The benchmark consumes pre-acquired workers
  described in a cluster YAML (see § 9.1); how those workers come into
  existence is out of scope.
- Comparing routing-policy algorithm quality in the abstract; we compare
  concrete gateway policies vs. concrete tc_router policy

## 3. What Is Being Compared

Four routing configurations are exercised against the **same workload, the
same N serving instances, and the same model**:

| Config | Component in front | Notes |
|---|---|---|
| `gw_load_aware` | sgl-model-gateway, `--policy power_of_two` | No cache awareness |
| `gw_cache_aware` | sgl-model-gateway, `--policy cache_aware` | Sticky session + imbalance fallback |
| `gw_load_aware_mooncake` | sgl-model-gateway, `--policy power_of_two`, SGLang HiCache backed by Mooncake | Isolates the value of a shared KV substrate from the value of programmability |
| `tc_router` | Our user-space Python router, Tensorcast-backed | Eager session-level KV migration via `publish` / `hydrate` |

The serving instances and SGLang configuration are identical across all
four configs. Only the routing component changes.

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
    result = await router.generate(
        session_id=traj.run_id,
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
       │  in-process call: router.generate(session_id, messages, tools, ...)
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
        session_id: str,
        messages: list[dict],          # OpenAI chat-format messages
        tools: list[dict] | None,      # OpenAI function specs, optional
        sampling_params: dict,
    ) -> GenerateResult: ...

    async def close(self) -> None: ...
```

Both the Tensorcast router and the gateway-baseline wrapper implement
this interface. Each wrapper internally posts to its serving endpoint
via `/v1/chat/completions` (OpenAI-compatible). The traffic generator
does not know which router it is talking to and never sees a
flattened prompt string.

`GenerateResult` carries the streaming response text (discarded in
faithful-replay mode), TTFT, total latency, served-instance label, and
the SGLang `meta_info` (specifically `prompt_tokens`, `cached_tokens`).

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

On each `generate(session_id, messages, tools, ...)` call:

1. Resolve target instance:
   - if `session_id` has a `home_instance`, route there
   - else assign `home_instance` (round-robin or load-aware among
     instances) and route there
2. Generate a unique `rid` for this turn, store it as
   `session_state[session_id].last_engine_request_id`, then post the
   chat-completions request to the chosen instance. Record per-turn
   metrics.
3. Update `last_active_ts`, `turn_count`, `last_prompt_tokens`.
4. **Asynchronously** invoke `Rebalancer.maybe_rebalance(...)`. This
   never blocks the response.

A background `Rebalancer` task ticks every `rebalance_period_ms` (default
500 ms):

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

v1 ships **one** concrete `ThresholdPolicy`:

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

For session `S` with `home_instance = A` and target instance `B`:

```python
# Inside the rebalancer task; never on the request hot path.
last_rid = session_state[S].last_engine_request_id

ctx_pub = tc.context(
    request_id=f"migrate-pub:{S}:{epoch}",
    idempotency_key=f"migrate-pub:{S}:{epoch}",
    deadline_ms=15_000,
)
plan_pub = runtime.plan(ctx_pub)
pub_ref = plan_pub.on_instance(A).publish(
    engine_request_id=last_rid,
    ttl_ms=migration_ttl_ms,  # default 120 s
)
res_pub = plan_pub.run()
publish_manifest = res_pub.step(pub_ref).artifact_result.publish_manifest

ctx_hyd = tc.context(...)
plan_hyd = runtime.plan(ctx_hyd)
hyd_ref = plan_hyd.on_instance(B).hydrate(publish_manifest=publish_manifest)
plan_hyd.run()

session_state[S].home_instance = B
session_state[S].last_published_manifest = publish_manifest
```

Subtleties locked in for v1:

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

v1 is a cross-host setup. Each SGLang serving instance runs on a
distinct worker so that Tensorcast `publish` / `hydrate` traffic
goes over the actual cluster network rather than local PCIe / NVLink.
Same-host migration would short-circuit the substrate's transport
cost and make the comparison unrepresentative of real deployments.

The benchmark **consumes pre-acquired workers** described in a cluster
YAML (see § 9.1). It does not acquire or release workers itself.

### 7.1 Worker layout

`N` worker hosts (default `N = 3`), one SGLang instance per worker:

- worker `i` hosts: 1 SGLang instance + 1 Tensorcast daemon (always)
- one chosen worker (default `worker_0`) additionally hosts the
  Tensorcast global store, and (when applicable) the Mooncake master
  + metadata service

Each worker contributes `tp_size` GPUs to its SGLang instance.
Default model is `Qwen/Qwen3-32B` with `tp_size = 2` →
`N × tp_size = 6` GPUs across the cluster. `Qwen/Qwen3-14B` with
`tp_size = 1` is supported as the lightweight alternative.

### 7.2 Driver host

The router and traffic generator run on a **separate driver host**
(typically the local machine running `run_benchmark.py`, or a cluster
login node). The driver host:

- has network reachability to every worker's SGLang HTTP endpoint and
  Tensorcast daemon
- does not need GPUs
- runs the `tc_router` Python process (router + workload generator +
  Tensorcast runtime client) for tc_router configs, and the
  gateway-baseline wrapper for `gw_*` configs

This keeps load on the `N` workers symmetric. Putting router code on
one of the workers would unfairly burden that worker's serving
instance with control-plane overhead.

### 7.3 Distinct-host requirement

Workers MUST be on distinct physical hosts. The cluster YAML loader
asserts this; the benchmark refuses to start otherwise. Running
multiple instances on one worker is explicitly out of scope (§ 2).

### 7.4 RDMA capability and transport mode

Workers MUST be RDMA-capable (have an HCA exposed and configured).
The actual transport used by Tensorcast / Mooncake is selectable
per-run via `transport.use_rdma` in `benchmark.yaml` (see § 9.2),
mapping to:

- Tensorcast: `communicator.enable_rdma = true | false`
- Mooncake: `protocol = rdma | tcp`

This lets one cluster of workers serve both RDMA and TCP runs for
apples-to-apples backend comparison (same hardware, same workload,
only transport changes). The chosen mode is recorded as a column in
`summary.csv`.

### 7.5 RDMA smoke

For runs that select RDMA, the driver issues a star-shaped smoke
test (`worker_0 → worker_i` for `i ≥ 1`) before launching real
services. This reuses the contract from
[`share_remote/arch.md` § 6](../share_remote/arch.md). Smoke is
skipped for TCP runs and may be skipped for re-runs against an
already-validated cluster YAML via a `--skip-rdma-smoke` flag.

### 7.6 RDMA env injection

Per-worker RDMA environment variables (`NCCL_IB_HCA`,
`NCCL_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME`, `NCCL_SOCKET_FAMILY`,
`MASTER_ADDR`) are stored in each worker's entry in the cluster YAML
under `base_env` and **automatically injected** into every command
the benchmark runs on that worker via `Worker.run`. The driver
itself does not assume worker-global shell state is sufficient. This
mirrors `share_remote/arch.md` § 6.3.

## 8. Driver Structure

```
tc_router/
  arch.md                  # this file
  README.md                # how to run (per § 9.3 invocation form)
  run_benchmark.py         # entry: combines cluster_*.yaml + benchmark.yaml
  scripts/                 # service lifecycle wrappers (reused from request_transfer / share_remote)
  configs/
    cluster_brainctl_<id>.yaml   # one per acquired set of workers
    cluster_static_<id>.yaml     # one per pre-existing worker set
    benchmark_<id>.yaml          # one per experiment definition
  resource/                # cluster-portable resource abstraction (§ 7)
    __init__.py
    base.py                # Worker, ResourceProvider, RemoteProcess Protocols
    factory.py             # dispatch on cluster_yaml.provider.kind
    brainctl.py            # BrainctlProvider: Worker.run wraps `brainctl exec`
    static.py              # StaticProvider: Worker.run wraps generic shell exec (e.g., direct ssh)
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
python run_benchmark.py \
  --cluster configs/cluster_brainctl_001.yaml \
  --bench   configs/benchmark_qwen3_32b_main.yaml
```

Responsibilities:

- load `cluster.yaml` via `resource.factory.from_cluster_config(...)` →
  obtain a `ResourceProvider` and its `list[Worker]`
- run `health_check()` on the workers
- if `transport.use_rdma` is true, run the RDMA smoke test
- for each `(config, c_target, trial, preset)` cell of the sweep:
  - launch services on the appropriate workers (via `services/`,
    backed by `Worker.run`)
  - wait for service health
  - run the traffic generator on the driver host for `T_wall` seconds
  - tear down services launched for this cell
  - via `Worker.get_file`, pull per-instance logs into the run output
    directory
- write a top-level `summary.csv` indexed by
  `(config, c_target, trial, preset)` with `transport_mode` recorded

The driver does **not** acquire or release workers and does not assume
which cluster CLI was used — see § 14 for the portability story.

### 8.2 `resource/` — cluster-portable abstraction

```python
class RemoteProcess(Protocol):
    pid: int | None       # may be None if Provider doesn't expose PIDs
    async def wait(self) -> int: ...                  # exit code
    async def kill(self) -> None: ...
    async def stdout(self) -> bytes: ...
    async def stderr(self) -> bytes: ...

class Worker(Protocol):
    id: str                       # human-readable label
    address: str                  # routable IP for inter-worker traffic
    gpu_indices: list[int]        # GPUs this benchmark may use on this worker
    scratch_dir: str              # writable per-worker path
    base_env: dict[str, str]      # always merged into Worker.run env (RDMA vars, MASTER_ADDR, ...)

    async def run(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,    # merged on top of base_env
        cwd: str | None = None,
        background: bool = False,
        log_path: str | None = None,           # redirect stdout/stderr on the worker
    ) -> RemoteProcess: ...

    async def put_file(self, local_path: Path, remote_path: str) -> None: ...
    async def get_file(self, remote_path: str, local_path: Path) -> None: ...
    async def read_file(self, remote_path: str, *, max_bytes: int | None = None) -> bytes: ...

class ResourceProvider(Protocol):
    @classmethod
    def from_cluster_config(cls, path: str | Path) -> "ResourceProvider": ...

    def workers(self) -> list[Worker]: ...
    async def health_check(self) -> None: ...
```

The Provider is **not** responsible for acquiring workers. It only
adapts a YAML description of already-acquired workers to a uniform
`Worker` interface. Acquisition lives in out-of-band scripts that the
operator runs once per cluster lease.

### 8.3 `services/` — Provider-agnostic service launchers

Each launcher takes a `Worker` plus configuration and emits the
right command to start its service. None of them know about
`brainctl` or any specific cluster CLI; they only call `Worker.run`.

For example, `services/sglang.py::launch_instance(worker, config)`
constructs the SGLang launch command (with `--tp-size`, `--port`,
`--host worker.address`, model flags, **without `--tool-call-parser`**
per § 5.2.3) and calls `Worker.run(..., background=True,
log_path=f"{worker.scratch_dir}/sglang_{port}.log")`. The same
command shape works on every provider.

### 8.4 Traffic generator and routers

Stand-alone modules in `workload/generator.py` and `router/`. They
run on the driver host (not on workers). They take a `Router`
instance via constructor and a list of (instance_address, instance_id)
pairs derived from the launched SGLang services. They are unaware of
the resource layer.

## 9. Configuration Model

The benchmark consumes **two** YAML files: a cluster description and
an experiment description. They are intentionally separate so that
acquired workers can be reused across many experiments without
re-running any cluster CLI, and so that a new cluster's onboarding
touches exactly one file (the cluster YAML's provider section).

### 9.1 `cluster.yaml` — describes pre-acquired workers

```yaml
provider:
  kind: brainctl                      # dispatch key for resource/factory.py
  # Provider-specific settings, used when Worker.run executes commands.
  # Example for brainctl:
  exec:
    cli: brainctl
    subcmd: exec
driver_host:
  scratch_dir: /home/i-zhouyuhan/tot/thirdparty/sglang/benchmark/tensorcast_benchmark/kv/tc_router/outputs
workers:
  - id: worker_a
    address: 10.42.1.5                # routable IP for inter-worker traffic
    process_handle: rjob-XXX-001      # provider-specific identifier (used by brainctl exec)
    gpu_indices: [0, 1]
    scratch_dir: /workspace/scratch
    base_env:                         # injected into every Worker.run on this worker
      NCCL_IB_HCA: "mlx5_2,mlx5_3"
      NCCL_IB_GID_INDEX: "3"
      NCCL_SOCKET_FAMILY: AF_INET
      NCCL_SOCKET_IFNAME: bond0
      MASTER_ADDR: 10.42.1.5
  - id: worker_b
    address: 10.42.1.7
    process_handle: rjob-XXX-002
    gpu_indices: [0, 1]
    scratch_dir: /workspace/scratch
    base_env: { ... }
  - id: worker_c
    address: 10.42.1.9
    process_handle: rjob-XXX-003
    gpu_indices: [0, 1]
    scratch_dir: /workspace/scratch
    base_env: { ... }
service_placement:
  global_store_worker_id: worker_a    # which worker hosts the Tensorcast global store
  mooncake_master_worker_id: worker_a # which worker hosts Mooncake master (when applicable)
```

The cluster YAML is **acquired-state**, not a request. The operator
populates it once per cluster lease (via whatever cluster CLI is
appropriate), then runs many experiments against it.

### 9.2 `benchmark.yaml` — describes an experiment

```yaml
run_id: tcrouter-2026-06-15-001
model:
  path: Qwen/Qwen3-32B
  tp_size: 2
instances:
  count: 3                            # MUST equal len(cluster.workers)
  kv_pool_size_gb: auto
transport:
  use_rdma: true                      # mapped to Tensorcast / Mooncake transport flags
workload:
  dataset_path: /data/datasets/OpenHands-Sampled-Trajectories
  pool_filter:
    min_turns: 8
    min_total_tokens: 8000
  inter_turn_delay:
    preset: agent_medium              # one of agent_fast | agent_medium | agent_slow | custom
    # custom_mu / custom_sigma only honored when preset == custom
  max_new_tokens_clip: 512
  start_jitter_s: 2.0
  wall_seconds: 600
  warmup_seconds: 120
  trials: 3
  c_target_sweep: [3, 5, 8, 12, 18, 24]
configs:
  - kind: gw_load_aware
  - kind: gw_cache_aware
  - kind: gw_load_aware_mooncake
  - kind: tc_router
    policy:
      kind: threshold
      rebalance_ratio: 2.0
      rebalance_abs: 4
      max_migrations_per_tick: 1
      migration_cooldown_s: 30.0
      migration_ttl_ms: 120000
      inter_turn_delay_p90_s: 56.0    # P90 of the active preset; loader fills this
load_polling:
  period_ms: 250
rebalancer:
  period_ms: 500
```

Validation rules at load time:

- `instances.count == len(cluster.workers)` (one instance per worker).
- For every worker, `len(gpu_indices) >= model.tp_size`.
- `inter_turn_delay.preset == custom` requires `custom_mu` and
  `custom_sigma`; otherwise they MUST be absent.
- `inter_turn_delay_p90_s` is **derived** from the active preset by
  the loader (operator does not have to re-compute when changing
  preset).

### 9.3 Invocation

```bash
python run_benchmark.py \
  --cluster configs/cluster_brainctl_001.yaml \
  --bench   configs/benchmark_qwen3_32b_main.yaml
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
  "session_id": "gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t04-run_1",
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
  "rid": "tcrouter:run_1:turn05"
}
```

`session_id` is the trajectory's `run_id` from the dataset.
`instance_id` (column inside the per-turn record, **not** the SGLang
serving-instance label) is the SWE-Gym task ID for traceability back
to the source dataset row.

### 10.2 Per-migration record

Written to `outputs/<run_id>/<config>/c<C_target>/trial<i>/migrations.jsonl`
(only emitted by `tc_router`):

```jsonc
{
  "ts": 1717919998.901,
  "session_id": "gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t04-run_1",
  "source_instance": "inst-0",
  "target_instance": "inst-1",
  "publish_latency_ms": 41.0,
  "hydrate_latency_ms": 88.3,
  "transferred_bytes_estimated": 134217728,
  "decided_by": "ThresholdPolicy",
  "consumed_by_turn_rid": "tcrouter:run_1:turn05",
  "consumed_within_s": 6.4,
  "wasted": false
}
```

`consumed_by_turn_rid` is populated lazily after the next turn of the
session lands on the target. If no such turn occurs before the bundle's
TTL expires, the row is rewritten with `wasted: true`.

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
   behave identically to `gw_load_aware` (no migrations).
4. `tc_router` with `should_rebalance` stubbed to always-True for one
   target session. Verify the migration flow: `publish` succeeds,
   `hydrate` succeeds, `prepared-bundle attached` appears in the target
   log, the next turn lands on the new instance with high
   `cached_tokens`. This reuses the verification logic from
   `request_transfer`.
5. Full `C_target` sweep on the smallest valid grid
   (`C_target in {3, 8}`) for all four configs, single trial.
6. Full sweep with `trials = 3`.

Only after step 6 is reproducibly green do we treat the headline plot as
a result.

## 13. Open Questions Tracked Outside This Doc

The following are deferred and will be specified when their cost / risk
becomes clear during implementation:

- exact knobs of `ThresholdPolicy` (`rebalance_ratio`, etc.) — tuned in
  step 5 of validation.
- whether to add a "placeholder-request" mechanism for whole-session
  snapshots (would upgrade partial → full coverage). v1 is partial; this
  is a possible v2.
- bandwidth budgeting for the rebalancer under high migration rates.
- TP > 1 launch-harness support.
- richer policy variants (length-aware target picking, predictive
  pre-migration based on inter-turn delay distribution, etc.).

## 14. Cluster Portability

The benchmark is designed so that running it on a new cluster is a
**one-file delta**. This section spells out the contract.

### 14.1 What is cluster-specific

Exactly one component is allowed to know about a particular cluster
CLI: a `ResourceProvider` implementation in
`resource/<your_cluster>.py`. Everything else
(`driver/`, `services/`, `router/`, `workload/`, `metrics/`,
`run_benchmark.py`) is cluster-agnostic and interacts with workers
only through the `Worker` and `RemoteProcess` Protocols defined in
`resource/base.py`.

### 14.2 Onboarding a new cluster

1. **Acquire workers** out of band, using whatever the cluster
   provides (its CLI, its scheduler, its web UI). The benchmark does
   not call this. Acquisition results in a set of long-lived worker
   processes / pods / VMs.
2. **Write `resource/<your_cluster>.py`** implementing
   `ResourceProvider`. The bulk of the work is `Worker.run`, which
   needs to:
   - run a command on a specific worker
   - inject the worker's `base_env` plus the per-call `env`
   - support `background=True` with `log_path` redirection
   - return a `RemoteProcess` whose `wait` / `kill` / `stdout` /
     `stderr` work
   - typically this wraps a cluster-specific exec command
     (`brainctl exec`, `kubectl exec`, plain `ssh`, `docker exec`,
     ...). For brainctl the implementation in
     `resource/brainctl.py` is the reference.
3. **Register the provider**: add a `kind` → class mapping in
   `resource/factory.py`.
4. **Write `configs/cluster_<your_cluster>_<id>.yaml`** describing
   the acquired workers (per § 9.1 schema). `provider.kind` matches
   step 3.

That is the entire delta. `benchmark.yaml`, `run_benchmark.py`, the
service launchers in `services/`, the router, the workload, and the
metrics code do not change.

### 14.3 What lives outside the benchmark

- **acquisition / release scripts**: not part of the benchmark.
  Operators may keep convenience helpers like
  `scripts/acquire_brainctl.py` and `scripts/release_brainctl.py`
  next to the benchmark, but they are **not** invoked by
  `run_benchmark.py`.
- **cluster-credential management**: SSH keys, k8s contexts,
  brainctl tokens, etc. Provided through whatever channel the
  cluster CLI uses; the benchmark inherits the operator's
  environment.
- **per-cluster RDMA discovery**: which HCAs to use is recorded as
  static `base_env` in the cluster YAML. Re-deriving HCA names per
  worker is a job for the acquisition script, not the benchmark.

### 14.4 Static fallback

`resource/static.py::StaticProvider` is a generic provider that
takes the cluster YAML at face value and runs commands via plain
shell exec (e.g. `ssh user@address`). Any cluster that exposes SSH
to its workers can use it without writing a custom provider. Custom
providers are only needed when the cluster doesn't expose a
generic shell channel and instead requires its own exec CLI.

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
