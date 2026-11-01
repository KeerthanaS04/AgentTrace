# AgentTrace: Findings Log

> Record anything learned while building: surprises, failures, decisions, and measured results.
> Add new entries at the top of each section. Every entry should name the task (`P{phase}-T{n}`) and the date.
> These entries feed `docs/limitations.md` (P9-T8) and the submission write-up.

**Entry format:**

```
### YYYY-MM-DD · P?-T? · <short title>
- **What happened:** observation, with run/span/scenario IDs where relevant
- **Why it matters:** impact on correctness, cost, schedule, or claims
- **Action:** what we changed, or "open"
```

---

## 1. Decisions and Resolved Open Items
<!-- Hackathon rules, deadline, budget, chosen models/hosting, schema changes approved by the user, scope cuts -->

| Date | Item | Decision | Decided by |
|---|---|---|---|
| 2026-10-28 | Hackathon deadline | Nov 4, 2026; build the full plan, cut line only as a safety net | User |
| 2026-10-28 | `make` not installed on Windows dev machine | `tasks.py` is the task runner; `Makefile` is a thin wrapper over it | User |
| 2026-10-28 | Git workflow | One branch per task (`p{phase}-t{n}-<slug>`); Claude creates the branch, user commits and merges. Base branch is `main`; remote `origin` = github.com/KeerthanaCL/AgentTrace | User |
| 2026-10-28 | `phases.md` vs `PROGRESS.md` | `phases.md` serves as `PROGRESS.md`; `docs/lessons.md` points to `findings.md` | Claude |
| 2026-10-29 | P0-T2 schema additions beyond §4.2 | Approved: `parent_span_id`, `kind="agent"`, `prompt_version` (Span/AnomalyScore/Diagnosis), `Run.replay_of/seed/created_at`, `Diagnosis.diagnosis_id/escalation_reason/narrative`, `ReplayResult.diagnosis_id/patch`; new `RootCause`, `PropagatedError`, `Patch` (3 kinds), `Fix`, `ExplainabilityScore` | User |
| 2026-10-29 | Where injected faults live | `faults` sit with `ground_truth` in `schemas/ground_truth.py` (not in `ScenarioInput`): the fault list names the root node, so it is ground truth | Claude |
| 2026-10-29 | `Diagnosis.judge_agreement` range | [-1, 1], since Cohen's/Fleiss' κ can be negative (spec implied 0–1) | Claude |
| 2026-10-29 | Gateway HTTP client | `httpx` directly (one OpenAI-compatible adapter for Meta / NIM / OpenRouter / vLLM), not the `openai` SDK, so caching, retries, and cost stay in our code | Claude (user didn't object) |
| 2026-10-29 | Gateway cache key | Keyed by provider + model + messages + params + json_mode, **not** role, so roles that share a model share cached results | Claude |
| 2026-10-31 | KB format and domain | Fictional retailer "Brightcart". 47 docs in `data/kb/<topic>.yaml` (one YAML per topic, not one file per doc), 108 checkable facts. 4 superseded docs + confusable windows (30 / 15 / 60 / 14 days) give F1 faults a detectable effect | Claude |
| 2026-10-31 | Fact matching | Fact text is matched whitespace-insensitively (`normalize_ws`), since bodies wrap across lines; the P1-T7 checker should use the same normalization | Claude |
| 2026-11-01 | Reproducible numbers | Every number reported must come from a repo command, not an ad-hoc script. Added `python tasks.py retrieval`; README lists the measurement commands | User request |
| 2026-11-01 | Tool design | Tools compute day counts from the fixture DB's `as_of` date (never the real clock, never the LLM). `refund_api` has `quote` (check only) and `issue` modes; the Tool Agent uses `quote`, since it runs before the Policy Agent decides | Claude |
| | Team size / submission format | | |
| 2026-10-28 | Non-Meta models | Include NVIDIA + OSS as judges and agents under test, so results hold up beyond Meta's own models (still check the rules) | User |
| 2026-10-31 | Muse Spark model ID | `muse-spark-1.3` ($1.25 / $4.25 per M), confirmed via `GET /v1/models` + dashboard. Wonder confirmed at $4 / $20. Rate limits not yet known | User + Claude |
| | Spark Contributor tier? | **Open, user's call.** `muse-spark-1.3-contributor` is the same model at $0.10 / $0.20 (~12–20× cheaper) but data is used to train Meta models, it is rate-limited, and only in select countries. Our data is synthetic, so the privacy cost is low. Priced in `models.yaml`; switching is a one-line edit | User |
| | NVIDIA / OSS models + hosting | | |
| 2026-10-28 | Meta budget | $150 Model API credits; hard cap $140, warn at $100 | User |
| | NVIDIA / OSS budget | | |

---

## 1b. Tooling Findings

### 2026-10-28 · P0-T1 · ruff 0.17 formats Markdown code blocks
- **What happened:** `ruff format --check` flagged the Python schema snippet inside `CLAUDE_meta.md`.
- **Why it matters:** lint would fail on docs, and ruff would rewrite the spec.
- **Action:** `extend-exclude = ["*.md"]` in `[tool.ruff]`. Pre-commit hooks call `uv run ruff` so the hook and lint use the same ruff version.

### 2026-10-28 · P0-T1 · Anaconda `VIRTUAL_ENV` leaks into uv
- **What happened:** the shell has `VIRTUAL_ENV=...\anaconda3`, and uv prints a mismatch warning on every command.
- **Why it matters:** harmless (uv ignores it and uses `.venv`), but noisy.
- **Action:** open. Clear `VIRTUAL_ENV` in the shell, or don't auto-activate conda base in this terminal.

---

### 2026-10-30 · P0-T5 · Budget guard limits to remember
- **What happened:** the guard checks a worst-case estimate before each call (prompt chars / 3 + full `max_tokens` output) and records actual cost after.
- **Why it matters:**
  1. The per-run scope is a `ContextVar`, which does **not** carry into `ThreadPoolExecutor` worker threads automatically. When judging is parallelized (P6-T4), submit work with `contextvars.copy_context().run`, or spend won't count toward the run.
  2. Check and record are separate steps, so concurrent calls can overshoot a cap by up to (workers × one call's worst case).
  3. Unpriced models estimate $0, so the guard can't stop them. Price every model that costs money.
- **Action:** revisit 1–2 in P6-T4. 3 is handled by `test_verified_meta_models_are_priced` for Meta.

---

### 2026-10-31 · P1-T2 · Retrieval: vocabulary gap, and a held-out leak I caught
- **What happened:**
  - Plain BM25 got recall@3 = 0.875 on the 24-query dev set. All 3 misses were wording gaps: "cracked" vs "damaged/defective", "laptop" vs "electronics", "approve" vs "approval".
  - I added a small query-expansion table: damage synonyms, product → "electronics", approval wording.
  - I wrote 10 held-out queries to check generalization, but finished the table *after* writing them, and 3 entries (`authorize`, `coupon`, `earbuds`) came from held-out wording. That is leakage, and it inflated held-out recall to 0.900. I removed those entries.
- **Final numbers:**

  | Set | n | recall@1 | recall@3 | recall@5 |
  |---|---|---|---|---|
  | dev | 24 | 0.688 | **0.979** | 0.979 |
  | held-out | 10 | 0.800 | **0.800** | 0.900 |

- **Why it matters:**
  - The dev number meets Done-when (≥ 0.9), but held-out 0.80 (n=10, wide CI) is the more honest estimate for raw ticket text.
  - The two held-out misses are exactly the removed words, which confirms the leak.
  - Keyword retrieval stays brittle to synonyms. In the real workflow the Retriever agent's LLM writes the query, which should use policy wording and narrow the gap. Measure that in P1-T5.
- **Action:**
  - The held-out set is now frozen (a test checks that dev and held-out don't overlap).
  - Future expansion entries may only come from dev-set misses.
  - Revisit if the P4-T4 scenario checks show clean-run retrieval failures.

### 2026-10-31 · P1-T2 · BM25 implemented in-house
- **What happened:** I wrote BM25 directly (Lucene-style IDF, k1 = 1.5, b = 0.75, titles weighted ×2) instead of using `rank-bm25`.
- **Why it matters:** no numpy dependency, no negative-IDF quirk on a 47-doc corpus, and deterministic tie-breaking (score rounded to 1e-9, then doc_id), so rankings are reproducible across machines.
- **Action:** none. Superseded docs are left out of the index by default; F1 faults inject them.

---

## 2. Model and Provider Findings
<!-- Verified model IDs, API quirks, rate limits, pricing, non-determinism at temp 0, structured-output reliability -->

### 2026-10-31 · P0-T5 · Muse models spend hidden reasoning tokens inside `max_tokens`
- **What happened:** with `max_tokens=8` (and again with 300), Spark returned `content: null`, `finish_reason: length`. 298 of 300 tokens were `reasoning_tokens`. A one-word reply cost ~480 reasoning tokens at default effort.
- **Why it matters:**
  - A judge with too small a `max_tokens` returns nothing but is still billed.
  - Default reasoning would make Spark judging about 5× more expensive than needed.
  - Reasoning tokens are billed as output and included in `completion_tokens`, so our cost math is already correct.
- **Measured (one-word reply):**

  | Model | Default | `low` | `minimal` |
  |---|---|---|---|
  | Spark 1.3 | ~480 | ~90 | ~10 |
  | Wonder 1.0 | ~60 | ~50 | ~14 |

  Accepted values are `minimal, low, medium, high, xhigh, max` (`none` is rejected).
- **Action:**
  - `judge_fast` set to `reasoning_effort: low`, `max_tokens: 4096`; `judge_deep` set to `max_tokens: 8192`.
  - Gateway now records `reasoning_tokens` and `finish_reason`, and warns on truncated empty replies.
  - Smoke fails on empty replies instead of reporting them as `ok`.
  - Revisit the effort level against judge accuracy in P6-T1.

### 2026-10-31 · P0-T5 · Meta API supports JSON mode
- **What happened:** `response_format: {"type": "json_object"}` was accepted and returned valid JSON.
- **Action:** `supports_json_mode: true` for the meta provider.

### 2026-10-31 · P0-T5 · Probe calls outside the ledger
- **What happened:** the raw-API probes behind the two findings above were sent directly, not through the gateway, so they're missing from `.cache/calls.jsonl`.
- **Why it matters:** actual Meta spend is about $0.02 higher than the ledger shows. Negligible, but the ledger is a slight undercount.
- **Action:** future probes go through the gateway.

### 2026-10-29 · P0-T4 · Still unverified in `config/models.yaml`
- **What happened:** the first config was written without dashboard access.
- **Why it matters:** the gateway refuses `TODO_VERIFY` roles, so only `judge_deep` (Muse Wonder) is callable until these are filled in.
- **Action:** open. Confirm each of these:
  - Model IDs for `judge_fast` (Muse Spark), `judge_nvidia`, `judge_oss`, `agent_default`, `agent_alt_1`, `agent_alt_2`
  - Exact Muse Wonder pricing (spec says "roughly $4/$20")
  - Whether the Meta API accepts `response_format` JSON mode (`supports_json_mode` is false until then)
  - NVIDIA and OpenRouter base URLs, taken from public docs
  - Pricing for any NVIDIA / OpenRouter models chosen

---

## 3. Parse Failures and Structured-Output Issues
<!-- parse_failure counts by model/prompt version, common failure shapes, prompt fixes -->

---

## 4. Replay and Determinism Issues
<!-- Flaky replays, cache misses, live-vs-cached drift, stability tolerances (P5-T6) -->

---

## 5. Judge Behavior and Disagreements
<!-- Where judges disagree, systematic biases, κ values, cases that needed Wonder escalation -->

---

## 6. Scenario and Fault-Injection Issues
<!-- Ineffective faults, mislabeled scenarios, dropped scenarios and why (P4-T4) -->

---

## 7. Diagnosis Errors
<!-- Wrong root cause, missed propagation, false positives on F0, non-recovering patches -->

---

## 8. Explainability Metric Findings
<!-- Agreement with human labels, metrics that didn't separate faulty/clean, drift detection lag -->

---

## 9. Cost and Performance
<!-- Cost per run/diagnosis, Wonder spend, latency, budget-guard trips -->

| Date | What was measured | Value | Notes |
|---|---|---|---|

---

## 10. Measured Results (headline numbers)
<!-- Only numbers produced by the evaluation harness. Always include n and CI. Keep oracle and model/rule results separate. -->

| Date | Metric | Value | 95% CI | n | Config / commit |
|---|---|---|---|---|---|

---

## 11. Scope Cuts
<!-- Anything cut per the cut line in CLAUDE_meta.md §8, with reason. Mirror into docs/limitations.md. -->
