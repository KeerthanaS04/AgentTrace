# AgentTrace: Phases and Tasks

> Task checklist derived from `CLAUDE_meta.md`. Tick each box when its **Done when** criterion is met and tests pass.
> Commit per task as `P{phase}-T{n}: <summary>`. Log anything surprising in `findings.md`.

**Total:** 13 phases, 83 tasks.

---

## Before Starting: Open Items

- [x] Deadline: **Nov 4, 2026**. Scope: **the full plan** (cut line kept only as a safety net)
- [ ] Team size and submission format (check hackathon checklist)
- [x] Non-Meta models: **include them**. NVIDIA and open-source models act as independent judges and agents under test, so results hold up outside Meta's own models
- [ ] Double-check the hackathon rules don't penalize non-Meta models
- [ ] Exact Muse Spark model ID and rate limits (Model API dashboard)
- [ ] Pick NVIDIA + open-source models and hosting (NIM / OpenRouter / local vLLM), and their budget
- [x] Meta budget: **$150 Model API credits** (see Budget Plan)

---

## Schedule (Oct 28 → Nov 4)

Rule: build a **thin** version of each stage first, then deepen. Feature freeze **Nov 3, 18:00**.

| Day | Date | Tasks | Exit check |
|---|---|---|---|
| 1 | Tue Oct 28 | P0 (all), P1-T1–T4 | Gateway calls every provider; KB + tools + runtime work |
| 2 | Wed Oct 29 | P1-T5–T7, P2 (all), P3 (all), P4-T1 (F1 + F3 only) | Traced runs with correct `input_refs`; graph builds |
| 3 | Thu Oct 30 | P5 (all), P6-T1–T3, T5, T7 (thin), P7 (all) | **M1:** one scenario → verified root cause + recovered replay |
| 4 | Fri Oct 31 | P4 (rest), P6-T4, T6, T8, P11-T1 (scaffold in parallel) | 80–120 verified scenarios; ensemble + escalation working |
| 5 | Sat Nov 1 | P9-T1–T3, T5, T7, P8-T1–T3, T5–T6 | **M2:** headline numbers + baselines + ablations exist |
| 6 | Sun Nov 2 | P8-T4, T7–T9, P10 (all), P9-T4, T6, T8 | **M3:** scoring + drift alerts; API serves everything |
| 7 | Mon Nov 3 | P11-T2–T9 | **M4:** full demo can be driven from the UI, offline mode works |
| 8 | Tue Nov 4 | P12 (all) | **M5:** submitted |

If a day's exit check fails, finish it before starting the next day's tasks. If you're still behind by the end of Day 5, apply the cut line.

---

## Budget Plan ($150 Meta credits)

Muse Wonder is about $4/M input and $20/M output tokens. Muse Spark pricing is still to be confirmed.

| Use | Meta budget | Notes |
|---|---|---|
| Dev + smoke + prompt iteration | $15 | Iterate prompts on cheap/OSS models first, then confirm on Spark |
| Spark: Stage A judging + continuous scoring | $40 | Every call cached; repeat runs cost ~$0 |
| Wonder: escalations in Full pipeline | $25 | Only on escalation triggers |
| Wonder: B3 baseline (Wonder-only) | $35 | Run on a fixed subset (~40 scenarios, K=1) if the full set would exceed this |
| Demo pre-recording + reruns | $10 | |
| Reserve | $25 | Don't touch before Nov 3 |

- Set `MAX_USD_TOTAL` for the Meta provider to **$140** (hard stop) and warn at $100.
- **Agents under test run on non-Meta models** by default (`agent_default` = an OSS model). Muse models stay the diagnosis engine. This saves Meta credits and backs the model-agnostic claim.
- Track Meta spend separately from NVIDIA/OSS spend in the gateway cost log.

---

## Build Order (Milestones)

Phases are numbered by component, **not** by build order. Build in this order:

| Milestone | Scope | Outcome |
|---|---|---|
| **M1: Vertical slice** | P0, P1, P2, P3, P5, P6 (minimal), P7, plus a minimal P4 (one injector + one scenario) | One scenario goes trace → graph → diagnosis → verified replay |
| **M2: Evidence at scale** | P4 (full dataset), P9 (metrics + baselines) | Real numbers exist |
| **M3: Continuous explainability** | P8 | Metrics, scorer, drift alerts |
| **M4: Platform + UI** | P10, P11 | API and dashboard |
| **M5: Polish + submission** | P12 | Demo, docs, submission |

**Can run in parallel:** P1 ∥ P4 (fault catalog) ∥ P5 (replay cache); P8 ∥ P11 (UI scaffold).

**Never cut:** schemas, gateway, provenance edges, fault dataset with ground truth, replay engine, verification loop, headline metrics, 3-minute demo.

**Cut first if late (in this order):** P11-T9 + P10-T7 → P9-T4 + P9-T6 → P10-T6 → P8-T4 → third ensemble judge.

---

## Phase 0: Foundations
**Goal:** clean repo, shared schemas, working model gateway. · **Milestone:** M1

- [x] **P0-T1: Repo scaffold.** Layout per §4.1, `pyproject.toml`, `Makefile` (`setup test lint serve eval`), `.env.example`, `pre-commit`, `PROGRESS.md`.
  - *Done when:* `make setup && make test` is green on an empty suite.
- [x] **P0-T2: Schemas.** `Ref`, `Span`, `Explanation`, `NodeContract`, `Run`, `AnomalyScore`, `Diagnosis` (+ `RootCause`, `PropagatedError`, `Fix`), `ReplayResult`, `Scenario` in `agenttrace/schemas/`.
  - *Done when:* every schema round-trips through JSON; invalid data is rejected.
- [x] **P0-T3: Model gateway.** Provider adapters (Meta, NVIDIA NIM, generic OpenAI-compatible), role lookup, content-hash cache, retry/backoff, token + cost accounting, Pydantic structured-output with retry.
  - *Done when:* `gateway.complete(role="judge_fast", ...)` works per provider; repeat call is a logged `cache_hit`; cost recorded.
  - *Status:* verified live 2026-10-31 on Meta (Spark + Wonder): call, then `cache_hit` at $0, cost recorded. NVIDIA / OpenRouter covered by mocked tests; re-run `smoke` once their models are chosen.
- [x] **P0-T4: Model config.** `config/models.yaml` with roles `judge_fast`, `judge_deep`, `judge_nvidia`, `judge_oss`, `agent_default`, `agent_alt_1`, `agent_alt_2`, plus pricing. Unknown IDs = `TODO_VERIFY`.
  - *Done when:* swapping a role's model needs a config edit only.
- [x] **P0-T5: Smoke tests + budget guard.** `make smoke` pings each model; `MAX_USD_PER_RUN` / `MAX_USD_TOTAL` abort over-budget runs.
  - *Done when:* smoke passes for available providers and skips unavailable ones gracefully.
  - *Status:* verified 2026-10-31: `judge_fast` + `judge_deep` ok, 5 TODO_VERIFY roles skipped, exit 0. Budget guard: total / per-provider / per-run caps, lifetime spend from ledger, checked before sending.

---

## Phase 1: Demo Workflow (Customer Support)
**Goal:** a realistic workflow to instrument and break. · **Milestone:** M1

- [ ] **P1-T1: Knowledge base.** 30–50 policy/FAQ docs in `data/kb/` with IDs and checkable facts (dates, amounts, windows).
  - *Done when:* each doc has ID, title, body, and ≥1 checkable fact.
- [ ] **P1-T2: Retrieval index.** Vector or BM25 index; `search(query, k)` → doc IDs + scores.
  - *Done when:* top-3 recall ≥ 0.9 on 20 hand-written queries.
- [ ] **P1-T3: Mock tools.** `order_lookup`, `refund_api`, `account_status` on a JSON fixture DB; strict response schemas.
  - *Done when:* each tool has typed I/O and unit tests.
- [ ] **P1-T4: Agent runtime.** `workflow/runtime.py`; agents are `run(inputs, ctx) -> AgentOutput` (output + `Explanation`); workflow is a declared step graph.
  - *Done when:* a toy 2-agent workflow runs end to end.
- [ ] **P1-T5: The five agents.** Router → Retriever → Tool Agent → Policy Agent → Response Writer, each with a `NodeContract` in `workflow/contracts.py`.
  - *Done when:* ≥90% of 20 clean tickets reach the correct final decision.
- [ ] **P1-T6: Model-swappable agents.** Per-agent model from config.
  - *Done when:* workflow runs on ≥3 different models via config only.
- [ ] **P1-T7: Final-outcome checker.** Deterministic check (expected decision + required facts in reply).
  - *Done when:* unit tested on correct and incorrect outputs.

---

## Phase 2: Tracing and Instrumentation
**Goal:** capture every step with true data-flow provenance. · **Milestone:** M1

- [ ] **P2-T1: `@traced` + `TraceContext`.** Record input, output, timings, tokens, cost, model, params, explanation; stable `node_id`.
  - *Done when:* a run yields a full `Run` with spans for all 5 agents + nested tool/retrieval/LLM spans.
- [ ] **P2-T2: Provenance via `Ref`.** Inputs carry their producer span; no inferring edges from execution order.
  - *Done when:* clean-run `input_refs` exactly match the declared data flow (tested).
- [ ] **P2-T3: Trace store.** SQLAlchemy models/repos for runs, spans, scores, diagnoses, replays; CLI `agenttrace run --scenario X`.
  - *Done when:* runs persist and reload byte-identically; queries by run/node/scenario work.
- [ ] **P2-T4: Export/import.** JSON (optional OTel-style).
  - *Done when:* export → import yields an identical `Run`.
- [ ] **P2-T5: Integrity checks.** No orphan refs, no cycles, consistent timestamps.
  - *Done when:* corrupted-trace fixtures are rejected.

---

## Phase 3: Graph Reconstruction
**Goal:** traces → queryable causal DAG. · **Milestone:** M1

- [ ] **P3-T1: Graph builder.** `networkx.DiGraph`: nodes = spans, edges = `input_refs`, with agent/kind/contract attributes.
  - *Done when:* graph matches declared topology on clean runs.
- [ ] **P3-T2: Graph queries.** `ancestors`, `descendants`, `topological_order`, `paths_to_final`, `critical_path`, `first_divergence`.
  - *Done when:* tested on hand-built graphs.
- [ ] **P3-T3: Logical-node alignment.** Map spans across runs by `node_id`.
  - *Done when:* original vs. replay yields the correct changed/unchanged set.
- [ ] **P3-T4: UI serialization.** Graph JSON (nodes, edges, status colors, layout hints) for React Flow.
  - *Done when:* JSON validates against the UI schema.

---

## Phase 4: Fault Injection and Scenario Dataset
**Goal:** labeled dataset with known root causes. · **Milestone:** M2 (minimal slice needed for M1)

- [ ] **P4-T1: Fault catalog.** Pure-function injectors:
  - `F0` control (no fault) · `F1` wrong/stale retrieval · `F2` empty/irrelevant retrieval · `F3` wrong tool value · `F4` malformed/error tool response · `F5` invalid decision · `F6` hallucinated claim · `F7` multi-fault · `F8` silent fault
  - *Done when:* each injector is tested to mutate only its target node.
- [ ] **P4-T2: Scenario schema + generator.** Ticket + fixtures + fault spec + ground truth (`root_cause_node_ids`, `affected_node_ids`, `expected_final`, `fault_type`). 80–120 scenarios, ≥10 per fault type, ≥15 controls.
  - *Done when:* dataset validates, is balanced, and is committed to `data/scenarios/`.
- [ ] **P4-T3: Ground-truth isolation.** Only `scenarios/ground_truth.py` exposes truth; `diagnosis/` and `explainability/` can't import it.
  - *Done when:* a test fails on any forbidden import path.
- [ ] **P4-T4: Scenario verification.** Confirm each fault produces its labeled effect (F8: intentionally none).
  - *Done when:* ≥95% behave as labeled; the rest are fixed, documented, or removed.
- [ ] **P4-T5: Multi-run protocol.** K runs per scenario with different seeds (default K=3).
  - *Done when:* K runs stored per scenario; pass-rate variance reported.

---

## Phase 5: Record/Replay Engine (hardest — start early)
**Goal:** reproducible re-execution with surgical interventions. · **Milestone:** M1

- [ ] **P5-T1: Record/replay cache.** Keyed by `(node_id, model, params, hash(inputs))`.
  - *Done when:* replaying an unmodified run gives zero diffs.
- [ ] **P5-T2: Patcher.** Target span + replacement → plan: freeze non-descendants, mark target `patched=True`, re-run descendants.
  - *Done when:* frozen/re-run sets are correct on hand-built graphs.
- [ ] **P5-T3: Re-executor.** Topological execution; frozen nodes reuse outputs; replay stored linked to original.
  - *Done when:* patching a known-bad node changes only its descendants.
- [ ] **P5-T4: Patch sources.** `oracle` (ground truth, upper bound), `model` (model-proposed), `rule` (deterministic fix). Tag `patch_source`.
  - *Done when:* each source works end to end on ≥1 fault type.
- [ ] **P5-T5: Differ + recovery scoring.** Changed spans, final diff, `recovered`, `recovery_score`.
  - *Done when:* output feeds both UI and evaluation harness.
- [ ] **P5-T6: Replay stability.** Same replay ×3: identical (cached) or within documented tolerance (live).
  - *Done when:* stability report generated; tolerance documented.

---

## Phase 6: Diagnosis Engine
**Goal:** failed run → ranked, evidence-backed root cause. · **Milestone:** M1 (minimal), deepened later

- [ ] **P6-T1: Judge prompts + schemas.** Versioned prompts in `judges/prompts/` → `AnomalyScore`. Judge answers separately: (a) inputs trustworthy? (b) output correct given inputs? Must cite span/doc IDs.
  - *Done when:* schema-valid output on 20 sample spans per judge model.
- [ ] **P6-T2: Deterministic validators.** Schema, ranges, tool-error flags, retrieval relevance threshold, numeric claim-vs-source match.
  - *Done when:* catch all `F4` and most `F2` with no LLM call.
- [ ] **P6-T3: Stage A — per-node anomaly scoring.** Fast judges over every node in topological order, combined with validators.
  - *Done when:* every span gets an `AnomalyScore`; cost per run logged.
- [ ] **P6-T4: Judge ensemble.** `judge_fast` + `judge_nvidia` + `judge_oss`; weighted vote; Cohen's κ; per-node disagreement.
  - *Done when:* output has per-judge verdicts, aggregate verdict, agreement score.
- [ ] **P6-T5: Stage B — causal attribution.** Root cause = bad output + trustworthy inputs (or unexplained anomalous parents). Propagated = bad output + anomalous parent that explains it (with path). Rank by topo order, confidence, descendant count.
  - *Done when:* correct on 10 hand-checked scenarios; every propagated node has a valid path.
- [ ] **P6-T6: Stage C — escalation to Muse Wonder.** Rules in `config/escalation.yaml`: low agreement, close top-2, multiple roots, long chain, validator/judge conflict.
  - *Done when:* triggers correctly in tests; cost tracked separately.
- [ ] **P6-T7: Stage D — fix proposal.** Structured `Fix` with a machine-applicable patch per fault type.
  - *Done when:* every root cause has an applicable `Fix`.
- [ ] **P6-T8: Diagnosis report.** `Diagnosis` + narrative citing only recorded spans.
  - *Done when:* a lint test confirms every narrative claim references an existing span ID.

---

## Phase 7: Counterfactual Verification Loop
**Goal:** prove or reject diagnoses by experiment. · **Milestone:** **M1 complete here**

- [ ] **P7-T1: Verification loop.** Per candidate: patch → replay → check recovery; verified, or demote and try next (max 3).
  - *Done when:* loop terminates; `ReplayResult` rows linked to the diagnosis.
- [ ] **P7-T2: Interventional ranking.** Rank by measured recovery; label "necessary" vs. "contributing".
  - *Done when:* `F7` scenarios get correct combined attribution when both roots are patched.
- [ ] **P7-T3: Propagation verification.** After patching root, propagated nodes should become correct; still-wrong nodes are re-labeled as roots.
  - *Done when:* labels update automatically from replay evidence.
- [ ] **P7-T4: End-to-end command.** `agenttrace diagnose --run <id>` and `--scenario <id>`.
  - *Done when:* **M1** — one command takes a failing scenario to a verified root cause with recovered replay.

---

## Phase 8: Continuous Explainability
**Goal:** measure grounding and explainability per step and over time. · **Milestone:** M3

- [ ] **P8-T1: Metric definitions doc.** `docs/explainability_metrics.md` + weights in `config/explainability.yaml`.
  - *Done when:* each metric has a precise, testable definition.
- [ ] **P8-T2: Grounding (G).** Atomic claims checked against available evidence; `G = supported / total`.
  - *Done when:* agreement with 30 hand-labeled spans is reported.
- [ ] **P8-T3: Rationale-Action Consistency (RAC).** Does rationale entail the action? 0–1 via judge + deterministic checks.
  - *Done when:* flags all injected `F5`/`F6` inconsistencies.
- [ ] **P8-T4: Faithfulness via ablation (Fa).** `Fa = P(change | cited removed) − P(change | uncited removed)`, clipped [0,1], sampled.
  - *Done when:* separates faithful from unfaithful explanations on a constructed set.
- [ ] **P8-T5: Calibration (C).** ECE per agent over a window; `C = 1 − ECE`.
  - *Done when:* per-agent ECE + reliability diagram export.
- [ ] **P8-T6: Composite XI.** `XI = Σ wᵢ · metricᵢ`; always report components.
  - *Done when:* stored per span, agent, and run.
- [ ] **P8-T7: Continuous scoring pipeline.** Background job + `agenttrace score --run <id>`; cheap judge tier; scheduled ablations.
  - *Done when:* 20 scenarios auto-populate scores for every span.
- [ ] **P8-T8: Drift detection + alerts.** Rolling windows, z-score/CUSUM, `Alert` records; auto-diagnose low-XI runs even if final answer looks right.
  - *Done when:* an injected degradation raises an alert within a defined number of runs.
- [ ] **P8-T9: Link XI to diagnosis.** XI as Stage A/B features; diagnosis labels low-XI nodes. Flag-switchable.
  - *Done when:* toggleable by flag for Phase 9 ablation.

---

## Phase 9: Evaluation Harness
**Goal:** turn everything into numbers, with baselines and ablations. · **Milestone:** M2

- [ ] **P9-T1: Metrics.** Root-cause top-1/top-3; propagation P/R/F1; recovery rate (oracle vs. model/rule, separate); false-positive rate on F0; early detection on F8; judge κ (Cohen + Fleiss) and ensemble gain; XI AUROC; cost + latency, tiering savings.
  - *Done when:* each metric has a unit-tested function.
- [ ] **P9-T2: Baselines.** B0 whole-trace single LLM · B1 Spark-only graph · B2 ensemble, no replay · B3 Wonder-only · Full system.
  - *Done when:* all run through the same harness on the same scenarios.
- [ ] **P9-T3: Ablations.** No validators / ensemble / escalation / replay verification / XI features / provenance edges.
  - *Done when:* ablation table shows each component's contribution.
- [ ] **P9-T4: Model-agnostic evaluation.** ≥3 agent models and multiple judge sets.
  - *Done when:* results reported per agent-model and per judge-set.
- [ ] **P9-T5: Statistical reporting.** K runs, bootstrap 95% CIs, per-fault-type breakdown.
  - *Done when:* every headline number has a CI and `n`.
- [ ] **P9-T6: Before/after experiment.** Apply recommended fixes, re-run suite, report change in success and XI.
  - *Done when:* measured improvement (or honest null) documented.
- [ ] **P9-T7: Report generator.** `agenttrace eval --report` → `reports/results.md`, CSVs, plot data.
  - *Done when:* one command regenerates all results.
- [ ] **P9-T8: Failure analysis.** Confusions by fault type, judge errors, non-recovery.
  - *Done when:* `docs/limitations.md` written from real error cases.

---

## Phase 10: Platform Backend
**Goal:** an API over everything. · **Milestone:** M4

- [ ] **P10-T1: FastAPI skeleton.** App factory, config, CORS, errors, OpenAPI.
  - *Done when:* `make serve` serves `/health`.
- [ ] **P10-T2: Run/trace endpoints.** `POST /runs`, `GET /runs`, `GET /runs/{id}`, `/graph`, `/spans/{span_id}`.
  - *Done when:* tested with fixture runs.
- [ ] **P10-T3: Diagnosis/replay endpoints.** `POST/GET /runs/{id}/diagnose|diagnosis`, `POST /runs/{id}/replay`, `GET /replays/{id}`, `GET /runs/{id}/compare/{other_id}`; background jobs + polling.
  - *Done when:* diagnosis can be started and polled to completion.
- [ ] **P10-T4: Explainability endpoints.** `GET /explainability/agents`, `/explainability/runs/{id}`, `GET /alerts`, `POST /alerts/{id}/ack`.
  - *Done when:* time series match stored scores.
- [ ] **P10-T5: Evaluation/regression endpoints.** `POST /suites/{name}/run`, `GET .../results`, `GET .../compare?a=&b=`.
  - *Done when:* two suite runs give a comparable delta report.
- [ ] **P10-T6: SDK / ingestion adapter.** `agenttrace-sdk` with `@traced` + HTTP exporter; second toy workflow example.
  - *Done when:* second workflow's traces are ingested and diagnosed.
- [ ] **P10-T7: Settings endpoints.** Model roles, judge sets, thresholds, sampling (non-secret only).
  - *Done when:* changes apply without restart and are logged.

---

## Phase 11: UI
**Goal:** failure chain and explainability obvious in seconds. · **Milestone:** M4

- [ ] **P11-T1: Scaffold.** Vite + React + TS + Tailwind, routing, OpenAPI-generated client, layout.
  - *Done when:* app lists runs from the API.
- [ ] **P11-T2: Causal graph (React Flow).** Status colors (ok / suspicious / propagated / root cause); click node → span panel.
  - *Done when:* a diagnosed scenario shows root cause and propagation path.
- [ ] **P11-T3: Evidence panel.** Narrative with clickable span refs; judge agreement; Wonder escalation flag.
  - *Done when:* every cited span ID highlights its node.
- [ ] **P11-T4: Replay comparison.** Side-by-side graph diff, final output diff, recovery verdict, patch source label.
  - *Done when:* shows exactly which nodes changed and whether the outcome recovered.
- [ ] **P11-T5: Explainability dashboard.** Per-agent G/RAC/Fa/C/XI time series, alerts, reliability diagram, cross-model comparison.
  - *Done when:* injected degradation shows a visible dip and alert.
- [ ] **P11-T6: Regression suite page.** Version A vs. B deltas; newly failing/fixed scenarios.
  - *Done when:* shows a meaningful diff.
- [ ] **P11-T7: Evaluation results page.** Metric cards with CIs, baseline chart, ablation table, per-fault breakdown, cost tiering.
  - *Done when:* renders from `reports/` with no manual steps.
- [ ] **P11-T8: Scenario launcher.** Pick scenario/fault, run, watch pipeline progress.
  - *Done when:* full live demo driven from the UI.
- [ ] **P11-T9: Polish + resilience.** Loading/error/empty states, light/dark, keyboard nav, offline demo mode.
  - *Done when:* demo works with network off.

---

## Phase 12: Demo, Documentation, Submission
**Goal:** a reliable 3-minute demo and a reproducible submission. · **Milestone:** M5

- [ ] **P12-T1: Demo script.** `docs/demo_script.md`: failing ticket → causal graph → Spark evidence + Wonder escalation → replay recovery → silent-drift alert → headline metrics.
  - *Done when:* dry run fits the time limit with no manual fixes.
- [ ] **P12-T2: Curated demo scenarios.** Easy (bad retrieval), subtle (tool value propagating through 2 agents), hard (multi-fault, needs Wonder). Pre-cached.
  - *Done when:* each runs < 60 s or loads instantly from cache.
- [ ] **P12-T3: README + architecture docs.** Setup < 10 min, diagram, metric definitions, SDK guide.
  - *Done when:* a fresh clone reproduces the demo from the README.
- [ ] **P12-T4: Reproducibility package.** `make reproduce` from cached model outputs.
  - *Done when:* numbers match the report exactly.
- [ ] **P12-T5: Submission assets.** Description, architecture figure, results table, screenshots/GIFs, limitations, short video.
  - *Done when:* matches the hackathon checklist.
- [ ] **P12-T6: Final QA.** Tests, lint, clean-clone setup, offline demo.
  - *Done when:* `docs/release_checklist.md` fully ticked.
