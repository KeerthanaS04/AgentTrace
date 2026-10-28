# CLAUDE.md: AgentTrace

> Project guide for Claude Code. Read this file fully before starting any task.
> Work **one task at a time**, in phase order, and tick items in `PROGRESS.md` as you finish them.

---

## 1. Project Summary

**AgentTrace** is a debugging and continuous-explainability platform for multi-agent AI workflows.

It does five things:

1. **Instruments** multi-agent workflows (agents, tools, retrieval, intermediate decisions) and records full traces.
2. **Reconstructs** the execution as a data-flow dependency graph (DAG).
3. **Diagnoses** failures: separates *root causes* from *propagated errors*, using Meta's **Muse Spark** (fast pass) and **Muse Wonder** (deep investigation), with **NVIDIA** and other **open-source** models as independent judges.
4. **Verifies** diagnoses through **counterfactual replay**: patch the suspected node, re-run only its descendants, and measure whether the final outcome actually improves.
5. **Monitors continuously**: scores every agent step for explainability (grounding, rationale-action consistency, faithfulness, calibration, drift) and tracks it over time in an evaluation platform.

**One-line pitch:** *AgentTrace continuously monitors how well multi-agent systems explain their decisions, and when something breaks, it traces the error to its origin and proves the fix through replay.*

**Demo:** a multi-agent customer-support workflow with deliberately injected failures (bad retrieval, faulty tool responses, invalid intermediate outputs), with a visualized failure chain and a side-by-side original vs. corrected replay.

**Success is measured, not narrated.** Every claim in the demo must be backed by a number from the evaluation harness (Phase 9).

---

## 2. Non-Negotiable Engineering Rules

1. **Never hardcode secrets.** Read keys from environment variables (`.env`, git-ignored). Provide `.env.example`.
2. **Never invent model IDs or endpoints.** Model IDs live only in `config/models.yaml`. If an ID is marked `TODO_VERIFY`, stop and ask the user to confirm it from the provider dashboard.
3. **All LLM calls go through the model gateway** (`agenttrace/gateway/`). No direct SDK/HTTP calls elsewhere. The gateway handles caching, retries, logging, cost accounting, and structured-output validation.
4. **Determinism first.** Temperature 0 (or lowest allowed) for judges and replay. Every LLM/tool call is cached by content hash. Replay must be reproducible.
5. **All structured model output is validated with Pydantic.** On validation failure: retry up to 2 times with the error message, then record a `parse_failure` and continue; never crash a pipeline run.
6. **Evidence-grounded explanations only.** Every diagnosis must cite span IDs from the trace. A diagnosis with no evidence pointers is invalid.
7. **Ground truth is never visible to the diagnosis pipeline.** Only the evaluation harness may read `ground_truth` fields. Enforce this in code (separate loader, no import path from `diagnosis/` to `scenarios/ground_truth`).
8. **Report oracle and realistic results separately.** Replay with ground-truth patches is an *upper bound*; replay with model-proposed patches is the *real* result. Never mix them.
9. **Tests per task.** Each task ships with unit tests and meets its "Done when" criteria. Commit once per task with message `P{phase}-T{n}: <summary>`.
10. **Ask before changing a schema** in `agenttrace/schemas/`. Schemas are the contract between all components.
11. **Cost awareness.** Log tokens and USD per call. Use cheap/open models for high-volume scoring; escalate to Muse Wonder only when triggered by explicit rules.

---

## 3. Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ (backend, agents, analysis), TypeScript (UI) |
| Schemas | Pydantic v2 |
| Graph | `networkx` |
| API server | FastAPI + Uvicorn |
| Storage | SQLite (via SQLAlchemy) for runs, spans, scores, diagnoses; JSON files for scenarios |
| Cache | On-disk content-addressed cache (`.cache/`, SQLite or `diskcache`) |
| Demo agents | Plain Python with a lightweight agent runtime (no heavy framework) so tracing and replay are fully controlled; optional LangGraph adapter as a stretch |
| Retrieval | Local vector store (Chroma or FAISS) + small embedding model, or BM25 as fallback |
| Frontend | React + Vite + TypeScript + Tailwind, **React Flow** (graph), **Recharts** (time series) |
| Testing | `pytest`, `pytest-asyncio`, `vitest` for UI |
| Tooling | `ruff`, `mypy` (lenient), `pre-commit`, `Makefile` |
| Model access | Meta Model API (Muse Spark, Muse Wonder), NVIDIA NIM (Nemotron etc.), open-source models via NIM/OpenRouter/local vLLM, all through an OpenAI-compatible client where possible |

### Model access notes (verify before use)

- **Meta Model API:** base URL `https://api.meta.ai/v1`, Bearer auth with `MODEL_API_KEY`. Supports Responses, Chat Completions, and Messages formats (OpenAI-compatible and Anthropic-compatible). Muse Wonder model ID: `muse-wonder-1.0`. Muse Spark exact ID: `TODO_VERIFY` (check the Model API dashboard). Wonder pricing is roughly $4/M input and $20/M output tokens, so use it sparingly.
- **NVIDIA:** hosted NIM endpoints (OpenAI-compatible) with `NVIDIA_API_KEY`. Exact Nemotron model name: `TODO_VERIFY`.
- **Open-source judge/agent models:** Llama / Qwen / Mistral via NIM, OpenRouter, or local vLLM. Exact names: `TODO_VERIFY`.
- Check the hackathon rules on whether non-Meta models are allowed and how central Muse models must be. Default framing: **Muse models are the primary reasoning engine; open-source/NVIDIA models are swappable judges and agents-under-test.**

---

## 4. Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          DEMO WORKFLOW                             │
│  Router → Retriever → Tool Agent → Policy Agent → Response Writer  │
│        (agents under test; swappable underlying models)            │
└───────────────┬────────────────────────────────────────────────────┘
                │ @traced spans (inputs, outputs, provenance refs)
                ▼
┌──────────────────────┐    ┌──────────────────────────────────────┐
│  TRACE STORE         │◄───│  FAULT INJECTOR (scenario harness)   │
│  runs/spans/scores/  │    │  injects faults, holds ground truth  │
│  diagnoses (SQLite)  │    └──────────────────────────────────────┘
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  GRAPH BUILDER       │  spans → data-flow DAG (+ contracts per node)
└─────────┬────────────┘
          ▼
┌────────────────────────────────────────────────────────────────────┐
│  DIAGNOSIS ENGINE                                                  │
│  A. Per-node anomaly scoring (cheap judges: Spark / NVIDIA / OSS)  │
│  B. Causal attribution: root cause vs propagated (graph walk)      │
│  C. Escalation to Muse Wonder (disagreement / low confidence)      │
│  D. Fix proposal (targeted, per node type)                         │
└─────────┬──────────────────────────────────────────────────────────┘
          ▼
┌────────────────────────────────────────────────────────────────────┐
│  COUNTERFACTUAL REPLAY ENGINE                                      │
│  freeze upstream · patch suspect · re-run descendants · compare    │
│  → verification loop: try next candidate if no recovery            │
└─────────┬──────────────────────────────────────────────────────────┘
          │
          ├──────────────► CONTINUOUS EXPLAINABILITY SCORER
          │                (grounding, RAC, faithfulness, calibration, drift)
          ▼
┌────────────────────────────────────────────────────────────────────┐
│  EVALUATION HARNESS  (metrics, baselines, ablations, reports)      │
└─────────┬──────────────────────────────────────────────────────────┘
          ▼
┌────────────────────────────────────────────────────────────────────┐
│  PLATFORM: FastAPI backend + React UI                              │
│  graph view · diagnosis · replay diff · explainability dashboard   │
│  regression suite · alerts · eval results                          │
└────────────────────────────────────────────────────────────────────┘
```

### 4.1 Repository Layout

```
agenttrace/
├── CLAUDE.md
├── PROGRESS.md                  # phase/task checklist (you maintain this)
├── README.md
├── Makefile
├── pyproject.toml
├── .env.example
├── config/
│   ├── models.yaml              # model roles → provider, id, params, pricing
│   ├── explainability.yaml      # metric weights, thresholds, sampling rates
│   └── escalation.yaml          # rules for escalating to Muse Wonder
├── agenttrace/
│   ├── schemas/                 # Pydantic: Span, Run, Graph, Diagnosis, Score, Scenario
│   ├── gateway/                 # model gateway: providers, cache, retry, cost, structured output
│   ├── tracing/                 # @traced, TraceContext, Ref (provenance), exporters
│   ├── store/                   # SQLAlchemy models, repositories, migrations
│   ├── graph/                   # graph builder, queries (ancestors, descendants, critical path)
│   ├── workflow/                # demo support workflow
│   │   ├── agents/              # router, retriever, tool_agent, policy_agent, writer
│   │   ├── tools/               # mock order/refund/account APIs
│   │   ├── kb/                  # knowledge base docs + index builder
│   │   └── runtime.py           # lightweight agent runtime
│   ├── faults/                  # fault injectors + fault catalog
│   ├── scenarios/               # scenario generator, loader (ground truth isolated here)
│   ├── replay/                  # record/replay cache, patcher, re-executor, differ
│   ├── diagnosis/               # anomaly scoring, attribution, escalation, fix proposal, verification loop
│   ├── explainability/          # metrics, claim extraction, ablation, calibration, drift, aggregator
│   ├── judges/                  # judge prompts, ensemble, agreement stats
│   ├── evaluation/              # metrics, baselines, ablations, report generator
│   ├── api/                     # FastAPI app and routers
│   └── cli.py                   # `agenttrace run|diagnose|replay|score|eval|serve`
├── ui/                          # React app
├── data/
│   ├── scenarios/               # generated scenario JSON (inputs + ground truth)
│   └── kb/                      # source KB documents
├── tests/
├── notebooks/                   # analysis for the final report (optional)
└── docs/                        # architecture diagram, metric definitions, demo script
```

### 4.2 Core Data Contracts (`agenttrace/schemas/`)

Define these first (Phase 0). All other components depend on them.

```python
class Ref(BaseModel):                 # provenance-carrying pointer to a value
    span_id: str                      # span that produced the value
    key: str                          # which output field
    content_hash: str

class Span(BaseModel):
    span_id: str
    run_id: str
    node_id: str                      # LOGICAL node id (stable across runs), e.g. "retriever.search"
    agent: str
    kind: Literal["llm", "tool", "retrieval", "decision", "message"]
    input: dict                       # resolved input values
    input_refs: list[Ref]             # DATA-FLOW edges: which prior outputs fed this span
    output: dict
    explanation: Explanation | None   # structured self-explanation (see below)
    model: str | None
    params: dict
    started_at: datetime
    ended_at: datetime
    tokens_in: int; tokens_out: int; cost_usd: float
    error: str | None
    patched: bool = False             # True if replaced during counterfactual replay

class Explanation(BaseModel):
    rationale: str
    cited_evidence: list[str]         # span_ids / doc_ids the agent says it relied on
    confidence: float                 # 0..1 self-reported

class NodeContract(BaseModel):        # what a step SHOULD do; used by judges
    node_id: str
    description: str
    input_expectations: str
    output_expectations: str
    programmatic_checks: list[str]    # names of deterministic validators (schema, range, etc.)

class Run(BaseModel):
    run_id: str; scenario_id: str | None; workflow_version: str
    agent_models: dict[str, str]
    final_output: dict; final_correct: bool | None
    spans: list[Span]

class AnomalyScore(BaseModel):
    span_id: str; judge: str
    output_ok_given_inputs: float     # 0..1 local correctness
    inputs_ok: float                  # 0..1 trust in inputs
    verdict: Literal["ok", "suspicious", "faulty"]
    evidence: list[str]; reasoning: str

class Diagnosis(BaseModel):
    run_id: str
    root_causes: list[RootCause]      # ranked; each has node_id, span_id, fault_type, confidence, evidence
    propagated: list[PropagatedError] # span_id, caused_by (root span ids), path
    recommended_fixes: list[Fix]
    escalated_to_deep: bool
    judge_agreement: float
    cost_usd: float

class ReplayResult(BaseModel):
    original_run_id: str; replay_run_id: str
    patched_span_id: str; patch_source: Literal["oracle", "model", "rule"]
    changed_spans: list[str]
    original_final: dict; replay_final: dict
    recovered: bool; recovery_score: float
```

---

## 5. Phases and Tasks

**Legend:** each task lists *what to build*, *key files*, and **Done when**.
**Milestones** (see §8): M1 = vertical slice (end of Phase 7, single scenario), M2 = full eval, M3 = explainability, M4 = platform + UI, M5 = polish.

---

### Phase 0: Foundations

**Goal:** a clean repo, shared schemas, and a working model gateway.

- **P0-T1: Repo scaffold.** Create the layout in §4.1, `pyproject.toml`, `Makefile` (`make setup test lint serve eval`), `.env.example`, `pre-commit`, `PROGRESS.md` with every task below as a checkbox.
  *Done when:* `make setup && make test` runs green on an empty test suite.
- **P0-T2: Schemas.** Implement all models in §4.2 under `agenttrace/schemas/` with JSON-serializable round-trip tests.
  *Done when:* every schema round-trips to/from JSON; invalid data is rejected in tests.
- **P0-T3: Model gateway.** `gateway/` with provider adapters (Meta Model API, NVIDIA NIM, generic OpenAI-compatible), role-based lookup from `config/models.yaml`, content-hash cache, retry with backoff, token and cost accounting, JSON/structured-output helper with Pydantic validation and retry.
  *Done when:* a single call `gateway.complete(role="judge_fast", ...)` works for each configured provider; a second identical call is served from cache and logged as `cache_hit`; cost is recorded.
- **P0-T4: Model config.** `config/models.yaml` defining roles: `judge_fast` (Muse Spark), `judge_deep` (Muse Wonder `muse-wonder-1.0`), `judge_nvidia`, `judge_oss`, `agent_default`, `agent_alt_1`, `agent_alt_2`, plus pricing per provider. Unknown IDs marked `TODO_VERIFY`.
  *Done when:* swapping a role's model requires editing config only.
- **P0-T5: Smoke tests and budget guard.** A `make smoke` command pings each configured model with a trivial prompt; a global budget cap (`MAX_USD_PER_RUN`, `MAX_USD_TOTAL`) aborts runs that exceed it.
  *Done when:* smoke passes for available providers and gracefully skips unavailable ones.

---

### Phase 1: Demo Workflow (Customer-Support Multi-Agent System)

**Goal:** a realistic, deterministic-enough workflow to instrument and break.

- **P1-T1: Knowledge base.** 30–50 short policy/FAQ docs (refund policy, shipping, warranty, account issues, escalation rules) in `data/kb/`, with unique IDs and explicit facts (dates, amounts, windows) so wrong retrieval produces detectable wrong answers.
  *Done when:* each doc has an ID, title, body, and 1+ checkable facts.
- **P1-T2: Retrieval index.** Build the vector (or BM25) index; expose `search(query, k)` returning doc IDs and scores.
  *Done when:* top-3 recall ≥ 0.9 on a hand-written set of 20 queries.
- **P1-T3: Mock tools.** `order_lookup`, `refund_api`, `account_status`, backed by a small JSON fixture database; deterministic outputs with strict response schemas.
  *Done when:* each tool has typed input/output and unit tests.
- **P1-T4: Agent runtime.** `workflow/runtime.py`: a minimal runtime where every agent is `run(inputs: dict[str, Ref|value], ctx) -> AgentOutput`, and a workflow is a declared graph of steps. Each agent returns both an `output` and an `Explanation`.
  *Done when:* a toy 2-agent workflow runs end to end via the runtime.
- **P1-T5: The five agents.**
  1. **Router:** classifies the ticket (refund, shipping, account, other).
  2. **Retriever:** forms a query and selects policy docs.
  3. **Tool Agent:** decides which tool to call, calls it, interprets the result.
  4. **Policy Agent:** combines retrieved policy and tool data into a decision (approve/deny/escalate) with justification.
  5. **Response Writer:** drafts the customer reply.
  Each has a `NodeContract` (description, expectations, programmatic checks) stored in `workflow/contracts.py`.
  *Done when:* 90%+ of 20 clean tickets get the correct final decision with the default model.
- **P1-T6: Model-swappable agents.** Each agent's underlying model is configurable per agent (`agent_default`, `agent_alt_*`), enabling model-agnostic evaluation later.
  *Done when:* the workflow runs with at least 3 different underlying models via config only.
- **P1-T7: Final-outcome checker.** Deterministic correctness check per scenario (expected decision plus required facts in the reply), so `final_correct` doesn't depend on an LLM.
  *Done when:* checker is unit tested on correct and incorrect final outputs.

---

### Phase 2: Tracing and Instrumentation

**Goal:** capture every step with true data-flow provenance.

- **P2-T1: `@traced` decorator and `TraceContext`.** Wrap agent/tool/retrieval/LLM calls; record input, output, timings, tokens, cost, model, params, explanation. Stable logical `node_id` per step.
  *Done when:* running the workflow yields a complete `Run` with spans for all five agents plus nested tool/retrieval/LLM spans.
- **P2-T2: Provenance via `Ref`.** Values passed between steps are wrapped so the tracer records exactly which producer span each input came from (`input_refs`). No inferring edges from execution order.
  *Done when:* for a clean run, each span's `input_refs` exactly matches the workflow's declared data flow (asserted in a test).
- **P2-T3: Trace store.** SQLAlchemy models and repositories for runs, spans, scores, diagnoses, replays; CLI `agenttrace run --scenario X`.
  *Done when:* runs persist and reload byte-identically; queries by run, node, and scenario work.
- **P2-T4: Trace export/import.** JSON export/import (and optional OpenTelemetry-style export) so traces from other systems could be loaded.
  *Done when:* exporting then importing produces an identical `Run`.
- **P2-T5: Trace integrity checks.** Validate traces (no orphan refs, no cycles, timestamps consistent); fail loudly with clear errors.
  *Done when:* corrupted-trace fixtures are rejected by tests.

---

### Phase 3: Graph Reconstruction

**Goal:** turn traces into a queryable causal DAG.

- **P3-T1: Graph builder.** Build a `networkx.DiGraph` from spans: nodes = spans, edges = `input_refs`. Attach node attributes (agent, kind, contract, anomaly placeholders).
  *Done when:* graph matches declared workflow topology on clean runs.
- **P3-T2: Graph queries.** `ancestors`, `descendants`, `topological_order`, `paths_to_final(span)`, `critical_path`, `first_divergence(run_a, run_b)`.
  *Done when:* queries are tested on hand-built graphs.
- **P3-T3: Logical-node alignment.** Map spans between two runs of the same workflow by `node_id` (needed for replay diffs and regression comparison).
  *Done when:* aligning an original and a replay run yields a correct changed/unchanged node set.
- **P3-T4: Graph serialization for UI.** Export graph as JSON (nodes, edges, status colors, layout hints) for React Flow.
  *Done when:* the JSON validates against a schema the UI will consume.

---

### Phase 4: Fault Injection and Scenario Dataset

**Goal:** a labeled dataset with known root causes (the basis of every metric).

- **P4-T1: Fault catalog.** Implement injectors, each a pure function that mutates one node's output at record time:
  - `F1` wrong/stale retrieval (returns an incorrect but plausible doc)
  - `F2` empty or irrelevant retrieval
  - `F3` faulty tool response (wrong value, e.g. wrong order status or amount)
  - `F4` malformed or error tool response (schema violation, timeout payload)
  - `F5` invalid intermediate decision (wrong routing, wrong policy decision)
  - `F6` hallucinated intermediate claim (fabricated fact in an agent's output)
  - `F7` multi-fault (two independent roots in one run)
  - `F8` silent fault (a fault that does *not* change the final outcome; tests early detection)
  - `F0` control (no fault; tests false-positive rate)
  *Done when:* each injector has unit tests proving the mutation is applied at the intended node only.
- **P4-T2: Scenario schema and generator.** A `Scenario` = ticket + fixture data + fault spec + **ground truth** (`root_cause_node_ids`, `affected_node_ids`, `expected_final`, `fault_type`). Generate 80–120 scenarios (≥10 per fault type, ≥15 clean controls), varied across ticket categories.
  *Done when:* the dataset validates, is balanced, and is committed as JSON in `data/scenarios/`.
- **P4-T3: Ground-truth isolation.** `scenarios/ground_truth.py` is the only module that exposes truth; `diagnosis/` and `explainability/` cannot import it (enforced by an import-linter or test).
  *Done when:* a test fails if any forbidden import path exists.
- **P4-T4: Scenario verification.** Run every scenario; confirm faults actually cause the expected downstream effect (or, for `F8`, intentionally do not). Drop or fix scenarios whose fault is ineffective.
  *Done when:* ≥95% of scenarios behave as labeled; the rest are documented or removed.
- **P4-T5: Multi-run protocol.** Support K repeated runs per scenario with different seeds (default K=3) for variance estimates.
  *Done when:* the runner stores K runs per scenario and reports pass-rate variance.

---

### Phase 5: Record/Replay Engine (hardest component; start early)

**Goal:** reproducible re-execution with surgical interventions.

- **P5-T1: Record/replay cache.** Content-addressed cache keyed by `(node_id, model, params, hash(inputs))` for LLM and tool calls. Replay mode: unchanged inputs ⇒ cache hit ⇒ identical output.
  *Done when:* replaying an unmodified run reproduces every span output exactly (zero diffs).
- **P5-T2: Patcher.** Given a run, a target `span_id`, and a replacement output, produce a replay plan: freeze all non-descendants, mark the target `patched=True`, and mark descendants for re-execution.
  *Done when:* the plan's frozen and re-run sets are correct on hand-built graphs.
- **P5-T3: Re-executor.** Execute the plan in topological order; descendants run live (cache-aware), frozen nodes reuse recorded outputs; the replay run is stored linked to the original.
  *Done when:* patching a known-bad node on a verified scenario changes only its descendants.
- **P5-T4: Patch sources.** Implement three sources: `oracle` (ground-truth value, upper bound), `model` (model-proposed fix), `rule` (deterministic fix, e.g. re-run retrieval with a corrected query, re-call the tool with validation). Tag `patch_source` on every result.
  *Done when:* each source works on at least one fault type end to end.
- **P5-T5: Differ and recovery scoring.** Compare original vs. replay: changed spans, final output diff, `recovered` (final now correct per the deterministic checker), and `recovery_score` (partial credit, e.g. required-facts coverage).
  *Done when:* the differ's output feeds both the UI and the evaluation harness.
- **P5-T6: Replay stability test.** Repeat the same replay 3 times; results must be identical (cached) or within a documented tolerance (live).
  *Done when:* a stability report is generated and the tolerance is documented.

---

### Phase 6: Diagnosis Engine

**Goal:** from a failed (or suspicious) run to a ranked, evidence-backed root-cause hypothesis.

- **P6-T1: Judge prompts and schemas.** Versioned prompts in `judges/prompts/` producing `AnomalyScore`. The judge sees: the node contract, the node's inputs, the node's output, the node's explanation, and **parent outputs summarized**, and must answer two separate questions: *(a) Are the inputs trustworthy? (b) Given these inputs, is the output correct?* Output must cite span IDs and doc IDs.
  *Done when:* prompts return schema-valid output on 20 sample spans per judge model.
- **P6-T2: Deterministic validators.** Programmatic checks per node (schema validity, value ranges, tool-error flags, retrieved-doc relevance threshold, claim-vs-source numeric match). These produce hard evidence and short-circuit expensive judging.
  *Done when:* validators catch all `F4` faults and most `F2` faults without any LLM call.
- **P6-T3: Per-node anomaly scoring (Stage A).** Run the fast judge(s) over every node in topological order; combine with validator results.
  *Done when:* every span in a run gets an `AnomalyScore`; cost per run is logged.
- **P6-T4: Judge ensemble.** Run `judge_fast` (Muse Spark), `judge_nvidia`, and `judge_oss` independently; aggregate by weighted vote; compute pairwise agreement (Cohen's κ) and per-node disagreement.
  *Done when:* ensemble output includes per-judge verdicts, the aggregate verdict, and an agreement score.
- **P6-T5: Causal attribution (Stage B).** Algorithm:
  1. Mark nodes with aggregate verdict `faulty|suspicious`.
  2. A node is a **root-cause candidate** if its output is bad *and* its inputs are judged trustworthy (`inputs_ok` high) *or* all anomalous parents are themselves unexplained.
  3. A node is a **propagated error** if its output is bad and at least one parent feeding it is anomalous and plausibly explains it; record the path back to the root(s).
  4. Rank candidates by (earliest in topological order, anomaly confidence, size of affected descendant set).
  *Done when:* on 10 hand-checked scenarios, the attribution output is correct and each propagated node has a valid path.
- **P6-T6: Escalation to Muse Wonder (Stage C).** Rules in `config/escalation.yaml`: escalate when (a) ensemble agreement < threshold, (b) top-2 candidates have close confidence, (c) multiple roots suspected, (d) chain longer than N hops, or (e) validators and judges conflict. Wonder receives the compact graph, candidate list, and evidence, and returns a final ranking with reasoning.
  *Done when:* escalation triggers on the right cases in tests and its cost is tracked separately.
- **P6-T7: Fix proposal (Stage D).** Per-candidate targeted fix recommendation by fault type (e.g. "re-query with a more specific query," "validate tool payload against schema before use," "add a routing confidence threshold"). Output a structured `Fix` with a concrete `patch` the replay engine can apply.
  *Done when:* every diagnosed root cause has a `Fix` containing a machine-applicable patch.
- **P6-T8: Diagnosis report.** Assemble a `Diagnosis` object plus a human-readable evidence narrative that only cites recorded spans.
  *Done when:* a lint test confirms every claim in the narrative references an existing span ID.

---

### Phase 7: Counterfactual Verification Loop

**Goal:** prove or reject diagnoses by experiment. (**Milestone M1: full vertical slice on one scenario.**)

- **P7-T1: Verification loop.** For each ranked root-cause candidate: apply its proposed patch → replay → check recovery. If recovered, mark the diagnosis **verified**; if not, mark **unverified**, demote its confidence, and try the next candidate. Cap attempts (default 3).
  *Done when:* the loop terminates, and results are stored as `ReplayResult` rows linked to the diagnosis.
- **P7-T2: Interventional ranking.** When several candidates exist, rank by measured recovery (not just judge confidence). Distinguish "necessary" (fix recovers) from "contributing" (partial recovery).
  *Done when:* multi-fault scenarios (`F7`) yield correct combined attribution when patching both roots.
- **P7-T3: Propagation verification.** Confirm propagated errors: after patching the root cause, nodes labeled "propagated" should become correct in the replay; nodes that stay wrong are re-labeled as additional roots.
  *Done when:* propagation labels are updated automatically from replay evidence.
- **P7-T4: End-to-end command.** `agenttrace diagnose --run <id>` runs Stages A→D plus verification and prints a summary; `agenttrace diagnose --scenario <id>` runs the whole pipeline from a scenario.
  *Done when:* **M1 reached**: one command takes an injected-failure scenario from "wrong final answer" to "verified root cause with recovered replay."

---

### Phase 8: Continuous Explainability

**Goal:** measure, per step and over time, how well agents' decisions are grounded and explainable.

- **P8-T1: Metric definitions doc.** `docs/explainability_metrics.md` with formulas, ranges, inputs, and failure interpretation for each metric below. Configurable weights in `config/explainability.yaml`.
  *Done when:* each metric has a precise, testable definition.
- **P8-T2: Grounding score (G).** Extract atomic claims from a step's output/explanation; check each against the evidence the agent had (retrieved docs, tool results) via an entailment judge. `G = supported_claims / total_claims`.
  *Done when:* G matches human labels on 30 hand-labeled spans (report agreement).
- **P8-T3: Rationale-Action Consistency (RAC).** Does the stated rationale actually entail the action/decision taken (e.g. rationale says "outside refund window" but decision is "approve")? Score 0–1 via judge plus deterministic checks where possible.
  *Done when:* RAC flags all injected `F5` and `F6` inconsistencies in the labeled set.
- **P8-T4: Faithfulness via ablation (Fa).** For sampled spans: remove the evidence the agent *cited* and re-run the step; faithfulness is high if the output materially changes when cited evidence is removed and does *not* change when random uncited evidence is removed. `Fa = P(change | cited removed) − P(change | uncited removed)`, clipped to [0,1]. Uses the replay cache and runs on a sampled fraction (config) to control cost.
  *Done when:* Fa is computed for a sampled subset and separates faithful from unfaithful explanations on a constructed test set.
- **P8-T5: Calibration (C).** Compare self-reported `confidence` to correctness (deterministic checker for outcomes; ensemble consensus for step-level). Compute Expected Calibration Error (ECE) per agent over a window; `C = 1 − ECE`.
  *Done when:* ECE is computed per agent with a reliability diagram data export.
- **P8-T6: Composite Explainability Index (XI).** `XI = Σ w_i · metric_i` (G, RAC, Fa, C) with weights from config; always report components alongside the composite.
  *Done when:* XI is stored per span, per agent, and per run.
- **P8-T7: Continuous scoring pipeline.** A scorer that, on every new run, scores all steps using the cheap judge tier (Spark + open models), schedules ablations for sampled steps, and writes scores to the store. Runs as a background job and via CLI (`agenttrace score --run <id>`).
  *Done when:* running 20 scenarios populates scores for every span automatically.
- **P8-T8: Drift detection and alerts.** Rolling-window statistics per agent and metric against a baseline; detect drops using z-score/CUSUM; emit `Alert` records with links to affected runs and the causal graph. Trigger a diagnosis automatically for runs whose XI falls below a threshold even when the final answer looks correct (this is the "silent fault" use case).
  *Done when:* injecting a degradation (e.g. a worse model or a corrupted KB doc mid-stream) raises an alert within a defined number of runs.
- **P8-T9: Link explainability to diagnosis.** Use XI components as additional evidence features in Stage A/B (low grounding on a node boosts its anomaly prior) and use diagnosis results to label which low-XI nodes were true failures.
  *Done when:* the integration is switchable by flag so its benefit can be ablated in Phase 9.

---

### Phase 9: Evaluation Harness

**Goal:** turn the whole system into numbers, with baselines and ablations.

- **P9-T1: Metrics implementation.**
  - **Root-cause accuracy:** top-1 and top-3 match against ground-truth root nodes.
  - **Propagation detection:** precision, recall, and F1 over the set of affected downstream nodes.
  - **Recovery rate:** fraction of scenarios where replay recovers the correct final output, reported separately for `oracle` and `model`/`rule` patch sources.
  - **False-positive rate:** diagnoses raised on `F0` control runs.
  - **Early-detection rate:** fraction of `F8` silent faults flagged by continuous scores before/without a visible final failure.
  - **Judge agreement:** Cohen's κ pairwise and Fleiss' κ overall; ensemble accuracy vs. each single judge.
  - **Explainability validity:** AUROC of XI (and each component) for predicting step-level failure; correlation with ground-truth faultiness.
  - **Cost and latency per diagnosis;** savings from tiering vs. Wonder-only.
  *Done when:* each metric has a unit-tested function on synthetic inputs.
- **P9-T2: Baselines.**
  - **B0:** paste the entire trace into one LLM and ask "what went wrong" (no graph, no replay).
  - **B1:** Spark-only graph diagnosis, no ensemble, no replay.
  - **B2:** graph diagnosis with ensemble, no replay verification.
  - **B3:** Wonder-only on everything (accuracy ceiling and cost ceiling).
  - **Full:** tiered ensemble + escalation + replay verification + explainability signals.
  *Done when:* all baselines run through the same harness on the same scenarios.
- **P9-T3: Ablations.** Toggle one component at a time: no validators, no ensemble, no escalation, no replay verification, no explainability features, no provenance edges (use execution order instead). Run each on the full dataset.
  *Done when:* an ablation table shows each component's contribution.
- **P9-T4: Model-agnostic evaluation.** Re-run the pipeline with ≥3 different agent-under-test models and with different judge sets, to show robustness.
  *Done when:* results are reported per agent-model and per judge-set.
- **P9-T5: Statistical reporting.** K repeated runs per scenario; bootstrap 95% confidence intervals for headline metrics; per-fault-type breakdown.
  *Done when:* every headline number in the report carries a CI and `n`.
- **P9-T6: Before/after improvement experiment.** Apply AgentTrace's recommended fixes to the workflow (e.g. add validators, improve retrieval query), re-run the suite, and report the change in task success and XI.
  *Done when:* a measured improvement (or an honest null result) is documented.
- **P9-T7: Report generator.** `agenttrace eval --report` produces `reports/results.md`, CSVs, and plot data for the UI and the submission.
  *Done when:* one command regenerates all results from stored runs.
- **P9-T8: Failure analysis.** Categorize where AgentTrace is wrong (confusions by fault type, judge errors, replay non-recovery) and write limitations honestly.
  *Done when:* `docs/limitations.md` is written from real error cases.

---

### Phase 10: Platform Backend

**Goal:** an API over everything so the UI and external users can use it.

- **P10-T1: FastAPI skeleton.** App factory, config, CORS, error handling, OpenAPI docs.
  *Done when:* `make serve` starts the API with `/health`.
- **P10-T2: Run and trace endpoints.**
  `POST /runs` (ingest trace), `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/graph`, `GET /runs/{id}/spans/{span_id}`.
  *Done when:* endpoints are tested with fixture runs.
- **P10-T3: Diagnosis and replay endpoints.**
  `POST /runs/{id}/diagnose`, `GET /runs/{id}/diagnosis`, `POST /runs/{id}/replay`, `GET /replays/{id}`, `GET /runs/{id}/compare/{other_id}`.
  Long jobs run in the background with status polling.
  *Done when:* a diagnosis can be started and polled to completion via API.
- **P10-T4: Explainability endpoints.**
  `GET /explainability/agents` (time series), `GET /explainability/runs/{id}`, `GET /alerts`, `POST /alerts/{id}/ack`.
  *Done when:* time-series responses match stored scores.
- **P10-T5: Evaluation and regression endpoints.**
  `POST /suites/{name}/run`, `GET /suites/{name}/results`, `GET /suites/{name}/compare?a=&b=` (score deltas between workflow versions).
  *Done when:* running the scenario suite twice yields a comparable delta report.
- **P10-T6: SDK / ingestion adapter.** A tiny client library (`agenttrace-sdk`) with `@traced` and an HTTP exporter so another Python workflow can send traces. Include an example integrating a second, toy workflow to prove it isn't hardcoded to the demo.
  *Done when:* a second workflow's traces are ingested and diagnosed.
- **P10-T7: Settings endpoints.** Read/update model roles, judge sets, escalation thresholds, and sampling rates (non-secret fields only).
  *Done when:* config changes apply without restart and are logged.

---

### Phase 11: UI

**Goal:** make the failure chain and the explainability story obvious in seconds.

- **P11-T1: UI scaffold.** Vite + React + TS + Tailwind, routing, API client generated from OpenAPI, layout shell.
  *Done when:* app loads and lists runs from the API.
- **P11-T2: Run detail with causal graph (React Flow).** Nodes colored by status: **ok / suspicious / propagated error / root cause**; edges show data flow; click a node to open the span panel (input, output, explanation, anomaly scores per judge, validators, XI components).
  *Done when:* a diagnosed scenario visually shows the root cause and the propagation path.
- **P11-T3: Evidence panel.** Side panel showing the diagnosis narrative with clickable span references that highlight nodes in the graph; shows which judges agreed/disagreed and whether Wonder was escalated.
  *Done when:* every cited span ID in the narrative is clickable and highlights the node.
- **P11-T4: Replay comparison view.** Side-by-side original vs. corrected replay: graph diff (changed nodes highlighted), final output diff, recovery verdict, patch source label (`oracle` / `model` / `rule`).
  *Done when:* the view shows exactly which nodes changed and whether the outcome recovered.
- **P11-T5: Continuous explainability dashboard.** Time-series per agent for G, RAC, Fa, C, XI; drift alerts with links to affected runs; reliability diagram; per-agent comparison across underlying models.
  *Done when:* injecting a degradation produces a visible dip and an alert in the UI.
- **P11-T6: Regression suite page.** Run the scenario suite against workflow version A vs. B; show score deltas, newly failing and newly fixed scenarios.
  *Done when:* the page shows a meaningful diff between two versions.
- **P11-T7: Evaluation results page.** Headline metric cards (with CIs), baseline comparison chart, ablation table, per-fault-type breakdown, cost tiering chart.
  *Done when:* page renders from `reports/` data with no manual steps.
- **P11-T8: Scenario launcher.** UI control to choose a scenario/fault, run the workflow, and watch the diagnosis pipeline progress (for live demos).
  *Done when:* a full live demo can be driven entirely from the UI.
- **P11-T9: Polish and resilience.** Loading/error/empty states, light/dark readability, keyboard-friendly navigation, offline demo mode that replays stored runs without any API calls.
  *Done when:* the demo works with the network off using cached data.

---

### Phase 12: Demo, Documentation, and Submission

- **P12-T1: Demo script (3 minutes).** `docs/demo_script.md`: (1) show a failing support ticket with a wrong final answer, (2) show the causal graph and the root cause vs. propagated errors, (3) show Muse Spark's evidence and the Wonder escalation on a hard multi-fault case, (4) show counterfactual replay recovering the answer, (5) show the explainability dashboard catching a silent degradation, (6) show headline metrics.
  *Done when:* a dry run fits in the time limit with no manual fixes.
- **P12-T2: Curated demo scenarios.** Pick 3: one easy (bad retrieval), one subtle (faulty tool value propagating through two agents), one hard (multi-fault needing Wonder). Pre-record and cache them for reliability.
  *Done when:* each runs in under 60 seconds or loads instantly from cache.
- **P12-T3: README and architecture docs.** Setup in under 10 minutes, architecture diagram, metric definitions, how to add a new workflow via the SDK.
  *Done when:* a fresh clone can reproduce the demo by following the README.
- **P12-T4: Reproducibility package.** One command (`make reproduce`) re-runs the evaluation from cached model outputs and regenerates all reported numbers.
  *Done when:* numbers match the report exactly.
- **P12-T5: Submission assets.** Project description, architecture figure, results table, screenshots/GIFs, limitations section, and a short video.
  *Done when:* all assets match the hackathon's submission checklist.
- **P12-T6: Final QA pass.** Run all tests, lint, a clean-clone setup, and the offline demo; fix anything flaky.
  *Done when:* checklist in `docs/release_checklist.md` is fully ticked.

---

## 6. Evaluation Targets (fill in real numbers; do not pre-claim)

| Metric | How measured | Baseline to beat |
|---|---|---|
| Root-cause top-1 / top-3 accuracy | vs. ground-truth root nodes | B0 (single LLM on raw trace) |
| Propagation P / R / F1 | affected downstream nodes | B0, B1 |
| Recovery rate (oracle) | replay with ground-truth patch | upper bound |
| Recovery rate (model/rule patch) | replay with proposed patch | B2 (no verification) |
| False-positive rate | diagnoses on clean controls | lower is better |
| Early-detection rate | silent faults flagged by XI | no-XI ablation |
| Judge agreement (κ) and ensemble gain | across Spark / NVIDIA / OSS | best single judge |
| XI validity (AUROC) | XI predicting step failure | each component alone |
| Cost per diagnosis | tiered vs. Wonder-only | B3 |

All results include CIs, `n`, and per-fault-type breakdowns.

---

## 7. Model Roles and Escalation Policy

| Role | Default model | Used for |
|---|---|---|
| `judge_fast` | Muse Spark | per-node anomaly scoring, claim checks, first-pass diagnosis |
| `judge_nvidia` | NVIDIA Nemotron (via NIM) | independent judge, open-weight option |
| `judge_oss` | Llama / Qwen / Mistral | independent judge, high-volume continuous scoring |
| `judge_deep` | Muse Wonder (`muse-wonder-1.0`) | escalated investigation of complex or conflicting cases |
| `agent_*` | configurable | agents under test (model-agnostic evaluation) |

**Escalate to `judge_deep` only when** ensemble agreement is low, top candidates are close, multiple roots are suspected, the chain is long, or validators and judges conflict (see `config/escalation.yaml`). Log the reason for every escalation.

---

## 8. Build Order, Milestones, and Cut Line

**Everything above is in scope.** Because time is limited, build in this order and keep each milestone demoable:

1. **M1: Vertical slice** (Phases 0, 1, 2, 3, 5, 6 minimal, 7): one scenario goes from trace → graph → diagnosis → verified replay. Build a thin version of every stage first, then deepen.
2. **M2: Evidence at scale** (Phase 4 full dataset, Phase 9 metrics and baselines): numbers exist before the UI is pretty.
3. **M3: Continuous explainability** (Phase 8): metrics, scorer, drift alerts.
4. **M4: Platform and UI** (Phases 10, 11).
5. **M5: Polish and submission** (Phase 12).

**Parallelizable work** (if multiple people or sessions): Phase 1 (workflow) ∥ Phase 4 (fault catalog) ∥ Phase 5 (replay cache); Phase 8 (metrics) ∥ Phase 11 (UI scaffold).

**If the schedule slips, cut from the bottom of this list first**, and document it in `docs/limitations.md`:
1. P11-T9 offline polish extras, P10-T7 settings endpoints
2. P9-T4 multi-agent-model evaluation, P9-T6 before/after experiment
3. P10-T6 second-workflow SDK demo
4. P8-T4 faithfulness ablation (keep G, RAC, C)
5. Third judge in the ensemble (keep Spark + one open model + Wonder escalation)

**Never cut:** schemas, gateway, provenance edges, fault dataset with ground truth, replay engine, verification loop, headline metrics, the 3-minute demo.

---

## 9. Working Agreements for Claude Code

- Start each session by reading this file and `PROGRESS.md`; state which task you're doing.
- Prefer small, composable modules; avoid cleverness. Type-hint everything.
- Don't add dependencies without a one-line justification in the commit message.
- When uncertain about a provider's API behavior or model ID, check the provider docs or ask the user. Do not guess.
- When a task's "Done when" is met, update `PROGRESS.md`, run `make test lint`, and commit.
- Keep prompts in versioned files, never inline in code, and record the prompt version on every `AnomalyScore` and `Diagnosis`.
- Record every surprising failure (parse errors, flaky replays, judge disagreements) in `docs/lessons.md`; these feed the limitations section.

---

## 10. Open Items (resolve with the user early)

- [ ] Hackathon deadline, team size, and submission format
- [ ] Confirmation that non-Meta models (NVIDIA, open-source) are allowed alongside Muse models
- [ ] Exact Muse Spark model ID and available rate limits / credits
- [ ] Which NVIDIA and open-source models to use (and hosting: NIM vs OpenRouter vs local vLLM)
- [ ] Overall token/USD budget for the hackathon
