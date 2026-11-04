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
| 2026-11-01 | Agents under test | No OpenRouter key, so all agent roles run on NVIDIA's free tier (see §2). Each agent asks its model for JSON that includes its own rationale / cited evidence / confidence, so the `Explanation` comes from the model that decided | Claude |
| 2026-11-01 | Retrieval design | The model writes 1–3 queries; BM25 does the search (reproducible). The ticket text is also searched, plus a standing "escalation rules" query whose docs are never cut by the 6-doc cap | Claude |
| 2026-11-02 | Per-agent model config | `config/agents.yaml` holds named profiles (step -> role, with a `default_role`); `config/models.yaml` maps role -> model. The runner takes `--profile` (one or several), never a model name. Profiles are validated against the models config at startup | Claude |
| 2026-11-02 | Tracing design | Active trace and open span are `ContextVar`s, so calls nest without passing a tracer. `@traced` sits on the three choke points (`Gateway.complete` → `llm`, `ToolBox.call` → `tool`, `BM25Index.search` → `retrieval`); the runtime opens one `agent` span per step (span_id = step id, so existing `input_refs` resolve) and a `message` span for the workflow input. node_id is logical and stable (`router`, `router.llm`, `tool_agent.order_lookup`, `retriever.search`); repeats keep the node_id and get span_id `#2`, `#3` (each JSON repair is its own `llm` span). Tokens/cost live only on `llm` spans (no double counting); an agent span's `model` is its LLM calls' model. Skipped steps get a zero-length span with the error. Without an active trace the decorated functions behave exactly as before | Claude |
| 2026-11-02 | Provenance (P2-T2) | Refs are minted by the runtime when a value is produced (producer span id, key, content hash) and handed to consumers via their declared sources, never recomputed by the consumer or inferred from order. Schema addition: `Ref.input_key` (optional), the consumer input the value was bound to, so provenance is exact per input (`writer.decision <- policy.decision`). `workflow/provenance.py::check_provenance(run, workflow)` checks every step span input by input, plus the hash chain producer output = ref = consumer input; `clean_eval` reports "provenance exact n/n". Data flow is step-level: nested llm/tool/search spans carry no refs and relate to their step by nesting (intra-agent refs would be a later refinement) | Claude |
| 2026-11-02 | Trace store (P2-T3) | SQLite via SQLAlchemy 2 (new dependency `sqlalchemy`), default `.cache/agenttrace.db`, override `--db` / `$AGENTTRACE_DB`. Each row stores the record's exact Pydantic JSON (`doc`, source of truth) plus indexed copies of query fields (run, scenario, node, kind, metric), so reloads are byte-identical by construction and re-validated (a corrupted row raises). Spans keep their recorded order (`seq`), key `(run_id, span_id)`, cascade-deleted with their run. A `meta.schema_version` row makes a mismatched database fail on open (no Alembic migrations yet). Stored run ids are `{scenario}-{profile}-r{n}`, so repeated runs of a scenario never collide | Claude |
| 2026-11-02 | Export/import (P2-T4) | `tracing/export.py`. Native: `{"format": "agenttrace.run" \| "agenttrace.runs", "version": 1, ...}`, re-validated strictly on import. OTLP/JSON: one resourceSpans per run, ids = sha256-derived hex (trace 32, span 16, deterministic), integer-nanosecond times, error status code 2, LLM spans with GenAI conventions (`gen_ai.request.model`, `gen_ai.usage.*_tokens`), `input.value` / `output.value`. Our exports also carry each full record (`agenttrace.span`, resource `agenttrace.run`) so they round-trip byte-identically (key order must be preserved: sorted keys broke identity in testing). Foreign OTel traces are mapped best-effort (name → node_id, OpenInference/GenAI attributes → kind, model, tokens; workflow_version "otel-import") | Claude |
| 2026-11-03 | Causal graph (P3-T1) | New dependency `networkx` (+ dev `types-networkx`). `graph/builder.py::build_graph(run, contracts=, nested=True)`: nodes = spans (key span_id; attrs node_id, agent, kind, model, status ok/error/skipped, parent, tokens, cost, contract, anomaly/verdict placeholders). Edges carry `kind`: "data" = input refs producer -> consumer (one edge per producer/consumer pair, all bindings listed); "nested" = child -> parent, so a fault inside a tool/LLM/search call is causally connected to its step's output (without them nested spans would be isolated). `data_subgraph` drops nesting; `node_topology` gives (producer node_id, consumer node_id) pairs, comparable across runs | Claude |
| 2026-11-04 | Causal attribution, Stage B (P6-T5) | `diagnosis/stage_b.py::attribute(run, scores, findings)`, no model calls, over one aggregate score per span (`aggregate_scores`: ensemble, else Stage A deterministic, else a single judge; inherited model-call copies dropped). **Feeders:** a step is fed by the steps it read from *and* the calls inside it; a call is fed by its step's data inputs. In causal order: a span is **anomalous** if its verdict is faulty or suspicious, or if a call inside it is anomalous (a step passes its calls' results on: its contract, which a validator checks; found by the hand-built tests, where a faulty refund quote inside a judge-passed tool_agent otherwise made Policy a spurious second root). Search spans count as anomalous only when a validator settled them (default; `judge_searches=True` lets judges flag them), following P6-T4 where 35 of 60 judge splits were on searches. **Root** = anomalous with no anomalous feeder, or with data inputs judged trustworthy (`inputs_ok` ≥ 0.5) and no anomalous call. **Propagated** = explained by an anomalous call inside it, or by an anomalous data feeder when its inputs are judged untrustworthy; `caused_by` = those feeders' roots; `path` = root → the explaining chain → span (`valid_path` checks it against the graph). Roots ranked by causal position, then confidence (1 − output_ok), then downstream size. Fault type comes from node kind and validator checks. `to_diagnosis()` gives the `schemas.Diagnosis` | Claude |
| 2026-11-04 | Judge ensemble (P6-T4) | **judge_oss = `qwen/qwen-2.5-72b-instruct` on OpenRouter** ($0.36 / $0.40 per M, verified on its OpenRouter page; `response_format` supported there, but json mode stays off provider-wide), a third model family besides Muse (Meta) and Nemotron (NVIDIA). **judge_nvidia thinking off** (`enable_thinking: false`), per the P6-T1 measurement (about 56 s per span and one all-reasoning truncation with it on), re-validated with `judge-sample`. **Aggregation** (`diagnosis/ensemble.py`, `config/ensemble.yaml`): weighted vote over the judges with a valid reply on the span (equal weights until Phase 9 calibrates them); ties go to the more severe verdict (faulty > suspicious > ok), so a split never hides a possible fault; `inputs_ok` / `output_ok` are weighted means; stored as judge `ensemble`. Spans settled without judges (validators, input, failed / skipped) keep their deterministic score and don't count toward agreement; model calls inherit their step's aggregate. **Agreement:** Cohen's κ per judge pair (3 categories; degenerate single-category cases are reported as 1.0 / 0.0) and Fleiss' κ, per run and pooled over all judged spans; the mean pairwise κ is the number `Diagnosis.judge_agreement` will hold; per-node `disagreement` = 1 − the winning verdict's weight share | Claude |
| 2026-11-04 | Stage A anomaly scoring (P6-T3) | `diagnosis/stage_a.py::score_run(run, gateway, roles)` visits spans in causal order and takes each span's score from the first source that applies: **input** (given; ok) / **failed** step (faulty, its error) / **skipped** step (suspicious) / **validators** (any validator *error*: faulty, no model call, judge skipped) / **judge** (steps, tool calls and searches with each role; validator warnings appended to the evidence, the verdict stays the judge's) / **no answer** (a judge that never gave a valid reply: suspicious 0.5 / 0.5, counted, never guessed) / **inherited** (model-call spans copy their step's score as judge `inherited:<role>`, since they repeat the step's prompt). For deterministic scores, `inputs_ok` = the lowest `output_ok_given_inputs` among the producer steps, already scored in causal order. Stored in a new `anomaly_scores` table (added by `create_all`, no schema-version bump, so old stores still open; one row per span + judge + prompt version, re-scoring replaces). Runs are chosen by id: `scenarios/devset.py` writes a stratified dev set as run ids only (2 verified runs per code F0–F8), so the diagnosis side reads which runs to score without seeing labels | Claude |
| 2026-11-04 | Deterministic validators (P6-T2) | `diagnosis/validators.py` (protected, no truth): `validate_run(run, ValidationContext(index, today))` gives `schemas.ValidatorFinding`s (span, check, error\|warning, message, evidence refs), and `short_circuit()` is true on any error, so a judge can be skipped. **Tool calls:** error payload; the tool's output-model schema; *record completeness* (the real tools always serialize every field, so a dropped optional field is caught even though the schema allows it); consistency within the record (day counts vs dates, lateness vs dates, subtotal vs items, fraud flag vs account status, Plus flag vs Plus date, quote status vs reject reason). **Searches:** empty (warning); hit not in the index (superseded docs aren't indexed); recorded score not reproducible from the index (`BM25Index.score`, the same arithmetic as `search`); a hit sharing no term with the query. **Steps:** the P1 contract checks plus: the router's order id is named in the ticket; retriever docs are not superseded and came from its own searches; the tool agent passed results on unchanged and the refund quote agrees with the order and account; policy doesn't approve what refund_api rejected; numbers stated in a justification or reply ($, %, durations) appear in the step's inputs (warning; ids and dates are stripped from the sources first, and a range must appear as that range). **Caveat:** the consistency, score-reproduction and completeness checks catch corruption *between* a component and its record, which is how faults are injected. They are the deterministic equivalent of real data corruption or tool and serving bugs, not of a model reasoning badly (that is the judges' job) | Claude |
| 2026-11-04 | Node judge prompt and schema (P6-T1) | Prompts are versioned files `judges/prompts/<name>_<vN>.md`; `VERSIONS.json` records each version's SHA-256 (CRLF normalized), and a test fails if a released prompt is edited. To change one, add v2 so old scores keep their `prompt_version` (`node-judge-v1`). **Context** (`judges/context.py`): for a step, its contract, full inputs, output and self-explanation, plus the outputs it read from its producers (via its input refs, only the keys it bound, summarized: strings cut at 400 chars, lists at 8) and the calls inside it (summarized). For a tool or search call: the tool's description, its args and result, and the enclosing step's inputs. Model-call spans aren't judged separately (they repeat their step's prompt). Nothing truth-derived is shown: no `final_correct`, no scenario id (a test checks the rendered prompt). **Reply** (`JudgeReply`): `inputs_ok` and `output_ok_given_inputs` (probabilities) with separate reasoning for (a) and (b), a verdict ok / suspicious / faulty (faulty = the output is wrong *given its inputs*, so bad inputs alone make a node suspicious, not faulty, which is what root-cause separation needs), and 1–5 evidence items `{ref, claim}`. Every ref must be a span id the context lists or a KB doc id appearing in it, checked inside the schema validation so a bad citation goes back to the model in the gateway's repair loop (max 2). Maps to `schemas.AnomalyScore` (judge = role, evidence = 'ref: claim'). Judges for the node prompt: judge_fast, judge_nvidia, judge_oss. Muse Wonder is the Stage C escalation model with its own prompt (P6-T6), so it isn't sampled here | Claude |
| 2026-11-04 | Replay stability (P5-T6) | `scenarios/stability.py`: the same plan replayed K times. **cached** = normal gateway; **live** = `Gateway.with_cache(empty)` per repetition (same config, keys, transports, ledger and budget, so the spend is still tracked). The tolerance is documented in §4. Patches are computed once, so variation is the replay's alone | Claude |
| 2026-11-04 | Differ and recovery scoring (P5-T5) | **Differ** (`replay/differ.py`, no truth): `diff_runs(original, replay)` labels every span of either run patched / changed (with how: input, output, model, ...) / new / missing / unchanged, on top of `align_runs`, and diffs the final output key by key. **Recovery** (`evaluation/recovery.py`, harness): `recovered` = the deterministic checker passes the replay's final output. `recovery_score` = 0.5 × decision right + 0.5 × share of required facts stated (the decision alone when there are none), so a right decision with half the facts scores 0.75. Scored against the same seed's clean run too: `clean_correct` (recovery was possible) and `matches_clean` (the replay reached the clean outcome, right or wrong). Recovery rates are reported over *recoverable* replays (P5-T3 finding). **Feeds the evaluation harness:** `replay_result()` builds the stored `schemas.ReplayResult` (two new optional fields, `clean_correct` and `matches_clean`; old records still load; `changed_spans` are local ids), saved by `patch-check --store` and by `agenttrace replay` for sourced patches on scenario runs. **Feeds the UI:** `to_ui_replay()` = the replay's React Flow graph plus `replay_status` / `replay_changes` per node and a `replay` block (patched, missing spans, final changes, counts, optional recovery, own legend). The committed JSON Schema was regenerated (additive; version stays 1) and a test validates a replay view against it | Claude |
| 2026-11-04 | Patch sources (P5-T4) | Three sources, tagged as `patch_source` on every `Patch`, plan, and replay `Run` (new field; `PatchSource` now has one definition, in `schemas.diagnosis`). **oracle** (`scenarios/oracle.py`, harness only): the node's output in the scenario's clean run with the same `model_params`, recomputed through the gateway cache; the upper bound. **rule** (`replay/sources.py`, no truth): re-call a tool with its recorded args (the result is schema-validated, so a wrong value or a malformed/error reply is replaced); re-run a search with its query; re-run the retriever step's searches from its recorded queries (reusing the Retriever's own `select_docs`, extracted for this); re-call every tool-agent call. Rules don't apply (n/a) to router, policy, writer and model calls, because their output is a model's judgement. **model** (`replay/sources.py`, no truth): role `patcher` (new; Muse Spark like judge_fast, since repair is part of diagnosis) sees the node contract, its inputs, its suspect output and an optional hint, and returns the output with the same keys. A wrong key set or no valid reply raises `PatchSourceError`, counted n/a. A test checks the prompt carries no ground truth. Rules are computed up front and handed over as replace-output patches, matching the schema's `ReplaceOutputPatch` (`ReplayPlan.schema_patch()`); the schema's override-inputs/params kinds stay for P7 fixes. The executor's in-memory result is renamed `ExecutedReplay`, so it isn't confused with the stored `schemas.ReplayResult` (P5-T5) | Claude |
| 2026-11-04 | Re-executor (P5-T3) | `replay/executor.py::execute_plan(run, workflow, plan, resources, faults=)` re-runs the workflow in topological order with a `PatchingReplay` hook over a replay cache filed from the original run. **Frozen** spans are served from the record; a frozen span with no recorded output raises `ReplayMissError` (the plan doesn't match the run), and is never silently re-run. **The target** gets the replacement and `patched=True`. A patched agent's explanation is dropped (it described the output the agent computed, not this one), but its own calls still replay. Patching `__input__` re-runs the workflow with the new input. **Re-run** spans are served from the record if their inputs are unchanged, else run through the gateway (whose cache still applies). Replays are stored with `replay_of`. `ReplayResult.violations()` lists anything outside the target and its descendants that changed. **Other faults stay in force** via `faults=`: without them, fixing one root of an F7 also silently removes a second root downstream, because Policy re-runs fault-free, which overstates the counterfactual (a test shows it). For this, `FaultHook.skip()` counts calls a replay served, so occurrence-numbered faults still hit the same call. The P5-T3 tests also caught that the decorator's default replay rebuilt a recorder-less traced call from the whole span output instead of its `result`. A patch is always the span's *recorded* output shape. CLI `agenttrace replay`; check `python tasks.py patch-check` | Claude |
| 2026-11-04 | Patcher (P5-T2) | `replay/patcher.py::plan_patch(run, workflow, Patch(target, output, patch_source))` returns a `ReplayPlan`: the target is patched, and every other span is frozen or re-run. Reached steps come from the workflow's **declared** wiring, not the recorded refs. A step skipped in the original because the target's step failed has no recorded inputs, so P3's `downstream()` misses it, yet it must run once the patch fixes that step (a test shows the gap). Which steps are reached depends on the target: a step reaches its downstream steps; a call inside a step reaches that step plus its downstream steps; the workflow input reaches its consumers. Every span of a reached step re-runs, except two kinds that stay frozen: calls in the target's own step opened *before* the target (they ran before its output existed), and spans nested inside the target (an agent's own model calls are what it was asked, not what the patch changes). So the plan is `downstream()` minus earlier sibling calls, plus formerly skipped steps. Decided at step level, not field level: a downstream step re-runs even if it reads only unchanged fields, and the replay cache serves it at no cost. Validation: unknown target, wrong output keys for a step or input (they must equal the declared keys), an empty call output, or a workflow/version mismatch raise `PatchError`. `patch_source` is carried for P5-T4. CLI `agenttrace plan` | Claude |
| 2026-11-04 | Record/replay cache (P5-T1) | `agenttrace/replay/`. `ReplayCache` implements a tracing `ReplayHook` passed as `Workflow.run(replay=)`. **Calls** (llm, tool, retrieval) are keyed *before* they run by the content hash of (node_id, kind, model, params, input); for model calls, model and params are the resolved model and merged params (seed included), declared via `@traced(replay_key=, rebuild=)`, and tools/searches reuse their `inject` converters. A hit reproduces the recorded span (cost 0) and returns the rebuilt value without calling anything. **Agents** are keyed after their body ran: own node id, model(s), step params, inputs, plus a fingerprint (key + output) of each nested call in order. So a recorded agent output is reused only if its calls answered exactly as recorded. That reproduces agent-level faults and patches, and keeps runs of different seeds apart in one cache (a plain node+inputs key would serve seed 2's Policy output to a seed-1 replay). **Strict** mode raises `ReplayMissError` on a *model-call* miss (tools, searches and agents are deterministic code and recompute; failed calls are never recorded). `HarnessError` is the new base of `InjectionError` and `ReplayMissError`, and the runtime re-raises it, so a miss is never recorded as an agent failure. One output per key, first recorded wins. Optional SQLite persistence. `Run.model_params` (new, default empty; older runs fall back to `seed`) lets a replay use the run's own seed. Check: `python tasks.py replay-check` | Claude |
| 2026-11-03 | Multi-run protocol (P4-T5) | `scenarios/multirun.py` (`python tasks.py multirun`). Run k of K (default 3) passes `seed=k` to **every** model call: `Workflow.run(model_params=)` → `StepContext.model_params` → `ask()` → gateway params. The seed is part of the cache key, so each seed is its own cached run, and adding a seed only pays for the new one; runs with no `model_params` keep their old keys (P4-T4's verified runs are untouched). Decoding stays as configured (temperature 0), so the protocol measures the *system as configured*: provider non-determinism under a seed, not sampling noise. `--temperature` overrides it for a sampling-variance study (separate run ids `-s{k}-t{T}` and report). Each seeded run is judged with the P4-T4 criteria against the clean run of the **same** seed. Stored as `{scenario}-{profile}-s{k}` (`Run.seed` set), controls included. Variance = sample variance / std of the per-run rates (behaves-as-labelled, final-correct) across the K runs, plus per code; scenarios whose verdict or correctness differs across seeds are listed as unstable. `verify --store` now also stores each control's run (P7 needs them for the false-positive rate) | Claude |
| 2026-11-03 | Scenario verification criteria (P4-T4) | `scenarios/verify.py` (`python tasks.py verify`): each scenario's ticket runs clean (once per ticket, cached) and with its fault plan, same models. Outcome = (completed, decision, required facts missing). Behaves: F0 = no fault and clean decision correct; F1–F7 = every fault applied, root output changed (via `align_runs`), outcome changed and wrong; F8 = applied, root changed, outcome identical. Misbehaviour is classified (not applied / root unchanged / no effect / still correct / silent fault changed outcome / fault error), to drive fixing or dropping scenarios. Report in `reports/scenario_verification.json`; `--store` saves faulty runs (id `{scenario}-{profile}`, replaced on re-run) for P7 | Claude |
| 2026-11-03 | Ground-truth isolation (P4-T3) | `scenarios/ground_truth.py` is the access module (`load_dataset`, `scenario`, `truth_for`). `scenarios/isolation.py` statically builds the package import graph (plain, relative, in-function, `importlib.import_module` / `__import__` literals; importing a submodule counts its parent packages) and enforces: (1) only the harness (`scenarios`, `faults`, `evaluation`, `workflow.clean_eval`, `cli`) imports truth (`schemas.ground_truth`, `scenarios.*`) directly; (2) nothing else *reaches* truth through any import chain (so diagnosis cannot use a harness module either); (3) non-harness code may not name `scenarios.json`. Protected packages (diagnosis, explainability, judges) must exist so the check cannot pass vacuously. No import-linter dependency. **Leak found and fixed:** the Policy Agent imported `Decision` from `schemas.ground_truth`; `Decision` now lives in `schemas.scenario` (public) | Claude |
| 2026-11-03 | Scenario dataset (P4-T2) | `data/scenarios/scenarios.json`, 111 scenarios (F0 18, F1 11, F2 11, F3 12, F4 12, F5 12, F6 12, F7 11, F8 12; refund 84 / shipping 15 / account 12), generated by `agenttrace/scenarios/generator.py` (seed 20261103; a test keeps the file current). Bases: the 18 clean tickets the default model resolves correctly (clean-05 and clean-18 excluded, since a fault there could not be told from the model's own miss). Every fault is hand-written against a decision-relevant fact of its base: F1 swaps the governing doc for a superseded/confusable one, F3 the field the decision rests on, F6 a claim contradicting the required fact, F8 a field nothing reads (email, card digits, unused outputs). Ground truth: roots = fault targets; affected = steps the corrupted value reaches, computed from the declared wiring per field (a wrong `router.order_id` skips the Retriever; unused outputs reach nothing); expected_final = the base's correct outcome. Ids are opaque (`sc-001`…, seeded shuffle) because diagnosis can see scenario ids. Tool faults target occurrence 1. Bases reuse dev-set tickets: the dataset measures *diagnosis*, not agent generalization | Claude |
| 2026-11-03 | Escalation precedence in the KB | KB-ESC-001 gains: when more than one rule applies, account and payment rules (fraud review -> Trust & Safety, chargeback -> Payments team) take precedence over refund amount limits. Prompted by Ultra and Spark both sending clean-06 to a supervisor. Results before this date used the old KB | User |
| 2026-11-03 | Muse Spark as an agent | Added role `agent_spark` (muse-spark-1.3, reasoning low, max_tokens 4096) and profile `spark` for model comparisons only; default agents stay NVIDIA so Meta credits stay on diagnosis. Don't judge `spark` runs with judge_fast (same model) | User |
| 2026-11-03 | Fault injection (P4-T1) | `faults/injectors.py`: pure JSON -> JSON injectors, each restricted to span kinds: retrieval (`retrieval_wrong_doc`, `retrieval_empty`, `retrieval_irrelevant`), tool (`tool_wrong_value`, `tool_malformed` error / missing_fields / wrong_type), agent (`set_field`, `hallucinated_claim`). `FAULT_CATALOG` maps F0–F8 to injectors; F7 (two roots) and F8 (silent) are scenario-level. `faults/plan.py::FaultPlan(scenario.faults)` is passed to `Workflow.run(faults=)`; it is consulted when a node produces its output ("at record time": `@traced(inject=(to_json, from_json))` for searches and tools, the runtime for agent outputs), so the faulty value is recorded and passed on as if the node produced it. Param `occurrence` (1-based) targets one call of a repeated node; default all. Nothing about the fault enters the trace; `plan.applied` / `unapplied()` are for the harness. Misconfiguration raises `InjectionError`, which the runtime re-raises (never recorded as an agent failure). A broken tool may now return raw JSON (`ToolResult = model \| dict`) | Claude |
| 2026-11-03 | Topology check counts steps that ran | User's run: Bolt showed `graph topology 18/20`. The 2 "mismatches" were its Tool Agent parse failures: policy and writer were skipped, so their input edges were rightly absent. `graph.executed_topology(g, declared)` keeps only declared edges into steps that ran; `clean_eval` now compares against it | Claude (from user's run) |
| 2026-11-03 | Blast radius = `downstream()` | Found by the P4-T1 tests: a wrong policy decision changed `writer.llm`, which is not a graph descendant of `policy` (nesting edges point child -> parent, but a step's inputs flow into its calls). `graph.downstream(g, span)` = descendants + calls nested in every affected step; diagnosis (P7) should use it for "affected nodes" | Claude |
| 2026-11-03 | UI graph JSON (P3-T4) | `graph/ui.py::to_ui_graph(g, run) -> UiGraph`, React Flow field names (nodes: id, type = span kind, position, width/height, data; edges: id, source, target, type, label, animated, style, data). `display_status` priority root_cause > propagated > suspicious > error > skipped > ok (builder gained a `role` placeholder for diagnosis); colour-blind-safe palette sent as `legend` so the UI never hard-codes it. Layout hints: steps in columns by causal depth (longest data path), nested calls stacked below their step; `layer` / `row` in data. JSON Schema generated from the Pydantic models, committed at `ui/schema/run-graph.schema.json` (regenerate: `uv run python -m agenttrace.graph.ui`; a test keeps it current). Dev dependency `jsonschema` (+ `types-jsonschema`) validates exports in tests | Claude |
| 2026-11-03 | Run alignment (P3-T3) | `graph/alignment.py::align_runs(a, b)`: spans matched by logical node + occurrence (= local id, so run ids never matter); each pair compared on status, input, output (cache-only fields ignored), model, params, and `patched`; `changes` names which. Pair status unchanged / changed / only_a / only_b; `node_status()` rolls occurrences up per node (an extra or missing call makes the node "changed"). Pairs follow A's causal order, B-only spans last. Runs of different workflow versions are refused. CLI `agenttrace diff RUN_A RUN_B [--all]` | Claude |
| 2026-11-03 | Graph queries (P3-T2) | `graph/queries.py`, all deterministic (ties by recorded `seq`, then id): `ancestors`, `descendants`, `topological_order` (lexicographic Kahn), `final_spans` (builder records the producers of `final_output`; fallback = sinks), `paths_to_final` (all simple paths, shortest first), `critical_path` (heaviest chain by duration / cost / tokens; by default over steps along data edges: a step's time already includes its nested calls, but tokens/cost live only on LLM spans, so for those each step counts its nested calls' total. Caught by the user's run: without the rollup the token "path" was a lone `policy.llm` span), `first_divergence(a, b)` (match spans by local id, walk a's causal order, first span whose status or output hash differs, so its own inputs did not differ). Output hashes ignore cache-only fields (`cache_hit`, `original_cost_usd`) so a cached rerun is not a divergence. CLI `agenttrace graph RUN [--from SPAN] [--diff RUN_B]` | Claude |
| 2026-11-03 | Trace integrity (P2-T5) | `tracing/integrity.py::check_integrity(run)` is workflow-independent (works on foreign imports). Codes: `orphan_parent`, `orphan_ref`, `ref_key_missing`, `ref_hash_mismatch`, `ref_from_future` (consumed before the producer ended), `child_outside_parent`, `parent_cycle`, `ref_cycle`, `empty`; every issue is listed at once. Enforced on `TraceStore.save_run` (nothing written), `TraceStore.get_run`, and `import_runs`. Schema-level breakage (duplicate ids, end < start, self-parent, foreign span, unknown field) is already rejected by Pydantic. Fixtures in `tests/fixtures/traces/` are generated by `make_traces.py`; a test keeps the committed files identical to the generator | Claude |
| 2026-11-02 | Final-outcome check (`final_correct`) | `agenttrace/evaluation/outcome.py`: correct = expected decision AND every `required_facts` string in the reply. No LLM. Matching normalizes NFKC, all dash variants, curly quotes, `&`/"and", case, and whitespace; facts must stand alone ("79" matches "$79.00", not "$179" / "79.50" / "O1079"); ordinals ("31st") and a plural "s" are allowed. Synonyms are **not** ("30-day" does not match "30 days"): a fact is an exact KB phrase. Amounts and `escalate_to` are not graded separately; the escalation team is a required fact where the KB names one | Claude |
| 2026-11-01 | Runtime design | Sync, sequential (gateway is sync). Each step binds every input explicitly to `FromInput(key)` or `FromStep(step, key)`; the graph is checked at construction (input/output keys, unknown steps, cycles). Each step record carries `input_refs` (producer step id + content hash), so P2-T2 provenance comes from declared wiring, not execution order. Step ids are deterministic (`{run_id}/{step}`) so replays line up. Values are JSON-copied across steps (no shared mutation). An agent exception becomes a step `error`; only its downstream steps are skipped | Claude |
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

### 2026-11-01 · P1-T5 · NVIDIA: most listed models are not callable on this key
- **What happened:** `GET /v1/models` lists 85 models (free call). One test prompt each (NVIDIA free credits, $0 Meta):
  - **Work:** `nemotron-3.5-lightning-30b-a3b` (~5s), `nemotron-3-ultra-550b-a55b` (~7s, one 503 "overloaded"), `nemotron-3.5-bolt-e4b` (~2s), `mistral-large-4-1t-a52b` (~30s).
  - **404 "Function not found for account":** mistral-large-2, nemotron-3-super-120b, kimi-k2.6, llama-3.1-nemotron-70b/ultra-253b, nemotron-nano-3, mistral-nemo-12b, mistral-7b, gemma-3-12b, phi-3.5-moe, nemotron-4-340b.
  - **Time out (even at 200s):** gpt-oss-20b, gemma-4-31b-it, deepseek-v4.1-flash.
- **Why it matters:** "listed" ≠ "usable". Any future model choice needs a test call first.
- **Action:** `agent_default` = lightning, `agent_alt_1` = mistral-large-4, `agent_alt_2` = bolt, `judge_nvidia` = ultra. JSON mode works (`supports_json_mode: true`). Free tier priced at $0. `judge_oss` stays TODO_VERIFY (no OpenRouter key).

### 2026-11-01 · P1-T5 · Nemotron thinking: off is faster and more reliable here
- **What happened:** Nemotron reasons by default. `chat_template_kwargs: {enable_thinking: false}` cut a policy answer from ~507 to 43 output tokens (5.5s → 1.3s), still correct. `reasoning_effort: low` had no effect. With thinking on and `max_tokens: 4096`, some agent calls spent all 4096 tokens reasoning and returned nothing.
- **Action:** agents run with thinking off. The judge keeps thinking on (revisit in P6-T1).

### 2026-11-02 · P1-T6 · Mistral Large 4 always reasons, and is slow on the free tier
- **What happened:** `enable_thinking: false` still reasoned (34s, 195 tokens); default took 62s; `reasoning_effort: none` → HTTP 400. At `max_tokens: 1024` a call spent 1023 tokens reasoning and returned nothing; two calls hit the 120s read timeout.
- **Action:** `agent_alt_1` (Mistral) gets `max_tokens: 4096`; NVIDIA `timeout_s` raised to 240. The `mistral` profile stays (a different model family matters for credibility) but a full 20-ticket run is ~2 hours, so it's for spot checks. Added `agent_alt_3` = Nemotron Ultra (thinking off, ~2s) as the third fast model. Ultra is also `judge_nvidia`, so `ultra` runs must not be judged by `judge_nvidia`.

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

### 2026-11-04 · P4-T5 · NVIDIA free-tier rate limit broke a run's results without failing it
During `python tasks.py multirun` (user's run), seeds 1 and 2 finished cleanly. Partway through seed 3 NVIDIA returned bursts of HTTP 429. The provider retried 4 times (about 1+2+4+8 s), which is shorter than a per-minute window. The 5th attempt raised, and the runtime recorded the error as an *agent failure*. 23 stored seed-3 runs had `ProviderRequestError: nvidia: HTTP 429` in a span. Seed 3's clean runs were among them, so every later seed-3 verdict was judged against a broken baseline. Nothing in the output showed this: the runs "completed" with step errors.
- **Fix 1, provider:** 429 has its own retry budget (`rate_limit_retries` 8, waits doubling from 2 s to `rate_limit_max_wait_s` 60 s, about 4 min in all, `Retry-After` honoured). Other errors keep `max_retries`.
- **Fix 2, harness:** a run containing a provider error (`ProviderRequestError`, `ProviderUnavailableError`, `BudgetExceededError`) says nothing about the scenario. `verify` and `multirun` now mark it "provider error", leave it out of every rate, never cache it as a clean baseline, never store it, and exit `INCOMPLETE`. A rerun redoes only those calls, because failed calls are never cached. A full rerun replaces or prunes the 23 bad stored runs.
- **Lesson for later phases:** infrastructure failures must never enter the evaluation as model behaviour. The same guard belongs in the P7 and P9 runners.

### 2026-11-04 · P6-T4 · A rejected OpenRouter key crashed the judge sample
`python tasks.py judge-sample --roles judge_nvidia judge_oss` (user's run): OpenRouter answered `HTTP 401 {"message": "User not found."}`, meaning it rejected the key. The `ProviderRequestError` propagated and ended the whole command. Nemotron's 20 judgements had already finished (cached), but no report was written.
- **Fix:** `judge_span` now returns provider errors as a result (`provider_error`, and `auth_error` for 401 / 403 or a missing key) instead of raising. `judge-sample` reports such a judge as skipped and carries on; Stage A disables the judge for the rest of the run (no repeated failing calls); `ensemble` stops with a clear message, so an ensemble is never silently computed without one of its judges.
- **The key itself is for the user to fix** (it must be a live OpenRouter key, `sk-or-v1-...`, in `.env` as `OPENROUTER_API_KEY`).

## 3. Parse Failures and Structured-Output Issues

### 2026-11-04 · P6-T1 · Nemotron Ultra (thinking on) can spend its whole budget reasoning
In `python tasks.py judge-sample` (user's run), one of 20 judge_nvidia calls hit `max_tokens=4096` with 4,096 reasoning tokens and no visible output. The gateway's repair loop retried it and the second reply was valid (19/20 valid on the first try, 20/20 overall). The model also reasons at length on the rest: 46k output tokens for 20 judgements, against Spark's 21k, and about 56 s per span against Spark's 12.5 s.
- **For P6-T3 (every node of every run):** raise `judge_nvidia` max_tokens (e.g. 8192), or turn thinking off for it as the agent roles do, and measure which keeps validity at 20/20. Truncation wastes a full call each time.
- **Cost planning:** Spark costs about $0.0077 per judged span. Every judged span of all 312 seeded runs (about 21 per run) would be roughly $50 of the $150 Meta budget, so P6-T3/P7 should judge a planned subset or use the cheap tier first.
- **A partial re-run (`--roles judge_nvidia`) overwrote the full report** in the first version; it now keeps the other judges' results when the sample matches.

<!-- parse_failure counts by model/prompt version, common failure shapes, prompt fixes -->

### 2026-11-01 · P1-T5 · Extra keys broke strict parsing, and repairs repeated the mistake
- **What happened:** the Writer's reply had an unrequested `"reply_length_check": 84` key. Our schemas forbid extra keys, so it failed; at temperature 0 the model sent the identical JSON on both repair attempts, so the step failed.
- **Action:** LLM reply schemas (`LLMReply`) now drop extra keys. The agents' own outputs stay strict. `AgentFailure` messages now include the last validation error.

### 2026-11-01 · P1-T5 · Text details for the P1-T7 checker
- Replies write "5‑7" with a non-breaking hyphen (U+2011), so a plain `"5-7 business days"` match would miss it. The checker must normalize dashes (and whitespace, per `normalize_ws`).
- One Writer run said "5-10 business days" (the policy says 5-7): a natural hallucination. The checker's required facts will catch it.
- The Policy Agent sometimes gives the right decision with the wrong amount (it missed KB-REF-009's $6.95 return fee on a $48 order). Decision accuracy doesn't grade amounts yet.
- **Handled in P1-T7:** all three points above are covered by `check_outcome` and its tests (U+2011 normalization, "5-10" failing "5-7 business days"). Amounts are still graded only where a ticket lists the amount as a required fact.

### 2026-11-02 · P1-T7 · Right decision, wrong reply: what the outcome check catches on the saved runs
- Lightning's replies often say refunds take "5-10 business days"; KB-REF-007 says 5-7 (a natural hallucination, in the Writer).
- Ultra's clean-06 escalated a fraud-review account to a "Support Supervisor team"; KB-ACC-003 says Trust & Safety. Decision right, routing wrong.
- Lightning's clean-19 approved correctly but never said why (the holiday window to January 31).
- Bolt often says the refund time "is not specified in the provided documents", a sign the timing doc wasn't retrieved.
- These are decision-correct but outcome-wrong runs: exactly the "silent" errors the debugger must localize later.
- Measured (cached, all 20 tickets): decision vs. final is Lightning 19 vs. 17, Ultra 20 vs. 18, Bolt 16 vs. 12. clean-01 fails final on all three models (Lightning and Ultra say "5-10 business days"; Bolt says the timing "is not specified"), so the refund-timing claim is the most common right-decision error.

### 2026-11-03 · P4-T1 · Spark ties Ultra; both strongest models miss the same two tickets
- **Measured** (user's run, `python tasks.py clean --profile default ultra bolt spark -q`): Spark 20/20 decision, 18/20 final, same as Ultra; Lightning 19/17, Bolt 16/12. Spark is slow live (934 s for 20 tickets: it reasons before every answer, ~500 output tokens per call), instant from cache afterwards.
- **clean-01:** Ultra states a wrong fact ("5-10 business days"; KB-REF-007 says 5-7); Spark omits the timing altogether. Same miss, different failure mode (hallucination vs. omission).
- **clean-06:** both Ultra and Spark escalate to "support supervisor"; the label says Trust & Safety. The ticket triggers two rules: fraud review -> Trust & Safety (KB-ACC-003) and refund $219 > $200 -> supervisor (KB-REF-005). The KB never states which takes precedence, so this may be a **KB gap**, not only a model error. **Resolved (user's call, 2026-11-03):** KB-ESC-001 now states the precedence ("account and payment rules take precedence over refund amount limits": fraud review -> Trust & Safety, chargeback -> Payments team, even over $200), with fact `escalation_precedence`. The clean-06 label is unchanged; it is now backed by an explicit rule. All earlier clean-ticket numbers were measured on the KB *before* this change.
- **After the change** (user's runs, same commands): Spark 20/20, final **19/20** (clean-06 fixed); Ultra 20/20, final 18/20 (clean-06 fixed; clean-10 now misses a fact); Lightning **18/20**, final 16/20; Bolt 14/20, final 10/20 (3 parse failures).
- **Side effect (Lightning, clean-05):** it now approves the Final Sale rain jacket, writing "not final sale" although `order_lookup` returned `final_sale: true`. KB-REF-003 (the Final Sale policy) is no longer retrieved: the model's queries came from cache (identical), but the longer KB-ESC-001 shifts BM25 scores and KB-REF-003 fell out of the 6-doc cap. A one-sentence KB edit fixed one ticket and silently broke retrieval for another: a real cross-effect of the kind the debugger must localize.
- **Retrieval benchmark unchanged** after the edit (user's run, `python tasks.py retrieval`): dev recall@3 0.979, held-out 0.800, same three misses. The benchmark scores single queries; the workflow merges several searches under a 6-doc cap, and KB-REF-003 was the marginal doc for clean-05. A cap-level effect invisible to per-query recall.
- **Decision:** keep the cap and wording (Lightning still meets the 90% target). Keep clean-05 as a *natural* upstream fault: the wrong decision surfaces at `policy`, the cause is `retriever` (missing doc), so it is a real test for root-cause localization in P7.

### 2026-11-03 · P3-T3 · Same decision, different text: two models differ in almost every span
- **Measured** (`uv run agenttrace diff clean-07-default-r1 clean-07-bolt-r1`, user's run): Lightning and Bolt both escalate clean-07 correctly, yet 16 of 21 spans changed. Unchanged: `__input__`, the two fixed-query searches (`retriever.search#4`, `#5`: ticket text and the standing escalation query), and `order_lookup` / `account_status` (same arguments, same facts).
- `router` changed only in model/params: same category and order id, differently worded rationale. So the first divergence (`router.llm`) is in explanation text, not in routing.
- `refund_api` changed only its input (different arguments, e.g. the reason text) with the same quote returned.
- **Implication:** "the text changed" is not "the result changed". Replay recovery (P5) and regression comparison must judge outcomes with the P1-T7 checker (and step outputs by their decision-relevant fields), not by byte equality; raw divergence is for locating where runs split.

---

## 4. Replay and Determinism Issues
<!-- Flaky replays, cache misses, live-vs-cached drift, stability tolerances (P5-T6) -->

### 2026-11-04 · P5-T6 · Replay stability tolerance (fixed before the first stability run)
`python tasks.py replay-stability` repeats each replay `--reps` times (default 3), with the patch computed once and held fixed. It runs in two modes:
- **cached (the normal gateway): IDENTICAL is required.** Every span must have the same status, input, output, model, params and patched flag (`align_runs`), and the final output must be equal. Cache bookkeeping in a model call's output (`cache_hit`, `original_cost_usd`) is ignored, because repetition 1 may make the call that the later ones are served from. (A first draft compared those fields too and flagged every replay; caught by the tests.)
- **live (each repetition on its own empty completion cache, so every re-run model call goes to the provider): the decision and the `recovered` verdict must be identical in every repetition.** Reply wording may differ: the share of replays with identical reply text, and the largest `recovery_score` spread, are reported but are not pass criteria.

Why this tolerance: P4-T5 showed the provider is not bit-deterministic across *seeds* even at temperature 0. Whether it is deterministic for the *same* seed repeated is what the live mode measures, so wording is reported rather than required. Every metric built on replays (`recovered`, root-cause verification in P7) depends only on the decision and the checker's verdict, so those are what must hold.

### 2026-11-04 · P5-T6 · Cached replays are exactly stable; live replays keep the decision but not always the verdict
`python tasks.py replay-stability` (user's run): 24 verified runs (3 per code), 27 oracle-patched replays, 3 repetitions per mode, report `reports/replay_stability.json`.
- **cached: 27/27 within tolerance**: identical in every span and the final output, recovery-score spread 0.0.
- **live: 23/27 within tolerance, FAIL against the tolerance fixed in advance** (kept as declared, not loosened after the fact). Decision identical in 27/27, `recovered` verdict identical in 23/27, reply text identical in only **2/27**, largest score spread 0.5.
- **The 4 failures share one shape:** the decision is stable, but the Writer's reply states the single required fact in some repetitions and not others (sc-003 on T-0101, "5-7 business days"; sc-011, sc-020 and sc-032 on T-0117, "14 days"). A spread of 0.5 is one fact flipping.
- **The provider is not deterministic even for the same seed at temperature 0:** the wording changed in 25/27 live replays. (P4-T5 showed the same across different seeds.)

What follows:
1. **Evaluation replays run cached** (the record plus the gateway cache), as `patch-check`, `replay-check` and the replay CLI already do. Cached replays are exactly reproducible, so P7/P9 recovery numbers are deterministic given the cache.
2. **Reproducing the numbers on another machine needs the cache** (`.cache/completions.sqlite`) or the stored runs. Re-running cold re-asks the provider and the wording will differ. The cache is part of the reproducibility artifact.
3. **A live recovery verdict is a sample, not a fact:** about 15% of live replays (4/27) flip `recovered` by wording alone. Any live-mode evaluation must repeat K times and report recovered as a rate (P4-T5 protocol), never a single live verdict.
4. **The flips sit on facts the Writer adds from its own wording** ("5-7 business days" is not in T-0101's retrieved docs; P4-T5 sc-004). Such required facts are the noisiest part of the outcome check.

---

## 5. Judge Behavior and Disagreements
<!-- Where judges disagree, systematic biases, κ values, cases that needed Wonder escalation -->

---

### 2026-11-04 · P6-T4 · Spark and Nemotron agree only fairly (κ 0.32), and most of their disagreement is about searches
`python tasks.py ensemble --judges judge_fast judge_nvidia` (user's run; judge_oss left out because OpenRouter rejects the key, 401 "User not found" even on a direct call):
- **Pooled κ 0.321 ("fair")**, with large per-run swings (−0.13 to +0.68) because each run has only 7–13 judged spans. Pooled agreement is the number to quote.
- **Spark flags far more than Nemotron:** 75 of 205 spans not ok, against 31. Nemotron (thinking off) is the lenient judge. Which one is right is a Phase 9 question (accuracy against ground truth); agreement alone can't say.
- **35 of the 60 splits are on `retriever.search` spans.** Whether a BM25 hit list is "good enough" is a soft call the two judges make differently. The deterministic validators already check searches exactly (index membership, score reproduction, relevance, emptiness; P6-T2), so for P6-T5 search spans should lean on validators and be down-weighted in judge agreement, rather than letting judge noise there drive root-cause candidates.
- **With two judges the weighted vote is degenerate:** every disagreement is a tie, and ties go to the more severe verdict, so the ensemble is effectively "flag if either judge flags" (faulty 29 is more than either judge alone). That is by design (a split never hides a fault), but it makes the ensemble as noisy as the noisier judge. A third judge (judge_oss) would turn ties into majorities; re-run with all three once the OpenRouter key works (the two judges' calls are cached).

## 6. Scenario and Fault-Injection Issues
<!-- Ineffective faults, mislabeled scenarios, dropped scenarios and why (P4-T4) -->

### 2026-11-01 · P1-T5 · `ScenarioInput.category` would leak the Router's answer
- **What happened:** `ScenarioInput` (the "workflow may see this" view) carries the true ticket category, which is exactly what the Router must predict.
- **Action:** the workflow's only input is `ticket`; `category` is used by the eval harness only. A test checks no prompt contains ground-truth fields.

### 2026-11-01 · P1-T5 · How the 20 clean tickets got to 95%, and the one natural miss
Each run is `python tasks.py clean` (agent_default, thinking off):

| Change | Score | Misses |
|---|---|---|
| First run | 17/20 (85%) | 03, 15: escalation policy not retrieved; 18: date reasoning |
| Standing escalation query | 17/20 (85%) | standing docs cut off by the doc cap (my bug); 05: final-sale doc crowded out |
| Standing docs added after the cap (user's run) | 18/20 (90%) | 15: prompt told the agent to ignore ticket text, so it ignored the legal threat (my bug); 18 |
| Prompt fix: escalation rules include what the customer says | **19/20 (95%)** | 18 only |

- **clean-18 is left as a natural error:** the model claims a Jan 2 delivery is "between Nov 15 and Dec 31". The right policy and data were provided. No ticket-specific prompt rule was added for it.
- **Caveat:** prompts were adjusted while looking at these 20 tickets, so they're a dev set; the honest held-out number comes from the Phase 4 scenarios.

### 2026-11-02 · P1-T6 · Same workflow, three models (config only): where they fail differs
`python tasks.py clean --profile default ultra bolt -q` (user's run):

| Profile | Model | Accuracy | Completed | Contract violations |
|---|---|---|---|---|
| default | Nemotron 3.5 Lightning (30B MoE) | 19/20 (95%) | 20/20 | 1/20 |
| ultra | Nemotron 3 Ultra (550B MoE), thinking off | 20/20 (100%) | 20/20 | 0/20 |
| bolt | Nemotron 3.5 Bolt (small) | 16/20 (80%) | 18/20 | 8/20 |

- **Different failure types per model.** Lightning's one miss (clean-18) is a reasoning slip. Bolt's misses are mostly *format* failures: 6 runs cited mangled doc IDs (`": "`, `".KB-REF-002"`, a doc title); 2 Tool Agent runs never produced valid JSON (one truncated mid-string, one missing fields) after 2 repairs, so no decision. Bolt also shares the clean-18 slip and gets clean-10 wrong.
- **Why it matters for later phases:** these are *natural* faults (not injected), useful for checking that diagnosis finds the right node on real failures. The contract checks caught every Bolt citation error.
- Bolt's settings were left unchanged (no tuning per model), so the comparison is like-for-like.

### 2026-11-03 · P4-T4 · First verification: 77/111 behave; 34 faults landed but did not matter
`python tasks.py verify --store` (profile default, user's run): F0 18/18, F1 6/11, F2 8/11, F3 9/12, F4 4/12, F5 10/12, F6 2/12, F7 8/11, F8 12/12. All 34 misses had the same reason: **no effect on outcome**. The fault was applied and the root's output changed, and Policy and Writer changed too, but the decision and required facts did not.
- **Root cause: the workflow is redundant.** A fault only matters if it hits the *only* source of a fact. In the faulty runs, the model kept finding a second source:
  - **the ticket text:** "I disputed the charge"; "gift card"; the order id in the body when the Router's was wrong;
  - **the refund_api quote:** it is computed from the database, not from order_lookup's output, so `chargeback_open` gave away a hidden chargeback;
  - **sibling fields:** `delivered_at` next to a corrupted `days_since_delivery` or `delivered_late`. The model counted the days itself;
  - **a second KB document restating the rule:** KB-ACC-004 and KB-REF-011 both state the Plus window.
- **Superseded documents label themselves** ("Superseded by KB-REF-001…"). Swapping one in as a "stale" document had no effect, because the model rightly ignored it. That is good model behaviour, but it's a weak F1.
- **Appended hallucinations lose to the agent's own reasoning.** The claim came after a justification that already stated the true fact, or contradicted `escalate_to` / `refund_amount`, and the Writer followed those. Both F6 that worked added a fact the justification did not already state. In one case (sc-012) the claim partly reached the reply ("a supervisor will be in touch"), but the required fact survived.
- **One F7 cancelled itself:** with retrieval emptied, the 7-day damage rule was gone, so the wrong day count no longer mattered.
- **Action:** all 34 specs are moved to `DROPPED` in [generator.py](agenttrace/scenarios/generator.py), each with its measured reason. Their replacements:
  - aim at single-source facts;
  - use the new `values` mode of `tool_wrong_value`, which changes a record consistently (O1013 rewritten as on time across all of its date fields);
  - use `mode="replace"` claims, where the whole justification is fabricated, with the same decision and a wrong fact.

  The dataset is now 108 scenarios (≥11 per fault code). It needs a second verification run.

### 2026-11-03 · P4-T4 · Second verification: 96/108 behave; one more second source, and a checker gap
Same command (user's run): F0 18/18, F1 10/11, F2 10/11, F3 10/11, F4 6/11, F5 12/12, F6 8/11, F7 10/11, F8 12/12. The 12 misses, again all "no effect on outcome":
- **The Policy Agent falls back on the customer's own claims (5 of the 12).** Its prompt says to take facts from the tools, not from the customer. With the account tool broken, it still accepted "I'm a Plus member" from the ticket. With the delivery made on time, it still accepted "arrived three days late". So the ticket text is a second source for anything the customer asserts. Natural workflow behaviour, kept as is (no prompt change).
- **The model knew or loosely read the rule.**
  - With no documents, it still applied the 30-day window.
  - It read KB-SHP-008 ("may cancel if not shipped within 5 business days") as allowing the cancellation.
  - It inferred "not shipped" from an empty `delivered_at`.
  - It approved a fee refund for an on-time delivery with no policy for it.
- **The Writer used its own inputs over a fabricated justification.** It took the 30-minute expiry from KB-ACC-001 in its documents, and $14.99 from `refund_amount`.
- **Outcome-checker gap (sc-102):** the fabricated 75% rule reached the reply ("your refund is $59.25"). But the reply also mentions "the $79.00 purchase price", so the required fact "79" still matched and the run counts as correct. A wrong answer passes when the required fact appears in another role. This matters for P5-T5 recovery scoring. A fix would be a per-ticket "forbidden facts" list, not done here.
- **Action:** the 12 specs are moved to `DROPPED` with these reasons. 8 replacements follow the patterns that worked:
  - account facts the customer never states (refund count, join date);
  - the delivery date for a ticket that gives none;
  - fabricated justifications on facts the Writer has no other source for.

  The dataset is now 104 scenarios (≥10 per fault code).

### 2026-11-03 · P4-T4 · Third verification: 103/104 behave (PASS); the one miss is kept and documented
Same command (user's run): every code at 100% except F4 at 10/11. Overall 99% against the 95% target. **8 of the 8 replacements behaved.**
- **sc-042 (F4, clean-17, `account_status` without `plus_since`), no effect on outcome.** The Policy Agent noted that the join date was missing and that "it cannot be verified that the 14-day window applies". It denied anyway, and the reply still states the 14-day rule. That is a *correct* outcome reached by an assumption, not from evidence.
- **Kept, not removed.** The done-when allows a documented remainder. Removing it would reshuffle every scenario id, which would invalidate the verified runs. P4-T5's K=3 runs will show whether this miss is stable or just one sample's luck, and the decision on it waits for that. Until then, evaluations that need "the fault changed the outcome" should read `reports/scenario_verification.json` (row `behaves`).
- **Update (P4-T5):** under seeds 1–3, sc-042 behaved 2/3 times. It is flaky, not a consistent miss, so it stays.
- **Across the three passes:** 46 specs were dropped (all documented in `DROPPED` with their measured reasons), and 104 remain.

---

### 2026-11-04 · P4-T5 · Three seeded runs: 83% behave as labelled, not 99%; one run overstates the dataset
`python tasks.py multirun` (K=3, seeds 1–3, temperature as configured = 0; user's run, complete, 312 runs stored):
- **Behaves as labelled:** 80.8% / 85.6% / 82.7%, mean **83.0%, std 2.4 pp**.
- **Final correct:** 29.8% / 36.5% / 33.7%, mean 33.3%, std 3.4 pp. If every fault behaved, it would be 30/104 = 28.8% (F0 + F8).
- **By code (mean, std):** F0 94.4% (5.6), F1 96.7% (5.8), F2 90.0% (0.0), F3 70.0% (10.0), F4 75.8% (10.5), F5 83.3% (8.3), F6 69.7% (13.9), F7 73.3% (5.8), F8 86.1% (9.6).
- **38/104 scenarios are unstable** (verdict or correctness differs between seeds). 4 never behave (0/3): sc-008, sc-085, sc-092, sc-095. All four behaved in the unseeded P4-T4 run.

What this shows:
- **A seed changes the output even at temperature 0.** NIM's decoding isn't deterministic across seeds. So a single run is one sample, including P4-T4's.
- **P4-T4's 99% is optimistic, by selection.** Specs were kept or dropped on whether they behaved in that one unseeded sample, which favours specs that happened to behave in it. On fresh samples the rate falls to 83%, a regression to the mean. The P4-T4 number stands as what its done-when measured, but **83% ± 2.4 pp is the honest rate**.
- **Controls vary too.** The clean decision is wrong on some seeds for T-0115 (legal threat), T-0110 and T-0103. So "the fault caused the miss" can't be assumed per run; it has to be checked per run.
- **Fault types that rest on one fact (F1, F2) are the stablest.** Types that rely on the Writer's wording or the Policy Agent's reading (F3, F4, F6, F7) vary most.

Decision: **the dataset is not tuned further.** Dropping specs again on these three runs would repeat the same selection bias. Instead, later phases score per run:
- the multirun report records, for every `(scenario, seed)` run, whether the fault behaved as labelled;
- diagnosis metrics (P7, P9) should be reported on all runs **and** on the runs where the fault behaved, so a fault that didn't matter in a run isn't counted as a missed root cause;
- variance comes from these K=3 runs.

### 2026-11-04 · P5-T3 · Oracle patches: replays are surgical; recovery is bounded by the clean run and by F7's second root
`python tasks.py patch-check` (user's run). All 230 replays changed only the patched node and its descendants. The patch replaces the known-bad node with what it produced in the clean run of the same seed. Two caveats about recovery, for P5-T5:
- **A single-root oracle patch can only be as good as the clean run.** With the root's clean output in place, everything downstream sees the clean run's inputs. So the final outcome should be the clean run's, including when that seed's clean run was itself wrong (P4-T5: clean runs miss on some seeds, e.g. sc-004 seed 1 says "5-10 business days"). The 3–5 non-recovered cases per single-root code are probably those. P5-T5 should compare recovery against the same seed's clean outcome, not against "correct", to separate "the patch failed" from "the workflow fails here anyway".
- **F7: patching one of two roots recovers 19/44 (43%).** The other fault stays in force, by design. Without it the second root disappears and recovery would be overstated. Full recovery on F7 needs both roots patched. P5-T2's plans have one target, so that is a case for multi-patch plans (or sequential replays) in P5-T5 / P7.
- **The free tier rate-limited the run** (429 waits up to 60 s), and the longer budget from P4-T5 absorbed every one.

### 2026-11-04 · P5-T4 · Each patch source has its own reach: rules match the oracle on tool and retrieval faults; only a model can patch a judgement
`python tasks.py patch-check --source oracle | rule | model` (user's runs). Every replay from every source changed only the target and its descendants, and was tagged.
- **rule = oracle on F1–F4** (26/29, 25/27, 18/21, 24/25 for both). Re-calling the real tool or re-running the real search reproduces the clean value exactly, so for tool and retrieval faults a fix with no ground truth reaches the upper bound. The remaining misses are runs whose clean run is itself wrong at that seed (P5-T3 finding).
- **rule cannot touch policy:** 66 n/a (every F5/F6 root, plus the policy roots in F7/F8), because a decision or justification is a model's judgement and there's no deterministic recompute.
- **model covers exactly that gap.** The patcher (Muse Spark) repaired 9/11 F5 and 8/9 F6 on the first 20 verified F5/F6 runs, with 0 malformed replies. That's a subset, so not directly comparable to the oracle's 27/30 and 23/23 over all runs; P5-T5 / P9 should run the model source on the full set for a like-for-like recovery rate. Cost: 20 Spark calls for the patches, plus 20 Writer re-runs on NVIDIA.
- **Implication for P7:** a fix policy can use the rule source first for tool and retrieval nodes (free, deterministic, oracle-level) and the model source for agent judgements, with the oracle kept as the upper bound in evaluation only.

### 2026-11-04 · P5-T5 · Scored against the clean run, single-root rule patches recover everything that was recoverable
`python tasks.py patch-check --source rule --store` (user's run), with recovery now scored against the same seed's clean run:
- **F1–F4: recovered = recoverable (26/26, 25/25, 18/18, 24/24), and every replay matches the clean outcome.** This confirms the P5-T3 hypothesis. Every "not recovered" single-root replay was a run whose clean run is itself wrong at that seed. Example: sc-003 seed 1 went from deny to approve after the rule patch, but the reply says "5-10 business days", exactly as that seed's clean run does. So the score rose 0.00 -> 0.50 without counting as recovered. Without the clean comparison these looked like patch failures.
- **F7: 13/34 recoverable recovered, as expected.** One root is fixed while the other stays in force. Multi-root recovery needs both patched (multi-patch plans).
- **F8: 23/25 match the clean outcome, and 2 recovered although the clean run was wrong.** Likely cause: F8 includes faults on the retriever's `queries` output. The retriever rule re-runs the searches from those (faulty) recorded queries, so its docs differ from both the original and the clean run. A rule replaying a step from its own recorded outputs inherits any fault in them. Noted for P7: the retriever rule should take queries from the step's model call, not its output.
- **Reporting rule from now on: quote recovery over recoverable replays**, and separately how often a replay reaches the clean outcome.

### 2026-11-04 · P6-T2 · Validators settle tool and retrieval faults for free; judgement faults are left to the judges
`python tasks.py validator-check` (user's run, 312 seeded runs, no model calls). PASS: every F4 fault was caught by an error (33/33), and every F2 fault had a finding on its root (30/30: 9 errors from irrelevant or score-mismatched hits, 21 empty-search warnings).
- **Free hard evidence well beyond the done-when:** F1 30/30 (a swapped hit's score doesn't reproduce from the index), F7 24/30 errors, F3 21/30. The 9 F3 misses are wrong values that are *consistent* with the rest of the record: a refund count of 1, a recent Plus join date, and the on-time delivery rewritten across all its date fields. No internal check can see those; that's the judges' (or a rule re-call's) job.
- **F8 6/36 is early detection:** the silent F8 that appends `KB-GEN-001` to Policy's cited docs is flagged (`cites_retrieved_docs`) although the outcome never changes.
- **F5 4/36 and F6 0 errors are expected.** A wrong decision or a fabricated justification is a judgement; only the router's wrong order id (`order_id_in_ticket`) and numeric hallucinations (F6, 22/33 warnings) leave a deterministic trace.
- **Controls: the 1 "false alarm" is a real model error.** In sc-048 seed 3, Policy cited the tool name `account_status` as a policy document (as Bolt did in P1-T6).
- **28/54 controls carry warnings, almost all the Writer inventing response times not in its inputs:** "5-7 business days" ×18 (three dash styles), "5-10" ×4, "2 business days" ×2, "1-2" / "2-3" / "3-5" business days. That is the same behaviour behind the P5-T6 live-replay flips. Policy's computed "$12.90" (a 10% fee) is the one arguable false positive: derived arithmetic isn't recognised. These stay warnings, so they don't short-circuit a judge; whether to make derived amounts count as supported is for P6-T3 once judges weigh in.

### 2026-11-04 · P6-T5 · First attribution pass: the right roots are found, but weak upstream flags outranked proven faults
`python tasks.py attribution-check` (user's run, 18 dev-set runs, stored Spark + Nemotron ensemble scores, no model calls). Default mode (searches flagged only by validators): top-1 root = true root node **7/16**, = its step 9/16, a true root among the ranked roots **15/16**, controls with no root **0/2**, propagation paths valid **18/18**. With `--judge-searches`: top-1 5/16 (low-confidence search flags everywhere), which confirms leaving searches to the validators.

Hand-check of the 10 printed runs: correct sc-003, sc-005, sc-006 (1 of 2 roots), sc-010, sc-013; step-level sc-015; wrong sc-004 (F0, a 0.35 writer flag), sc-011, sc-012, sc-014 (F8: nothing visible to trace). Two of the misses share a cause, the spec's ranking order (earliest first):
- **sc-012:** a judge's *suspicious* retriever (0.45) outranked the *validator-proven* account_status fault (1.00).
- **sc-011:** a weak order_lookup flag (0.20) outranked the Policy fault (0.47).

The same fault on 5 search spans was also listed as 5 roots (sc-006).

**Changes (deviations from the spec's ranking, documented in `stage_b.py`):**
1. Rank by evidence tier first (deterministic proof > a judge's faulty > a judge's suspicious), then the spec's earliest / confidence / reach.
2. One root per node: the earliest span stands for the node, the others go into its evidence, and their propagated errors are credited to it.

These are principled (hard evidence beats a hunch, and a fault is a node), but they were found on this dev set, so Phase 9's numbers on the full dataset are the honest test.

**Re-run after the changes (user's run):** top-1 node 13/16, step 15/16, paths 18/18. sc-011 and sc-012 are now top-1, and sc-006 shows one retrieval root. Hand-checked 10: 8 exact + 1 step-level. What remains wrong:
- **Missed second root (sc-006, F7):** found the retrieval root, missed account_status.
- **Spurious extra roots (sc-003 policy 0.80, sc-012 retriever 0.45, sc-011 writer / order_lookup):** judged-faulty nodes whose judges trusted inputs that were actually bad. Some propagated errors are then credited to them (sc-012: policy "via retriever" instead of via the proven tool_agent chain).
- **No-fault and silent cases (sc-004 F0, sc-014 F8):** a low-confidence writer flag (0.35) becomes a root when nothing else is anomalous.

All three are what P7's verification loop targets: patch the top candidate, replay, and keep only roots whose fix recovers the outcome. Controls still show false alarms (low-confidence judge flags). Removing those is the job of P7's replay verification, which tests each candidate by patching it, rather than another threshold tuned here.

### 2026-11-05 · P6-T6 · Wonder confirms Stage B's top roots; the controls' "false alarms" are real Writer hallucinations
`python tasks.py escalate` (user's run). Wonder kept Stage B's top-1 root on all 15 escalated runs (accuracy 15/16 either way), so on this dev set its value is in the *rest* of the ranking and in the narrative, not the top pick:
- **It pruned spurious roots,** but not consistently: sc-003, sc-018 and sc-026 went from 2 to the true 1, sc-011 from 4 to 2, sc-015 from 3 to 2; sc-035 went from 2 to 3 (it added the Writer). On the two-fault sc-024 it kept one root (order_lookup) and did not add the second true root (account_status).
- **The two "false alarms" on controls are not false.** Wonder's narratives for sc-004 and sc-016 (F0) say the Writer invented a refund timeline ("5-10" / "5-7 business days") that is in none of its inputs. That is true, the same natural hallucination the validators flag (`numbers_supported`, P6-T2) and the cause of the P5-T6 live-replay flips. Ground truth knows only *injected* faults, so a control with a genuine model error is scored as a false alarm. Likewise sc-014 (F8): the injected fault (a changed email nobody reads) is silent by design, and Wonder named the Writer's natural one instead. On sc-018 (F8) Wonder correctly identified the injected fake citation KB-GEN-001.
- **Consequence for Phase 9:** the false-positive rate on F0 controls overstates real false alarms. Natural faults should be labelled separately (e.g. writer replies with unsupported timelines, which the validators already detect) before reporting false-positive rates.
- **Fix:** Wonder sometimes ranked two spans of one node (sc-013: `retriever.search` twice). Its ranking is now collapsed to one root per node, as Stage B's is.
- **Cost:** about $0.08 per escalation. 15 of 18 runs escalated, mostly on low per-run κ (12×), which with 7–13 judged spans per run is a noisy signal. If escalation needs to be rarer, raise the bar on that rule first.

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
| 2026-11-02 | Clean-ticket accuracy by model (dev set) | Lightning 95%, Ultra 100%, Bolt 80% | — (dev set) | 20 each | profiles default / ultra / bolt; P1-T6 branch |
| 2026-11-02 | Clean-ticket final outcome (decision + required facts; dev set) | Lightning 17/20 (85%), Ultra 18/20 (90%), Bolt 12/20 (60%) | — (dev set) | 20 each | `python tasks.py clean --profile default ultra bolt -q`; P1-T7 branch |
| 2026-11-03 | Scenarios behaving as labelled, first pass (before fixes) | 77/111 = 69% (F0 18/18, F1 6/11, F2 8/11, F3 9/12, F4 4/12, F5 10/12, F6 2/12, F7 8/11, F8 12/12) | — | 111 | `python tasks.py verify --store` (profile default); P4-T4 branch |
| 2026-11-05 | Escalation to Muse Wonder, Stage C (P6-T6) | `diagnosis/escalation.py` (protected, no truth). **Rules** (`config/escalation.yaml`; any one fires, each with an `enabled` switch): run mean pairwise κ < 0.4 (`run_agreement` recomputes it from the stored per-judge votes, ignoring placeholder failures); top two roots in the same evidence tier with confidences < 0.15 apart (different tiers never count as close); more than 1 root; a propagation path longer than 3 hops; a validator error on a call inside a step a judge called ok. **Budget:** at most 20 escalations / $5 per batch, after which runs keep their Stage B diagnosis. **Wonder** (`judge_deep`, prompt `escalation-v1`, versioned and hashed like the node judge) gets a compact context: data edges and calls per step, every span's aggregate verdict with each judge's vote, failed validator checks, the ranked candidates with evidence, the propagation paths, and the top 4 candidates' inputs and outputs (summarized). It returns 0–5 ranked roots (each must cite a listed span, checked in the schema so a bad citation is sent back) and a narrative. `final_diagnosis` uses Wonder's ranking, keeps only propagated errors its roots still explain, and sets `escalated_to_deep` / `escalation_reason`; `cost_usd` = Wonder's cost only (ledger tag `escalation/<run>`). If Wonder never gives a valid ranking, Stage B stands and the reason says so | Claude |
| 2026-11-05 | Escalation to Muse Wonder (P6-T6) | escalated **15/18** runs (judge agreement ×12, 2–4 root candidates ×11, validator/judge conflict ×6, top two close ×1); **15/15 valid rankings**; **Wonder cost $1.1954 total, about $0.08 per escalation**, kept apart from Stage A. Top-1 root (node or step) **15/16 before and after**; controls with no root 0/2 before and after. Root counts: pruned to the true single root in sc-003, sc-018, sc-026 (2 → 1); fewer in sc-011 (4 → 2) and sc-015 (3 → 2); more in sc-035 (2 → 3) | — | 18 dev-set runs | `python tasks.py escalate` (judge_deep = muse-wonder-1.0, prompt escalation-v1); P6-T6 branch |
| 2026-11-05 | **Causal attribution, Stage B (P6-T5)** | top-1 root = true root node **13/16** (first pass 7/16), = true root or its step **15/16**, a true root among the ranked roots 15/16; controls with no root **0/2**; propagation paths valid **18/18**. Hand-checked 10: 8 exact, 1 step-level, 2 wrong (sc-004 F0 false alarm, sc-014 F8 silent). Hand-built algorithm cases: 10/10 + ranking/grouping tests | — | 18 dev-set runs | `python tasks.py attribution-check` (stored ensemble scores, validators; no model calls); P6-T5 branch |
| 2026-11-04 | Judge ensemble, Spark + Nemotron (P6-T4) | **pooled Cohen's κ 0.321** (Fleiss 0.292) over 205 judged spans; per run κ −0.13 to +0.68. Verdicts: Spark ok 130 / suspicious 49 / faulty 26; Nemotron (thinking off) ok 174 / 21 / 10; ensemble ok 127 / 49 / 29. Splits by node: retriever.search 35, writer 8, retriever 7, policy 4, tool_agent 4. $0 (Spark cached from P6-T3; Nemotron free). judge_oss not included: OpenRouter rejects the key (401) | — | 18 runs, 205 spans | `python tasks.py ensemble --judges judge_fast judge_nvidia`; P6-T4 branch |
| 2026-11-04 | Nemotron node judge with thinking off (P6-T4 re-check) | **20/20 schema-valid** (19/20 first try); 60,905 in / 10,171 out tokens (vs 46,071 out with thinking on); verdicts ok 13 / suspicious 2 / faulty 5 | — | 20 spans | `python tasks.py judge-sample --roles judge_nvidia`; P6-T4 branch |
| 2026-11-04 | Stage A on the diagnosis dev set (P6-T3) | **every span scored: 18/18 runs** (18–22 spans per run); 205 judgements in 205 model calls (no repairs, 0 judge failures); 23 spans settled by validators (no judge call); **cost $0.92 total, $0.051 per run** (Muse Spark; spans already judged in the P6-T1 sample were served from the cache at $0, so per-run cost ranges $0.012–$0.084) | — | 18 runs (2 per code F0–F8, seed 1) | `python tasks.py devset` then `python tasks.py stage-a` (judge_fast, node-judge-v1); P6-T3 branch |
| 2026-11-04 | **Faults caught by deterministic validators, no model call (P6-T2)** | root span flagged, error / any: **F4 33/33 / 33/33**, **F2 9/30 / 30/30**, F1 30/30 / 30/30, F3 21/30 / 21/30, F5 4/36 / 4/36, F6 0/33 / 22/33 (warnings), F7 24/30 / 30/30, F8 6/36 / 6/36. Controls: 1/54 with an error (a real model citation error), 28/54 with warnings only | — | 312 runs (258 faulty, 54 controls) | `python tasks.py validator-check`; P6-T2 branch |
| 2026-11-04 | Node judges schema-valid on 20 sampled spans (P6-T1) | **judge_fast (Muse Spark) 20/20** (20/20 on the first try; verdicts ok 8 / suspicious 6 / faulty 6; 50,658 in / 21,252 out tokens, **$0.154**, 250 s); **judge_nvidia (Nemotron Ultra) 20/20** (19/20 first try; ok 12 / suspicious 4 / faulty 4; 59,831 in / 46,071 out, $0, 1,124 s); judge_oss skipped (no key). Re-run fully cached (0.0 s) | — | 20 spans × 2 judges | `python tasks.py judge-sample` (seed 20261104, seed-1 scenario runs, prompt node-judge-v1); P6-T1 branch |
| 2026-11-04 | Replay stability, 3 repetitions (P5-T6) | **cached 27/27 identical** (score spread 0.0); **live 23/27 within tolerance** (decision 27/27, recovered verdict 23/27, reply text 2/27, largest score spread 0.5) | — | 27 replays × 3 × 2 modes | `python tasks.py replay-stability` (oracle patches, 3 runs per code); P5-T6 branch |
| 2026-11-04 | **Recovery of recoverable replays, rule patches (P5-T5)** | **85.7% (147 recoverable)**; by code recovered/recoverable: F1 26/26, F2 25/25, F3 18/18, F4 24/24, F7 13/34, F8 20/20. Matches the clean outcome: F1 29/29, F2 27/27, F3 21/21, F4 25/25, F7 15/37, F8 23/25. Mean score before -> after: F1 0.17 -> 0.95, F2 0.17 -> 0.96, F3 0.12 -> 0.93, F4 0.16 -> 0.98, F7 0.03 -> 0.47, F8 0.86 -> 0.92 | — | 164 replays | `python tasks.py patch-check --source rule --store` (164 ReplayResult records stored); P5-T5 branch |
| 2026-11-04 | Patch sources end to end (P5-T4): final correct after one root patch | **oracle** 230 replays: F1 26/29, F2 25/27, F3 18/21, F4 24/25, F5 27/30, F6 23/23, F7 19/44, F8 26/31. **rule** 164 replays (66 n/a, all policy targets): F1 26/29, F2 25/27, F3 18/21, F4 24/25, F7 13/37, F8 22/25. **model** (patcher = Muse Spark; first 20 verified F5/F6 runs): F5 9/11, F6 8/9, 0 n/a. Every replay descendants-only and tagged | — (model: n=20 subset) | 230 / 230 / 20 | `python tasks.py patch-check [--source rule \| --source model --code F5 F6 --limit 20]`; P5-T4 branch |
| 2026-11-04 | Patched replays change only the target and its descendants (P5-T3) | **230/230** (every fault code 100%) | — | 230 patches on 208 verified seeded runs | `python tasks.py patch-check` (oracle patch = the node's clean-run output of the same seed; other faults kept in force); P5-T3 branch |
| 2026-11-04 | Final correct after one oracle root patch (context for P5-T5) | F1 26/29, F2 25/27, F3 18/21, F4 24/25, F5 27/30, F6 23/23, F7 19/44, F8 26/31; 538 model calls not served from the record (gateway cache or live) | — | 230 | same run |
| 2026-11-04 | Unmodified replay reproduces stored runs (P5-T1) | **401/401 identical, zero diffs**, 0 live model calls, 0 recomputed. Served from the replay cache: 2,744 llm, 2,005 agent, 2,005 retrieval, 1,084 tool outputs | — | 401 | `python tasks.py replay-check` (strict; all stored runs: clean, P4 verify, and K=3 seeded runs, faulty included); P5-T1 branch |
| 2026-11-04 | **Scenarios behaving as labelled, K=3 seeded runs (P4-T5)** | **mean 83.0%, std 2.4 pp** (80.8 / 85.6 / 82.7%); final correct 33.3% ± 3.4 pp; 38/104 unstable, 4 never behave | — (K=3; bootstrap CIs in P9-T5) | 104 × 3 | `python tasks.py multirun` (profile default, seeds 1–3, temperature 0); P4-T5 branch |
| 2026-11-03 | **Scenarios behaving as labelled (P4-T4, final)** | **103/104 = 99%** (F0 18/18, F1 10/10, F2 10/10, F3 10/10, F4 10/11, F5 12/12, F6 11/11, F7 10/10, F8 12/12); miss: sc-042 | — (single run; variance in P4-T5) | 104 | `python tasks.py verify --store` (profile default); P4-T4 branch |
| 2026-11-03 | Scenarios behaving as labelled, second pass | 96/108 = 89% (F0 18/18, F1 10/11, F2 10/11, F3 10/11, F4 6/11, F5 12/12, F6 8/11, F7 10/11, F8 12/12) | — | 108 | `python tasks.py verify --store` (profile default); P4-T4 branch |
| 2026-11-03 | Muse Spark as the agent (dev set) | decision 20/20 (100%), final 18/20 (90%), 0 contract violations, 20/20 completed; 934 s live (~47 s/ticket); ~115 Meta requests, ~127k in / ~60k out tokens (~$0.42) per the Meta dashboard | — (dev set) | 20 | `python tasks.py clean --profile default ultra bolt spark -q`; P4-T1 branch |
| 2026-11-02 | Provenance exact (refs = declared data flow, hash chain intact) | 20/20 | — | 20 | `python tasks.py clean -q` (profile default); P2-T2 branch |
| 2026-11-03 | Trace integrity (no orphans, cycles, time errors) | 20/20 | — | 20 | `python tasks.py clean -q` (profile default); P2-T5 branch |
| 2026-11-01 | Clean-ticket decision accuracy (dev set) | 19/20 = 95% | — (dev set, not a headline number) | 20 | agent_default = nemotron-3.5-lightning, thinking off; P1-T5 branch |

---

## 11. Scope Cuts
<!-- Anything cut per the cut line in CLAUDE_meta.md §8, with reason. Mirror into docs/limitations.md. -->
