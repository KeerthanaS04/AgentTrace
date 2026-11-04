# AgentTrace

Debugging and continuous explainability for multi-agent AI workflows. AgentTrace records full traces, rebuilds them as a data-flow graph, separates root causes from propagated errors, and proves a fix by counterfactual replay.

- Project spec: [CLAUDE_meta.md](CLAUDE_meta.md)
- Task checklist (serves as `PROGRESS.md`): [phases.md](phases.md)
- Findings log: [findings.md](findings.md)

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env        # then fill in API keys
python tasks.py setup       # install deps + pre-commit hooks
python tasks.py test
python tasks.py lint
```

`make <target>` works the same where `make` is installed.

## Choosing models

Code never names a model. Two config files decide which model each agent uses:

- [config/models.yaml](config/models.yaml): **role -> model**, e.g. `agent_alt_2 -> nvidia/nemotron-3.5-bolt-e4b`, with its params and pricing.
- [config/agents.yaml](config/agents.yaml): **profile -> which role each agent uses**, e.g. profile `mixed` = small model everywhere except the Policy Agent.

Swap a model by editing one of these files, then run `python tasks.py clean --profile <name>`.

## Seeing the numbers

Tests only report pass/fail. These commands print the actual measurements:

| Command | Shows |
|---|---|
| `python tasks.py smoke` | Each model role: ok / skipped / failed, latency, cost, and lifetime spend vs. budget caps (makes tiny live API calls) |
| `python tasks.py retrieval` | Retrieval recall@1/3/5 on the dev and held-out query sets, every miss, and PASS/FAIL against the target |
| `python tasks.py toy` | A toy 2-agent workflow run through the agent runtime: each step's input refs (which step produced each input), output, and explanation (no model calls, free) |
| `python tasks.py clean -v` | The five-agent workflow on the 20 clean tickets: expected vs. actual decision, the final-outcome check (`final`: right decision **and** the reply states every required fact; missing facts are listed), router category, contract violations, provenance (each step's input refs match the declared data flow, with matching content hashes), trace integrity (no orphan refs, cycles, or timestamp errors), graph topology (the span graph's data flow equals the declared workflow), the reasoning behind each miss, and PASS/FAIL against 90%. Uses NVIDIA (free tier); repeat runs come from cache. Add `--only clean-05` for one ticket |
| `python tasks.py clean --profile default ultra bolt -q` | The same 20 tickets on several model profiles (from `config/agents.yaml`), then a side-by-side table: decision accuracy, final-outcome correct, router accuracy, completed runs, contract violations, time, and each ticket's decision per model |
| `python tasks.py clean --only clean-07 --trace` | One ticket's trace as a tree: the input span, each agent span with its input refs, and the model, tool, and search calls nested under it, with model, tokens, cost, and time per span. The full `Run` is saved in `reports/clean_eval_<profile>.json` |
| `uv run agenttrace run --scenario clean-07` | Runs one scenario and stores its trace in `.cache/agenttrace.db`: run id, decision vs. expected, final outcome, provenance. Add `--trace` for the span tree, `--profile bolt` for another model |
| `uv run agenttrace runs` / `uv run agenttrace show <run_id>` | Lists stored runs (filter with `--scenario`), and reloads one from the database to print its span tree (`--node policy` prints that span as JSON) |
| `uv run agenttrace export <run_id> -o run.json` / `uv run agenttrace import run.json` | Writes stored runs to a JSON file (`--scenario X` for all of a scenario's runs, `--otel` for OpenTelemetry OTLP/JSON) and loads such a file (ours or another system's OTLP) back into the store |
| `uv run agenttrace graph <run_id>` | Causal-graph queries on a stored run: final spans, causal order, critical path (`--weight tokens` / `cost_usd`). `--from tool_agent.order_lookup` lists every path from that span to the final output; `--diff <other_run>` reports where two runs first diverge; `--json out.json` writes the React Flow graph JSON the UI consumes (schema: `ui/schema/run-graph.schema.json`) |
| `uv run agenttrace diff <run_a> <run_b>` | Aligns two stored runs span by span: counts of unchanged / changed / only-in-one spans, each changed span with what changed (status, input, output, model, params, patched), and the first divergence. `--all` lists unchanged spans too |
| `uv run python -m agenttrace.scenarios.generator --check` | The labelled scenario dataset (`data/scenarios/scenarios.json`): a fault-code × ticket-category coverage table, and whether the committed file matches the generator. Without `--check` it rewrites the file |
| `uv run python -m agenttrace.scenarios.isolation` | Ground-truth isolation: modules scanned, which are harness (may see truth) and which are protected (diagnosis, explainability, judges), and any import path from outside the harness to the answers (exit 1 if one exists) |
| `python tasks.py verify` | Scenario verification: every scenario run clean and with its fault (default profile, NVIDIA free tier): per scenario the clean vs. faulty outcome and whether it behaves as labelled (F1–F7 change the outcome, F8 leaves it unchanged, F0 is correct), a per-code summary, reasons for misbehaving scenarios, PASS/FAIL against 95%. `--only sc-003 -v` for one scenario, `--code F3` for one fault type, `--store` to save each scenario's run (faulty, or the control) for diagnosis |
| `python tasks.py multirun` | Multi-run protocol: every scenario run K=3 times with seeds 1–3 (each seed passed to every model call and cached separately), each judged as in `verify` against the clean run of the same seed, all stored as `<scenario>-default-s<seed>`. Prints each run's behaves-as-labelled and final-correct rates, their mean / std / variance across runs, per-code spread, and the scenarios whose result differs between seeds. `--k 5`, `--code F4`, `--temperature 0.7` to sample instead of the configured decoding |
| `python tasks.py replay-check` | Record/replay check: every stored support-workflow run replayed *strictly* through the workflow from a replay cache filed from that run (a model call with no recorded output is an error, so no live call is possible). Per run: identical (zero diffs in every span's output, input, model, params, status, explanation, and the final output) / with diffs / missing a recording, plus how many outputs the cache served. `--prefix sc-001`, `--limit 20 -v` |
| `uv run pytest tests/<file>.py -v` | Each individual test by name, for one area |
