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

## Seeing the numbers

Tests only report pass/fail. These commands print the actual measurements:

| Command | Shows |
|---|---|
| `python tasks.py smoke` | Each model role: ok / skipped / failed, latency, cost, and lifetime spend vs. budget caps (makes tiny live API calls) |
| `python tasks.py retrieval` | Retrieval recall@1/3/5 on the dev and held-out query sets, every miss, and PASS/FAIL against the target |
| `python tasks.py toy` | A toy 2-agent workflow run through the agent runtime: each step's input refs (which step produced each input), output, and explanation (no model calls, free) |
| `uv run pytest tests/<file>.py -v` | Each individual test by name, for one area |
