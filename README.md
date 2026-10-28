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
