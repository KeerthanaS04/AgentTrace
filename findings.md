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
| 2026-10-28 | Git workflow | One branch per task (`p{phase}-t{n}-<slug>`); Claude creates the branch, user commits and merges. Base branch is `master` | User |
| 2026-10-28 | `phases.md` vs `PROGRESS.md` | `phases.md` serves as `PROGRESS.md`; `docs/lessons.md` points to `findings.md` | Claude |
| | Team size / submission format | | |
| 2026-10-28 | Non-Meta models | Include NVIDIA + OSS as judges and agents under test, so results hold up beyond Meta's own models (still check the rules) | User |
| | Muse Spark model ID + limits | | |
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

## 2. Model and Provider Findings
<!-- Verified model IDs, API quirks, rate limits, pricing, non-determinism at temp 0, structured-output reliability -->

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
