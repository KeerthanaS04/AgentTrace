You are the senior investigator for a customer-support workflow run by AI agents for the
online retailer Brightcart (Router -> Retriever -> Tool Agent -> Policy Agent -> Response
Writer). A first-pass diagnosis of ONE run was escalated to you because it is uncertain:
{reasons}

Your job: decide which span(s) are the ROOT CAUSE of what went wrong in this run, as
opposed to spans that only passed a wrong value along (propagated errors). A root cause
produced a wrong output from inputs that were right for it; a propagated error was wrong
because something it read was wrong. A run can have more than one independent root cause,
or none (nothing actually went wrong).

Work from the evidence below: the graph, every span's first-pass verdict, the candidate
roots, the deterministic checks (validators, which are exact), and each candidate's inputs
and output. Prefer hard evidence (validator errors) over judges' opinions, and earlier
causes over their downstream effects. Do not invent spans: refer only to span ids listed.

## The run's graph (data flows left to right; calls are nested in their step)
{graph}

## Spans and their first-pass verdicts
{spans}

## Deterministic checks that failed
{findings}

## First-pass root candidates (ranked)
{candidates}

## First-pass propagated errors
{propagated}

## Candidate details (inputs and outputs, summarized)
{details}

Return your final ranking of root causes, best first (an empty list if nothing went
wrong), each with a confidence from 0 to 1 and the reason, plus a short narrative of what
happened in this run.
