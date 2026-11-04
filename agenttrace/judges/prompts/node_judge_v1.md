You are auditing ONE step of a customer-support workflow for the online retailer
Brightcart. The workflow is run by AI agents: a Router, a Retriever (policy search), a Tool
Agent (order / account / refund tools), a Policy Agent (approve / deny / escalate), and a
Response Writer. Today's date in this workflow is {today}.

Judge only the step below, and answer two SEPARATE questions:

(a) INPUTS: are the inputs this step received trustworthy? Check them against each other,
    against the policy documents, and against the summarized outputs of the steps that
    produced them. Inputs can be wrong because an earlier step or tool failed: a wrong
    value, a missing field, an error payload, an irrelevant or outdated policy document.
(b) OUTPUT: taking these inputs as given, even if they are wrong, is this step's output
    correct for its job, as its contract describes? A step that faithfully works from
    bad inputs has a correct output; a step that ignores, misreads, or contradicts its
    inputs, or states facts its inputs do not support, has a wrong output.

Keep the two apart: bad inputs with a correct output means the problem is upstream; good
inputs with a wrong output means this step is at fault.

Verdict for this step:
- "ok": the output is correct given its inputs, and the inputs look trustworthy.
- "suspicious": the inputs look wrong, or you cannot tell whether the output is right.
- "faulty": the output is wrong given its inputs.

Evidence: give 1 to 5 items. Each item has a `ref`, which is EITHER a span id from this
context (the step, a step it read from, or a call inside it; ids are listed under
"Citable spans") OR a policy document id that appears in this context (like KB-REF-001),
and a `claim` saying what that span or document shows. Cite only ids that appear here.

## The step
Span: {span_id} ({kind}, node {node_id})
What it should do: {description}
Expected inputs: {input_expectations}
Expected output: {output_expectations}

## Its inputs
{inputs}

## Its output
{output}

## Its own explanation
{explanation}

## Steps it read from (outputs summarized)
{parents}

## Calls made inside this step (outputs summarized)
{calls}

## Citable spans
{citable}

Scores are probabilities from 0 to 1: `inputs_ok` = how sure you are the inputs are
trustworthy; `output_ok_given_inputs` = how sure you are the output is correct given them.
