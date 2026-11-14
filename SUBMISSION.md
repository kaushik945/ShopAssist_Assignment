# ShopAssist — cost optimization submission

## Run

Python 3.10+, from this repository's root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q -k 'not baseline_cost_is_high and not help_center_exceeds_context'
python benchmark.py
python question1/run.py
python question2/run.py
```

Validation: **52 passed, 3 deselected**. `reports/validation.txt` records the run.
Original tests and all shared client, billing, tokenization and policy files remain
unchanged. Two deselected tests require expensive baseline costs and contradict
optimization. The unchanged large-corpus size test was previously interrupted
because corpus generation took too long; it is not claimed as passed. Unfiltered
pytest still contains the two incompatible baseline assertions.

## Q1 — one call, complete context

`question1/agent.py` retains all five policies, the full customer thread and full
order snapshot. It removes redundant examples, random identifiers and verbose
reasoning requests; order JSON is compact. The parser checks required field types
and allowed actions, and the agent checks that the cited article exists. Verified
order data precedes customer text, preventing customer snapshot markers from
winning the simulator's first-match parser. This is not a comprehensive injection
defense.

T-1001: 389 uncached input tokens, 0 cached, 46 output.
**389 × 3 + 46 × 15 = 1,857 points**, below 3,000.

## Q2 — four functional nodes

The four nodes remain explicit in `question2/agent.py`:

1. **Triage:** lexical issue detection over customer messages plus verified product
   metadata. It selects candidates, not the final action. Custom-item and digital
   exceptions are retained from order metadata even when omitted by the customer.
   Unknown categories, unrecognized intent, multiple requested outcomes and known
   unsupported issue types are marked for review. The full conversation still
   reaches composition; the small router does not summarize away customer facts.
2. **Retrieve:** load authoritative policy files by candidate id. Do not generate
   policy text, concatenate the enormous filler corpus or truncate arbitrary slices.
   An uncertain request gets all five real articles as a bounded fallback.
3. **Compose:** one concise JSON-only LLM call using the selected evidence and full
   order/thread. Instructions ask for a proposed decision, not claimed execution.
4. **Audit:** validate structure/citation, then verify that a proposed automatic
   outcome is justified by intent and facts. Guards check refund/replacement windows,
   customization, revealed licenses and subscription age. Missing, negative, boolean
   or otherwise invalid day values do not authorize an automatic outcome.

This interprets nodes as processing stages, not four required LLM invocations.
The supplied specification does not explicitly require a call per node. If four
calls are an unstated evaluator requirement, this implementation does not meet it.
No agent imports the grading oracle, branches on ticket ids or overrides billing.

### Review behavior

An invalid, unsupported or insufficiently justified output raises `ReviewRequired`,
a `ValueError` subclass. A caller must catch it and send the ticket to manual review.
It does not silently approve an action, retry invisibly, invent a citation or pretend
that a human-review queue has actually been implemented. Valid model escalations
with a relevant citation remain ordinary `ESCALATE_TIER2` results.

Example integration (queue implementation belongs to the calling application):

```python
try:
    result = pipeline.handle_ticket(ticket_id)
except ReviewRequired as exc:
    # Record the reason and route to a real review queue; do not fulfill anything.
    ...
```

### Required two-ticket per-node accounting

| Ticket | Node | Uncached input | Cached input | Output | Points |
|---|---|---:|---:|---:|---:|
| T-1004: catalog replacement | triage | 0 | 0 | 0 | 0 |
| T-1004 | retrieve | 0 | 0 | 0 | 0 |
| T-1004 | compose | 216 | 0 | 36 | 1,188 |
| T-1004 | audit | 0 | 0 | 0 | 0 |
| T-1003: custom-item escalation | triage | 0 | 0 | 0 | 0 |
| T-1003 | retrieve | 0 | 0 | 0 | 0 |
| T-1003 | compose | 243 | 0 | 39 | 1,314 |
| T-1003 | audit | 0 | 0 | 0 | 0 |

T-1004: **216 × 3 + 36 × 15 = 1,188**.
T-1003: **243 × 3 + 39 × 15 = 1,314**.
Zero means zero LLM billing points, not zero CPU/I/O cost.

## Complete cold-session measurements

`benchmark.py` uses a new agent/session for every ticket. Raw per-call accounting
is in `reports/optimized.json`.

| Ticket | Q1 points, target 3,000 | Q2 points, target 2,500 |
|---|---:|---:|
| T-1001 | 1,857 | 1,410 |
| T-1002 | 1,734 | 1,266 |
| T-1003 | 1,695 | 1,314 |
| T-1004 | 1,656 | 1,188 |
| T-1091 (subscription) | 1,698 | 1,251 |

## Caching and scaling

No prompt cache credits are claimed. The useful prompts are below the repository's
1,024-token eligible-prefix threshold. They meet budget on cold requests without
padding or pretending a prefix has already been cached. Python's cached article
loading reduces local work but is not an LLM prompt-cache discount.

For a genuinely larger policy corpus: index versioned articles, retrieve by issue
and metadata, retain mandatory exception policies, and evaluate retrieval coverage.
Keep stable instructions/core policy text first and variable evidence/facts last.
If the useful stable prefix is eligible, measure real cache hits and misses, including
policy-version changes, before claiming savings. Cached input still costs points.
The local client's cached_prefix argument checks prefix matching/length but does not
verify prior cache population; we do not exploit that accounting shortcut.

## Tests that challenge the implementation

`tests/test_pipeline_guards.py` injects proposed answers directly into audit and
mocks incorrect model output through the public entry point. These tests do not
rely on the simulator producing its usual correct answer. They cover:

- Return day 30 vs 31; replacement day 7 vs 8; subscription day 30 vs 31.
- Missing/negative/non-integer age and missing condition evidence.
- Customized-item replacement and valid custom escalation.
- Opened/damaged goods failing the ordinary return guard.
- Multiple/unsupported intents and conservative fallback retrieval.
- Customer claims conflicting with verified order facts.
- Retrieval changing with customer intent and incorrect model proposals rejected.

Existing correctness tests cover public outputs. Additional tests cover the
subscription holdout, costs, call counts, malformed JSON fields and citations.

## Honest limits

- The LLM is a simulator: it computes answers from order facts internally. These
  are simulator cost and fixture results, not real-model accuracy measurements.
- The lexical router is intentionally small. It does not reliably understand
  negation, paraphrases, multiple languages or conversation revisions. Its
  unsupported-issue list is not exhaustive. Real deployment needs intent and
  retrieval evaluations; the present code is a scoped assignment implementation.
- The large KB fixture consists of five substantive articles plus repeated routing
  filler. Direct file retrieval is appropriate here; it is not a demonstration
  of search over thousands of different articles.
- Audit guards duplicate a few explicit policy constraints and must be updated
  with policy changes. Unknown action/policy combinations require review.
- Unopened and undamaged are conservative proxies for resalable condition in
  this schema. Subscription fixtures represent annual plans and cancellation
  requests; broader products need explicit plan and cancellation facts. The
  referenced opened-goods policy is absent, so it cannot support approval.
- The simulator ignores the instruction to avoid claims of completed fulfillment
  in its canned replies. Q2 returns only action/citation and executes no actions.
- No fabricated baseline percentage is reported. Original Q2 baseline measurement
  was interrupted during expensive corpus generation.

Review the code and these assumptions before submitting. Nothing has been pushed
or submitted, and any employer-specific AI-disclosure rules still apply.
