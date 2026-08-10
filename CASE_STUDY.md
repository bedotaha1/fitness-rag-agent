# Case Study: Why My Agent Kept Searching for Anglerfish Facts

## The symptom

My fitness-domain RAG agent evaluated at 83.33% (15/18) on a golden dataset of
tool-call decisions. Every single failure had the same shape: the agent
called its `search_fitness_knowledge` tool on questions that had nothing to
do with fitness — coffee brewing temperature, why anglerfish have a glowing
lure, how sourdough fermentation works.

A one-directional failure pattern (every miss in the same direction) is a
strong signal of something systematic, not random model variance — so
instead of re-running the eval and hoping for a better roll, I went looking
for the actual mechanism.

## First check: was it even a real bug?

Before touching any code, I checked the golden dataset's own labels against
precedent from an earlier session — an out-of-scope swimming question had
previously been labeled `should_call_tool: True`, reasoning "a reasonable
agent would search, then correctly find nothing." This session's labels
disagreed with that precedent without me noticing. Lesson: a failing eval
doesn't automatically mean the *system* is wrong — the golden dataset itself
can be the bug, and it's worth checking that before anything else.

## The real clue: a 3-2 split within one category

Looking closer, only 3 of my 5 out-of-scope questions actually leaked
through:

| Question | Tool called? |
|---|---|
| "What brewing temperature is recommended for pour-over coffee?" | Yes |
| "Why do anglerfish have a glowing lure?" | Yes |
| "How does sourdough fermentation work?" | Yes |
| "Who invented the bicycle?" | No |
| "What's the capital of Brazil?" | No |

If the agent were treating "out of scope" as one uniform bucket, all five
should have behaved the same way. They didn't — and that split, not the
overall score, was the actual signal worth explaining.

## Ruling out the obvious explanation

My tool's `description` field listed six explicit topics (strength training,
cardio, recovery, sleep, nutrition, injury prevention) — none of which
overlap with coffee, anglerfish, or sourdough. So the leak wasn't
topic-matching against the description; something else was driving the
decision.

## The actual mechanism

The system prompt told the model how to *behave once it decided to search*
(clarify ambiguous questions, don't hallucinate numbers, admit empty
results) — but never told it *whether* a question belonged in scope at all.
In that gap, the model fell back on an implicit heuristic: **use the tool
when uncertain about the answer, skip it when confident.**

"Who invented the bicycle?" and "capital of Brazil?" are single, well-known
facts the model is confident about — no search needed. "Brewing
temperature," "how does fermentation work," and "why do anglerfish glow"
are exactly the kind of detail-heavy questions a general-purpose assistant
instinctively double-checks — even though confidence has nothing to do with
whether the topic belongs in a *fitness* knowledge base.

The bug wasn't that the model's confidence was miscalibrated. It's that
**confidence was influencing the decision at all**, when domain membership
should have been a hard gate, checked first, with confidence irrelevant to
that gate.

## The fix

An earlier attempt at fixing this — "if not confident AND not in scope, say
I don't know" — was itself a logical bug: it only rejects when *both*
conditions hold, so a confident-but-out-of-scope question could still slip
through. The corrected version checks domain membership as its own
standalone gate, with confidence never entering that decision:

```
"The knowledge base covers exactly six topics: strength training, cardio,
recovery, sleep, nutrition, and injury prevention. If the user's question
is not about one of these six topics, do not call the tool, regardless of
how confident or unconfident you are in answering it yourself — respond
that the question is outside the scope of this knowledge base."
```

Paired with a tool description that states relevance only ("Do not use for
topics outside these six areas") — keeping the two fields doing two
separate jobs, since the tool description is read at tool-selection time and
the system prompt governs the model's overall behavior.

## Result

Re-running all five out-of-scope questions post-fix: 0/5 incorrectly
triggered the tool. The original swimming-question labeling inconsistency
also resolved itself — a hard domain gate makes "out of scope" behave
identically regardless of how confident-sounding the question is.

## What this taught me

The 5-bucket debugging framework I now use for any agent eval failure:

1. Wrong tool-call decision → system prompt or tool description need
   sharpening — not a retry.
2. Tool called correctly but query was bad → system prompt
   query-formulation issue.
3. Good query, retrieval still wrong/empty → not an agent problem —
   chunking or distance threshold.
4. Retrieval good, but the final answer fails a faithfulness check →
   generation-grounding instructions need tightening.
5. Everything looks correct, but the judge still marks it false → check
   the judge's own rubric before touching the system.

Re-running a failing case is only ever a *diagnostic* (to separate a real
bug from random model variance) — never a fix on its own.
