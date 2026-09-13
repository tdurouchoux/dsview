# Learning app — design proposal

*Status: proposed · date: 2026-09-13 · author: tdurouchoux*

## Problem

dsview maps the **data-science landscape as encountered**: what was ingested, how it connects, what
is worth reading next. It says nothing about what is actually *retained*, and — more importantly —
nothing about what *should* be known.

This proposes a second system that models **the reader against a standard**: a daily micro-quiz
that keeps the basics warm and surfaces which areas have gone cold.

## Goals and non-goals

**Goal.** Answer, per topic, a deliberately coarse question: *do I still have the basics on this
methodology / model family?* Recognition-level, not depth.

**Goal.** Become a habit. An unused app is worth nothing here, so friction and engagement design
are functional requirements, not polish.

**Non-goal — measuring mastery.** No calibrated skill level, no certification. This is what makes
multiple choice the right instrument rather than a compromise.

**Non-goal — being a view over dsview.** See below; this is the central design constraint.

## The curriculum is not the knowledge graph

**What I want to read is not what I need to learn.** A reading list is driven by novelty: new
papers, new tooling, this month's agent framework. The fundamentals — experiment design,
cross-validation pitfalls, probability calibration, bias–variance, window functions, statistical
power — are not blogged about weekly, so they are thin or absent in the graph precisely *because*
they are settled. They are also exactly what a quiz should keep warm.

An earlier version of this proposal built the item pool directly on `extraction.extractiontopic`
and weighted selection by `graph_analysis.pagerank`. That is backwards twice over: it can only ask
about topics that happen to appear in past reading, and it then *prioritises the most-read ones* —
ranking by what is already freshest. Recorded here because the failure is tempting: the graph is
right there, well-structured, and already deduplicated.

So the system carries **two distinct topic spaces**:

| | `curriculum` topics | `extraction.extractiontopic` |
|---|---|---|
| Answers | what a data scientist should have | what I was exposed to |
| Sourced from | syllabi, textbook tables of contents, standard references, web research | ingested content |
| Owns | quiz scope | personalisation and grounding |

and a **mapping** between them, which is informative in both directions:

- curriculum topic with no kb coverage → a blind spot the reading list never touched. A prime
  quiz target, and the reason coverage-gap analysis comes back almost free.
- kb topic with no curriculum entry → frontier or tool-specific material. Fine to have read,
  not necessarily worth committing to memory.
- both → consolidation: read recently, so check whether it stuck.

The kb's role is therefore **personalisation, grounding and gap detection — not scope.** Curriculum
coverage is decided before the graph is consulted.

## Architecture

Two tempos, and the split follows them:

- **Offline, agentic, expensive, slow.** Build and maintain the curriculum; source reference
  material; write and critique questions; fill the bank. Runs weekly/monthly as a job.
- **Online, deterministic, instant.** Pick five items, record answers, derive streak and progress.
  Pure SQL, **zero LLM calls**, no spinner. This is the habit surface and it must stay boring.

Nothing in the serving path calls a model. Every LLM cost is amortised over weeks of sessions.

### Where the code lives — open decision

The earlier answer was "generation inside dsview, client in its own repo". The reframing above
weakens that: a canonical DS curriculum is not dsview's domain, and hosting it there gives dsview a
second mission plus a web-research agent it has no other use for.

Recommended instead, **three components**:

1. **dsview, unchanged** — stays the landscape map. Already exposes exactly the read interface an
   agent needs: `search_topic`, `search_content`, `get_connected_topics`,
   `get_connected_contents`, `explore_graph`, `get_nodes_ranking`. The MCP server *is* the
   integration point.
2. **Learning backend** (new repo) — curriculum, agents, question bank, answer log, HTTP API.
   Imports `dsview` as a uv git dependency for the `LLMModel` abstraction, config loading, prompt
   conventions and the eval harness; consults the knowledge base **as an MCP client**.
3. **Client** (new repo) — phone-first web app, presentation and gamification only.

Consuming the kb through MCP rather than by importing its schema is what structurally enforces the
previous section: the graph becomes one tool an agent may call, not the backbone the quiz is built
on. It cannot quietly become the curriculum.

The cost is real: a git-dependency version coupling, and no shared Postgres transaction. Keeping
generation inside dsview instead is still defensible — it is less moving parts today. **This is the
one decision worth settling before any code.**

### Contract

```
GET  /learning/session    -> today's items: question, choices, no answer key
POST /learning/answer     -> {question_id, choice_id} -> {correct, explanation}
GET  /learning/progress   -> streak, per-area accuracy and coverage
POST /learning/flag       -> {question_id} : "this question is broken"
```

The answer key is withheld from `GET /learning/session` and revealed by `POST /learning/answer` —
otherwise the client ships the answers to the browser.

**The answer log lives server-side**, as the single source of truth; the client derives streaks and
scores from it and holds no durable state. A second client, or a rewrite, inherits the full
history.

## Data model

Own `learning` schema (own database if it becomes a separate service). References to dsview topic
and content ids are stored as **plain integers, not foreign keys**, so the learning store can be
separated from dsview's Postgres later without a migration.

- **`curriculumtopic`** — `id`, `name`, `area`, `kind` (`fundamental` | `frontier`),
  `description`, `embedding`, `status` (`proposed` | `accepted` | `rejected`), `review_date`.
- **`curriculumsource`** — `id`, `curriculum_topic_id`, `url`, `note`. Provenance for an
  agent-proposed curriculum is what makes it trustable; without it, accepting a topic is an act of
  faith.
- **`kbtopiclink`** — `curriculum_topic_id`, `dsview_topic_id`, `similarity`, `method`. The
  mapping; drives consolidation and blind-spot detection.
- **`quizquestion`** — `id`, `curriculum_topic_id`, `dsview_content_id` (nullable, set for
  grounded consolidation items), `archetype`, `question`, `explanation`, `generation_date`,
  `retired`.
- **`quizchoice`** — `id`, `question_id`, `text`, `correct`, `distractor_topic_id`.
- **`quizanswer`** — `id`, `question_id`, `choice_id`, `answer_date`, `flagged`.
- **`questionlabel`** — human verdicts, multi-label; the judge's training data (see *Evaluation*).

**No topic-state table, no stored streaks.** Freshness, accuracy and streak are derived by SQL over
`quizanswer`. Stored state is a second source of truth that can drift, for an aggregation that is
free at this scale. Materialise only if the selection query actually becomes slow.

## Agents (offline)

The open-ended parts of this problem are genuinely agentic — the shape of the curriculum is not
known in advance, and finding authoritative material is search-and-read work:

- **Curriculum agent.** Web-searches syllabi, textbook tables of contents and standard references;
  proposes canonical topics with provenance; dedupes against the existing curriculum by embedding.
  Output lands as `status="proposed"` behind a human accept/reject gate — an auto-accepted
  curriculum is an unbounded quality risk at the root of everything downstream.
- **Reference agent.** For an accepted topic, fetches grounding material so questions are written
  from a source rather than from parametric memory. This is the main defence against confidently
  mis-keyed items.
- **Writer + critic loop.** Writer drafts an item; the judge (below) critiques; writer revises;
  accept or discard, bounded iterations. The judge therefore does double duty — inline critic at
  generation time *and* offline eval metric. This is the most valuable single idea in the design.

Deliberately **not** an agent: the curriculum↔kb mapping. Embedding neighbours plus an
`ERClassifier`-style pairwise confirmation already solves it, and that task is tuned and evaluated
(ADR 0001). Reuse it rather than adding a third agent.

**Framework: `pydantic-ai`.** Logfire is by the same team and already instrumented across this
project, so multi-agent traces come free; it is pydantic-native, matching the existing
structured-output style; and it speaks MCP, which is how the kb is consulted. Nothing
LangGraph-scale is warranted for a single-user system. A hand-rolled loop over the existing
`LLMModel` plus tool calls is the fallback if the dependency is unwelcome.

Cost: agentic generation is far more expensive per item than one structured call. It is also
offline, batched, and amortised over weeks of serving — which is the whole point of separating the
tempos.

## Question generation

### Archetypes

| Archetype | Grounded in | Distractors from |
|---|---|---|
| `recognition` | curriculum topic + fetched reference | sibling curriculum topics in the same area |
| `discrimination` | a confusable topic pair | the pair's counterpart |
| `application` | topic + typical use | siblings in the same area |
| `consolidation` | a summary of content actually read (via kb) | other topics linked to that content |

### Distractors are selected, not invented

The known failure mode of LLM-written multiple choice is implausible options: the key is
identifiable without knowing anything. Candidates come from structure instead —

- embedding neighbours within the curriculum (close but genuinely distinct);
- for consolidation items, the kb's own topic neighbourhood;
- **`extraction.ercomparison`** — a log of topic pairs similar enough to require an
  entity-resolution decision, with `merge_topic` recording the verdict. Pairs judged *not* to merge
  are confusable-but-distinct by construction: ready-made discrimination items.

The model writes the stem and the explanation; the candidate set comes from data.

### A note on grounding depth

dsview persists summaries, not raw content (full text exists only in `labels.labelledcontent`). For
consolidation items that caps depth at what a summary supports — acceptable at recognition level.
Curriculum items are unaffected: they are grounded in material the reference agent fetches.

## Daily selection policy

Five items, mix configurable in YAML:

- **2 cold fundamentals** — `kind="fundamental"`, longest since asked or previously wrong.
- **1 consolidation** — curriculum topic mapped to content ingested in the last few weeks.
- **1 blind spot** — accepted curriculum topic with no kb coverage.
- **1 new** — never asked.

A readable policy, not a learned scheduler: when a session feels wrong, the reason should be one
SQL query away. Note where pagerank survives — *ordering within consolidation items only*, among
topics actually read. It never defines scope. FSRS-style intervals are a later refinement, once
there is answer history to justify them.

## Gamification

Stated as the most important aspect, so treated as load-bearing:

- **Speed above all.** Pre-generated bank, no LLM at answer time, five items, immediate feedback.
  Under a minute, never a spinner.
- **Streak with a grace/freeze mechanic.** Streak loss is the dominant churn cause, and an
  unforgiving streak punishes exactly the busy weeks when the habit is most fragile.
- **Explanation on every answer**, generated offline, so a wrong answer teaches rather than scores.
- **Progress against the curriculum**, per area — a real skill tree, with blind spots visible as
  unfilled branches. Stronger than an XP number because the structure means something.
- **A consistent daily trigger**, reusing `notification/email_sender.py` until a PWA push earns its
  keep.
- **Weekly recap** — streak, accuracy by area, coldest topics.

Explicitly rejected: leaderboards and social comparison (single user), currencies with nothing to
buy.

Honest risk: solo gamification decays once novelty passes. What survives is the streak, the visible
coverage map, and near-zero friction. Cosmetic layers are not worth early investment.

## Evaluation — a judge trained on labelled questions

Question quality is the main risk, and it is not measurable by plumbing tests. The judge is
therefore a first-class model with its own labelled dataset and its own eval.

### Failure taxonomy (multi-label)

A question can fail several ways at once, so verdicts are checkboxes, not one enum:

- `mis_keyed` — the marked answer is wrong.
- `ambiguous` — more than one defensible answer.
- `cue_leaking` — key guessable from phrasing, length, grammatical agreement, "all of the above".
- `weak_distractors` — wrong options obviously wrong.
- `off_level` — trivia, or depth beyond recognition.
- `false_premise` — the stem itself is wrong.

### Labelled data, including cold start

There are zero labelled questions today. Three sources, cheapest first:

- **Hand labelling** in a Streamlit form beside the existing ones — for MCQ this is read-and-tick,
  so a seed of 100–200 is an afternoon.
- **Synthetic negatives** — corrupt known-good items deterministically: swap the key, replace
  distractors with far-away topics, append "all of the above". Manufactures labelled failures for
  `mis_keyed`, `weak_distractors` and `cue_leaking` at no labelling cost, which is what makes a
  judge trainable before the human set is large.
- **Production signals** — `flagged` items are labelled negatives for free; items answered
  correctly every time are uninformative; an item whose distractor beats its key is probably
  mis-keyed. An active-learning loop that costs nothing.

### The judge itself

An `LLMModel` subclass with one structured output field per failure mode. "Trained" means
**prompt-optimised against the labelled set**, following existing practice — DSPy optimisation of
extraction prompts already lives in `notebooks/` (`dspy_entity_resolution_opt.py`,
`dspy_content_description_opt.py`). Fine-tuning a small model is the fallback if an optimised
prompt plateaus, and only then, since it needs an order of magnitude more labels.

### Meta-evaluation, and the circularity trap

A judge is worthless unmeasured: `evaluation/quiz_judge.py` scores **agreement with human labels**
per failure mode (F1, and Cohen's κ for the subjective ones) on the `eval` split, reusing
`assign_rows` and the existing split ratios, logged to MLflow like every other task. Only past an
acceptable agreement does the judge become a generation gate.

**The trap:** once the judge filters generation *and* measures it, the metric grades its own
filtering and will look excellent regardless of reality. Mitigations — keep a human-labelled
held-out set as the real measure, re-label a fresh sample periodically, and track judge agreement
over time as a first-class metric rather than assuming it holds. This is the kind of eval that rots
silently.

## Phasing

1. Settle the repo split. Curriculum agent + human review gate; a first accepted curriculum for
   two or three areas only.
2. Mapping to the kb; question generation for those areas; hand-inspect the bank.
3. Serving routes + the smallest client that runs a daily session (streak, instant feedback,
   explanations).
4. Labelling form, judge, meta-eval; then wire the judge into the writer/critic loop.
5. Per-area coverage view and weekly recap. Widen the curriculum.

## Deferred

- **Reading recommendations.** Now partly free: a curriculum blind spot *is* a reading suggestion.
  The complementary signal worth recording — **`extraction.extractionlink`** holds links extracted
  from content that was read but never ingested; a link referenced across several reads and never
  followed is a strong recommendation from one SQL query, no LLM. Belongs in the dsview weekly
  digest before it justifies any UI here.
- **Spaced repetition proper** (FSRS per topic) — needs answer history first.
- **Free-text answers with an LLM judge** — only if the goal shifts from recognition to depth.
