# Learning app — design proposal

*Status: proposed · date: 2026-09-13 · author: tdurouchoux*

## Problem

dsview maps the **data-science landscape**: what exists, what was ingested, how it connects. It
says nothing about what is actually *retained*. The reading list grows faster than the memory of
it, and there is currently no way to tell a topic that was genuinely absorbed from one that was
skimmed eighteen months ago and forgotten.

This proposes a second, complementary system that models **the reader** rather than the landscape:
a daily micro-quiz that keeps the basics warm and surfaces which areas have gone cold.

## Goals and non-goals

**Goal.** Answer, per topic, a deliberately coarse question: *do I still have the basics on this
methodology / model family?* Recognition-level, not depth.

**Goal.** Become a habit. A correct-but-unused app is worth nothing here, so friction and
engagement design are functional requirements, not polish.

**Non-goal — measuring mastery.** No attempt to score depth of understanding, produce a
calibrated skill level, or certify competence. This is what makes multiple-choice the right
instrument rather than a compromise (see *Question format*).

**Non-goal (v1) — recommending what to explore next.** Coverage-gap analysis is deferred; see
*Deferred*.

## Architecture — a producer/consumer split

Generation lives **inside dsview**; the user-facing client is a **separate repository**.

| | dsview (`dsview/learning/`) | client (new repo) |
|---|---|---|
| Owns | topic graph, question bank, answer log | presentation, gamification, session UX |
| Runs as | weekly generation cronjob + HTTP routes | phone-first web app |
| Talks to | Postgres (`learning` schema) | dsview HTTP API only |

Rationale: question generation is an LLM-extraction task over the knowledge graph and belongs
beside the other `LLMModel` tasks — it needs the topic embeddings, the ER history, the prompt
conventions, the eval harness and the model config that already exist. None of that belongs in a
client. Conversely the client needs to iterate on interaction design at a completely different
tempo, and should be replaceable without touching the pipeline.

**The answer log stays in dsview**, as the single source of truth. The client derives streaks,
scores and progress from it rather than keeping its own copy. This keeps the client thin and
swappable — a rewrite, or a second client, inherits the full history — and lets the generation
policy read real outcomes when choosing what to ask next. The client holds no durable state of
its own beyond a session token.

### Contract

Served from the existing FastAPI app (`dsview/api.py`), a `/learning` router:

```
GET  /learning/session    -> today's items: question, choices, no answer key
POST /learning/answer     -> {question_id, choice_id} -> {correct, explanation}
GET  /learning/progress   -> streak, per-tag accuracy and coverage
POST /learning/flag       -> {question_id} : "this question is broken"
```

The answer key is withheld from `GET /learning/session` and revealed by `POST
/learning/answer` — otherwise the client holds the answers and the quiz is trivially inspectable
in devtools.

## Data model — `learning` Postgres schema

A fourth schema alongside `content`, `extraction`, `labels`, in
`dsview/db/schemas/learning_schema.py`:

- **`quizquestion`** — `id`, `topic_id` (FK `extraction.extractiontopic`), `content_id`
  (FK `content.inputcontent`, set only for source-recall items), `archetype`, `question`,
  `explanation`, `generation_date`, `retired`.
- **`quizchoice`** — `id`, `question_id`, `text`, `correct`, `distractor_topic_id` (set when the
  distractor was drawn from a real topic rather than written by the model).
- **`quizanswer`** — `id`, `question_id`, `choice_id`, `answer_date`, `flagged`.

**No topic-state table.** Per-topic freshness ("last asked", "recently wrong") is derived by SQL
over `quizanswer` rather than materialised. Stored state would be a second source of truth that
can drift from the log, for a table of a few thousand rows where the aggregation is free.
Materialise it later only if the selection query actually becomes slow.

Streaks and scores are likewise derived from `quizanswer.answer_date` — never stored.

## Question generation

A weekly cronjob (same pattern as `dsview_digest_cronjob.yaml`) tops up the bank for topics that
are short of unseen items. Daily selection is then pure SQL with **zero LLM calls**, so a session
loads instantly — the single most important property for a habit.

New `LLMModel` subclass in `dsview/learning/`, with `prompts/system_quiz_generation.txt` +
`prompts/user_quiz_generation.txt` and a `ModelType.QUIZ_GENERATION` entry in `config.py` and
`config/model.yaml`, following `topics_extraction.py`.

### Archetypes

| Archetype | Grounded in | Distractors from |
|---|---|---|
| `recognition` | topic name + description | nearest topics of the same `type` by embedding |
| `discrimination` | a confusable topic pair | the pair's counterpart |
| `application` | topic + its `type` | sibling topics of the same `type` |
| `source_recall` | one content's summary | other contents sharing a tag |

### Distractors are selected, not invented

The known failure mode of LLM-written multiple choice is implausible wrong answers: the key is
identifiable without knowing anything. Two dsview-specific assets avoid this:

- **`ExtractionTopic.embedding`** — nearest neighbours of the same `type` are semantically close
  but genuinely distinct, which is exactly the right difficulty.
- **`extraction.ercomparison`** — a log of topic pairs that were similar enough to require an
  entity-resolution decision, with `merge_topic` recording the verdict. Pairs judged *not* to
  merge are, by construction, confusable-but-distinct: ready-made discrimination items.

So the model writes the stem and the explanation; the candidate set comes from the graph.

### Depth is bounded by summaries — and that is fine here

Raw content is not persisted: `extraction` stores summaries, topics, tags and links, and full
text exists only in `labels.labelledcontent` for labelled rows. Questions can therefore probe what
a summary supports, not what section 4 of a paper argued. Under the recognition-level goal this
costs nothing. It would be the binding constraint if the goal ever became depth — at which point
the fix is persisting markdown at ingest time, a separate dsview change.

## Daily selection policy

Five items, drawn by a transparent weighted policy with the mix in `config/learning.yaml`:

- **2 cold** — topics answered wrong, or not asked for longest.
- **2 recently read** — topics linked to content ingested in the last few weeks (consolidation:
  did last month's reading stick?).
- **1 new** — a never-asked topic, weighted by `graph_analysis.pagerank` so central topics in
  *this* landscape come up before peripheral ones.

Deliberately a readable policy, not a learned scheduler: when a session feels wrong, the reason
should be inspectable in one SQL query. Spaced repetition (FSRS-style intervals) is a later
refinement once there is enough answer history to justify it.

## Gamification

Stated as the most important aspect, so treated as load-bearing:

- **Speed above all.** Pre-generated bank, no LLM at answer time, five items, immediate
  per-question feedback. A session should be under a minute and never show a spinner.
- **Streak** with a grace/freeze mechanic. Streak loss is the dominant churn cause in daily-habit
  apps, and an unforgiving streak punishes exactly the busy weeks when the habit is most fragile.
- **Explanation on every answer**, generated offline alongside the question, so a wrong answer
  teaches instead of only scoring.
- **Progress against your own map.** The 27 tags in `config/extraction.yaml` form a natural skill
  tree: per-tag coverage and accuracy bars. This is the strongest available hook — progress
  measured against a landscape you actually curated, rather than an arbitrary XP number.
- **A consistent daily trigger**, reusing `notification/email_sender.py` until a PWA push is
  worth building.
- **Weekly recap** — streak, accuracy by tag, coldest areas. Doubles as the retention hook.

Explicitly rejected: leaderboards and social comparison (single-user app), and currencies with
nothing to spend them on.

Honest risk: solo gamification decays once novelty passes. The mechanics that survive that are
the streak, the visible coverage map, and near-zero friction — the cosmetic layers are not worth
early investment.

## Evaluation path

Per the project rule that a task without an eval path is incomplete, and because bad questions
are the main quality risk:

- **`labels.quizquestionlabels`** — verdict per question: `valid` / `ambiguous` / `wrong_key` /
  `trivial`, plus a distractor-plausibility judgement.
- **A labelling form** in the Streamlit interface, alongside the existing ones.
- **`dsview/evaluation/quiz_generation.py`** — share of valid questions and share of plausible
  distractors on the `eval` split, logged to MLflow like every other task, reusing `assign_rows`
  and the existing split ratios.
- **Production item statistics as a free secondary signal.** Items answered correctly every time
  are too easy to be informative; an item whose distractor is chosen more often than its key is
  probably mis-keyed. Both are computable from `quizanswer` at no cost, and `flagged` gives a
  direct channel for the ones that are simply wrong.

## Phasing

1. `learning` schema, generation task + prompts, weekly cronjob, bank for the highest-pagerank
   topics. Verified by inspecting generated questions by hand.
2. `/learning` routes + the smallest client that can run a daily session (streak, instant
   feedback, explanations).
3. Labelling form + evaluation module; tune the generation prompt against it.
4. Per-tag coverage view and weekly recap.

## Deferred

- **Coverage-gap analysis** ("what is worth exploring next"). Parked deliberately, but the cheap
  high-signal source is worth recording: **`extraction.extractionlink`** holds links extracted
  from content that was read but never ingested. A link referenced across several reads and never
  followed is a strong reading recommendation, available from one SQL query with no LLM. Thin
  coverage on a tag is the second signal. Both belong in the weekly digest before they justify
  any UI.
- **Spaced repetition proper** (FSRS per topic) — needs answer history first.
- **Free-text answers with an LLM judge** — only if the goal ever shifts from recognition toward
  depth.

## Open question

Whether the `/learning` routes belong in the existing ingestion API deployment or a separate
service. The ingestion API is currently public-ish and stateless per request; the learning routes
are personal and write history. Same codebase either way — only the deployment unit is in
question.
