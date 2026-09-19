# Learning app — design proposal

*Status: proposed · date: 2026-09-19 · author: tdurouchoux*

**Scope.** This document specifies a new system that lives in **its own repository** (working name
`dsrecall`, placeholder). It is kept in the dsview repo because that is the repo that exists today,
and because one section of it is an integration contract against dsview. Every path, table and tool
written `dsview/...` refers to the **`tdurouchoux/dsview`** repository and its deployment, verified
against dsview `2.7.0`.

## Problem

dsview maps the **data-science landscape as encountered**: what was ingested, how it connects, what
is worth reading next. It says nothing about what is actually *retained*, and — more importantly —
nothing about what *should* be known.

This proposes a second system that models **the reader against a standard**: a daily micro-quiz
that keeps the basics warm and surfaces which areas have gone cold.

## Goals and non-goals

**Goal.** Answer, per topic, a deliberately coarse question: *do I still have the basics on this
methodology / model family?* Working knowledge — how it works, when to use it, what goes wrong —
not depth, and never mere name recognition.

**Goal.** Become a habit. An unused app is worth nothing here, so friction and engagement design
are functional requirements, not polish.

**Goal — exercise the current LLM tooling landscape.** Building this on 2026 tooling is part of the
point, not incidental. dsview's LLM layer is a 2024 design that has since been overtaken by
framework-level equivalents; reusing it here would save a week and forfeit the exercise. So
**nothing is imported from dsview** — see *Stack* and *Integration*.

**Non-goal — measuring mastery.** No calibrated skill level, no certification. This is what makes
multiple choice the right instrument rather than a compromise.

**Non-goal — being a view over dsview.** See below; this is the central design constraint.

## The curriculum is not the knowledge graph

**What I want to read is not what I need to learn.** A reading list is driven by novelty: new
papers, new tooling, this month's agent framework. The fundamentals — experiment design,
cross-validation pitfalls, probability calibration, bias–variance, window functions, statistical
power — are not blogged about weekly, so they are thin or absent in the graph precisely *because*
they are settled. They are also exactly what a quiz should keep warm.

An earlier version of this proposal built the item pool directly on dsview's extracted topics and
weighted selection by pagerank over the reading graph. That is backwards twice over: it can only ask
about topics that happen to appear in past reading, and it then *prioritises the most-read ones* —
ranking by what is already freshest. Recorded because the failure is tempting: the graph is right
there, well-structured, already deduplicated by entity resolution.

So `dsrecall` owns a **curriculum** — a topic space of its own, sourced independently of what was
read, each topic marked `fundamental` or `frontier`. dsview is then consulted for three things and
nothing else:

- **curriculum sources** — courses and documentation already curated in the knowledge base are
  syllabi, i.e. exactly the structured material the curriculum agent is looking for (see
  *Integration*);
- **grounding** — when a curriculum topic *was* covered by something read, questions can be written
  against that reading;
- **coverage** — whether a curriculum topic appears in the knowledge base at all.

Curriculum scope is decided before the graph is consulted. A topic absent from dsview is still in
the curriculum; it is simply a topic whose questions are written from fetched references rather than
from personal reading.

## Stack

- **`pydantic-ai`** — agent definitions, LLM provider access, and the MCP client. Replaces
  everything dsview's `model_utils` does, at framework level, and speaks MCP natively.
- **MLflow 3 (GenAI)** — evaluation, LLM judges, human-feedback alignment, and **prompt management
  via the prompt registry**. Prompts are registered artefacts with versions, not files in a repo
  directory.
- **Postgres** — own database, no relation to dsview's.
- A small HTTP API and a phone-first client (below).

Provider choice is left open on purpose; comparing a few is part of the exercise. Exact MLflow
GenAI API surface should be pinned down against the installed version in phase 1 rather than
assumed from this document.

## Architecture

Two tempos, and the split follows them:

- **Offline, agentic, expensive, slow.** Build and maintain the curriculum; source reference
  material; write and critique questions; fill the bank. Runs weekly/monthly as a job.
- **Online, deterministic, instant.** Pick five items, record answers, derive streak and progress.
  Pure SQL, **zero LLM calls**, no spinner. This is the habit surface and it must stay boring.

Nothing in the serving path calls a model. Every LLM cost is amortised over weeks of sessions.

### Components

1. **dsview — unchanged.** Consumed over its MCP server, never extended for this purpose.
2. **`dsrecall` backend — new repo.** Curriculum, agents, question bank, answer log, HTTP API.
3. **Client — new repo, or a subdirectory of the backend.** Presentation and gamification only, no
   durable state beyond a session token.

### Contract with the client

```
GET  /learning/session    -> today's items: question, choices, no answer key
POST /learning/answer     -> {question_id, choice_id} -> {correct, explanation}
GET  /learning/progress   -> streak, per-area accuracy and coverage
POST /learning/flag       -> {question_id} : "this question is broken"
```

The answer key is withheld from `GET /learning/session` and revealed by `POST /learning/answer` —
otherwise the client ships the answers to the browser.

**The answer log lives in the backend**, as the single source of truth; the client derives streaks
and scores from it. A second client, or a rewrite, inherits the full history.

---

## Integration with dsview — MCP only

One channel: the MCP server at **`https://dsview-mcp.lab.sspcloud.fr`**
(`kubernetes/dsview_deployment.yaml`, `dsview-mcp-service` → port 8000; streamable HTTP,
`stateless_http=True`, `json_response=True`). No direct database access, no shared database, no
imported code, and nothing written back. `dsrecall` **adapts to whatever the server exposes** —
where a tool is missing, it works differently rather than reaching around the boundary.

Tools available (`dsview/mcp/server.py`):

```python
get_content(content_id: int) -> DsviewContent
query_content(already_read: bool | None = None, read_priority: int | None = None,
              relevance: int | None = None, source: str | None = None,
              date_ordering: Literal["asc","desc"] | None = None,
              limit: int = 20) -> DsviewContentList
get_connected_contents(topic_id: int) -> DsviewContentList
search_content(search_term: str, limit: int = 10,
               types: list[str] | None = None) -> DsviewContentList
get_connected_topics(content_id: int) -> DsviewTopicList
search_topic(search_term: str, limit: int = 10,
             types: list[str] | None = None) -> DsviewTopicList
explore_graph(node_id: int, node_kind: Literal["topic","content"],
              radius: int = 3) -> Neighborhood
get_nodes_ranking(metric: Literal["betweenness","degree","pagerank"] = "betweenness",
                  node_kind: Literal["content","topic"] | None = None,
                  limit: int = 20) -> NodeRanking
```

Resources: `config://content_types`, `config://topic_types`.

`DsviewContent` carries `id, link, upload_date, already_read, read_priority, relevance, source,
title, type, tags, summary`. `DsviewTopic` carries `id, type, name, description`.

### What each tool is for here

| Need | Call |
|---|---|
| Find syllabi to seed the curriculum | `search_content(..., types=["Course"])`, also `["Documentation"]` |
| Is this curriculum topic covered at all? | `search_topic(topic_name)` |
| Ground a question in something actually read | `get_connected_contents(topic_id)` → `summary` |
| What was read recently | `query_content(date_ordering="desc", limit=...)`, filter on `upload_date` |
| Reads whose takeaway is worth testing | `query_content(already_read=True, relevance=4\|5, date_ordering="desc")`, keep `type == "Blog post"` / `"Scientific article"` |
| Which curriculum topic a read belongs to | `get_connected_topics(content_id)` |

**Courses as curriculum sources** is the most valuable of these and was missing from earlier drafts:
a course or documentation entry in the knowledge base is a curated table of contents. It gives the
curriculum agent structured, already-vetted material to propose topics from, which is a much better
use of dsview than treating it as a coverage database.

### Working without what is not exposed

| Not exposed | How `dsrecall` manages |
|---|---|
| Topic embeddings (excluded from the MCP payload) | Unnecessary: `search_topic` performs dsview's hybrid FTS + vector search server-side and returns ranked topics, which is the capability the vectors would have provided. `dsrecall` embeds its *own* curriculum topics, which is what it actually needs them for: deduping agent proposals, and finding the adjacent method a `neighbour_true` distractor is built from. |
| Entity-resolution history (`ercomparison`) | Not needed: distractors are statements, not topic names, so a catalogue of confusable topic pairs has nothing to attach to. See the note below. |
| Date-range content queries | Page `query_content(date_ordering="desc")` and filter on `upload_date` client-side. |
| Resolving a topic by id | Not needed: `dsrecall` stores no dsview ids (see *Data model*). Topic identity travels as a name from `search_topic`, provenance as a URL. |
| Hourly staleness of search and ranking indexes | Irrelevant: only the offline batch talks to dsview. The serving path never does. |

On entity resolution: dsview's ER log contains topic pairs judged similar-but-distinct, which an
earlier draft wanted as ready-made "which of these is X" items. That item shape is now out of scope
entirely (see *Question generation*), so the log has nothing to contribute. The residual confusion
value survives in a better form: a property that is genuinely true of a *neighbouring* method makes
an excellent wrong option for a question about this one, which tests the same confusion through
substance instead of through naming.

### Deliberately no stored coupling

`dsrecall` stores **no dsview identifiers at all** — not topic ids, not content ids. Where a
question was grounded in something read, provenance is kept as the **source URL and title**, which
stay meaningful if dsview is offline, re-ingested with new ids, or eventually replaced. There is no
mapping table between the two topic spaces: coverage is a question asked at generation time, not a
relation to maintain.

---

## Data model

- **`curriculumtopic`** — `id`, `name`, `area`, `kind` (`fundamental` | `frontier`), `description`,
  `embedding`, `status` (`proposed` | `accepted` | `rejected`), `review_date`.
- **`curriculumsource`** — `id`, `curriculum_topic_id`, `url`, `title`, `note`, `origin`
  (`web` | `dsview`). Provenance is what makes an agent-proposed curriculum trustable; without it,
  accepting a topic is an act of faith. `origin="dsview"` marks a topic proposed from a course or
  documentation entry in the knowledge base.
- **`quizquestion`** — `id`, `curriculum_topic_id`, `archetype` (`mechanism` | `application` |
  `pitfall` | `tradeoff` | `takeaway`), `question`, `explanation`, `source_url`, `source_title`
  (both nullable, set for grounded items), `generation_date`, `retired`.
- **`quizchoice`** — `id`, `question_id`, `text`, `correct`, `strategy` (`perturbed` |
  `neighbour_true` | `misconception`, null on the key). Recording how each distractor was built is
  what lets evaluation compare strategies rather than guess.
- **`quizanswer`** — `id`, `question_id`, `choice_id`, `answer_date`, `flagged`.
- **`questionlabel`** — human verdicts, multi-label; the judge's reference data. Carries its own
  `split` column (`eval` | `test`), assigned once at insert and never recomputed.

**No topic-state table, no stored streaks.** Freshness, accuracy and streak are derived by SQL over
`quizanswer`. Stored state is a second source of truth that can drift, for an aggregation that is
free at this scale. Materialise only if the selection query actually becomes slow.

On the split: dsview learned this the hard way (`docs/adr/0001-...`) — a split recomputed from row
position silently reshuffles as rows are added, invalidating every past comparison. Storing the
assignment at insert is the simpler fix when there is no reason not to.

## Agents (offline)

- **Curriculum agent.** Proposes canonical topics from two source families: web research (syllabi,
  textbook tables of contents, standard references) and dsview's own courses and documentation via
  `search_content(types=["Course"])`. Dedupes against the existing curriculum by embedding. Lands as
  `status="proposed"` behind a human accept/reject gate — an auto-accepted curriculum is an unbounded
  quality risk at the root of everything downstream.
- **Reference agent.** For an accepted topic, fetches grounding material so questions are written
  from a source rather than from parametric memory. The main defence against confidently mis-keyed
  items.
- **Writer + critic loop.** Writer drafts; the judge critiques; writer revises; accept or discard,
  bounded iterations. The judge does double duty — inline critic *and* offline evaluation metric.

## Question generation

### Options are statements, never topic names

The point of an item is *what is behind a topic* — how a method works, when it applies, how it
fails. That rules out one tempting shortcut: if the four options are topic names, the stem can only
ask "which of these is X", and the item tests vocabulary rather than understanding. An earlier draft
built exactly that, because drawing options from a pool of real topics is such a convenient way to
get plausible wrong answers.

So every option is a **proposition about the topic**: a mechanism, a consequence, a condition, a
failure mode. One is the **key** (true, and true *here*); the other three are **distractors** —
statements that are wrong, but wrong in a way worth catching.

### Where distractors come from

Distractor quality *is* item quality: if the wrong options are obviously wrong, the key is
identifiable without knowing anything. Free-writing them is the known weak point, so each one is
built by an explicit strategy, recorded on the row (`quizchoice.strategy`) so evaluation can tell
which strategies produce good items:

- **`perturbed`** — take a true statement from the reference material and change exactly one thing:
  the direction of an effect, the condition it holds under, the quantity that moves. Tempting
  because it is almost right.
- **`neighbour_true`** — a property that genuinely holds for an adjacent method, but not for this
  one. This is where the confusion between close concepts gets tested, without ever asking the
  reader to match a name to a definition.
- **`misconception`** — a documented common mistake, which the reference agent looks for explicitly
  ("common pitfalls", "frequently misunderstood"). The strongest distractor of the three, because it
  is what you would actually answer if rusty.

### Archetypes

| Archetype | Asks | Example shape |
|---|---|---|
| `mechanism` | how it works internally, what a component actually does | "What does the causal mask in a decoder block actually prevent?" |
| `application` | when it fits, which approach suits a described situation | "500 labelled rows, 200 features, heavy class imbalance — which approach, and why?" |
| `pitfall` | how it breaks, what invalidates a result | "What is wrong with tuning the decision threshold on the same folds used for model selection?" |
| `tradeoff` | why choose this over a near alternative, in a stated context | "Why prefer this over the obvious alternative when interpretability is a hard requirement?" |
| `takeaway` | the methodological conclusion a specific read argued | see below |

The model writes the stem, the key, the distractors and the explanation; the **strategies** above
constrain the distractors, and the reference material grounds the key.

One mechanical consequence of statement-options: they are long, and in badly written multiple choice
the longest option is the answer. Options must be normalised for length and grammatical shape, and
`cue_leaking` is the judge's job to catch (see *Evaluation*).

### `takeaway` — consolidating a specific read

The point is not to re-summarise something read; a summary played back is worth nothing. It is worth
asking about a read only when that read **drew a methodological conclusion** — a blog post arguing
that some approach beats another under given conditions, a paper's actual finding. The question then
tests whether the *conclusion* stuck.

This makes `takeaway` a **filtered** archetype, not one that can be generated on demand:

1. List candidates: `query_content(already_read=True, relevance=4|5, date_ordering="desc")`, keeping
   `type` in `Blog post` / `Scientific article` — high self-rated relevance is the cheapest available
   proxy for "had a point worth keeping".
2. Read each `summary` and keep only those carrying an extractable methodological claim. Many will
   not; that is expected, and the ones that do not are simply skipped rather than turned into
   summary-recall questions.
3. Distractors are alternative conclusions: the opposite finding, a stronger version than was
   actually claimed, or a claim that is true in general but not what this piece argued.

Because dsview persists summaries rather than raw content, a claim has to survive summarisation to
be usable. That bounds the yield of this archetype — it is a filter on which reads qualify, not a
cap on question depth for the other four.

Note what is *not* used here: topics linked to the same content. The graph is sparse enough that
co-occurrence in one read implies little, so it is a poor source of anything.

## Daily selection policy

Five items, mix configurable:

- **2 cold fundamentals** — `kind="fundamental"`, longest since asked or previously wrong.
- **1 `takeaway`** — the conclusion of something read recently, when a qualifying read exists;
  otherwise fall back to a curriculum item, since this archetype is filtered and can come up empty.
- **1 uncovered** — an accepted curriculum topic dsview has nothing on.
- **1 new** — never asked.

A readable policy, not a learned scheduler: when a session feels wrong, the reason should be one SQL
query away. FSRS-style intervals are a later refinement, once there is answer history.

## Gamification

Stated as the most important aspect, so treated as load-bearing:

- **Speed above all.** Pre-generated bank, no LLM at answer time, five items, immediate feedback.
  Under a minute, never a spinner.
- **Streak with a grace/freeze mechanic.** Streak loss is the dominant churn cause, and an
  unforgiving streak punishes exactly the busy weeks when the habit is most fragile.
- **Explanation on every answer**, generated offline, so a wrong answer teaches rather than scores.
- **Progress against the curriculum**, per area — a real skill tree, uncovered topics visible as
  unfilled branches. Stronger than an XP number because the structure means something.
- **A consistent daily trigger** — email until a PWA push earns its keep.
- **Weekly recap** — streak, accuracy by area, coldest topics.

Explicitly rejected: leaderboards and social comparison (single user), currencies with nothing to
buy.

Honest risk: solo gamification decays once novelty passes. What survives is the streak, the visible
coverage map, and near-zero friction. Cosmetic layers are not worth early investment.

## Evaluation — an MLflow judge aligned on labelled questions

Question quality is the main risk and is not measurable by plumbing tests. The judge is built with
MLflow's GenAI tooling rather than hand-rolled, which is also the point: this is the part of the
stack most worth learning.

### Failure taxonomy (multi-label)

A question can fail several ways at once, so verdicts are checkboxes, not one enum: `mis_keyed`,
`ambiguous`, `cue_leaking` (key guessable from phrasing, length, grammatical agreement, "all of the
above"), `weak_distractors`, `off_level`, `false_premise`.

### Reference data, including cold start

- **Hand labelling** — for MCQ this is read-and-tick, so a seed of 100–200 is an afternoon. Worth
  trying MLflow's own review/labelling surface before building a form.
- **Synthetic negatives** — corrupt known-good items deterministically: swap the key, replace
  distractors with far-away topics, append "all of the above". Manufactures labelled failures for
  `mis_keyed`, `weak_distractors` and `cue_leaking` at no labelling cost, which is what makes a
  judge trainable before the human set is large.
- **Production signals** — `flagged` items are labelled negatives for free; items answered correctly
  every time are uninformative; an item whose distractor beats its key is probably mis-keyed.

### The judge

One scorer per failure mode, evaluated over a stored dataset with `mlflow.genai.evaluate`, and
**aligned against the human labels** using MLflow's judge-alignment support rather than a
hand-tuned prompt. Judge prompts live in the **MLflow prompt registry**, versioned, so a scoring run
records which prompt version produced it. Everything logs to MLflow runs as usual, so judge versions
are comparable over time.

This replaces the DSPy-in-a-notebook approach dsview uses. Fine-tuning a small model stays the
fallback if alignment plateaus, and would need an order of magnitude more labels.

### Meta-evaluation, and the circularity trap

A judge is worthless unmeasured: score **agreement with human labels** per failure mode (F1, and
Cohen's κ for the subjective ones) on the held-out `test` split. Only past acceptable agreement does
the judge become a generation gate.

**The trap:** once the judge filters generation *and* measures it, the metric grades its own
filtering and will look excellent regardless of reality. Mitigations — keep a human-labelled
held-out set as the real measure, re-label a fresh sample periodically, and track judge agreement
over time as a first-class metric. This is the kind of evaluation that rots silently.

## Phasing

1. New repo skeleton: `pydantic-ai` agent that reaches the dsview MCP server, MLflow tracking
   reachable, learning schema created. Pin down the actual MLflow GenAI API surface here.
2. Curriculum agent + human review gate; a first accepted curriculum for two or three areas,
   seeded partly from dsview courses.
3. Question generation for those areas; hand-inspect the bank.
4. Serving routes + the smallest client that runs a daily session.
5. Labelling, judge, alignment, meta-evaluation; then wire the judge into the writer/critic loop.
6. Per-area coverage view and weekly recap. Widen the curriculum.

## Deferred

- **Reading recommendations.** Partly free already: an uncovered curriculum topic *is* a reading
  suggestion. The complementary signal worth recording — dsview's `extraction.extractionlink` holds
  links extracted from content that was read but never ingested; a link referenced across several
  reads and never followed is a strong recommendation. It is not exposed over MCP and belongs in
  dsview's own weekly digest rather than here.
- **Spaced repetition proper** (FSRS per topic) — needs answer history first.
- **Free-text answers with an LLM judge** — only if the goal shifts from working knowledge to
  depth.
