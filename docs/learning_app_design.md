# Learning app — design proposal

*Status: proposed · date: 2026-09-19 · author: tdurouchoux*

**Scope.** This document specifies a new system that lives in **its own repository** (working name
`dsrecall`, placeholder). It is kept in the dsview repo because that is the repo that exists today,
and because most of it is an integration contract *against* dsview. Every path written
`dsview/...`, and every table and tool named below, refers to the **`tdurouchoux/dsview`**
repository and its deployment — not to the new repo. Paths were verified against dsview `2.7.0`.

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
and weighted selection by `dsview/graph/graph_analysis.py::pagerank`. That is backwards twice over:
it can only ask about topics that happen to appear in past reading, and it then *prioritises the
most-read ones* — ranking by what is already freshest. Recorded because the failure is tempting:
the graph is right there, well-structured, already deduplicated by entity resolution.

So the system carries **two distinct topic spaces**:

| | `curriculum` topics (new repo) | `extraction.extractiontopic` (dsview) |
|---|---|---|
| Answers | what a data scientist should have | what I was exposed to |
| Sourced from | syllabi, textbook tables of contents, standard references, web research | ingested content |
| Owns | quiz scope | personalisation and grounding |

and a **mapping** between them, informative in both directions:

- curriculum topic with no kb coverage → a blind spot the reading list never touched. A prime quiz
  target, and why coverage-gap analysis comes back almost free.
- kb topic with no curriculum entry → frontier or tool-specific material. Fine to have read, not
  necessarily worth committing to memory.
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

### Three components — decided

1. **dsview — unchanged.** Stays the landscape map. Consumed, never extended for this purpose
   (two small optional additions are listed under *Gaps*, and neither is required to start).
2. **Learning backend — new repo.** Curriculum, agents, question bank, answer log, HTTP API. Own
   Postgres database. Imports `dsview` as a library for the LLM plumbing; consults the knowledge
   base over MCP and read-only SQL.
3. **Client — new repo (or a subdirectory of 2).** Phone-first web app: presentation and
   gamification only, no durable state beyond a session token.

Consuming the kb from outside, rather than growing inside dsview, is what structurally enforces the
previous section: the graph is one source an agent may consult, and cannot quietly become the
curriculum. It also keeps a web-research agent and a canonical DS syllabus out of a repo whose
mission is the reading landscape.

Accepted costs, stated plainly: a git-dependency version coupling on dsview; no shared Postgres
transaction between the two stores; and the new repo needs its own `CONF_DIR` mirroring dsview's
config shape (see *As a library*).

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

## Integration with dsview

Three channels, each for what it is actually good at.

| Need | Channel |
|---|---|
| Open-ended exploration by an agent | MCP server |
| Bulk reads, embeddings, date ranges, ER history | read-only Postgres (`dsview_ro`) |
| `LLMModel`, config, prompts, eval split helpers | `dsview` as a uv git dependency |

### 1. MCP server — the agent's view

Deployed at **`https://dsview-mcp.lab.sspcloud.fr`** (`kubernetes/dsview_deployment.yaml`,
`dsview-mcp-service` → port 8000), streamable HTTP, `stateless_http=True`, `json_response=True`
(`dsview/mcp/server.py`). Note that unlike the ingest, dashboard and labels ingresses, the MCP
ingress carries **no `nginx.ingress.kubernetes.io/auth-type: basic` annotation** — there is nothing
for the backend to authenticate with, and equally nothing protecting the endpoint. Worth settling
before a second consumer depends on it.

Tools available (exact signatures):

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
title, type, tags, summary` — enough to ground a consolidation item without touching SQL.
`DsviewTopic` carries `id, type, name, description`.

Two behaviours to design around:

- **Indexes and the graph are cached for an hour** (`INDEX_TTL = 3_600`, `lru_cache` over
  `get_extraction_index` / `get_topics_index` / `get_graph`), so `search_*` and
  `get_nodes_ranking` lag recent ingests. Irrelevant for a weekly batch; would matter if the
  serving path ever called MCP, which it must not.
- **`query_content` has no date-range filter.** For "ingested in the last N weeks", either page
  with `date_ordering="desc"` and filter client-side, or use SQL (dsview's own
  `get_content_by_date_range` is not exposed as a tool).

### 2. Read-only Postgres — the batch view

User **`dsview_ro`**, created by `kubernetes/create_readonly_user.sql`: `SELECT` on all tables in
`content`, `extraction` and `labels`, including future tables. Connection parameters follow
dsview's own (`dsview/config.py::PostgresConfig`): `POSTGRES_HOST`, `POSTGRES_USER`,
`POSTGRES_PASSWORD`, port `5432`, database `defaultdb` (`config/storage.yaml`).

Three things the MCP surface cannot give and SQL must:

- **Topic embeddings.** `search_topic` builds its payload with
  `model_dump(exclude=["embedding"])`, so `extraction.extractiontopic.embedding` is reachable only
  in SQL. Both the curriculum↔kb mapping and embedding-neighbour distractors depend on it.
- **Entity-resolution history.** `extraction.ercomparison` is not exposed over MCP at all. Note its
  shape: it stores `name_1/type_1/description_1`, `name_2/...`, `merge_topic`, `merge_name` — **names,
  not topic ids**, with no foreign key to `extractiontopic`. Joining back to topics goes through the
  name, which the unique `ix_extraction_topic_name_lower` index makes sound.
- **Bulk and date-ranged reads** of `content.inputcontent` and `extraction.extractionresult`
  without paging a tool call at a time.

Tables consumed, for reference:

| Table | Used for |
|---|---|
| `content.inputcontent` | `upload_date`, `already_read`, `relevance` — consolidation targeting |
| `extraction.extractionresult` | `title`, `content_type`, `summary` — grounding consolidation items |
| `extraction.extractiontopic` | `name`, `type`, `description`, `embedding` — mapping and distractors |
| `extraction.contenttopicrelation` | topic ↔ content edges |
| `extraction.extractiontag` | the 27-tag taxonomy, for area alignment |
| `extraction.ercomparison` | confusable pairs → discrimination items |
| `extraction.extractionlink` | deferred; see *Deferred* |

**The learning store is a separate database.** References to dsview rows are stored as **plain
integers, never foreign keys**, so the two can live in separate Postgres instances without a
migration. The backend opens its own read-write engine for its own database and a separate
read-only engine for dsview's — it does **not** import `dsview.db.engine`, which resolves a single
`POSTGRES_*` triple.

Nothing in the new repo ever writes to `content`, `extraction` or `labels`. `dsview_ro` enforces
that; the design should not rely on the grant alone.

### 3. dsview as a library

dsview is not published to PyPI, so a uv git dependency pinned to a tag (`v*.*.*`, currently
`2.7.0`):

```toml
dependencies = ["dsview @ git+https://github.com/tdurouchoux/dsview@v2.7.0"]
```

Worth importing:

- `dsview.model_utils` — `LLMModel`, `get_model_provider`: the prompt-file + structured-output
  pattern, multi-provider support and tenacity retries.
- `dsview.config` — `ModelType`, `ModelConfig`, `lazy`, `lazy_model_config`, `load_model_config`,
  `setup_logger`.
- `dsview.db.query.query_labels::assign_rows` — the hash-based deterministic split. Reuse the
  function, fix this repo's own random state and ratios at creation, and never change them
  afterwards (same discipline as dsview's rule, on a separate dataset).
- `dsview.notification.email_sender::send_email` — the daily nudge, until a PWA push earns its keep.

Three import hazards, all real:

- **`dsview.config` reads `CONF_DIR` and `PROMPT_DIR`** on first attribute access, and
  `load_model_config` expects a `model.yaml` of dsview's shape. The new repo therefore needs its
  **own** config directory with its own `model.yaml` (and `logging.yaml`), not dsview's. This is the
  main ongoing cost of the library dependency: a schema change in dsview's config is a breaking
  change here.
- **Do not import `dsview.extraction.models.*`.** `topics_extraction.py` and
  `description_generation.py` build `TopicType` / `TagsType` / `ContentType` enums from config **at
  import time** (pydantic needs them at class definition), so importing them requires a valid
  `CONF_DIR` immediately. Importing `dsview.model_utils` alone stays lazy.
- **Do not import `dsview.db.engine`** — see above; build engines locally.

Provider keys follow whatever this repo's `model.yaml` declares (`MISTRAL_API_KEY` if it mirrors
dsview's default). `.env` at the repo root is auto-loaded by `python-dotenv`, as in dsview.

### Gaps in dsview, and whether they matter

Neither of these blocks a start; both are cheap if wanted later.

- **No tool resolves a topic by id.** `explore_graph`'s own docstring refers to a `get_topic` tool
  that does not exist, and `get_nodes_ranking` / `explore_graph` return ids with only a label. The
  mapping table stores `dsview_topic_id`, so hydrating one is SQL today. A `get_topic(topic_id)`
  tool in `dsview/mcp/server.py` would be a handful of lines and would make the agent path
  self-sufficient.
- **`ercomparison` is invisible to agents.** Only matters if discrimination-item selection moves
  from the batch job into an agent. Leave it in SQL until then.

Both are *requests on dsview*, to be raised there as their own change, not reasons to fork or
vendor anything.

### Conventions to inherit

The new repo should start with its own `CLAUDE.md` seeded from dsview's: lazy config, `session`
last among required parameters, `uvx ruff format . && uvx ruff check .`, hermetic tests by default
with `@pytest.mark.llm` / `@pytest.mark.network` for the rest, ADRs in `docs/adr/` in MADR format,
MLflow run names that say what changed, and no prompt change merged without an eval.

---

## Data model (learning backend)

- **`curriculumtopic`** — `id`, `name`, `area`, `kind` (`fundamental` | `frontier`), `description`,
  `embedding`, `status` (`proposed` | `accepted` | `rejected`), `review_date`.
- **`curriculumsource`** — `id`, `curriculum_topic_id`, `url`, `note`. Provenance is what makes an
  agent-proposed curriculum trustable; without it, accepting a topic is an act of faith.
- **`kbtopiclink`** — `curriculum_topic_id`, `dsview_topic_id` (loose int), `similarity`, `method`.
- **`quizquestion`** — `id`, `curriculum_topic_id`, `dsview_content_id` (nullable loose int, set for
  grounded consolidation items), `archetype`, `question`, `explanation`, `generation_date`,
  `retired`.
- **`quizchoice`** — `id`, `question_id`, `text`, `correct`, `distractor_topic_id`.
- **`quizanswer`** — `id`, `question_id`, `choice_id`, `answer_date`, `flagged`.
- **`questionlabel`** — human verdicts, multi-label; the judge's training data.

**No topic-state table, no stored streaks.** Freshness, accuracy and streak are derived by SQL over
`quizanswer`. Stored state is a second source of truth that can drift, for an aggregation that is
free at this scale. Materialise only if the selection query actually becomes slow.

## Agents (offline)

- **Curriculum agent.** Web-searches syllabi, textbook tables of contents and standard references;
  proposes canonical topics with provenance; dedupes against the existing curriculum by embedding.
  Lands as `status="proposed"` behind a human accept/reject gate — an auto-accepted curriculum is an
  unbounded quality risk at the root of everything downstream.
- **Reference agent.** For an accepted topic, fetches grounding material so questions are written
  from a source rather than from parametric memory. The main defence against confidently mis-keyed
  items.
- **Writer + critic loop.** Writer drafts; the judge critiques; writer revises; accept or discard,
  bounded iterations. The judge does double duty — inline critic *and* offline eval metric.

Deliberately **not** an agent: the curriculum↔kb mapping. Embedding neighbours over
`extraction.extractiontopic.embedding` plus an `ERClassifier`-style pairwise confirmation already
solve it, and that task is tuned and evaluated (dsview `docs/adr/0001-entity-resolution-classifier-model-and-prompt.md`).

**Framework: `pydantic-ai`.** Logfire is by the same team and already instrumented across dsview,
so multi-agent traces come free; it is pydantic-native, matching the existing structured-output
style; and it speaks MCP, which is how the kb is consulted. Nothing LangGraph-scale is warranted
for a single-user system. A hand-rolled loop over `LLMModel` plus tool calls is the fallback.

Cost: agentic generation is far more expensive per item than one structured call. It is also
offline, batched, and amortised over weeks of serving — the point of separating the tempos.

## Question generation

| Archetype | Grounded in | Distractors from |
|---|---|---|
| `recognition` | curriculum topic + fetched reference | sibling curriculum topics in the same area |
| `discrimination` | a confusable topic pair | the pair's counterpart |
| `application` | topic + typical use | siblings in the same area |
| `consolidation` | a summary of content actually read | other topics linked to that content |

### Distractors are selected, not invented

The known failure mode of LLM-written multiple choice is implausible options: the key is
identifiable without knowing anything. Candidates come from structure instead — embedding
neighbours within the curriculum; for consolidation items, the kb's own topic neighbourhood; and
**`extraction.ercomparison` rows with `merge_topic = false`**, which are confusable-but-distinct by
construction. The model writes the stem and the explanation; the candidate set comes from data.

### Grounding depth

dsview persists summaries, not raw content (full text exists only in `labels.labelledcontent`, and
only for labelled rows). For consolidation items that caps depth at what a summary supports —
acceptable at recognition level. Curriculum items are unaffected: the reference agent fetches their
material.

## Daily selection policy

Five items, mix configurable in YAML:

- **2 cold fundamentals** — `kind="fundamental"`, longest since asked or previously wrong.
- **1 consolidation** — curriculum topic mapped to content ingested in the last few weeks.
- **1 blind spot** — accepted curriculum topic with no `kbtopiclink` row.
- **1 new** — never asked.

A readable policy, not a learned scheduler: when a session feels wrong, the reason should be one
SQL query away. Note where pagerank survives — *ordering within consolidation items only*, among
topics actually read. It never defines scope. FSRS-style intervals are a later refinement.

## Gamification

Stated as the most important aspect, so treated as load-bearing:

- **Speed above all.** Pre-generated bank, no LLM at answer time, five items, immediate feedback.
  Under a minute, never a spinner.
- **Streak with a grace/freeze mechanic.** Streak loss is the dominant churn cause, and an
  unforgiving streak punishes exactly the busy weeks when the habit is most fragile.
- **Explanation on every answer**, generated offline, so a wrong answer teaches rather than scores.
- **Progress against the curriculum**, per area — a real skill tree, blind spots visible as
  unfilled branches. Stronger than an XP number because the structure means something.
- **A consistent daily trigger**, via `dsview.notification.email_sender`.
- **Weekly recap** — streak, accuracy by area, coldest topics.

Explicitly rejected: leaderboards and social comparison (single user), currencies with nothing to
buy.

Honest risk: solo gamification decays once novelty passes. What survives is the streak, the visible
coverage map, and near-zero friction. Cosmetic layers are not worth early investment.

## Evaluation — a judge trained on labelled questions

### Failure taxonomy (multi-label)

A question can fail several ways at once, so verdicts are checkboxes, not one enum: `mis_keyed`,
`ambiguous`, `cue_leaking` (key guessable from phrasing, length, grammatical agreement, "all of the
above"), `weak_distractors`, `off_level`, `false_premise`.

### Labelled data, including cold start

- **Hand labelling** in a small Streamlit form — for MCQ this is read-and-tick, so a seed of
  100–200 is an afternoon.
- **Synthetic negatives** — corrupt known-good items deterministically: swap the key, replace
  distractors with far-away topics, append "all of the above". Manufactures labelled failures for
  `mis_keyed`, `weak_distractors` and `cue_leaking` at no labelling cost, which is what makes a
  judge trainable before the human set is large.
- **Production signals** — `flagged` items are labelled negatives for free; items answered
  correctly every time are uninformative; an item whose distractor beats its key is probably
  mis-keyed.

### The judge itself

An `LLMModel` subclass with one structured output field per failure mode. "Trained" means
**prompt-optimised against the labelled set**, following existing practice — DSPy optimisation of
extraction prompts already lives in dsview's `notebooks/` (`dspy_entity_resolution_opt.py`,
`dspy_content_description_opt.py`). Fine-tuning a small model is the fallback if an optimised
prompt plateaus, and needs an order of magnitude more labels.

### Meta-evaluation, and the circularity trap

A judge is worthless unmeasured: an `evaluation/quiz_judge.py` scores **agreement with human
labels** per failure mode (F1, and Cohen's κ for the subjective ones) on the `eval` split, reusing
`assign_rows`, logged to MLflow. Only past acceptable agreement does the judge become a generation
gate.

**The trap:** once the judge filters generation *and* measures it, the metric grades its own
filtering and will look excellent regardless of reality. Mitigations — keep a human-labelled
held-out set as the real measure, re-label a fresh sample periodically, and track judge agreement
over time as a first-class metric. This is the kind of eval that rots silently.

## Phasing

1. New repo skeleton: `CLAUDE.md`, config dir, pinned dsview dependency, learning schema, and a
   smoke test that reaches both the MCP endpoint and `dsview_ro`.
2. Curriculum agent + human review gate; a first accepted curriculum for two or three areas only.
3. Mapping to the kb; question generation for those areas; hand-inspect the bank.
4. Serving routes + the smallest client that runs a daily session.
5. Labelling form, judge, meta-eval; then wire the judge into the writer/critic loop.
6. Per-area coverage view and weekly recap. Widen the curriculum.

## Deferred

- **Reading recommendations.** Partly free now: a curriculum blind spot *is* a reading suggestion.
  The complementary signal worth recording — **`extraction.extractionlink`** holds links extracted
  from content that was read but never ingested; a link referenced across several reads and never
  followed is a strong recommendation from one SQL query, no LLM. Belongs in dsview's weekly digest
  before it justifies any UI here.
- **Spaced repetition proper** (FSRS per topic) — needs answer history first.
- **Free-text answers with an LLM judge** — only if the goal shifts from recognition to depth.
