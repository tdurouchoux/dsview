# DSView — Repository Review

*Review date: 2026-07-11 · branch `td_evaluation_makeover` (currently even with `main`)*

## What this project is

DSView is a personal data-science knowledge-management platform. Content (blog posts, papers,
repos, docs) is ingested from URLs, run through an LLM extraction pipeline (summary, description,
topics, relevant links), deduplicated via LLM-based entity resolution, stored in PostgreSQL, and
rendered as an Obsidian vault plus an interactive knowledge graph. Four deployable services share
the codebase: a FastAPI ingestion API, an MCP server exposing the knowledge graph to LLM clients,
a Marimo dashboard suite, and a Streamlit labelling interface. Deployment targets Kubernetes via
per-role Dockerfiles published by a tag-triggered GitHub Actions workflow.

The overall shape is good: for a solo project it has an unusually complete lifecycle — ingestion,
extraction, storage, retrieval, evaluation with labelled data and MLflow tracking, and multiple
consumption surfaces. The MCP server in particular is a genuinely interesting design: it treats
the content/topic graph as the navigation structure (search as entry point, neighborhood
exploration, centrality-based ranking for "what should I read next").

## Architecture map

| Area | Files | Notes |
|---|---|---|
| Config | [config.py](dsview/config.py) | OmegaConf structured configs + env interpolation; per-task model configs (`ModelType`) |
| Ingestion | [ingest_source.py](dsview/ingest_source.py), [extraction/content_loader.py](dsview/extraction/content_loader.py) | URL cleaning, Medium→readmedium redirect, HTML/PDF loaders |
| Extraction | [extraction/content_extraction.py](dsview/extraction/content_extraction.py), [extraction/models/](dsview/extraction/models/) | Parallel async LLM calls; declarative `LLMModel` subclasses (prompt file + pydantic output class) |
| LLM abstraction | [model_utils/](dsview/model_utils/) | Provider-agnostic (Mistral/OpenAI/Anthropic/Ollama), tenacity retries, sync+async |
| Storage | [db/schemas/](dsview/db/schemas/), [db/ingest/](dsview/db/ingest/), [db/query/](dsview/db/query/) | SQLModel over Postgres, separate schemas (`content`, `extraction`, labels) |
| Search index | [db/query/query_utils.py](dsview/db/query/query_utils.py) | In-memory DuckDB attached to Postgres; FTS (BM25) + HNSW vector index, RRF fusion — nice, lightweight hybrid-search design |
| Graph | [graph/build_graph.py](dsview/graph/build_graph.py), [graph/graph_analysis.py](dsview/graph/graph_analysis.py) | igraph bipartite content↔topic graph; BFS neighborhoods, centrality rankings |
| MCP server | [mcp/server.py](dsview/mcp/server.py) | 8 tools + resources, stateless HTTP, Accept-header fix middleware |
| Interfaces | [interface/dashboard/](dsview/interface/dashboard/), [interface/labelling/](dsview/interface/labelling/) | Marimo dashboards; Streamlit labelling for eval datasets |
| Evaluation | [evaluation/](dsview/evaluation/) | Per-task eval (topics, links, description, ER) against labelled data, MLflow metrics |

## Strengths

- **Clean LLM-task pattern.** `LLMModel` subclasses ([llm_model.py](dsview/model_utils/llm_model.py))
  declare a prompt file, format params, and a pydantic structured-output class. Adding a new
  extraction task is ~10 lines ([topics_extraction.py](dsview/extraction/models/topics_extraction.py)).
  Config-driven per-task model selection (`ModelType`) is a good cost/quality lever.
- **Hybrid search done simply.** The DuckDB index (`query_with_rff`) gets BM25 + HNSW + reciprocal
  rank fusion without running a dedicated search service. Attaching Postgres from an in-memory
  DuckDB is a clever trick for a single-user system.
- **Evaluation is a first-class concern.** Labelled datasets live in the DB with eval/test splits,
  the labelling UI feeds them, and notebooks exist for prompt optimization (incl. DSPy experiments).
  Most hobby projects never get here.
- **The ER audit trail** (`ERComparison` table storing every LLM merge decision with FTS/VSS scores)
  is exactly what you want for later evaluating/replacing the ER classifier.

## Issues found

### Correctness

1. **MCP index/graph "hourly refresh" likely never happens.**
   [server.py:71-89](dsview/mcp/server.py#L71-L89) computes `ttl_hash` once inside `app_lifespan`,
   which runs once per server process. The `@cache`d `get_extraction_index`/`get_topics_index`/
   `get_graph` are therefore called with a single hash value for the process lifetime — the tool
   docstrings claim "indexes are updated every hour", but nothing recomputes the hash per request.
   The TTL trick only works if `get_ttl_hash` is called on every access.

2. **One long-lived DB `Session` shared by all MCP requests** ([server.py:75](dsview/mcp/server.py#L75)).
   SQLAlchemy sessions are not thread/concurrency-safe, and a failed query poisons the session for
   every subsequent request until restart. A session-per-request dependency (as [api.py](dsview/api.py#L37-L39)
   does) would be safer. The same concern is flagged in the extraction code itself:
   `topics_er` gathers concurrent tasks against one session with a `# ! I am not sure but I may need thread safe session`
   comment ([content_extraction.py:204](dsview/extraction/content_extraction.py#L204)).

3. **Eval bug in topic matching:** [evaluation/topics_extraction.py:120-122](dsview/evaluation/topics_extraction.py#L120-L122)
   calls `find_close_topic(simple_er, topic, row["name"][:1], row["type"][:1])` — only the *first*
   labelled topic is ever considered for a fuzzy match. Recall/precision numbers are systematically
   understated. This looks like leftover debugging (`[:1]`), along with `print("ONE TOPIC MATCHED !!!")`.
   Given the branch name (`td_evaluation_makeover`), this is presumably part of what's being reworked.

4. **Degenerate eval scores:** when the extractor returns zero topics, precision and average
   precision are set to 1 ([topics_extraction.py:136-139](dsview/evaluation/topics_extraction.py#L136-L139)).
   An empty prediction scoring perfect precision will inflate means; most IR conventions would score 0
   or exclude the row.

5. **`get_connected_topics` takes `content_id: str`** while every other tool uses `int`
   ([server.py:363](dsview/mcp/server.py#L363)), and neither it nor `get_connected_contents` handles
   a missing row (`db_session.get(...)` returning `None` → `AttributeError` instead of a clean tool error,
   unlike `get_content` which raises a proper `ValueError`).

6. **Backup/restore drops primary keys.** `backup()` removes the `id` column
   ([cli.py:59](dsview/cli.py#L59)) before writing parquet. On restore, `InputContent` rows get fresh
   auto-increment ids — but `ExtractionResult.content_id` (not part of the backup set) references the
   old ids. A restore into a DB that still has extraction data, or any drift in insert order, silently
   re-associates content with the wrong extractions.

### Security / robustness

7. **SQL injection surface in the DuckDB index layer.** Queries are built by f-string interpolation
   ([query_utils.py:141-151, 198-209](dsview/db/query/query_utils.py#L141-L151)), and the code
   acknowledges it (`# ! Not safe sql ingestion is possible`). Notably, the MCP tools pass
   client-supplied `types` lists straight into a `WHERE ... IN ('...')` clause
   ([server.py:324-326, 405](dsview/mcp/server.py#L324-L326)) and `_clean_input` only strips single
   quotes from the search term. For a single-user personal tool the blast radius is small, but the MCP
   server listens on `0.0.0.0` and is driven by an LLM — parameterized queries (`conn.execute(sql, params)`
   works in DuckDB) would close this cheaply.

8. **No auth on any surface.** The FastAPI ingest API, the MCP server, and the dashboards all bind
   `0.0.0.0` with no authentication; protection presumably relies entirely on the Kubernetes ingress.
   Worth a one-line statement of that assumption in the deployment docs.

9. **`GithubVault.url` embeds the GitHub token** in the remote URL ([config.py:118-123](dsview/config.py#L118-L123)).
   Fine as a mechanism, but make sure it never gets logged (e.g. by git error output surfaced in logs).

10. **`requests.get` without a timeout** ([content_loader.py:42](dsview/extraction/content_loader.py#L42)) —
    a hung server stalls the whole ingestion worker.

### Code quality / hygiene

- **Module-level side effects everywhere:** `load_dotenv()`, `setup_logger()`, `load_model_config()`
  run at import time in [api.py](dsview/api.py#L20-L24), [llm_model.py](dsview/model_utils/llm_model.py#L14-L16),
  [query_utils.py](dsview/db/query/query_utils.py#L10-L11), etc. This makes imports order-sensitive,
  requires `CONF_DIR` to be set before *any* dsview import, and is why the CLI resorts to function-level
  imports. Deferring config loading to first use (a cached accessor) would untangle this.
- **`ModelConfigurationError` is defined twice** with different signatures
  ([config.py:66](dsview/config.py#L66), [model_provider.py:40](dsview/model_utils/model_provider.py#L40)).
- **TODO density is high in core paths** — e.g. `content_extraction.py` opens with four TODOs including
  "fix this script this should not be a class". These are honest notes, but the load-bearing ones
  (session thread-safety, SQL safety) deserve issues rather than comments.
- **`tests/test_db.py` is empty**, and there are no tests for the DB layer, MCP tools, graph module, or
  evaluation logic — the places where the issues above live. Existing tests (config, loaders, LLM
  provider mocks, ER classification) are reasonable.
- **No CI for tests/lint** — the only workflow is Docker publish on tags. A cheap `uv sync && pytest && ruff check`
  workflow on PRs would have caught some of the drift below.
- **Dead code / commented-out blocks** in `server.py` (resources), `cli.py` (mlflow lines),
  `_check_model_config` never called.

### Documentation drift

The README is polished but promises more than the repo delivers, which is worse than a modest README:

- `docker-compose up -d`, `docker-compose.prod.yml` — **no compose file exists** in the repo.
- MIT `LICENSE` file referenced — **not present** (a public repo without a license is "all rights reserved").
- Mentions ROUGE-based description eval, "minimum 80% coverage", GitHub wiki/discussions, Docker Hub —
  aspirational at best.
- CLI docs mention commands with flags that don't match [cli.py](dsview/cli.py) (`ingest --priority`,
  no `dsview --version`).

## Suggested priorities

1. **Land the evaluation makeover** (this branch's purpose): fix the `[:1]` matching bug, decide the
   zero-prediction scoring convention, replace `print`s with logging, and consider concurrency across
   rows (evals are currently row-sequential with nested `asyncio.run`).
2. **MCP server hardening:** per-request session, actually-refreshing indexes (call `get_ttl_hash` inside
   each tool), parameterized DuckDB queries, `int` content_id + not-found handling.
3. **Trim the README to reality** and add a LICENSE (or delete the license section).
4. **Add a test/lint CI workflow**; fill in `test_db.py` with the backup/restore round-trip — it would
   have exposed issue #6.
5. Longer term: the module-level config loading is the biggest structural drag; making config lazy would
   simplify testing and make the four service entrypoints less fragile.

## Overall impression

This is a well-conceived personal platform with several genuinely good engineering ideas (hybrid
DuckDB search over Postgres, declarative LLM task classes, ER decision auditing, graph-native MCP
tools). Its weaknesses are the classic solo-project ones: shared mutable sessions across async
boundaries, f-string SQL, import-time side effects, an aspirational README, and evaluation code that
drifted during experimentation. None of it is architectural — the fixes above are mostly local and
the structure would support them without a rewrite.
