# CLAUDE.md

DSView is a personal data-science knowledge-management platform: content (blog posts, papers,
repos) is ingested from URLs, run through an LLM extraction pipeline (summary, description,
topics, links, entity resolution), stored in PostgreSQL, and exposed as an Obsidian vault, a
knowledge graph, an MCP server, Marimo dashboards, and a Streamlit labelling interface.

## Commands

```bash
uv sync                          # install (dev group included by default)
uv sync --group dashboard        # marimo dashboards extras
uv sync --group labels           # streamlit labelling extras

pytest tests/test_config.py      # safe: no network, no LLM calls
pytest                           # CAUTION: see "Testing" below before running the full suite

uvx ruff format . && uvx ruff check .   # formatting / linting (must be run via uvx)
dsview --help                    # CLI (ingest, rebuild, backup, regen-vault, ...)

# evaluation — costs real API money, see "Model evaluation" below
dsview evaluate --help                                        # topics, links, description, er
dsview evaluate topics --set-type eval --run-name NAME        # pipeline defaults
dsview evaluate topics CONFIG.yaml --set-type eval --run-name NAME
```

### Evaluation CLI

`dsview evaluate <task> [CONFIG.yaml]` runs one task's `evaluate()` inside an MLflow run
(`dsview/evaluation/cli.py`). `--set-type` defaults to `eval`; `--run-name` should always be
passed (see the naming rule below). Each run logs params, metrics, and an `eval_results.parquet`
artifact of per-row scores — **analysing that artifact afterwards costs nothing, so never re-run
an eval just to inspect its results.**

Runs go to one MLflow experiment per task ("Topics extraction", "Entity resolution", …), overridable
with `--experiment`. The code never calls `set_tracking_uri`, so the tracking backend is whatever
the environment resolves to — don't hard-code a store when querying past runs. Fetch a past run's
artifact with
`mlflow.artifacts.download_artifacts(run_id=..., artifact_path="eval_results.parquet")`. Note that a
failed or cancelled run still creates an MLflow run with the same `--run-name` but no artifact, so
resolve a name to a run id **newest-first** rather than assuming names are unique.

The optional YAML overrides the pipeline defaults; every key is optional and anything omitted
falls back to the default (`LLMModel` coalesces). Configs live in **`eval_configs/`** — not in a
scratchpad — with their variant prompt files alongside:

```yaml
model_config:                    # built as a full ModelConfig
  chat_model: mistral-small-2603
  embedding_model: mistral-embed
  provider: MISTRAL
  token_limit: 100000
  model_specific_config:         # spread into every provider call
    temperature: 0               # pin for any comparison: halves run-to-run noise
system_prompt_file: eval_configs/topics_exp3_system_prompt.txt
user_prompt_file: eval_configs/topics_exp2_user_prompt.txt
```

Two things that are easy to get wrong:

- **An overridden `user_prompt_file` still gets `DEFAULT_USER_PROMPT_ADD_FORMAT` applied.** A
  variant prompt must keep the placeholders the task injects (for topics: `{content}` **and**
  `{tags}`) and contain no other braces, or `.format()` raises at call time.
- `er` additionally takes `--sync` (bypass the provider batch API) and `--remove-descr`. Other
  tasks have no `--sync` flag; set `DSVIEW_DISABLE_NATIVE_BATCH=1` in the environment instead.

**Mistral's batch API is intermittently unreliable.** Evals default to it, and it periodically
fails a whole run with `"No batch job matches the given query."` (404) shortly after submitting —
observed several times in a single afternoon. It is transient, not a config error: a plain retry
of the identical command usually succeeds. If it keeps failing, fall back to the synchronous path
(`--sync` for `er`, `DSVIEW_DISABLE_NATIVE_BATCH=1` otherwise), which is slower and forgoes the
batch discount but is reliable. A failed run leaves a params-only MLflow run behind — see the
duplicate-run-name note above before fetching artifacts by run name.

## Configuration is lazy — keep it that way

Module-level config objects are **lazy proxies** (`lazy()` / `lazy_model_config()` in
`dsview/config.py`): they are typed as the real config class for IDE/type-checker purposes but
only read `CONF_DIR`, `PROMPT_DIR`, `POSTGRES_*` env vars and YAML files on first attribute
access. The DB engine is likewise created on first use (`dsview/db/__init__.py` module
`__getattr__`). Consequences:

- Importing `dsview` modules and running `dsview --help` needs **no environment**; config errors
  surface at first *use*, not at import.
- When adding a module-level config or provider global, declare it with `lazy(...)` — never call
  `load_*_config()` or `get_model_provider(...)` at module scope.
- **The one exception**: `extraction/models/topics_extraction.py` and `description_generation.py`
  build `TopicType` / `TagsType` / `ContentType` enums from config at import time (pydantic needs
  them at class definition), so importing the extraction models still requires a valid `CONF_DIR`.
  Don't import them at module level from low-level packages (see the deferred imports in
  `db/query/query_topic.py`).
- Runtime env requirements: `CONF_DIR`/`PROMPT_DIR` for any config/prompt use, `POSTGRES_*` for DB
  access, and the API key for whichever provider `config/model.yaml` declares (e.g.
  `MISTRAL_API_KEY`). `.env` at the repo root is auto-loaded via `python-dotenv`.

## Architecture map

| Area | Location |
|---|---|
| Config schemas (OmegaConf dataclasses) | `dsview/config.py` + `config/*.yaml` |
| LLM abstraction (multi-provider, retries) | `dsview/model_utils/` |
| Extraction tasks (one `LLMModel` subclass per task) | `dsview/extraction/models/` |
| Pipeline orchestration | `dsview/ingest_source.py`, `dsview/extraction/content_extraction.py` |
| DB schemas / queries (SQLModel, Postgres) | `dsview/db/` |
| Hybrid search (DuckDB FTS + HNSW + RRF) | `dsview/db/query/query_utils.py` |
| Knowledge graph (igraph) | `dsview/graph/` |
| MCP server | `dsview/mcp/server.py` |
| Evaluation | `dsview/evaluation/` |
| Labelling UI (feeds eval datasets) | `dsview/interface/labelling/` |
| Prompts (plain text, `{placeholders}`) | `prompts/` |

Each LLM task is declared as an `LLMModel` subclass: a system/user prompt file pair in `prompts/`,
optional format params, and a pydantic structured-output class. To add a task, follow the pattern
in `dsview/extraction/models/topics_extraction.py` and register a `ModelType` in `config.py` +
`config/model.yaml` if it needs its own model.

## Model evaluation — rules

Evaluation is a first-class part of this project. Labelled ground truth lives **in the database**
(`labels` schema), produced by the Streamlit labelling interface. Eval code is in
`dsview/evaluation/` (one module per task: `topics_extraction`, `links_extraction`,
`description_generation`, `entity_resolution`, `semantic_score`).

- **Splits are deterministic, not stored.** `assign_rows()` in `dsview/db/query/query_labels.py`
  assigns eval/test membership at query time from `DEFAULT_RANDOM_STATE = 42` and the split
  ratios. **Never change the random state or split ratios** — doing so silently reshuffles which
  rows are eval vs. test and invalidates every past MLflow run's comparability. Entity resolution
  has its own ratios (`SPLIT_RATIOS` in `evaluation/entity_resolution.py`); same rule applies.
- **Develop on `set_type="eval"`, touch `"test"` only for a final measurement** of a chosen
  prompt/model. The test split is small (10–30%); don't burn it iterating.
- **Always log runs to MLflow.** Every `evaluate()` logs params (prompts, model config) and
  metrics when a run is active — wrap calls in `mlflow.start_run()`. An eval result that isn't in
  MLflow can't be compared against later.
- **MLflow run names must be explicit.** Include at least the model name, and the tried
  config/prompt variant if possible (e.g. `claude_haiku_4_5_v2_prompt_eval`, not `run1` or `test`).
  A run name that doesn't say what changed defeats the point of comparing runs later.
- **Comparisons must hold everything else constant.** `evaluate()` accepts `model_config`,
  `system_prompt`, `user_prompt` overrides — change one at a time, and log which one changed.
- **Evals cost real API money.** Topic-extraction eval also fans out an LLM-judge (`SimpleERModel`)
  per predicted×labelled topic pair. Never run `evaluate()` casually or in tests; ask before
  launching a full eval run.
- **Optimize the number of experiments, don't just iterate.** Before launching a run, have a
  specific hypothesis for why this prompt/model/config change might help. Batch or skip
  variants that are unlikely to move the metric rather than trying everything.
- **Never overfit to the eval set.** In particular, never copy examples (rows, IDs, exact
  phrasings) from the eval split into a prompt — that inflates the eval metric without improving
  real generalization, and the gap will show up as a drop on the test split. Prompt improvements
  should come from general reasoning about failure modes, not from eval-set specifics.
- Prompt-optimization experiments (incl. DSPy) live in `notebooks/` as Marimo scripts — check
  there before re-deriving an optimization workflow.
- **A notebook is a test harness — it MUST NEVER drive a change to production code.** There is
  no condition under which working in a `notebooks/` Marimo script (running it, debugging it,
  hitting an error inside it) justifies editing anything under `dsview/` or any other production
  module. If a notebook hits a bug or limitation in production code (e.g. an `asyncio.run` error
  when a batch fallback runs inside the kernel's event loop), **diagnose it, report it, and work
  around it inside the notebook only** (e.g. `nest_asyncio`, or run outside marimo). Never edit
  production code to make a notebook work. Any production-code fix is a separate, explicitly
  requested task — propose it and wait for the user to ask.
- When adding a new extraction task, add its labels table (`dsview/db/schemas/labels_schema.py`),
  a labelling form, and an `evaluation/` module alongside the model — a task without an eval
  path is considered incomplete here.

## Testing — rules

**The existing suite is not hermetic.** Know what each test file touches before running it:

- `tests/test_extraction_models.py`, `test_llm_provider.py`, `test_er_classification.py`:
  make **live LLM API calls** (cost money) and fetch **live URLs** (flaky). Don't run these to
  "check nothing broke" after an unrelated change.
- `tests/test_config.py`, `test_model_utils.py`, `test_content_loader.py`, `test_obsidian.py`:
  mostly safe/local.
- `tests/test_db.py` is **empty** — the DB layer, MCP tools, graph module, and evaluation logic
  have no coverage. New work in those areas should come with tests.

Rules for tests you write:

- **New tests must be hermetic by default**: no live LLM calls, no network, no writes to the real
  Postgres. Mock at the `ModelProvider` / `LLMModel.predict` boundary (structured outputs are
  pydantic models — easy to fabricate), and use a throwaway engine (SQLite in-memory works with
  SQLModel for most schema logic) or a fixture-scoped schema for DB tests.
- If a test genuinely needs a live LLM or network, mark it (`@pytest.mark.llm` /
  `@pytest.mark.network`) so the default `pytest` run stays free and offline. Prefer extending
  this convention over adding more unmarked live tests.
- Metric logic in `dsview/evaluation/` (precision/recall/AP computation, split assignment) is pure
  pandas — it is unit-testable without any LLM and is exactly where silent bugs hurt most. Test it
  with hand-built DataFrames.
- Don't test LLM output *quality* in pytest — that's what the evaluation modules + MLflow are for.
  Tests assert plumbing (types, schemas, prompt formatting, metric math), evals assert quality.

## Code style

Bare-bones but modular: the minimal implementation that solves the task, structured so it's easy
to extend later — not padded with flexibility for that later.

- No speculative surface: no "might be useful later" parameters, return values, branches, or
  abstractions. If nothing calls it, delete it — don't keep it "just in case."
- Modular means correct boundaries, not extra options: a function owns one concern and queries its
  own inputs rather than accepting precomputed state threaded in from a sibling. That's what makes
  it reusable later without a refactor, not extra configurability now.
- Two near-identical blocks (same shape, one differing parameter) become one parameterized
  function/factory, not copy-paste-and-tweak.
- Push work to the layer that does it best — e.g. a SQL join over hand-rolled Python-side
  cross-referencing — instead of building intermediate structures to compensate.

## Conventions

- Prefer an existing, well-established library over hand-rolled parsing/extraction logic (regex,
  manual string processing, ...) when one already does the job — check `pyproject.toml`/`uv.lock`
  first, including transitive dependencies (e.g. `markdown-it-py` comes in via `rich`), before
  writing custom logic or reaching for a new dependency.
- Python ≥ 3.11, `uv` for dependency management, `ruff` for format + lint, `pyright` configured.
- SQLModel tables live in per-domain schema modules under `dsview/db/schemas/`; Postgres schemas
  are `content`, `extraction`, `labels`.
- Prompts are plain-text files with `str.format` placeholders — changing a prompt file changes
  model behavior; treat prompt edits like code changes (evaluate before merging).
- Version is bumped in `pyproject.toml`; Docker images are published by tagging `v*.*.*`.
