# ADR 0002: Topics extraction — prompt, model and output schema

## Status

Accepted

## Context

Topic extraction (`TopicsExtractor`) ran on `mistral-small-2603` with the original
`prompts/user_topics_extraction.txt` and scored **P 0.42 / R 0.43 / AP 0.34** — low enough to
warrant a systematic tuning round.

Non-LLM baselines were measured first, to check an LLM is the right tool at all
(`notebooks/topic_modeling_baselines.py`; `yake`/`keybert`/`gliner` are ephemeral notebook
dependencies). Corpus-level topic modelling (LDA/NMF/BERTopic) was rejected without being run: it
assumes documents are mixtures over shared latent topics, whereas this corpus is a heterogeneous
reading list processed one document at a time — it would have measured a different task.

| Method | What it is given | P | R | AP |
|---|---|---|---|---|
| YAKE | the document only (statistical) | 0.07 | 0.09 | 0.02 |
| KeyBERT | general sentence embeddings | 0.28 | 0.13 | 0.21 |
| GLiNER `gliner_medium-v2.1` | the `TopicType` taxonomy as label set | 0.25 | 0.29 | 0.13 |
| `mistral-small-2603`, default prompt | taxonomy + domain knowledge | 0.42 | 0.43 | 0.34 |

*(Pre-dates the dataset repair and `score_row` fix below — comparable to each other, not to later
tables. The logged KeyBERT/YAKE runs also pre-date the MMR and `dedup_lim` settings now in the
notebook.)*

**The task is not keyword extraction.** Scores track how much domain structure each method is
handed: statistics alone effectively fail, embeddings triple AP, the taxonomy gives the best
non-LLM recall. Two structural limits emerged that shaped everything after:

- **GLiNER is entity-only, so it caps at the ~60% of labels that are named entities** — the other
  40% are `Concept`. That 40/60 split turns out to be the axis the whole prompt result turns on.
- **The metric rewards redundant near-duplicate names.** KeyBERT's whole-document similarity
  ranking returns ten paraphrases of one theme, which `score_row` counts as ten hits — a measured
  **3.3×** precision inflation before MMR. The same failure mode recurs in the LLM and is what
  Exp 2's name-normalisation rule fixes.

Two defects were fixed before any prompt comparison was trusted: two degenerate rows in the
labelled set (a fetch-block message and a truncated stub, both re-extractable and repaired via
`notebooks/repair_labelled_content.py`), and `score_row` scoring precision/AP as **1** for empty
predictions, which rewarded abstention. Both fixes *lowered* the baseline; pre-fix numbers are
excluded below. Run-to-run noise is ~0.05 at default temperature, ~0.02 at `temperature: 0`.

## Options considered

Labelled **eval** split (25 rows), `mistral-small-2603`, temperature 0 unless noted. Variants in
`eval_configs/`.

| Variant | P | R | AP | type acc |
|---|---|---|---|---|
| default prompt (default temp) | 0.60 | 0.51 | 0.52 | 0.75 |
| Exp 1 — named-entity-first analysis (default temp) | 0.70 | 0.65 | 0.63 | 0.76 |
| **Exp 2 — Exp 1 + name normalization** (runs 1/2) | **0.72 / 0.70** | **0.66 / 0.64** | **0.66 / 0.64** | 0.75 / 0.72 |
| Exp 3 — Exp 2 + type definitions (runs 1/2) | 0.72 / 0.69 | 0.62 / 0.58 | 0.65 / 0.64 | 0.79 / 0.76 |
| Fable-authored pair — confidence-ranked output | 0.62 | 0.67 | 0.54 | 0.81 |

Held-out **test** split (16 rows): default **0.66 / 0.47 / 0.59**, Exp 2 **0.67 / 0.56 / 0.57**.

Model comparison (eval split): `mistral-medium-latest` scores **0.65 / 0.61 / 0.58** on the default
prompt but **0.57 / 0.57 / 0.50** on Exp 2.

Key findings:

- **The `analysis` field was the single largest win** (AP 0.34 → 0.57). Structured-output decoding
  fills fields in schema order, so an `analysis: str` declared *first* conditions the topic list
  that follows — mirroring `ERResult.analysis`. Field **order is load-bearing**; after `topics` it
  would be inert.
- **Exp 2 is a calibration correction, not a general improvement.** Ground truth is 40% `Concept`.
  The default prompt makes small emit **67%** Concepts; Exp 2 shifts it to **33%**, and that is the
  entire gain.
- **The same prompt makes `mistral-medium-latest` worse** (AP 0.58 → 0.50). Medium is already
  well-calibrated (**36.5%** Concepts); Exp 2 shifts both models toward named entities by a similar
  magnitude, overshooting medium to **14.7%**. Its predictions under Exp 2 harvest incidental proper
  nouns — README integration lists, competitor asides, dated build IDs — while dropping Concepts.
- **Small's advantage over medium is confined to short documents**: +0.17 precision on the short
  tercile, +0.02 on medium, 0.00 on long.
- **Precision and AP decline with content length** for small+Exp 2 (r = −0.42 / −0.40; recall
  unrelated; controlling for label count strengthens it to −0.48). Mechanism is the
  `MAX_N_TOPICS = 10` cap — predict/label ratio runs 0.87 on short documents vs 1.05–1.20 on long
  ones, i.e. the model pads toward the cap where marginal predictions are wrong. Medium shows no
  length effect.
- **Type accuracy trades against recall.** Exp 3 bought +0.04 type accuracy for −0.05 recall,
  reproducibly; 8 of its 15 lost labels were `Concept`s (RAG, Quantization, RL, …) — explicit type
  definitions demote `Concept` to a residual bucket. Rejected. Medium's higher type accuracy is the
  same artifact: it is computed over *matched* topics only, and named entities type unambiguously.
- **Confidence-ranking the output backfired** — the Fable variant gave the worst AP (0.54) despite
  the best type accuracy.
- **Eval-split gains are optimistically biased.** Only the **recall** gain survived on test (+0.09);
  precision was flat, AP slightly negative.

## Decision

Promote the **Exp 2 user prompt** (`eval_configs/topics_exp2_user_prompt.txt` →
`prompts/user_topics_extraction.txt`) with **`temperature: 0`** on **`mistral-small-2603`**, keeping
the `analysis` field first in `TopicList`. System prompt stays at its default — no variant beat it.

## Consequences

- Expected production effect is **recall +0.09 at flat precision** (the held-out number), not the
  larger eval-split gains.
- **The prompt is not model-portable** — it corrects a bias specific to `mistral-small-2603` and
  actively harms a better-calibrated model. Any model change must re-evaluate it. Comparing
  predicted `Concept` share against the ~40% ground truth is a fast diagnostic.
- `mistral-medium-latest` is the better *base* model but not worth its cost: small+Exp 2 beats
  medium+default on all three metrics. Worth revisiting if the prompt is ever retired.
- Remaining headroom is in **long documents** (~0.18 precision below short ones); the untested
  hypothesis is cap-padding, ceiling ~+0.06 overall.
- **Native reasoning could not be evaluated.** Both models support `reasoning_effort`, defaulting to
  `"none"` (verified: `"high"` returns a `ThinkChunk` at ~40× completion tokens). It is unreachable
  here because `chat.parse` raises `TypeError` on the list-valued `content` reasoning returns
  (`mistralai/extra/struct_chat.py` handles only `str`/`None`); testing it needs a provider change in
  `dsview/model_utils/providers/mistral.py`. Reasoning also rejects greedy sampling unless `top_p`
  is explicitly 1 (API error 3054).
- All metrics are 25 eval / 16 test rows; differences under ~0.02 are noise, and every
  `mistral-medium-latest` result is single-run.
