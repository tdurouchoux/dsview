---
status: "accepted"
date: "2026-07-18"
decision-makers: "tdurouchoux"
---

# Entity Resolution classifier — model and prompt

## Context and Problem Statement

The entity-resolution (ER) classifier used `mistral-medium-2508` with the original
`prompts/system_entity_resolution.txt` prompt in production. `mistral-medium-2508` is being
deprecated, forcing a replacement decision. Before committing to a like-for-like swap to
`mistral-medium-latest`, we used the opportunity to re-evaluate both model and prompt.

Mid-investigation we found that the eval/test split (`assign_rows` in
`dsview/db/query/query_labels.py`) assigned membership by **row position** rather than by a
stable key, so every time new labels were added the eval/test membership of *existing* rows
silently reshuffled. This made some prompt-vs-prompt comparisons noisier than they appeared. We
fixed this (hash-based split, keyed on each row's DB id) before the final round of comparisons
below. All results in the decision table use the new, stable split; some numbers seen earlier in
the investigation (under the old split) are not directly comparable and are omitted here.

Prompt iterations (`eval_configs/prompts/system_entity_resolution_v2.txt` through `v8.txt`) explored
tightening or loosening the merge rules, the number/order of reasoning steps in the
`<topic_comparison>` block, and the few-shot examples — see those files and the eval configs in
`eval_configs/` for the exact variants tried.

## Considered Options

* `mistral-small-latest` + prompt v7 (chosen)
* `mistral-medium-2508` (prior production, default prompt)
* `mistral-medium-latest`, default prompt
* `mistral-medium-latest` + prompt v2
* `mistral-medium-latest` + prompt v6
* `mistral-small-latest` + prompt v2
* `mistral-small-latest` + prompt v6
* `claude-haiku-4-5` + prompt v6

## Decision Outcome

Chosen option: "`mistral-small-latest` with prompt v7", because it ties the prior production
F0.75 score (0.90) while being the cheapest model evaluated, with a recall-favoring error profile
(perfect recall, 0.85 precision) that is the safer failure mode for a deduplication task — false
merges are easier to catch/undo than silently-missed duplicates.

Promoted **`mistral-small-latest` with prompt `eval_configs/prompts/system_entity_resolution_v7.txt`**
to production for entity-resolution classification (`config/model.yaml`,
`prompts/system_entity_resolution.txt`).

### Consequences

* Good, because it has a lower per-call cost than the deprecated `mistral-medium-2508`, at
  matching measured quality.
* Neutral, because the error profile shifts to recall-favoring (occasional over-merge) rather than
  the prior balanced profile — worth a periodic spot-check of merged topics post-rollout.
* Good, because the hash-based split fix means all future eval/test comparisons are stable across
  labelling-set growth; re-running old configs will no longer silently drift due to split
  reshuffling.
* Neutral, because further prompt tuning on this task looks close to a plateau for Mistral-family
  models (±0.01–0.02 per iteration in later rounds); the next material gain, if needed, is more
  likely to come from a model change than another prompt revision.

## Pros and Cons of the Options

All figures are **F0.75** on the labelled `test` split (small, ~80 pairs — treat differences
under ~0.02 as noise), sync (non-batch) API calls, hash-based split.

| Model | Prompt | Precision | Recall | F0.75 |
|---|---|---|---|---|
| `mistral-medium-2508` (prior production) | default | 0.90 | 0.91 | **0.90** |
| `mistral-medium-latest` | default | 0.83 | 0.94 | 0.87 |
| `mistral-medium-latest` | v2 (heavily-tuned rules) | 0.93 | 0.83 | 0.89 |
| `mistral-medium-latest` | v6 (simplified rules) | 0.82 | 1.00 | 0.88 |
| `mistral-small-latest` | v2 | 0.90 | 0.81 | 0.87 |
| `mistral-small-latest` | v6 | 0.85 | 0.98 | 0.89 |
| **`mistral-small-latest`** | **v7 (v6 + broadened category/instance rule)** | **0.85** | **1.00** | **0.90** |
| `claude-haiku-4-5` | v6 | 0.87 | 0.98 | 0.90 |

Key findings:

* Model choice mattered less than expected. Every model tested (small/medium/haiku) landed
  in a tight 0.87–0.90 F0.75 band on test once paired with a decent prompt — no model was a clear
  loser or winner on its own.
* Prompt tuning had a real but model-dependent effect. The v2→v6→v7 line of iteration (fewer
  reasoning steps, a broadened "category vs. specific named instance" merge rule) measurably
  helped both `claude-haiku-4-5` and `mistral-small-latest`, but the same v6 prompt made
  `mistral-medium-latest` *worse* (it over-merges under v6, precision 0.82 vs 0.93 under v2) —
  prompt changes do not transfer uniformly across models.
* `mistral-medium-latest` never matched the prior production score (0.90) under any prompt
  tried (best: 0.89, v2). The deprecated `2508` pin appears to have been unusually well-suited to
  the original default prompt specifically, not a stand-in for "any mistral-medium".
* `mistral-small-latest` + v7 ties prior production exactly (0.90) while being the cheapest
  model evaluated, with a recall-favoring error profile (perfect recall, 0.85 precision) — it
  never misses a true duplicate merge, only occasionally over-merges, which is the safer failure
  mode for a deduplication task (false merges are easier to catch/undo than silently-missed
  duplicates).
* `claude-haiku-4-5` + v6 ties the same score (0.90) via a different provider; kept as a fallback
  option but not needed since a same-provider (Mistral) option already matches.
