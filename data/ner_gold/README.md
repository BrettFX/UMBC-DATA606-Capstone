# NER gold, silver and held-out labels

Label schema and guidelines live in `src/ner/` (`schema.py`, `prompt.py`). Entity dicts are
`{"text", "label", "span": [start, end)}` with character offsets into `text`.

## Final files (use these)

| file | utterances | what it is |
|---|---|---|
| `gold.jsonl` | 41 | the original hand-labeled set. Left untouched: the NER notebook's cached ablation results depend on it |
| `gold_expanded.jsonl` | 95 | `gold.jsonl` + 54 utterances reviewed by the project author (34 adjudicated disagreements, 20 audit items with labels shown). Human-verified. Used for tuning and few-shot retrieval |
| `silver.jsonl` | 167 | labeled by two annotators (Claude, blind; and the 9B model) and **not** reviewed by a human: 112 where they agreed, 55 where the disagreement was explained by a decided guideline. Fine for training, not for evaluation |
| `heldout_gold.jsonl` | 100 | random validation/test utterances, never used for tuning or few-shot retrieval. 38 human-adjudicated, 62 agreed by both annotators. **Frozen: do not tune on it.** One span is flagged `needs_confirmation` |

Every row carries `tier`, `origin`, `selection_reason`, `dataset_split` and `dataset_source`.
`selection_reason` of `random` marks an unbiased sample; `targeted:*` and `quota:*` rows oversample
rare labels, so use them only for per-label (balanced-slice) metrics.

## `work/` (provenance, rarely needed)

Intermediate files from the labeling rounds, kept so the gold set can be audited or rebuilt:
candidate lists (`candidates*.jsonl`, `heldout.jsonl`), model pre-labels (`*_prelabeled.jsonl`),
Claude's blind labels (`claude_labels.jsonl`, `heldout_claude.jsonl`), the reviewer queues
(`audit_queue.jsonl`, `heldout_queue.jsonl`) and the raw reviewer exports (`gold_new.jsonl`,
`heldout_reviewed.jsonl`). The scripts that made them are in `scripts/ner/gold/` and refuse to
overwrite an existing output without `--overwrite`.

`candidates.jsonl` was produced before the selection script excluded identical texts, so re-running the
script would select a slightly different set. Treat these files as records, not as regenerable outputs.
