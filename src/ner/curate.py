"""Curate LLM annotations into training data: review routing, leakage removal and export.

One LLM pass labels every utterance, so there is no second annotator to flag doubtful labels. Instead
cheap signals computed from the run itself mark labels that are probably wrong (needed a retry, a label
cue such as 'squawk' with no matching span, several unlabeled number words, ...). On held-out human labels,
flagged utterances had a wrong label about 60% of the time against about 15% for unflagged ones, so flagged
utterances go to a review queue and only unflagged ones are used for training.

Evaluation text must never reach training: the corpus is deduplicated on text but a training utterance can
share its wording with a gold or held-out utterance, so any utterance whose text appears in an evaluation
set is dropped from every LLM-labeled split.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .rules import RULE_PATTERNS
from .schema import DIGIT_WORDS, NER_LABELS, entity_spans

REVIEW_CUES = {
    "SQUAWK": re.compile(r"\bsquawk\b"), "RUNWAY": re.compile(r"\brunway\b"),
    "HEADING": re.compile(r"\bheading\b"), "FREQUENCY": RULE_PATTERNS["FREQUENCY"],
}
SPLITS = {"train": "train", "validation": "dev", "test": "test_llm"}  # corpus split -> export name
BERT_LABELS = ["O"] + [f"{p}-{label}" for label in NER_LABELS for p in ("B", "I")]


def review_signals(record: dict) -> list[str]:
    """Reasons an LLM-labeled utterance should not be trusted (empty list = unflagged)."""
    text, spans = record["text"], entity_spans(record["text"], record["entities"])
    signals = []
    if not record["ok"]:
        signals.append("unresolved")
    if record["n_rounds"] > 1:
        signals.append("retry")
    if record.get("n_flags", 0) > 0:
        signals.append("flag-left")
    if not spans:
        signals.append("no-entity")
    for label, pattern in REVIEW_CUES.items():
        if pattern.search(text) and not any(s[2] == label for s in spans):
            signals.append("cue-without-span")
            break
    stray = [m for m in re.finditer(r"\S+", text)
             if m.group() in DIGIT_WORDS and not any(s[0] <= m.start() and m.end() <= s[1] for s in spans)]
    if len(stray) >= 2:
        signals.append("unlabeled-numbers")
    return signals


def to_iob2(text: str, spans: list[tuple[int, int, str]]) -> tuple[list[str], list[str]]:
    """Whitespace tokens and their IOB2 tags for character-offset spans."""
    toks = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    tags = ["O"] * len(toks)
    for s, e, label in spans:
        inside = [i for i, (_, ts, te) in enumerate(toks) if ts >= s and te <= e]
        for k, i in enumerate(inside):
            tags[i] = ("B-" if k == 0 else "I-") + label
    return [t for t, _, _ in toks], tags


@dataclass
class Curated:
    """Curated splits (lists of export rows) plus the bookkeeping that explains them."""

    splits: dict[str, list[dict]] = field(default_factory=dict)
    review_queue: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _row(rec: dict, split: str, origin: str) -> dict:
    text = rec["text"]
    spans = entity_spans(text, rec["entities"])
    tokens, tags = to_iob2(text, spans)
    return {"id": rec["utterance_id"], "text": text, "tokens": tokens, "ner_tags": tags, "split": split, "origin": origin,
            "dataset_source": rec.get("dataset_source"),
            "entities": [{"text": text[s:e], "label": label, "span": [s, e]} for s, e, label in spans]}


def curate(annotations: list[dict], human_sets: dict[str, list[dict]]) -> Curated:
    """Split `annotations` into train/dev/test_llm (unflagged only) and a review queue.

    `human_sets` maps an export name (e.g. 'gold', 'heldout') to hand-verified records with
    `utterance_id`, `text` and `entities`; they are exported unchanged and their texts are excluded
    from every LLM-labeled split.
    """
    eval_texts = {r["text"] for rows in human_sets.values() for r in rows}
    out, dropped, flags = Curated(), Counter(), Counter()
    for name in [*SPLITS.values(), *human_sets]:
        out.splits[name] = []
    for rec in annotations:
        split = SPLITS.get(rec["dataset_split"])
        if split is None:
            continue
        if rec["text"] in eval_texts:
            dropped["text shared with an evaluation set"] += 1
            continue
        signals = review_signals(rec)
        flags.update(signals)
        if signals:
            out.review_queue.append({"utterance_id": rec["utterance_id"], "text": rec["text"], "split": split,
                                     "signals": signals, "entities": rec["entities"]})
            dropped["flagged for review"] += 1
        else:
            out.splits[split].append(_row(rec, split, "llm-unflagged"))
    for name, rows in human_sets.items():
        out.splits[name] = [_row(r, name, "human") for r in rows]
    out.stats = {
        "utterances": {k: len(v) for k, v in out.splits.items()}, "review_queue": len(out.review_queue),
        "dropped": dict(dropped), "signal_counts": dict(flags),
        "entities": {k: dict(Counter(e["label"] for r in v for e in r["entities"])) for k, v in out.splits.items()},
    }
    return out


def write_exports(curated: Curated, out_dir: Path) -> dict:
    """Write spaCy DocBins, BERT IOB2 JSONL, label ids, the review queue and a manifest; returns the manifest."""
    import spacy
    from spacy.tokens import DocBin

    out_dir.mkdir(parents=True, exist_ok=True)
    blank, misaligned = spacy.blank("en"), Counter()
    for name, rows in curated.splits.items():
        docbin = DocBin()
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in rows:
                doc = blank.make_doc(r["text"])
                ents = [doc.char_span(e["span"][0], e["span"][1], label=e["label"], alignment_mode="strict")
                        for e in r["entities"]]
                if any(e is None for e in ents):
                    misaligned[name] += 1
                    continue
                doc.ents = ents
                docbin.add(doc)
                f.write(json.dumps(r) + "\n")
        docbin.to_disk(out_dir / f"{name}.spacy")
    with open(out_dir / "review_queue.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in curated.review_queue)
    (out_dir / "label2id.json").write_text(json.dumps({label: i for i, label in enumerate(BERT_LABELS)}, indent=2))
    manifest = {**curated.stats, "misaligned_dropped": dict(misaligned), "labels": list(NER_LABELS)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
