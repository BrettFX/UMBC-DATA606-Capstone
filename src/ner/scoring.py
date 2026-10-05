"""Span-level scoring against the hand-labeled gold set."""

import json
from collections import Counter

import pandas as pd

from .schema import NER_LABELS, entity_spans

def load_gold(path: str) -> dict[str, dict]:
    with open(path) as f:
        return {r["utterance_id"]: r for r in map(json.loads, f)}


def _span_set(text: str, entities: list[dict]) -> set[tuple[int, int, str]]:
    return set(entity_spans(text, entities))


def score_spans(gold: dict[str, dict], pred: dict[str, list[dict]]) -> pd.DataFrame:
    """Exact-match (start, end, label) precision/recall/F1, per label and overall."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for uid, g in gold.items():
        gs, ps = _span_set(g["text"], g["entities"]), _span_set(g["text"], pred.get(uid, []))
        for s in ps & gs: tp[s[2]] += 1
        for s in ps - gs: fp[s[2]] += 1
        for s in gs - ps: fn[s[2]] += 1
    rows = {}
    for label in list(NER_LABELS) + ["ALL"]:
        t, p, n = ((sum(d.values()) for d in (tp, fp, fn)) if label == "ALL"
                   else (tp[label], fp[label], fn[label]))
        prec, rec = t / (t + p) if t + p else 0.0, t / (t + n) if t + n else 0.0
        rows[label] = {"precision": prec, "recall": rec,
                       "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "support": t + n}
    return pd.DataFrame(rows).T
