"""Score any NER model against curated sets with the span metric used for the LLM annotators."""

from __future__ import annotations

import time
from typing import Callable

from .schema import NER_LABELS
from .scoring import score_spans

Predict = Callable[[list[str]], list[list[tuple[int, int, str]]]]  # texts -> per-text (start, end, label) spans


def evaluate_sets(predict: Predict, sets: dict[str, list[dict]]) -> dict:
    """Span-level (exact start, end, label) scores per set: overall P/R/F1, macro-F1 over labels that
    occur in the set, per-label F1, and exact-match utterances."""
    results = {}
    for name, rows in sets.items():
        if not rows:
            continue
        preds = predict([r["text"] for r in rows])
        gold = {r["id"]: {"text": r["text"], "entities": r["entities"]} for r in rows}
        pred = {r["id"]: [{"text": r["text"][s:e], "label": label, "span": [s, e]} for s, e, label in spans]
                for r, spans in zip(rows, preds)}
        table = score_spans(gold, pred)
        present = [label for label in NER_LABELS if table.loc[label, "support"] > 0]
        exact = sum({(e["span"][0], e["span"][1], e["label"]) for e in r["entities"]} == set(p)
                    for r, p in zip(rows, preds))
        results[name] = {
            "n": len(rows), "precision": table.loc["ALL", "precision"], "recall": table.loc["ALL", "recall"],
            "f1": table.loc["ALL", "f1"], "macro_f1": float(table.loc[present, "f1"].mean()), "exact": exact,
            "per_label_f1": {label: round(float(table.loc[label, "f1"]), 3) for label in NER_LABELS},
            "support": {label: int(table.loc[label, "support"]) for label in NER_LABELS}}
    return results


def time_predict(predict: Predict, texts: list[str]) -> float:
    """Utterances per second on `texts`."""
    t0 = time.perf_counter()
    predict(texts)
    return len(texts) / (time.perf_counter() - t0)
