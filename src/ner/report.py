"""Per-label classification report (precision, recall, F1, support) for NER models on the frozen human-verified held-out set.

The layout follows scikit-learn's `classification_report`, but the counts are exact (start, end, label) spans, so there is
no accuracy row (a span task has no true negatives); the `micro avg` row, with all spans pooled, takes its place.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .inference import SpacyNer
from .scoring import load_gold, score_spans

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_ORDER = ["CALLSIGN", "COMMAND", "FACILITY", "ALTITUDE", "WAYPOINT", "FREQUENCY", "RUNWAY", "HEADING", "SQUAWK"]
HELDOUT = REPO_ROOT / "data/ner_gold/heldout_gold.jsonl"
ANNOTATIONS = REPO_ROOT / "data/processed/ner_dataset/annotations"
TEACHER = "LLM teacher (Qwen3.5-9B production labels)"


def teacher_predictions(gold: dict[str, dict], annotations_dir: Path = ANNOTATIONS) -> dict[str, list[dict]]:
    """Production (Qwen3.5-9B) labels for each held-out utterance, matched by transcript text (the annotated pool is deduplicated)."""
    path = next(annotations_dir.glob("annotations_*.jsonl"))
    by_text = {r["text"]: r["entities"] for r in map(json.loads, open(path))}
    return {u: by_text[g["text"]] for u, g in gold.items() if g["text"] in by_text}


def student_predictions(gold: dict[str, dict], name: str, models_dir: Path = REPO_ROOT / "models/ner") -> dict[str, list[dict]]:
    ner = SpacyNer(models_dir / name / "model-best")
    ids = list(gold)
    return dict(zip(ids, ner.predict_many([gold[u]["text"] for u in ids])))


def classification_report(gold: dict[str, dict], pred: dict[str, list[dict]]) -> pd.DataFrame:
    """Rows: each label with support, then micro avg (all spans pooled), macro avg and weighted avg over labels with support."""
    gold = {u: g for u, g in gold.items() if u in pred}
    table = score_spans(gold, pred)
    labels = [label for label in LABEL_ORDER if table.loc[label, "support"] > 0]
    rows = table.loc[labels, ["precision", "recall", "f1", "support"]].copy()
    sup = rows["support"]
    rows.loc["micro avg"] = table.loc["ALL", ["precision", "recall", "f1", "support"]]
    rows.loc["macro avg"] = [*rows.loc[labels, ["precision", "recall", "f1"]].mean(), sup.sum()]
    rows.loc["weighted avg"] = [*(rows.loc[labels, c].mul(sup).sum() / sup.sum() for c in ("precision", "recall", "f1")), sup.sum()]
    return rows


def build_reports(students: list[str], heldout: Path = HELDOUT) -> dict[str, pd.DataFrame]:
    """The teacher's report and one report per student model directory, keyed by display title."""
    gold = load_gold(str(heldout))
    reports = {TEACHER: classification_report(gold, teacher_predictions(gold))}
    for name in students:
        reports[f"Student: {name}"] = classification_report(gold, student_predictions(gold, name))
    return reports


def to_text(rows: pd.DataFrame) -> str:
    width = max(len(i) for i in rows.index)
    lines = [f"{'':>{width}}  precision    recall  f1-score   support", ""]
    for i, r in rows.iterrows():
        if i == "micro avg":
            lines.append("")
        lines.append(f"{i:>{width}}  {r.precision:9.2f} {r.recall:9.2f} {r.f1:9.2f} {int(r.support):9d}")
    return "\n".join(lines)


def render_image(rows: pd.DataFrame, title: str, path: Path, *, subtitle: str = "", min_support: int = 5, dpi: int = 200) -> list[str]:
    """Draws the report as a slide-ready table; returns the labels left out of the table (support below `min_support`).

    Omitted labels still count toward the averages, which the footnote says. F1 cells are shaded by value.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    surface, ink, ink2, grid, blue = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
    labels = [i for i in rows.index if i not in ("micro avg", "macro avg", "weighted avg")]
    shown = [l for l in labels if rows.loc[l, "support"] >= min_support]
    hidden = [f"{l} ({int(rows.loc[l, 'support'])})" for l in labels if l not in shown]
    body = [(l, "label") for l in shown] + [(l, "avg") for l in ("micro avg", "macro avg", "weighted avg")]

    cols = [("", 0.02, "left"), ("precision", 0.50, "right"), ("recall", 0.66, "right"), ("f1-score", 0.82, "right"), ("support", 0.97, "right")]
    row_h, top = 0.36, 1.05
    height = top + row_h * (len(body) + 1) + 0.15 + (0.35 if hidden else 0.1)
    fig = plt.figure(figsize=(7.4, height), facecolor=surface)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, height)
    ax.axis("off")

    ax.text(0.02, height - 0.28, title, fontsize=13, fontweight="bold", color=ink, va="center")
    if subtitle:
        ax.text(0.02, height - 0.6, subtitle, fontsize=9.5, color=ink2, va="center")

    y = height - top
    for text, x, ha in cols:
        ax.text(x, y - row_h / 2, text, fontsize=10.5, fontweight="bold", color=ink2, ha=ha, va="center")
    y -= row_h
    ax.plot([0.02, 0.97], [y, y], color=ink2, linewidth=1)
    for name, kind in body:
        if name == "micro avg":
            y -= 0.15
            ax.plot([0.02, 0.97], [y + 0.075, y + 0.075], color=grid, linewidth=1)
        r = rows.loc[name]
        f1 = float(r.f1)
        ax.add_patch(Rectangle((0.725, y - row_h + 0.04), 0.10, row_h - 0.08, facecolor=blue, alpha=max(0.0, (f1 - 0.5) / 0.5) * 0.45,
                               edgecolor="none"))
        weight = "bold" if name == "micro avg" else "normal"
        ax.text(cols[0][1], y - row_h / 2, name, fontsize=11, fontweight=weight, color=ink, va="center")
        for (text, x, ha), value in zip(cols[1:], (r.precision, r.recall, r.f1, r.support)):
            shown_value = f"{int(value)}" if text == "support" else f"{value:.2f}"
            ax.text(x, y - row_h / 2, shown_value, fontsize=11, fontweight=weight, color=ink, ha=ha, va="center")
        y -= row_h
    if hidden:
        ax.text(0.02, 0.2, f"Not shown (too few held-out spans, but counted in the averages): {', '.join(hidden)}.",
                fontsize=8.5, color=ink2, va="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=surface)
    plt.close(fig)
    return hidden


def training_label_stats(rows: list[dict]) -> pd.DataFrame:
    """Per label over training rows: span count, share of all spans, distinct span texts, and the share of spans that are distinct."""
    texts: dict[str, list[str]] = {}
    for r in rows:
        for e in r["entities"]:
            texts.setdefault(e["label"], []).append(e["text"])
    total = sum(len(v) for v in texts.values())
    stats = pd.DataFrame({label: {"spans": len(v), "share_pct": 100 * len(v) / total, "distinct": len(set(v)), "distinct_ratio": len(set(v)) / len(v)}
                          for label, v in texts.items()}).T
    return stats.sort_values("spans", ascending=False)


def runway_template(rows: list[dict]) -> dict:
    """How template-like RUNWAY spans are: how many start with the word 'runway' and how many are exactly three words."""
    spans = [e["text"] for r in rows for e in r["entities"] if e["label"] == "RUNWAY"]
    return {"spans": len(spans), "start_with_runway": sum(t.startswith("runway") for t in spans),
            "three_words": sum(len(t.split()) == 3 for t in spans)}


def label_errors(gold: dict[str, dict], pred: dict[str, list[dict]], label: str) -> list[dict]:
    """Held-out utterances where the model's spans for `label` differ from the human spans (texts, not offsets)."""
    from .schema import entity_spans

    out = []
    for uid, g in gold.items():
        if uid not in pred:
            continue
        want = {s for s in entity_spans(g["text"], g["entities"]) if s[2] == label}
        got = {s for s in entity_spans(g["text"], pred[uid]) if s[2] == label}
        if want != got:
            text = g["text"]
            out.append({"text": text, "human": [text[a:b] for a, b, _ in sorted(want)], "model": [text[a:b] for a, b, _ in sorted(got)]})
    return out
