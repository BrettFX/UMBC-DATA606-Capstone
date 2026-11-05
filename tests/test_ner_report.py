
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ner import report  # noqa: E402


def _row(text, *ents):
    return {"text": text, "entities": [{"text": text[s:e], "label": l, "span": [s, e]} for s, e, l in ents]}


def test_report_matches_hand_computed_precision_recall_f1_and_averages():
    gold = {"a": _row("runway two four left", (0, 14, "RUNWAY")),
            "b": _row("descend and contact tower", (0, 7, "COMMAND"), (12, 19, "COMMAND"), (20, 25, "FACILITY"))}
    pred = {"a": gold["a"]["entities"],                                             # RUNWAY: 1 of 1
            "b": [gold["b"]["entities"][0], {"text": "tower", "label": "COMMAND", "span": [20, 25]}]}  # 1 COMMAND hit, 1 COMMAND false positive, 2 misses
    rows = report.classification_report(gold, pred)

    assert list(rows.index) == ["COMMAND", "FACILITY", "RUNWAY", "micro avg", "macro avg", "weighted avg"]
    assert rows.loc["COMMAND", ["precision", "recall", "f1", "support"]].tolist() == pytest.approx([0.5, 0.5, 0.5, 2])
    assert rows.loc["FACILITY", ["precision", "recall", "f1"]].tolist() == [0.0, 0.0, 0.0]
    assert rows.loc["RUNWAY", "f1"] == 1.0
    assert rows.loc["micro avg", ["precision", "recall", "support"]].tolist() == pytest.approx([2 / 3, 2 / 4, 4])
    assert rows.loc["macro avg", "recall"] == pytest.approx((0.5 + 0.0 + 1.0) / 3)
    assert rows.loc["weighted avg", "recall"] == pytest.approx((0.5 * 2 + 0.0 * 1 + 1.0 * 1) / 4)


def test_labels_without_support_are_left_out_and_text_layout_has_a_micro_row():
    gold = {"a": _row("runway two four left", (0, 14, "RUNWAY"))}
    rows = report.classification_report(gold, {"a": gold["a"]["entities"]})
    assert "CALLSIGN" not in rows.index
    text = report.to_text(rows)
    assert "precision    recall  f1-score   support" in text and "micro avg" in text


def test_image_omits_rare_labels_but_still_renders(tmp_path):
    gold = {"a": _row("runway two four left", (0, 14, "RUNWAY")), "b": _row("squawk one two", (0, 6, "SQUAWK"))}
    rows = report.classification_report(gold, {u: g["entities"] for u, g in gold.items()})
    out = tmp_path / "table.png"
    assert report.render_image(rows, "t", out, min_support=2) == ["RUNWAY (1)", "SQUAWK (1)"]
    assert out.stat().st_size > 1000


def test_training_stats_runway_template_and_label_errors():
    rows = [_row("descend runway two four left", (8, 28, "RUNWAY")), _row("runway one three", (0, 16, "RUNWAY")),
            _row("direct oscar kilo golf", (7, 22, "WAYPOINT"))]
    stats = report.training_label_stats(rows)
    assert stats.loc["RUNWAY", "spans"] == 2 and stats.loc["RUNWAY", "share_pct"] == pytest.approx(200 / 3)
    assert stats.loc["WAYPOINT", "distinct_ratio"] == 1.0
    assert report.runway_template(rows) == {"spans": 2, "start_with_runway": 2, "three_words": 1}
    gold = {"x": rows[2], "y": rows[1]}
    pred = {"x": [{"text": "oscar", "label": "WAYPOINT", "span": [7, 12]}], "y": rows[1]["entities"]}
    assert report.label_errors(gold, pred, "WAYPOINT") == [{"text": "direct oscar kilo golf", "human": ["oscar kilo golf"], "model": ["oscar"]}]
    assert report.label_errors(gold, pred, "RUNWAY") == []
