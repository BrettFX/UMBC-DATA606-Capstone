"""Behavior checks moved verbatim from the NER notebook's cell-level sanity asserts."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ner.schema import *  # noqa: F401,F403
from ner.schema import NUMBER_WORDS, DIGIT_WORDS, GREETING_PHRASES  # noqa: F401


def test_schema_checks():
    # Sanity checks
    _t = "lufthansa four one alpha descend flight level one two zero"
    _a, _r = align_entities(_t, [EntitySpan(text="Lufthansa four one alpha", label="CALLSIGN"),
                                 EntitySpan(text="flight level one two zero", label="ALTITUDE"),
                                 EntitySpan(text="descend to", label="COMMAND"),
                                 EntitySpan(text="four one", label="ALTITUDE")])
    assert [(_t[s:e], l) for s, e, l in _a] == [("lufthansa four one alpha", "CALLSIGN"), ("flight level one two zero", "ALTITUDE")]
    assert len(_r) == 2
    for _bad in [EntitySpan(text="good day", label="COMMAND"), EntitySpan(text="rhein one three two decimal four", label="FACILITY"),
                 EntitySpan(text="on one three three decimal four", label="FREQUENCY"), EntitySpan(text="six seven zero", label="CALLSIGN"),
                 EntitySpan(text="runway", label="RUNWAY"), EntitySpan(text="one eight zero", label="FREQUENCY"),
                 EntitySpan(text="ils approach", label="WAYPOINT"), EntitySpan(text="speed two fifty", label="ALTITUDE"),
                 EntitySpan(text="one eight zero knots", label="ALTITUDE")]:
        try:
            check_constraints(_bad)
            raise AssertionError(f"constraint should have rejected {_bad}")
        except ValueError:
            pass
    _e = with_spans(_t, [{"text": "descend", "label": "COMMAND"}, {"text": "lufthansa four one alpha", "label": "CALLSIGN"}])
    assert _e == [{"text": "lufthansa four one alpha", "label": "CALLSIGN", "span": [0, 24]},
                  {"text": "descend", "label": "COMMAND", "span": [25, 32]}]
    try:
        entity_spans(_t, [{"text": "descend", "label": "COMMAND", "span": [0, 7]}])
        raise AssertionError("a wrong span must be rejected")
    except ValueError:
        pass
    try:
        check_constraints(EntitySpan(text="one eight zero", label="FREQUENCY"), "speed one eight zero")
        raise AssertionError("a number after 'speed' must be rejected")
    except ValueError:
        pass
    assert [is_trainable_utterance(x) for x in ["roger", "thank you", "standby", "ah stand by", "roger turn left"]] == [False, False, False, True, True]


def test_rules_and_parser():
    from ner.rules import rule_candidates
    from ner.prompt import parse_tool_calls
    assert ("FREQUENCY", "one three three decimal four") in rule_candidates("contact zurich one three three decimal four")
    assert ("FACILITY", "zurich") in rule_candidates("contact zurich one three three decimal four")
    raw = ("<tool_call>\n<function=add_entity>\n<parameter=text>\nflight level one two zero\n</parameter>\n"
           "<parameter=label>\nALTITUDE\n</parameter>\n</function>\n</tool_call><|im_end|>")
    assert parse_tool_calls(raw) == [("add_entity", {"text": "flight level one two zero", "label": "ALTITUDE"})]


def test_gold_is_self_consistent():
    from ner.schema import EntitySpan, check_constraints, entity_spans
    from ner.scoring import load_gold, score_spans
    gold = load_gold(str(pathlib.Path(__file__).resolve().parents[1] / "data" / "ner_gold" / "gold.jsonl"))
    for g in gold.values():
        entity_spans(g["text"], g["entities"])
        for e in g["entities"]:
            check_constraints(EntitySpan(text=e["text"], label=e["label"]), g["text"])
    assert score_spans(gold, {u: g["entities"] for u, g in gold.items()}).loc["ALL", "f1"] == 1.0
