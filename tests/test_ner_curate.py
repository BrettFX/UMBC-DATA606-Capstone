"""Tests for NER curation: review routing, IOB2 conversion, leakage removal and the exports."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ner import curate, review_signals, to_iob2, write_exports  # noqa: E402


def rec(uid, text, ents, split="train", ok=True, rounds=1, flags=0):
    return {"utterance_id": uid, "text": text, "entities": ents, "dataset_split": split, "dataset_source": "atcosim",
            "ok": ok, "n_rounds": rounds, "n_flags": flags}


def ent(text, label, start):
    return {"text": text, "label": label, "span": [start, start + len(text)]}


def test_review_signals():
    clean = rec("a", "descend flight level one two zero", [ent("descend", "COMMAND", 0), ent("flight level one two zero", "ALTITUDE", 8)])
    assert review_signals(clean) == []
    assert review_signals({**clean, "n_rounds": 2}) == ["retry"]
    assert "unresolved" in review_signals({**clean, "ok": False})
    assert "no-entity" in review_signals(rec("b", "good evening", []))
    # a squawk cue without a SQUAWK span, and several unlabeled number words
    sq = rec("c", "squawk four two one zero", [ent("squawk", "COMMAND", 0)])
    assert "cue-without-span" in review_signals(sq) and "unlabeled-numbers" in review_signals(sq)


def test_to_iob2_tags_multiword_spans():
    toks, tags = to_iob2("contact rhein radar one three two", [(0, 7, "COMMAND"), (8, 19, "FACILITY")])
    assert toks == ["contact", "rhein", "radar", "one", "three", "two"]
    assert tags == ["B-COMMAND", "B-FACILITY", "I-FACILITY", "O", "O", "O"]


def make_inputs():
    good = rec("t1", "descend flight level one two zero", [ent("descend", "COMMAND", 0), ent("flight level one two zero", "ALTITUDE", 8)])
    return [
        good,
        rec("t2", "climb flight level three one zero", [ent("climb", "COMMAND", 0), ent("flight level three one zero", "ALTITUDE", 6)], rounds=2),  # flagged
        rec("t3", "turn left heading two seven zero", [ent("turn left", "COMMAND", 0), ent("heading two seven zero", "HEADING", 10)]),  # leaks: same text as gold
        rec("d1", "descend flight level nine zero", [ent("descend", "COMMAND", 0), ent("flight level nine zero", "ALTITUDE", 8)], split="validation"),
        rec("e1", "contact rhein one three two decimal four", [ent("contact", "COMMAND", 0)], split="test"),  # unlabeled numbers -> flagged
    ]


def test_curate_routes_flagged_and_drops_text_shared_with_human_sets():
    human = {"gold": [{"utterance_id": "g1", "text": "turn left heading two seven zero",
                       "entities": [ent("turn left", "COMMAND", 0), ent("heading two seven zero", "HEADING", 10)]}]}
    out = curate(make_inputs(), human)
    assert [r["id"] for r in out.splits["train"]] == ["t1"]  # t2 flagged, t3 leaked
    assert [r["id"] for r in out.splits["dev"]] == ["d1"] and out.splits["test_llm"] == []
    assert {r["utterance_id"] for r in out.review_queue} == {"t2", "e1"}
    assert out.stats["dropped"] == {"flagged for review": 2, "text shared with an evaluation set": 1}
    assert [r["id"] for r in out.splits["gold"]] == ["g1"] and out.splits["gold"][0]["origin"] == "human"
    assert out.splits["train"][0]["ner_tags"][:1] == ["B-COMMAND"]


def test_write_exports_roundtrips_through_spacy(tmp_path):
    import spacy
    from spacy.tokens import DocBin

    out = curate(make_inputs(), {"gold": []})
    manifest = write_exports(out, tmp_path)
    assert manifest["utterances"]["train"] == 2 and manifest["misaligned_dropped"] == {}  # t3 only leaks when gold contains it
    docs = list(DocBin().from_disk(tmp_path / "train.spacy").get_docs(spacy.blank("en").vocab))
    assert [(e.text, e.label_) for e in docs[0].ents] == [("descend", "COMMAND"), ("flight level one two zero", "ALTITUDE")]
    first = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert first["tokens"][0] == "descend" and (tmp_path / "label2id.json").exists() and (tmp_path / "review_queue.jsonl").exists()
