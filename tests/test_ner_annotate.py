"""Tests for the annotation engine and the resumable run, using a fake backend (no GPU or vLLM)."""
import json
import pathlib
import re
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ner import AnnotationConfig, annotate_all, build_pool, load_cache, run_annotation, validate  # noqa: E402


class FakeBackend:
    """Returns scripted JSON per transcript; a list means 'first call, second call, ...'."""

    def __init__(self, script: dict[str, str | list[str]]):
        self.script, self.calls, self.batch_sizes = script, {}, []

    def render(self, messages):
        return "\n".join(m["content"] for m in messages)

    def generate(self, prompts):
        self.batch_sizes.append(len(prompts))
        outs = []
        for prompt in prompts:
            text = re.findall(r"Transcript: (.+)", prompt)[-1].strip()
            n = self.calls[text] = self.calls.get(text, 0) + 1
            reply = self.script.get(text, '{"entities": []}')
            outs.append(reply[min(n, len(reply)) - 1] if isinstance(reply, list) else reply)
        return outs


def ents(*pairs):
    return json.dumps({"entities": [{"text": t, "label": label} for t, label in pairs]})


def test_validate_accepts_valid_and_reports_rejections():
    ok, errors = validate("lufthansa four one alpha descend", ents(("lufthansa four one alpha", "CALLSIGN"), ("descend", "COMMAND")))
    assert [e.text for e in ok] == ["lufthansa four one alpha", "descend"] and not errors
    ok, errors = validate("descend now", ents(("climb", "COMMAND")))  # not in the transcript
    assert not ok and len(errors) == 1 and "does not appear" in errors[0]
    ok, errors = validate("descend now", "not json")
    assert not ok and "not valid JSON" in errors[0]


def test_retry_only_reasks_failed_utterances_and_feeds_errors_back():
    texts = {"a": "descend now", "b": "climb now"}
    backend = FakeBackend({
        "descend now": ents(("descend", "COMMAND")),
        "climb now": [ents(("hello", "COMMAND")), ents(("climb", "COMMAND"))],  # wrong first, right second
    })
    result = annotate_all(backend, texts, max_rounds=3)
    assert backend.batch_sizes == [2, 1]  # the second round only re-asks the failing utterance
    assert result["a"]["ok"] and result["a"]["n_rounds"] == 1
    assert result["b"]["ok"] and result["b"]["n_rounds"] == 2 and result["b"]["n_rejected"] == 1
    assert [e["text"] for e in result["b"]["entities"]] == ["climb"]


def test_unresolved_utterance_keeps_valid_entities_and_is_not_ok():
    backend = FakeBackend({"descend now": ents(("descend", "COMMAND"), ("zzz", "COMMAND"))})
    r = annotate_all(backend, {"a": "descend now"}, max_rounds=2)["a"]
    assert not r["ok"] and r["n_rounds"] == 2 and [e["text"] for e in r["entities"]] == ["descend"]


def test_missed_verb_flag_retries_but_never_marks_not_ok():
    backend = FakeBackend({"direct canne": [ents(("canne", "WAYPOINT")), ents(("canne", "WAYPOINT"))]})
    r = annotate_all(backend, {"a": "direct canne"}, max_rounds=2, check_missed=True)["a"]
    assert r["ok"] and r["n_rounds"] == 2 and r["n_flags"] == 1


def test_postprocess_merges_split_facility():
    backend = FakeBackend({"contact rhein radar": ents(("contact", "COMMAND"), ("rhein", "FACILITY"), ("radar", "FACILITY"))})
    r = annotate_all(backend, {"a": "contact rhein radar"}, postprocess=True)["a"]
    assert [(e["text"], e["label"]) for e in r["entities"]] == [("contact", "COMMAND"), ("rhein radar", "FACILITY")]


def make_utterances():
    rows = [("u1", "descend flight level one two zero"), ("u2", "climb flight level one two zero"),
            ("u3", "descend flight level one two zero"),  # duplicate text of u1
            ("u4", "roger"),  # single word
            ("u5", "thank you"),  # filler only
            ("u6", "contact rhein one three two decimal four")]
    return pd.DataFrame({"utterance_id": [r[0] for r in rows], "transcript_normalized": [r[1] for r in rows],
                         "dataset_split": "train", "dataset_source": "atcosim"})


def test_build_pool_dedupes_and_drops_untrainable():
    pool = build_pool(make_utterances())
    assert sorted(pool["utterance_id"]) == ["u1", "u2", "u6"]


def test_run_annotation_checkpoints_resumes_and_writes_manifest(tmp_path):
    pool = build_pool(make_utterances())
    config = AnnotationConfig(model="fake/model", prompt_version=3, check_missed=False)
    backend = FakeBackend({})
    path = run_annotation(backend, pool, tmp_path, config, chunk_size=2, limit=2)
    assert len(load_cache(path)) == 2 and backend.batch_sizes == [2]
    # resume: only the one missing utterance is sent to the model
    backend2 = FakeBackend({})
    run_annotation(backend2, pool, tmp_path, config, chunk_size=2)
    cache = load_cache(path)
    assert sorted(cache) == ["u1", "u2", "u6"] and backend2.batch_sizes == [1]
    lines = [json.loads(line)["utterance_id"] for line in open(path)]
    assert len(lines) == len(set(lines)) == 3  # no utterance written twice
    manifest = json.loads((tmp_path / f"manifest_{config.signature()}.json").read_text())
    assert manifest["complete"] and manifest["annotated"] == 3 and manifest["config"]["model"] == "fake/model"
    progress = json.loads((tmp_path / f"progress_{config.signature()}.json").read_text())
    assert progress["status"] == "finished" and progress["annotated"] == 3 and progress["this_run_done"] == 1
    assert progress["labels"] == {} and len(progress["chunk_seconds"]) == 1  # the fake model labels nothing
    # a finished run asks the model for nothing
    backend3 = FakeBackend({})
    run_annotation(backend3, pool, tmp_path, config)
    assert backend3.batch_sizes == []


def test_signature_depends_on_config_and_prompt():
    a = AnnotationConfig(model="m", prompt_version=3)
    assert a.signature() == AnnotationConfig(model="m", prompt_version=3).signature()
    assert a.signature() != AnnotationConfig(model="m", prompt_version=2).signature()
    assert a.signature() != AnnotationConfig(model="other", prompt_version=3).signature()
    assert a.signature() != AnnotationConfig(model="m", prompt_version=3, postprocess=False).signature()
