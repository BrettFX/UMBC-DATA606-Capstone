"""Tests for artifact collection and the non-overwriting S3 upload, using a fake S3 client (no AWS calls)."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import artifacts  # noqa: E402


class FakeS3:
    def __init__(self, existing: list[str] | None = None):
        self.existing, self.uploaded, self.bodies = existing or [], [], {}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys):
        return {"KeyCount": sum(k.startswith(Prefix) for k in self.existing)}

    def upload_file(self, path, bucket, key):
        self.uploaded.append((bucket, key))

    def put_object(self, Bucket, Key, Body, ContentType):
        self.uploaded.append((Bucket, Key))
        self.bodies[Key] = Body


def touch(path: pathlib.Path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_repo(tmp_path):
    touch(tmp_path / "models/ner/spacy/model-best/ner/model")
    touch(tmp_path / "models/ner/spacy/results.json", "{}")
    touch(tmp_path / "models/ner/bert/model-best/model.safetensors")
    touch(tmp_path / "models/ner/bert/checkpoints/checkpoint-100/optimizer.pt")  # must never be uploaded
    touch(tmp_path / "models/whisper-small-en-atc-finetuned-full/final/model.safetensors")
    touch(tmp_path / "models/whisper-small-en-atc-finetuned-full/checkpoint-100/optimizer.pt")
    touch(tmp_path / "models/whisper-small-en-atc-finetuned-full/test_scores.json", "{}")
    touch(tmp_path / "data/processed/asr_results/finetuned_whisper-small-en_full.csv")
    touch(tmp_path / "data/processed/ner_dataset/curated/manifest.json", "{}")
    return tmp_path


def test_collect_ner_excludes_checkpoints_and_respects_variants(tmp_path):
    repo = make_repo(tmp_path)
    keys = {i.rel_key for i in artifacts.collect("ner", repo)}
    assert "spacy/model-best/ner/model" in keys and "spacy/results.json" in keys and "bert/model-best/model.safetensors" in keys
    assert "data/curated_manifest.json" in keys
    assert not any("checkpoint" in k or "optimizer" in k for k in keys)
    only_spacy = {i.rel_key.split("/")[0] for i in artifacts.collect("ner", repo, ["spacy"])}
    assert only_spacy == {"spacy", "data"}  # the bert variant is filtered out; shared data manifests remain


def test_collect_asr_takes_final_model_and_scores_not_checkpoints(tmp_path):
    keys = {i.rel_key for i in artifacts.collect("asr", make_repo(tmp_path))}
    assert "whisper-small-en-atc-finetuned-full/final/model.safetensors" in keys
    assert "whisper-small-en-atc-finetuned-full/test_scores.json" in keys
    assert "test_results/finetuned_whisper-small-en_full.csv" in keys
    assert not any("checkpoint" in k for k in keys)


def test_collect_asr_selected_variant_takes_its_own_results_and_adapter(tmp_path):
    repo = make_repo(tmp_path)
    touch(repo / "models/whisper-medium-en-atc-finetuned-full-lora/final/model.safetensors")
    touch(repo / "models/whisper-medium-en-atc-finetuned-full-lora/final-adapter/adapter_model.safetensors")
    touch(repo / "models/whisper-medium-en-atc-finetuned-full-lora/checkpoint-9/optimizer.pt")
    for name in ("finetuned_whisper-medium-en_full-lora.csv", "finetuned_whisper-small-en_full-8ep.csv", "experiments_summary.md"):
        touch(repo / "data/processed/asr_results" / name)
    keys = {i.rel_key for i in artifacts.collect("asr", repo, ["whisper-medium-en-atc-finetuned-full-lora"])}
    assert "whisper-medium-en-atc-finetuned-full-lora/final-adapter/adapter_model.safetensors" in keys
    assert "test_results/finetuned_whisper-medium-en_full-lora.csv" in keys and "test_results/experiments_summary.md" in keys
    assert "test_results/finetuned_whisper-small-en_full-8ep.csv" not in keys  # another run's results are not dragged along
    assert not any("checkpoint" in k or k.startswith("whisper-small") for k in keys)


def test_collect_rejects_unknown_task(tmp_path):
    with pytest.raises(ValueError):
        artifacts.collect("tts", tmp_path)


def test_upload_writes_files_and_manifest_under_prefix(tmp_path):
    repo = make_repo(tmp_path)
    items = artifacts.collect("ner", repo, ["spacy"])
    manifest = artifacts.build_manifest("ner", "run1", items, repo)
    s3 = FakeS3()
    n = artifacts.upload(items, manifest, "s3://bucket/ml-tasks/ner/run1", s3)
    assert n == len(items) + 1
    assert ("bucket", "ml-tasks/ner/run1/spacy/results.json") in s3.uploaded
    body = json.loads(s3.bodies["ml-tasks/ner/run1/manifest.json"])
    assert body["run_id"] == "run1" and {f["key"] for f in body["files"]} == {i.rel_key for i in items}
    assert all(len(f["sha256"]) == 64 for f in body["files"])


def test_upload_refuses_existing_prefix_unless_overwrite(tmp_path):
    repo = make_repo(tmp_path)
    items = artifacts.collect("ner", repo, ["spacy"])
    s3 = FakeS3(existing=["ml-tasks/ner/run1/old.bin"])
    with pytest.raises(FileExistsError):
        artifacts.upload(items, {}, "s3://bucket/ml-tasks/ner/run1", s3)
    assert s3.uploaded == []  # nothing was written
    assert artifacts.upload(items, {}, "s3://bucket/ml-tasks/ner/run1", s3, overwrite=True) > 0
    # a sibling run folder is a different prefix, even when its name starts the same
    assert artifacts.upload(items, {}, "s3://bucket/ml-tasks/ner/run", FakeS3(existing=["ml-tasks/ner/run1/old.bin"])) > 0


def test_split_s3_uri():
    assert artifacts.split_s3_uri("s3://b/ml-tasks/ner/") == ("b", "ml-tasks/ner")
    with pytest.raises(ValueError):
        artifacts.split_s3_uri("https://example.com/x")
