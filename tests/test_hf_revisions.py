"""The Hugging Face revisions are pinned, and the code that downloads from the Hub actually uses the pins."""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hf_revisions  # noqa: E402


def test_every_pin_is_a_full_commit_sha_and_ids_are_unique():
    entries = {**hf_revisions.DATASETS, **hf_revisions.MODELS}
    assert len(entries) == len(hf_revisions.DATASETS) + len(hf_revisions.MODELS)  # no repo id in both tables
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in entries.values())  # a branch name or short SHA would not pin anything


def test_pin_and_revision():
    assert hf_revisions.pin("openai/whisper-small.en") == {"revision": hf_revisions.MODELS["openai/whisper-small.en"]}
    assert hf_revisions.pin("some/new-experiment") == {} and hf_revisions.pin("/local/model/dir") == {} and hf_revisions.pin(None) == {}
    assert hf_revisions.revision("jlvdoorn/atcosim") == hf_revisions.DATASETS["jlvdoorn/atcosim"]


def test_snapshot_passes_local_directories_through(tmp_path):
    assert hf_revisions.snapshot(str(tmp_path)) == str(tmp_path)  # a local model directory is not a Hub repo


def test_every_hub_id_the_project_defaults_to_is_pinned():
    import asr  # via the package: importing the asr.train submodule directly would shadow the package's lazily exposed train()
    from preprocessing.ingest import DEFAULT_SOURCES

    wanted = set(DEFAULT_SOURCES.values()) | {asr.TrainConfig().model, "bert-base-uncased", "openai/whisper-medium.en", "hexgrad/Kokoro-82M"}
    wanted |= set(re.findall(r'model="([^"]+)"', (ROOT / "scripts/run_ner_annotate.py").read_text()))  # the two vLLM presets
    wanted |= {"jacktol/whisper-medium.en-fine-tuned-for-ATC"}
    assert wanted <= (set(hf_revisions.DATASETS) | set(hf_revisions.MODELS)), wanted - set(hf_revisions.DATASETS) - set(hf_revisions.MODELS)


def test_datasets_are_loaded_at_the_pinned_commit(monkeypatch, tmp_path):
    """Both of the pipeline's download paths (`_load` and `download_raw`) must ask the Hub for the pinned commit."""
    from preprocessing import ingest

    for method in ("_load", "download_raw"):
        seen = {}
        monkeypatch.setattr(ingest, "load_dataset", lambda repo, **kw: seen.setdefault(repo, kw) and {} or {})
        pipe = ingest.DataIngestPipeline(data_dir=str(tmp_path))
        getattr(pipe, method)()
        assert set(seen) == set(pipe.sources.values()), method
        assert all(kw["revision"] == hf_revisions.revision(repo) for repo, kw in seen.items()), method


def test_the_readme_table_matches_the_pins():
    """The README lists 7-character commit prefixes; each must belong to a pinned repo of the same name."""
    rows = re.findall(r"^\| (?:Dataset|Model) \| (.+?) \| (.+?) \|$", (ROOT / "README.md").read_text(), flags=re.M)
    assert len(rows) >= 11
    pins = {**hf_revisions.DATASETS, **hf_revisions.MODELS}
    for repos, commits in rows:
        names, prefixes = re.findall(r"`([^`]+)`", repos), re.findall(r"`([0-9a-f]{7})`", commits)
        names = [n for n in names if n in pins]
        assert names and len(names) == len(prefixes), (repos, commits)
        for name, prefix in zip(names, prefixes):
            assert pins[name].startswith(prefix), (name, prefix)


def test_loader_calls_in_the_code_are_pinned():
    """A guard against a new from_pretrained/load_dataset call that forgets the pin (hub ids only; local paths are fine)."""
    checks = {
        "src/preprocessing/ingest.py": [r"load_dataset\(repo_id, cache_dir=cache_dir, \*\*pin\(repo_id\)\)"],
        "src/asr/train.py": [r"WhisperProcessor\.from_pretrained\(config\.model, \*\*pin\(config\.model\)\)", r"\*\*pin\(config\.model\)\)\.to\(device\)"],
        "src/ner/train_bert.py": [r"\*\*pin\(model_name\)"],
        "scripts/run_ner_annotate.py": [r"snapshot\(preset\[\"model\"\]\)"],
        "app/services/speech.py": [r"hub_file\(KOKORO_REPO"],
    }
    for path, patterns in checks.items():
        text = (ROOT / path).read_text()
        for pattern in patterns:
            assert re.search(pattern, text), (path, pattern)
    assert "load_dataset(repo_id, cache_dir=cache_dir)" not in (ROOT / "src/preprocessing/ingest.py").read_text()
