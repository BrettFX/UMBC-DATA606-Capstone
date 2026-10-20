import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import app_bundle  # noqa: E402


def _fake_repo(root: Path):
    for rel in ("ct2-int8/model.bin", "ct2-int8/config.json"):
        f = root / app_bundle.ASR_DIR / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x" * 10)
    for v in ("spacy", "spacy-balanced"):
        f = root / "models/ner" / v / "model-best/meta.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{}")


def test_package_fetch_roundtrip(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _fake_repo(src)
    sums = tmp_path / "app.sha256"
    zip_path = tmp_path / "dist" / app_bundle.BUNDLE_NAME
    app_bundle.package(zip_path, root=src, checksum_file=sums)
    app_bundle.fetch(zip_path, dest=dest, checksum_file=sums)
    assert (dest / app_bundle.ASR_DIR / "ct2-int8/model.bin").read_bytes() == b"x" * 10
    assert (dest / "models/ner/spacy-balanced/model-best/meta.json").exists()


def test_checksum_mismatch_extracts_nothing(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _fake_repo(src)
    sums = tmp_path / "app.sha256"
    zip_path = tmp_path / app_bundle.BUNDLE_NAME
    app_bundle.package(zip_path, root=src, checksum_file=sums)
    zip_path.write_bytes(zip_path.read_bytes() + b"tamper")
    with pytest.raises(SystemExit, match="Checksum mismatch"):
        app_bundle.fetch(zip_path, dest=dest, checksum_file=sums)
    assert not dest.exists()


@pytest.mark.parametrize("name", ["../evil.txt", "/abs/evil.txt", "other/evil.txt"])
def test_unsafe_zip_members_are_refused(tmp_path, name):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr(name, "x")
    with pytest.raises(SystemExit, match="Refusing"):
        app_bundle.safe_extract(bad, tmp_path / "dest")


def test_missing_required_model_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        app_bundle.bundle_files(tmp_path)
