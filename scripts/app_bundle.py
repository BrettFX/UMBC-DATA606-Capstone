#!/usr/bin/env python3
"""Package and fetch the single-zip model bundle that ClearanceIQ needs (the Google Drive fallback to S3).

The bundle holds only what the app loads (~0.8 GB): the CTranslate2 int8 ASR model, both spaCy NER models and the small
JSON files behind the model cards. The zip is hosted on Google Drive, but its sha256 is committed to this repo
(`app-models.sha256`), so a tampered or corrupted download is rejected before anything is extracted.

Maintainer (after retraining or promoting a model):
    python scripts/app_bundle.py package            # writes dist/clearanceiq-models.zip and updates app-models.sha256
    # upload the zip to the Drive folder (replace the old file), then commit app-models.sha256

Anyone without S3 access:
    python scripts/app_bundle.py fetch              # downloads from Drive, verifies, extracts into models/
    python scripts/app_bundle.py fetch --zip path/to/clearanceiq-models.zip    # already downloaded by hand
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_NAME = "clearanceiq-models.zip"
CHECKSUM_FILE = REPO_ROOT / "app-models.sha256"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1F4tGwPjIgFPdo3ZClRX5-y3gFGThBWZ1"

ASR_DIR = "models/whisper-medium-en-atc-finetuned-full-lora"
# (path relative to the repo root, optional) -- directories are included recursively
BUNDLE_PATHS = [
    (f"{ASR_DIR}/ct2-int8", False),
    (f"{ASR_DIR}/test_scores.json", True),
    ("models/ner/spacy/model-best", False),
    ("models/ner/spacy/results.json", True),
    ("models/ner/spacy-balanced/model-best", False),
    ("models/ner/spacy-balanced/results.json", True),
]
ALLOWED_PREFIX = "models/"
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def bundle_files(root: Path = REPO_ROOT) -> list[Path]:
    files = []
    for rel, optional in BUNDLE_PATHS:
        p = root / rel
        if not p.exists():
            if optional:
                continue
            raise FileNotFoundError(f"{rel} is missing; train or download the models first (./download-models.sh).")
        files += sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
    return files


def package(out: Path, root: Path = REPO_ROOT, checksum_file: Path = CHECKSUM_FILE) -> str:
    files = bundle_files(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in files:
            z.write(f, f.relative_to(root).as_posix())
    digest = sha256_of(out)
    checksum_file.write_text(f"{digest}  {BUNDLE_NAME}\n")  # the name Drive must hold, whatever --out was
    print(f"Wrote {out} ({out.stat().st_size / 1e9:.2f} GB, {len(files)} files)\nsha256 {digest} -> {checksum_file}")
    return digest


def expected_digest(checksum_file: Path = CHECKSUM_FILE) -> str:
    try:
        return checksum_file.read_text().split()[0].lower()
    except (FileNotFoundError, IndexError) as exc:
        raise SystemExit(f"{checksum_file.name} is missing or empty, so the download cannot be verified.") from exc


def safe_extract(zip_path: Path, dest: Path) -> list[str]:
    """Extracts only regular files under models/, refusing absolute paths and '..' (zip-slip)."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        for n in names:
            target = (dest / n).resolve()
            if not n.startswith(ALLOWED_PREFIX) or not target.is_relative_to(dest):
                raise SystemExit(f"Refusing to extract unexpected path in bundle: {n}")
        z.extractall(dest, members=[i for i in z.infolist() if not i.is_dir()])
    return names


def download_from_drive(workdir: Path) -> Path:
    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown is not installed. Run ./setup.sh (or pip install gdown), or download the zip by hand from\n"
                         f"  {DRIVE_FOLDER_URL}\nand run: python scripts/app_bundle.py fetch --zip <file>") from None
    gdown.download_folder(url=DRIVE_FOLDER_URL, output=str(workdir), quiet=False, use_cookies=False)
    found = list(workdir.rglob(BUNDLE_NAME))
    if not found:
        raise SystemExit(f"{BUNDLE_NAME} was not found in the Drive folder {DRIVE_FOLDER_URL}")
    return found[0]


def fetch(zip_path: Path | None, dest: Path = REPO_ROOT, checksum_file: Path = CHECKSUM_FILE) -> int:
    want = expected_digest(checksum_file)
    with tempfile.TemporaryDirectory() as tmp:
        src = zip_path or download_from_drive(Path(tmp))
        got = sha256_of(src)
        if got != want:
            raise SystemExit(f"Checksum mismatch for {src.name}: expected {want}, got {got}. Nothing was extracted.")
        names = safe_extract(src, dest)
    print(f"Verified and extracted {len(names)} files into {dest / 'models'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    pk = sub.add_parser("package", help="Build the zip and update app-models.sha256.")
    pk.add_argument("--out", type=Path, default=REPO_ROOT / "dist" / BUNDLE_NAME)
    fe = sub.add_parser("fetch", help="Download (or use --zip), verify against app-models.sha256, extract into models/.")
    fe.add_argument("--zip", type=Path, default=None, help="Use an already-downloaded bundle instead of downloading.")
    args = p.parse_args(argv)
    if args.command == "package":
        package(args.out)
        return 0
    return fetch(args.zip)


if __name__ == "__main__":
    sys.exit(main())
