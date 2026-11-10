# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for the standalone ClearanceIQ desktop app (onedir layout).

Usage (after `pip install -r requirements-app.txt pyinstaller`):
    ./download-models.sh --profile cpu      # fetch the model bundle into models/ first (~0.8 GB)
    pyinstaller clearanceiq.spec            # writes dist/ClearanceIQ/

The CI build (.github/workflows/release.yml) runs the same two steps per OS and zips dist/ClearanceIQ/
as the release asset, so the result needs no network access and no separate install step on the user's
machine. Training-only dependencies (torch, transformers, datasets, accelerate, bitsandbytes, peft) and
the optional Kokoro voice read-back are excluded on purpose -- see requirements-app.txt.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

REPO_ROOT = Path(SPECPATH).resolve()  # PyInstaller exec()s this file, so __file__ isn't defined here; it injects SPECPATH instead

datas = [
    (str(REPO_ROOT / "app"), "app"),
    (str(REPO_ROOT / "src"), "src"),
    (str(REPO_ROOT / ".streamlit"), ".streamlit"),
]
# models/ is gitignored and fetched by ./download-models.sh; only bundle it if it's actually there, so a
# quick local build (e.g. to sanity-check the spec itself) doesn't require the ~0.8 GB download first.
if (REPO_ROOT / "models").is_dir():
    datas.append((str(REPO_ROOT / "models"), "models"))

binaries = []
# vendor/ffmpeg/ is produced by ./scripts/fetch_static_ffmpeg.sh; bundled only if present (same reasoning as
# models/ above) -- without it the frozen app still runs, just with m4a/mp3 uploads disabled (audio_prep.py
# already degrades gracefully when ffmpeg isn't on PATH, which is exactly the case this leaves it in).
_ffmpeg_dir = REPO_ROOT / "vendor" / "ffmpeg"
if _ffmpeg_dir.is_dir():
    binaries += [(str(f), "ffmpeg") for f in _ffmpeg_dir.iterdir() if f.is_file()]

hiddenimports = []
# These packages load plugins, language data or native backends dynamically, which PyInstaller's static
# import scan misses -- collect_all grabs their data files, binaries and submodules up front instead of
# discovering missing pieces one failed launch at a time.
for pkg in ("streamlit", "spacy", "faster_whisper", "ctranslate2", "plotly"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    [str(REPO_ROOT / "scripts" / "launch_app.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "torchvision", "transformers", "datasets", "accelerate", "bitsandbytes", "peft", "kokoro"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ClearanceIQ",
    console=False,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    name="ClearanceIQ",
)
