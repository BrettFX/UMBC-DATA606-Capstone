#!/usr/bin/env python3
"""Entry point for the frozen ClearanceIQ desktop build (see clearanceiq.spec).

A PyInstaller bundle has no separate `streamlit` executable to shell out to, so this re-enters
Streamlit's own CLI in-process instead, pointed at the copy of app/main.py inside the bundle.
Not used in development -- for that, use ./run-app.sh or `streamlit run app/main.py` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bundle_root() -> Path:
    # PyInstaller's onedir layout puts bundled data files (app/, src/, models/, .streamlit/) next to the
    # executable; sys._MEIPASS points there both in that layout and in a onefile build's unpack directory.
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def main() -> None:
    root = _bundle_root()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))

    import streamlit.web.cli as stcli

    sys.argv = [
        "streamlit", "run", str(root / "app" / "main.py"),
        "--global.developmentMode=false",
        "--server.headless=false",  # open the user's browser automatically; config.toml defaults this to true
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
