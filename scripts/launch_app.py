#!/usr/bin/env python3
"""Entry point for the frozen ClearanceIQ desktop build (see clearanceiq.spec).

A PyInstaller bundle has no separate `streamlit` executable to shell out to, so this re-enters
Streamlit's own CLI in-process instead, pointed at the copy of app/main.py inside the bundle.
Not used in development -- for that, use ./run-app.sh or `streamlit run app/main.py` instead.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback
from pathlib import Path


def _bundle_root() -> Path:
    # PyInstaller's onedir layout puts bundled data files (app/, src/, models/, .streamlit/) next to the
    # executable; sys._MEIPASS points there both in that layout and in a onefile build's unpack directory.
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def _expose_bundled_ffmpeg(root: Path) -> None:
    # scripts/fetch_static_ffmpeg.sh + clearanceiq.spec put a static ffmpeg here when one was fetched at
    # build time; putting its folder on PATH is all app/services/audio_prep.py needs (it looks ffmpeg up
    # with shutil.which). A build without it leaves PATH untouched, and that lookup -- and the app -- just
    # falls back to its existing "m4a/mp3 uploads are disabled" message instead of failing.
    ffmpeg_dir = root / "ffmpeg"
    if ffmpeg_dir.is_dir():
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")


def _skip_onboarding_prompt() -> None:
    # On a machine that has never run Streamlit before, its CLI asks an interactive "enter your email"
    # onboarding question on first launch and blocks on stdin for the answer -- which never comes here,
    # since a windowed (console=False) build has no stdin attached. That hang (not a crash) is the likeliest
    # explanation for "a loading cursor, then nothing": pre-writing an empty credentials.toml is Streamlit's
    # own documented way to mark onboarding already done, so it's skipped unconditionally.
    try:
        path = Path.home() / ".streamlit" / "credentials.toml"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[general]\nemail = \"\"\n")
    except OSError:
        pass


def _report_crash(root: Path, exc: BaseException) -> None:
    # A windowed (console=False) build has no terminal to print to -- sys.stdout/stderr may be None or a
    # silent null-writer -- so this would otherwise fail with nothing but a flash of the loading cursor and
    # no trace of why. Write the traceback next to the exe and, on Windows, pop up a message box pointing
    # at it, so a double-click launch surfaces *something* actionable instead of just dying quietly.
    log_path = root / "clearanceiq_error.log"
    try:
        log_path.write_text(traceback.format_exc())
    except OSError:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                None,
                f"ClearanceIQ failed to start:\n\n{exc}\n\nDetails were written to:\n{log_path}",
                "ClearanceIQ",
                0x10,  # MB_ICONERROR
            )
        except Exception:
            pass


def main() -> None:
    root = _bundle_root()
    try:
        os.chdir(root)  # so Streamlit's own project-relative config.toml lookup finds the bundled .streamlit/
        _skip_onboarding_prompt()
        _expose_bundled_ffmpeg(root)
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
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 -- anything here would otherwise vanish silently; see _report_crash
        _report_crash(root, exc)
        raise


if __name__ == "__main__":
    # Windows has no fork(): any dependency that spawns a worker process (Python's multiprocessing, or a
    # library using it internally) re-execs this very exe, and without freeze_support() that re-exec runs
    # main() again instead of behaving as a worker -- symptoms range from a silent hang to runaway relaunches.
    # A cheap, harmless no-op on other platforms and on the main process itself.
    multiprocessing.freeze_support()
    main()
