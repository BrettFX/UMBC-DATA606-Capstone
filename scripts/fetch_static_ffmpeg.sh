#!/usr/bin/env bash
# Downloads a static ffmpeg binary for the current OS into vendor/ffmpeg/, so clearanceiq.spec can bundle it
# into the standalone build (mp3/m4a uploads then work with no system ffmpeg install). Best-effort and
# non-fatal: a failure here (e.g. one of the third-party static-build hosts is down) only warns and exits 0,
# because the app already degrades gracefully without ffmpeg (see app/services/audio_prep.py) -- it just
# disables m4a/mp3 uploads and says why.
#
# Usage: ./scripts/fetch_static_ffmpeg.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/vendor/ffmpeg"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

warn_and_skip() {
    echo "WARNING: $1 -- continuing without a bundled ffmpeg (m4a/mp3 uploads will be disabled in this build)." >&2
    exit 0
}

mkdir -p "$DEST"

case "$(uname -s)" in
    Linux)
        case "$(uname -m)" in
            x86_64|amd64) arch=linux64 ;;
            aarch64|arm64) arch=linuxarm64 ;;
            *) warn_and_skip "unsupported Linux architecture '$(uname -m)'" ;;
        esac
        URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-${arch}-gpl.tar.xz"
        curl -sL --fail "$URL" -o "$TMP/ffmpeg.tar.xz" || warn_and_skip "could not download $URL"
        tar -xf "$TMP/ffmpeg.tar.xz" -C "$TMP" || warn_and_skip "could not extract the Linux ffmpeg archive"
        cp "$TMP"/ffmpeg-*/bin/ffmpeg "$DEST/ffmpeg" || warn_and_skip "the Linux ffmpeg archive layout changed; see $URL"
        chmod +x "$DEST/ffmpeg"
        ;;
    Darwin)
        INFO_URL="https://evermeet.cx/ffmpeg/info/ffmpeg/release"
        ZIP_URL="$(curl -sL --fail "$INFO_URL" | python3 -c "import sys,json; print(json.load(sys.stdin)['download']['zip']['url'])" 2>/dev/null)"
        [[ -n "$ZIP_URL" ]] || warn_and_skip "could not resolve the macOS static ffmpeg download URL from $INFO_URL"
        curl -sL --fail "$ZIP_URL" -o "$TMP/ffmpeg.zip" || warn_and_skip "could not download $ZIP_URL"
        unzip -q "$TMP/ffmpeg.zip" -d "$TMP" || warn_and_skip "could not extract the macOS ffmpeg archive"
        cp "$TMP/ffmpeg" "$DEST/ffmpeg" || warn_and_skip "the macOS ffmpeg archive layout changed; see $INFO_URL"
        chmod +x "$DEST/ffmpeg"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        curl -sL --fail "$URL" -o "$TMP/ffmpeg.zip" || warn_and_skip "could not download $URL"
        7z x "$TMP/ffmpeg.zip" -o"$TMP" -y >/dev/null || warn_and_skip "could not extract the Windows ffmpeg archive"
        cp "$TMP"/ffmpeg-*/bin/ffmpeg.exe "$DEST/ffmpeg.exe" || warn_and_skip "the Windows ffmpeg archive layout changed; see $URL"
        ;;
    *)
        warn_and_skip "unrecognized OS '$(uname -s)'"
        ;;
esac

echo "Static ffmpeg ready at $DEST"
