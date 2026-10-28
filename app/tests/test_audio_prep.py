import io
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from app.services import audio_prep
from app.services.errors import AppError


def wav(y, sr=16000, channels=1):
    buf = io.BytesIO()
    sf.write(buf, y if channels == 1 else np.stack([y] * channels, 1), sr, format="WAV")
    return buf.getvalue()


def speechlike(seconds, sr=16000, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(0)
    return (amp * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.005 * rng.standard_normal(len(t))).astype(np.float32)


def test_basic_passthrough():
    p = audio_prep.prepare(wav(speechlike(3)), "a.wav")
    assert p.sr == 16000 and len(p.segments) == 1 and 2.5 < p.duration_s <= 3.1


@pytest.mark.parametrize("sr", [8000, 44100, 48000])
def test_resamples(sr):
    p = audio_prep.prepare(wav(speechlike(2, sr), sr), "a.wav")
    assert p.sr == 16000 and any("Resampled" in s for s in p.applied_steps)


def test_stereo_downmix():
    p = audio_prep.prepare(wav(speechlike(2), channels=2), "a.wav")
    assert p.original_channels == 2 and p.samples.ndim == 1


def test_silence_rejected():
    with pytest.raises(AppError, match="No speech"):
        audio_prep.prepare(wav(np.zeros(32000, np.float32)), "a.wav")


def test_too_short_rejected():
    with pytest.raises(AppError, match="too short"):
        audio_prep.prepare(wav(speechlike(0.2)), "a.wav")


def test_corrupt_and_empty_and_bad_ext():
    with pytest.raises(AppError):
        audio_prep.prepare(b"not audio at all" * 100, "a.wav")
    with pytest.raises(AppError, match="empty"):
        audio_prep.prepare(b"", "a.wav")
    with pytest.raises(AppError, match="Unsupported"):
        audio_prep.prepare(b"x", "a.exe")


def test_long_audio_split_at_pauses():
    parts = []
    for _ in range(4):
        parts += [speechlike(10), np.zeros(8000, np.float32)]
    p = audio_prep.prepare(wav(np.concatenate(parts)), "a.wav")
    assert len(p.segments) >= 2 and all(s.end_s - s.start_s <= 28.5 for s in p.segments)


def test_too_long_rejected():
    with pytest.raises(AppError, match="too long"):
        audio_prep.prepare(wav(np.zeros(16000 * 400, np.float32) + 0.1), "a.wav")


def test_clipping_warning():
    y = np.clip(speechlike(2, amp=3.0), -1, 1)
    assert any("clipped" in w for w in audio_prep.prepare(wav(y), "a.wav").warnings)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("ext,codec", [("m4a", "aac"), ("mp3", "libmp3lame"), ("ogg", "libvorbis")])
def test_compressed_formats_decode_through_ffmpeg(tmp_path, ext, codec):
    """m4a has no libsndfile support; before the ffmpeg fallback, uploads of it failed in a fresh environment."""
    src, dst = tmp_path / "a.wav", tmp_path / f"a.{ext}"
    src.write_bytes(wav(speechlike(2)))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-codec:a", codec, str(dst)], check=True)
    y, sr = audio_prep.decode(dst.read_bytes(), dst.name)
    assert abs(len(y) / sr - 2.0) < 0.2 and y.shape[1] == 1
    assert audio_prep.prepare(dst.read_bytes(), dst.name).segments


def test_m4a_error_points_to_ffmpeg_when_it_is_missing(monkeypatch):
    monkeypatch.setattr(audio_prep.shutil, "which", lambda _name: None)
    with pytest.raises(AppError, match="needs ffmpeg"):
        audio_prep.decode(b"not really an m4a file", "voice.m4a")
