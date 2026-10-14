"""Inference-time audio preparation: decode, validate, normalize, trim and segment for the ASR model.

The fine-tuned Whisper models take mono 16 kHz audio of at most 30 s. Everything between "whatever the user
uploaded" and that contract happens here, and every change is recorded in `applied_steps` so the UI can show it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import librosa
import numpy as np
import soundfile as sf

from app import config
from app.services.errors import AppError


@dataclass
class Segment:
    start_s: float
    end_s: float
    samples: np.ndarray = field(repr=False)


@dataclass
class PreparedAudio:
    samples: np.ndarray = field(repr=False)          # mono float32 at 16 kHz, trimmed and normalized
    sr: int
    original_sr: int
    original_channels: int
    original_duration_s: float
    segments: list[Segment]
    warnings: list[str]
    applied_steps: list[str]
    snr_db: float | None = None
    trim_start_s: float = 0.0     # leading silence removed: prepared time + this = time in the original recording

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sr


def decode(data: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    """Decode bytes to (samples [n] or [n, ch] float32, sample rate). Raises AppError on failure."""
    if not data:
        raise AppError("The file is empty.", "Choose a different recording.")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext and ext not in config.ALLOWED_EXTENSIONS:
        raise AppError(f"Unsupported file type '.{ext}'.", f"Supported: {', '.join(config.ALLOWED_EXTENSIONS)}.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise AppError(f"The file is larger than {config.MAX_UPLOAD_MB} MB.")
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return y, int(sr)
    except Exception:
        pass
    try:  # mp3/m4a and other formats libsndfile cannot read (needs ffmpeg/audioread)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=f".{ext or 'bin'}") as f:
            f.write(data)
            f.flush()
            y, sr = librosa.load(f.name, sr=None, mono=False)
        y = y[:, None] if y.ndim == 1 else y.T
        return y.astype(np.float32), int(sr)
    except Exception as exc:
        raise AppError("The audio could not be decoded; the file may be corrupt or in an unsupported format.",
                       f"Technical detail: {type(exc).__name__}") from exc


def estimate_snr_db(y: np.ndarray, sr: int) -> float | None:
    """Rough SNR: loud-frame RMS over quiet-frame RMS (10th percentile). None if it cannot be estimated."""
    rms = librosa.feature.rms(y=y, frame_length=int(0.025 * sr), hop_length=int(0.010 * sr))[0]
    if len(rms) < 10:
        return None
    noise, speech = np.percentile(rms, 10), np.percentile(rms, 90)
    if noise <= 0 or speech <= 0:
        return None
    return float(20 * np.log10(speech / noise))


def segment_on_pauses(y: np.ndarray, sr: int, max_s: float) -> list[tuple[int, int]]:
    """Split at the longest pauses so no piece exceeds `max_s`. Falls back to fixed windows with overlap-free cuts."""
    n, max_n = len(y), int(max_s * sr)
    if n <= max_n:
        return [(0, n)]
    voiced = librosa.effects.split(y, top_db=config.TRIM_TOP_DB, frame_length=1024, hop_length=256)
    gaps = [(voiced[i][1], voiced[i + 1][0]) for i in range(len(voiced) - 1) if voiced[i + 1][0] > voiced[i][1]]
    cuts, start = [], 0
    while n - start > max_n:
        window = [(a, b) for a, b in gaps if start + int(max_n * 0.4) < (a + b) // 2 <= start + max_n]
        if window:  # cut in the middle of the longest pause inside the window
            a, b = max(window, key=lambda g: g[1] - g[0])
            cut = (a + b) // 2
        else:  # no pause found: hard cut
            cut = start + max_n
        cuts.append((start, cut))
        start = cut
    cuts.append((start, n))
    return cuts


def prepare(data: bytes, filename: str = "") -> PreparedAudio:
    y2, sr0 = decode(data, filename)
    channels = y2.shape[1]
    steps: list[str] = []
    warnings: list[str] = []
    y = y2.mean(axis=1) if channels > 1 else y2[:, 0]
    if channels > 1:
        steps.append(f"Mixed {channels} channels down to mono")
    if not np.isfinite(y).all():
        y = np.nan_to_num(y)
        warnings.append("The audio contained invalid samples; they were replaced with silence.")
    original_duration = len(y) / sr0
    if original_duration < config.MIN_DURATION_S:
        raise AppError(f"The clip is too short ({original_duration:.2f} s).",
                       f"Provide at least {config.MIN_DURATION_S:.1f} s of speech.")
    if original_duration > config.MAX_DURATION_S:
        raise AppError(f"The clip is too long ({original_duration / 60:.1f} min).",
                       f"The maximum is {config.MAX_DURATION_S / 60:.0f} minutes; trim it first.")
    if sr0 < 8000:
        warnings.append(f"Low sample rate ({sr0} Hz); recognition quality may suffer.")
    if np.abs(y).max() >= 0.999 and (np.abs(y) >= 0.999).mean() > 0.001:
        warnings.append("The recording is clipped (distorted); recognition quality may suffer.")

    y = y - y.mean()
    if sr0 != config.SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr0, target_sr=config.SAMPLE_RATE)
        steps.append(f"Resampled {sr0 / 1000:g} kHz → {config.SAMPLE_RATE / 1000:g} kHz")
    y = y.astype(np.float32)

    if float(np.sqrt(np.mean(y ** 2))) < config.MIN_RMS:
        raise AppError("No speech detected: the recording is silent or nearly silent.")

    trimmed, (a, b) = librosa.effects.trim(y, top_db=config.TRIM_TOP_DB, frame_length=1024, hop_length=256)
    pad = int(config.TRIM_PAD_S * config.SAMPLE_RATE)
    a, b = max(0, a - pad), min(len(y), b + pad)
    trim_start = 0.0
    if (a + (len(y) - b)) / config.SAMPLE_RATE >= 0.3:
        trim_start = a / config.SAMPLE_RATE
        steps.append(f"Trimmed {(a + len(y) - b) / config.SAMPLE_RATE:.1f} s of leading/trailing silence")
        y = y[a:b]
    if len(y) / config.SAMPLE_RATE < config.MIN_DURATION_S:
        raise AppError("No speech detected: only a very short sound was found.")

    peak = float(np.abs(y).max())
    if 0 < peak < 0.5:  # raise quiet recordings, but never by more than 12 dB (avoid boosting pure noise)
        gain = min(0.9 / peak, 4.0)
        y = (y * gain).astype(np.float32)
        steps.append(f"Raised level by {20 * np.log10(gain):.0f} dB")
    snr = estimate_snr_db(y, config.SAMPLE_RATE)
    if snr is not None and snr < 8:
        warnings.append(f"Very noisy recording (estimated SNR ≈ {snr:.0f} dB); expect more transcription errors.")

    spans = segment_on_pauses(y, config.SAMPLE_RATE, config.MAX_SEGMENT_S)
    segments = [Segment(a / config.SAMPLE_RATE, b / config.SAMPLE_RATE, y[a:b]) for a, b in spans]
    if len(segments) > 1:
        steps.append(f"Split into {len(segments)} segments at pauses (model limit is 30 s)")
    return PreparedAudio(samples=y, sr=config.SAMPLE_RATE, original_sr=sr0, original_channels=channels,
                         original_duration_s=original_duration, segments=segments, warnings=warnings,
                         applied_steps=steps, snr_db=snr, trim_start_s=trim_start)
