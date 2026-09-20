"""Acoustic feature extraction for individual utterance waveforms.

Pure numpy/librosa functions with no Hugging Face `datasets` dependency, so
they're independently testable against a synthetic waveform (e.g. a sine
wave or a silence array) without touching the ingestion pipeline. Called
once per row from `ingest.DataIngestPipeline._add_utterance_metadata`'s
existing batched audio-decode map, so every clip is decoded exactly once.
"""

from __future__ import annotations

from typing import Dict

import librosa
import numpy as np

FRAME_LENGTH = 2048
HOP_LENGTH = 512
SILENCE_TOP_DB = 30
SNR_NOISE_PERCENTILE = 10
SNR_SIGNAL_PERCENTILE = 95

FEATURE_NAMES = (
    "rms_energy",
    "silence_pct",
    "snr_db_proxy",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
)


def compute_acoustic_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Derive six acoustic summary features from one decoded mono waveform.

    Returns NaN for every feature if `y` is empty/degenerate rather than
    raising -- corrupt-row handling is centralized in the pipeline's
    per-row try/except, not duplicated here.

    rms_energy (unitless, >=0):
        Mean, across FRAME_LENGTH=2048/hop=512 frames, of
        sqrt(mean(y_frame**2)) (`librosa.feature.rms`). `y` is float PCM in
        [-1, 1], so this is a loudness proxy, not dBFS/SPL.

    silence_pct (percent, 0-100):
        `100 * (1 - sum(non_silent_interval_lengths) / len(y))`, where
        non-silent intervals come from `librosa.effects.split(y,
        top_db=30)`. A fully-silent clip (no intervals returned) is defined
        as 100.0, not an error. `top_db=30` is a fixed, documented constant
        -- changing it changes every value, so it's pinned at module level
        rather than tuned per source.

    snr_db_proxy (dB, HEURISTIC -- not a physical SNR measurement):
        Percentile-gap noise-floor proxy: per-frame RMS converted to dB,
        then `P95(frame_db) - P10(frame_db)` -- the gap between the
        loudest (speech-active) and quietest (background) frames, using
        percentiles instead of min/max so a single clipping transient
        doesn't dominate. CAVEATS: no clean noise-only reference exists in
        either corpus, so this is a *relative dynamic-range proxy*, valid
        only for comparative claims ("source X trends noisier than source
        Y"), never as an absolute physical SNR value. Assumes background
        noise is quieter and roughly stationary relative to speech, which
        can break down on real ATC recordings where the noise floor and
        speech overlap.

    zero_crossing_rate (unitless fraction, 0-1):
        Mean, across frames, of `librosa.feature.zero_crossing_rate`:
        fraction of adjacent-sample sign changes per frame. Higher values
        correlate with noisiness/unvoiced speech and radio static.

    spectral_centroid_hz (Hz):
        Mean, across frames, of `librosa.feature.spectral_centroid`: the
        amplitude-weighted mean frequency of the magnitude spectrum.
        Higher = more high-frequency ("brighter") energy content.

    spectral_bandwidth_hz (Hz):
        Mean, across frames, of `librosa.feature.spectral_bandwidth`:
        amplitude-weighted standard deviation of frequency around the
        centroid -- how spread out (vs. tonal/narrowband) the spectrum is.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0 or not sr:
        return {name: float("nan") for name in FEATURE_NAMES}

    rms_frames = librosa.feature.rms(
        y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    rms_energy = float(np.mean(rms_frames))

    non_silent_intervals = librosa.effects.split(
        y, top_db=SILENCE_TOP_DB, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    if len(non_silent_intervals) == 0:
        silence_pct = 100.0
    else:
        non_silent_samples = sum(end - start for start, end in non_silent_intervals)
        silence_pct = float(100 * (1 - non_silent_samples / len(y)))

    frame_db = librosa.amplitude_to_db(rms_frames, ref=1.0)
    snr_db_proxy = float(
        np.percentile(frame_db, SNR_SIGNAL_PERCENTILE)
        - np.percentile(frame_db, SNR_NOISE_PERCENTILE)
    )

    zero_crossing_rate = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
            )[0]
        )
    )

    spectral_centroid_hz = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
            )[0]
        )
    )
    spectral_bandwidth_hz = float(
        np.mean(
            librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
            )[0]
        )
    )

    return {
        "rms_energy": rms_energy,
        "silence_pct": silence_pct,
        "snr_db_proxy": snr_db_proxy,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_bandwidth_hz": spectral_bandwidth_hz,
    }
