"""Canonical final schema for `utterance_df`, kept as executable
documentation (a data dictionary) rather than a prose-only table, so drift
between what the pipeline actually produces and what's documented is
caught by `validate_final_schema` instead of found by hand later.
"""

from __future__ import annotations

import pandas as pd

# Column order + one-line definitions. `info` (ATCO2-ASR-specific) is
# deliberately absent: neither current source (ATCOSIM, ATC-ASR-Dataset)
# populates it, so it's dead weight now that ATCO2-ASR has been replaced.
FINAL_UTTERANCE_COLUMNS = {
    "utterance_id": (
        "Reproducible id assigned by DataIngestPipeline: "
        "'<dataset_source>-<original_split>-<row index>'."
    ),
    "dataset_source": "Which corpus this row came from: 'atcosim' or 'atc-asr-dataset'.",
    "original_split": (
        "The split as originally published by the source, before this "
        "project's own resplit."
    ),
    "dataset_split": (
        "This project's own train/validation/test split "
        "(see DataIngestPipeline._resplit)."
    ),
    "source_id": (
        "The publisher's own row id, where available (atc-asr-dataset only; "
        "null for atcosim). Not used for joins/dedup -- `utterance_id`/"
        "`audio_checksum` are canonical for that."
    ),
    "audio_path": "Original filename of the row's audio clip.",
    "audio_checksum": (
        "sha1 of the decoded waveform's bytes; canonical audio-identity key "
        "for dedup."
    ),
    "audio_decode_error": (
        "None if the row's audio decoded cleanly; the exception message "
        "otherwise."
    ),
    "transcript": "Raw transcript text as published by the source.",
    "transcript_normalized": (
        "Lowercased, whitespace/punctuation-normalized transcript; "
        "canonical text-identity key for dedup."
    ),
    "duration_sec": "Audio duration in seconds, from the decoded waveform.",
    "sample_rate": (
        "Audio sample rate in Hz after resampling "
        "(see DataIngestPipeline.target_sample_rate); 0 for a corrupt row."
    ),
    "word_count": "Whitespace-delimited token count of `transcript`.",
    "character_count": "Character count of `transcript`.",
    "speech_rate_wpm": (
        "word_count / (duration_sec / 60); NaN if duration_sec is non-positive."
    ),
    "numeric_token_count": (
        "Count of ATC numeral-word tokens in `transcript_normalized` "
        "(see preprocessing.vocab)."
    ),
    "command_verb_count": (
        "Count of ATC command-verb tokens in `transcript_normalized`; a "
        "proxy pending real NER (deferred)."
    ),
    "callsign_like_count": (
        "Count of callsign-like token runs in `transcript_normalized`; a "
        "proxy pending real NER (deferred)."
    ),
    "rms_energy": (
        "Mean per-frame RMS amplitude (unitless, waveform in [-1, 1]); "
        "loudness proxy."
    ),
    "silence_pct": (
        "Percent of the clip librosa.effects.split(top_db=30) marks as "
        "below-threshold."
    ),
    "snr_db_proxy": (
        "P95-P10 gap of per-frame RMS-in-dB; a heuristic dynamic-range "
        "proxy, NOT a physical SNR measurement -- comparative use only."
    ),
    "zero_crossing_rate": (
        "Mean per-frame zero-crossing rate; noisiness/unvoiced-speech proxy."
    ),
    "spectral_centroid_hz": (
        "Mean per-frame amplitude-weighted mean frequency (Hz); 'brightness'."
    ),
    "spectral_bandwidth_hz": "Mean per-frame amplitude-weighted frequency spread (Hz).",
    "recorded_at": (
        "ISO-8601 recording timestamp parsed from the filename, where the "
        "source's filenames embed one; null otherwise (structural, not "
        "missing data -- see DataIngestPipeline._parse_recorded_at)."
    ),
    "is_repeated_transcript": (
        "True if `transcript_normalized` recurs across 2+ rows with "
        "*different* audio content (expected fixed ATC phraseology; never "
        "dropped -- see preprocessing.quality)."
    ),
    "repeated_transcript_count": (
        "How many rows (post-cleaning) share this row's `transcript_normalized`."
    ),
}


def validate_final_schema(df: pd.DataFrame) -> None:
    """Raise if `df`'s columns don't match `FINAL_UTTERANCE_COLUMNS` exactly.

    Catches schema drift early: a retired column (e.g. `info`) sneaking
    back in via `_standardize_schema`'s generic backfill, or a
    `compute_*_features` typo silently omitting an expected column.
    """
    expected = set(FINAL_UTTERANCE_COLUMNS)
    actual = set(df.columns)
    missing = expected - actual
    unexpected = actual - expected
    assert not missing and not unexpected, (
        f"utterance_df schema drift -- missing columns: {sorted(missing) or 'none'}; "
        f"unexpected columns: {sorted(unexpected) or 'none'}."
    )
