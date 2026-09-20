"""Reusable, auditable data-quality checks for `utterance_df`.

Formalizes the ad hoc checks `notebooks/proposal_eda.ipynb` already ran by
hand (missing audio/transcript, zero-duration rows, duplicate/repeated
transcripts) into reusable functions behind a single entry point,
`run_quality_checks`, that produces a cleaned DataFrame *and* an auditable
cleaning log -- nothing is dropped without a corresponding logged reason.

The key distinction this module encodes (`check_duplicate_row` vs.
`check_repeated_transcript`): a duplicate *audio recording* is an
accidental data-quality issue (drop it), but a duplicate *transcript*
alone is expected -- ATC uses fixed phraseology, so many distinct
recordings legitimately share the same wording (e.g. "contact milan one
three four five two good bye" recurring across unrelated sessions). The
fork is audio identity (`audio_checksum`), not transcript identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple

import pandas as pd

CLEANING_LOG_COLUMNS = [
    "stage",
    "check_description",
    "rows_in",
    "rows_flagged",
    "rows_removed",
    "rows_out",
    "action",
    "notes",
]


@dataclass
class QualityCheckResult:
    name: str
    description: str
    mask: pd.Series  # boolean, aligned to the df it was computed against; True = flagged
    action: Literal["drop", "flag_only"]
    notes: str = ""


def check_missing_audio(df: pd.DataFrame) -> QualityCheckResult:
    mask = (
        df["audio_path"].isna()
        | (df["audio_path"].astype(str).str.strip() == "")
        | df["audio_decode_error"].notna()
    )
    return QualityCheckResult(
        "missing_audio",
        "Audio path is null/blank, or the row's audio failed to decode.",
        mask,
        "drop",
        "Rows with no usable audio can't be used for ASR training or acoustic feature EDA.",
    )


def check_missing_transcript(df: pd.DataFrame) -> QualityCheckResult:
    mask = df["transcript"].isna() | (df["transcript"].astype(str).str.strip() == "")
    return QualityCheckResult(
        "missing_transcript",
        "Transcript is null or blank after stripping whitespace.",
        mask,
        "drop",
        "Rows with no transcript can't be used for ASR training or text feature EDA.",
    )


def check_zero_or_negative_duration(df: pd.DataFrame) -> QualityCheckResult:
    mask = df["duration_sec"].isna() | (df["duration_sec"] <= 0)
    return QualityCheckResult(
        "zero_or_negative_duration",
        "Derived audio duration is missing, zero, or negative.",
        mask,
        "drop",
        "A non-positive duration means the audio decode produced no usable samples.",
    )


def check_corrupt_audio(df: pd.DataFrame) -> QualityCheckResult:
    mask = df["audio_decode_error"].notna()
    return QualityCheckResult(
        "corrupt_audio",
        "Audio failed to decode (see `audio_decode_error` for the exception).",
        mask,
        "drop",
        "Kept separate from `missing_audio` so the cleaning log shows *why* audio was unusable.",
    )


def check_duplicate_row(df: pd.DataFrame) -> QualityCheckResult:
    mask = df.duplicated(subset=["audio_checksum", "transcript_normalized"], keep="first")
    return QualityCheckResult(
        "duplicate_row",
        "Same audio content AND same normalized transcript as an earlier row.",
        mask,
        "drop",
        "A true accidental duplicate (e.g. the same clip published under two ids) -- distinct "
        "from `repeated_transcript`, which is expected, meaning-bearing repeated ATC phraseology.",
    )


def check_repeated_transcript(df: pd.DataFrame) -> QualityCheckResult:
    same_text = df.duplicated(subset=["transcript_normalized"], keep=False)
    same_audio_and_text = df.duplicated(
        subset=["audio_checksum", "transcript_normalized"], keep=False
    )
    mask = same_text & ~same_audio_and_text
    return QualityCheckResult(
        "repeated_transcript",
        "Same normalized transcript as another row, but different audio content.",
        mask,
        "flag_only",
        "Fixed ATC phraseology recurring across distinct recordings -- retained unconditionally, "
        "annotated via `is_repeated_transcript`/`repeated_transcript_count` rather than dropped.",
    )


DEFAULT_QUALITY_CHECKS: List[Callable[[pd.DataFrame], QualityCheckResult]] = [
    check_missing_audio,
    check_missing_transcript,
    check_zero_or_negative_duration,
    check_corrupt_audio,
    check_duplicate_row,
    check_repeated_transcript,
]


def run_quality_checks(
    df: pd.DataFrame,
    checks: Optional[List[Callable[[pd.DataFrame], QualityCheckResult]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply `checks` in order, each against the rows still surviving after
    prior 'drop' stages, and return `(cleaned_df, dropped_df, cleaning_log_df)`.

    `cleaned_df` keeps every 'flag_only' hit (annotated via a boolean
    `is_<check name>` column) and excludes every 'drop' hit. `dropped_df`
    is every row any 'drop' check removed, tagged with which check removed
    it and why -- nothing is discarded unaudited.
    """
    checks = checks if checks is not None else DEFAULT_QUALITY_CHECKS
    working = df.copy()
    log_rows = []
    dropped_frames = []

    for check in checks:
        result = check(working)
        rows_in = len(working)
        rows_flagged = int(result.mask.sum())

        if result.action == "drop":
            if rows_flagged:
                dropped_frames.append(
                    working.loc[result.mask].assign(
                        dropped_by=result.name, drop_reason=result.notes
                    )
                )
            working = working.loc[~result.mask]
            rows_removed = rows_flagged
        else:
            working[f"is_{result.name}"] = False
            working.loc[result.mask, f"is_{result.name}"] = True
            if result.name == "repeated_transcript":
                working["repeated_transcript_count"] = working.groupby(
                    "transcript_normalized"
                )["transcript_normalized"].transform("count")
            rows_removed = 0

        log_rows.append(
            {
                "stage": result.name,
                "check_description": result.description,
                "rows_in": rows_in,
                "rows_flagged": rows_flagged,
                "rows_removed": rows_removed,
                "rows_out": len(working),
                "action": result.action,
                "notes": result.notes,
            }
        )

    cleaning_log_df = pd.DataFrame(log_rows, columns=CLEANING_LOG_COLUMNS)
    if dropped_frames:
        dropped_df = pd.concat(dropped_frames, ignore_index=True)
    else:
        dropped_df = df.iloc[0:0].assign(dropped_by=[], drop_reason=[])

    return working.reset_index(drop=True), dropped_df, cleaning_log_df
