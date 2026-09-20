"""Reusable ingestion/feature/quality-check logic for the ATC speech EDA
and (later) model-training pipeline.

Public API:
    - `DataIngestPipeline`: load, join, resample, enrich, and clean the
      source datasets end to end (see `ingest.py`).
    - `run_quality_checks`: the auditable data-quality pass `DataIngestPipeline`
      runs internally, importable directly for ad hoc use (see `quality.py`).
    - `compute_acoustic_features` / `compute_transcript_features`: the
      per-row feature functions `DataIngestPipeline` calls internally,
      importable directly for ad hoc/exploratory use on a single clip.
    - `FINAL_UTTERANCE_COLUMNS`: the data dictionary for `utterance_df`
      (see `schema.py`).
"""

from .audio_features import compute_acoustic_features
from .ingest import DataIngestPipeline
from .quality import run_quality_checks
from .schema import FINAL_UTTERANCE_COLUMNS, validate_final_schema
from .text_features import compute_transcript_features

__all__ = [
    "DataIngestPipeline",
    "run_quality_checks",
    "compute_acoustic_features",
    "compute_transcript_features",
    "FINAL_UTTERANCE_COLUMNS",
    "validate_final_schema",
]
