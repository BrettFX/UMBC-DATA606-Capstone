"""Data ingestion pipeline for the ATC speech corpora used in this capstone.

Loads the ATCO2-ASR and ATCOSIM datasets from Hugging Face, standardizes them
onto a common schema, resamples audio to a common sample rate, and derives
the utterance-level metadata (`utterance_id`, `duration_sec`, `word_count`,
`speech_rate_wpm`, etc.) documented in `docs/proposal.md` section 3.5
("Dataset Integration").

Example:
    >>> from data_ingest_pipeline import DataIngestPipeline
    >>> pipeline = DataIngestPipeline(data_dir="../data")
    >>> combined_dataset, utterance_df = pipeline.run()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from datasets import Audio, DatasetDict, Value, concatenate_datasets, load_dataset

# Hugging Face repo IDs, keyed by the short `dataset_source` name used
# throughout this project.
DEFAULT_SOURCES = {
    "atco2-asr": "jlvdoorn/atco2-asr",
    "atcosim": "jlvdoorn/atcosim",
}

# ASR-standard sample rate (also ATCO2-ASR's native rate; Whisper/Wav2Vec2
# and most pretrained speech models expect 16kHz mono input).
DEFAULT_TARGET_SAMPLE_RATE = 16_000

# ATCO2-ASR filenames embed a recording date/time, e.g.
# `LKPR_RUZYNE_Radar_120_520MHz_20201025_091112.wav`. ATCOSIM's generic
# session-based filenames (e.g. `gf1_01_001.wav`) don't match this pattern.
_RECORDING_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")


def _parse_recorded_at(audio_path: Optional[str]) -> Optional[str]:
    """Parse a `YYYYMMDD_HHMMSS` timestamp out of an audio filename, if present.

    Returns an ISO-8601 string (e.g. `"2020-10-25T09:09:15"`) for ATCO2-ASR
    rows, or `None` for rows (e.g. ATCOSIM's) whose filename doesn't embed one.
    """
    if not audio_path:
        return None
    match = _RECORDING_TIMESTAMP_RE.search(audio_path)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(0), "%Y%m%d_%H%M%S").isoformat()
    except ValueError:
        return None


@dataclass
class DataIngestPipeline:
    """Loads, joins, resamples, and enriches the ATC speech datasets.

    Attributes:
        data_dir: Root data directory. Raw downloads are cached under
            `<data_dir>/raw/<dataset_source>/`, and processed output is
            optionally saved under `<data_dir>/processed/`.
        sources: Mapping of `dataset_source` name -> Hugging Face repo ID.
        target_sample_rate: Sample rate (Hz) every audio row is resampled to.
        num_proc: Worker processes for the metadata-derivation map step.
            `None` runs single-process.

    After `run()`, the original per-source dataset counts (pre-join) are
    available on `source_counts_`, and a per-(dataset_source, original_split)
    row-count summary (used to document class imbalance) is available on
    `validation_report_`.
    """

    data_dir: str = "../data"
    sources: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SOURCES))
    target_sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE
    num_proc: Optional[int] = None

    source_counts_: Dict[str, Dict[str, int]] = field(default_factory=dict, init=False)
    validation_report_: Optional[pd.DataFrame] = field(default=None, init=False)

    def run(
        self, save_to_disk: bool = True, force: bool = False
    ) -> tuple[DatasetDict, pd.DataFrame]:
        """Execute the full ingestion pipeline end to end.

        The audio decode + resample step is the expensive part (roughly
        5-6 minutes single-process for the ~10k rows across both source
        datasets). Pass `num_proc` to the constructor to parallelize it.
        By default this method reuses a previous run's saved output instead
        of repeating that cost; pass `force=True` to always recompute.

        Args:
            save_to_disk: If True, persist the outputs under
                `<data_dir>/processed/` (`combined_dataset/` via
                `DatasetDict.save_to_disk`, `utterance_df.parquet` via
                `DataFrame.to_parquet`) so downstream notebooks/scripts don't
                need to re-download and re-resample every run.
            force: If False (default) and a previous `save_to_disk=True` run's
                output already exists under `<data_dir>/processed/`, load and
                return that instead of recomputing. If True, always re-run
                the full pipeline (e.g. after changing `target_sample_rate`
                or `sources`).

        Returns:
            `(combined_dataset, utterance_df)`:
              - `combined_dataset`: a `DatasetDict` (train/validation) with
                every source concatenated, audio resampled to
                `target_sample_rate`, and per-row metadata columns added.
                Training-ready, and includes the decoded `audio` column.
              - `utterance_df`: a flat `pandas.DataFrame` across all splits
                with one row per utterance and no raw audio arrays.
                Analysis-ready, matching the data dictionary in
                `docs/proposal.md` section 3.5.
        """
        if not force:
            cached = self._load_cached()
            if cached is not None:
                combined_dataset, utterance_df = cached
                self._validate(combined_dataset, utterance_df)
                return combined_dataset, utterance_df

        datasets = self._load()
        datasets = self._standardize_schema(datasets)
        datasets = self._resample(datasets)
        combined_dataset = self._join(datasets)
        combined_dataset = self._add_utterance_metadata(combined_dataset)
        utterance_df = self._to_utterance_dataframe(combined_dataset)
        self._validate(combined_dataset, utterance_df)

        if save_to_disk:
            self._save(combined_dataset, utterance_df)

        return combined_dataset, utterance_df

    def _load_cached(self) -> Optional[tuple[DatasetDict, pd.DataFrame]]:
        """Return a previously saved `(combined_dataset, utterance_df)` from
        `<data_dir>/processed/`, or None if no cached output exists yet."""
        processed_dir = Path(self.data_dir) / "processed"
        combined_path = processed_dir / "combined_dataset"
        utterance_path = processed_dir / "utterance_df.parquet"
        if not combined_path.exists() or not utterance_path.exists():
            return None
        combined_dataset = DatasetDict.load_from_disk(str(combined_path))
        utterance_df = pd.read_parquet(utterance_path)
        return combined_dataset, utterance_df

    def _load(self) -> Dict[str, DatasetDict]:
        """Download (or load from local cache) each source dataset."""
        datasets = {}
        for source_name, repo_id in self.sources.items():
            cache_dir = str(Path(self.data_dir) / "raw" / source_name)
            dsd = load_dataset(repo_id, cache_dir=cache_dir)
            datasets[source_name] = dsd
            self.source_counts_[source_name] = {split: len(ds) for split, ds in dsd.items()}
        return datasets

    def _standardize_schema(self, datasets: Dict[str, DatasetDict]) -> Dict[str, DatasetDict]:
        """Tag rows with provenance/a reproducible ID and align columns.

        Adds `dataset_source`, `original_split`, and a reproducible
        `utterance_id` (`"<dataset_source>-<original_split>-<row index>"`)
        to every row, then backfills any column present in one source but
        not another (e.g. ATCOSIM has no `info` column) as an all-null
        string column so every source shares one schema before concatenation.
        """
        all_columns = {
            col for dsd in datasets.values() for col in dsd["train"].column_names
        }

        standardized = {}
        for source_name, dsd in datasets.items():
            tagged = DatasetDict()
            for split_name, split in dsd.items():
                split = split.map(
                    lambda _example, idx, source_name=source_name, split_name=split_name: {
                        "dataset_source": source_name,
                        "original_split": split_name,
                        "utterance_id": f"{source_name}-{split_name}-{idx:06d}",
                    },
                    with_indices=True,
                )
                for column in sorted(all_columns - set(split.column_names)):
                    split = split.map(lambda _example, column=column: {column: None})
                    split = split.cast_column(column, Value("string"))
                tagged[split_name] = split
            standardized[source_name] = tagged
        return standardized

    def _resample(self, datasets: Dict[str, DatasetDict]) -> Dict[str, DatasetDict]:
        """Cast the `audio` column to `target_sample_rate` on every split.

        `cast_column` is lazy: it only changes the feature's declared
        sample rate; the actual resampling happens on demand (via
        librosa/soundfile) the next time a row's `audio` is decoded, e.g.
        inside `_add_utterance_metadata`'s map below.
        """
        for dsd in datasets.values():
            for split_name in dsd:
                dsd[split_name] = dsd[split_name].cast_column(
                    "audio", Audio(sampling_rate=self.target_sample_rate)
                )
        return datasets

    def _join(self, datasets: Dict[str, DatasetDict]) -> DatasetDict:
        """Concatenate every source's matching splits into one DatasetDict."""
        split_names = set.intersection(*(set(dsd.keys()) for dsd in datasets.values()))
        return DatasetDict(
            {
                split_name: concatenate_datasets(
                    [dsd[split_name] for dsd in datasets.values()]
                )
                for split_name in sorted(split_names)
            }
        )

    def _add_utterance_metadata(self, combined: DatasetDict) -> DatasetDict:
        """Derive `audio_path`, `duration_sec`, `sample_rate`, `word_count`,
        `character_count`, `speech_rate_wpm`, and `recorded_at` from each
        row's (now-resampled) audio + text.

        This is the step that actually triggers audio decode/resample, so
        it's the expensive part of the pipeline, batched and, if
        `num_proc` is set, parallelized.
        """

        def _derive(batch):
            audio_paths, durations, sample_rates = [], [], []
            word_counts, char_counts, speech_rates, recorded_ats = [], [], [], []
            for audio, text in zip(batch["audio"], batch["text"]):
                audio_paths.append(audio["path"])
                sample_rates.append(audio["sampling_rate"])
                duration = len(audio["array"]) / audio["sampling_rate"]
                durations.append(duration)
                text = text or ""
                word_count = len(text.split())
                word_counts.append(word_count)
                char_counts.append(len(text))
                # Words per minute; guards against a divide-by-zero on a
                # (currently nonexistent, see Data Quality Checks) zero-duration row.
                speech_rates.append(word_count / (duration / 60) if duration > 0 else float("nan"))
                recorded_ats.append(_parse_recorded_at(audio["path"]))
            return {
                "audio_path": audio_paths,
                "duration_sec": durations,
                "sample_rate": sample_rates,
                "word_count": word_counts,
                "character_count": char_counts,
                "speech_rate_wpm": speech_rates,
                "recorded_at": recorded_ats,
            }

        for split_name in combined:
            combined[split_name] = combined[split_name].map(
                _derive, batched=True, num_proc=self.num_proc
            )
            # `recorded_at` is all-None for entire batches of ATCOSIM rows;
            # without an explicit cast, Arrow can infer a `null` type for
            # those batches vs. `string` for ATCO2-ASR's, and then choke on
            # the mismatch when the split's underlying table is finalized
            # (same failure mode `_standardize_schema` guards against for
            # the backfilled `info` column).
            combined[split_name] = combined[split_name].cast_column(
                "recorded_at", Value("string")
            )
        return combined

    def _to_utterance_dataframe(self, combined: DatasetDict) -> pd.DataFrame:
        """Flatten every split into one lightweight, audio-free DataFrame.

        Matches the "standardized utterance-level analytical dataset" data
        dictionary in `docs/proposal.md` section 3.5 (renaming `text` ->
        `transcript` to match it); raw audio arrays are dropped here on
        purpose. Use `combined_dataset` for anything that needs the audio.
        """
        frames = [
            split.remove_columns("audio").to_pandas().assign(dataset_split=split_name)
            for split_name, split in combined.items()
        ]
        df = pd.concat(frames, ignore_index=True)
        df = df.rename(columns={"text": "transcript"})
        column_order = [
            "utterance_id",
            "dataset_source",
            "original_split",
            "audio_path",
            "transcript",
            "duration_sec",
            "sample_rate",
            "word_count",
            "character_count",
            "speech_rate_wpm",
            "info",
            "recorded_at",
        ]
        remaining = [c for c in df.columns if c not in column_order]
        return df[[c for c in column_order if c in df.columns] + remaining]

    def _validate(self, combined: DatasetDict, utterance_df: pd.DataFrame) -> None:
        """Check the join didn't drop/duplicate rows and record source
        imbalance for `validation_report_`.

        Skips the cross-check against `source_counts_` when loading from a
        disk cache (`_load()` never ran, so there's nothing fresh to compare
        against); it still checks `combined_dataset`/`utterance_df` agree
        with each other and that `utterance_id` is unique either way.

        Raises:
            AssertionError: if the combined row count doesn't match the sum
                of the original per-source, per-split row counts, or if
                `combined_dataset` and `utterance_df` disagree on row count.
        """
        actual_total = sum(len(split) for split in combined.values())
        if self.source_counts_:
            expected_total = sum(
                n for split_counts in self.source_counts_.values() for n in split_counts.values()
            )
            assert actual_total == expected_total, (
                f"Combined row count ({actual_total}) does not match the sum of "
                f"source dataset row counts ({expected_total}); a row was lost "
                f"or duplicated during the join."
            )
        assert len(utterance_df) == actual_total, (
            f"utterance_df row count ({len(utterance_df)}) does not match "
            f"combined_dataset row count ({actual_total})."
        )
        assert utterance_df["utterance_id"].is_unique, "utterance_id is not unique."

        report = (
            utterance_df.groupby(["original_split", "dataset_source"])
            .size()
            .rename("row_count")
            .reset_index()
        )
        report["pct_of_split"] = (
            100 * report["row_count"] / report.groupby("original_split")["row_count"].transform("sum")
        ).round(1)
        self.validation_report_ = report

    def _save(self, combined: DatasetDict, utterance_df: pd.DataFrame) -> None:
        """Persist both outputs under `<data_dir>/processed/`."""
        processed_dir = Path(self.data_dir) / "processed"
        combined.save_to_disk(str(processed_dir / "combined_dataset"))
        utterance_df.to_parquet(processed_dir / "utterance_df.parquet", index=False)
