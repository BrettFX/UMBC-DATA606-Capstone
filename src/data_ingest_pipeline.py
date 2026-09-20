"""Data ingestion pipeline for the ATC speech corpora used in this capstone.

Loads the ATC-ASR-Dataset (real ATC utterances) and ATCOSIM (simulated ATC
utterances) datasets from Hugging Face, standardizes them onto a common
schema, resamples audio to a common sample rate, and derives
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
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import Audio, DatasetDict, Value, concatenate_datasets, load_dataset

# Hugging Face repo IDs, keyed by the short `dataset_source` name used
# throughout this project.
#
# `atc-asr-dataset` replaces `jlvdoorn/atco2-asr` (rather than supplementing
# it): its dataset card states it's built in part from the "ATCO2 1-Hour Test
# Subset", the same public release `atco2-asr` wraps, so running both would
# risk duplicate/overlapping utterances (and possible train/test leakage)
# between "sources" that are really the same recordings.
DEFAULT_SOURCES = {
    "atcosim": "jlvdoorn/atcosim",
    "atc-asr-dataset": "jacktol/ATC-ASR-Dataset"
}

# ASR-standard sample rate (also ATCO2-ASR's native rate; Whisper/Wav2Vec2
# and most pretrained speech models expect 16kHz mono input).
DEFAULT_TARGET_SAMPLE_RATE = 16_000

# ATCO2-ASR filenames embed a recording date/time, e.g.
# `LKPR_RUZYNE_Radar_120_520MHz_20201025_091112.wav`. ATCOSIM's generic
# session-based filenames (e.g. `gf1_01_001.wav`) don't match this pattern.
_RECORDING_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")

# ATCOSIM's filenames are `<speaker>_<session>_<utterance>.wav` (e.g.
# `gf1_01_001.wav`, `gf1_01_002.wav`, ...): utterances sharing a
# `<speaker>_<session>` prefix are clips from the same recording. Grouping by
# this key before re-splitting keeps same-session utterances from straddling
# a split boundary. `atc-asr-dataset`'s row `id`s (e.g. `01d7425638602ab696a4`)
# are opaque hashes with no recoverable session info, so it isn't re-split at
# all (see `resplit_sources`/`_resplit`) -- its published train/validation/test
# split (already ~80/10/10, per its dataset card manually filtered/cleaned) is
# kept as-is rather than risk reintroducing leakage a random reshuffle can't
# detect.
_SESSION_KEY_PATTERNS = {
    "atcosim": re.compile(r"^(.+)_\d+$"),
}


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
        resplit_sources: Sources to re-split into a fresh `train`/
            `validation`/`test` partition (`test_size`/`val_size` of the
            source's total each), grouped by recording session so utterances
            from the same session all land in the same split (see
            `_SESSION_KEY_PATTERNS`). Sources not listed here keep their
            originally published split as-is -- do not add a source here
            without also adding its session-grouping pattern to
            `_SESSION_KEY_PATTERNS`, or the group-by-session leakage
            protection this exists for is silently skipped.
        test_size: Fraction of each `resplit_sources` source assigned to the
            new `test` split.
        val_size: Fraction of each `resplit_sources` source assigned to the
            new `validation` split.
        split_seed: Seed for the shuffle that assigns session groups to
            splits in `_resplit`, so re-running `run(force=True)` reproduces
            the same split.

    After `run()`, the original per-source dataset counts (pre-join) are
    available on `source_counts_`, and a per-(dataset_source, original_split)
    row-count summary (used to document class imbalance) is available on
    `validation_report_`. `original_split` in that report is the split as
    originally published by the source; the row's actual (post-`_resplit`)
    split is `dataset_split` in `utterance_df`.
    """

    data_dir: str = "../data"
    sources: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SOURCES))
    target_sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE
    num_proc: Optional[int] = None
    resplit_sources: tuple[str, ...] = ("atcosim",)
    test_size: float = 0.1
    val_size: float = 0.1
    split_seed: int = 42

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
        datasets = self._resplit(datasets)
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

    def _resplit(self, datasets: Dict[str, DatasetDict]) -> Dict[str, DatasetDict]:
        """Re-split each `resplit_sources` source into a fresh, session-grouped
        `train`/`validation`/`test` partition; leave every other source's
        published split untouched.

        Pooling every source's own splits and randomly reshuffling rows
        (the simplest way to give the combined corpus a `test` split that
        spans all sources) would risk leaking same-recording-session
        utterances across the new split boundaries for any source whose rows
        aren't independent -- exactly the failure mode a published split is
        usually curated to avoid. So only sources in `resplit_sources` (ones
        with a recoverable session id, per `_SESSION_KEY_PATTERNS`) are
        reshuffled; the rest keep whatever split their publisher assigned.
        """
        resplit = dict(datasets)
        for source_name in self.resplit_sources:
            if source_name not in resplit:
                continue
            pattern = _SESSION_KEY_PATTERNS.get(source_name)
            if pattern is None:
                raise NotImplementedError(
                    f"{source_name!r} is in resplit_sources but has no entry in "
                    "_SESSION_KEY_PATTERNS. Re-splitting it without a session "
                    "grouping key would risk leaking same-recording utterances "
                    "across train/validation/test; add a pattern (or remove it "
                    "from resplit_sources to keep its published split as-is)."
                )
            dsd = resplit[source_name]
            flat = concatenate_datasets([dsd[split_name] for split_name in sorted(dsd.keys())])

            # `decode=False` reads each row's filename without decoding its
            # audio -- cheap, since it skips the librosa/soundfile resample
            # `_add_utterance_metadata` later does for real.
            audio_paths = flat.cast_column("audio", Audio(decode=False))["audio"]
            groups = [self._session_group(p["path"], pattern) for p in audio_paths]

            assignment = self._group_split(groups, self.test_size, self.val_size, self.split_seed)
            flat = flat.add_column("_resplit_assignment", assignment)
            resplit[source_name] = DatasetDict(
                {
                    split_name: flat.filter(
                        lambda row, split_name=split_name: row["_resplit_assignment"] == split_name
                    ).remove_columns("_resplit_assignment")
                    for split_name in ("train", "validation", "test")
                }
            )
        return resplit

    @staticmethod
    def _session_group(audio_path: Optional[str], pattern: re.Pattern) -> str:
        """Return `audio_path`'s session-grouping key, or `audio_path` itself
        (i.e. treat it as its own singleton group) if `pattern` doesn't match."""
        stem = Path(audio_path).stem if audio_path else ""
        match = pattern.match(stem)
        return match.group(1) if match else stem

    @staticmethod
    def _group_split(
        groups: List[str], test_size: float, val_size: float, seed: int
    ) -> List[str]:
        """Assign every row to `train`/`validation`/`test` by its `groups`
        value, so all rows sharing a group land in the same split.

        Shuffles the unique groups (seeded, for reproducibility) and greedily
        fills `test` then `validation` up to `test_size`/`val_size` of the
        total row count, assigning whatever's left to `train`. Exact
        proportions aren't guaranteed (groups aren't split), but with many
        small, similarly-sized groups (as ATCOSIM's ~50 recording sessions
        are) the result lands close to the target split.
        """
        group_sizes = Counter(groups)
        unique_groups = list(group_sizes.keys())
        np.random.default_rng(seed).shuffle(unique_groups)

        n_total = len(groups)
        target_test = round(n_total * test_size)
        target_val = round(n_total * val_size)

        split_by_group = {}
        n_test = n_val = 0
        for group in unique_groups:
            size = group_sizes[group]
            if n_test < target_test:
                split_by_group[group] = "test"
                n_test += size
            elif n_val < target_val:
                split_by_group[group] = "validation"
                n_val += size
            else:
                split_by_group[group] = "train"
        return [split_by_group[group] for group in groups]

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
        """Concatenate every source's splits into one DatasetDict.

        Uses the union (not intersection) of split names: a source missing
        a given split (e.g. ATCOSIM has no `test` split) simply contributes
        no rows to it, rather than that split being dropped entirely.
        """
        split_names = set.union(*(set(dsd.keys()) for dsd in datasets.values()))
        return DatasetDict(
            {
                split_name: concatenate_datasets(
                    [dsd[split_name] for dsd in datasets.values() if split_name in dsd]
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
