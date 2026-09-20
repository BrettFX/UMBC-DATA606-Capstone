# src

Reusable project code, importable from notebooks via `sys.path.append("../src")`.

## `preprocessing/`

- **`ingest.py`** — `DataIngestPipeline` loads the ATCOSIM (simulated) and ATC-ASR-Dataset
  (real) corpora from Hugging Face, tags each row with `dataset_source`/`original_split`/a
  reproducible `utterance_id`, aligns their schemas, re-splits ATCOSIM by recording session
  (train/validation/test, avoiding same-session leakage — see `ingest._SESSION_KEY_PATTERNS`),
  resamples all audio to a common sample rate, derives per-utterance acoustic + text features,
  and runs auditable data-quality checks. Intended to be reused by model-training code later
  so ingestion logic isn't duplicated.
- **`audio_features.py`** — `compute_acoustic_features(y, sr)`: RMS energy, silence %, an SNR
  dynamic-range proxy, zero-crossing rate, spectral centroid/bandwidth. Pure numpy/librosa, no
  `datasets` dependency.
- **`text_features.py`** / **`vocab.py`** — `normalize_transcript`, plus lightweight proxy
  counts (numeric tokens, ATC command verbs, callsign-like sequences) standing in for the
  deferred, ground-truth NER work.
- **`quality.py`** — `run_quality_checks(df)`: missing audio/transcript, zero-duration, corrupt
  audio, and duplicate-row checks (all dropped), plus repeated-transcript detection (flagged,
  never dropped — see the module docstring for why duplicate *audio* and repeated *phrasing*
  are treated differently). Returns a cleaned df, the dropped rows, and an auditable stage-by-
  stage cleaning log.
- **`schema.py`** — `FINAL_UTTERANCE_COLUMNS`: the data dictionary for `utterance_df`, plus
  `validate_final_schema` to catch drift.

```python
from preprocessing import DataIngestPipeline

pipeline = DataIngestPipeline(data_dir="../data")
combined_dataset, utterance_df = pipeline.run()
```

`run()` caches its output under `data/processed/` (see [../data/README.md](../data/README.md))
so the expensive audio decode/resample/feature-extraction step only runs once; pass
`force=True` to recompute. Reproducing from scratch: `pip install -r requirements.txt`, then
`rm -rf <data_dir>/processed` (raw HF downloads under `<data_dir>/raw/` are untouched, cached
inputs — leave them), then `DataIngestPipeline(data_dir=...).run(force=True)`.
