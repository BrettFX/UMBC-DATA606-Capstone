# src

Reusable project code, importable from notebooks via `sys.path.append("../src")`.

## `data_ingest_pipeline.py`

`DataIngestPipeline` loads the ATCO2-ASR and ATCOSIM datasets from Hugging Face, tags each
row with `dataset_source`/`original_split`/a reproducible `utterance_id`, aligns their schemas,
resamples all audio to a common sample rate, and derives per-utterance metadata
(`duration_sec`, `sample_rate`, `word_count`, `character_count`, `speech_rate_wpm`,
`recorded_at` for ATCO2-ASR only, parsed from its filenames). Used from
[notebooks/allen_data_606_proposal_eda.ipynb](../notebooks/allen_data_606_proposal_eda.ipynb),
and intended to be reused by model-training code later so ingestion logic isn't duplicated.

```python
from data_ingest_pipeline import DataIngestPipeline

pipeline = DataIngestPipeline(data_dir="../data")
combined_dataset, utterance_df = pipeline.run()
```

`run()` caches its output under `data/processed/` (see [../data/README.md](../data/README.md))
so the ~5-6 minute audio decode/resample step only runs once; pass `force=True` to recompute.
