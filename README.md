# UMBC-DATA606-Capstone

## TODO
- [x] Create run script for kicking off data curation and ingest process so that it doesn't have to be ran via notebook.
- [x] Add optional post-process purge flag so that large raw datasets are removed from local file system after processing is complete.

## Running the pipeline from the command line

`run-pipeline.sh` is the single entry point for the project's pipeline (data ingest today;
training and inference will be added as their own stages later). It walks through each
stage interactively, asking whether to run it and what arguments to pass:

```bash
./run-pipeline.sh
```

Each stage's questions map directly to its underlying script's CLI flags, so the same
questions will apply once training/inference are wired in.

### Running data ingestion directly

On machines with less RAM than `notebooks/comprehensive_eda.ipynb` needs (that notebook's
plots, audio playback widgets, and word clouds all sit in the same kernel as the pipeline's
own data), the ingest stage can also be run directly, non-interactively:

```bash
python scripts/run_ingest.py                    # uses cached output under data/processed/ if present
python scripts/run_ingest.py --force             # recompute from scratch
python scripts/run_ingest.py --num-proc 4        # parallelize the metadata-derivation step
python scripts/run_ingest.py --purge-raw         # delete data/raw/<source>/ after a successful run
```

See `python scripts/run_ingest.py --help` for all options.
