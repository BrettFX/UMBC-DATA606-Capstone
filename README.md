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

## Getting and publishing the trained models (S3)

The best ASR and NER models are stored in S3 at
`s3://endurasoft-dev-ml-ops/ml-tasks/<task>/<model-name>/<run-id>/`, with a `latest/` copy of the
run that inference should use (for example `ml-tasks/asr/lora-whisper-medium-en/latest/` and
`ml-tasks/ner/spacy-balanced/latest/`). Two driver scripts wrap `scripts/model_store.py`; they need AWS
credentials (environment or `~/.aws`) and are safe to re-run.

```bash
./download-models.sh --profile cpu # a device WITHOUT a GPU: CTranslate2 int8 ASR model + NER (~0.8 GB, no torch needed)
./download-models.sh --profile gpu # a device with a GPU: full-precision ASR model + NER (~1.5 GB)
./download-models.sh               # everything (full ASR model, CTranslate2 ASR model, NER)
./download-models.sh --only ner    # one group (asr, asr-ct2, ner); --dry-run shows the plan; --run-id X fetches a specific run
./upload-models.sh                 # publish the current best models as new runs and update latest (asks first)
./upload-models.sh --dry-run       # show the plan only; a model that already matches latest is skipped
```

The ASR model is published twice: the merged full-precision model (`asr/lora-whisper-medium-en`) and its
CTranslate2 int8 conversion (`asr/lora-whisper-medium-en-ct2-int8`), which is what a CPU device runs
(`asr.inference.best_transcriber()` picks the right one; see `res/benchmarks/EXPERIMENTS.md` for why).
Which local models count as "the best" is a short list in `scripts/lib/model_common.sh`; change a row
there when a better model replaces one. `python scripts/model_store.py --help` has the lower-level commands
(`list`, `promote`, per-run `upload`/`download`).
