# ATC Communications Intelligence

**UMBC DATA 606 Capstone, Fall 2026** | Brett Allen | Advisor: Dr. Chaojie (Jay) Wang

Speech recognition and information extraction for air traffic control (ATC) radio. The project fine-tunes Whisper on real and
simulated ATC audio, extracts nine kinds of aviation entities (callsigns, commands, altitudes, runways, ...) from the transcripts,
compares models by aviation-specific metrics as well as word error rate (WER), and ships the result as **ClearanceIQ**, a
Streamlit prototype that runs on a laptop CPU.

> **ClearanceIQ is an academic research prototype. It is not validated, certified or intended for operational use.**

| Deliverable | Where |
|---|---|
| Final report | [docs/report.md](docs/report.md) |
| Final presentation | [docs/Allen_DATA_606_Capstone_Final.pptx](docs/Allen_DATA_606_Capstone_Final.pptx) (earlier decks: [Progress Update](docs/Allen_DATA_606_Capstone_Progress_Update.pptx), [Proposal](docs/Allen_DATA_606_Capstone_Proposal.pptx)) |
| Notebooks | [notebooks/](notebooks/README.md): proposal EDA, comprehensive EDA, machine learning |
| Application | [app/](app/README.md), launched with `./run-app.sh` |
| Resume | [docs/Resume.md](docs/Resume.md) |

## Architecture

```
ATC audio -> audio prep (mono 16 kHz, trim, split at pauses) -> ASR (Whisper medium.en + LoRA)
          -> entity extraction (spaCy, 9 labels) -> clearance card / live transcript
          -> (optional) Claude plain-English explanation, Kokoro voice read-back
```

Data: ATC-ASR-Dataset (real radio) + ATCOSIM (simulated), 17,681 utterances, about 17.9 hours, split so that no recording session
crosses train/validation/test. Entity labels come from a staged process: an LLM teacher (Qwen3.5-9B) labels the corpus, humans verify
small evaluation sets, and a small spaCy student is trained on the curated labels.

## Headline results

Full 1,909-utterance test split, each model scored once (details, confidence intervals and caveats in the report, Sections 5 and 7).

| Question | Result |
|---|---|
| ASR accuracy (RQ1) | `whisper-medium.en` + LoRA: **3.8% WER overall, 7.5% on real audio** (zero-shot: 35.7% and 53.6%) |
| Aviation entities after ASR | Entity F1 0.935 (callsign recall 0.886, the weakest entity) |
| Entity extraction (RQ2) | Held-out span F1 0.878 (LLM teacher) and 0.861 (selected spaCy student, about 4 MB, about 3,700 utterances/s on a CPU) |
| What makes audio hard (RQ3) | Real vs. simulated audio and very short clips matter most; per-utterance correlations are weak (|rho| at most 0.17) |
| CPU deployment | About 0.7 s per utterance (real-time factor about 0.2), 2.2 GB RAM, CTranslate2 int8 |

## Repository structure

| Path | Contents |
|---|---|
| `docs/` | Report (`report.md`), presentations, data dictionaries, resume |
| `notebooks/` | Proposal EDA, comprehensive EDA, machine learning (prototype and production results) |
| `app/` | ClearanceIQ Streamlit app: `services/` (audio prep, models, pipeline), `ui/`, `samples/`, tests |
| `src/` | Reusable code: `preprocessing/` (ingest, quality checks, features), `asr/`, `ner/`, `artifacts.py`, `benchmarking.py` |
| `scripts/` | Command-line entry points (ingest, ASR training, NER annotate/curate/train, benchmarks, model store) |
| `tests/` | Unit tests for the pipelines and the download/setup scripts |
| `data/` | Documentation and the human-verified NER evaluation sets; raw and processed data are git-ignored ([data/README.md](data/README.md)) |
| `res/` | Figures used by the report and deck, benchmark logs, NER annotator experiment results |
| `setup.sh`, `run-app.sh`, `run-pipeline.sh`, `download-models.sh`, `upload-models.sh` | Environment setup, app launch, pipeline driver, model store |

## Launch ClearanceIQ

```bash
./setup.sh                              # once: conda env, system packages (espeak-ng, ffmpeg), Jupyter kernel
./download-models.sh --profile cpu      # once: ASR + NER models (S3, or the Google Drive bundle if you have no AWS access)
./run-app.sh                            # opens http://localhost:8501
```

Pick a bundled sample clip, upload a recording (wav, mp3, flac, ogg, m4a), or record one, then press **Analyze**. The app needs no
datasets and no GPU. See [app/README.md](app/README.md) for features and development notes.

## Setup

The project requires **Python 3.12** (pinned to 3.12.12): the pinned `torch==2.7.1` and
`torchvision==0.22.1` (`+cu128`) builds are validated on it (cu128 is needed for Blackwell/RTX 50-series GPUs and also runs on older ones), and an unpinned
or mismatched torchvision fails at import with `operator torchvision::nms does not exist`.

```bash
./setup.sh
conda activate data-science
```

`setup.sh` creates the conda environment defined in `environment.yml` (or updates it in
place if it already exists, so it is safe to re-run after `requirements.txt` changes) and
registers it as a Jupyter kernel named `data-science`, matching the notebooks' kernelspec,
so it appears in Jupyter and VS Code without manual `ipykernel` setup. If
you already have a `data-science` env on a different Python version, remove it first with
`conda env remove -n data-science`. Non-conda users can use `pyenv`/`uv`, which read
`.python-version`, then `pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt`.

## Running the pipeline from the command line

`run-pipeline.sh` is the entry point for the project's pipeline stages (data ingest, NER annotation, curation and training, ASR
training, and model download/upload). It walks through each stage interactively, asking whether to run it and what arguments to pass:

```bash
./run-pipeline.sh
```

Each stage's questions map directly to its underlying script's CLI flags.

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

## Reproducing the experiments

All experiments use the `data-science` conda environment unless noted. Seeds (42 by default) and the session-grouped split are fixed in
code, so reruns reproduce the splits; fine-tuned model scores vary by a few tenths of a point from GPU non-determinism (report, Section 5.4).

1. **Data.** Raw data is not in the repo. `python scripts/run_ingest.py` downloads both corpora from Hugging Face, resamples, derives
   features, runs the data-quality checks and writes `data/processed/utterances.parquet` (see [data/README.md](data/README.md) for
   sources and licensing).
2. **EDA.** Run `notebooks/comprehensive_eda.ipynb` (figures are saved to `res/figures/`).
3. **ASR.** `python scripts/run_asr_train.py --model openai/whisper-medium.en --lora --batch-size 2 --grad-accum 16 --tag lora` (about 4 h
   on an 8 GB GPU); other experiments and a `--dry-run` are described in the script header.
4. **NER labels and students.** In a separate vLLM environment: `scripts/run_ner_annotate.py` (about 12 h on the dev GPU); then
   `python scripts/run_ner_curate.py` and `python scripts/run_ner_train.py --model spacy --balance`.
5. **Benchmarks.** `python scripts/benchmark_inference.py` (see `res/benchmarks/EXPERIMENTS.md`).
6. **Report tables and figures.** `notebooks/machine_learning.ipynb` loads the saved results; to skip the multi-hour training runs,
   fetch them with `./download-models.sh` (the `asr-results` and `ner-data` groups).

`./run-pipeline.sh` walks through the ingest, NER and ASR stages interactively.

## Tests

```bash
python -m pytest tests app/tests        # pipelines, download and setup scripts, app UI (browser tests need Playwright; see app/README.md)
```

## Limitations

Two English corpora with no accent or speaker metadata, about 7.4 hours of real audio from limited airspace; NER evaluation sets of
95 to 100 utterances; models trained once; latency measured on one laptop CPU; and recognition errors flow straight into entity
extraction (the NER cannot detect a misheard digit). ClearanceIQ has not been tested with controllers or pilots. See the report,
Section 7.3.

## Getting and publishing the trained models

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

**No S3 access? Run the app from a Google Drive bundle.** `./download-models.sh` (default `--source auto`) falls back to a
single zip, `clearanceiq-models.zip` (~0.7 GB: the CTranslate2 ASR model and both spaCy NER models, everything the Streamlit
app loads), in a public view-only [Drive folder](https://drive.google.com/drive/folders/1F4tGwPjIgFPdo3ZClRX5-y3gFGThBWZ1)
whenever S3 can't be used (no AWS credentials, credentials for a different AWS account, an expired SSO login, no network or no `boto3`) or a download fails part-way, with no error shown. Force Drive with `--source gdrive`; `--source s3` never falls back, and an error appears only then or when every method fails. The zip is checked against `app-models.sha256`
(committed here) before extraction, and you can also download it by hand and run
`python scripts/app_bundle.py fetch --zip <file>`. The bundle does not include what the notebooks and training need
(full-precision model, results CSVs, NER data). Maintainers rebuild it with `./package-app-models.sh`,
upload `dist/clearanceiq-models.zip` to the Drive folder (replacing the old one) and commit the updated `app-models.sha256`.

The ASR model is published twice: the merged full-precision model (`asr/lora-whisper-medium-en`) and its
CTranslate2 int8 conversion (`asr/lora-whisper-medium-en-ct2-int8`), which is what a CPU device runs
(`asr.inference.best_transcriber()` picks the right one; see `res/benchmarks/EXPERIMENTS.md` for why).
Which local models count as "the best" is a short list in `scripts/lib/model_common.sh`; change a row
there when a better model replaces one. `python scripts/model_store.py --help` has the lower-level commands
(`list`, `promote`, per-run `upload`/`download`).

## Data and licensing

Raw data is acquired from Hugging Face by the ingest script and is never committed. Both corpora are research and coursework use only; see
[data/README.md](data/README.md) for the licensing notes and redistribution restrictions.
