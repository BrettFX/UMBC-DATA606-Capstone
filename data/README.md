# Data

This project uses two air traffic control (ATC) speech corpora, both accessed via their
Hugging Face mirrors. Raw downloads and any derived/processed artifacts are **not** committed
to this repo (see [.gitignore](../.gitignore)). Only this documentation and the folder
placeholders are tracked. Anyone cloning the repo needs to re-download the data locally using
the steps below.

## Directory layout

```
data/
├── raw/                 # Untouched downloads, as pulled from Hugging Face (git-ignored)
│   ├── atcosim/         # cache_dir target for jlvdoorn/atcosim
│   └── atc-asr-dataset/ # cache_dir target for jacktol/ATC-ASR-Dataset
├── processed/            # Cleaned / merged / feature-engineered outputs (git-ignored)
└── samples/              # Small, hand-picked example files safe to keep for demos (git-ignored)
```

Each subfolder keeps a `README.md` placeholder so the empty directory structure survives
`git clone`; everything else under `data/` is ignored by git.

> **Note:** an earlier phase of this project (see `docs/report.md` §3's "A note on dataset
> selection") used `jlvdoorn/atco2-asr` as the real-ATC-audio source instead of
> `jacktol/ATC-ASR-Dataset`. It was replaced, not supplemented, once it became clear
> `ATC-ASR-Dataset` is itself built in part from the same public "ATCO2 1-Hour Test Subset"
> `atco2-asr` wraps — running both risked duplicate/overlapping recordings and train/test
> leakage. `data/raw/atco2-asr/` is no longer part of the ingestion pipeline's `sources`.

## Data sources

| Dataset | HF repo | Upstream / original source | Rows | Size |
|---|---|---|---|---|
| ATCOSIM | [jlvdoorn/atcosim](https://huggingface.co/datasets/jlvdoorn/atcosim) | [ATCOSIM project](https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html), TU Graz SPSC lab, featuring simulated ATC communications | 9,559 (7,646 train / 1,913 validation, as published; re-split by this project into 7,459/1,004/1,096 train/validation/test) | ~2.4 GB |
| ATC-ASR-Dataset | [jacktol/ATC-ASR-Dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset) | Built from the [UWB ATC Corpus](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0) and the [ATCO2 project](https://www.atco2.org/data)'s 1-Hour Test Subset, featuring real, recorded ATC communications | 8,122 (6,497 train / 812 validation / 813 test) | ~848 MB |

Both HF datasets ship as audio (`audio`) + transcription (`text`) pairs, English only.
`ATC-ASR-Dataset` additionally includes an `id` field (an opaque per-row hash; kept as
`source_id` after ingestion). Neither ships recording-level metadata (airport, position,
waypoints) — the retired `atco2-asr`'s `info` field was the only source that did.

## Download steps

Datasets are pulled with Hugging Face `datasets.load_dataset`, using `cache_dir` (not `data_dir`!) to
point the download at the expected local path. See
[src/preprocessing/ingest.py](../src/preprocessing/ingest.py)'s `_load` method (or
[notebooks/comprehensive_eda.ipynb](../notebooks/comprehensive_eda.ipynb) for the same thing run
interactively) for the working example:

```python
from datasets import load_dataset
import os

data_dir = "../data/"
atcosim_path = os.path.join(data_dir, "raw/atcosim")
atc_asr_dataset_path = os.path.join(data_dir, "raw/atc-asr-dataset")

atcosim_dataset = load_dataset("jlvdoorn/atcosim", cache_dir=atcosim_path)
atc_asr_dataset = load_dataset("jacktol/ATC-ASR-Dataset", cache_dir=atc_asr_dataset_path)
```

Notes:
- `cache_dir` controls where the data lands locally; `data_dir` means something different (a
  subfolder *inside* the HF repo to filter on) and will 404 if misused. Don't swap them.
- `datasets` nests its own cache structure (a `<org>___<name>/` folder, arrow files, `.lock`
  files, etc.) under whatever `cache_dir` you give it. Expect that structure, not flat audio
  files.
- No Hugging Face auth token is required; both repos are public.
- Re-running the cells is safe/idempotent. `datasets` will reuse the local cache instead of
  re-downloading once the files exist.

## Expected paths

After downloading, you should have:

```
data/raw/atcosim/jlvdoorn___atcosim/...
data/raw/atc-asr-dataset/jacktol___atc-asr-dataset/...
```

Downstream notebooks/scripts should read from these paths (or from `data/processed/` for
cleaned/merged output) rather than hitting the Hub again.

## Licensing & redistribution restrictions

None of the Hugging Face dataset cards involved (`jlvdoorn/atcosim`, `jacktol/ATC-ASR-Dataset`)
state an explicit license, so licensing terms fall back to the **upstream/original corpora**.
Treat all of them as **research/coursework use only**. Do not redistribute the raw audio or
transcriptions outside this project, and do not use them for any commercial purpose:

- **ATCOSIM**: Distributed by TU Graz's SPSC lab "free of charge" for research/educational use,
  but with no formal open-source license attached. Consult the
  [full manual](http://www2.spsc.tugraz.at/databases/ATCOSIM/DOC/atcosim_report.pdf) before any
  use beyond coursework, and attribute TU Graz SPSC when citing.
- **ATC-ASR-Dataset**: built in part from the **ATCO2 1-Hour Test Subset** — the same tier of
  the ATCO2 corpus released by the project for **research purposes only** (the full training
  set and official 4h test set are separately licensed, commercially/non-commercially, through
  [ELRA](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0484/)). Cite the ATCO2
  project (https://www.atco2.org/) if used in written work. It's also built in part from the
  **UWB ATC Corpus**, hosted on the LINDAT/CLARIAH-CZ repository; consult that repository's own
  terms before any use beyond coursework.

**Practical implications for this repo:**
- Raw/processed audio and transcripts stay out of git (enforced via `.gitignore`).
- Don't upload raw samples to third-party services (model hosts, public demo apps, etc.)
  without re-checking the current upstream terms.
- If in doubt about a specific use (e.g., publishing a fine-tuned model trained on this data),
  check the upstream ATCOSIM/ATCO2/UWB terms first. The Hugging Face mirrors' silence on
  licensing is not the same as public-domain/unrestricted use.
