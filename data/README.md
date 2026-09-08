# Data

This project uses two air traffic control (ATC) speech corpora, both accessed via their
Hugging Face mirrors. Raw downloads and any derived/processed artifacts are **not** committed
to this repo (see [.gitignore](../.gitignore)). Only this documentation and the folder
placeholders are tracked. Anyone cloning the repo needs to re-download the data locally using
the steps below.

## Directory layout

```
data/
├── raw/            # Untouched downloads, as pulled from Hugging Face (git-ignored)
│   ├── atco2-asr/  # cache_dir target for jlvdoorn/atco2-asr
│   └── atcosim/    # cache_dir target for jlvdoorn/atcosim
├── processed/       # Cleaned / merged / feature-engineered outputs (git-ignored)
└── samples/         # Small, hand-picked example files safe to keep for demos (git-ignored)
```

Each subfolder keeps a `README.md` placeholder so the empty directory structure survives
`git clone`; everything else under `data/` is ignored by git.

## Data sources

| Dataset | HF repo | Upstream / original source | Rows | Size |
|---|---|---|---|---|
| ATCO2-ASR | [jlvdoorn/atco2-asr](https://huggingface.co/datasets/jlvdoorn/atco2-asr) | [ATCO2 project](https://www.atco2.org/data) ([GitHub](https://github.com/idiap/atco2-corpus)), featuring real, recorded ATC communications | 559 (446 train / 113 validation) | ~126 MB |
| ATCOSIM | [jlvdoorn/atcosim](https://huggingface.co/datasets/jlvdoorn/atcosim) | [ATCOSIM project](https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html), TU Graz SPSC lab, featuring simulated ATC communications | 9,559 (7,646 train / 1,913 validation) | ~2.4 GB |

Both HF datasets ship as audio (`audio`) + transcription (`text`) pairs in Parquet format,
English only. ATCO2-ASR additionally includes an `info` field with metadata about active
aircraft/navigation waypoints.

## Download steps

Datasets are pulled with Hugging Face `datasets.load_dataset`, using `cache_dir` (not `data_dir`!) to
point the download at the expected local path. See
[notebooks/allen_data_606_proposal_eda.ipynb](../notebooks/allen_data_606_proposal_eda.ipynb) for
the working example:

```python
from datasets import load_dataset
import os

data_dir = "../data/"
atco2_asr_path = os.path.join(data_dir, "raw/atco2-asr")
atcosim_path = os.path.join(data_dir, "raw/atcosim")

atco2_asr_dataset = load_dataset("jlvdoorn/atco2-asr", cache_dir=atco2_asr_path)
atcosim_dataset = load_dataset("jlvdoorn/atcosim", cache_dir=atcosim_path)
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
data/raw/atco2-asr/jlvdoorn___atco2-asr/...
data/raw/atcosim/jlvdoorn___atcosim/...
```

Downstream notebooks/scripts should read from these paths (or from `data/processed/` for
cleaned/merged output) rather than hitting the Hub again.

## Licensing & redistribution restrictions

Neither Hugging Face dataset card (`jlvdoorn/atco2-asr`, `jlvdoorn/atcosim`) states an
explicit license, so licensing terms fall back to the **upstream/original corpora**. Treat both
datasets as **research/coursework use only**. Do not redistribute the raw audio or
transcriptions outside this project, and do not use them for any commercial purpose:

- **ATCO2**: The full ATCO2 corpus is tiered. The full training set and official 4h test set
  are distributed commercially/non-commercially through
  [ELRA](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0484/), while the smaller
  1-hour test subset (which the `jlvdoorn/atco2-asr` mirror's size with 559 rows / 126 MB
  matches) is released by the project for **research purposes only**. Cite the ATCO2 paper
  ([arXiv:2211.04054](https://arxiv.org/abs/2211.04054)) if used in written work.
- **ATCOSIM**: Distributed by TU Graz's SPSC lab "free of charge" for research/educational use,
  but with no formal open-source license attached. Consult the
  [full manual](http://www2.spsc.tugraz.at/databases/ATCOSIM/DOC/atcosim_report.pdf) before any
  use beyond coursework, and attribute TU Graz SPSC when citing.

**Practical implications for this repo:**
- Raw/processed audio and transcripts stay out of git (enforced via `.gitignore`).
- Don't upload raw samples to third-party services (model hosts, public demo apps, etc.)
  without re-checking the current upstream terms.
- If in doubt about a specific use (e.g., publishing a fine-tuned model trained on this data),
  check the upstream ATCO2/ATCOSIM terms first. The Hugging Face mirrors' silence on licensing is not
  the same as public-domain/unrestricted use.
