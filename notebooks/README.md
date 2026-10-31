# Notebooks

Run them with the `data-science` Jupyter kernel that `../setup.sh` registers (the notebooks' kernelspec display name).

| Notebook | Purpose | Needs |
|---|---|---|
| [proposal_eda.ipynb](proposal_eda.ipynb) | Proposal-stage exploration of the first real-ATC source (ATCO2-ASR, since replaced; see `docs/report.md` Section 3) and ATCOSIM | The Hugging Face datasets (downloaded on first run) |
| [comprehensive_eda.ipynb](comprehensive_eda.ipynb) | Report Section 4: data cleansing, acoustic and transcript features, outliers, class balance, split composition | The Hugging Face datasets, or the cached `data/processed/utterances.parquet` (also produced by `python scripts/run_ingest.py`) |
| [machine_learning.ipynb](machine_learning.ipynb) | Report Section 5: ASR prototype (Sections A to E, runs on a GPU), then the production ASR, NER and benchmark results (Sections F to H, loaded from saved outputs) | `./download-models.sh --only asr-results` and `--only ner-data` for the saved results; a GPU only for the prototype sections |

Figures are saved to `../res/figures/`, where the report and presentation use them. The long training runs live in
`../scripts/`, not in the notebooks (see the root README, "Reproducing the experiments").
