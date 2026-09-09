# ATC Communications Intelligence: Speech Recognition and Information Extraction for Aviation Safety

|  |  |
| --- | --- |
| **Author** | Brett Allen |
| **GitHub Repository** | https://github.com/BrettFX/UMBC-DATA606-Capstone |
| **LinkedIn** | https://linkedin.com/in/brett-allen-586ba4121 |
| **PowerPoint Presentation** | *TBD* |
| **YouTube Channel** | https://www.youtube.com/@brettallen2199 |

---

# 2. Background

## 2.1 Topic Overview

Air traffic control (ATC) operations rely heavily on voice communication between air traffic controllers and pilots. These radio transmissions communicate operational information such as aircraft callsigns, altitude assignments, headings, speeds, runway instructions, and radio-frequency changes.

Unlike ordinary conversational speech, ATC communications use specialized aviation phrasing and frequently contain dense numeric and operational information. Radio communications may also be affected by background noise, interference, rapid speech, accents, varying audio quality, and overlapping or abbreviated phrasing.

This project investigates the application of modern Automatic Speech Recognition (ASR) and Natural Language Processing (NLP) techniques to ATC radio communications. The primary goal is to determine how effectively ATC audio can be converted into accurate text and then transformed into structured aviation information.

The project will follow a general communication-intelligence pipeline:

> **ATC Audio → Speech Recognition → Aviation Information Extraction → Communication Analysis**
> 

A prototype application (likely using `Streamlit`) is planned for development to be used to demonstrate the trained models and analytical results. The goal is to implement a real-world use case that Air Traffic Controllers and Pilots can use to enhance daily operations and promote aviation safety.

---

## 2.2 Why This Topic Matters

Clear and accurate communication is a fundamental component of aviation operations. ATC transmissions frequently contain operationally significant information including:

- Aircraft callsigns
- Altitude assignments
- Heading instructions
- Speed restrictions
- Runway information
- Radio frequencies
- Other controller instructions

Modern speech-recognition models can achieve strong performance on general speech, but ATC communications represent a specialized domain. For example, a model could produce a transcript that appears mostly correct but incorrectly recognizing an operationally important number, callsign, or command.

For instance, confusing an altitude of "five thousand" with "six thousand" may represent only a small number of incorrect words when measured using traditional transcription metrics, yet the error is considerably more meaningful in an aviation context.

This creates an opportunity to evaluate speech-recognition systems not only by overall transcription accuracy, but also by their ability to correctly recognize aviation-specific information.

NLP techniques may provide an additional layer of analysis by identifying structured operational entities from ATC transcripts via Named Entity Extraction (NER). Combining ASR and NLP could make it possible to transform unstructured aviation audio into structured data suitable for analysis and decision-support applications.

---

## 2.3 Research Questions

This project will investigate the following research questions.

### Research Question 1: Automatic Speech Recognition (ASR)

**How accurately can general-purpose and aviation-domain-adapted (pre-trained and/or fine-tuned) speech recognition models transcribe air traffic control communications?**

Potential evaluation measures include:

- Word Error Rate (WER)
- Character Error Rate
- Callsign recognition accuracy
- Numeric-value recognition accuracy
- Command recognition accuracy

---

### Research Question 2: Aviation Information Extraction

**Can NLP techniques reliably identify operational aviation information from ATC transcripts?**

> **NOTE:** The datasets explored so far do not provide token-level entity labels (e.g., callsign spans, command spans, etc.) inside a transcript. Therefore, training or evaluating extraction will require a rule-based/pattern approach (e.g., regex over standardized ATC phrasing) as well as an additional small (likely manually annotated) evaluation dataset, rather than full supervised training on existing labels. As an aside, there likely exists annotated datasets already but it may be possible to leverage the use of an LLM to help curate such a training dataset to then train a `spaCy` NER model to extract different information categories of interest.
> 

Potential information categories include:

- Callsigns
- Commands
- Altitudes
- Headings
- Speeds
- Runways
- Radio frequencies
- Other operational values

Potential evaluation measures include:

- Precision
- Recall
- F1 score
- Per-entity F1 score

---

### Research Question 3: Communication Characteristics and Model Performance

**What acoustic and linguistic characteristics of ATC transmissions are associated with speech-recognition performance?**

Not every transmission is equally difficult to transcribe, and this question is about *why*. *Acoustic characteristics** describe the audio signal itself, such as how long a transmission runs, how quickly someone speaks, and how noisy or clear the recording is. **Linguistic characteristics** describe the spoken content, such as how many words are packed into a transmission, and how much operationally dense information (e.g., callsigns, numbers, commands, etc.) it carries. If certain acoustic or linguistic profiles turn out to correlate with higher error rates, then it's something that can be addressed. It points to where a model is likely to fail, and suggests targeted responses (e.g., more training data for that condition, preprocessing to compensate for noise, flagging low-confidence transcripts for manual review, etc.).

**Core characteristics** (already computed in the preliminary EDA work and populated for every transmission):

- Transmission duration (`duration_sec`)
- Word count (`word_count`) and character count (`character_count`)
- Speech rate (`speech_rate_wpm`)
- Recording origin (`dataset_source`): whether the audio was collected from live operational communications or a controlled/simulated environment, since collection method often correlates with background noise and phrasing consistency.

**Planned characteristics** (requires additional work not yet built; pursued only if there's time after the core ASR work is complete): Signal-to-Noise Ratio, RMS energy, silence percentage, and the number of commands/numeric values/aviation entities per transmission, the last three depend on Research Question 2's extraction output (e.g., information categories extracted via NER).

Potential relationships to investigate include:

- Word Error Rate (WER) vs. transmission duration
- WER vs. speech rate
- WER vs. recording origin (real vs. simulated)
- (Planned) WER vs. Signal to Noise Ratio (SNR); aviation-entity error rate vs. audio quality

---

# 3. Data

Two aviation speech datasets were acquired, joined, and inspected for this research. The full
data-ingestion pipeline, data-quality checks, and exploratory analysis are documented in
[notebooks/allen_data_606_proposal_eda.ipynb](../notebooks/allen_data_606_proposal_eda.ipynb).

1. **ATCO2-ASR** (real ATC communications)
2. **ATCOSIM** (simulated ATC communications)

After combining the two datasets, they provide **10,118 utterances** (8,092 train / 2,026 validation) and
**~11.55 hours** of ATC audio. `DataIngestPipeline` (`src/data_ingest_pipeline.py`) joins them,
resamples all audio to a common 16kHz sample rate, and derives utterance-level metadata.

---

## 3.1 ATCO2-ASR Dataset

### Data Source

**ATCO2-ASR: Air Traffic Control Automatic Speech Recognition Dataset**

Source:

https://www.atco2.org/

Accessed via the Hugging Face mirror
[`jlvdoorn/atco2-asr`](https://huggingface.co/datasets/jlvdoorn/atco2-asr), a 559-utterance
subset of the larger ATCO2 corpus (the full corpus is gated behind an ELRA license). See
[data/README.md](../data/README.md) for the licensing/redistribution notes established during
data ingestion.

### Dataset Description

ATCO2-ASR is a real air-traffic-control speech corpus. The audio is recorded from live ATC
communications, paired with ground-truth transcripts and recording-level metadata (e.g., airport,
position, published waypoints, and observed callsigns/airlines).

Because it's real audio, ATCO2-ASR is expected to serve as the harder,
higher-value evaluation target for this project since it's closer to the noisy, accented, overlapping
speech the research questions care about, even though it's the smaller of the two datasets.

---

### Data Size and Shape

| Characteristic | Value |
| --- | --- |
| Size on disk | ~126 MB |
| Number of rows / utterances | 559 (446 train / 113 validation) |
| Number of columns | 3 raw on Hugging Face (`audio`, `text`, `info`); 12 total after `DataIngestPipeline` adds 9 derived columns |
| Number of audio files | 559 |
| Total audio duration | ~1.10 hours (mean 7.06s per utterance, std 3.93s, range 1.86-32.77s) |
| Audio format | WAV, mono |
| Sample rate | 16,000 Hz (native; matches this project's target rate) |

---

### Time Period

Although a time period is not explicitly provided as a field, audio filenames embed a recording date/time
(e.g., `LKPR_RUZYNE_Radar_120_520MHz_20201025_091112.wav` maps to 2020-10-25 09:11:12), which the
`DataIngestPipeline` parses into a `recorded_at` column. The inspected date range is as follows:

**2020-10-25 to 2021-05-05 (192 days)**. However, it's not one continuous window. Instead, it's broken into two distinct
recording waves about 6 months apart:

| Wave | Airports | Date range |
| --- | --- | --- |
| 1 | LKPR (Prague), LKTB (Brno) | 2020-10-25 to 2020-10-29 |
| 2 | LSGS (Sion), LSZB (Bern), LSZH (Zurich), LZIB (Bratislava), YSSY (Sydney) | 2021-04-12 to 2021-05-05 |

> **NOTE:** For modeling purposes, any date-correlated recording-condition differences (e.g., equipment, weather, traffic patterns, etc.) line up with *which airports* are in the data, not a spread-out timeline.
> 

---

### Unit of Observation

**One ATC utterance**: a 16kHz mono audio clip and its ground-truth transcript, plus an `info` metadata field which is only present in the ATCO2-ASR dataset and contains airport ICAO code and name, position (e.g., Radar/Tower), published waypoints, and the callsigns/airlines observed in that recording. 

Through preliminary EDA, the corpus spans 7 airports: LKPR (Prague), LKTB (Brno), LSGS (Sion), LSZB (Bern), LSZH (Zurich), LZIB (Bratislava), and YSSY (Sydney). Given all these airports are Non-United States (NONUS), it explains the (likely British) accents in all the audio recordings.

---

### Data Dictionary

> **NOTE:** Generated directly from `utterance_df` in
> [notebooks/allen_data_606_proposal_eda.ipynb](../notebooks/allen_data_606_proposal_eda.ipynb)
> and exported to [data_dictionary_atco2-asr.csv](data_dictionary_atco2-asr.csv). "Observed
> Value" is each column's first non-null example.
> 

| Column | Data Type | Definition | Observed Value |
| --- | --- | --- | --- |
| `audio` | Audio (array + sampling_rate + path) | Raw waveform, decoded on access | `path='LKPR_RUZYNE_Radar_120_520MHz_20201025_091112.wav', sampling_rate=16000, array=<~117,760 samples>` |
| `utterance_id` | String | Reproducible ID assigned by `DataIngestPipeline` | `atco2-asr-train-000000` |
| `original_split` | String | Split (train/validation) within the row's own source dataset, before joining | `train` |
| `audio_path` | String | Original audio filename | `LKPR_RUZYNE_Radar_120_520MHz_20201025_091112.wav` |
| `transcript` | String | Ground-truth transcript (renamed from the raw `text` column) | "Oscar Kilo Papa Mike Bravo descend flight level one hundred level one hundred Oscar Kilo Papa Mike …" |
| `duration_sec` | Float | Clip length in seconds | `7.36` |
| `sample_rate` | Int | Audio sampling frequency (Hz), after resampling to a common rate | `16000` |
| `word_count` | Int | Transcript word count | `18` |
| `character_count` | Int | Transcript character count | `105` |
| `speech_rate_wpm` | Float | `word_count / (duration_sec / 60)` | `146.74` |
| `info` | String | Recording-level metadata: airport ICAO/name, position, waypoints, callsigns/airlines (ATCO2-ASR only) | "LKPR \| Praha Ruzyne \| Radar \| AKEVA ARVEG BAGRU BAROX BAVIN BEKVI ELMEK ELPON ERASU EVEMI KENOK KUV…" |
| `recorded_at` | String | Recording date/time parsed from `audio_path` (ATCO2-ASR only) | `2020-10-25T09:11:12` |
| `dataset_split` | String | Split the row landed in after joining (identical to `original_split` today) | `train` |

> **NOTE:** `speaker_role`, per-token `callsign`/`command`/`value` annotations are not included. Also, neither
> the raw schema nor `info` provides token-level labels so aviation entity extraction from Research Question 2 has no ready-made ground truth to train or evaluate against.
> 

---

## 3.2 ATCOSIM Dataset

### Data Source

**ATCOSIM: Air Traffic Control Simulation Speech Corpus**

Source:

https://huggingface.co/datasets/Jzuluaga/atcosim_corpus

### Dataset Description

ATCOSIM is an aviation speech corpus containing simulated ATC communications produced by professional air traffic controllers.

The corpus contains speech recordings and corresponding transcripts and provides a much larger
source of aviation-domain audio which is useful for:

- ASR model training or fine-tuning
- ASR model evaluation on clean, uniformly-phrased speech
- Cross-dataset model comparison against real ATCO2-ASR audio

Accessed via the Hugging Face mirror
[`jlvdoorn/atcosim`](https://huggingface.co/datasets/jlvdoorn/atcosim). ATCOSIM serves as the
primary training-volume dataset for this project (94.5% of combined rows, 90.5% of combined
audio hours). However, the real-audio ATCO2-ASR data remains the evaluation target.

---

### Data Size and Shape

| Characteristic | Value |
| --- | --- |
| Size on disk | ~2.4 GB |
| Number of rows / utterances | 9,559 (7,646 train / 1,913 validation) |
| Number of columns | 2 raw on Hugging Face (`audio`, `text`); 12 total after `DataIngestPipeline` backfills a null `info` column (for schema parity with ATCO2-ASR) and adds 9 derived columns |
| Number of audio files | 9,559 |
| Total audio duration | ~10.46 hours (mean 3.94s per utterance, std 1.51s, range 0.14-38.88s) |
| Audio format | WAV, mono |
| Number of speakers | Not provided; `jlvdoorn/atcosim` has no speaker-identifier column |

---

### Time Period

**Not applicable.** No recording dates are provided, and filenames (e.g., `gf1_01_001.wav`) use
a generic recording/session naming convention with no embedded date, unlike ATCO2-ASR's.

---

### Unit of Observation

**One simulated ATC utterance**: a 32kHz mono audio clip (resampled to 16kHz by
the `DataIngestPipeline`) and its ground-truth transcript. No recording-level metadata is provided beyond the audio and transcript.

---

### Data Dictionary

> **NOTE:** Generated directly from `utterance_df` in
> [notebooks/allen_data_606_proposal_eda.ipynb](../notebooks/allen_data_606_proposal_eda.ipynb)
> and exported to [data_dictionary_atcosim.csv](data_dictionary_atcosim.csv). "Observed Value" is
> each column's first non-null example.
> 

| Column | Data Type | Definition | Observed Value |
| --- | --- | --- | --- |
| `audio` | Audio (array + sampling_rate + path) | Raw waveform, decoded on access | `path='gf1_01_001.wav', sampling_rate=16000, array=<~46,978 samples>` |
| `utterance_id` | String | Reproducible ID assigned by `DataIngestPipeline` | `atcosim-train-000000` |
| `original_split` | String | Split (train/validation) within the row's own source dataset, before joining | `train` |
| `audio_path` | String | Original audio filename | `gf1_01_001.wav` |
| `transcript` | String | Ground-truth transcript (renamed from the raw `text` column) | " contact geneva one two eight decimal one five good bye " |
| `duration_sec` | Float | Clip length in seconds | `2.94` |
| `sample_rate` | Int | Audio sampling frequency (Hz), after resampling to a common rate | `16000` |
| `word_count` | Int | Transcript word count | `10` |
| `character_count` | Int | Transcript character count | `56` |
| `speech_rate_wpm` | Float | `word_count / (duration_sec / 60)` | `204.35` |
| `info` | String | Recording-level metadata: airport ICAO/name, position, waypoints, callsigns/airlines (ATCO2-ASR only) | `N/A` |
| `recorded_at` | String | Recording date/time parsed from `audio_path` (ATCO2-ASR only) | `N/A` |
| `dataset_split` | String | Split the row landed in after joining (identical to `original_split` today) | `train` |

> **NOTE:** `info` and `recorded_at` are always null in the ATCOSIM dataset. Both show as `N/A` in the table since the
> `DataIngestPipeline` backfills them for schema parity with the ATCO2-ASR dataset. The Hugging Face
> mirror used provides only audio + transcript natively.
>

---

## 3.3 Target Variables / Labels

Because the project contains more than one machine-learning task, there may not be a single target variable for the entire project.

### Automatic Speech Recognition

For the ASR task, the primary target will be:

**`transcript`**

The model will receive an audio signal as input and attempt to predict the corresponding transcript.

---

### Aviation Information Extraction

As stated in previous sections, neither dataset provides ready-made entity annotations. Thus, the target labels will be derived rather than taken directly from the source data. For instance, an initial rule-based/pattern pass over transcripts, refined against a small manually annotated evaluation subset (e.g., via LLM or an additional data source found elsewhere).

Potential target labels include:

- Callsign
- Command
- Altitude
- Heading
- Speed
- Runway
- Frequency
- Other operational values

---

## 3.4 Candidate Features / Predictors

Different model tasks will require different predictors.

### ASR Model Inputs

The primary predictor for speech recognition will be the:

- Raw audio waveform

Potential derived audio representations or features may include:

- Mel-frequency spectrogram
- Mel-Frequency Cepstral Coefficients (MFCCs)
- Signal-to-Noise Ratio
- Root Mean Square (RMS) energy
- Spectral characteristics
- Silence percentage

> **NOTE:** For pre-trained transformer-based speech models, the raw or appropriately pre-processed waveform may be used directly rather than manually engineered features.
> 

---

### Information-Extraction Model Inputs

Potential predictors include:

- Transcript tokens
- Word sequence
- Context surrounding each token
- Aviation phrasing patterns
- Numeric expressions

Transformer-based NLP models or `spaCy` models may generate learned contextual representations directly from transcript text.

---

### Model-Performance Analysis Features

Matches Research Question 3's core/planned split above:

- **Core (implemented):** transmission duration, word count, character count, speech rate,
  dataset source
- **Planned (not yet implemented):** Signal-to-Noise Ratio, RMS energy, silence percentage,
  number of commands/numeric values/aviation entities, speaker role

These variables can be compared with model performance measurements such as WER to investigate which communication characteristics are associated with higher or lower transcription accuracy.

---

## 3.5 Dataset Integration

The two datasets were preserved separately through ingestion (`_load`, `_standardize_schema`) so
each source's original structure and metadata stayed intact, then joined
(`concatenate_datasets`) into one combined `DatasetDict` once their schemas were aligned.

The `DataIngestPipeline` produces a standardized utterance-level analytical table
(`utterance_df`, 10,118 rows) with the following structure:

| Column | Description |
| --- | --- |
| `utterance_id` | Unique observation |
| `dataset_source` | ATCO2 or ATCOSIM |
| `audio_path` | Audio source |
| `transcript` | Ground-truth text |
| `duration_sec` | Transmission duration |
| `speaker_role` | Speaker category when available |
| `sample_rate` | Audio sampling frequency |
| `word_count` | Transcript word count |
| `speech_rate_wpm` | Estimated speech rate |
| `callsign_count` | Number of callsigns |
| `command_count` | Number of commands |
| `numeric_count` | Number of numeric values |
| `entity_count` | Number of aviation entities |

> **NOTE:** The `DataIngestPipeline` will likely evolve through iterations of this project and additional pipeline scripts will likely be created to streamline the end-to-end flow from loading the data, pre-processing it, training and evaluating models, and inferencing the models.
> 
