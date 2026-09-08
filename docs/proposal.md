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

Unlike ordinary conversational speech, ATC communications use specialized aviation phraseology and frequently contain dense numeric and operational information. Radio communications may also be affected by background noise, interference, rapid speech, accents, varying audio quality, and overlapping or abbreviated phraseology.

This project investigates the application of modern Automatic Speech Recognition (ASR) and Natural Language Processing (NLP) techniques to ATC radio communications. The primary goal is to determine how effectively ATC audio can be converted into accurate text and then transformed into structured aviation information.

The project will follow a general communication-intelligence pipeline:

> **ATC Audio → Speech Recognition → Aviation Information Extraction → Communication Analysis**
> 

A prototype application, tentatively named **ClearanceIQ**, may later be used to demonstrate the trained models and analytical results.

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

Modern speech-recognition models can achieve strong performance on general speech, but ATC communications represent a specialized domain. A model could produce a transcript that appears mostly correct while still incorrectly recognizing an operationally important number, callsign, or command.

For example, confusing an altitude of "five thousand" with "six thousand" may represent only a small number of incorrect words when measured using traditional transcription metrics, yet the error is considerably more meaningful in an aviation context.

This creates an opportunity to evaluate speech-recognition systems not only by overall transcription accuracy, but also by their ability to correctly recognize aviation-specific information.

Natural language processing techniques may provide an additional layer of analysis by identifying structured operational entities from ATC transcripts. Combining ASR and NLP could make it possible to transform unstructured aviation audio into structured data suitable for analysis and, eventually, decision-support applications.

---

## 2.3 Research Questions

This project will investigate the following research questions.

**Scope note (post-EDA):** The original proposal sketch included a fourth research question on
readback-consistency analysis (matching pilot readbacks against controller instructions). It has
been dropped rather than kept as a fallback: it depended on associating controller/pilot turns
within a conversation, and neither ATCO2-ASR nor ATCOSIM's utterance-level schema provides that
turn-level structure, so it would have required scope well beyond a 15-week course. Research
Question 2 has also been narrowed below, since the EDA confirmed neither dataset provides
token-level entity annotations to train or evaluate against.

### Research Question 1: Automatic Speech Recognition (ASR)

**How accurately can general-purpose and aviation-domain-adapted speech recognition models transcribe air traffic control communications?**

Potential evaluation measures include:

- Word Error Rate (WER)
- Character Error Rate
- Callsign recognition accuracy
- Numeric-value recognition accuracy
- Command recognition accuracy

---

### Research Question 2: Aviation Information Extraction

**Can natural language processing techniques reliably identify operational aviation information from ATC transcripts?**

**Scope:** treated as a secondary objective. Neither ATCO2-ASR's `info` field nor ATCOSIM's
schema provides token-level entity labels (callsign spans, command spans, etc.) inside a
transcript, only recording-level context for ATCO2-ASR. Training or evaluating extraction
therefore starts from a rule-based/pattern baseline (e.g., regex over standardized ATC
phraseology) plus a small, manually annotated evaluation subset, rather than full supervised
training on existing labels.

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

**Core characteristics** (already implemented by `DataIngestPipeline`, confirmed populated for
every row in the EDA):

- Transmission duration (`duration_sec`)
- Word count (`word_count`) and character count (`character_count`)
- Speech rate (`speech_rate_wpm`)
- Dataset source (`dataset_source`): real (ATCO2-ASR) vs. simulated (ATCOSIM)

**Stretch characteristics** (require work not yet built; pursued only if time allows after the
core ASR work): Signal-to-Noise Ratio, RMS energy, silence percentage, and number of
commands/numeric values/aviation entities per utterance, the last three depend on Research
Question 2's extraction output. `speaker_role` is not pursued at all: it's absent from both
source schemas (see Section 3, ATCO2-ASR Data Dictionary).

Potential relationships to investigate include:

- WER vs. transmission duration
- WER vs. speech rate
- WER vs. dataset source (confirming the per-source evaluation split called for in the EDA's
  Implications for the Modeling Plan)
- (Stretch) WER vs. SNR; aviation-entity error rate vs. audio quality

---

# 3. Data

Two aviation speech datasets were acquired, joined, and inspected for this research. The full
data-ingestion pipeline, data-quality checks, and exploratory analysis are documented in
[notebooks/allen_data_606_proposal_eda.ipynb](../notebooks/allen_data_606_proposal_eda.ipynb).

1. **ATCO2-ASR** (real ATC communications)
2. **ATCOSIM** (simulated ATC communications)

Combined, the two datasets provide **10,118 utterances** (8,092 train / 2,026 validation) and
**~11.55 hours** of ATC audio. `DataIngestPipeline` (`src/data_ingest_pipeline.py`) joins them,
resamples all audio to a common 16kHz sample rate, and derives the utterance-level metadata used
throughout this section.

---

## 3.1 ATCO2 Dataset

### Data Source

**ATCO2: Air Traffic Control Automatic Speech Recognition Dataset**

Source:

https://www.atco2.org/

Accessed via the Hugging Face mirror
[`jlvdoorn/atco2-asr`](https://huggingface.co/datasets/jlvdoorn/atco2-asr), a 559-utterance
subset of the larger ATCO2 corpus (the full corpus is gated behind an ELRA license). See
[data/README.md](../data/README.md) for the licensing/redistribution notes established during
data ingestion.

### Dataset Description

ATCO2-ASR is a real air-traffic-control speech corpus: audio recorded from live ATC
communications, paired with ground-truth transcripts and recording-level metadata (airport,
position, published waypoints, and observed callsigns/airlines).

Because it's real (not simulated) audio, ATCO2-ASR is expected to serve as the harder,
higher-value evaluation target for this project, closer to the noisy, accented, overlapping
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

Not provided as an explicit field. ATCO2-ASR audio filenames do embed a recording date/time
(e.g., `LKPR_RUZYNE_Radar_120_520MHz_20201025_091112.wav` → 2020-10-25), so an approximate range
could be derived from `audio_path` in future work; this wasn't computed as part of the current
EDA pass, so it's documented as **not applicable / not provided** rather than estimated.

---

### Unit of Observation

**One ATC utterance**: a 16kHz mono audio clip and its ground-truth transcript, plus (ATCO2-ASR
only) recording-level `info` metadata: airport ICAO code and name, position (e.g., Radar/Tower),
published waypoints, and the callsigns/airlines observed in that recording. The corpus spans 7
airports (including Prague/LKPR and Sydney/YSSY, confirmed by inspection; the remaining 5 are
identified only by ICAO code: LSGS, LSZB, LSZH, LZIB, LKTB).

---

### Data Dictionary

Confirmed from the actual `jlvdoorn/atco2-asr` schema and `DataIngestPipeline`'s derived columns
(see [notebooks/allen_data_606_proposal_eda.ipynb](../notebooks/allen_data_606_proposal_eda.ipynb)):

| Column | Data Type | Definition | Observed Values |
| --- | --- | --- | --- |
| `audio` | Audio, 16kHz mono | Raw waveform, decoded on access | duration 1.86-32.77s |
| `text` | String | Ground-truth transcript | e.g. "Ryanair Seven Three Alpha Hotel turn left heading three six zero" |
| `info` | String | Recording-level metadata (airport, position, waypoints, callsigns, airlines) | populated for all 559 rows |
| `utterance_id` | String | Reproducible ID assigned by `DataIngestPipeline` | e.g. `atco2-asr-train-000000` |
| `duration_sec` | Float | Clip length in seconds | mean 7.06, std 3.93 |
| `word_count` | Int | Transcript word count | mean 19.1, std 10.3, max 88 |
| `character_count` | Int | Transcript character count | mean 110.7, std 61.1 |
| `speech_rate_wpm` | Float | `word_count / (duration_sec / 60)` | mean 168.4 wpm, std 36.0 |

**Not available:** `speaker_role`, per-token `callsign`/`command`/`value` annotations. Neither
the raw schema nor `info` provides token-level labels, only recording-level context, so aviation
entity extraction (Research Question 2) has no ready-made ground truth to train or evaluate
against; see the scope note under Research Question 2.

---

## 3.2 ATCOSIM Dataset

### Data Source

**ATCOSIM: Air Traffic Control Simulation Speech Corpus**

Source:

https://huggingface.co/datasets/Jzuluaga/atcosim_corpus

### Dataset Description

ATCOSIM is an aviation speech corpus containing simulated ATC communications produced by professional air traffic controllers.

The corpus contains speech recordings and corresponding transcripts and provides a much larger
source of aviation-domain audio, useful for:

- ASR model training or fine-tuning
- ASR model evaluation on clean, uniformly-phrased speech
- Cross-dataset model comparison against real ATCO2-ASR audio

Accessed via the Hugging Face mirror
[`jlvdoorn/atcosim`](https://huggingface.co/datasets/jlvdoorn/atcosim). ATCOSIM serves as the
primary training-volume dataset for this project (94.5% of combined rows, 90.5% of combined
audio hours); ATCO2-ASR remains the harder, real-audio evaluation target.

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
`DataIngestPipeline`) and its ground-truth transcript. No recording-level metadata (no airport,
speaker, or session identifier) is provided beyond audio and transcript.

---

### Data Dictionary

Confirmed from the actual `jlvdoorn/atcosim` schema and `DataIngestPipeline`'s derived columns:

| Column | Data Type | Definition | Observed Values |
| --- | --- | --- | --- |
| `audio` | Audio, 32kHz mono (native) | Raw waveform; resampled to 16kHz during ingestion | duration 0.14-38.88s |
| `text` | String | Ground-truth transcript | e.g. "contact geneva one two eight decimal one five good bye" |
| `utterance_id` | String | Reproducible ID assigned by `DataIngestPipeline` | e.g. `atcosim-train-000000` |
| `duration_sec` | Float | Clip length in seconds, post-resample | mean 3.94, std 1.51 |
| `word_count` | Int | Transcript word count | mean 11.3, std 4.2, max 71 |
| `character_count` | Int | Transcript character count | mean 64.7, std 23.9 |
| `speech_rate_wpm` | Float | `word_count / (duration_sec / 60)` | mean 174.4 wpm, std 37.9 |

**Not available:** `speaker_id`, `info`/session metadata originally anticipated in the earlier
draft of this dictionary. The Hugging Face mirror used provides only audio + transcript.

---

## 3.3 Target Variables / Labels

Because the project contains more than one machine-learning task, there may not be a single target variable for the entire project.

### Automatic Speech Recognition

For the ASR task, the primary target will be:

**`transcript`**

The model will receive an audio signal as input and attempt to predict the corresponding transcript.

Conceptually:

> **Audio → Transcript**
> 

---

### Aviation Information Extraction

Neither dataset provides ready-made entity annotations (see Research Question 2's scope note),
so target labels will be derived rather than taken directly from the source data: an initial
rule-based/pattern pass over transcripts, refined against a small manually annotated evaluation
subset.

Potential target labels include:

- Callsign
- Command
- Altitude
- Heading
- Speed
- Runway
- Frequency
- Other operational values

Conceptually:

> **Transcript → Aviation Entity Labels**
> 

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
- RMS energy
- Spectral characteristics
- Silence percentage

For pretrained transformer-based speech models, the raw or appropriately preprocessed waveform may be used directly rather than manually engineered features.

---

### Information-Extraction Model Inputs

Potential predictors include:

- Transcript tokens
- Word sequence
- Context surrounding each token
- Aviation phraseology patterns
- Numeric expressions

Transformer-based NLP models may generate learned contextual representations directly from transcript text.

---

### Model-Performance Analysis Features

Matches Research Question 3's core/stretch split above:

- **Core (implemented):** transmission duration, word count, character count, speech rate,
  dataset source
- **Stretch (not yet implemented):** Signal-to-Noise Ratio, RMS energy, silence percentage,
  number of commands/numeric values/aviation entities, speaker role

These variables can be compared with model performance measurements such as Word Error Rate to investigate which communication characteristics are associated with higher or lower transcription accuracy.

---

## 3.5 Dataset Integration

The two datasets were preserved separately through ingestion (`_load`, `_standardize_schema`) so
each source's original structure and metadata stayed intact, then joined
(`concatenate_datasets`) into one combined `DatasetDict` once their schemas were aligned.

`DataIngestPipeline` produces a standardized utterance-level analytical table
(`utterance_df`, 10,118 rows) with this structure:

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

**Implementation status (Proposal EDA notebook):** `DataIngestPipeline`
(`src/data_ingest_pipeline.py`) already implements `utterance_id`, `dataset_source`,
`original_split`, `audio_path`, `transcript`, `duration_sec`, `sample_rate`, `word_count`,
`character_count`, `speech_rate_wpm`, and `info` (ATCO2-ASR only). `speaker_role`,
`callsign_count`, `command_count`, `numeric_count`, and `entity_count` remain stretch goals,
pending the rule-based/weakly-supervised entity-extraction work described under Research
Question 2, and are not part of the core deliverable for this course.
