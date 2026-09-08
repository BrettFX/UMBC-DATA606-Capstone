# ATC Communications Intelligence: Speech Recognition and Information Extraction for Aviation Safety

|  |  |
| --- | --- |
| **Author** | Brett Allen |
| **GitHub Repository** | https://github.com/BrettFX/UMBC-DATA606-Capstone |
| **LinkedIn** | https://linkedin.com/in/brett-allen-586ba4121 |
| **PowerPoint Presentation** | *To be added* |
| **YouTube Presentation** | https://www.youtube.com/@brettallen2199 |

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

### Research Question 1: Automatic Speech Recognition

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

Potential characteristics include:

- Transmission duration
- Signal-to-Noise Ratio (SNR)
- Speech rate
- Word count
- Number of commands
- Number of numeric values
- Number of aviation entities
- Speaker role
- Audio energy
- Silence percentage

Potential relationships to investigate include:

- WER vs. SNR
- WER vs. transmission duration
- WER vs. speech rate
- WER vs. number of commands
- Aviation-entity error rate vs. audio quality

---

### Research Question 4: Readback Analysis

**Can structured representations of controller instructions and pilot readbacks be used to identify potentially inconsistent readbacks?**

Potential discrepancies may include:

- Altitude mismatches
- Heading mismatches
- Frequency mismatches
- Speed mismatches
- Runway mismatches

This research question will be treated as a secondary objective and will depend on whether the available datasets contain sufficient information to reliably associate controller instructions with pilot responses.

---

# 3. Data

Two aviation speech datasets are currently planned for this research:

1. **ATCO2**
2. **ATCOSIM**

The datasets will be inspected and analyzed in a Jupyter Notebook before the final values in this section are completed.

---

## 3.1 ATCO2 Dataset

### Data Source

**ATCO2: Air Traffic Control Automatic Speech Recognition Dataset**

Source:

https://www.atco2.org/

### Dataset Description

ATCO2 is an air traffic control speech corpus developed to support machine learning research involving ATC radio communications.

The dataset contains ATC audio and associated annotations that may support tasks such as:

- Automatic speech recognition
- Callsign recognition
- Command recognition
- Aviation entity extraction
- Speaker-role analysis

ATCO2 is expected to serve as the primary dataset for this project because its aviation-specific audio and annotations closely align with the project's research questions.

---

### Data Size and Shape

The exact dataset characteristics will be calculated after acquisition and inspection.

| Characteristic | Value |
| --- | --- |
| Size on disk | TBD |
| Number of rows / utterances | TBD |
| Number of columns | TBD |
| Number of audio files | TBD |
| Total audio duration | TBD |
| Audio format | TBD |
| Sample rate | TBD |

---

### Time Period

**TBD**

If the dataset documentation does not provide a meaningful time range for the recordings, this will be documented as **not applicable / not provided** rather than estimated.

---

### Unit of Observation

The expected unit of observation is:

> **One ATC transmission or utterance and its associated transcript, audio, and metadata.**
> 

The exact unit will be confirmed after the dataset schema is inspected.

---

### Data Dictionary

The final data dictionary will be generated after loading the ATCO2 data.

Expected fields may include the following:

| Column | Data Type | Definition | Potential Values |
| --- | --- | --- | --- |
| `utterance_id` | String / Categorical | Unique identifier for an ATC transmission | Unique ID |
| `audio` / `audio_path` | Audio / String | Audio recording or path to recording | Audio file |
| `transcript` | String | Ground-truth transcription of the transmission | Free text |
| `speaker_role` | Categorical | Role of the speaker if available | Controller, pilot, other |
| `callsign` | String / Categorical | Aircraft callsign if annotated | Aircraft identifier |
| `command` | String / Categorical | ATC command if annotated | Climb, descend, turn, contact, etc. |
| `value` | String / Numeric | Operational value associated with a command | Altitude, heading, frequency, etc. |
| `duration` | Numeric | Length of the transmission | Seconds |

**Note:** This table is preliminary. Column names and definitions will be replaced with the actual ATCO2 schema after the dataset is acquired.

---

## 3.2 ATCOSIM Dataset

### Data Source

**ATCOSIM: Air Traffic Control Simulation Speech Corpus**

Source:

https://huggingface.co/datasets/Jzuluaga/atcosim_corpus

### Dataset Description

ATCOSIM is an aviation speech corpus containing simulated ATC communications produced by professional air traffic controllers.

The corpus contains speech recordings and corresponding transcripts and may provide an additional source of aviation-domain audio for:

- ASR model evaluation
- ASR model training or fine-tuning
- Acoustic analysis
- Cross-dataset model comparison

ATCOSIM is expected to serve as a secondary dataset complementing ATCO2.

---

### Data Size and Shape

The exact dataset characteristics will be calculated after acquisition and inspection.

| Characteristic | Value |
| --- | --- |
| Size on disk | TBD |
| Number of rows / utterances | TBD |
| Number of columns | TBD |
| Number of audio files | TBD |
| Total audio duration | TBD |
| Audio format | TBD |
| Number of speakers | TBD |

---

### Time Period

**TBD / Not applicable**

The appropriate value will be determined after reviewing the corpus metadata.

---

### Unit of Observation

The expected unit of observation is:

> **One simulated ATC speech utterance and its associated transcript and metadata.**
> 

This will be confirmed after inspecting the dataset.

---

### Data Dictionary

The final data dictionary will use the actual ATCOSIM schema.

Expected fields may include:

| Column | Data Type | Definition | Potential Values |
| --- | --- | --- | --- |
| `utterance_id` | String / Categorical | Unique speech-record identifier | Unique ID |
| `audio` / `audio_path` | Audio / String | Speech recording or audio-file location | Audio file |
| `transcript` | String | Ground-truth transcription | Free text |
| `speaker_id` | Categorical | Identifier for the speaker | Speaker identifier |
| `duration` | Numeric | Length of the utterance | Seconds |
| `sample_rate` | Numeric | Audio sampling frequency | Hz |
| `metadata` | Mixed | Additional recording information | Dataset dependent |

**Note:** This table is preliminary and will be replaced with the actual dataset columns after acquisition.

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

For the information-extraction task, the target labels will be aviation entity annotations when available.

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

The exact target structure will depend on the annotations available in the acquired datasets.

---

### Readback Analysis

If controller/pilot exchange analysis is feasible, an additional derived target may represent whether an operational value in a pilot readback is consistent with the preceding controller instruction.

Potential values could include:

- Consistent
- Possible discrepancy

This target would be derived later in the project rather than taken directly from the source datasets.

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

The following variables may also be created to investigate Research Question 3:

- Transmission duration
- Word count
- Speech rate
- Signal-to-Noise Ratio
- RMS energy
- Silence percentage
- Number of commands
- Number of numeric values
- Number of aviation entities
- Speaker role
- Dataset source

These variables can be compared with model performance measurements such as Word Error Rate to investigate which communication characteristics are associated with higher or lower transcription accuracy.

---

## 3.5 Dataset Integration

The datasets will initially be preserved separately so their original structures and metadata remain intact.

During data preparation, a standardized utterance-level analytical dataset may be created with a common structure similar to:

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
`callsign_count`, `command_count`, `numeric_count`, and `entity_count` remain future work,
pending speaker-role labeling and the NLP entity-extraction work described under Research
Questions 2 and 4.

The final set of columns will be determined by the actual attributes available in the two source datasets.
