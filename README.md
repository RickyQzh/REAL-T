# REAL-T: Real Conversational Mixtures for Target Speaker Extraction

<p align="center">
  <img src="./figure/logo.drawio.svg" alt="REAL-T Logo" width="200"/>
</p>

<p align="center">
  <a href="xxxxxxxx">
    <img src="https://img.shields.io/badge/Paper-ArXiv-red" alt="Paper">
  </a>
  <a href="https://real-tse.github.io/">
    <img src="https://img.shields.io/badge/REAL--T-Page-blue" alt="REAL-T Page">
  </a>
  <a href="https://huggingface.co/datasets/SLbaba/REAL-T">
    <img src="https://img.shields.io/badge/Datasets-HuggingFace-yellow" alt="Datasets on Hugging Face">
  </a>
</p>

![Pipeline](./figure/pipeline.svg)


## Introduction

Target Speaker Extraction (TSE) models have demonstrated impressive performance on synthetic datasets such as **LibriMix** and **WSJMix**. However, these benchmarks lack the acoustic realism and conversational dynamics of actual human interactions — such as spontaneous speech, overlapping turns, and environmental noise — limiting their relevance to real-world scenarios.

Efforts like **REAL-M** and **LibriCSS** have attempted to bridge this gap. REAL-M collects simultaneous speech in shared environments through multi-speaker read-aloud sessions, while LibriCSS re-records isolated utterances via synchronized playback. Though valuable, these datasets still fall short of capturing the nuances of real conversations — including irregular speaker turns, sporadic utterances, and authentic background conditions.

To address these limitations, we introduce **REAL-T**, the first conversation-centric dataset specifically designed for TSE under real-world conditions. Built from speaker diarization corpora, REAL-T naturally includes overlapping speech, enrollment-ready segments, and complex conversational behaviors.

Key features of REAL-T include:

- **Multi-lingual**: English and Mandarin recordings
- **Multi-genre**: Covering diverse conversational scenarios
- **Multi-enrollment**: Multiple enrollment utterance from different parts of the conversation

To support controlled evaluation, we define two test sets:

- **BASE**: A filtered, balanced subset for initial testing
- **PRIMARY**: A more realistic and challenging benchmark

Evaluations reveal that existing TSE models suffer significant performance degradation on REAL-T, highlighting the need for more robust approaches tailored to real conversational speech.

For more details, refer to our paper: [REAL-T Paper](xxxxxxxxx)


## Installation

Datasets are at [huggingface](https://huggingface.co/datasets/SLbaba/REAL-T).

### 1. Clone the repository

```bash
git clone https://github.com/REAL-TSE/REAL-T.git
cd REAL-T

# install wesep
git submodule init
git submodule update
```

### 2. Create a Conda environment and install dependencies

```bash
conda create -n REAL-T python=3.9
conda activate REAL-T
pip install -r requirements.txt
```

### 3. Set up Linux PATH and PYTHONPATH

> Please replace `$PWD` below with the absolute path to this project

```
$ export PATH=$PWD/FireRedASR/fireredasr/:$PWD/FireRedASR/fireredasr/utils/:$PATH
$ export PYTHONPATH=$PWD/FireRedASR/:$PYTHONPATH
$ export PYTHONPATH=$PWD/wesep/:$PYTHONPATH
```


### 4. Prepare Dataset and Checkpoints

Evaluation requires the `REAL-T` dataset and the ASR model checkpoint `FireRedASR-AED-L` and `whisper-large-v2` from Hugging Face. The dataset must be prepared in a specific format before running evaluation. To automatically set up everything, run:

```bash
bash -i ./pre.sh
```

## Evaluation

### TSE Inference on REAL-T

The `run_tse.sh` script below demonstrates how to perform TSE inference with the [Wesep toolkit](https://github.com/wenet-e2e/wesep) using a **BSRNN model** trained on **VoxCeleb1**. You can adapt its `input/output` structure to suit your own TSE model.

➡️ See [Bash Script Parameters: run_tse.sh](#run_tsesh) for detailed setup.

```bash
cd REAL-T
bash -i run_tse.sh
```

### ASR-based Evaluation of TSE

The `transcribe_and_evaluation.sh` script automates the evaluation of ASR outputs using Token Error Rate (TER) metrics. You can run transcription, evaluation, or both, depending on your needs. Before execution, make sure to review the script’s parameters—such as whether to include the Fisher corpus, the evaluation subset, input/output directories, and other configurable options.
➡️ See [Bash Script Parameters: transcribe_and_evaluation.sh](#transcribe_and_evaluationsh) for detailed setup.

```bash
cd REAL-T

# Only ASR
bash -i ./transcribe_and_evaluation.sh 1

# Only Evaluation
bash -i ./transcribe_and_evaluation.sh 2

# Both ASR and Evaluation
bash -i ./transcribe_and_evaluation.sh 1 2
```

#### `run_tse.sh`

This script runs Target Speaker Extraction (TSE) inference for multiple datasets using a specified model. Each dataset will be processed individually, generating separated target speaker audio files.

| **Variable Name** | **Description** |
| :--- | :--- |
| `MODEL_NAME 🚩` | Name of the TSE model used for inference (e.g., `bsrnn_vox1`). |
| `DATASETS 🚩` | List of datasets to process (e.g., AliMeeting, AMI, CHiME6, AISHELL-4, DipCo). Fisher can also be included if needed. |
| `TEST_SET 🚩` | Test subset to use: `PRIMARY` or `BASE`. |
| `DEVICE 🚩` | Device on which to run inference (`cuda` for GPU, `cpu` for CPU). |
| `BASE_META_PATH` | Base directory containing metadata CSV files for each dataset. |
| `BASE_OUTPUT_DIR` | Directory where the separated audios will be saved. |
| `TSE_SCRIPT` | Path to the TSE inference Python script (`tse.py`). |
| `META_CSV_PATH` | Path to the CSV file containing mixture and enrolment utterance metadata. |
| `UTTERANCE_MAP_CSV` | Path to the CSV mapping enrolment utterances to mixture utterances. |
| `OUTPUT_DIR` | Directory where output audios for each dataset will be stored. |

---

#### transcribe_and_evaluation.sh

This script automates two main stages for the REAL-T datasets:

- **ASR transcription** using different models depending on dataset language.
- **ASR evaluation** using TER (Token Error Rate) metrics.

The operation mode is controlled by passing one or two arguments:

- `1` → only run ASR transcription
- `2` → only run evaluation
- `1 2` → run both transcription and evaluation sequentially

Typically, run `1 2` for a full process.

| Step | Function | Description |
| --- | --- | --- |
| 1 | `run_asr` | Transcribe all audio files using the appropriate ASR model (Chinese or English) for each dataset. |
| 2 | `run_asr_evaluation` | Evaluate the ASR outputs by calculating Token Error Rate (TER) against the ground-truth transcripts. |

| Variable Name | Description |
| --- | --- |
| `TEST_SET_DIR 🚩` | **(Required)** Directory containing the ground-truth test set metadata (BASE or PRIMARY). |
| `BASE_DIRS 🚩` | **(Required)** List of base directories containing predicted TSE outputs to be transcribed and/or evaluated, as listed below. |
| `INCLUDING_FISHER` | Whether to include the Fisher corpus in evaluation (`True` or `False`). |
| `MODES` | Runtime arguments specifying which mode(s) to execute (1 for ASR, 2 for Evaluation). |
| `ASR_SCRIPT` | Path to the Python script performing ASR inference (`asr_inference.py`). |
| `EVAL_SCRIPT` | Path to the Python script performing ASR evaluation (`asr_evaluation.py`). |
| `CHINESE_ASR_MODEL` | Name of the model used for Chinese datasets (e.g., `FireRedASR-AED-L`). |
| `ENGLISH_ASR_MODEL` | Name of the model used for English datasets (e.g., `whisper-large-v2`). |
| `MAPPING_CSV_NAME` | Name of the CSV file mapping mixture audio to clean references (default: `tse_audio_mapping.csv`). |

```bash
BASE_DIRS=(
    "YourPath/output/PRIMARY/bsrnn_vox1"
    "YourPath/output/PRIMARY/bsrnn_hr_vox1"
)
```

---


## Results

The table below compares the performance of several recently proposed TSE models on the simulated Libri2Mix and PRIMARY test sets. 

<div align="center">

| Model       | Training Data     | Libri2Mix SI-SDR (dB) | PRIMARY zh (%) | PRIMARY en (%) |
|:-------------:|:-------------------:|:------------------------:|:----------------:|:----------------:|
| TSELM-L     | Libri2Mix-360     | /                      | 331.73         | 192.39         |
| USEF-TFGridnet | Libri2Mix-100  | **18.05**              | 67.98          | 87.27          |
| **BSRNN**   | Libri2Mix-100     | 12.95                  | 81.74          | 91.20          |
|             | Libri2Mix-360     | 16.57                  | 69.80          | 73.61          |
|             | VoxCeleb1         | 16.50                  | **57.61**      | 69.63          |
| **BSRNN_HR**| Libri2Mix-100     | 15.91                  | 70.03          | 78.96          |
|             | Libri2Mix-360     | 17.99                  | 63.38          | 74.64          |
|             | VoxCeleb1         | 16.38                  | 58.77          | **66.46**      |

</div>

## Citation

```
@inproceedings{li25da_interspeech,
  title     = {{REAL-T: Real Conversational Mixtures for Target Speaker Extraction}},
  author    = {{Shaole Li and Shuai Wang and Jiangyu Han and Ke Zhang and Wupeng Wang and Haizhou Li}},
  year      = {{2025}},
  booktitle = {{Interspeech 2025}},
  pages     = {{1923--1927}},
  doi       = {{10.21437/Interspeech.2025-2662}},
  issn      = {{2958-1796}},
}
```


## Contact

For any questions, please contact: `shuaiwang@nju.edu.cn`
