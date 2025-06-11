# REAL-T: Real Conversational Mixtures for Target Speaker Extraction

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

![Pipeline](REAL-T.png)


## Introduction

Target Speaker Extraction (TSE) models have demonstrated impressive performance on synthetic datasets such as **LibriMix** and **WSJMix**. However, these benchmarks lack the acoustic realism and conversational dynamics of actual human interactions — such as spontaneous speech, overlapping turns, and environmental noise — limiting their relevance to real-world scenarios.

Efforts like **REAL-M** and **LibriCSS** have attempted to bridge this gap. REAL-M collects simultaneous speech in shared environments through multi-speaker read-aloud sessions, while LibriCSS re-records isolated utterances via synchronized playback. Though valuable, these datasets still fall short of capturing the nuances of real conversations — including irregular speaker turns, sporadic utterances, and authentic background conditions.

To address these limitations, we introduce **REAL-T**, the first conversation-centric dataset specifically designed for TSE under real-world conditions. Built from speaker diarization corpora, REAL-T naturally includes overlapping speech, enrollment-ready segments, and complex conversational behaviors.

Key features of REAL-T include:

- **Multi-lingual**: English and Mandarin recordings
- **Multi-genre**: Covering diverse conversational scenarios
- **Multi-enrollment**: Multiple enrollment utterances per target speaker for robustness

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
```

### 2. Create a Conda environment and install dependencies

```bash
conda create -n REAL-T python=3.9
conda activate REAL-T
pip install -r requirements.txt
```

### 3. Set up Linux PATH and PYTHONPATH

```
$ export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
$ export PYTHONPATH=$PWD/:$PYTHONPATH
$ export PYTHONPATH=$PWD/wesep:$PYTHONPATH
```


### 4. Prepare Dataset and Checkpoints

Evaluation requires the `REAL-T` dataset and the ASR model checkpoint `FireRedASR-AED-L` from Hugging Face. The dataset must be prepared in a specific format before running evaluation. To automatically set up everything, run:

```bash
bash -i /pre.sh
```
After that, navigate to the working directory:

```bash
cd ./wesep/examples/librimix/tse/v2
```

Create an `exp/` directory and inside it, create subdirectories for each model listed below. Each subdirectory must contain:

* avg_model.pt — the model checkpoint
* 
* config.yaml — the model configuration file

⚠️ Important: The directory names must exactly match those defined in `./tse_baseline/tse_model/` to ensure proper model loading.

The required directory structure is (you only need to include the models you intend to evaluate):

```
exp
├── bsrnn_100
│   ├── avg_model.pt
│   └── config.yaml
├── bsrnn_360
│   ├── avg_model.pt
│   └── config.yaml
├── bsrnn_feats_SDR
│   ├── avg_model.pt
│   └── config.yaml
├── bsrnn_hr_100
│   ├── avg_model.pt
│   └── config.yaml
├── bsrnn_hr_360
│   ├── avg_model.pt
│   └── config.yaml
├── bsrnn_hr_vox1
│   ├── avg_model.pt
│   └── config.yaml
├── spex_plus_100
│   ├── avg_model.pt
│   └── config.yaml
├── spex_plus_360
│   ├── avg_model.pt
│   └── config.yaml
└── usef_tfgridnet_100
    ├── avg_model.pt
    └── config.yaml
```

## Evaluation

### TSE Inference on REAL-T

If you are using the Wesep toolkit, you can directly run the script below. Otherwise, please refer to `run_tse.sh` to build your own TSE system based on its structure and logic. Before execution, make sure to review the script's parameters—such as the evaluation subset, output directory, and other configurable options.

➡️ See [Bash Script Parameters: run_tse.sh](#run_tsesh) for detailed setup.

```bash
cd REAL-T
bash -i run_tes.sh
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
| `MODEL_NAME 🚩` | Name of the TSE model used for inference (e.g., `bsrnn_hr_360`). |
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
    "/root/shared-nvme/open-source/REAL-T/output/PRIMARY/bsrnn_vox1"
    "/root/shared-nvme/open-source/REAL-T/output/PRIMARY/bsrnn_hr_vox1"
)
```

---


## Results

Results of the several recently proposed TSE models on the PRIMARY test set can be found on our [REAL-T Page](https://real-tse.github.io/).


## Citation

```
xxxx
```


## Contact

For any questions, please contact: `xxxxx@gmail.com`