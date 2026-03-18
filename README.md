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


## 1. Introduction

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


## 2. Installation

Datasets are at [huggingface](https://huggingface.co/datasets/SLbaba/REAL-T).

### 2.1 Clone the repository

```bash
git clone https://github.com/REAL-TSE/REAL-T.git
cd REAL-T

# install submodules (wesep + FireRedASR2S)
git submodule update --init --recursive
```

### 2.2 Create a Conda environment and install dependencies

```bash
conda create -n REAL-T python=3.10
conda activate REAL-T
pip install -r requirements.txt
# Reinstall GPU ORT last so silero-vad / wespeakerruntime do not leave CPU ORT active.
pip install --force-reinstall --no-deps onnxruntime-gpu==1.19.2
```

`requirements.txt` is the only supported Python dependency entrypoint for this repo. For RTX 5090 / `sm_120`, it resolves the `cu128` PyTorch wheels automatically. `wespeaker` remains the only GitHub dependency because local `wesep` imports it directly.

All top-level scripts source `env_setup.sh` by default. That helper activates `REAL-T` and appends the local `FireRedASR` / `FireRedASR2S` / `wesep` paths automatically. If you want to use a different env name temporarily, run them with `REALT_CONDA_ENV=<your_env_name>`.

### 2.3 Set up Linux PATH and PYTHONPATH

> Please replace `$PWD` below with the absolute path to this project (REAL-T repo root).

FireRedASR (ASR transcription) and **FireRedASR2S** (e.g. FireRedVAD) are expected under the REAL-T repo root. Initialize/update the submodules first, then add the repo roots to `PYTHONPATH` so that `import fireredasr` / `import fireredasr2s` work.

```
$ export PATH=$PWD/FireRedASR/fireredasr/:$PWD/FireRedASR/fireredasr/utils/:$PATH
$ export PYTHONPATH=$PWD/FireRedASR/:$PYTHONPATH
$ export PYTHONPATH=$PWD/FireRedASR2S/:$PYTHONPATH
$ export PYTHONPATH=$PWD/wesep/:$PYTHONPATH
```


### 2.4 Prepare Dataset and Checkpoints

Evaluation requires the `REAL-T` dataset and the ASR model checkpoint `FireRedASR-AED-L` and `whisper-large-v2` from Hugging Face. The dataset must be prepared in a specific format before running evaluation. To automatically set up everything, run:

```bash
bash -i ./pre.sh
```

## 3. Inference and Evaluation

### 3.1 TSE Inference on REAL-T

The `run_tse.sh` script below demonstrates how to perform TSE inference with the [Wesep toolkit](https://github.com/wenet-e2e/wesep) using a **BSRNN model** trained on **VoxCeleb1**. You can adapt its `input/output` structure to suit your own TSE model.

```bash
cd REAL-T
bash -i run_tse.sh
```

This script runs TSE inference for multiple datasets using a specified model. Each dataset will be processed individually, generating separated target speaker audio files.

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

### 3.2 Evaluation / 评估

#### 3.2.1 ASR-based Evaluation of TSE

##### 3.2.1.1 `transcribe_and_evaluation.sh` (FireRedASR-AED-L for CN, whisper-large-v2 for EN)

The `transcribe_and_evaluation.sh` script automates the evaluation of ASR outputs using Token Error Rate (TER) metrics. You can run transcription, evaluation, or both, depending on your needs. Before execution, make sure to review the script's parameters—such as whether to include the Fisher corpus, the evaluation subset, input/output directories, and other configurable options.

```bash
cd REAL-T

# Only ASR
bash -i ./transcribe_and_evaluation.sh 1

# Only Evaluation
bash -i ./transcribe_and_evaluation.sh 2

# Both ASR and Evaluation
bash -i ./transcribe_and_evaluation.sh 1 2
```

This script automates two main stages for the REAL-T datasets:

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
| `CHINESE_DATASETS` | Space-separated dataset names that use Chinese ASR (default: `AliMeeting AISHELL-4`). |
| `ENGLISH_DATASETS` | Space-separated dataset names that use English ASR (default: `AMI DipCo CHiME6 Fisher`). |
| `MAPPING_CSV_NAME` | Name of the CSV file mapping mixture audio to clean references (default: `tse_audio_mapping.csv`). |

This script uses **only FireRedASR-AED-L** (Chinese) and **whisper** (English), producing `{BASE_NAME}_TER.txt` / `{BASE_NAME}_TER.csv`. For transcription and WER with FireRedASR2-AED, use **transcribe_and_evaluation_asr2.sh** below.

```bash
BASE_DIRS=(
    "YourPath/output/PRIMARY/bsrnn_vox1"
    "YourPath/output/PRIMARY/bsrnn_hr_vox1"
)
```

##### 3.2.1.2 `transcribe_and_evaluation_asr2.sh` (FireRedASR2-AED for all 5 datasets)

This script is **independent** of `transcribe_and_evaluation.sh`: it processes **all datasets** using the vendored **FireRedASR2-AED** Python inference code under `./FireRedASR2S`. Outputs include:

- Transcription results: `{BASE_DIR}/{dataset}/FireRedASR2-AED/predicted.csv` (e.g., `REAL-T/output/PRIMARY/BSRNN/AISHELL-4/FireRedASR2-AED`)
- WER statistics: `{BASE_DIR}/{BASE_NAME}_TER_ASR2_AED.txt` and `{BASE_DIR}/{BASE_NAME}_TER_ASR2_AED.csv` (e.g., `REAL-T/output/PRIMARY/BSRNN/BSRNN_TER_ASR2_AED.txt` / `BSRNN_TER_ASR2_AED.csv`)

The script expects local FireRedASR2-AED weights in `./FireRedASR2S/pretrained_models/FireRedASR2-AED`. That directory must contain at least:

- `model.pth.tar`
- `cmvn.ark`
- `dict.txt`
- `train_bpe1000.model`

```bash
# ASR2 transcription (all datasets)
bash -i ./transcribe_and_evaluation_asr2.sh 1

# Only evaluation -> generate TER_ASR2_AED
bash -i ./transcribe_and_evaluation_asr2.sh 2

# Both transcription and evaluation
bash -i ./transcribe_and_evaluation_asr2.sh 1 2
```

Configurable variables such as `BASE_DIRS` (default: `./output/PRIMARY/BSRNN`) and `TEST_SET_DIR` follow the same conventions as `transcribe_and_evaluation.sh`. Unlike the main ASR script, this ASR2 script uses **FireRedASR2-AED for all 5 datasets** and does not split Chinese vs English models.

---

#### 3.2.2 Timing-based Evaluation of TSE (FireRedVAD)

In addition to TER, this repo provides a **time-segment evaluation** pipeline for TSE. It does not rely on ASR text; instead, it evaluates which time segments were extracted.

- Run **FireRedVAD** on TSE outputs (`output/.../<dataset>/wav/*.wav`) to obtain predicted speech timestamps.
- Use REAL-T metadata (`mixture_utterance + speaker`) and overlap annotations from  
  `./datasets/REAL-T/json/<dataset>/overlap_records.json`  
  to build target-speaker ground-truth timestamps.
- Compute **frame-level (10 ms) TP/FP/FN** and report **Micro Precision / Recall / F1**.

**Prerequisites**

1. **FireRedASR2S**: Initialize/update the submodule inside the REAL-T project directory (see Installation step 1). Set **PYTHONPATH** as in Installation step 3 so that `import fireredasr2s` works (same scheme as FireRedASR).
   **FireRedVAD weights are not auto-downloaded.** The script expects the model in `FireRedASR2S/pretrained_models/FireRedVAD/VAD/` (that directory must contain at least `cmvn.ark` and `model.pth.tar`). Recommended: download from [ModelScope (魔搭)](https://www.modelscope.cn/models/xukaituo/FireRedVAD/files). The ModelScope repo already has a `VAD` subfolder, so download to `FireRedASR2S/pretrained_models/FireRedVAD/` (do **not** append `/VAD`); the downloaded `VAD/` will then match the path the script uses:
   ```bash
   # Initialize FireRedASR2S first (if not done)
   git submodule update --init --recursive FireRedASR2S

   # Download FireRedVAD from ModelScope (saves to .../FireRedVAD/, repo's VAD/ becomes .../FireRedVAD/VAD/)
   pip install modelscope
   mkdir -p ./FireRedASR2S/pretrained_models/FireRedVAD
   python -c "from modelscope import snapshot_download; snapshot_download('xukaituo/FireRedVAD', local_dir='./FireRedASR2S/pretrained_models/FireRedVAD')"
   ```
   If the directory is missing or invalid, Mode 1 will raise an error and exit.
2. **Environment for VAD**: The script defaults to `conda activate REAL-T`. You can activate it manually:
   ```bash
   conda activate REAL-T
   ```
   Or just run the top-level shell script directly and let `env_setup.sh` do it for you.
3. **GT overlap JSON for Mode 2 (evaluation)**: VAD **evaluation** (Mode 2) requires overlap JSON in the following layout. Copy the output of [REAL-T-Ext-channel-re-seclection](https://github.com/REAL-TSE/REAL-T-Ext-channel-re-seclection) into the project:
   ```bash
   # From REAL-T repo root: copy json output to datasets
   mkdir -p ./datasets/REAL-T/json
   cp -r /path/to/REAL-T-Ext-channel-re-seclection/output/REAL-T-datasets/json/* ./datasets/REAL-T/json/
   ```
   The directory must contain per-dataset subdirs (e.g. `AMI`, `AliMeeting`, ...), each with `overlap_records.json` (and any other files produced by REAL-T-Ext). **Without this layout, VAD timing evaluation (Mode 2) cannot run.**

```bash
cd REAL-T

# Only VAD (generate timestamp predictions)
bash -i ./vad_and_evaluation.sh 1

# Only timing evaluation (requires existing vad_segments.jsonl and ./datasets/REAL-T/json)
bash -i ./vad_and_evaluation.sh 2

# Full pipeline
bash -i ./vad_and_evaluation.sh 1 2
```

The two modes produce the following outputs:

- **Mode 1 (VAD)**: Predicted timestamps at `"{BASE_DIR}/{dataset}/FireRedVAD/vad_segments.jsonl"`.
- **Mode 2 (Evaluation)** (two sub-steps):
  1. **Prepare labels**: From meta + overlap JSON, write `{BASE_DIR}/{dataset}/FireRedVAD/label_segments.jsonl` (one row per utterance: `utterance`, `mix_duration`, `label_segments` in relative time with collar applied).
  2. **Eval from jsonl only**: Read only `vad_segments.jsonl` and `label_segments.jsonl` (and mapping CSV) to compute frame-level metrics; no meta or overlap files are read in this step. Outputs:
     - Per-utterance details: `{BASE_DIR}/{BASE_NAME}_TSE_TIMING.csv`
     - Report: `{BASE_DIR}/{BASE_NAME}_TSE_TIMING.txt`

Default metric settings:

- Frame shift: `FRAME_SHIFT=0.01` (10 ms)
- GT boundary collar: `COLLAR=0.05` (±50 ms)
- Aggregation: Micro (accumulate TP/FP/FN first, then compute P/R/F1)
- VAD threshold: `SPEECH_THRESHOLD=0.5`

Evaluation definitions (frame-level):

- `TP_dur`: Duration where both prediction and GT are speech
- `FP_dur`: Duration where prediction is speech but GT is not (false extraction)
- `FN_dur`: Duration where GT is speech but prediction is not (missed extraction)
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2TP / (2TP + FP + FN)`

| Variable Name | Description |
| --- | --- |
| `BASE_DIRS` | List of TSE output root directories to evaluate (e.g., `./output/PRIMARY/bsrnn_vox1`). |
| `TEST_SET_DIR` | REAL-T metadata directory (`PRIMARY` or `BASE`). |
| `GT_JSON_BASE_DIR` | Root directory for overlap JSON (default `./datasets/REAL-T/json`). Must be prepared by copying REAL-T-Ext output as in README. |
| `FIREREDASR2S_ROOT` | Path to FireRedASR2S repo (default `./FireRedASR2S`; clone under REAL-T). |
| `FIRERED_VAD_MODEL_DIR` | FireRedVAD model directory (default `$FIREREDASR2S_ROOT/pretrained_models/FireRedVAD/VAD`; must be locally available). |
| `FRAME_SHIFT` | Frame-level evaluation time granularity (default: `0.01`). |
| `COLLAR` | GT boundary expansion (default: `0.05`). |
| `MATCH_TOLERANCE` | Tolerance for matching `mixture` start/end times (default: `0.02`). |
| `SPEECH_THRESHOLD` | FireRedVAD speech detection threshold (default: `0.5`). |

Notes:
- If `vad_segments.jsonl` has not been generated, Mode 2 will skip the corresponding dataset and log it in the report.
- FireRedVAD weights are **not** auto-downloaded; the model directory must exist and contain the manually downloaded files (see Prerequisites above). If it does not exist, Mode 1 will raise an error and exit.

---

#### 3.2.3 Speaker Similarity Evaluation (WeSpeaker)

The `compute_spk_similarity.sh` script computes speaker cosine similarity between TSE estimation (or mixture) audio and enrolment audio. **Mode 1** produces a per-utterance details CSV; **Mode 2** generates a summary TXT with Per-dataset and Per-language Statistics from existing CSV. `wespeakerruntime` is already included in `requirements.txt`.

**Pair mode** (env `SPK_SIM_PAIR_MODE`):
- **`tse_enrol`** (default): TSE output vs enrolment. Outputs `{BASE_NAME}_spk_similarity.csv` and `{BASE_NAME}_spk_similarity_summary.txt`.
- **`mixture_enrol`**: Mixture (input) vs enrolment, used as a baseline. Outputs `{BASE_NAME}_spk_similarity_mixture_enrol.csv` and `{BASE_NAME}_spk_similarity_mixture_enrol_summary.txt`.

```bash
cd REAL-T

# 1: compute & generate CSV only (requires WeSpeaker)
bash -i ./compute_spk_similarity.sh 1

# 2: generate TXT from existing CSV (no WeSpeaker needed)
bash -i ./compute_spk_similarity.sh 2

# run 1 then 2
bash -i ./compute_spk_similarity.sh 1 2

# Mixture baseline: compare mixture (input) vs enrolment instead of TSE vs enrolment.
# Produces *_spk_similarity_mixture_enrol.csv and *_spk_similarity_mixture_enrol_summary.txt.
SPK_SIM_PAIR_MODE=mixture_enrol bash -i ./compute_spk_similarity.sh 1 2
```

| Variable Name | Description |
| --- | --- |
| `TEST_SET_DIR` | Directory with REAL-T metadata CSVs (default: `./datasets/REAL-T/PRIMARY`). |
| `BASE_DIRS` | Space-separated TSE output roots. Each root will generate one CSV + one summary TXT. |
| `MAPPING_CSV` | Global REAL-T mapping file (default: `./datasets/REAL-T/mapping.csv`). |
| `SPK_SIM_PAIR_MODE` | Pair mode: `tse_enrol` (default) or `mixture_enrol` (baseline). |
| `WESPEAKER_LANG` | Default WeSpeaker language (`en` or `chs`). |
| `WESPEAKER_PROVIDER` | ONNX Runtime provider policy: `auto`, `cuda`, or `cpu`. |
| `WESPEAKER_DATASET_LANG_OVERRIDES` | Comma-separated dataset-to-language map (default: `AISHELL-4:chs,AliMeeting:chs`; other datasets use `WESPEAKER_LANG`). |
| `MAX_SAMPLES` | Optional cap for quick smoke tests. |
| `NUM_WORKERS` | Reserved argument; current implementation runs serially for stability. |

---

#### 3.2.4 DNSMOS Speech Quality Evaluation (SIG, BAK, OVRL, P808)

The `compute_dnsmos.sh` script computes **DNSMOS** (non-intrusive perceptual speech quality) scores for TSE output audios: **SIG** (speech quality), **BAK** (background noise quality), **OVRL** (overall quality), and **P808** (ITU-T P.808 overall MOS). **Mode 1** produces a per-utterance CSV (`{BASE_NAME}_dnsmos.csv`) and optionally a summary TXT (`{BASE_NAME}_dnsmos.txt`); **Mode 2** regenerates the summary TXT from an existing CSV (no DNSMOS model needed).

**Prerequisites**

- Python dependencies are already in `requirements.txt` (onnxruntime, librosa, soundfile, pandas, tqdm). Auto-download of models also requires `huggingface_hub` (usually present).
- **DNSMOS ONNX models**: If `sig_bak_ovr.onnx` and `model_v8.onnx` are missing under `DNSMOS_MODEL_DIR`, the script will **automatically download** them from [Hugging Face (Vyvo-Research/dnsmos)](https://huggingface.co/Vyvo-Research/dnsmos) on first run. To disable auto-download (e.g. offline), set `DNSMOS_NO_DOWNLOAD=1` or use `--no_download_models` and place the two files manually (from [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge) or the same Hugging Face repo).

```bash
cd REAL-T

# 1: compute DNSMOS & generate CSV (and optionally TXT)
bash -i ./compute_dnsmos.sh 1

# 2: generate TXT from existing CSV (no DNSMOS model needed)
bash -i ./compute_dnsmos.sh 2

# run 1 then 2
bash -i ./compute_dnsmos.sh 1 2
```

| Variable Name | Description |
| --- | --- |
| `TEST_SET_DIR` | Directory with REAL-T metadata CSVs (default: `./datasets/REAL-T/PRIMARY`). |
| `BASE_DIRS` | Space-separated TSE output roots. Each root will generate `{BASE_NAME}_dnsmos.csv` and `{BASE_NAME}_dnsmos.txt`. |
| `DNSMOS_MODEL_DIR` | Directory for ONNX models (default: `./DNSMOS`). Missing files are auto-downloaded from Hugging Face unless `DNSMOS_NO_DOWNLOAD=1`. Required for Mode 1 only. |
| `DNSMOS_NO_DOWNLOAD` | Set to `1` to disable auto-download of models; you must provide `sig_bak_ovr.onnx` and `model_v8.onnx` in `DNSMOS_MODEL_DIR` (default: `0`). |
| `DNSMOS_PROVIDER` | ONNX Runtime provider: `auto` (prefer CUDA if available), `cuda`, or `cpu`. Use `cuda` to speed up inference when GPU is available (default: `auto`). |
| `MAX_SAMPLES` | Optional cap on number of utterances per base_dir (for quick tests). |

Outputs:

- **CSV**: `{BASE_DIR}/{BASE_NAME}_dnsmos.csv` — columns: base_dir, dataset, utterance, path, SIG, BAK, OVRL, P808, status, error_message.
- **TXT**: `{BASE_DIR}/{BASE_NAME}_dnsmos.txt` — overall and per-dataset / per-language statistics (mean, std, min, max for SIG, BAK, OVRL, P808).

---

## 4. FireRedASR2-AED Local Setup

To obtain **FireRedASR2-AED** transcription and WER for all datasets, prepare the local model directory first, then run **transcribe_and_evaluation_asr2.sh**.

### 4.1 Preparing Local Weights

1. Initialize `FireRedASR2S` inside the REAL-T project if it is not already present:
   ```bash
   git submodule update --init --recursive FireRedASR2S
   ```

2. Download the Python inference weights to `./FireRedASR2S/pretrained_models/FireRedASR2-AED`:
   ```bash
   pip install -U modelscope
   modelscope download --model xukaituo/FireRedASR2-AED --local_dir ./FireRedASR2S/pretrained_models/FireRedASR2-AED
   ```
   Or with Hugging Face:
   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli download FireRedTeam/FireRedASR2-AED --local-dir ./FireRedASR2S/pretrained_models/FireRedASR2-AED
   ```

3. Confirm the directory contains `model.pth.tar`, `cmvn.ark`, `dict.txt`, and `train_bpe1000.model`.

### 4.2 Running Transcription and Evaluation

From the **REAL-T** repository root:

```bash
cd /path/to/REAL-T
bash -i ./transcribe_and_evaluation_asr2.sh 1 2
```

This runs local FireRedASR2-AED transcription on all datasets and produces `BSRNN_TER_ASR2_AED.txt` / `BSRNN_TER_ASR2_AED.csv`.

### 4.3 Script `asr_inference_fireredasr2.py`

| Item | Description |
|------|-------------|
| **Purpose** | Uses the vendored `FireRedASR2S` Python API to read `tse_audio_mapping.csv` (columns `utterance`, `path`) and write `predicted.csv` compatible with the existing ASR evaluation pipeline. |
| **Usage** | Typically invoked automatically by **transcribe_and_evaluation_asr2.sh** in Mode 1 for all datasets. To run on a single dataset manually:<br><br>`python3 asr/asr_inference_fireredasr2.py --audio_mapping /path/to/tse_audio_mapping.csv --output_dir /path/to/FireRedASR2-AED --dataset_name AISHELL-4 --model_dir ./FireRedASR2S/pretrained_models/FireRedASR2-AED` |
| **Expected Output** | Generates **predicted.csv** (columns `utterance`, `transcript`) under `--output_dir`, in the same format as `asr_inference.py` output, for use with `asr_evaluation.py` to compute TER. When used with **transcribe_and_evaluation_asr2.sh**, results for all datasets are stored under each dataset's `FireRedASR2-AED/` directory, and the evaluation stage produces **`{BASE_NAME}_TER_ASR2_AED.txt`** and **`{BASE_NAME}_TER_ASR2_AED.csv`**. |


## 5. Results

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

## 6. Citation

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


## 7. Contact

For any questions, please contact: `shuaiwang@nju.edu.cn`
