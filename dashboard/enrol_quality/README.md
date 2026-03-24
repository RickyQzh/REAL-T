# Enrol Quality: ASR Transcription

这个目录用于评估 `enrolment_speakers` 的转录质量。

## 文件说明

- `prepare_enrol_inventory.py`
  - 生成 `enrol_inventory.csv`
  - 从 `datasets/REAL-T/mapping/*_mixture_and_enrolment.csv` 确定 `enrol_id -> dataset`
  - 从 `datasets/REAL-T/metadata/*_meta.csv` 的 `language` 列读取中英文证据并映射成 `ch/en`
- `transcribe_enrol_asr.py`
  - 对 inventory 里的 enrol 执行 ASR，支持 `whisper` / `firered`
  - 支持 `--language ch|en`，可直接输出单语 CSV，也可继续按 shard 落盘
- `merge_enrol_transcripts.py`
  - 合并 `shards/` 下分片 CSV，生成最终转录 CSV
- `extract_enrol_ground_truth.py`
  - 从 `datasets/REAL-T/json/*/no_overlap_segments.json` 提取每个 enrol 的文本真值
  - 规则与 `REAL-T-Ext/generate_datasets/construct.py` 一致：
    `enrol_id = "{utterance_id}_{speaker}_{start:.2f}_{end:.2f}"`
- `build_enrol_ter_report.py`
  - 合并中文 ASR、英文 ASR 与 `enrol_ground_truth_from_no_overlap.csv`
  - 生成全量 enrol 评估表（含 `TER`）
- `run_enrol_quality_pipeline.py`
  - 一键生成最终 `enrol_ter_full.csv`
  - 默认将 inventory / GT / ASR 中间 CSV 写入临时目录
  - 完成后自动清理中间产物；调试时可用 `--keep_intermediates 1`

## 推荐用法：一键流程

```bash
cd /home/qzh88/Data/REAL-T-Challenge/REAL-T

# 单卡全量：中文默认 FireRed，英文默认 Whisper
CUDA_VISIBLE_DEVICES=0 python dashboard/enrol_quality/run_enrol_quality_pipeline.py

# 快速冒烟：CPU + Whisper tiny，且只跑前 10 条
python dashboard/enrol_quality/run_enrol_quality_pipeline.py \
  --ch_model whisper \
  --en_model whisper \
  --whisper_model_name openai/whisper-tiny \
  --device cpu \
  --max_samples 10

# 调试：保留中间 CSV
CUDA_VISIBLE_DEVICES=0 python dashboard/enrol_quality/run_enrol_quality_pipeline.py \
  --keep_intermediates 1
```

默认最终只保留 `dashboard/enrol_quality/enrol_ter_full.csv`。  
`enrol_inventory*.csv`、`enrol_transcripts_*.csv`、`enrol_ground_truth_from_no_overlap.csv`、
`shards/`、`logs/` 都视为中间产物，不再建议提交到版本库。

## 手动流程

如果需要分步排查，仍然可以单独运行这些脚本：

```bash
python dashboard/enrol_quality/prepare_enrol_inventory.py

python dashboard/enrol_quality/transcribe_enrol_asr.py \
  --model whisper \
  --inventory_csv dashboard/enrol_quality/enrol_inventory.csv \
  --language en \
  --num_shards 1 --shard_index 0 \
  --device cuda:0

python dashboard/enrol_quality/transcribe_enrol_asr.py \
  --model firered \
  --inventory_csv dashboard/enrol_quality/enrol_inventory.csv \
  --language ch \
  --num_shards 1 --shard_index 0 \
  --device cuda:0

python dashboard/enrol_quality/extract_enrol_ground_truth.py \
  --inventory_csv dashboard/enrol_quality/enrol_inventory.csv \
  --output_csv dashboard/enrol_quality/enrol_ground_truth_from_no_overlap.csv

python dashboard/enrol_quality/build_enrol_ter_report.py \
  --gt_csv dashboard/enrol_quality/enrol_ground_truth_from_no_overlap.csv \
  --ch_asr_csv dashboard/enrol_quality/ch_asr.csv \
  --en_asr_csv dashboard/enrol_quality/en_asr.csv \
  --output_csv dashboard/enrol_quality/enrol_ter_full.csv
```

## 中间 ASR CSV 字段

最终合并文件 `enrol_transcripts_<model>.csv` 字段如下：

- `enrol_id`: enrol 标识
- `dataset`: 数据集标识（AMI/CHiME6/DipCo/AliMeeting/AISHELL-4）
- `ch_en`: 语言标签（`ch` 或 `en`）
- `transcript`: ASR 转录文本
