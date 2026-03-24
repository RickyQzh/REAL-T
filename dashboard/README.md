# REAL-T Dashboard

This dashboard provides model-to-model comparison for results under `output/BASE`.

## Conda Environment

Create a dedicated conda environment for the dashboard and install dependencies from this directory:

```bash
cd /home/qzh88/Data/REAL-T-Challenge/REAL-T

conda create -n realt_dashboard_py312 python=3.12 -y
conda run -n realt_dashboard_py312 python -m pip install -r dashboard/requirements.txt
```

Verify the environment:

```bash
conda run -n realt_dashboard_py312 python - <<'PY'
import opencc
import streamlit
import plotly
import pandas
import transformers
from utils.asr_metrics import normalizer_for_transcript

print("opencc:", opencc.__file__)
print("streamlit:", streamlit.__version__)
print("normalized zh:", normalizer_for_transcript("嗯 主题不一样", "Ground Truth", "zh"))
PY
```

## Launch

Run from the `REAL-T` repo root.

`conda run` **captures stdout/stderr by default**, so Streamlit’s startup lines (local/network URLs, errors) can appear delayed or look like the process is hanging. Use `--no-capture-output` so output streams live:

```bash
conda run --no-capture-output -n realt_dashboard_py312 \
  streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

Alternatively, activate the env and run Streamlit directly (also live output):

```bash
conda activate realt_dashboard_py312
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

Then open `http://<host>:8501` (on a remote machine, use the server’s IP or SSH port forwarding).

## Data Rules

- Only reads `output/BASE`.
- Uses `datasets/REAL-T/BASE/*_meta.csv` as the metric-join metadata source.
- Uses `datasets/REAL-T/metadata/*_meta.csv` to derive `primary speaker` and the official `PRIMARY` subset with the same full-data logic as `REAL-T-Ext/generate_datasets/filter_subset.py`.
- Enrol-level filters additionally read `dashboard/enrol_quality/enrol_ter_full.csv` by default.
  - Override path via env var `REALT_ENROL_TER_FULL_CSV`.
  - Join key is `BASE.enrolment_speakers_utterance == enrol_ter_full.enrol_id`.
- Excludes `Fisher` from all views and statistics.
- `PRIMARY` is no longer derived by recomputing max `speaker_ratio` inside BASE.
  - It is first identified from full metadata (`datasets/REAL-T/metadata`) and then applied to `output/BASE`.
  - This matches the official `filter_PRIMARY(full)` semantics instead of the old BASE-only approximation.
- Custom subsets still read metrics from `output/BASE`, but their `primary speaker` semantics now follow the full metadata rather than BASE-local recomputation.

## Metrics

The dashboard supports these 11 single-select metrics:

- `TER / fireredasr-1/whisper`
- `TER / fireredasr-2`
- `SIM / enrol-mixture`
- `SIM / enrol-tse`
- `DNSMOS / SIG`
- `DNSMOS / BAK`
- `DNSMOS / OVRL`
- `DNSMOS / P808`
- `RATIO / precision`
- `RATIO / recall`
- `RATIO / f1`

## Notes

- Missing or unfinished metric CSVs are tolerated. Tables show `NA`, and charts skip missing values.
- The recommended environment above includes the dependencies needed for `utils.asr_metrics.normalizer_for_transcript`, so transcript-length filtering should use the repo's strict normalization logic instead of the dashboard fallback path.
- Non-SIM tables keep extra precision for small values so the mean is easier to compare with `*_TER.txt` / `*_summary.txt`.
- New sidebar filters:
  - `enrol quality (TER)`: `<= 0.05/0.10/0.15/0.20/0.30` or `不限`
  - `enrol length (GT词数)`: `0-5`, `>=5`, `>=10`, `>=15`, `>=20`, or `不限`
