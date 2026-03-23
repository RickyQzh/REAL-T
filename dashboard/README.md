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
- Uses `datasets/REAL-T/BASE/*_meta.csv` as the single metadata source.
- Excludes `Fisher` from all views and statistics.
- `PRIMARY` is implemented as a BASE-derived preset:
  - primary speaker only
  - `speaker_ratio >= 20%`
  - `mixture_duration <= 30s`
  - `transcript_length > 5`
- All other custom subsets are also derived from the BASE metadata.

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
