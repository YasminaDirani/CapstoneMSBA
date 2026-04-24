# Financial Aid Analytics

Streamlit dashboard for thesis-grade financial aid analysis, decision patterns, and model evidence.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set the main file path to `streamlit_app.py`.
4. Deploy with the included `requirements.txt` and `runtime.txt`.

The app is now self-contained for deployment: the enhanced cleaned-data exports, eligibility artifacts, percentage-model artifacts, and purchasing-power experiment runtime code are packaged in this repository.

## Default artifact behavior

For local development, when a sibling source repo named `Yasmina-MSBA` is present next to this folder, or when `YASMINA_SOURCE_ROOT` is set, the app prefers refreshed files from that source repo:

- cleaned data from `cleaned data/faid_cleaned.csv`
- eligibility artifacts from `artifacts/yasmina_eligibility_notebook/`
- percentage-model artifacts from `artifacts/yasmina_percentage/`
- purchasing-power experiment code from `faid_models/POC/`

In deployment, that sibling repo will not exist, so the app falls back to the packaged copies in this folder:

- `cleaned data/`
- `artifacts/yasmina_eligibility_notebook/`
- `artifacts/yasmina_percentage/`
- `faid_models/`

The percentage pipeline artifact is large, around 96 MB, but remains under GitHub's 100 MB per-file limit.

## Data sensitivity

The packaged data includes financial-aid application records and raw review fields. Deploy from a private repository or a protected Streamlit workspace unless the data has been approved for public sharing.

## Optional environment overrides

- `YASMINA_SOURCE_ROOT`
- `YASMINA_DATA_PATH`
- `YASMINA_MODEL_PATH`
- `YASMINA_MODEL_METADATA_PATH`
- `YASMINA_MODEL_1_NOTEBOOK_PATH`
- `YASMINA_PERCENTAGE_METADATA_PATH`
- `YASMINA_PERCENTAGE_DECISION_PATH`
- `YASMINA_PERCENTAGE_FAIRNESS_PATH`
- `YASMINA_PERCENTAGE_FAIRNESS_ACTIONS_PATH`
- `YASMINA_PERCENTAGE_STAGE1_IMPORTANCE_PATH`
- `YASMINA_PERCENTAGE_STAGE2_IMPORTANCE_PATH`

These are optional. The app can run either from the refreshed MSBA outputs or from packaged fallback assets.
