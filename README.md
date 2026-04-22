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

## Packaged deployment assets

- Data file: `faid_cleaned.csv`
- Saved model: `artifacts/yasmina_eligibility/yasmina_eligibility_pipeline.joblib`
- Model metadata: `artifacts/yasmina_eligibility/yasmina_eligibility_metadata.json`
- Model 1 benchmark summary: `artifacts/model_1/model_1_summary.json`

## Optional environment overrides

- `YASMINA_DATA_PATH`
- `YASMINA_MODEL_PATH`
- `YASMINA_MODEL_1_NOTEBOOK_PATH`

These are optional. The app is configured to run from the packaged repo files by default.
