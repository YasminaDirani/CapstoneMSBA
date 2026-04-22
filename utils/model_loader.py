import os
from pathlib import Path
from typing import Any

import joblib
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "YASMINA_MODEL_PATH",
        ROOT_DIR / "artifacts" / "yasmina_eligibility" / "yasmina_eligibility_pipeline.joblib",
    )
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a trained model from disk with joblib and cache the resource."""
    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)
