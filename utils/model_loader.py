import os
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

from utils.source_paths import (
    DEFAULT_MODEL_PATH as RESOLVED_DEFAULT_MODEL_PATH,
    ROOT_DIR,
)

DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "YASMINA_MODEL_PATH",
        RESOLVED_DEFAULT_MODEL_PATH,
    )
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a trained model or model bundle from disk with joblib and cache the resource."""
    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)
