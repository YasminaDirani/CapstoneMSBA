import os
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = Path(
    os.environ.get("YASMINA_DATA_PATH", ROOT_DIR / "faid_cleaned.csv")
)


@st.cache_data(show_spinner=False)
def load_csv(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame and cache the result."""
    path = Path(csv_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return pd.read_csv(path)
