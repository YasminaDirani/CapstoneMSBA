from __future__ import annotations

import pandas as pd


POSITIVE_DECISION_LABELS = {
    "awarded",
    "awarded_affidavit_of_promise",
}
NEGATIVE_DECISION_LABELS = {
    "denied",
}
COMPARABLE_DECISION_LABELS = POSITIVE_DECISION_LABELS | NEGATIVE_DECISION_LABELS


def normalize_decision_series(series: pd.Series) -> pd.Series:
    """Normalize decision labels into comparable awarded/denied buckets."""
    normalized = series.astype("string").str.strip().str.lower()
    mapped = pd.Series(pd.NA, index=series.index, dtype="string")
    mapped = mapped.mask(normalized.isin(POSITIVE_DECISION_LABELS), "awarded")
    mapped = mapped.mask(normalized.isin(NEGATIVE_DECISION_LABELS), "denied")
    return mapped
