from typing import Any

import numpy as np
import pandas as pd

from utils.model_artifact import resolve_explainability_estimator


def clean_feature_name(feature_name: str) -> str:
    """Make transformed pipeline feature names easier to read in charts."""
    cleaned = str(feature_name)
    for token in ("num__", "cat__", "remainder__", "imputer__", "scaler__"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.replace("encoder__", "")
    cleaned = cleaned.replace("missingindicator_", "is_missing:")
    cleaned = cleaned.replace("__missing__", "[missing]")
    return cleaned


def get_transformed_feature_names(preprocessor: Any) -> list[str]:
    """Recover readable feature names after preprocessing when supported."""
    if not hasattr(preprocessor, "get_feature_names_out"):
        raise AttributeError("The model preprocessor does not expose transformed feature names.")

    raw_feature_names = preprocessor.get_feature_names_out()
    return [clean_feature_name(name) for name in raw_feature_names]


def _resolve_model_components(model: Any) -> tuple[Any | None, Any]:
    """Return the pipeline preprocessor and final estimator when available."""
    pipeline = resolve_explainability_estimator(model)
    if pipeline is None:
        raise AttributeError("The loaded model artifact does not contain an explainability estimator.")

    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor")
    estimator = getattr(pipeline, "named_steps", {}).get("model", pipeline)
    return preprocessor, estimator


def _resolve_feature_names(preprocessor: Any, estimator: Any, value_count: int) -> list[str]:
    """Resolve readable feature names for the fitted estimator output."""
    feature_names: list[str] | None = None
    if preprocessor is not None:
        try:
            feature_names = get_transformed_feature_names(preprocessor)
        except Exception:
            feature_names = None

    if feature_names is None and hasattr(estimator, "feature_names_in_"):
        feature_names = [str(name) for name in estimator.feature_names_in_]

    if feature_names is None or len(feature_names) != value_count:
        feature_names = [f"feature_{index + 1}" for index in range(value_count)]

    return feature_names


def extract_top_feature_importances(
    model: Any,
    *,
    top_n: int = 15,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Extract top feature importances or coefficient magnitudes from a fitted model."""
    preprocessor, estimator = _resolve_model_components(model)

    metadata = {
        "value_label": "Importance",
        "caption": "Bars show the model's top feature importance scores.",
    }

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float).ravel()
        signed_values = values.copy()
        metadata["caption"] = (
            "Bars show the model's top feature importance scores from the fitted estimator."
        )
    elif hasattr(estimator, "coef_"):
        signed_values = np.asarray(estimator.coef_, dtype=float).ravel()
        values = np.abs(signed_values)
        metadata["value_label"] = "Absolute Coefficient"
        metadata["caption"] = (
            "Bars show the largest coefficient magnitudes; color indicates positive or negative direction."
        )
    else:
        raise AttributeError(
            "The loaded model does not expose feature_importances_ or coef_."
        )

    feature_names = _resolve_feature_names(preprocessor, estimator, len(values))

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
            "signed_value": signed_values,
        }
    )
    importance_df["direction"] = np.where(
        importance_df["signed_value"] < 0,
        "Negative",
        "Positive",
    )
    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
        ignore_index=True,
    ).head(top_n)

    return importance_df, metadata


def extract_top_feature_drivers(
    model: Any,
    *,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Extract the strongest positive and negative coefficient drivers."""
    preprocessor, estimator = _resolve_model_components(model)

    if not hasattr(estimator, "coef_"):
        raise AttributeError(
            "Top positive and negative drivers require a model with signed coefficients."
        )

    coefficients = np.asarray(estimator.coef_, dtype=float).ravel()
    feature_names = _resolve_feature_names(preprocessor, estimator, len(coefficients))

    driver_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    positive_drivers = (
        driver_df[driver_df["coefficient"] > 0]
        .sort_values(by="coefficient", ascending=False, ignore_index=True)
        .head(top_n)
    )
    negative_drivers = (
        driver_df[driver_df["coefficient"] < 0]
        .sort_values(by="coefficient", ascending=True, ignore_index=True)
        .head(top_n)
    )

    if positive_drivers.empty and negative_drivers.empty:
        raise ValueError("No directional coefficients are available for driver analysis.")

    metadata = {
        "value_label": "Coefficient",
        "caption": (
            "Positive drivers increase the model's tendency toward an awarded prediction, "
            "while negative drivers decrease it."
        ),
    }
    return positive_drivers, negative_drivers, metadata
