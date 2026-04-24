from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.model_artifact import resolve_prediction_estimator, resolve_selected_threshold


TARGET_COLUMN = "decision"
LEAKAGE_COLUMNS = {
    "decision",
    "bin_status",
    "need_pct",
    "need_comment",
    "merit_pct",
    "merit_hist_pct",
    "merit_comment",
    "over_and_above_decision",
    "over_and_above_percentage_awarded",
    "has_over_and_above",
    "raw_decision",
    "raw_bin_status",
    "parsed_decision",
    "parsed_bin_status",
    "parsed_need_pct",
    "parsed_need_comment",
    "parsed_merit_pct",
    "parsed_merit_hist_pct",
    "parsed_merit_comment",
    "parsed_over_and_above_decision",
    "parsed_over_and_above_amount_awarded",
    "parsed_over_and_above_percentage_awarded",
    "inferred_has_over_and_above",
}


def prepare_scoring_features(
    dataframe: pd.DataFrame,
    *,
    model: Any | None = None,
) -> pd.DataFrame:
    """Prepare dataset features for inference using the training logic when available."""
    estimator = resolve_prediction_estimator(model) if model is not None else None
    expected_columns = getattr(estimator, "feature_names_in_", None)
    if expected_columns is not None:
        return dataframe.reindex(columns=list(expected_columns)).copy()

    try:
        from faid_models.eligibility.modeling.yasmina_eligibility_model import (
            LEAKAGE_COLUMNS as MODEL_LEAKAGE_COLUMNS,
            TARGET_COLUMN as MODEL_TARGET_COLUMN,
            build_domain_features,
            select_feature_frame,
        )
    except Exception:
        try:
            from yasmina_eligibility_model import (
                LEAKAGE_COLUMNS as MODEL_LEAKAGE_COLUMNS,
                TARGET_COLUMN as MODEL_TARGET_COLUMN,
                build_domain_features,
                select_feature_frame,
            )
        except Exception:
            features = dataframe.drop(columns=sorted(LEAKAGE_COLUMNS), errors="ignore").copy()
            return features

    try:
        features, _ = select_feature_frame(
            dataframe,
            target_column=MODEL_TARGET_COLUMN,
            leakage_columns=MODEL_LEAKAGE_COLUMNS,
        )
    except TypeError:
        features, _ = select_feature_frame(dataframe, target_column=MODEL_TARGET_COLUMN)
    except Exception:
        features = dataframe.drop(columns=sorted(LEAKAGE_COLUMNS), errors="ignore").copy()
        return features

    return build_domain_features(features)


def get_positive_class_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Extract positive-class probabilities from a compatible binary classifier."""
    estimator = resolve_prediction_estimator(model)
    if estimator is None:
        raise AttributeError("The loaded model artifact does not contain a scoring estimator.")

    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
        if probabilities.ndim == 1:
            return probabilities

        if probabilities.shape[1] == 1:
            return probabilities[:, 0]

        positive_index = 1
        model_classes = getattr(estimator, "classes_", None)
        if model_classes is not None:
            classes_list = list(model_classes)
            normalized_classes = [str(value).strip().lower() for value in classes_list]

            if "awarded" in normalized_classes:
                positive_index = normalized_classes.index("awarded")
            elif 1 in classes_list:
                positive_index = classes_list.index(1)

        return probabilities[:, positive_index]

    if hasattr(estimator, "decision_function"):
        decision_scores = np.asarray(estimator.decision_function(features), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision_scores))

    raise AttributeError("The loaded model does not expose predict_proba or decision_function.")


def add_prediction_probabilities(dataframe: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Return a copy of the dataset with predicted award probabilities attached."""
    estimator = resolve_prediction_estimator(model)
    if estimator is None:
        raise AttributeError("The loaded model artifact does not contain a scoring estimator.")

    features = prepare_scoring_features(dataframe, model=model)
    expected_columns = getattr(estimator, "feature_names_in_", None)
    if expected_columns is not None:
        features = features.reindex(columns=list(expected_columns))

    probabilities = get_positive_class_probabilities(model, features)
    selected_threshold = resolve_selected_threshold(model, default=0.50)
    scored_df = dataframe.copy()
    scored_df["predicted_award_probability"] = probabilities
    scored_df["predicted_award_percentage"] = probabilities * 100.0
    scored_df["predicted_award_label"] = np.where(
        probabilities >= selected_threshold,
        "Awarded",
        "Denied",
    )
    scored_df["prediction_threshold_used"] = selected_threshold
    return scored_df


def build_roc_curve_data(
    dataframe: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    probability_column: str = "predicted_award_probability",
    positive_label: str = "awarded",
) -> tuple[pd.DataFrame, float]:
    """Build ROC curve coordinates and AUC from labeled rows."""
    if target_column not in dataframe.columns:
        raise KeyError(f"Target column not found: {target_column}")

    if probability_column not in dataframe.columns:
        raise KeyError(f"Probability column not found: {probability_column}")

    normalized_target = dataframe[target_column].astype("string").str.strip().str.lower()
    labeled_mask = normalized_target.notna() & normalized_target.ne("")
    labeled_df = dataframe.loc[labeled_mask].copy()

    if labeled_df.empty:
        raise ValueError("No labeled rows are available to compute the ROC curve.")

    probabilities = pd.to_numeric(labeled_df[probability_column], errors="coerce")
    valid_mask = probabilities.notna()
    labeled_df = labeled_df.loc[valid_mask].copy()
    probabilities = probabilities.loc[valid_mask]

    binary_target = (
        labeled_df[target_column].astype("string").str.strip().str.lower()
        == positive_label.strip().lower()
    ).astype(int)

    if binary_target.nunique() < 2:
        raise ValueError("ROC curve requires both positive and negative labeled examples.")

    y_true = binary_target.to_numpy(dtype=int)
    y_score = probabilities.to_numpy(dtype=float)

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    distinct_threshold_indexes = np.where(np.diff(y_score) != 0)[0]
    threshold_indexes = np.r_[distinct_threshold_indexes, len(y_true) - 1]

    true_positives = np.cumsum(y_true)[threshold_indexes]
    false_positives = np.cumsum(1 - y_true)[threshold_indexes]
    positive_total = int(y_true.sum())
    negative_total = int(len(y_true) - positive_total)

    true_positive_rate = np.r_[0.0, true_positives / positive_total, 1.0]
    false_positive_rate = np.r_[0.0, false_positives / negative_total, 1.0]
    thresholds = np.r_[1.0, y_score[threshold_indexes], 0.0]

    roc_auc = float(
        np.sum(
            (false_positive_rate[1:] - false_positive_rate[:-1])
            * (true_positive_rate[1:] + true_positive_rate[:-1])
            / 2.0
        )
    )
    roc_df = pd.DataFrame(
        {
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "threshold": thresholds,
        }
    )
    return roc_df, roc_auc


def build_confusion_matrix_data(
    dataframe: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    probability_column: str = "predicted_award_probability",
    positive_label: str = "awarded",
    threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build confusion-matrix cell counts from labeled rows and predicted probabilities."""
    if target_column not in dataframe.columns:
        raise KeyError(f"Target column not found: {target_column}")

    if probability_column not in dataframe.columns:
        raise KeyError(f"Probability column not found: {probability_column}")

    normalized_target = dataframe[target_column].astype("string").str.strip().str.lower()
    labeled_mask = normalized_target.notna() & normalized_target.ne("")
    labeled_df = dataframe.loc[labeled_mask].copy()

    if labeled_df.empty:
        raise ValueError("No labeled rows are available to compute the confusion matrix.")

    probabilities = pd.to_numeric(labeled_df[probability_column], errors="coerce")
    valid_mask = probabilities.notna()
    labeled_df = labeled_df.loc[valid_mask].copy()
    probabilities = probabilities.loc[valid_mask]

    binary_target = (
        labeled_df[target_column].astype("string").str.strip().str.lower()
        == positive_label.strip().lower()
    ).astype(int)

    if binary_target.nunique() < 2:
        raise ValueError(
            "Confusion matrix requires both positive and negative labeled examples."
        )

    resolved_threshold = threshold
    if resolved_threshold is None and "prediction_threshold_used" in labeled_df.columns:
        threshold_series = pd.to_numeric(
            labeled_df["prediction_threshold_used"],
            errors="coerce",
        ).dropna()
        if not threshold_series.empty:
            resolved_threshold = float(threshold_series.iloc[0])
    if resolved_threshold is None:
        resolved_threshold = 0.50

    predicted_positive = (probabilities >= float(resolved_threshold)).astype(int)

    true_negative = int(((binary_target == 0) & (predicted_positive == 0)).sum())
    false_positive = int(((binary_target == 0) & (predicted_positive == 1)).sum())
    false_negative = int(((binary_target == 1) & (predicted_positive == 0)).sum())
    true_positive = int(((binary_target == 1) & (predicted_positive == 1)).sum())

    matrix_df = pd.DataFrame(
        [
            {
                "actual_label": "Denied",
                "predicted_label": "Denied",
                "count": true_negative,
                "outcome": "True Negative",
            },
            {
                "actual_label": "Denied",
                "predicted_label": "Awarded",
                "count": false_positive,
                "outcome": "False Positive",
            },
            {
                "actual_label": "Awarded",
                "predicted_label": "Denied",
                "count": false_negative,
                "outcome": "False Negative",
            },
            {
                "actual_label": "Awarded",
                "predicted_label": "Awarded",
                "count": true_positive,
                "outcome": "True Positive",
            },
        ]
    )
    summary = {
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }
    return matrix_df, summary
