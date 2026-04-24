#!/usr/bin/env python3
"""Full-model purchasing-power proxy experiment for the eligibility pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faid_models.eligibility.modeling.yasmina_eligibility_model import (  # noqa: E402
    DATA_PATH,
    SELECTED_THRESHOLD_POLICY,
    apply_feature_pruning,
    build_domain_features,
    build_preprocessors,
    compute_probability_metrics,
    evaluate_threshold_policies,
    fit_feature_pruner,
    fit_model_prototype,
    get_model_specs,
    get_positive_class_probabilities,
    get_recommended_class_weight,
    get_recommended_scale_pos_weight,
    load_data,
    make_model_pipeline,
    prepare_target,
    select_feature_frame,
    split_dataset,
    tune_thresholds,
)


PP_FACTOR = {
    2024: 0.7,
    2025: 0.85,
}

FATHER_INCOME_CANDIDATES = (
    "father_income",
    "parsed_father_gross_income",
    "parsed_father_net_income",
)
APPLICATION_TERM_CANDIDATES = (
    "Application Term",
    "application_term",
    "parsed_application_term",
)
METADATA_CANDIDATES = (
    PROJECT_ROOT / "artifacts/yasmina_eligibility_notebook/yasmina_eligibility_metadata.json",
    PROJECT_ROOT / "artifacts/yasmina_eligibility/yasmina_eligibility_metadata.json",
)
MODEL_FALLBACK_ORDER = (
    "XGBoost",
    "HistGradientBoosting",
    "Random Forest",
    "Logistic Regression",
)


def print_section(title: str) -> None:
    """Render a small console section header."""
    print(f"\n{title}")
    print("-" * 80)


def resolve_column_name(dataframe: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    """Resolve a column by exact or case-insensitive match."""
    exact_matches = [column for column in candidates if column in dataframe.columns]
    if exact_matches:
        return exact_matches[0]

    lowered = {str(column).strip().lower(): column for column in dataframe.columns}
    for candidate in candidates:
        match = lowered.get(candidate.strip().lower())
        if match is not None:
            return match

    raise KeyError(
        f"None of the requested columns were found: {', '.join(candidates)}"
    )


def extract_application_year(series: pd.Series) -> pd.Series:
    """Extract the year from the first four characters of the application term."""
    return pd.to_numeric(
        series.astype("string").str.strip().str.slice(0, 4),
        errors="coerce",
    ).astype("Int64")


def resolve_model_name(model_specs: dict[str, Any]) -> str:
    """Prefer the current eligibility model when available, else fall back safely."""
    for metadata_path in METADATA_CANDIDATES:
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected_model = (
            metadata.get("artifacts", {}).get("selected_model_name")
            or metadata.get("selected_model_name")
        )
        if selected_model in model_specs:
            return str(selected_model)

    for model_name in MODEL_FALLBACK_ORDER:
        if model_name in model_specs:
            return model_name

    return next(iter(model_specs))


def summarize_income_series(raw_income: pd.Series, adjusted_income: pd.Series) -> pd.DataFrame:
    """Summarize raw vs adjusted income with consistent numeric stats."""
    rows: list[dict[str, Any]] = []
    for feature_name, series in (
        ("raw_father_income", raw_income),
        ("adjusted_income", adjusted_income),
    ):
        rows.append(
            {
                "feature": feature_name,
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


# Purchasing power experiment start
def build_pp_factor_table() -> pd.DataFrame:
    """Render the purchasing-power proxy mapping as a small table."""
    return pd.DataFrame(
        [{"application_year": year, "pp_factor": factor} for year, factor in sorted(PP_FACTOR.items())]
    )


def build_confusion_matrix_frame(matrix: list[list[int]]) -> pd.DataFrame:
    """Format a confusion matrix for readable console output."""
    return pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )


def extract_feature_importance_tables(
    fitted_model: Any,
    *,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract top feature importances when the fitted estimator exposes them."""
    empty = pd.DataFrame(columns=["rank", "feature", "importance"])

    if not hasattr(fitted_model, "named_steps"):
        return empty, empty

    preprocessor = fitted_model.named_steps.get("preprocessor")
    estimator = fitted_model.named_steps.get("model")
    if preprocessor is None or estimator is None or not hasattr(preprocessor, "get_feature_names_out"):
        return empty, empty

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        return empty, empty

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_, dtype=float)).reshape(-1)
    else:
        return empty, empty

    row_count = min(len(feature_names), len(importances))
    if row_count == 0:
        return empty, empty

    full_table = pd.DataFrame(
        {
            "feature": feature_names[:row_count],
            "importance": importances[:row_count],
        }
    ).sort_values(by=["importance", "feature"], ascending=[False, True], ignore_index=True)
    full_table.insert(0, "rank", full_table.index + 1)
    return full_table.head(top_n).copy(), full_table


def summarize_income_related_importance(
    importance_table: pd.DataFrame,
    *,
    feature_token: str,
    label: str,
) -> dict[str, Any]:
    """Aggregate importance for the income feature family."""
    if importance_table.empty:
        return {
            "label": label,
            "matching_feature_count": 0,
            "importance_sum": 0.0,
            "best_rank": None,
            "top_matching_feature": "not_supported",
        }

    matching = importance_table[
        importance_table["feature"].str.contains(feature_token, regex=False, na=False)
    ].copy()
    if matching.empty:
        return {
            "label": label,
            "matching_feature_count": 0,
            "importance_sum": 0.0,
            "best_rank": None,
            "top_matching_feature": "not_found",
        }

    best_row = matching.sort_values(by=["rank", "importance"], ascending=[True, False]).iloc[0]
    return {
        "label": label,
        "matching_feature_count": int(len(matching)),
        "importance_sum": float(matching["importance"].sum()),
        "best_rank": int(best_row["rank"]),
        "top_matching_feature": str(best_row["feature"]),
    }


def build_segment_performance_table(
    *,
    target_true: pd.Series,
    application_year: pd.Series,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate AUC and F1 by application year on the shared test set."""
    rows: list[dict[str, Any]] = []
    year_labels = application_year.astype("string").fillna("missing_or_malformed")

    for year_value in sorted(year_labels.unique().tolist()):
        segment_mask = year_labels == year_value
        segment_target = target_true.loc[segment_mask]
        segment_probabilities_a = model_a["test_probabilities"].loc[segment_mask]
        segment_probabilities_b = model_b["test_probabilities"].loc[segment_mask]
        segment_predictions_a = model_a["test_predictions"].loc[segment_mask]
        segment_predictions_b = model_b["test_predictions"].loc[segment_mask]

        try:
            auc_a = float(roc_auc_score(segment_target, segment_probabilities_a))
        except Exception:
            auc_a = float("nan")
        try:
            auc_b = float(roc_auc_score(segment_target, segment_probabilities_b))
        except Exception:
            auc_b = float("nan")

        rows.append(
            {
                "application_year": year_value,
                "row_count": int(segment_mask.sum()),
                "auc_model_a": auc_a,
                "f1_model_a": float(f1_score(segment_target, segment_predictions_a, zero_division=0)),
                "auc_model_b": auc_b,
                "f1_model_b": float(f1_score(segment_target, segment_predictions_b, zero_division=0)),
            }
        )

    return pd.DataFrame(rows)


def build_decision_impact_summary(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
) -> dict[str, Any]:
    """Translate prediction deltas into business-style decision terms."""
    predictions_a = model_a["test_predictions"]
    predictions_b = model_b["test_predictions"]
    model_a_positive = int((predictions_a == 1).sum())
    model_b_positive = int((predictions_b == 1).sum())
    additional_positive_decisions = int(((predictions_a == 0) & (predictions_b == 1)).sum())
    lost_positive_decisions = int(((predictions_a == 1) & (predictions_b == 0)).sum())
    net_positive_change = model_b_positive - model_a_positive
    total_cases = int(len(predictions_a))

    return {
        "model_a_positive_predictions": model_a_positive,
        "model_b_positive_predictions": model_b_positive,
        "additional_positive_decisions": additional_positive_decisions,
        "lost_positive_decisions": lost_positive_decisions,
        "net_positive_change": net_positive_change,
        "net_positive_change_pct": (net_positive_change / total_cases) if total_cases else 0.0,
        "total_cases": total_cases,
    }


def build_distribution_sanity_check(
    diagnostics: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Compare how much the raw and adjusted income features actually differ."""
    raw_income = diagnostics["raw_income"]
    adjusted_income = diagnostics["adjusted_income"]
    valid_mask = raw_income.notna() & adjusted_income.notna()

    if int(valid_mask.sum()) > 1:
        correlation = float(raw_income.loc[valid_mask].corr(adjusted_income.loc[valid_mask]))
    else:
        correlation = float("nan")

    raw_mean = float(raw_income.mean())
    adjusted_mean = float(adjusted_income.mean())
    mean_shift_pct = ((adjusted_mean - raw_mean) / raw_mean) * 100.0 if raw_mean else float("nan")
    non_default_factor_rate = (
        1.0 - (diagnostics["default_factor_row_count"] / diagnostics["row_count"])
        if diagnostics["row_count"]
        else 0.0
    )
    active_adjustment_rate = (
        diagnostics["active_adjustment_row_count"] / diagnostics["row_count"]
        if diagnostics["row_count"]
        else 0.0
    )

    sanity_frame = pd.DataFrame(
        [
            {
                "raw_adjusted_correlation": correlation,
                "mean_shift_pct": mean_shift_pct,
                "non_default_factor_rate": non_default_factor_rate,
                "active_adjustment_rate": active_adjustment_rate,
            }
        ]
    )

    if np.isnan(correlation):
        explanation = "The distribution sanity check could not compute a stable correlation."
    else:
        explanation = (
            f"Raw and adjusted income remain highly aligned with correlation {correlation:.4f}. "
            f"The mean shifted by {mean_shift_pct:.2f}%, so the feature changed in scale more than in rank ordering."
        )

    return sanity_frame, explanation


def build_experiment_feature_sets(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build full-feature Model A and Model B frames with one controlled income swap."""
    father_income_column = resolve_column_name(dataframe, FATHER_INCOME_CANDIDATES)
    application_term_column = resolve_column_name(dataframe, APPLICATION_TERM_CANDIDATES)

    base_features, _ = select_feature_frame(dataframe, target_column)
    base_features = build_domain_features(base_features)
    if father_income_column not in base_features.columns:
        raise KeyError(
            f"Resolved father income column '{father_income_column}' is not present in the model feature set."
        )

    father_income = pd.to_numeric(dataframe[father_income_column], errors="coerce")
    application_year = extract_application_year(dataframe[application_term_column])
    year_factor = application_year.map(PP_FACTOR).fillna(1.0).astype(float)
    adjusted_income = father_income * year_factor
    active_adjustment_mask = father_income.notna() & year_factor.ne(1.0)

    model_a_features = base_features.copy()
    model_b_features = base_features.drop(columns=[father_income_column], errors="ignore").copy()
    insert_at = list(base_features.columns).index(father_income_column)
    model_b_features.insert(insert_at, "adjusted_income", adjusted_income)

    year_distribution = (
        application_year.astype("string")
        .fillna("missing_or_malformed")
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("application_year")
        .reset_index(name="row_count")
    )

    diagnostics = {
        "father_income_column": father_income_column,
        "application_term_column": application_term_column,
        "row_count": int(len(dataframe)),
        "missing_father_income_count": int(father_income.isna().sum()),
        "default_factor_row_count": int((year_factor == 1.0).sum()),
        "active_adjustment_row_count": int(active_adjustment_mask.sum()),
        "year_distribution": year_distribution,
        "income_summary": summarize_income_series(father_income, adjusted_income),
        "application_year": application_year,
        "raw_income": father_income,
        "adjusted_income": adjusted_income,
    }
    return model_a_features, model_b_features, diagnostics


def align_feature_sets_for_modeling(
    *,
    model_a_features: pd.DataFrame,
    model_b_features: pd.DataFrame,
    target: pd.Series,
    father_income_column: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Apply one shared split and one shared pruning policy to both experiments."""
    (
        features_train_a,
        features_valid_a,
        features_test_a,
        target_train,
        target_valid,
        target_test,
    ) = split_dataset(model_a_features, target)

    features_train_b = model_b_features.loc[features_train_a.index].copy()
    features_valid_b = model_b_features.loc[features_valid_a.index].copy()
    features_test_b = model_b_features.loc[features_test_a.index].copy()

    columns_to_drop, _ = fit_feature_pruner(features_train_a)
    columns_to_drop = [column for column in columns_to_drop if column != father_income_column]

    features_train_a = apply_feature_pruning(features_train_a, columns_to_drop)
    retained_columns_a = features_train_a.columns.tolist()
    retained_columns_b = [
        "adjusted_income" if column == father_income_column else column
        for column in retained_columns_a
    ]

    features_valid_a = apply_feature_pruning(
        features_valid_a,
        columns_to_drop,
        retained_columns=retained_columns_a,
    )
    features_test_a = apply_feature_pruning(
        features_test_a,
        columns_to_drop,
        retained_columns=retained_columns_a,
    )
    features_train_b = apply_feature_pruning(
        features_train_b,
        columns_to_drop,
        retained_columns=retained_columns_b,
    )
    features_valid_b = apply_feature_pruning(
        features_valid_b,
        columns_to_drop,
        retained_columns=retained_columns_b,
    )
    features_test_b = apply_feature_pruning(
        features_test_b,
        columns_to_drop,
        retained_columns=retained_columns_b,
    )

    return (
        features_train_a,
        features_valid_a,
        features_test_a,
        features_train_b,
        features_valid_b,
        features_test_b,
        target_train,
        target_valid,
        target_test,
    )


def run_full_feature_model(
    *,
    variant_name: str,
    model_name: str,
    model_spec: Any,
    features_train: pd.DataFrame,
    features_valid: pd.DataFrame,
    features_test: pd.DataFrame,
    target_train: pd.Series,
    target_valid: pd.Series,
    target_test: pd.Series,
) -> dict[str, Any]:
    """Fit and evaluate one full-feature model with shared pipeline logic."""
    preprocessors, _, _ = build_preprocessors(features_train)
    pipeline = make_model_pipeline(preprocessors, model_spec)
    fitted_model = fit_model_prototype(pipeline, features_train, target_train)

    validation_probabilities = get_positive_class_probabilities(fitted_model, features_valid)
    _, validation_threshold_results = tune_thresholds(target_valid, validation_probabilities)
    selected_validation_metrics = validation_threshold_results[SELECTED_THRESHOLD_POLICY]

    test_probabilities = get_positive_class_probabilities(fitted_model, features_test)
    test_probability_metrics = compute_probability_metrics(target_test, test_probabilities)
    test_threshold_results = evaluate_threshold_policies(
        target_test,
        test_probabilities,
        {SELECTED_THRESHOLD_POLICY: selected_validation_metrics.threshold},
    )
    selected_test_metrics = test_threshold_results[SELECTED_THRESHOLD_POLICY]
    test_predictions = pd.Series(
        (test_probabilities >= float(selected_validation_metrics.threshold)).astype(int),
        index=features_test.index,
        name=f"{variant_name.lower().replace(' ', '_')}_prediction",
    )
    test_probabilities_series = pd.Series(
        np.asarray(test_probabilities, dtype=float),
        index=features_test.index,
        name=f"{variant_name.lower().replace(' ', '_')}_probability",
    )
    top_feature_importance, full_feature_importance = extract_feature_importance_tables(
        fitted_model,
        top_n=10,
    )

    return {
        "variant_name": variant_name,
        "model_name": model_name,
        "threshold_policy": SELECTED_THRESHOLD_POLICY,
        "selected_threshold": float(selected_validation_metrics.threshold),
        "auc": float(test_probability_metrics["roc_auc"]),
        "f1": float(selected_test_metrics.f1),
        "feature_count": int(features_train.shape[1]),
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities_series,
        "confusion_matrix": selected_test_metrics.confusion_matrix,
        "top_feature_importance": top_feature_importance,
        "full_feature_importance": full_feature_importance,
    }
# Purchasing power experiment end


def build_prediction_impact_summary(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare final test-set predictions between both model variants."""
    predictions_a = model_a["test_predictions"]
    predictions_b = model_b["test_predictions"]
    prediction_diff_mask = predictions_a.ne(predictions_b)
    changed_count = int(prediction_diff_mask.sum())
    total_count = int(len(predictions_a))

    return {
        "changed_case_count": changed_count,
        "changed_case_pct": (changed_count / total_count) if total_count else 0.0,
        "negative_to_positive_count": int(((predictions_a == 0) & (predictions_b == 1)).sum()),
        "positive_to_negative_count": int(((predictions_a == 1) & (predictions_b == 0)).sum()),
        "total_case_count": total_count,
    }


def build_interpretation(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    prediction_impact: dict[str, Any],
    decision_impact: dict[str, Any],
    diagnostics: dict[str, Any],
    income_importance_summary: pd.DataFrame,
) -> str:
    """Write a short plain-English interpretation."""
    auc_delta = model_b["auc"] - model_a["auc"]
    f1_delta = model_b["f1"] - model_a["f1"]
    changed_pct = float(prediction_impact["changed_case_pct"])
    default_factor_share = (
        diagnostics["default_factor_row_count"] / diagnostics["row_count"]
        if diagnostics["row_count"]
        else 0.0
    )
    income_importance_delta = 0.0
    if len(income_importance_summary) >= 2:
        income_importance_delta = float(
            income_importance_summary.iloc[1]["importance_sum"]
            - income_importance_summary.iloc[0]["importance_sum"]
        )

    if auc_delta > 0 and f1_delta > 0:
        verdict = "The purchasing-power adjustment helped on both AUC and F1."
    elif auc_delta < 0 and f1_delta < 0:
        verdict = "The purchasing-power adjustment hurt both AUC and F1."
    else:
        verdict = "The purchasing-power adjustment gave mixed results."

    if max(abs(auc_delta), abs(f1_delta)) < 0.005:
        magnitude = "The performance difference is marginal rather than decisive."
    elif max(abs(auc_delta), abs(f1_delta)) < 0.015:
        magnitude = "The performance difference is small but noticeable."
    else:
        magnitude = "The performance difference is meaningful."

    if changed_pct < 0.02:
        prediction_shift = "Very few final test predictions changed between Model A and Model B."
    elif changed_pct < 0.10:
        prediction_shift = "A modest share of final test predictions changed between the two models."
    else:
        prediction_shift = "A large share of final test predictions changed between the two models."

    if decision_impact["net_positive_change"] > 0:
        decision_note = (
            f"If deployed, Model B would approve {decision_impact['net_positive_change']} more students on the test set."
        )
    elif decision_impact["net_positive_change"] < 0:
        decision_note = (
            f"If deployed, Model B would approve {abs(decision_impact['net_positive_change'])} fewer students on the test set."
        )
    else:
        decision_note = "If deployed, Model B would approve the same number of students on the test set."

    if default_factor_share >= 0.50:
        business_implication = (
            "Most rows still used the default factor 1.0, so this proxy currently has limited reach. "
            "That means the experiment is directionally useful for fairness-across-years and economic-realism thinking, "
            "but not yet strong enough to justify a production change on its own."
        )
    elif auc_delta >= 0 or f1_delta >= 0:
        business_implication = (
            "Because the proxy injects some year sensitivity into income, it may help the model treat older applications "
            "more economically realistically. The effect here is still small, so it is better viewed as an exploratory fairness-across-years improvement than a finished policy change."
        )
    else:
        business_implication = (
            "This proxy does not currently improve the business outcome enough to justify replacing the raw income field. "
            "It is still a useful thesis experiment because it tests whether year-aware income scaling can improve fairness across application years."
        )

    if income_importance_delta > 0.0:
        importance_note = "Income-related importance increased after the adjustment."
    elif income_importance_delta < 0.0:
        importance_note = "Income-related importance decreased after the adjustment."
    else:
        importance_note = "Income-related importance stayed effectively flat."

    return (
        f"{verdict} Model B changed AUC by {auc_delta:.4f} and F1 by {f1_delta:.4f} versus Model A. "
        f"{magnitude} {prediction_shift} {importance_note} {decision_note} {business_implication}"
    )


def build_final_recommendation(
    *,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    prediction_impact: dict[str, Any],
    diagnostics: dict[str, Any],
) -> tuple[str, str]:
    """Produce a simple final recommendation based on effect size and coverage."""
    auc_delta = model_b["auc"] - model_a["auc"]
    f1_delta = model_b["f1"] - model_a["f1"]
    max_metric_delta = max(abs(auc_delta), abs(f1_delta))
    changed_case_pct = float(prediction_impact["changed_case_pct"])
    active_adjustment_rate = (
        diagnostics["active_adjustment_row_count"] / diagnostics["row_count"]
        if diagnostics["row_count"]
        else 0.0
    )

    if auc_delta > 0.005 and f1_delta > 0.005 and changed_case_pct >= 0.02:
        recommendation = "replace"
        rationale = (
            "Model B shows a meaningful and consistent lift with enough prediction movement to matter operationally."
        )
    elif max_metric_delta < 0.005 and changed_case_pct < 0.02 and active_adjustment_rate < 0.10:
        recommendation = "keep raw"
        rationale = (
            "The measured effect is marginal, very few decisions change, and the adjustment only touches a limited share of rows."
        )
    else:
        recommendation = "explore further"
        rationale = (
            "The idea is directionally plausible, but the current heuristic does not yet provide a clear enough operational win."
        )

    return recommendation, rationale


def main() -> int:
    dataframe = load_data(DATA_PATH)
    training_dataframe, target, target_metadata = prepare_target(dataframe)

    # Purchasing power experiment start
    model_a_features, model_b_features, diagnostics = build_experiment_feature_sets(
        training_dataframe,
        target_column=target_metadata["target_column"],
    )
    (
        features_train_a,
        features_valid_a,
        features_test_a,
        features_train_b,
        features_valid_b,
        features_test_b,
        target_train,
        target_valid,
        target_test,
    ) = align_feature_sets_for_modeling(
        model_a_features=model_a_features,
        model_b_features=model_b_features,
        target=target,
        father_income_column=diagnostics["father_income_column"],
    )
    # Purchasing power experiment end

    if diagnostics["father_income_column"] not in features_train_a.columns:
        raise KeyError(
            f"Model A does not contain the expected raw income column: {diagnostics['father_income_column']}"
        )
    if "adjusted_income" not in features_train_b.columns:
        raise KeyError("Model B does not contain the expected 'adjusted_income' feature.")
    if diagnostics["father_income_column"] in features_train_b.columns:
        raise ValueError("Model B still contains the raw father income column after replacement.")
    if features_train_a.shape[1] != features_train_b.shape[1]:
        raise ValueError("Model A and Model B must have the same feature count.")

    class_weight = get_recommended_class_weight(target_train)
    scale_pos_weight = get_recommended_scale_pos_weight(target_train)
    model_specs = get_model_specs(
        class_weight=class_weight,
        scale_pos_weight=scale_pos_weight,
    )
    model_name = resolve_model_name(model_specs)
    model_spec = model_specs[model_name]

    model_a = run_full_feature_model(
        variant_name="Model A",
        model_name=model_name,
        model_spec=model_spec,
        features_train=features_train_a,
        features_valid=features_valid_a,
        features_test=features_test_a,
        target_train=target_train,
        target_valid=target_valid,
        target_test=target_test,
    )
    model_b = run_full_feature_model(
        variant_name="Model B",
        model_name=model_name,
        model_spec=model_spec,
        features_train=features_train_b,
        features_valid=features_valid_b,
        features_test=features_test_b,
        target_train=target_train,
        target_valid=target_valid,
        target_test=target_test,
    )
    prediction_impact = build_prediction_impact_summary(model_a, model_b)
    decision_impact = build_decision_impact_summary(model_a, model_b)
    segment_performance = build_segment_performance_table(
        target_true=target_test,
        application_year=diagnostics["application_year"].loc[features_test_a.index],
        model_a=model_a,
        model_b=model_b,
    )
    distribution_sanity_check, distribution_explanation = build_distribution_sanity_check(diagnostics)
    income_importance_summary = pd.DataFrame(
        [
            summarize_income_related_importance(
                model_a["full_feature_importance"],
                feature_token=diagnostics["father_income_column"],
                label="Model A raw income",
            ),
            summarize_income_related_importance(
                model_b["full_feature_importance"],
                feature_token="adjusted_income",
                label="Model B adjusted income",
            ),
        ]
    )
    recommendation, recommendation_rationale = build_final_recommendation(
        model_a=model_a,
        model_b=model_b,
        prediction_impact=prediction_impact,
        diagnostics=diagnostics,
    )

    comparison_table = pd.DataFrame(
        [
            {
                "model_version": model_a["variant_name"],
                "income_feature": diagnostics["father_income_column"],
                "estimator": model_a["model_name"],
                "feature_count": model_a["feature_count"],
                "threshold_policy": model_a["threshold_policy"],
                "selected_threshold": model_a["selected_threshold"],
                "auc": model_a["auc"],
                "f1": model_a["f1"],
            },
            {
                "model_version": model_b["variant_name"],
                "income_feature": "adjusted_income",
                "estimator": model_b["model_name"],
                "feature_count": model_b["feature_count"],
                "threshold_policy": model_b["threshold_policy"],
                "selected_threshold": model_b["selected_threshold"],
                "auc": model_b["auc"],
                "f1": model_b["f1"],
            },
        ]
    )

    print_section("Purchasing Power Experiment")
    print(
        "PP_FACTOR is a simple proxy for inflation / purchasing power, not a formal CPI model.\n"
        "2024 receives a lower factor than 2025 because an earlier nominal income is assumed to overstate real purchasing power more strongly in a high-inflation / currency-collapse context.\n"
        "These values are heuristic and directional: they are meant to test whether year-aware income scaling helps, not to claim a precise macroeconomic adjustment.\n"
        f"Resolved father income column: {diagnostics['father_income_column']}\n"
        f"Resolved application term column: {diagnostics['application_term_column']}\n"
        f"Missing father income count: {diagnostics['missing_father_income_count']}\n"
        f"Rows using default factor 1.0: {diagnostics['default_factor_row_count']}\n"
        f"Shared estimator: {model_name}"
    )

    print_section("Purchasing Power Mapping")
    print(build_pp_factor_table().to_string(index=False))

    print_section("Extracted Year Distribution")
    print(diagnostics["year_distribution"].to_string(index=False))

    print_section("Income Summary")
    print(diagnostics["income_summary"].round(4).to_string(index=False))

    print_section("Model Comparison")
    print(comparison_table.round(4).to_string(index=False))

    print_section("Segment-Level Analysis By Application Year")
    print(segment_performance.round(4).to_string(index=False))

    print_section("Prediction Impact Analysis")
    prediction_impact_frame = pd.DataFrame(
        [
            {
                "changed_case_count": prediction_impact["changed_case_count"],
                "changed_case_pct": prediction_impact["changed_case_pct"],
                "negative_to_positive_count": prediction_impact["negative_to_positive_count"],
                "positive_to_negative_count": prediction_impact["positive_to_negative_count"],
            }
        ]
    )
    print(prediction_impact_frame.round(4).to_string(index=False))

    print_section("Decision Impact Framing")
    decision_impact_frame = pd.DataFrame([decision_impact])
    print(decision_impact_frame.round(4).to_string(index=False))
    if decision_impact["net_positive_change"] > 0:
        print(
            f"If deployed, Model B would approve {decision_impact['net_positive_change']} more students "
            f"({decision_impact['net_positive_change_pct']:.2%} of the test set)."
        )
    elif decision_impact["net_positive_change"] < 0:
        print(
            f"If deployed, Model B would approve {abs(decision_impact['net_positive_change'])} fewer students "
            f"({abs(decision_impact['net_positive_change_pct']):.2%} of the test set)."
        )
    else:
        print("If deployed, Model B would approve the same number of students as Model A on the test set.")

    print_section("Model A Confusion Matrix")
    print(build_confusion_matrix_frame(model_a["confusion_matrix"]).to_string())

    print_section("Model B Confusion Matrix")
    print(build_confusion_matrix_frame(model_b["confusion_matrix"]).to_string())

    print_section("Top 10 Feature Importance - Model A")
    if model_a["top_feature_importance"].empty:
        print("Feature importance is not supported by the selected estimator.")
    else:
        print(model_a["top_feature_importance"].round(6).to_string(index=False))

    print_section("Top 10 Feature Importance - Model B")
    if model_b["top_feature_importance"].empty:
        print("Feature importance is not supported by the selected estimator.")
    else:
        print(model_b["top_feature_importance"].round(6).to_string(index=False))

    print_section("Income Importance Comparison")
    print(income_importance_summary.round(6).to_string(index=False))
    income_importance_delta = 0.0
    if len(income_importance_summary) >= 2:
        income_importance_delta = float(
            income_importance_summary.iloc[1]["importance_sum"]
            - income_importance_summary.iloc[0]["importance_sum"]
        )
    if income_importance_delta > 0:
        print("Income-related importance increased after the purchasing-power adjustment.")
    elif income_importance_delta < 0:
        print("Income-related importance decreased after the purchasing-power adjustment.")
    else:
        print("Income-related importance stayed effectively flat after the purchasing-power adjustment.")

    print_section("Distribution Sanity Check")
    print(distribution_sanity_check.round(4).to_string(index=False))
    print(distribution_explanation)

    print_section("Interpretation")
    print(
        build_interpretation(
            model_a,
            model_b,
            prediction_impact,
            decision_impact,
            diagnostics,
            income_importance_summary,
        )
    )

    print_section("Recommendation")
    print(f"Recommendation: {recommendation}")
    print(recommendation_rationale)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
