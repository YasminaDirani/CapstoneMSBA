from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.chart_components import (
    build_grouped_feature_importance,
    make_confusion_matrix_chart,
    make_feature_importance_chart,
    make_grouped_importance_chart,
    render_chart_card,
    render_table_card,
)
from utils.decision_support import STATUS_CAUTION, STATUS_NOT_PRODUCTION, STATUS_RELIABLE
from utils.source_paths import DEFAULT_POC_EXPERIMENT_PATH, DEFAULT_POC_NOTEBOOK_PATH, SOURCE_ROOT
from utils.ui import (
    ACCENT_COLOR,
    CHART_NEUTRALS,
    DANGER_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    apply_design_system,
    render_decision_journey,
    render_insight_action_panel,
    render_kpi_row,
    render_page_header,
    render_takeaway_box,
)


MODEL_COLORS = {"Model A": PRIMARY_COLOR, "Model B": SECONDARY_COLOR}
SCENARIO_COLORS = {
    "legacy_original": "#7570b3",
    "conservative": "#66a61e",
    "baseline": "#1b9e77",
    "aggressive": "#e7298a",
}
PP_SCENARIO_META = {
    "conservative": {"annual_decay": 0.95, "label": "Lower adjustment"},
    "baseline": {"annual_decay": 0.90, "label": "Baseline adjustment"},
    "aggressive": {"annual_decay": 0.85, "label": "Higher adjustment"},
    "legacy_original": {"annual_decay": None, "label": "Original notebook heuristic"},
}
PRIMARY_SCENARIO_NAME = "baseline"


def safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


@st.cache_resource(show_spinner=False)
def load_poc_module() -> Any:
    module_path = Path(DEFAULT_POC_EXPERIMENT_PATH)
    if not module_path.exists():
        return None

    source_root = Path(SOURCE_ROOT)
    if source_root.exists():
        source_root_string = str(source_root)
        if source_root_string not in sys.path:
            sys.path.insert(0, source_root_string)

    module_name = "yasmina_purchasing_power_experiment_page"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner=False)
def load_eligibility_modeling_module() -> Any:
    module = load_poc_module()
    if module is None:
        return None
    return importlib.import_module("faid_models.eligibility.modeling.yasmina_eligibility_model")


def extract_confusion_counts(matrix: list[list[int]]) -> dict[str, int]:
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    return {
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
    }


def compute_policy_weighted_error(matrix: list[list[int]]) -> float:
    eligibility_module = load_eligibility_modeling_module()
    counts = extract_confusion_counts(matrix)
    if eligibility_module is None:
        false_positive_cost = 1.0
        false_negative_cost = 3.0
    else:
        weights = getattr(eligibility_module, "FAID_POLICY_PRIORITY_WEIGHTS", {})
        false_positive_cost = float(weights.get("cost_false_positive_review", 1.0))
        false_negative_cost = float(weights.get("cost_false_negative_missed_need", 3.0))

    return float(
        false_negative_cost * counts["false_negatives"]
        + false_positive_cost * counts["false_positives"]
    )


def build_pp_scenarios(application_year: pd.Series, legacy_map: dict[int, float]) -> tuple[list[int], int, dict[str, dict[int, float]]]:
    observed_years = sorted(
        int(year)
        for year in pd.Series(application_year).dropna().astype(int).unique().tolist()
    )
    reference_year = max(observed_years)
    scenarios = {
        name: {
            year: round(float(meta["annual_decay"]) ** (reference_year - year), 4)
            for year in observed_years
        }
        for name, meta in PP_SCENARIO_META.items()
        if meta["annual_decay"] is not None
    }
    scenarios["legacy_original"] = {int(year): float(value) for year, value in legacy_map.items()}
    return observed_years, reference_year, scenarios


def build_factor_scenario_table(
    observed_years: list[int],
    reference_year: int,
    scenarios: dict[str, dict[int, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_name, meta in PP_SCENARIO_META.items():
        factor_map = scenarios.get(scenario_name, {})
        row: dict[str, Any] = {
            "scenario": scenario_name,
            "scenario_label": meta["label"],
            "reference_year": reference_year,
            "adjusted_year_count": int(
                sum(1 for year in observed_years if factor_map.get(year, 1.0) != 1.0)
            ),
        }
        for year in observed_years:
            row[f"factor_{year}"] = float(factor_map.get(year, 1.0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_income_variant_feature_sets(
    module: Any,
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    factor_map: dict[int, float],
    income_transform: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    father_income_column = module.resolve_column_name(dataframe, module.FATHER_INCOME_CANDIDATES)
    application_term_column = module.resolve_column_name(
        dataframe,
        module.APPLICATION_TERM_CANDIDATES,
    )

    base_features, _ = module.select_feature_frame(dataframe, target_column)
    base_features = module.build_domain_features(base_features)
    father_income = pd.to_numeric(dataframe[father_income_column], errors="coerce")
    application_year = module.extract_application_year(dataframe[application_term_column])
    year_factor = application_year.map(factor_map).fillna(1.0).astype(float)
    adjusted_income = father_income * year_factor

    raw_feature_name = father_income_column
    adjusted_feature_name = "adjusted_income"
    model_a_features = base_features.copy()
    model_b_features = base_features.drop(columns=[father_income_column], errors="ignore").copy()
    insert_at = list(base_features.columns).index(father_income_column)

    if income_transform == "log1p":
        raw_feature_name = "log_raw_income"
        adjusted_feature_name = "log_adjusted_income"
        model_a_features = base_features.drop(columns=[father_income_column], errors="ignore").copy()
        model_b_features = base_features.drop(columns=[father_income_column], errors="ignore").copy()
        model_a_features.insert(
            insert_at,
            raw_feature_name,
            np.log1p(father_income.clip(lower=0)),
        )
        model_b_features.insert(
            insert_at,
            adjusted_feature_name,
            np.log1p(adjusted_income.clip(lower=0)),
        )
    else:
        model_b_features.insert(insert_at, adjusted_feature_name, adjusted_income)

    coverage_source = pd.DataFrame(
        {
            "application_year": application_year.astype("Int64"),
            "has_income": father_income.notna().astype(int),
            "actively_adjusted": (father_income.notna() & year_factor.ne(1.0)).astype(int),
        }
    )
    coverage_by_year = (
        coverage_source.groupby("application_year", dropna=False)
        .agg(
            row_count=("has_income", "size"),
            income_non_null_count=("has_income", "sum"),
            actively_adjusted_count=("actively_adjusted", "sum"),
        )
        .reset_index()
    )
    coverage_by_year["income_non_null_share"] = (
        coverage_by_year["income_non_null_count"] / coverage_by_year["row_count"]
    )
    coverage_by_year["actively_adjusted_share_of_year"] = (
        coverage_by_year["actively_adjusted_count"] / coverage_by_year["row_count"]
    )
    coverage_by_year["actively_adjusted_share_of_income"] = (
        coverage_by_year["actively_adjusted_count"]
        / coverage_by_year["income_non_null_count"].replace(0, np.nan)
    )

    diagnostics = {
        "father_income_column": father_income_column,
        "application_term_column": application_term_column,
        "row_count": int(len(dataframe)),
        "missing_father_income_count": int(father_income.isna().sum()),
        "default_factor_row_count": int((year_factor == 1.0).sum()),
        "active_adjustment_row_count": int((father_income.notna() & year_factor.ne(1.0)).sum()),
        "year_distribution": (
            application_year.astype("string")
            .fillna("missing_or_malformed")
            .value_counts(dropna=False)
            .sort_index()
            .rename_axis("application_year")
            .reset_index(name="row_count")
        ),
        "income_summary": module.summarize_income_series(father_income, adjusted_income),
        "application_year": application_year,
        "raw_income": father_income,
        "adjusted_income": adjusted_income,
        "year_factor": year_factor,
        "factor_map": {int(year): float(factor) for year, factor in factor_map.items()},
        "income_transform": income_transform or "linear",
        "raw_income_feature_name": raw_feature_name,
        "adjusted_income_feature_name": adjusted_feature_name,
        "coverage_by_year": coverage_by_year,
    }
    return model_a_features, model_b_features, diagnostics


def align_feature_sets_general(
    module: Any,
    *,
    model_a_features: pd.DataFrame,
    model_b_features: pd.DataFrame,
    target: pd.Series,
    raw_feature_name: str,
    adjusted_feature_name: str,
    original_income_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    (
        features_train_a,
        features_valid_a,
        features_test_a,
        target_train,
        target_valid,
        target_test,
    ) = module.split_dataset(model_a_features, target)

    features_train_b = model_b_features.loc[features_train_a.index].copy()
    features_valid_b = model_b_features.loc[features_valid_a.index].copy()
    features_test_b = model_b_features.loc[features_test_a.index].copy()

    columns_to_drop, _ = module.fit_feature_pruner(features_train_a)
    protected_features = {raw_feature_name}
    if original_income_column in features_train_a.columns:
        protected_features.add(original_income_column)
    columns_to_drop = [column for column in columns_to_drop if column not in protected_features]

    features_train_a = module.apply_feature_pruning(features_train_a, columns_to_drop)
    retained_columns_a = features_train_a.columns.tolist()
    retained_columns_b = [
        adjusted_feature_name if column == raw_feature_name else column
        for column in retained_columns_a
    ]

    features_valid_a = module.apply_feature_pruning(
        features_valid_a,
        columns_to_drop,
        retained_columns=retained_columns_a,
    )
    features_test_a = module.apply_feature_pruning(
        features_test_a,
        columns_to_drop,
        retained_columns=retained_columns_a,
    )
    features_train_b = module.apply_feature_pruning(
        features_train_b,
        columns_to_drop,
        retained_columns=retained_columns_b,
    )
    features_valid_b = module.apply_feature_pruning(
        features_valid_b,
        columns_to_drop,
        retained_columns=retained_columns_b,
    )
    features_test_b = module.apply_feature_pruning(
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


def run_full_feature_model_enhanced(
    module: Any,
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
    preprocessors, _, _ = module.build_preprocessors(features_train)
    pipeline = module.make_model_pipeline(preprocessors, model_spec)
    fitted_model = module.fit_model_prototype(pipeline, features_train, target_train)

    validation_probabilities = module.get_positive_class_probabilities(fitted_model, features_valid)
    _, validation_threshold_results = module.tune_thresholds(target_valid, validation_probabilities)
    selected_validation_metrics = validation_threshold_results[module.SELECTED_THRESHOLD_POLICY]

    test_probabilities = module.get_positive_class_probabilities(fitted_model, features_test)
    test_probability_metrics = module.compute_probability_metrics(target_test, test_probabilities)
    test_threshold_results = module.evaluate_threshold_policies(
        target_test,
        test_probabilities,
        {module.SELECTED_THRESHOLD_POLICY: selected_validation_metrics.threshold},
    )
    selected_test_metrics = test_threshold_results[module.SELECTED_THRESHOLD_POLICY]
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
    top_feature_importance, full_feature_importance = module.extract_feature_importance_tables(
        fitted_model,
        top_n=10,
    )

    return {
        "variant_name": variant_name,
        "model_name": model_name,
        "threshold_policy": module.SELECTED_THRESHOLD_POLICY,
        "selected_threshold": float(selected_validation_metrics.threshold),
        "auc": float(test_probability_metrics["roc_auc"]),
        "average_precision": float(test_probability_metrics["average_precision"]),
        "brier_score": float(test_probability_metrics["brier_score"]),
        "precision": float(selected_test_metrics.precision),
        "recall": float(selected_test_metrics.recall),
        "specificity": float(selected_test_metrics.specificity),
        "balanced_accuracy": float(selected_test_metrics.balanced_accuracy),
        "accuracy": float(selected_test_metrics.accuracy),
        "f1": float(selected_test_metrics.f1),
        "feature_count": int(features_train.shape[1]),
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities_series,
        "confusion_matrix": selected_test_metrics.confusion_matrix,
        "top_feature_importance": top_feature_importance,
        "full_feature_importance": full_feature_importance,
    }


def build_model_comparison_tables(
    *,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    diagnostics: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts_a = extract_confusion_counts(model_a["confusion_matrix"])
    counts_b = extract_confusion_counts(model_b["confusion_matrix"])

    comparison_table = pd.DataFrame(
        [
            {
                "model_version": model_a["variant_name"],
                "income_feature": diagnostics["raw_income_feature_name"],
                "estimator": model_a["model_name"],
                "feature_count": model_a["feature_count"],
                "selected_threshold": model_a["selected_threshold"],
                "roc_auc": model_a["auc"],
                "average_precision": model_a["average_precision"],
                "brier_score": model_a["brier_score"],
                "precision": model_a["precision"],
                "recall": model_a["recall"],
                "f1": model_a["f1"],
                "false_positives": counts_a["false_positives"],
                "false_negatives": counts_a["false_negatives"],
                "positive_predictions": int(model_a["test_predictions"].sum()),
                "positive_prediction_rate": float(model_a["test_predictions"].mean()),
                "policy_weighted_error": compute_policy_weighted_error(model_a["confusion_matrix"]),
            },
            {
                "model_version": model_b["variant_name"],
                "income_feature": diagnostics["adjusted_income_feature_name"],
                "estimator": model_b["model_name"],
                "feature_count": model_b["feature_count"],
                "selected_threshold": model_b["selected_threshold"],
                "roc_auc": model_b["auc"],
                "average_precision": model_b["average_precision"],
                "brier_score": model_b["brier_score"],
                "precision": model_b["precision"],
                "recall": model_b["recall"],
                "f1": model_b["f1"],
                "false_positives": counts_b["false_positives"],
                "false_negatives": counts_b["false_negatives"],
                "positive_predictions": int(model_b["test_predictions"].sum()),
                "positive_prediction_rate": float(model_b["test_predictions"].mean()),
                "policy_weighted_error": compute_policy_weighted_error(model_b["confusion_matrix"]),
            },
        ]
    )

    delta_table = pd.DataFrame(
        [
            {"metric": "ROC AUC", "model_b_minus_a": model_b["auc"] - model_a["auc"]},
            {
                "metric": "Average Precision",
                "model_b_minus_a": model_b["average_precision"] - model_a["average_precision"],
            },
            {
                "metric": "Brier Score",
                "model_b_minus_a": model_b["brier_score"] - model_a["brier_score"],
            },
            {"metric": "Precision", "model_b_minus_a": model_b["precision"] - model_a["precision"]},
            {"metric": "Recall", "model_b_minus_a": model_b["recall"] - model_a["recall"]},
            {"metric": "F1", "model_b_minus_a": model_b["f1"] - model_a["f1"]},
            {
                "metric": "False Positives",
                "model_b_minus_a": counts_b["false_positives"] - counts_a["false_positives"],
            },
            {
                "metric": "False Negatives",
                "model_b_minus_a": counts_b["false_negatives"] - counts_a["false_negatives"],
            },
            {
                "metric": "Policy-Weighted Error",
                "model_b_minus_a": compute_policy_weighted_error(model_b["confusion_matrix"])
                - compute_policy_weighted_error(model_a["confusion_matrix"]),
            },
        ]
    )
    return comparison_table, delta_table


def build_year_metric_table(
    *,
    target_true: pd.Series,
    application_year: pd.Series,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    year_labels = application_year.astype("string").fillna("missing_or_malformed")

    for year_value in sorted(year_labels.unique().tolist()):
        segment_mask = year_labels == year_value
        segment_target = target_true.loc[segment_mask]
        if segment_target.empty:
            continue

        for model_label, model in (("Model A", model_a), ("Model B", model_b)):
            probabilities = model["test_probabilities"].loc[segment_mask]
            predictions = model["test_predictions"].loc[segment_mask]
            matrix = confusion_matrix(segment_target, predictions, labels=[0, 1]).tolist()
            counts = extract_confusion_counts(matrix)
            actual_positive = max(int((segment_target == 1).sum()), 1)
            actual_negative = max(int((segment_target == 0).sum()), 1)

            try:
                auc_value = float(roc_auc_score(segment_target, probabilities))
            except Exception:
                auc_value = float("nan")
            try:
                ap_value = float(average_precision_score(segment_target, probabilities))
            except Exception:
                ap_value = float("nan")

            rows.append(
                {
                    "application_year": year_value,
                    "model_version": model_label,
                    "row_count": int(segment_mask.sum()),
                    "actual_positive_rate": float(segment_target.mean()),
                    "mean_predicted_probability": float(probabilities.mean()),
                    "calibration_gap": float(probabilities.mean() - segment_target.mean()),
                    "roc_auc": auc_value,
                    "average_precision": ap_value,
                    "brier_score": float(brier_score_loss(segment_target, probabilities)),
                    "precision": float(precision_score(segment_target, predictions, zero_division=0)),
                    "recall": float(recall_score(segment_target, predictions, zero_division=0)),
                    "false_positive_rate": safe_rate(counts["false_positives"], actual_negative),
                    "false_negative_rate": safe_rate(counts["false_negatives"], actual_positive),
                    "positive_prediction_rate": float(predictions.mean()),
                }
            )

    return pd.DataFrame(rows)


def build_temporal_consistency_tables(
    year_metric_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus_metrics = [
        "roc_auc",
        "recall",
        "false_positive_rate",
        "false_negative_rate",
        "brier_score",
        "calibration_gap",
        "positive_prediction_rate",
    ]
    rows: list[dict[str, Any]] = []
    for model_version in ["Model A", "Model B"]:
        subset = year_metric_table[year_metric_table["model_version"] == model_version]
        for metric in focus_metrics:
            values = subset[metric].dropna()
            rows.append(
                {
                    "model_version": model_version,
                    "metric": metric,
                    "range": float(values.max() - values.min()) if not values.empty else float("nan"),
                    "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
                }
            )
    consistency_table = pd.DataFrame(rows)
    comparison = (
        consistency_table.pivot(index="metric", columns="model_version", values="range")
        .rename(columns={"Model A": "model_a_range", "Model B": "model_b_range"})
        .reset_index()
    )
    comparison["range_improvement"] = comparison["model_a_range"] - comparison["model_b_range"]
    return consistency_table, comparison


def build_distribution_sanity_check(diagnostics: dict[str, Any]) -> tuple[pd.DataFrame, str]:
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
            f"The mean shifted by {mean_shift_pct:.2f}%, so the feature changed in scale more than rank ordering."
        )

    return sanity_frame, explanation


def build_income_plot_frame(diagnostics: dict[str, Any], sample_size: int = 400) -> pd.DataFrame:
    valid_mask = diagnostics["raw_income"].notna() & diagnostics["adjusted_income"].notna()
    frame = pd.DataFrame(
        {
            "raw_income": diagnostics["raw_income"].loc[valid_mask],
            "adjusted_income": diagnostics["adjusted_income"].loc[valid_mask],
            "application_year": diagnostics["application_year"].loc[valid_mask].astype("string"),
        }
    )
    if len(frame) > sample_size:
        frame = frame.sample(sample_size, random_state=getattr(load_poc_module(), "RANDOM_STATE", 42))
    return frame


def build_prediction_impact_summary(model_a: dict[str, Any], model_b: dict[str, Any]) -> dict[str, Any]:
    prediction_diff_mask = model_a["test_predictions"].ne(model_b["test_predictions"])
    changed_count = int(prediction_diff_mask.sum())
    total_count = int(len(model_a["test_predictions"]))
    return {
        "changed_case_count": changed_count,
        "changed_case_pct": safe_rate(changed_count, total_count),
        "negative_to_positive_count": int(
            ((model_a["test_predictions"] == 0) & (model_b["test_predictions"] == 1)).sum()
        ),
        "positive_to_negative_count": int(
            ((model_a["test_predictions"] == 1) & (model_b["test_predictions"] == 0)).sum()
        ),
    }


def build_changed_predictions_frame(
    *,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    diagnostics: dict[str, Any],
) -> pd.DataFrame:
    changed_mask = model_a["test_predictions"].ne(model_b["test_predictions"])
    changed_index = model_a["test_predictions"].loc[changed_mask].index
    if len(changed_index) == 0:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "source_index": changed_index,
            "application_year": diagnostics["application_year"].reindex(changed_index).astype("string"),
            "raw_income": diagnostics["raw_income"].reindex(changed_index),
            "adjusted_income": diagnostics["adjusted_income"].reindex(changed_index),
            "probability_before": model_a["test_probabilities"].reindex(changed_index),
            "probability_after": model_b["test_probabilities"].reindex(changed_index),
            "decision_before": model_a["test_predictions"].reindex(changed_index).map({0: "Deny/No award", 1: "Award"}),
            "decision_after": model_b["test_predictions"].reindex(changed_index).map({0: "Deny/No award", 1: "Award"}),
        }
    )
    frame["probability_delta"] = frame["probability_after"] - frame["probability_before"]
    frame["income_ratio"] = frame["adjusted_income"] / frame["raw_income"].replace(0, np.nan)
    return frame.sort_values("probability_delta", key=lambda series: series.abs(), ascending=False)


def build_sensitivity_row(scenario_name: str, pair_result: dict[str, Any], transform_label: str) -> dict[str, Any]:
    diagnostics = pair_result["diagnostics"]
    delta_lookup = pair_result["metric_delta_table"].set_index("metric")["model_b_minus_a"]
    income_non_null_count = int(diagnostics["raw_income"].notna().sum())
    return {
        "scenario": scenario_name,
        "scenario_label": PP_SCENARIO_META.get(scenario_name, {}).get("label", scenario_name),
        "transform": transform_label,
        "active_adjustment_share_all_rows": safe_rate(
            diagnostics["active_adjustment_row_count"],
            diagnostics["row_count"],
        ),
        "active_adjustment_share_non_missing_income": safe_rate(
            diagnostics["active_adjustment_row_count"],
            max(income_non_null_count, 1),
        ),
        "delta_roc_auc": float(delta_lookup.get("ROC AUC", 0.0)),
        "delta_average_precision": float(delta_lookup.get("Average Precision", 0.0)),
        "delta_brier_score": float(delta_lookup.get("Brier Score", 0.0)),
        "delta_precision": float(delta_lookup.get("Precision", 0.0)),
        "delta_recall": float(delta_lookup.get("Recall", 0.0)),
        "delta_f1": float(delta_lookup.get("F1", 0.0)),
        "delta_false_positives": float(delta_lookup.get("False Positives", 0.0)),
        "delta_false_negatives": float(delta_lookup.get("False Negatives", 0.0)),
        "delta_policy_weighted_error": float(delta_lookup.get("Policy-Weighted Error", 0.0)),
        "changed_prediction_pct": float(pair_result["prediction_impact"]["changed_case_pct"]),
    }


def fit_controlled_model_pair(
    module: Any,
    *,
    training_dataframe: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    factor_map: dict[int, float],
    income_transform: str | None = None,
) -> dict[str, Any]:
    model_a_features, model_b_features, diagnostics = build_income_variant_feature_sets(
        module,
        training_dataframe,
        target_column=target_column,
        factor_map=factor_map,
        income_transform=income_transform,
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
    ) = align_feature_sets_general(
        module,
        model_a_features=model_a_features,
        model_b_features=model_b_features,
        target=target,
        raw_feature_name=diagnostics["raw_income_feature_name"],
        adjusted_feature_name=diagnostics["adjusted_income_feature_name"],
        original_income_column=diagnostics["father_income_column"],
    )

    class_weight = module.get_recommended_class_weight(target_train)
    scale_pos_weight = module.get_recommended_scale_pos_weight(target_train)
    model_specs = module.get_model_specs(
        class_weight=class_weight,
        scale_pos_weight=scale_pos_weight,
    )
    model_name = module.resolve_model_name(model_specs)
    model_spec = model_specs[model_name]

    model_a = run_full_feature_model_enhanced(
        module,
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
    model_b = run_full_feature_model_enhanced(
        module,
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

    comparison_table, metric_delta_table = build_model_comparison_tables(
        model_a=model_a,
        model_b=model_b,
        diagnostics=diagnostics,
    )
    prediction_impact = build_prediction_impact_summary(model_a, model_b)
    year_metric_table = build_year_metric_table(
        target_true=target_test,
        application_year=diagnostics["application_year"].loc[features_test_a.index],
        model_a=model_a,
        model_b=model_b,
    )
    _, temporal_consistency_delta = build_temporal_consistency_tables(year_metric_table)
    distribution_sanity_check, distribution_explanation = build_distribution_sanity_check(diagnostics)

    return {
        "diagnostics": diagnostics,
        "comparison_table": comparison_table,
        "metric_delta_table": metric_delta_table,
        "prediction_impact": prediction_impact,
        "year_metric_table": year_metric_table,
        "temporal_consistency_delta": temporal_consistency_delta,
        "distribution_sanity_check": distribution_sanity_check,
        "distribution_explanation": distribution_explanation,
        "income_plot_frame": build_income_plot_frame(diagnostics),
        "changed_predictions_frame": build_changed_predictions_frame(
            model_a=model_a,
            model_b=model_b,
            diagnostics=diagnostics,
        ),
        "confusion_matrix_a": module.build_confusion_matrix_frame(model_a["confusion_matrix"]),
        "confusion_matrix_b": module.build_confusion_matrix_frame(model_b["confusion_matrix"]),
        "top_feature_importance_a": model_a["top_feature_importance"],
        "top_feature_importance_b": model_b["top_feature_importance"],
        "coverage_by_year": diagnostics["coverage_by_year"],
    }


def infer_recommendation(
    *,
    primary_result: dict[str, Any],
    sensitivity_frame: pd.DataFrame,
    temporal_consistency_delta: pd.DataFrame,
) -> tuple[str, str]:
    comparison_table = primary_result["comparison_table"]
    model_a_row = comparison_table.loc[comparison_table["model_version"] == "Model A"].iloc[0]
    model_b_row = comparison_table.loc[comparison_table["model_version"] == "Model B"].iloc[0]

    false_negative_delta = int(model_b_row["false_negatives"] - model_a_row["false_negatives"])
    policy_error_delta = float(
        model_b_row["policy_weighted_error"] - model_a_row["policy_weighted_error"]
    )
    recall_delta = float(model_b_row["recall"] - model_a_row["recall"])
    temporal_improvements = int((temporal_consistency_delta["range_improvement"] > 0).sum())
    scenario_fn_improvements = int((sensitivity_frame["delta_false_negatives"] < 0).sum())
    scenario_policy_improvements = int(
        (sensitivity_frame["delta_policy_weighted_error"] < 0).sum()
    )

    if (
        false_negative_delta < 0
        and policy_error_delta < 0
        and recall_delta > 0
        and temporal_improvements >= 3
        and scenario_fn_improvements >= 2
    ):
        recommendation = "Promising, but still experimental"
        rationale = (
            "The adjusted feature improves missed-need protection and lowers policy-weighted error "
            "while staying reasonably stable across years and scenarios."
        )
    elif policy_error_delta > 0 and scenario_policy_improvements == 0:
        recommendation = "Keep the raw income field"
        rationale = (
            "The adjusted feature increases weighted policy cost in the primary run and does not recover "
            "under the tested purchasing-power scenarios."
        )
    else:
        recommendation = "Use as a thesis-side experiment"
        rationale = (
            "The idea is analytically useful, but the operational evidence remains mixed across scenarios, "
            "years, or both."
        )

    rationale += (
        f" Primary delta: recall {recall_delta:+.3f}, false negatives {false_negative_delta:+d}, "
        f"policy-weighted error {policy_error_delta:+.1f}. "
        f"Scenario improvements: {scenario_fn_improvements} lower-FN scenarios and "
        f"{scenario_policy_improvements} lower-cost scenarios."
    )
    return recommendation, rationale


@st.cache_data(show_spinner="Running the purchasing-power notebook experiment...")
def run_purchasing_power_dashboard() -> dict[str, Any] | None:
    module = load_poc_module()
    if module is None:
        return None

    dataframe = module.load_data(module.DATA_PATH)
    training_dataframe, target, target_metadata = module.prepare_target(dataframe)
    notebook_application_year = module.extract_application_year(
        training_dataframe[module.resolve_column_name(training_dataframe, module.APPLICATION_TERM_CANDIDATES)]
    )
    observed_years, reference_year, scenarios = build_pp_scenarios(
        notebook_application_year,
        module.PP_FACTOR,
    )
    factor_scenario_table = build_factor_scenario_table(observed_years, reference_year, scenarios)

    primary_result = fit_controlled_model_pair(
        module,
        training_dataframe=training_dataframe,
        target=target,
        target_column=target_metadata["target_column"],
        factor_map=scenarios[PRIMARY_SCENARIO_NAME],
    )

    sensitivity_rows = []
    for scenario_name in ["legacy_original", "conservative", "baseline", "aggressive"]:
        pair = fit_controlled_model_pair(
            module,
            training_dataframe=training_dataframe,
            target=target,
            target_column=target_metadata["target_column"],
            factor_map=scenarios[scenario_name],
        )
        sensitivity_rows.append(build_sensitivity_row(scenario_name, pair, "linear"))
    sensitivity_frame = pd.DataFrame(sensitivity_rows)

    log_pair = fit_controlled_model_pair(
        module,
        training_dataframe=training_dataframe,
        target=target,
        target_column=target_metadata["target_column"],
        factor_map=scenarios[PRIMARY_SCENARIO_NAME],
        income_transform="log1p",
    )
    log_robustness_frame = pd.DataFrame(
        [build_sensitivity_row(PRIMARY_SCENARIO_NAME, log_pair, "log1p")]
    )

    recommendation, recommendation_rationale = infer_recommendation(
        primary_result=primary_result,
        sensitivity_frame=sensitivity_frame,
        temporal_consistency_delta=primary_result["temporal_consistency_delta"],
    )

    return {
        "factor_scenario_table": factor_scenario_table,
        "primary_result": primary_result,
        "sensitivity_frame": sensitivity_frame,
        "log_robustness_frame": log_robustness_frame,
        "recommendation": recommendation,
        "recommendation_rationale": recommendation_rationale,
    }


def build_confusion_heatmap_frame(matrix: pd.DataFrame, model_label: str) -> pd.DataFrame:
    frame = matrix.reset_index().rename(columns={"index": "actual"})
    melted = frame.melt(id_vars=["actual"], var_name="predicted", value_name="cases")
    melted["model_version"] = model_label
    return melted


def build_income_distribution_frame(income_plot_frame: pd.DataFrame) -> pd.DataFrame:
    distribution = income_plot_frame.melt(
        value_vars=["raw_income", "adjusted_income"],
        var_name="income_type",
        value_name="income_value",
    )
    distribution["income_type"] = distribution["income_type"].map(
        {"raw_income": "Raw income", "adjusted_income": "Adjusted income"}
    )
    return distribution


apply_design_system()

poc_result = run_purchasing_power_dashboard()
if poc_result is None:
    st.error(f"Purchasing-power experiment source not found: {DEFAULT_POC_EXPERIMENT_PATH}")
else:
    primary_result = poc_result["primary_result"]
    comparison_table = primary_result["comparison_table"]
    model_a_row = comparison_table.loc[comparison_table["model_version"] == "Model A"].iloc[0]
    model_b_row = comparison_table.loc[comparison_table["model_version"] == "Model B"].iloc[0]
    prediction_impact = primary_result["prediction_impact"]

    render_decision_journey("Model")
    render_page_header(
        title="Purchasing Power Experiment",
        description=(
            "Thesis-side sensitivity experiment that swaps raw father income for a year-adjusted "
            "purchasing-power proxy, then measures how eligibility behavior changes."
        ),
        takeaway=(
            "Purchasing-power adjustment is analytically useful but not production-ready; "
            "the experiment shows income representation is not the main bottleneck."
        ),
        kicker="POC Notebook Workspace",
        pills=[
            (poc_result["recommendation"], "primary"),
            (
                f"{prediction_impact['changed_case_pct']:.1%} changed predictions",
                "warning",
            ),
            (
                f"{int(model_b_row['false_negatives'] - model_a_row['false_negatives']):+d} false negatives",
                "danger",
            ),
            (DEFAULT_POC_NOTEBOOK_PATH.name, "secondary"),
        ],
    )
    render_takeaway_box(
        (
            f"Baseline deltas: ROC AUC {float(model_b_row['roc_auc'] - model_a_row['roc_auc']):+.4f}, "
            f"F1 {float(model_b_row['f1'] - model_a_row['f1']):+.4f}, weighted cost "
            f"{float(model_b_row['policy_weighted_error'] - model_a_row['policy_weighted_error']):+.1f}, "
            f"and {prediction_impact['changed_case_pct']:.1%} changed predictions. "
            "This supports sensitivity analysis, not production deployment."
        ),
        status="Caution",
    )
    render_kpi_row(
        [
            {
                "label": "ROC AUC Delta",
                "value": f"{float(model_b_row['roc_auc'] - model_a_row['roc_auc']):+.4f}",
                "note": "Model B minus Model A under the baseline adjustment scenario.",
                "tone": "primary",
            },
            {
                "label": "F1 Delta",
                "value": f"{float(model_b_row['f1'] - model_a_row['f1']):+.4f}",
                "note": "Binary award/no-award F1 difference under the same shared split.",
                "tone": "secondary",
            },
            {
                "label": "Weighted Cost Delta",
                "value": f"{float(model_b_row['policy_weighted_error'] - model_a_row['policy_weighted_error']):+.1f}",
                "note": "False negatives are weighted more heavily than review-inducing false positives.",
                "tone": "warning",
            },
            {
                "label": "Changed Predictions",
                "value": f"{prediction_impact['changed_case_pct']:.1%}",
                "note": "Share of test cases where the raw-income and adjusted-income models disagree.",
                "tone": "danger",
            },
        ]
    )
    render_insight_action_panel(
        insight=(
            "This page follows the notebook framing: the goal is not to prove a production policy change, "
            "but to test whether year-aware income scaling meaningfully changes fairness, ranking, or decision cost."
        ),
        implication=(
            "Read the baseline comparison together with the robustness charts. A one-off metric lift is not enough "
            "if the effect disappears under alternative purchasing-power assumptions."
        ),
    )

    overview_tab, income_tab, comparison_tab, robustness_tab, temporal_tab, drivers_tab = st.tabs(
        ["Overview", "Income Shift", "Comparison", "Robustness", "Temporal", "Drivers"]
    )

    with overview_tab:
        st.subheader("Notebook Recommendation")
        render_chart_card(
            title="Effect-Size Badge",
            subtitle=poc_result["recommendation"],
            takeaway=(
                f"Changed predictions: {prediction_impact['changed_case_pct']:.1%}. "
                "Operational impact is low, so this remains a thesis-side sensitivity experiment."
            ),
            caption=poc_result["recommendation_rationale"],
            status=STATUS_NOT_PRODUCTION,
            sample_size=int(prediction_impact["changed_case_count"]),
            coverage=f"{prediction_impact['changed_case_pct']:.1%} changed",
        )
        render_table_card(
            title="Baseline vs Adjusted Model Deltas",
            dataframe=comparison_table.round(4),
            takeaway="The tiny AUC gain comes with F1 and weighted-cost tradeoffs, so it is not production-ready.",
            caption="Model A uses raw income; Model B uses the purchasing-power adjusted income proxy.",
            status=STATUS_CAUTION,
            sample_size=len(comparison_table),
        )
        with st.expander("Purchasing-Power Scenario Factors", expanded=False):
            st.dataframe(poc_result["factor_scenario_table"], width="stretch", hide_index=True)

    with income_tab:
        st.subheader("Raw Vs Adjusted Income")
        income_plot_frame = primary_result["income_plot_frame"]
        top_row = st.columns(2)
        with top_row[0]:
            if income_plot_frame.empty:
                st.info("Income-shift plot data is not available.")
            else:
                scatter_chart = (
                    alt.Chart(income_plot_frame)
                    .mark_circle(size=55, opacity=0.65)
                    .encode(
                        x=alt.X("raw_income:Q", title="Raw income"),
                        y=alt.Y("adjusted_income:Q", title="Adjusted income"),
                        color=alt.Color(
                            "application_year:N",
                            scale=alt.Scale(range=CHART_NEUTRALS),
                            legend=alt.Legend(title="Application year"),
                        ),
                        tooltip=[
                            alt.Tooltip("application_year:N", title="Year"),
                            alt.Tooltip("raw_income:Q", title="Raw income", format=",.0f"),
                            alt.Tooltip("adjusted_income:Q", title="Adjusted income", format=",.0f"),
                        ],
                    )
                    .properties(height=340, title="Raw vs Adjusted Income")
                )
                render_chart_card(
                    title="Raw vs Adjusted Income",
                    subtitle="Sampled points by application year.",
                    takeaway="The adjustment mostly rescales income rather than changing rank order.",
                    chart=scatter_chart,
                    caption="This scatter should stay close to a monotonic relationship if the adjustment is not materially changing applicant order.",
                    status=STATUS_CAUTION,
                    sample_size=len(income_plot_frame),
                    warning="Income remains currency-sensitive; purchasing-power adjustment does not solve missingness or currency ambiguity.",
                )
        with top_row[1]:
            if income_plot_frame.empty:
                st.info("Income-distribution data is not available.")
            else:
                distribution_frame = build_income_distribution_frame(income_plot_frame)
                histogram = (
                    alt.Chart(distribution_frame)
                    .mark_bar(opacity=0.55)
                    .encode(
                        x=alt.X("income_value:Q", bin=alt.Bin(maxbins=30), title="Income value"),
                        y=alt.Y("count():Q", title="Cases"),
                        color=alt.Color(
                            "income_type:N",
                            scale=alt.Scale(
                                domain=["Raw income", "Adjusted income"],
                                range=[PRIMARY_COLOR, SECONDARY_COLOR],
                            ),
                        ),
                    )
                    .properties(height=340, title="Income Distribution Shift")
                )
                render_chart_card(
                    title="Income Distribution Shift",
                    subtitle="Raw and adjusted income distributions.",
                    takeaway="Distribution movement is visible, but operational decision changes remain very small.",
                    chart=histogram,
                    caption="Use this as sensitivity evidence, not as proof that income representation is production-ready.",
                    status=STATUS_CAUTION,
                    sample_size=len(distribution_frame),
                )

        if not income_plot_frame.empty:
            ratio_frame = income_plot_frame.copy()
            ratio_frame["income_ratio"] = ratio_frame["adjusted_income"] / ratio_frame["raw_income"].replace(0, np.nan)
            ratio_frame = ratio_frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["income_ratio"])
            if not ratio_frame.empty:
                ratio_chart = (
                    alt.Chart(ratio_frame)
                    .mark_boxplot(extent="min-max", color=PRIMARY_COLOR)
                    .encode(
                        x=alt.X("application_year:N", title="Application year"),
                        y=alt.Y("income_ratio:Q", title="Adjusted / raw income ratio"),
                        tooltip=[
                            alt.Tooltip("application_year:N", title="Year"),
                            alt.Tooltip("income_ratio:Q", title="Ratio", format=".2f"),
                        ],
                    )
                    .properties(height=310)
                )
                render_chart_card(
                    title="Adjustment Ratio By Year",
                    subtitle="Box plot of adjusted/raw income ratio.",
                    takeaway="Year-level ratios explain the experiment more clearly than raw currency-scale scatter alone.",
                    chart=ratio_chart,
                    caption="A ratio near 1.0 means little adjustment; wider boxes indicate stronger year-level scaling differences.",
                    status=STATUS_CAUTION,
                    sample_size=len(ratio_frame),
                )

        render_table_card(
            title="Distribution Sanity Check",
            dataframe=primary_result["distribution_sanity_check"].round(4),
            takeaway="High raw-adjusted correlation means rank order barely changes.",
            caption=primary_result["distribution_explanation"],
            status=STATUS_CAUTION,
        )

    with comparison_tab:
        st.subheader("Headline Model Comparison")
        confusion_row = st.columns([1, 1, 1.2])
        with confusion_row[0]:
            matrix_a = build_confusion_heatmap_frame(primary_result["confusion_matrix_a"], "Model A")
            render_chart_card(
                title="Model A Confusion Matrix",
                subtitle="Raw-income baseline.",
                takeaway="False negatives are the highest-risk error type for aid review.",
                chart=make_confusion_matrix_chart(matrix_a),
                caption="Cells show count, row share, and column share.",
                status=STATUS_CAUTION,
                sample_size=int(matrix_a["cases"].sum()),
            )
        with confusion_row[1]:
            matrix_b = build_confusion_heatmap_frame(primary_result["confusion_matrix_b"], "Model B")
            render_chart_card(
                title="Model B Confusion Matrix",
                subtitle="Purchasing-power adjusted income.",
                takeaway="The adjusted model does not materially change the error profile.",
                chart=make_confusion_matrix_chart(matrix_b),
                caption="Cells show count, row share, and column share.",
                status=STATUS_CAUTION,
                sample_size=int(matrix_b["cases"].sum()),
            )
        with confusion_row[2]:
            policy_cost_frame = comparison_table[["model_version", "policy_weighted_error"]].copy()
            policy_chart = (
                alt.Chart(policy_cost_frame)
                .mark_bar()
                .encode(
                    x=alt.X("model_version:N", title=None),
                    y=alt.Y("policy_weighted_error:Q", title="Weighted error units"),
                    color=alt.Color(
                        "model_version:N",
                        scale=alt.Scale(
                            domain=["Model A", "Model B"],
                            range=[MODEL_COLORS["Model A"], MODEL_COLORS["Model B"]],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("model_version:N", title="Model"),
                        alt.Tooltip("policy_weighted_error:Q", title="Weighted error", format=".1f"),
                    ],
                )
                .properties(height=220, title="Policy-Weighted Error")
            )
            render_chart_card(
                title="Policy-Weighted Error",
                subtitle="False negatives carry heavier review cost.",
                takeaway="Weighted cost increases under the adjusted model in the baseline scenario.",
                chart=policy_chart,
                caption="Lower is better; this makes the tiny AUC change less persuasive for production use.",
                status=STATUS_CAUTION,
                sample_size=len(policy_cost_frame),
            )

        render_table_card(
            title="Comparison Table",
            dataframe=comparison_table.round(4),
            takeaway="The production question is not whether one metric moved, but whether routing behavior improved enough to justify the data risk.",
            status=STATUS_CAUTION,
        )
        changed_predictions = primary_result.get("changed_predictions_frame", pd.DataFrame()).copy()
        if changed_predictions.empty:
            render_chart_card(
                title="Changed Predictions Only",
                takeaway="No final predictions changed between the two model variants.",
                status=STATUS_RELIABLE,
            )
        else:
            display_changed = changed_predictions.copy()
            for column in ["raw_income", "adjusted_income"]:
                display_changed[column] = pd.to_numeric(display_changed[column], errors="coerce").map(
                    lambda value: f"{value:,.0f}" if pd.notna(value) else "N/A"
                )
            for column in ["probability_before", "probability_after", "probability_delta", "income_ratio"]:
                display_changed[column] = pd.to_numeric(display_changed[column], errors="coerce").map(
                    lambda value: f"{value:.3f}" if pd.notna(value) else "N/A"
                )
            render_table_card(
                title="Changed Predictions Only",
                dataframe=display_changed.head(25),
                takeaway="Only changed cases are shown because the operational impact is small.",
                caption="This table is the audit trail for cases where the sensitivity experiment actually changes a decision label.",
                status=STATUS_CAUTION,
                sample_size=len(changed_predictions),
            )

    with robustness_tab:
        st.subheader("Sensitivity And Robustness Analysis")
        sensitivity_frame = poc_result["sensitivity_frame"].copy()
        sensitivity_frame["scenario_order"] = sensitivity_frame["scenario"].map(
            {"legacy_original": 0, "conservative": 1, "baseline": 2, "aggressive": 3}
        )
        sensitivity_frame = sensitivity_frame.sort_values("scenario_order", ignore_index=True)

        chart_row = st.columns(3)
        scenario_color_scale = alt.Scale(
            domain=list(SCENARIO_COLORS),
            range=list(SCENARIO_COLORS.values()),
        )
        chart_specs = [
            ("delta_roc_auc", "ROC AUC Delta"),
            ("delta_f1", "F1 Delta"),
            ("delta_policy_weighted_error", "Policy-Weighted Error Delta"),
        ]
        for column, (metric_key, title) in zip(chart_row, chart_specs):
            with column:
                metric_chart = (
                    alt.Chart(sensitivity_frame)
                    .mark_bar()
                    .encode(
                        x=alt.X("scenario_label:N", title=None, sort=None),
                        y=alt.Y(f"{metric_key}:Q", title=title),
                        color=alt.Color("scenario:N", scale=scenario_color_scale, legend=None),
                        tooltip=[
                            alt.Tooltip("scenario_label:N", title="Scenario"),
                            alt.Tooltip(f"{metric_key}:Q", title=title, format=".4f"),
                        ],
                    )
                    .properties(height=300, title=title)
                )
                zero_rule = (
                    alt.Chart(pd.DataFrame({"y": [0.0]}))
                    .mark_rule(color="#111827")
                    .encode(y="y:Q")
                )
                render_chart_card(
                    title=title,
                    subtitle="Robustness tornado component by scenario.",
                    takeaway="Scenario sensitivity stays small relative to production-governance risks.",
                    chart=alt.layer(metric_chart, zero_rule),
                    caption="The zero line shows no change versus the raw-income model.",
                    status=STATUS_CAUTION,
                    sample_size=len(sensitivity_frame),
                )

        render_table_card(
            title="Robustness Scenario Table",
            dataframe=sensitivity_frame.round(4),
            takeaway="Changed-prediction share remains low across scenarios.",
            status=STATUS_CAUTION,
        )
        st.markdown("**Log-Transform Check**")
        render_table_card(
            title="Log-Transform Check",
            dataframe=poc_result["log_robustness_frame"].round(4),
            takeaway="Transform choice does not turn the experiment into a production-ready income fix.",
            status=STATUS_NOT_PRODUCTION,
        )

    with temporal_tab:
        st.subheader("Temporal Fairness And Year-Level Consistency")
        year_metric_table = primary_result["year_metric_table"]
        if year_metric_table.empty:
            st.info("Year-level metrics are not available.")
        else:
            temporal_specs = [
                ("roc_auc", "ROC AUC"),
                ("recall", "Recall"),
                ("false_positive_rate", "False Positive Rate"),
                ("brier_score", "Brier Score"),
            ]
            chart_slots = [col for row in (st.columns(2), st.columns(2)) for col in row]
            for slot, (metric_key, metric_label) in zip(chart_slots, temporal_specs):
                with slot:
                    chart = (
                        alt.Chart(year_metric_table)
                        .mark_line(point=True, strokeWidth=3)
                        .encode(
                            x=alt.X("application_year:N", title="Application year"),
                            y=alt.Y(f"{metric_key}:Q", title=metric_label),
                            color=alt.Color(
                                "model_version:N",
                                scale=alt.Scale(
                                    domain=["Model A", "Model B"],
                                    range=[MODEL_COLORS["Model A"], MODEL_COLORS["Model B"]],
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip("application_year:N", title="Year"),
                                alt.Tooltip("model_version:N", title="Model"),
                                alt.Tooltip(f"{metric_key}:Q", title=metric_label, format=".4f"),
                            ],
                        )
                        .properties(height=260, title=metric_label)
                    )
                    year_counts = year_metric_table.groupby("application_year").size().sum()
                    render_chart_card(
                        title=f"Temporal Consistency: {metric_label}",
                        subtitle="Model A and Model B by application year.",
                        takeaway="Temporal stability should be interpreted with year-level sample size in mind.",
                        chart=chart,
                        caption="Use the same y-axis interpretation across years; small year slices should not drive production policy.",
                        status=STATUS_CAUTION,
                        sample_size=int(year_counts),
                    )

            render_table_card(
                title="Temporal Consistency Delta",
                dataframe=primary_result["temporal_consistency_delta"].round(4),
                takeaway="Year-level movement is monitoring evidence, not a production go/no-go result.",
                status=STATUS_CAUTION,
            )

    with drivers_tab:
        st.subheader("Explainability And Model Behavior Change")
        driver_cols = st.columns(2)
        driver_frames = [
            ("Model A", primary_result["top_feature_importance_a"], MODEL_COLORS["Model A"]),
            ("Model B", primary_result["top_feature_importance_b"], MODEL_COLORS["Model B"]),
        ]
        for column, (label, driver_frame, color) in zip(driver_cols, driver_frames):
            with column:
                if driver_frame.empty:
                    st.info(f"{label} driver export is unavailable for the selected estimator.")
                else:
                    render_chart_card(
                        title=f"Raw Top Drivers: {label}",
                        subtitle="Human-readable feature labels.",
                        takeaway="Feature importance explains model behavior, not causal policy rules.",
                        chart=make_feature_importance_chart(driver_frame, title=f"Top Drivers: {label}"),
                        caption="Raw features are grouped and translated where possible; reviewers should treat them as behavior diagnostics.",
                        status=STATUS_CAUTION,
                        sample_size=len(driver_frame),
                    )

                    grouped_driver_frame = build_grouped_feature_importance(driver_frame)
                    render_chart_card(
                        title=f"Grouped Importance: {label}",
                        subtitle="Feature importance rolled into thesis-friendly evidence families.",
                        takeaway="Grouped drivers make the model easier to explain to non-technical stakeholders.",
                        chart=make_grouped_importance_chart(grouped_driver_frame),
                        caption="Grouping is a readability layer; it does not change the underlying model.",
                        status=STATUS_RELIABLE,
                        sample_size=len(driver_frame),
                    )

        st.caption(
            "This follows the notebook's behavior-change check: even when the accuracy deltas are small, "
            "the dashboard still shows whether the model starts leaning on a different income representation."
        )
