#!/usr/bin/env python3

from __future__ import annotations

import inspect
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MISSING_DEPENDENCIES: list[str] = []

try:
    import joblib
except ImportError:
    MISSING_DEPENDENCIES.append("joblib")

try:
    import numpy as np
except ImportError:
    MISSING_DEPENDENCIES.append("numpy")

try:
    import pandas as pd
except ImportError:
    MISSING_DEPENDENCIES.append("pandas")

try:
    import sklearn  # noqa: F401
except ImportError:
    MISSING_DEPENDENCIES.append("scikit-learn")

if MISSING_DEPENDENCIES:
    missing_list = ", ".join(sorted(MISSING_DEPENDENCIES))
    raise ImportError(
        "Missing required Python packages: "
        f"{missing_list}. Install them with:\n"
        "/opt/homebrew/bin/python3 -m pip install "
        "joblib numpy pandas scikit-learn matplotlib"
    )

try:
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        make_scorer,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import (
        RandomizedSearchCV,
        RepeatedStratifiedKFold,
        StratifiedKFold,
        cross_validate,
        train_test_split,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:
    raise ImportError(
        "Missing required scikit-learn components. Install them with:\n"
        "/opt/homebrew/bin/python3 -m pip install scikit-learn"
    ) from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def _resolve_config_path(
    *,
    env_var: str,
    default_path: Path,
    base_dir: Path,
) -> Path:
    """Resolve a configurable path from the environment or a sensible default."""
    configured = os.environ.get(env_var)
    if not configured:
        return default_path

    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = _resolve_config_path(
    env_var="YASMINA_ELIGIBILITY_DATA_PATH",
    default_path=PROJECT_ROOT / "cleaned data/faid_cleaned.csv",
    base_dir=PROJECT_ROOT,
)
FALLBACK_DATA_PATHS = (
    DATA_PATH,
    PROJECT_ROOT / "cleaned data/faid_cleaned.csv",
    PROJECT_ROOT / "cleaned/faid_cleaned.csv",
    Path("cleaned data/faid_cleaned.csv"),
    Path("cleaned/faid_cleaned.csv"),
)
TARGET_COLUMN = "decision"
POSITIVE_TARGET_LABEL = "awarded"
POSITIVE_TARGET_LABELS = (
    POSITIVE_TARGET_LABEL,
    "awarded_affidavit_of_promise",
)
NEGATIVE_TARGET_LABELS = ("denied",)
EXCLUDED_TARGET_LABELS = ("usaid",)
ALLOW_ONLY_EXPLICIT_TARGET_LABELS = True

ARTIFACT_DIR = _resolve_config_path(
    env_var="YASMINA_ELIGIBILITY_ARTIFACT_DIR",
    default_path=PROJECT_ROOT / "artifacts/yasmina_eligibility",
    base_dir=PROJECT_ROOT,
)
MODEL_OUTPUT_PATH = ARTIFACT_DIR / "yasmina_eligibility_pipeline.joblib"
METADATA_OUTPUT_PATH = ARTIFACT_DIR / "yasmina_eligibility_metadata.json"
REPORT_OUTPUT_PATH = ARTIFACT_DIR / "yasmina_eligibility_report.txt"
ROC_CURVE_PATH = ARTIFACT_DIR / "yasmina_eligibility_roc_curve.png"
PR_CURVE_PATH = ARTIFACT_DIR / "yasmina_eligibility_pr_curve.png"
DECISION_OUTPUT_PATH = ARTIFACT_DIR / "yasmina_eligibility_decision_recommendations.csv"
MODEL_ARTIFACT_TYPE = "yasmina_eligibility_bundle"
MODEL_ARTIFACT_VERSION = 3

RANDOM_STATE = 42
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
CV_FOLDS = 5
REPEATED_CV_REPEATS = 3
BOOTSTRAP_ITERATIONS = 250
TOP_FEATURES_TO_SHOW = 20
DECISION_EXAMPLE_COUNT_PER_ACTION = 2
# Serial CV is more reliable than joblib multiprocessing in this desktop setup.
N_JOBS = 1

THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)
SELECTED_THRESHOLD_POLICY = "faid_need_sensitive"
MIN_PRECISION_FLOOR = 0.60
MIN_RECALL_FLOOR: float | None = None

FAID_POLICY_PRIORITY_WEIGHTS = {
    "benefit_true_positive": 1.0,
    "cost_false_positive_review": 1.0,
    "cost_false_negative_missed_need": 3.0,
}
FAID_POLICY_WEIGHT_SENSITIVITY_RATIOS = (1.0, 2.0, 3.0, 5.0)
FAID_POLICY_RATIONALE = (
    "Missing a deserving student is worse than sending an extra borderline case "
    "to committee review, so threshold selection should prioritize recall as long "
    "as shortlist quality remains operationally usable."
)
FAID_POLICY_WEIGHT_RATIONALE = (
    "The 3:1 weighting between missed-need errors and extra-review burden is a policy assumption "
    "that reflects committee priorities rather than an empirically learned truth. It encodes the view "
    "that excluding a deserving student is materially more harmful than reviewing one additional borderline case."
)

DECISION_APPROVE_MIN_PROBABILITY = 0.80
DECISION_REJECT_MAX_PROBABILITY = 0.20
DECISION_REVIEW_MARGIN = 0.10
DECISION_BOUNDARY_RATIONALE = (
    "Approve and reject actions require materially stronger confidence than the policy threshold itself, "
    "while a review band around the threshold preserves human oversight for borderline cases."
)

FAIRNESS_GAP_THRESHOLD = 0.10
FAIRNESS_MIN_GROUP_SIZE = 30
FAIRNESS_THRESHOLD_RATIONALE = (
    "A 0.10 subgroup gap is used as a practical governance trigger: smaller differences are monitored, "
    "while larger differences are treated as operationally meaningful enough to override automation."
)
FAIRNESS_MIN_GROUP_SIZE_RATIONALE = (
    "A minimum subgroup size of 30 is used to avoid overreacting to very small groups where error-rate estimates are unstable."
)
FAIRNESS_LIMITATIONS_NOTE = (
    "Fairness monitoring in this workflow is rule-based rather than causal. Group gaps should be interpreted as governance signals "
    "for review, not definitive proof of unfair treatment."
)
LOCAL_EXPLANATION_NOTE = (
    "Local case explanations are model-native approximations unless SHAP is separately added. They are intended to improve transparency, "
    "not to claim a full causal explanation of any individual eligibility decision."
)
TEMPORAL_VALIDATION_MIN_CYCLE_ROWS = 50
FAIRNESS_AUDIT_COLUMNS = (
    "gender",
    "applicant_gender",
    "school",
    "region",
    "applicant_region",
    "level",
    "nationality",
    "applicant_citizenship",
    "citizenship",
    "parsed_gender",
    "parsed_school",
    "parsed_region",
    "parsed_level",
    "parsed_nationality",
    "parsed_applicant_citizenship",
)
IDENTIFIER_COLUMNS = {
    "source_row_number",
    "source_sheet_name",
    "application_id",
    "student_id",
    "id",
    "raw_source_row_number",
    "raw_source_sheet_name",
}
AUDIT_ONLY_COLUMNS = {
    "gender",
    "applicant_gender",
    "school",
    "region",
    "applicant_region",
    "nationality",
    "applicant_citizenship",
    "citizenship",
    "parsed_gender",
    "parsed_school",
    "parsed_region",
    "parsed_nationality",
    "parsed_applicant_citizenship",
}

MODEL_SELECTION_CV_ROC_AUC_TOLERANCE = 0.005
MODEL_SELECTION_VALIDATION_ROC_AUC_TOLERANCE = 0.003
MODEL_SELECTION_CALIBRATION_ROC_AUC_DROP_LIMIT = 0.002
MODEL_SELECTION_CALIBRATION_AVG_PRECISION_DROP_LIMIT = 0.003
MODEL_SELECTION_CALIBRATION_MIN_BRIER_IMPROVEMENT = 0.001

USE_RARE_CATEGORY_GROUPING = True
RARE_CATEGORY_MIN_FREQUENCY = 10
AUTO_CLASS_WEIGHT = True
CLASS_WEIGHT_TRIGGER = 0.30
RUN_RANDOM_SEARCH = True
RANDOM_SEARCH_ITERATIONS = 12
CALIBRATE_SELECTED_MODEL = True
GENERATE_PLOTS = True

RUN_LEAKAGE_STRESS_TEST = True
SUSPICIOUS_FEATURES = {
    "father_income_document",
    "father_income_document_status",
    "father_income_document_suspicious_flag",
    "mother_income_document",
    "mother_income_document_status",
    "certificate_ownership_clean",
    "certificate_ownership_status",
    "travel_records_clean",
    "travel_records_category",
    "travel_records_missing_flag",
    "faid_missing_documents",
    "faid_missing_documents_count",
}
SUSPICIOUS_FEATURE_KEYWORDS = (
    "income_document",
    "missing_documents",
    "certificate_ownership",
    "travel_records",
)
DEFAULT_WORKFLOW_EXCLUSION_COLUMNS = {
    "father_income_document_status",
    "father_income_document_suspicious_flag",
    "mother_income_document_status",
    "certificate_ownership_clean",
    "certificate_ownership_status",
    "travel_records_clean",
    "travel_records_category",
    "travel_records_missing_flag",
    "faid_missing_documents",
    "faid_missing_documents_count",
}

DROP_HIGH_MISSINGNESS_COLUMNS = True
HIGH_MISSINGNESS_THRESHOLD = 0.985
DROP_NEAR_CONSTANT_COLUMNS = True
NEAR_CONSTANT_SHARE_THRESHOLD = 0.995
DROP_HIGH_CARDINALITY_TEXT_COLUMNS = True
HIGH_CARDINALITY_MIN_UNIQUE = 200
HIGH_CARDINALITY_UNIQUE_RATIO = 0.90
HIGH_CARDINALITY_AVG_LENGTH_THRESHOLD = 18.0
HIGH_CARDINALITY_NAME_HINTS = (
    "name",
    "clean",
    "detail",
    "comment",
    "document",
)

# Explicitly remove post-decision fields from modeling.
LEAKAGE_COLUMNS = {
    "decision",
    "raw_decision",
    "parsed_decision",
    "bin_status",
    "raw_bin_status",
    "parsed_bin_status",
    "need_pct",
    "parsed_need_pct",
    "need_comment",
    "parsed_need_comment",
    "merit_pct",
    "parsed_merit_pct",
    "merit_hist_pct",
    "parsed_merit_hist_pct",
    "merit_comment",
    "parsed_merit_comment",
    "over_and_above_decision",
    "raw_over_and_above_decision",
    "parsed_over_and_above_decision",
    "over_and_above_percentage_awarded",
    "parsed_over_and_above_percentage_awarded",
    "parsed_over_and_above_amount_awarded",
    "has_over_and_above",
    "inferred_has_over_and_above",
}

LEAKAGE_COLUMN_KEYWORDS = (
    "decision",
    "bin_status",
    "need_pct",
    "need_comment",
    "merit_pct",
    "merit_hist_pct",
    "merit_comment",
    "over_and_above",
)

OPTIONAL_DEPENDENCY_STATUS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "available": False,
        "included_in_candidates": False,
        "message": "xgboost availability not checked yet.",
    },
    "matplotlib": {
        "available": False,
        "requested": bool(GENERATE_PLOTS),
        "generated": False,
        "message": "matplotlib availability not checked yet.",
    },
    "shap": {
        "available": False,
        "requested": True,
        "generated": False,
        "message": "SHAP availability not checked yet.",
    },
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    estimator: Any
    tune_param_distributions: dict[str, list[Any]] | None = None


@dataclass(frozen=True)
class ThresholdMetrics:
    label: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    balanced_accuracy: float
    f1: float
    confusion_matrix: list[list[int]]


@dataclass
class ExperimentArtifacts:
    deployed_model: Any | None = None
    explainability_model: Pipeline | None = None
    plot_probabilities: np.ndarray | None = None
    decision_recommendations: pd.DataFrame | None = None


def print_section(title: str) -> None:
    """Render a clean console section header."""
    print(f"\n{title}")
    print("-" * 80)


def resolve_data_path(data_path: Path = DATA_PATH) -> Path:
    """Resolve the first available dataset path from configured candidates."""
    candidate_paths = [data_path, *FALLBACK_DATA_PATHS]
    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidate_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path.resolve()) for path in ordered_candidates)
    raise FileNotFoundError(f"No modeling CSV file was found. Paths checked: {searched}")


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the modeling dataset from disk with basic validation."""
    resolved_path = resolve_data_path(data_path)

    try:
        dataframe = pd.read_csv(resolved_path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV file: {resolved_path.resolve()}") from exc

    if dataframe.empty:
        raise ValueError(f"CSV file is empty: {resolved_path.resolve()}")

    return dataframe


def normalize_label(series: pd.Series) -> pd.Series:
    """Normalize string labels without mutating the raw dataframe."""
    return series.astype("string").str.strip().str.lower()


def normalize_label_values(values: tuple[str, ...] | list[str] | set[str]) -> set[str]:
    """Normalize a label collection into a lower-cased set."""
    return {str(value).strip().lower() for value in values if str(value).strip()}


def resolve_target_column(dataframe: pd.DataFrame, preferred_target: str = TARGET_COLUMN) -> str:
    """Find the target column using a strict preference with safe fallbacks."""
    if preferred_target in dataframe.columns:
        return preferred_target

    case_insensitive_matches = [
        column for column in dataframe.columns if column.strip().lower() == preferred_target.lower()
    ]
    if case_insensitive_matches:
        return case_insensitive_matches[0]

    decision_matches = [
        column for column in dataframe.columns if column.strip().lower() == "decision"
    ]
    if decision_matches:
        return decision_matches[0]

    parsed_decision_matches = [
        column for column in dataframe.columns if column.strip().lower() == "parsed_decision"
    ]
    if parsed_decision_matches:
        return parsed_decision_matches[0]

    raise KeyError(
        f"Target column '{preferred_target}' was not found. "
        f"Available columns: {', '.join(map(str, dataframe.columns))}"
    )


def prepare_target(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    positive_labels: tuple[str, ...] = POSITIVE_TARGET_LABELS,
    negative_labels: tuple[str, ...] = NEGATIVE_TARGET_LABELS,
    excluded_labels: tuple[str, ...] = EXCLUDED_TARGET_LABELS,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """
    Build the binary target.

    Explicit award labels -> 1
    Explicit denial labels -> 0

    Non-standard labels are excluded instead of being silently folded into the
    negative class.

    Missing target rows are dropped because they are unlabeled.
    """
    resolved_target = resolve_target_column(dataframe, target_column)
    raw_target = dataframe[resolved_target].copy()
    normalized_target = normalize_label(raw_target)
    normalized_positive_labels = normalize_label_values(positive_labels)
    normalized_negative_labels = normalize_label_values(negative_labels)
    normalized_excluded_labels = normalize_label_values(excluded_labels)

    if not normalized_positive_labels:
        raise ValueError("At least one positive target label is required.")
    if not normalized_negative_labels:
        raise ValueError("At least one negative target label is required.")
    overlap = normalized_positive_labels & normalized_negative_labels
    if overlap:
        raise ValueError(f"Positive and negative target labels overlap: {sorted(overlap)}")

    labeled_mask = normalized_target.notna() & normalized_target.fillna("").ne("")
    labeled_rows = int(labeled_mask.sum())
    dropped_unlabeled_rows = int((~labeled_mask).sum())

    if labeled_rows == 0:
        raise ValueError("No labeled rows remain after excluding missing target values.")

    labeled_target = normalized_target.loc[labeled_mask]
    allowed_labels = normalized_positive_labels | normalized_negative_labels
    recognized_mask = labeled_target.isin(allowed_labels)
    explicitly_excluded_mask = labeled_target.isin(normalized_excluded_labels)
    ambiguous_mask = ~recognized_mask

    if ALLOW_ONLY_EXPLICIT_TARGET_LABELS:
        training_mask = labeled_mask.copy()
        training_mask.loc[labeled_mask] = recognized_mask.to_numpy()
    else:
        training_mask = labeled_mask.copy()

    training_rows = int(training_mask.sum())
    dropped_non_binary_rows = int(labeled_rows - training_rows)
    if training_rows == 0:
        raise ValueError("No explicit binary target rows remain after label filtering.")

    training_dataframe = dataframe.loc[training_mask].copy()
    training_target_labels = normalized_target.loc[training_mask]
    binary_target = training_target_labels.isin(normalized_positive_labels).astype(int)

    metadata = {
        "target_column": resolved_target,
        "positive_labels": sorted(normalized_positive_labels),
        "negative_labels": sorted(normalized_negative_labels),
        "excluded_labels": sorted(normalized_excluded_labels),
        "target_definition_note": (
            "Only explicit award and denial outcomes are used for binary training. "
            "Rows with unlabeled, ambiguous, or structurally different outcomes remain out of the training target."
        ),
        "excluded_label_rationale": (
            "Excluded labels represent programs or outcomes that are not directly comparable to the core "
            "eligibility decision, so they are removed rather than forced into the negative class."
        ),
        "rows_total": int(len(dataframe)),
        "rows_labeled": labeled_rows,
        "rows_dropped_missing_target": dropped_unlabeled_rows,
        "rows_used_for_training": training_rows,
        "rows_dropped_non_binary_target": dropped_non_binary_rows,
        "target_distribution": binary_target.value_counts().sort_index().to_dict(),
        "target_positive_rate": float(binary_target.mean()),
        "raw_target_value_counts": labeled_target.value_counts().to_dict(),
        "training_target_value_counts": training_target_labels.value_counts().to_dict(),
        "explicitly_excluded_target_value_counts": labeled_target.loc[
            explicitly_excluded_mask
        ].value_counts().to_dict(),
        "ambiguous_target_value_counts": labeled_target.loc[
            ambiguous_mask & ~explicitly_excluded_mask
        ].value_counts().to_dict(),
    }
    return training_dataframe, binary_target, metadata


def resolve_available_columns(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...] | list[str] | set[str],
) -> list[str]:
    """Return the subset of requested columns that exists in the dataframe."""
    return [column for column in candidates if column in dataframe.columns]


def resolve_model_exclusion_summary(
    dataframe: pd.DataFrame,
    target_column: str,
    leakage_columns: set[str] = LEAKAGE_COLUMNS,
) -> dict[str, Any]:
    """Separate model exclusions into governance-friendly categories."""
    target_columns = [column for column in dataframe.columns if column == target_column]
    post_decision_columns = sorted(
        column
        for column in dataframe.columns
        if (
            column != target_column
            and (
                column in leakage_columns
                or any(keyword in column.lower() for keyword in LEAKAGE_COLUMN_KEYWORDS)
            )
        )
    )
    identifier_columns = sorted(resolve_available_columns(dataframe, IDENTIFIER_COLUMNS))
    audit_only_columns = sorted(resolve_available_columns(dataframe, AUDIT_ONLY_COLUMNS))

    columns_to_drop = sorted(
        set(target_columns + post_decision_columns + identifier_columns + audit_only_columns)
    )
    return {
        "target_columns": target_columns,
        "post_decision_columns": post_decision_columns,
        "identifier_columns": identifier_columns,
        "audit_only_columns": audit_only_columns,
        "columns_to_drop": columns_to_drop,
    }


def select_feature_frame(
    dataframe: pd.DataFrame,
    target_column: str,
    leakage_columns: set[str] = LEAKAGE_COLUMNS,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop the target and explicitly post-outcome columns."""
    exclusion_summary = resolve_model_exclusion_summary(
        dataframe=dataframe,
        target_column=target_column,
        leakage_columns=leakage_columns,
    )
    columns_to_drop = exclusion_summary["columns_to_drop"]
    feature_frame = dataframe.drop(columns=columns_to_drop, errors="ignore").copy()

    if feature_frame.empty:
        raise ValueError("No feature columns remain after excluding target/leakage columns.")

    return feature_frame, columns_to_drop


def build_audit_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep identifiers and fairness/audit columns out of the model but available for review."""
    audit_candidates = tuple(sorted(IDENTIFIER_COLUMNS)) + tuple(FAIRNESS_AUDIT_COLUMNS)
    audit_columns = []
    for column in audit_candidates:
        if column in dataframe.columns and column not in audit_columns:
            audit_columns.append(column)
    if not audit_columns:
        return pd.DataFrame(index=dataframe.index)
    return dataframe.loc[:, audit_columns].copy()


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create a train/validation/test split with stratification."""
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError(
            "validation_size and test_size must be positive and sum to less than 1."
        )

    features_train_valid, features_test, target_train_valid, target_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    relative_validation_size = validation_size / (1.0 - test_size)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features_train_valid,
        target_train_valid,
        test_size=relative_validation_size,
        random_state=RANDOM_STATE,
        stratify=target_train_valid,
    )

    return (
        features_train,
        features_valid,
        features_test,
        target_train,
        target_valid,
        target_test,
    )


def supports_parameter(callable_obj: Any, parameter_name: str) -> bool:
    """Safely check whether an estimator or transformer accepts a parameter."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return parameter_name in signature.parameters


def numeric_or_nan(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric version of a column or an all-NaN series if it is missing."""
    if column not in dataframe.columns:
        return pd.Series(np.nan, index=dataframe.index, dtype=float)
    return pd.to_numeric(dataframe[column], errors="coerce")


def numeric_or_zero(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column with missing values filled by zero."""
    return numeric_or_nan(dataframe, column).fillna(0.0)


def coalesce_numeric_series(*series_list: pd.Series) -> pd.Series:
    """Use the first non-missing value across a list of numeric series."""
    if not series_list:
        raise ValueError("At least one series is required for coalescing.")

    combined = series_list[0].copy()
    for series in series_list[1:]:
        combined = combined.fillna(series)
    return combined.fillna(0.0)


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series | float,
    offset: float = 1.0,
    minimum_denominator: float = 1.0,
) -> pd.Series:
    """Safely divide two values with configurable denominator protection."""
    numerator_series = numerator.fillna(0.0).astype(float)

    if isinstance(denominator, pd.Series):
        denominator_series = denominator.fillna(0.0).astype(float) + float(offset)
        denominator_series = denominator_series.clip(lower=minimum_denominator)
        return numerator_series / denominator_series

    protected_denominator = max(float(denominator) + float(offset), minimum_denominator)
    return numerator_series / protected_denominator


def has_any_columns(dataframe: pd.DataFrame, columns: list[str]) -> bool:
    """Return True when at least one of the requested columns exists."""
    return any(column in dataframe.columns for column in columns)


def build_domain_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add conservative domain features without mutating the input dataframe.

    Features are only created when their source columns exist.
    """
    features = dataframe.copy()
    existing_columns = set(features.columns)

    father_income = coalesce_numeric_series(
        numeric_or_nan(features, "father_gross_income"),
        numeric_or_nan(features, "father_net_income"),
    )
    mother_income = coalesce_numeric_series(
        numeric_or_nan(features, "mother_gross_income"),
        numeric_or_nan(features, "mother_net_income"),
    )
    investment_income = numeric_or_zero(features, "investments_total_annual_profit")
    rented_income = numeric_or_zero(features, "properties_rented_income_total")
    planted_income = numeric_or_zero(features, "properties_planted_income_total")
    assistant_support = numeric_or_zero(features, "financial_assistants_total_est_annual_amount")
    dependents_count = numeric_or_zero(features, "dependents_count")
    siblings_at_aub = numeric_or_zero(features, "siblings_at_aub_count")
    siblings_not_at_aub = numeric_or_zero(features, "siblings_not_at_aub_count")
    cars_count = numeric_or_zero(features, "cars_count")
    property_value = numeric_or_zero(features, "properties_total_estimated_value")
    property_area = numeric_or_zero(features, "properties_total_area")
    siblings_tuition = numeric_or_zero(features, "siblings_other_total_tuition")
    siblings_assistance = numeric_or_zero(features, "siblings_other_total_financial_assistance")
    loans_balance = numeric_or_zero(features, "loans_total_remaining_balance")
    loans_monthly_payment = numeric_or_zero(features, "loans_total_monthly_payment")

    parent_income_source_columns = [
        "father_gross_income",
        "father_net_income",
        "mother_gross_income",
        "mother_net_income",
    ]
    non_salary_income_columns = [
        "investments_total_annual_profit",
        "properties_rented_income_total",
        "properties_planted_income_total",
    ]
    household_size_columns = [
        "dependents_count",
        "siblings_at_aub_count",
        "siblings_not_at_aub_count",
    ]

    if has_any_columns(features, household_size_columns):
        features["household_size_proxy"] = (
            dependents_count + siblings_at_aub + siblings_not_at_aub + 1.0
        )

    if has_any_columns(features, parent_income_source_columns):
        features["parental_income_total"] = father_income + mother_income

    if has_any_columns(features, parent_income_source_columns + non_salary_income_columns):
        features["total_household_income_est"] = (
            father_income + mother_income + investment_income + rented_income + planted_income
        )

    if has_any_columns(
        features,
        parent_income_source_columns
        + non_salary_income_columns
        + ["financial_assistants_total_est_annual_amount"],
    ):
        features["total_household_support_est"] = (
            father_income
            + mother_income
            + investment_income
            + rented_income
            + planted_income
            + assistant_support
        )

    if {"parental_income_total", "household_size_proxy"}.issubset(features.columns):
        features["income_per_person"] = safe_divide(
            features["parental_income_total"],
            features["household_size_proxy"],
            offset=0.0,
            minimum_denominator=1.0,
        )

    if {"total_household_income_est", "household_size_proxy"}.issubset(features.columns):
        features["total_household_income_per_person"] = safe_divide(
            features["total_household_income_est"],
            features["household_size_proxy"],
            offset=0.0,
            minimum_denominator=1.0,
        )

    if {"total_household_support_est", "household_size_proxy"}.issubset(features.columns):
        features["support_per_person"] = safe_divide(
            features["total_household_support_est"],
            features["household_size_proxy"],
            offset=0.0,
            minimum_denominator=1.0,
        )

    if has_any_columns(features, ["siblings_other_total_tuition"]) and "total_household_support_est" in features.columns:
        features["tuition_burden_ratio"] = safe_divide(
            siblings_tuition,
            features["total_household_support_est"],
            offset=1.0,
            minimum_denominator=1.0,
        )

    if (
        has_any_columns(
            features,
            [
                "siblings_other_total_tuition",
                "siblings_other_total_financial_assistance",
            ],
        )
        and "total_household_support_est" in features.columns
    ):
        net_siblings_tuition = (siblings_tuition - siblings_assistance).clip(lower=0.0)
        features["siblings_tuition_burden_ratio"] = safe_divide(
            net_siblings_tuition,
            features["total_household_support_est"],
            offset=1.0,
            minimum_denominator=1.0,
        )

    if has_any_columns(features, ["properties_total_estimated_value"]) and "total_household_income_est" in features.columns:
        features["assets_to_income_ratio"] = safe_divide(
            property_value,
            features["total_household_income_est"],
            offset=1.0,
            minimum_denominator=1.0,
        )

    if has_any_columns(features, ["cars_count"]) and "household_size_proxy" in features.columns:
        features["cars_per_person"] = safe_divide(
            cars_count,
            features["household_size_proxy"],
            offset=0.0,
            minimum_denominator=1.0,
        )

    if has_any_columns(features, ["properties_total_estimated_value"]) and "household_size_proxy" in features.columns:
        features["property_value_per_person"] = safe_divide(
            property_value,
            features["household_size_proxy"],
            offset=0.0,
            minimum_denominator=1.0,
        )

    if has_any_columns(features, ["properties_total_area"]) and "household_size_proxy" in features.columns:
        features["property_area_per_person"] = safe_divide(
            property_area,
            features["household_size_proxy"],
            offset=0.0,
            minimum_denominator=1.0,
        )

    if has_any_columns(features, ["loans_total_monthly_payment"]) and "total_household_income_est" in features.columns:
        features["loan_payment_to_income_ratio"] = safe_divide(
            loans_monthly_payment * 12.0,
            features["total_household_income_est"],
            offset=1.0,
            minimum_denominator=1.0,
        )

    if has_any_columns(features, ["loans_total_remaining_balance"]) and "total_household_income_est" in features.columns:
        features["loan_balance_to_income_ratio"] = safe_divide(
            loans_balance,
            features["total_household_income_est"],
            offset=1.0,
            minimum_denominator=1.0,
        )

    new_columns = [column for column in features.columns if column not in existing_columns]
    if new_columns:
        ordered_new_columns = [column for column in features.columns if column in new_columns]
        features = features[[column for column in features.columns if column not in new_columns] + ordered_new_columns]

    return features


def fit_feature_pruner(features_train: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    """
    Fit conservative feature-pruning rules on the training split only.

    This is intentionally lightweight and designed to remove obvious noise rather than
    aggressively optimize the feature set.
    """
    high_missingness_columns: list[str] = []
    near_constant_columns: list[str] = []
    high_cardinality_columns: list[str] = []

    for column in features_train.columns:
        series = features_train[column]

        if DROP_HIGH_MISSINGNESS_COLUMNS:
            missing_rate = float(series.isna().mean())
            if missing_rate >= HIGH_MISSINGNESS_THRESHOLD:
                high_missingness_columns.append(column)
                continue

        if DROP_NEAR_CONSTANT_COLUMNS:
            value_counts = series.astype("string").fillna("__missing__").value_counts(
                normalize=True,
                dropna=False,
            )
            top_share = float(value_counts.iloc[0]) if not value_counts.empty else 1.0
            unique_values = int(series.nunique(dropna=False))
            if unique_values <= 1 or top_share >= NEAR_CONSTANT_SHARE_THRESHOLD:
                near_constant_columns.append(column)
                continue

        if DROP_HIGH_CARDINALITY_TEXT_COLUMNS and not pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if non_null.empty:
                continue

            unique_count = int(non_null.nunique())
            unique_ratio = unique_count / max(len(non_null), 1)
            avg_length = float(non_null.astype(str).str.len().mean())
            name_hint = any(token in column.lower() for token in HIGH_CARDINALITY_NAME_HINTS)

            if (
                unique_count >= HIGH_CARDINALITY_MIN_UNIQUE
                and unique_ratio >= HIGH_CARDINALITY_UNIQUE_RATIO
                and (avg_length >= HIGH_CARDINALITY_AVG_LENGTH_THRESHOLD or name_hint)
            ):
                high_cardinality_columns.append(column)

    columns_to_drop = sorted(
        set(high_missingness_columns + near_constant_columns + high_cardinality_columns)
    )

    summary = {
        "high_missingness_columns": high_missingness_columns,
        "near_constant_columns": near_constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "columns_dropped_total": len(columns_to_drop),
    }
    return columns_to_drop, summary


def apply_feature_pruning(
    dataframe: pd.DataFrame,
    columns_to_drop: list[str],
    retained_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Apply pre-fitted feature-pruning rules and align column order when requested."""
    pruned = dataframe.drop(columns=columns_to_drop, errors="ignore").copy()
    if retained_columns is not None:
        pruned = pruned.reindex(columns=retained_columns)
    return pruned


def build_numeric_imputer() -> SimpleImputer:
    """Median imputation with missing-value indicators when supported."""
    params: dict[str, Any] = {"strategy": "median", "add_indicator": True}
    if supports_parameter(SimpleImputer, "keep_empty_features"):
        params["keep_empty_features"] = True
    return SimpleImputer(**params)


def build_categorical_imputer() -> SimpleImputer:
    """Constant imputation is robust even when an entire column is missing."""
    params: dict[str, Any] = {"strategy": "constant", "fill_value": "__missing__"}
    if supports_parameter(SimpleImputer, "keep_empty_features"):
        params["keep_empty_features"] = True
    return SimpleImputer(**params)


def build_one_hot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder so every candidate model can use the same output."""
    params: dict[str, Any] = {"handle_unknown": "ignore"}

    if USE_RARE_CATEGORY_GROUPING and supports_parameter(OneHotEncoder, "min_frequency"):
        params["min_frequency"] = RARE_CATEGORY_MIN_FREQUENCY

    if supports_parameter(OneHotEncoder, "sparse_output"):
        params["sparse_output"] = False
    else:
        params["sparse"] = False

    return OneHotEncoder(**params)


def infer_column_types(feature_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric vs categorical columns using the training split."""
    numeric_columns = feature_frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in feature_frame.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build a ColumnTransformer for either linear or tree-based models."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [("imputer", build_numeric_imputer())]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(steps=numeric_steps)
        transformers.append(("num", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", build_categorical_imputer()),
                ("encoder", build_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable numeric or categorical feature columns were detected.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_preprocessors(
    feature_frame: pd.DataFrame,
) -> tuple[dict[str, ColumnTransformer], list[str], list[str]]:
    """Build separate preprocessors for linear and tree-based model families."""
    numeric_columns, categorical_columns = infer_column_types(feature_frame)
    preprocessors = {
        "linear": build_preprocessor(numeric_columns, categorical_columns, scale_numeric=True),
        "tree": build_preprocessor(numeric_columns, categorical_columns, scale_numeric=False),
    }
    return preprocessors, numeric_columns, categorical_columns


def get_recommended_class_weight(target: pd.Series) -> str | None:
    """Use balanced class weights only when imbalance is meaningful."""
    if not AUTO_CLASS_WEIGHT:
        return None

    positive_rate = float(target.mean())
    minority_rate = min(positive_rate, 1.0 - positive_rate)
    return "balanced" if minority_rate < CLASS_WEIGHT_TRIGGER else None


def get_recommended_scale_pos_weight(target: pd.Series) -> float:
    """Provide a mild imbalance-aware XGBoost weight when needed."""
    positive_count = float(target.sum())
    negative_count = float(len(target) - positive_count)
    if positive_count <= 0:
        return 1.0

    ratio = negative_count / positive_count
    return ratio if ratio > 1.25 else 1.0


def get_model_specs(
    class_weight: str | None = None,
    scale_pos_weight: float = 1.0,
) -> dict[str, ModelSpec]:
    """Build the candidate model registry."""
    OPTIONAL_DEPENDENCY_STATUS["xgboost"] = {
        "available": False,
        "included_in_candidates": False,
        "message": "xgboost not checked for this run.",
    }

    hist_kwargs: dict[str, Any] = {
        "l2_regularization": 1.0,
        "learning_rate": 0.03,
        "max_depth": 4,
        "max_iter": 300,
        "max_leaf_nodes": 23,
        "min_samples_leaf": 40,
        "random_state": RANDOM_STATE,
    }
    if class_weight is not None and supports_parameter(HistGradientBoostingClassifier, "class_weight"):
        hist_kwargs["class_weight"] = class_weight

    model_specs: dict[str, ModelSpec] = {
        "Logistic Regression": ModelSpec(
            name="Logistic Regression",
            family="linear",
            estimator=LogisticRegression(
                C=0.50,
                class_weight=class_weight,
                max_iter=4000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            ),
        ),
        "Random Forest": ModelSpec(
            name="Random Forest",
            family="tree",
            estimator=RandomForestClassifier(
                bootstrap=True,
                class_weight=class_weight,
                max_depth=10,
                max_features=0.50,
                min_samples_leaf=6,
                min_samples_split=12,
                n_estimators=400,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
            ),
            tune_param_distributions={
                "model__n_estimators": [300, 500, 700],
                "model__max_depth": [6, 8, 10, 12, 14],
                "model__min_samples_leaf": [4, 6, 8, 12],
                "model__min_samples_split": [8, 12, 20],
                "model__max_features": [0.30, 0.50, "sqrt"],
            },
        ),
        "Extra Trees": ModelSpec(
            name="Extra Trees",
            family="tree",
            estimator=ExtraTreesClassifier(
                bootstrap=False,
                class_weight=class_weight,
                max_depth=12,
                max_features=0.50,
                min_samples_leaf=4,
                min_samples_split=10,
                n_estimators=500,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
            ),
            tune_param_distributions={
                "model__n_estimators": [300, 500, 700],
                "model__max_depth": [8, 10, 12, 16, None],
                "model__min_samples_leaf": [2, 4, 6, 10],
                "model__min_samples_split": [4, 8, 10, 16],
                "model__max_features": [0.30, 0.50, "sqrt"],
            },
        ),
        "Gradient Boosting": ModelSpec(
            name="Gradient Boosting",
            family="tree",
            estimator=GradientBoostingClassifier(
                learning_rate=0.03,
                max_depth=2,
                max_features=0.60,
                min_samples_leaf=30,
                min_samples_split=30,
                n_estimators=200,
                random_state=RANDOM_STATE,
                subsample=0.75,
            ),
        ),
        "HistGradientBoosting": ModelSpec(
            name="HistGradientBoosting",
            family="tree",
            estimator=HistGradientBoostingClassifier(**hist_kwargs),
            tune_param_distributions={
                "model__learning_rate": [0.02, 0.03, 0.05],
                "model__max_depth": [3, 4, 5],
                "model__max_iter": [200, 300, 400],
                "model__max_leaf_nodes": [15, 23, 31],
                "model__min_samples_leaf": [30, 40, 60, 80],
                "model__l2_regularization": [0.5, 1.0, 2.0, 5.0],
            },
        ),
    }

    try:
        from xgboost import XGBClassifier

        OPTIONAL_DEPENDENCY_STATUS["xgboost"] = {
            "available": True,
            "included_in_candidates": True,
            "message": "XGBoost candidate enabled.",
        }
        model_specs["XGBoost"] = ModelSpec(
            name="XGBoost",
            family="tree",
            estimator=XGBClassifier(
                colsample_bytree=0.70,
                eval_metric="logloss",
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=6,
                n_estimators=350,
                n_jobs=N_JOBS,
                objective="binary:logistic",
                random_state=RANDOM_STATE,
                reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight,
                subsample=0.75,
                tree_method="hist",
                verbosity=0,
            ),
            tune_param_distributions={
                "model__n_estimators": [250, 350, 500],
                "model__max_depth": [2, 3, 4, 5],
                "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
                "model__subsample": [0.65, 0.75, 0.85],
                "model__colsample_bytree": [0.60, 0.70, 0.85],
                "model__min_child_weight": [4, 6, 8, 12],
                "model__reg_lambda": [1.0, 2.0, 5.0, 10.0],
            },
        )
    except Exception as exc:
        OPTIONAL_DEPENDENCY_STATUS["xgboost"] = {
            "available": False,
            "included_in_candidates": False,
            "message": (
                f"XGBoost unavailable, so the optional XGBoost candidate was skipped: {exc}. "
                "Install it with `pip install xgboost` to enable it."
            ),
        }
        warnings.warn(
            "XGBoost is not installed; skipping the optional XGBoost model. "
            f"Reason: {exc}. Install it with `pip install xgboost` to enable it.",
            stacklevel=2,
        )

    return model_specs


def make_model_pipeline(preprocessors: dict[str, ColumnTransformer], model_spec: ModelSpec) -> Pipeline:
    """Create a fresh sklearn Pipeline for a model specification."""
    if model_spec.family not in preprocessors:
        raise KeyError(f"Missing preprocessor for model family '{model_spec.family}'.")

    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessors[model_spec.family])),
            ("model", clone(model_spec.estimator)),
        ]
    )


def build_scoring() -> dict[str, Any]:
    """Create the scoring dictionary used for CV and randomized search."""
    return {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "neg_brier_score": "neg_brier_score",
    }


def sort_model_comparison_table(comparison_table: pd.DataFrame) -> pd.DataFrame:
    """Sort successful models first and rank them with an overfitting-aware tie-break."""
    successful = comparison_table[comparison_table["status"] == "ok"].copy()
    failed = comparison_table[comparison_table["status"] != "ok"].copy()

    if not successful.empty:
        ascending_map = {
            "validation_roc_auc": False,
            "cv_roc_auc_mean": False,
            "cv_f1_mean": False,
            "overfit_gap_roc_auc": True,
        }
        sort_columns = [
            column
            for column in [
                "validation_roc_auc",
                "cv_roc_auc_mean",
                "cv_f1_mean",
                "overfit_gap_roc_auc",
            ]
            if column in successful.columns
        ]
        ascending = [ascending_map[column] for column in sort_columns]
        successful = successful.sort_values(
            by=sort_columns,
            ascending=ascending,
            ignore_index=True,
        )

    if not failed.empty:
        failed = failed.sort_values(by=["source", "model"], ignore_index=True)

    return pd.concat([successful, failed], ignore_index=True)


def evaluate_models(
    model_specs: dict[str, ModelSpec],
    preprocessors: dict[str, ColumnTransformer],
    features_train: pd.DataFrame,
    target_train: pd.Series,
    cv: StratifiedKFold,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Cross-validate all base models and return comparison rows plus pipeline prototypes."""
    scoring = build_scoring()
    rows: list[dict[str, Any]] = []
    prototypes: dict[str, Pipeline] = {}

    for model_name, model_spec in model_specs.items():
        pipeline = make_model_pipeline(preprocessors, model_spec)
        prototypes[model_name] = clone(pipeline)

        try:
            cv_results = cross_validate(
                estimator=pipeline,
                X=features_train,
                y=target_train,
                cv=cv,
                error_score="raise",
                n_jobs=N_JOBS,
                return_train_score=True,
                scoring=scoring,
            )

            train_roc_auc_mean = float(np.mean(cv_results["train_roc_auc"]))
            cv_roc_auc_mean = float(np.mean(cv_results["test_roc_auc"]))
            train_f1_mean = float(np.mean(cv_results["train_f1"]))
            cv_f1_mean = float(np.mean(cv_results["test_f1"]))

            rows.append(
                {
                    "model": model_name,
                    "family": model_spec.family,
                    "source": "base",
                    "status": "ok",
                    "train_roc_auc_mean": train_roc_auc_mean,
                    "cv_roc_auc_mean": cv_roc_auc_mean,
                    "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
                    "overfit_gap_roc_auc": train_roc_auc_mean - cv_roc_auc_mean,
                    "train_f1_mean": train_f1_mean,
                    "cv_f1_mean": cv_f1_mean,
                    "overfit_gap_f1": train_f1_mean - cv_f1_mean,
                    "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
                    "cv_precision_mean": float(np.mean(cv_results["test_precision"])),
                    "cv_recall_mean": float(np.mean(cv_results["test_recall"])),
                    "train_brier_mean": float(-np.mean(cv_results["train_neg_brier_score"])),
                    "cv_brier_mean": float(-np.mean(cv_results["test_neg_brier_score"])),
                    "best_params": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "family": model_spec.family,
                    "source": "base",
                    "status": f"failed: {exc}",
                    "train_roc_auc_mean": np.nan,
                    "cv_roc_auc_mean": np.nan,
                    "cv_roc_auc_std": np.nan,
                    "overfit_gap_roc_auc": np.nan,
                    "train_f1_mean": np.nan,
                    "cv_f1_mean": np.nan,
                    "overfit_gap_f1": np.nan,
                    "cv_accuracy_mean": np.nan,
                    "cv_precision_mean": np.nan,
                    "cv_recall_mean": np.nan,
                    "train_brier_mean": np.nan,
                    "cv_brier_mean": np.nan,
                    "best_params": None,
                }
            )

    comparison_table = pd.DataFrame(rows)
    if comparison_table[comparison_table["status"] == "ok"].empty:
        raise RuntimeError("Every candidate model failed during cross-validation.")

    return sort_model_comparison_table(comparison_table), prototypes


def run_random_search(
    model_specs: dict[str, ModelSpec],
    preprocessors: dict[str, ColumnTransformer],
    features_train: pd.DataFrame,
    target_train: pd.Series,
    cv: StratifiedKFold,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Tune a small set of strong models with computationally reasonable searches."""
    if not RUN_RANDOM_SEARCH:
        return pd.DataFrame(), {}

    scoring = build_scoring()
    rows: list[dict[str, Any]] = []
    tuned_prototypes: dict[str, Pipeline] = {}

    for model_name, model_spec in model_specs.items():
        if not model_spec.tune_param_distributions:
            continue

        display_name = f"{model_name} (tuned)"
        search = RandomizedSearchCV(
            estimator=make_model_pipeline(preprocessors, model_spec),
            param_distributions=model_spec.tune_param_distributions,
            cv=cv,
            error_score="raise",
            n_iter=RANDOM_SEARCH_ITERATIONS,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            refit="roc_auc",
            return_train_score=True,
            scoring=scoring,
        )

        try:
            search.fit(features_train, target_train)
            best_index = int(search.best_index_)
            cv_results = search.cv_results_

            train_roc_auc_mean = float(cv_results["mean_train_roc_auc"][best_index])
            cv_roc_auc_mean = float(cv_results["mean_test_roc_auc"][best_index])
            train_f1_mean = float(cv_results["mean_train_f1"][best_index])
            cv_f1_mean = float(cv_results["mean_test_f1"][best_index])

            rows.append(
                {
                    "model": display_name,
                    "family": model_spec.family,
                    "source": "tuned",
                    "status": "ok",
                    "train_roc_auc_mean": train_roc_auc_mean,
                    "cv_roc_auc_mean": cv_roc_auc_mean,
                    "cv_roc_auc_std": float(cv_results["std_test_roc_auc"][best_index]),
                    "overfit_gap_roc_auc": train_roc_auc_mean - cv_roc_auc_mean,
                    "train_f1_mean": train_f1_mean,
                    "cv_f1_mean": cv_f1_mean,
                    "overfit_gap_f1": train_f1_mean - cv_f1_mean,
                    "cv_accuracy_mean": float(cv_results["mean_test_accuracy"][best_index]),
                    "cv_precision_mean": float(cv_results["mean_test_precision"][best_index]),
                    "cv_recall_mean": float(cv_results["mean_test_recall"][best_index]),
                    "train_brier_mean": float(-cv_results["mean_train_neg_brier_score"][best_index]),
                    "cv_brier_mean": float(-cv_results["mean_test_neg_brier_score"][best_index]),
                    "best_params": search.best_params_,
                }
            )
            tuned_prototypes[display_name] = clone(search.best_estimator_)
        except Exception as exc:
            rows.append(
                {
                    "model": display_name,
                    "family": model_spec.family,
                    "source": "tuned",
                    "status": f"failed: {exc}",
                    "train_roc_auc_mean": np.nan,
                    "cv_roc_auc_mean": np.nan,
                    "cv_roc_auc_std": np.nan,
                    "overfit_gap_roc_auc": np.nan,
                    "train_f1_mean": np.nan,
                    "cv_f1_mean": np.nan,
                    "overfit_gap_f1": np.nan,
                    "cv_accuracy_mean": np.nan,
                    "cv_precision_mean": np.nan,
                    "cv_recall_mean": np.nan,
                    "train_brier_mean": np.nan,
                    "cv_brier_mean": np.nan,
                    "best_params": None,
                }
            )

    tuned_table = pd.DataFrame(rows)
    if tuned_table.empty:
        return tuned_table, tuned_prototypes

    return sort_model_comparison_table(tuned_table), tuned_prototypes


def fit_model_prototype(model_prototype: Any, features: pd.DataFrame, target: pd.Series) -> Any:
    """Clone and fit a pipeline or estimator prototype."""
    fitted_model = clone(model_prototype)
    fitted_model.fit(features, target)
    return fitted_model


def fit_calibrated_model(model_prototype: Any, features: pd.DataFrame, target: pd.Series) -> Any:
    """Fit a calibrated wrapper around a model prototype using internal CV."""
    calibration_params: dict[str, Any] = {"cv": 3, "method": "sigmoid"}
    base_model = clone(model_prototype)

    if supports_parameter(CalibratedClassifierCV, "estimator"):
        calibration_params["estimator"] = base_model
    else:
        calibration_params["base_estimator"] = base_model

    calibrated_model = CalibratedClassifierCV(**calibration_params)
    calibrated_model.fit(features, target)
    return calibrated_model


def get_positive_class_probabilities(fitted_model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for any compatible classifier."""
    if hasattr(fitted_model, "predict_proba"):
        probabilities = fitted_model.predict_proba(features)[:, 1]
        return np.asarray(probabilities, dtype=float)

    if hasattr(fitted_model, "decision_function"):
        decision_scores = np.asarray(fitted_model.decision_function(features), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision_scores))

    raise AttributeError("The fitted model does not expose predict_proba or decision_function.")


def safe_metric(metric_func: Any, *args: Any, **kwargs: Any) -> float:
    """Safely compute a scalar metric and return NaN on failure."""
    try:
        return float(metric_func(*args, **kwargs))
    except Exception:
        return float("nan")


def compute_probability_metrics(target_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    """Compute probability-based metrics that do not require a threshold."""
    return {
        "roc_auc": safe_metric(roc_auc_score, target_true, probabilities),
        "average_precision": safe_metric(average_precision_score, target_true, probabilities),
        "brier_score": safe_metric(brier_score_loss, target_true, probabilities),
    }


def compute_threshold_metrics(
    target_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    label: str,
) -> ThresholdMetrics:
    """Compute threshold-dependent classification metrics."""
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(target_true, predictions, labels=[0, 1]).tolist()
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    specificity_denominator = true_negative + false_positive
    specificity = (
        float(true_negative / specificity_denominator)
        if specificity_denominator > 0
        else 0.0
    )
    recall = float(recall_score(target_true, predictions, zero_division=0))

    return ThresholdMetrics(
        label=label,
        threshold=float(threshold),
        accuracy=float(accuracy_score(target_true, predictions)),
        precision=float(precision_score(target_true, predictions, zero_division=0)),
        recall=recall,
        specificity=specificity,
        balanced_accuracy=float((recall + specificity) / 2.0),
        f1=float(f1_score(target_true, predictions, zero_division=0)),
        confusion_matrix=matrix,
    )


def evaluate_candidates_on_holdout(
    candidate_prototypes: dict[str, Pipeline],
    comparison_table: pd.DataFrame,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_holdout: pd.DataFrame,
    target_holdout: pd.Series,
    *,
    holdout_label: str,
) -> pd.DataFrame:
    """Fit all candidate pipelines on the training split and score them on a holdout split."""
    rows: list[dict[str, Any]] = []

    comparison_lookup = comparison_table.set_index("model")
    successful_models = set(comparison_table.loc[comparison_table["status"] == "ok", "model"])

    for model_name, model_prototype in candidate_prototypes.items():
        if model_name not in successful_models:
            continue

        try:
            fitted_model = fit_model_prototype(model_prototype, features_train, target_train)
            probabilities = get_positive_class_probabilities(fitted_model, features_holdout)
            probability_metrics = compute_probability_metrics(target_holdout, probabilities)
            default_metrics = compute_threshold_metrics(
                target_holdout,
                probabilities,
                threshold=0.50,
                label="default_0.50",
            )

            base_row = comparison_lookup.loc[model_name]
            rows.append(
                {
                    "model": model_name,
                    "family": base_row["family"],
                    "source": base_row["source"],
                    "status": "ok",
                    f"{holdout_label}_roc_auc": probability_metrics["roc_auc"],
                    f"{holdout_label}_average_precision": probability_metrics["average_precision"],
                    f"{holdout_label}_brier_score": probability_metrics["brier_score"],
                    f"{holdout_label}_accuracy_0.50": default_metrics.accuracy,
                    f"{holdout_label}_precision_0.50": default_metrics.precision,
                    f"{holdout_label}_recall_0.50": default_metrics.recall,
                    f"{holdout_label}_f1_0.50": default_metrics.f1,
                    "train_roc_auc_mean": base_row["train_roc_auc_mean"],
                    "cv_roc_auc_mean": base_row["cv_roc_auc_mean"],
                    "cv_roc_auc_std": base_row["cv_roc_auc_std"],
                    "overfit_gap_roc_auc": base_row["overfit_gap_roc_auc"],
                    "train_f1_mean": base_row["train_f1_mean"],
                    "cv_f1_mean": base_row["cv_f1_mean"],
                    "overfit_gap_f1": base_row["overfit_gap_f1"],
                    "cv_brier_mean": base_row["cv_brier_mean"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "family": comparison_lookup.loc[model_name, "family"],
                    "source": comparison_lookup.loc[model_name, "source"],
                    "status": f"failed: {exc}",
                    f"{holdout_label}_roc_auc": np.nan,
                    f"{holdout_label}_average_precision": np.nan,
                    f"{holdout_label}_brier_score": np.nan,
                    f"{holdout_label}_accuracy_0.50": np.nan,
                    f"{holdout_label}_precision_0.50": np.nan,
                    f"{holdout_label}_recall_0.50": np.nan,
                    f"{holdout_label}_f1_0.50": np.nan,
                    "train_roc_auc_mean": comparison_lookup.loc[model_name, "train_roc_auc_mean"],
                    "cv_roc_auc_mean": comparison_lookup.loc[model_name, "cv_roc_auc_mean"],
                    "cv_roc_auc_std": comparison_lookup.loc[model_name, "cv_roc_auc_std"],
                    "overfit_gap_roc_auc": comparison_lookup.loc[model_name, "overfit_gap_roc_auc"],
                    "train_f1_mean": comparison_lookup.loc[model_name, "train_f1_mean"],
                    "cv_f1_mean": comparison_lookup.loc[model_name, "cv_f1_mean"],
                    "overfit_gap_f1": comparison_lookup.loc[model_name, "overfit_gap_f1"],
                    "cv_brier_mean": comparison_lookup.loc[model_name, "cv_brier_mean"],
                }
            )

    holdout_table = pd.DataFrame(rows)
    if holdout_table.empty:
        raise RuntimeError(f"No candidate models were available for the {holdout_label} holdout set.")

    successful = holdout_table[holdout_table["status"] == "ok"].copy()
    failed = holdout_table[holdout_table["status"] != "ok"].copy()

    if not successful.empty:
        successful = successful.sort_values(
            by=[
                f"{holdout_label}_roc_auc",
                f"{holdout_label}_average_precision",
                f"{holdout_label}_brier_score",
                "overfit_gap_roc_auc",
            ],
            ascending=[False, False, True, True],
            ignore_index=True,
        )

    if not failed.empty:
        failed = failed.sort_values(by=["source", "model"], ignore_index=True)

    return pd.concat([successful, failed], ignore_index=True)


def choose_best_candidate(validation_table: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    """
    Select the winning candidate using a practical shortlist strategy.

    Strategy:
    1. Shortlist models within a small ROC AUC margin of the best CV ROC AUC.
    2. Within that shortlist, keep models within a small margin of the best validation ROC AUC.
    3. If performance is effectively tied, prefer the model with the smaller overfit gap,
       better Brier score, and lower CV variance.
    """
    successful = validation_table[validation_table["status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError("No successful candidate is available for final selection.")

    best_cv_roc_auc = float(successful["cv_roc_auc_mean"].max())
    cv_shortlist = successful[
        successful["cv_roc_auc_mean"] >= best_cv_roc_auc - MODEL_SELECTION_CV_ROC_AUC_TOLERANCE
    ].copy()

    best_validation_roc_auc = float(cv_shortlist["validation_roc_auc"].max())
    validation_shortlist = cv_shortlist[
        cv_shortlist["validation_roc_auc"]
        >= best_validation_roc_auc - MODEL_SELECTION_VALIDATION_ROC_AUC_TOLERANCE
    ].copy()

    # Keep the number of ascending flags aligned with the tie-break columns below.
    validation_shortlist = validation_shortlist.sort_values(
        by=[
            "overfit_gap_roc_auc",
            "cv_roc_auc_std",
            "validation_brier_score",
            "validation_average_precision",
            "validation_roc_auc",
            "cv_roc_auc_mean",
        ],
        ascending=[True, True, True, False, False, False],
        ignore_index=True,
    )

    selected_row = validation_shortlist.iloc[0]
    selection_summary = {
        "strategy": (
            "Shortlist by CV ROC AUC, then validation ROC AUC, then prefer lower overfit "
            "gap and better calibration quality when differences are effectively tied."
        ),
        "candidate_pool_size": int(len(successful)),
        "best_cv_roc_auc": best_cv_roc_auc,
        "cv_roc_auc_tolerance": MODEL_SELECTION_CV_ROC_AUC_TOLERANCE,
        "validation_roc_auc_tolerance": MODEL_SELECTION_VALIDATION_ROC_AUC_TOLERANCE,
        "cv_shortlist_models": cv_shortlist["model"].tolist(),
        "validation_shortlist_models": validation_shortlist["model"].tolist(),
        "selected_model": str(selected_row["model"]),
        "selected_model_family": str(selected_row["family"]),
        "selected_model_validation_roc_auc": float(selected_row["validation_roc_auc"]),
        "selected_model_cv_roc_auc": float(selected_row["cv_roc_auc_mean"]),
        "selected_model_overfit_gap_roc_auc": float(selected_row["overfit_gap_roc_auc"]),
        "selected_model_validation_brier_score": float(selected_row["validation_brier_score"]),
        "selected_model_cv_roc_auc_std": float(selected_row["cv_roc_auc_std"]),
        "selection_reason": (
            "Chosen from the near-top ROC AUC candidates because it provided the best "
            "stability/overfit trade-off on validation."
        ),
    }
    return str(selected_row["model"]), selection_summary


def build_threshold_grid_report(
    target_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: np.ndarray = THRESHOLD_GRID,
) -> pd.DataFrame:
    """Evaluate a dense threshold grid on a holdout split."""
    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        metrics = compute_threshold_metrics(target_true, probabilities, float(threshold), "grid")
        true_negative, false_positive = metrics.confusion_matrix[0]
        false_negative, true_positive = metrics.confusion_matrix[1]
        rows.append(
            {
                "threshold": metrics.threshold,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "specificity": metrics.specificity,
                "balanced_accuracy": metrics.balanced_accuracy,
                "f1": metrics.f1,
                "true_positive": int(true_positive),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "true_negative": int(true_negative),
                "predicted_positive_count": int(true_positive + false_positive),
                "predicted_negative_count": int(true_negative + false_negative),
            }
        )

    threshold_frame = pd.DataFrame(rows)
    if threshold_frame.empty:
        raise RuntimeError("Threshold tuning grid is empty.")

    return threshold_frame


def score_priority_thresholds(
    threshold_grid: pd.DataFrame,
    *,
    benefit_true_positive: float,
    cost_false_positive_review: float,
    cost_false_negative_missed_need: float,
) -> pd.DataFrame:
    """Score threshold candidates using explicit confusion-count policy weights."""
    return threshold_grid.assign(
        faid_priority_score=(
            threshold_grid["true_positive"] * float(benefit_true_positive)
            - threshold_grid["false_positive"] * float(cost_false_positive_review)
            - threshold_grid["false_negative"] * float(cost_false_negative_missed_need)
        )
    )


def build_priority_weight_sensitivity(
    *,
    selection_target: pd.Series,
    selection_probabilities: np.ndarray,
    evaluation_target: pd.Series | None = None,
    evaluation_probabilities: np.ndarray | None = None,
    missed_need_weight_ratios: tuple[float, ...] = FAID_POLICY_WEIGHT_SENSITIVITY_RATIOS,
    thresholds: np.ndarray = THRESHOLD_GRID,
    min_precision_floor: float | None = MIN_PRECISION_FLOOR,
) -> list[dict[str, Any]]:
    """Compare explicit missed-need/review-burden tradeoffs across policy ratios."""
    selection_grid = build_threshold_grid_report(
        selection_target,
        selection_probabilities,
        thresholds=thresholds,
    )

    evaluation_grid = selection_grid
    evaluation_split = "validation"
    if evaluation_target is not None and evaluation_probabilities is not None:
        evaluation_grid = build_threshold_grid_report(
            evaluation_target,
            evaluation_probabilities,
            thresholds=thresholds,
        )
        evaluation_split = "test"

    rows: list[dict[str, Any]] = []
    for missed_need_weight in missed_need_weight_ratios:
        candidates = selection_grid.copy()
        if min_precision_floor is not None:
            candidates = candidates[candidates["precision"] >= float(min_precision_floor)]
        precision_floor_applied = not candidates.empty
        if candidates.empty:
            candidates = selection_grid.copy()

        scored = score_priority_thresholds(
            candidates,
            benefit_true_positive=FAID_POLICY_PRIORITY_WEIGHTS["benefit_true_positive"],
            cost_false_positive_review=FAID_POLICY_PRIORITY_WEIGHTS["cost_false_positive_review"],
            cost_false_negative_missed_need=float(missed_need_weight),
        )
        selected_row = scored.sort_values(
            by=["faid_priority_score", "recall", "precision", "threshold"],
            ascending=[False, False, False, True],
            ignore_index=True,
        ).iloc[0]

        evaluation_row = evaluation_grid.loc[
            (evaluation_grid["threshold"] - float(selected_row["threshold"])).abs().idxmin()
        ]
        rows.append(
            {
                "missed_need_to_review_ratio": f"{int(missed_need_weight)}:1",
                "missed_need_weight": float(missed_need_weight),
                "selected_threshold": float(selected_row["threshold"]),
                "selection_split": "validation",
                "evaluation_split": evaluation_split,
                "precision_floor_applied": bool(precision_floor_applied),
                "priority_score_on_selection_split": float(selected_row["faid_priority_score"]),
                "precision": float(evaluation_row["precision"]),
                "recall": float(evaluation_row["recall"]),
                "missed_deserving_students": int(evaluation_row["false_negative"]),
                "extra_review_burden": int(evaluation_row["false_positive"]),
                "students_shortlisted": int(evaluation_row["predicted_positive_count"]),
                "true_positives": int(evaluation_row["true_positive"]),
                "false_positives": int(evaluation_row["false_positive"]),
                "false_negatives": int(evaluation_row["false_negative"]),
            }
        )

    return rows


def tune_thresholds(
    target_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: np.ndarray = THRESHOLD_GRID,
    min_precision_floor: float | None = MIN_PRECISION_FLOOR,
    min_recall_floor: float | None = MIN_RECALL_FLOOR,
) -> tuple[pd.DataFrame, dict[str, ThresholdMetrics]]:
    """
    Build an explicit threshold policy set.

    Policies:
    - default_0.50
    - best_f1
    - best_balanced_accuracy
    - max_recall_with_precision_floor
    - max_precision_with_recall_floor (only when configured)
    """
    threshold_grid = build_threshold_grid_report(target_true, probabilities, thresholds=thresholds)
    policy_results: dict[str, ThresholdMetrics] = {}

    policy_results["default_0.50"] = compute_threshold_metrics(
        target_true,
        probabilities,
        threshold=0.50,
        label="default_0.50",
    )

    best_f1_row = threshold_grid.sort_values(
        by=["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
        ignore_index=True,
    ).iloc[0]
    policy_results["best_f1"] = compute_threshold_metrics(
        target_true,
        probabilities,
        threshold=float(best_f1_row["threshold"]),
        label="best_f1",
    )

    best_balanced_accuracy_row = threshold_grid.sort_values(
        by=["balanced_accuracy", "f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, False, True],
        ignore_index=True,
    ).iloc[0]
    policy_results["best_balanced_accuracy"] = compute_threshold_metrics(
        target_true,
        probabilities,
        threshold=float(best_balanced_accuracy_row["threshold"]),
        label="best_balanced_accuracy",
    )

    recall_candidates = threshold_grid.copy()
    if min_precision_floor is not None:
        recall_candidates = recall_candidates[recall_candidates["precision"] >= float(min_precision_floor)]

    if recall_candidates.empty:
        recall_candidates = threshold_grid.copy()
        recall_label = "max_recall_without_precision_floor"
    else:
        recall_label = "max_recall_with_precision_floor"

    best_recall_row = recall_candidates.sort_values(
        by=["recall", "precision", "f1", "threshold"],
        ascending=[False, False, False, True],
        ignore_index=True,
    ).iloc[0]
    policy_results["max_recall_with_precision_floor"] = compute_threshold_metrics(
        target_true,
        probabilities,
        threshold=float(best_recall_row["threshold"]),
        label=recall_label,
    )

    faid_priority_candidates = threshold_grid.copy()
    if min_precision_floor is not None:
        faid_priority_candidates = faid_priority_candidates[
            faid_priority_candidates["precision"] >= float(min_precision_floor)
        ]
    if faid_priority_candidates.empty:
        faid_priority_candidates = threshold_grid.copy()

    faid_priority_candidates = score_priority_thresholds(
        faid_priority_candidates,
        benefit_true_positive=FAID_POLICY_PRIORITY_WEIGHTS["benefit_true_positive"],
        cost_false_positive_review=FAID_POLICY_PRIORITY_WEIGHTS["cost_false_positive_review"],
        cost_false_negative_missed_need=FAID_POLICY_PRIORITY_WEIGHTS["cost_false_negative_missed_need"],
    )
    best_faid_row = faid_priority_candidates.sort_values(
        by=["faid_priority_score", "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
        ignore_index=True,
    ).iloc[0]
    policy_results["faid_need_sensitive"] = compute_threshold_metrics(
        target_true,
        probabilities,
        threshold=float(best_faid_row["threshold"]),
        label="faid_need_sensitive",
    )

    if min_recall_floor is not None:
        precision_candidates = threshold_grid[
            threshold_grid["recall"] >= float(min_recall_floor)
        ].copy()
        if not precision_candidates.empty:
            best_precision_row = precision_candidates.sort_values(
                by=["precision", "recall", "f1", "threshold"],
                ascending=[False, False, False, True],
                ignore_index=True,
            ).iloc[0]
            policy_results["max_precision_with_recall_floor"] = compute_threshold_metrics(
                target_true,
                probabilities,
                threshold=float(best_precision_row["threshold"]),
                label="max_precision_with_recall_floor",
            )

    return threshold_grid, policy_results


def evaluate_threshold_policies(
    target_true: pd.Series,
    probabilities: np.ndarray,
    policy_thresholds: dict[str, float],
) -> dict[str, ThresholdMetrics]:
    """Evaluate fixed threshold policies on a new holdout split."""
    return {
        policy_name: compute_threshold_metrics(
            target_true,
            probabilities,
            threshold=threshold,
            label=policy_name,
        )
        for policy_name, threshold in policy_thresholds.items()
    }


def format_confusion_matrix(matrix: list[list[int]]) -> str:
    """Render a compact confusion matrix."""
    matrix_frame = pd.DataFrame(
        matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"],
    )
    return matrix_frame.to_string()


def threshold_results_to_frame(
    threshold_results: dict[str, ThresholdMetrics],
    *,
    selected_policy: str,
) -> pd.DataFrame:
    """Convert threshold policy results to a readable table."""
    rows: list[dict[str, Any]] = []
    for policy_name, metrics in threshold_results.items():
        rows.append(
            {
                "policy": policy_name,
                "selected": "yes" if policy_name == selected_policy else "",
                "threshold": metrics.threshold,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "specificity": metrics.specificity,
                "balanced_accuracy": metrics.balanced_accuracy,
                "f1": metrics.f1,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["selected", "balanced_accuracy", "f1", "recall"],
        ascending=[False, False, False, False],
        ignore_index=True,
    )


def choose_calibration_mode(summary: dict[str, Any]) -> str:
    """
    Choose between uncalibrated and calibrated probabilities using validation metrics.

    Calibration is preferred only when it improves Brier score meaningfully without
    materially hurting ranking quality.
    """
    calibrated = summary.get("calibrated")
    uncalibrated = summary["uncalibrated"]

    if not calibrated:
        return "uncalibrated"

    brier_improvement = uncalibrated["brier_score"] - calibrated["brier_score"]
    roc_auc_drop = uncalibrated["roc_auc"] - calibrated["roc_auc"]
    average_precision_drop = uncalibrated["average_precision"] - calibrated["average_precision"]

    if (
        brier_improvement >= MODEL_SELECTION_CALIBRATION_MIN_BRIER_IMPROVEMENT
        and roc_auc_drop <= MODEL_SELECTION_CALIBRATION_ROC_AUC_DROP_LIMIT
        and average_precision_drop <= MODEL_SELECTION_CALIBRATION_AVG_PRECISION_DROP_LIMIT
    ):
        return "calibrated"

    return "uncalibrated"


def compare_calibration_modes(
    model_prototype: Pipeline,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_valid: pd.DataFrame,
    target_valid: pd.Series,
) -> tuple[dict[str, Any], Any, np.ndarray]:
    """
    Evaluate uncalibrated vs calibrated probabilities on the validation split.

    When calibration is disabled, this still returns a consistent summary for reporting.
    """
    uncalibrated_model = fit_model_prototype(model_prototype, features_train, target_train)
    uncalibrated_probabilities = get_positive_class_probabilities(uncalibrated_model, features_valid)
    uncalibrated_metrics = compute_probability_metrics(target_valid, uncalibrated_probabilities)

    summary: dict[str, Any] = {
        "enabled": bool(CALIBRATE_SELECTED_MODEL),
        "selected_mode": "uncalibrated",
        "uncalibrated": uncalibrated_metrics,
        "calibrated": None,
        "status": "uncalibrated_only",
        "selection_reason": "Calibration disabled by configuration.",
    }

    if not CALIBRATE_SELECTED_MODEL:
        return summary, uncalibrated_model, uncalibrated_probabilities

    try:
        calibrated_model = fit_calibrated_model(model_prototype, features_train, target_train)
        calibrated_probabilities = get_positive_class_probabilities(calibrated_model, features_valid)
        calibrated_metrics = compute_probability_metrics(target_valid, calibrated_probabilities)

        summary["calibrated"] = calibrated_metrics
        summary["status"] = "calibration_compared"
        summary["brier_improvement"] = (
            uncalibrated_metrics["brier_score"] - calibrated_metrics["brier_score"]
        )
        summary["roc_auc_change"] = calibrated_metrics["roc_auc"] - uncalibrated_metrics["roc_auc"]
        summary["average_precision_change"] = (
            calibrated_metrics["average_precision"] - uncalibrated_metrics["average_precision"]
        )
        summary["selected_mode"] = choose_calibration_mode(summary)

        if summary["selected_mode"] == "calibrated":
            summary["selection_reason"] = (
                "Calibration improved Brier score meaningfully without materially reducing "
                "validation ROC AUC or average precision."
            )
            return summary, calibrated_model, calibrated_probabilities

        summary["selection_reason"] = (
            "Calibration did not improve probability quality enough to justify the validation "
            "ranking-quality trade-off."
        )
        return summary, uncalibrated_model, uncalibrated_probabilities
    except Exception as exc:
        warnings.warn(
            f"Calibration failed; falling back to uncalibrated probabilities. Reason: {exc}",
            stacklevel=2,
        )
        summary["status"] = f"calibration_failed: {exc}"
        summary["selection_reason"] = "Calibration failed, so uncalibrated probabilities were used."
        return summary, uncalibrated_model, uncalibrated_probabilities


def clean_feature_name(raw_name: str) -> str:
    """Make transformed feature names easier to read."""
    cleaned = raw_name.replace("num__", "").replace("cat__", "")
    cleaned = cleaned.replace("imputer__", "")
    cleaned = cleaned.replace("encoder__", "")
    cleaned = cleaned.replace("missingindicator_", "is_missing:")
    cleaned = cleaned.replace("__missing__", "[missing]")
    return cleaned


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Recover readable feature names after preprocessing."""
    try:
        raw_feature_names = preprocessor.get_feature_names_out()
    except Exception as exc:
        raise RuntimeError("Could not recover transformed feature names.") from exc

    return [clean_feature_name(name) for name in raw_feature_names]


def print_feature_insights(
    fitted_pipeline: Pipeline,
    features_reference: pd.DataFrame,
    target_reference: pd.Series,
    top_n: int = TOP_FEATURES_TO_SHOW,
) -> dict[str, Any]:
    """Print model explainability output with readable post-encoding feature names."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    estimator = fitted_pipeline.named_steps["model"]
    feature_names = get_transformed_feature_names(preprocessor)

    print_section("Feature Insights")

    if isinstance(estimator, LogisticRegression):
        coefficients = estimator.coef_.ravel()
        insights_frame = pd.DataFrame(
            {"feature": feature_names, "coefficient": coefficients}
        ).sort_values(by="coefficient", ascending=False, ignore_index=True)

        top_positive = insights_frame.head(top_n)
        top_negative = insights_frame.sort_values(
            by="coefficient", ascending=True, ignore_index=True
        ).head(top_n)

        print("Top positive coefficients:")
        print(top_positive.to_string(index=False))
        print("\nTop negative coefficients:")
        print(top_negative.to_string(index=False))

        return {
            "type": "logistic_coefficients",
            "note": LOCAL_EXPLANATION_NOTE,
            "top_positive": top_positive.to_dict(orient="records"),
            "top_negative": top_negative.to_dict(orient="records"),
        }

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        insights_frame = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values(by="importance", ascending=False, ignore_index=True)
        top_importances = insights_frame.head(top_n)

        print("Top feature importances:")
        print(top_importances.to_string(index=False))

        return {
            "type": "feature_importance",
            "note": LOCAL_EXPLANATION_NOTE,
            "top_features": top_importances.to_dict(orient="records"),
        }

    transformed_reference = preprocessor.transform(features_reference)
    permutation = permutation_importance(
        estimator=estimator,
        X=transformed_reference,
        y=target_reference,
        n_jobs=N_JOBS,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    insights_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False, ignore_index=True)
    top_importances = insights_frame.head(top_n)

    print("Top permutation importances:")
    print(top_importances.to_string(index=False))

    return {
        "type": "permutation_importance",
        "note": LOCAL_EXPLANATION_NOTE,
        "top_features": top_importances.to_dict(orient="records"),
    }


def evaluate_repeated_cv_stability(
    model_prototype: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    """Estimate selection stability with repeated stratified CV."""
    repeated_cv = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS,
        n_repeats=REPEATED_CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    scoring = build_scoring()
    cv_results = cross_validate(
        estimator=clone(model_prototype),
        X=features,
        y=target,
        cv=repeated_cv,
        error_score="raise",
        n_jobs=N_JOBS,
        scoring=scoring,
    )
    return {
        "cv_scheme": "RepeatedStratifiedKFold",
        "n_splits": CV_FOLDS,
        "n_repeats": REPEATED_CV_REPEATS,
        "roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        "average_precision_mean": float(np.mean(cv_results["test_average_precision"])),
        "precision_mean": float(np.mean(cv_results["test_precision"])),
        "recall_mean": float(np.mean(cv_results["test_recall"])),
        "f1_mean": float(np.mean(cv_results["test_f1"])),
        "brier_mean": float(-np.mean(cv_results["test_neg_brier_score"])),
    }


def resolve_fairness_columns(audit_frame: pd.DataFrame) -> list[str]:
    """Return audit columns that are usable for fairness monitoring."""
    columns: list[str] = []
    for column in FAIRNESS_AUDIT_COLUMNS:
        if column not in audit_frame.columns:
            continue
        non_null = audit_frame[column].dropna()
        if non_null.empty:
            continue
        if int(non_null.nunique()) <= 1:
            continue
        columns.append(column)
    return columns


def compute_group_fairness_metrics(
    audit_frame: pd.DataFrame,
    target_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    *,
    group_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute approval and error metrics per subgroup."""
    if not group_columns:
        return pd.DataFrame(), {}

    predictions = (probabilities >= threshold).astype(int)
    fairness_base = audit_frame.copy()
    fairness_base["target_true"] = target_true.to_numpy()
    fairness_base["predicted_positive"] = predictions
    fairness_base["predicted_probability"] = probabilities
    fairness_base["is_error"] = fairness_base["target_true"] != fairness_base["predicted_positive"]

    overall_metrics = {
        "approval_rate": float(np.mean(predictions)),
        "recall": float(recall_score(target_true, predictions, zero_division=0)),
        "precision": float(precision_score(target_true, predictions, zero_division=0)),
        "false_positive_rate": float(
            (((target_true == 0) & (predictions == 1)).sum()) / max(int((target_true == 0).sum()), 1)
        ),
        "false_negative_rate": float(
            (((target_true == 1) & (predictions == 0)).sum()) / max(int((target_true == 1).sum()), 1)
        ),
        "error_rate": float(np.mean(fairness_base["is_error"])),
    }

    rows: list[dict[str, Any]] = []
    for column in group_columns:
        grouped = fairness_base.assign(
            _group_value=fairness_base[column].astype("string").fillna("[missing]")
        ).groupby("_group_value", dropna=False)

        for group_value, group in grouped:
            actual_negative = max(int((group["target_true"] == 0).sum()), 1)
            actual_positive = max(int((group["target_true"] == 1).sum()), 1)
            false_positive_rate = float(
                (((group["target_true"] == 0) & (group["predicted_positive"] == 1)).sum())
                / actual_negative
            )
            false_negative_rate = float(
                (((group["target_true"] == 1) & (group["predicted_positive"] == 0)).sum())
                / actual_positive
            )
            rows.append(
                {
                    "group_column": column,
                    "group_value": str(group_value),
                    "sample_size": int(len(group)),
                    "approval_rate": float(group["predicted_positive"].mean()),
                    "actual_positive_rate": float(group["target_true"].mean()),
                    "precision": float(
                        precision_score(group["target_true"], group["predicted_positive"], zero_division=0)
                    ),
                    "recall": float(
                        recall_score(group["target_true"], group["predicted_positive"], zero_division=0)
                    ),
                    "false_positive_rate": false_positive_rate,
                    "false_negative_rate": false_negative_rate,
                    "error_rate": float(group["is_error"].mean()),
                    "avg_predicted_probability": float(group["predicted_probability"].mean()),
                }
            )

    fairness_frame = pd.DataFrame(rows)
    return fairness_frame, overall_metrics


def derive_fairness_actions(
    fairness_frame: pd.DataFrame,
    overall_metrics: dict[str, float],
) -> pd.DataFrame:
    """Convert fairness gaps into explicit intervention rules."""
    if fairness_frame.empty:
        return fairness_frame

    actions: list[dict[str, Any]] = []
    for record in fairness_frame.to_dict(orient="records"):
        false_positive_gap = record["false_positive_rate"] - overall_metrics["false_positive_rate"]
        false_negative_gap = record["false_negative_rate"] - overall_metrics["false_negative_rate"]
        approval_rate_gap = record["approval_rate"] - overall_metrics["approval_rate"]

        if record["sample_size"] < FAIRNESS_MIN_GROUP_SIZE:
            action = "monitor_only_small_sample"
            reason = "Subgroup too small for intervention."
        elif false_negative_gap >= FAIRNESS_GAP_THRESHOLD:
            action = "lower_threshold_and_force_review"
            reason = (
                "Subgroup false-negative rate is materially above the portfolio baseline, "
                "so potentially deserving students should receive extra scrutiny."
            )
        elif false_positive_gap >= FAIRNESS_GAP_THRESHOLD or abs(approval_rate_gap) >= FAIRNESS_GAP_THRESHOLD:
            action = "force_manual_review"
            reason = (
                "Subgroup decision pattern differs materially from the portfolio baseline, "
                "so automatic approve/reject recommendations should be overridden."
            )
        else:
            action = "monitor_only"
            reason = "No fairness intervention threshold was crossed."

        actions.append(
            {
                **record,
                "false_positive_gap": float(false_positive_gap),
                "false_negative_gap": float(false_negative_gap),
                "approval_rate_gap": float(approval_rate_gap),
                "recommended_action": action,
                "action_reason": reason,
            }
        )

    return pd.DataFrame(actions).sort_values(
        by=[
            "recommended_action",
            "false_negative_gap",
            "false_positive_gap",
            "sample_size",
        ],
        ascending=[True, False, False, False],
        ignore_index=True,
    )


def build_fairness_risk_lookup(fairness_actions: pd.DataFrame) -> dict[str, set[str]]:
    """Build a fast lookup for subgroup combinations that require intervention."""
    if fairness_actions.empty:
        return {}

    actionable = fairness_actions[
        fairness_actions["recommended_action"].isin(
            {"lower_threshold_and_force_review", "force_manual_review"}
        )
    ]
    lookup: dict[str, set[str]] = {}
    for record in actionable.to_dict(orient="records"):
        lookup.setdefault(record["group_column"], set()).add(str(record["group_value"]))
    return lookup


def summarize_fairness_assessment(
    fairness_frame: pd.DataFrame,
    fairness_actions: pd.DataFrame,
    overall_metrics: dict[str, float],
) -> dict[str, Any]:
    """Create a compact fairness summary for metadata/reporting."""
    if fairness_frame.empty:
        return {
            "available": False,
            "message": "No suitable fairness audit columns were available.",
            "overall_metrics": overall_metrics,
            "group_metrics": [],
            "actions": [],
        }

    return {
        "available": True,
        "overall_metrics": overall_metrics,
        "group_metrics": fairness_frame.to_dict(orient="records"),
        "actions": fairness_actions.to_dict(orient="records"),
        "actionable_group_count": int(
            fairness_actions["recommended_action"].isin(
                {"lower_threshold_and_force_review", "force_manual_review"}
            ).sum()
        ),
    }


def build_local_case_explanations(
    fitted_pipeline: Pipeline,
    features_reference: pd.DataFrame,
    *,
    top_n: int = 3,
) -> list[str]:
    """Explain individual rows using model-native logic without requiring SHAP."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    estimator = fitted_pipeline.named_steps["model"]
    feature_names = get_transformed_feature_names(preprocessor)
    transformed = preprocessor.transform(features_reference)
    transformed_array = np.asarray(transformed, dtype=float)

    if isinstance(estimator, LogisticRegression):
        contributions = transformed_array * estimator.coef_.ravel()
        explanations: list[str] = []
        for row in contributions:
            ranked = np.argsort(np.abs(row))[::-1]
            top_indices = [index for index in ranked if abs(row[index]) > 1e-9][:top_n]
            if not top_indices:
                explanations.append(
                    "Model-native approximation: no strong single driver was identified for this case."
                )
                continue
            parts = [
                f"{feature_names[index]} ({row[index]:+.3f})"
                for index in top_indices
            ]
            explanations.append("Model-native approximation: " + "; ".join(parts))
        return explanations

    if hasattr(estimator, "feature_importances_"):
        ranked = np.argsort(np.asarray(estimator.feature_importances_, dtype=float))[::-1][:top_n]
        summary = "; ".join(
            f"{feature_names[index]} ({float(estimator.feature_importances_[index]):.3f})"
            for index in ranked
        )
        return [f"Model-native approximation using global importance fallback: {summary}" for _ in range(len(features_reference))]

    return ["Model-native local explanation unavailable for this estimator." for _ in range(len(features_reference))]


def build_future_deployment_validation_plan() -> list[dict[str, str]]:
    """Describe the fallback plan when a true temporal split is unavailable."""
    return [
        {
            "step": "Freeze one approved feature/threshold policy bundle before the next admissions cycle opens.",
            "purpose": "Prevent silent threshold or feature drift during live deployment.",
        },
        {
            "step": "Score the next full cycle without retraining and compare later committee outcomes against the frozen predictions.",
            "purpose": "Measure future-cycle discrimination, calibration, and operational error rates honestly.",
        },
        {
            "step": "Re-run fairness action rules on the future cycle before allowing any broader automation.",
            "purpose": "Ensure subgroup gaps remain governed rather than assumed stable.",
        },
        {
            "step": "Update the model only after documenting drift, missed-need errors, and review burden under the previous bundle.",
            "purpose": "Keep iteration auditable and thesis-defensible.",
        },
    ]


def resolve_temporal_validation_split(
    dataframe: pd.DataFrame,
    *,
    minimum_cycle_rows: int = TEMPORAL_VALIDATION_MIN_CYCLE_ROWS,
) -> dict[str, Any]:
    """Create a defendable earlier-cycle/later-cycle split when cycle information exists."""
    candidate_tokens = ("date", "year", "term", "cycle", "semester", "cohort", "application")
    candidate_columns = [
        column for column in dataframe.columns
        if any(token in column.lower() for token in candidate_tokens)
    ]

    if "parsed_application_term" in dataframe.columns:
        raw_terms = pd.to_numeric(dataframe["parsed_application_term"], errors="coerce")
        cycle_year = (raw_terms // 100).astype("Int64")
        cycle_frame = pd.DataFrame({"cycle_year": cycle_year}, index=dataframe.index).dropna()
        cycle_summary = (
            cycle_frame.groupby("cycle_year", dropna=False)
            .size()
            .rename("rows")
            .reset_index()
            .sort_values("cycle_year")
            .reset_index(drop=True)
        )
        eligible_cycles = [
            int(value)
            for value in cycle_summary.loc[cycle_summary["rows"] >= int(minimum_cycle_rows), "cycle_year"].tolist()
        ]

        if len(eligible_cycles) >= 3:
            train_cycles = eligible_cycles[:-2]
            validation_cycles = [eligible_cycles[-2]]
            test_cycles = [eligible_cycles[-1]]

            train_index = cycle_year[cycle_year.isin(train_cycles)].index.tolist()
            validation_index = cycle_year[cycle_year.isin(validation_cycles)].index.tolist()
            test_index = cycle_year[cycle_year.isin(test_cycles)].index.tolist()

            return {
                "temporal_split_feasible": True,
                "source_column": "parsed_application_term",
                "cycle_unit": "application_cycle_year",
                "candidate_columns": candidate_columns,
                "viable_candidate_columns": ["parsed_application_term"],
                "train_cycles": train_cycles,
                "validation_cycles": validation_cycles,
                "test_cycles": test_cycles,
                "train_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "test_rows": int(len(test_index)),
                "train_index": train_index,
                "validation_index": validation_index,
                "test_index": test_index,
                "cycle_summary": cycle_summary.to_dict(orient="records"),
                "note": (
                    "A true temporal split is feasible using parsed_application_term grouped into application-cycle years. "
                    "Training uses earlier cycles, validation uses the next cycle, and final testing uses the latest cycle."
                ),
            }

        if len(eligible_cycles) >= 2:
            train_cycles = eligible_cycles[:-1]
            test_cycles = [eligible_cycles[-1]]
            train_index = cycle_year[cycle_year.isin(train_cycles)].index.tolist()
            test_index = cycle_year[cycle_year.isin(test_cycles)].index.tolist()

            return {
                "temporal_split_feasible": True,
                "source_column": "parsed_application_term",
                "cycle_unit": "application_cycle_year",
                "candidate_columns": candidate_columns,
                "viable_candidate_columns": ["parsed_application_term"],
                "train_cycles": train_cycles,
                "validation_cycles": [],
                "test_cycles": test_cycles,
                "train_rows": int(len(train_index)),
                "validation_rows": 0,
                "test_rows": int(len(test_index)),
                "train_index": train_index,
                "validation_index": [],
                "test_index": test_index,
                "cycle_summary": cycle_summary.to_dict(orient="records"),
                "note": (
                    "A partial temporal split is feasible using parsed_application_term. Earlier cycles can train the model "
                    "and the latest cycle can act as a future-style holdout, but validation still needs to be carved from the earlier pool."
                ),
            }

    return {
        "temporal_split_feasible": False,
        "source_column": None,
        "cycle_unit": None,
        "candidate_columns": candidate_columns,
        "viable_candidate_columns": [],
        "train_cycles": [],
        "validation_cycles": [],
        "test_cycles": [],
        "train_rows": 0,
        "validation_rows": 0,
        "test_rows": 0,
        "train_index": [],
        "validation_index": [],
        "test_index": [],
        "cycle_summary": [],
        "note": (
            "No reliable temporal cycle field was available for a defendable earlier-cycle/later-cycle split. "
            "Use the future deployment validation plan instead."
        ),
    }


def assess_temporal_split_feasibility(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Describe whether a cohort or temporal split is feasible with the available data."""
    temporal_split = resolve_temporal_validation_split(dataframe)
    temporal_split["future_deployment_validation_plan"] = build_future_deployment_validation_plan()
    return temporal_split


def bootstrap_metric_intervals(
    target_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Estimate simple bootstrap intervals for key holdout metrics."""
    target_array = np.asarray(target_true, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    metrics_by_name: dict[str, list[float]] = {
        "roc_auc": [],
        "average_precision": [],
        "precision": [],
        "recall": [],
        "balanced_accuracy": [],
    }

    for _ in range(iterations):
        indices = rng.integers(0, len(target_array), len(target_array))
        sampled_target = target_array[indices]
        sampled_probabilities = probability_array[indices]
        if len(np.unique(sampled_target)) < 2:
            continue

        probability_metrics = compute_probability_metrics(sampled_target, sampled_probabilities)
        threshold_metrics = compute_threshold_metrics(
            sampled_target,
            sampled_probabilities,
            threshold=float(threshold),
            label="bootstrap",
        )
        metrics_by_name["roc_auc"].append(probability_metrics["roc_auc"])
        metrics_by_name["average_precision"].append(probability_metrics["average_precision"])
        metrics_by_name["precision"].append(threshold_metrics.precision)
        metrics_by_name["recall"].append(threshold_metrics.recall)
        metrics_by_name["balanced_accuracy"].append(threshold_metrics.balanced_accuracy)

    summary: dict[str, Any] = {
        "iterations_requested": int(iterations),
        "iterations_used": int(len(metrics_by_name["roc_auc"])),
    }
    for metric_name, values in metrics_by_name.items():
        if not values:
            summary[metric_name] = None
            continue
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_95_lower": float(np.quantile(values, 0.025)),
            "ci_95_upper": float(np.quantile(values, 0.975)),
        }
    summary["fairness_gap_uncertainty_note"] = (
        "Fairness gaps are reported with subgroup sample sizes and intervention thresholds, but not bootstrapped here, "
        "because subgroup estimates can become unstable when repeatedly resampled."
    )
    return summary


def build_governance_table(exclusion_summary: dict[str, Any]) -> pd.DataFrame:
    """Render a compact governance table for the report."""
    rows = [
        {
            "column_type": "target",
            "examples": ", ".join(exclusion_summary.get("target_columns", [])[:5]),
            "used_in_model": "no",
            "why": "Outcome labels define the target and cannot be used as predictors.",
        },
        {
            "column_type": "post_decision",
            "examples": ", ".join(exclusion_summary.get("post_decision_columns", [])[:5]),
            "used_in_model": "no",
            "why": "These fields are only known after the committee decision and would leak the answer.",
        },
        {
            "column_type": "identifier",
            "examples": ", ".join(exclusion_summary.get("identifier_columns", [])[:5]),
            "used_in_model": "no",
            "why": "Identifiers do not represent financial need and can encourage memorization rather than generalization.",
        },
        {
            "column_type": "audit_only",
            "examples": ", ".join(exclusion_summary.get("audit_only_columns", [])[:5]),
            "used_in_model": "no",
            "why": "These fields are retained for fairness and governance monitoring rather than prediction.",
        },
        {
            "column_type": "model_input",
            "examples": "pre-decision financial, household, and asset features",
            "used_in_model": "yes",
            "why": "These fields are available before decision time and encode financial-need context.",
        },
    ]
    return pd.DataFrame(rows)


def build_decision_recommendations(
    *,
    audit_frame: pd.DataFrame,
    model_features: pd.DataFrame,
    target_true: pd.Series,
    probabilities: np.ndarray,
    selected_threshold: float,
    fairness_lookup: dict[str, set[str]],
    explainability_model: Pipeline | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Translate probabilities into approve/reject/review recommendations."""
    approve_threshold = max(
        DECISION_APPROVE_MIN_PROBABILITY,
        min(selected_threshold + DECISION_REVIEW_MARGIN, 0.95),
    )
    reject_threshold = min(
        DECISION_REJECT_MAX_PROBABILITY,
        max(selected_threshold - DECISION_REVIEW_MARGIN, 0.05),
    )

    fairness_flags: list[bool] = []
    fairness_reasons: list[str] = []
    for _, row in audit_frame.iterrows():
        triggered: list[str] = []
        for column, risky_values in fairness_lookup.items():
            if column not in audit_frame.columns:
                continue
            value = str(row.get(column, "[missing]"))
            if pd.isna(row.get(column)):
                value = "[missing]"
            if value in risky_values:
                triggered.append(f"{column}={value}")
        fairness_flags.append(bool(triggered))
        fairness_reasons.append(", ".join(triggered))

    explanations = (
        build_local_case_explanations(explainability_model, model_features)
        if explainability_model is not None
        else ["Explainability model unavailable."] * len(model_features)
    )

    rows: list[dict[str, Any]] = []
    for position, (index, probability) in enumerate(zip(model_features.index, probabilities)):
        confidence = float(max(probability, 1.0 - probability))
        borderline = abs(float(probability) - float(selected_threshold)) <= DECISION_REVIEW_MARGIN
        fairness_risk = fairness_flags[position]

        if fairness_risk:
            action = "review"
            reason = "Manual review required because subgroup fairness safeguards were triggered."
        elif borderline:
            action = "review"
            reason = "Manual review required because the case is too close to the policy threshold."
        elif probability >= approve_threshold:
            action = "approve"
            reason = "High-confidence approve recommendation under the calibrated probability policy."
        elif probability <= reject_threshold:
            action = "reject"
            reason = "High-confidence reject recommendation under the calibrated probability policy."
        else:
            action = "review"
            reason = "Manual review required because model confidence is not high enough for auto-routing."

        row_payload = audit_frame.loc[index].to_dict() if index in audit_frame.index else {}
        row_payload.update(
            {
                "actual_target": int(target_true.loc[index]) if index in target_true.index else None,
                "predicted_probability_awarded": float(probability),
                "confidence_score": confidence,
                "selected_threshold": float(selected_threshold),
                "approve_threshold": float(approve_threshold),
                "reject_threshold": float(reject_threshold),
                "fairness_risk_flag": fairness_risk,
                "fairness_risk_reason": fairness_reasons[position],
                "recommended_action": action,
                "action_reason": reason,
                "case_explanation": explanations[position],
            }
        )
        rows.append(row_payload)

    decision_frame = pd.DataFrame(rows)
    summary = {
        "approve_threshold": float(approve_threshold),
        "reject_threshold": float(reject_threshold),
        "review_margin": float(DECISION_REVIEW_MARGIN),
        "action_counts": decision_frame["recommended_action"].value_counts().to_dict(),
        "fairness_risk_count": int(decision_frame["fairness_risk_flag"].sum()),
    }
    return decision_frame, summary


def compute_optional_shap_explanations(
    fitted_pipeline: Pipeline | None,
    features_reference: pd.DataFrame,
    example_indices: list[Any],
    *,
    top_n: int = 3,
) -> dict[Any, str]:
    """Attempt SHAP explanations for a small set of example rows."""
    if fitted_pipeline is None or not example_indices:
        OPTIONAL_DEPENDENCY_STATUS["shap"] = {
            "available": False,
            "requested": False,
            "generated": False,
            "message": "SHAP was not requested because no explainability model or example rows were available.",
        }
        return {}

    try:
        import shap  # type: ignore
    except Exception as exc:
        OPTIONAL_DEPENDENCY_STATUS["shap"] = {
            "available": False,
            "requested": True,
            "generated": False,
            "message": f"SHAP unavailable: {exc}",
        }
        return {}

    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    estimator = fitted_pipeline.named_steps["model"]
    feature_names = get_transformed_feature_names(preprocessor)

    try:
        background_features = features_reference.sample(
            n=min(100, len(features_reference)),
            random_state=RANDOM_STATE,
        )
        background = np.asarray(preprocessor.transform(background_features), dtype=float)
        example_features = features_reference.loc[example_indices]
        transformed_examples = np.asarray(preprocessor.transform(example_features), dtype=float)

        explainer = shap.Explainer(estimator, background, feature_names=feature_names)
        shap_values = explainer(transformed_examples)

        explanations: dict[Any, str] = {}
        for row_position, row_index in enumerate(example_indices):
            values = np.asarray(shap_values.values[row_position], dtype=float)
            ranked = np.argsort(np.abs(values))[::-1][:top_n]
            parts = [
                f"{feature_names[index]} ({values[index]:+.3f})"
                for index in ranked
                if abs(values[index]) > 1e-9
            ]
            explanations[row_index] = (
                "SHAP explanation: " + "; ".join(parts)
                if parts
                else "SHAP explanation: no dominant local driver identified."
            )

        OPTIONAL_DEPENDENCY_STATUS["shap"] = {
            "available": True,
            "requested": True,
            "generated": True,
            "message": "SHAP explanations generated successfully for example cases.",
        }
        return explanations
    except Exception as exc:
        OPTIONAL_DEPENDENCY_STATUS["shap"] = {
            "available": True,
            "requested": True,
            "generated": False,
            "message": f"SHAP available but explanation generation failed: {exc}",
        }
        return {}


def build_decision_examples(
    decision_recommendations: pd.DataFrame,
    model_features: pd.DataFrame,
    explainability_model: Pipeline | None,
) -> list[dict[str, Any]]:
    """Select a small set of concrete decision examples for reporting."""
    if decision_recommendations.empty:
        return []

    review_candidates = decision_recommendations.assign(
        threshold_distance=(
            decision_recommendations["predicted_probability_awarded"]
            - decision_recommendations["selected_threshold"]
        ).abs()
    )

    slices = []
    approve_examples = (
        decision_recommendations[decision_recommendations["recommended_action"] == "approve"]
        .sort_values("confidence_score", ascending=False)
        .head(DECISION_EXAMPLE_COUNT_PER_ACTION)
    )
    reject_examples = (
        decision_recommendations[decision_recommendations["recommended_action"] == "reject"]
        .sort_values("confidence_score", ascending=False)
        .head(DECISION_EXAMPLE_COUNT_PER_ACTION)
    )
    review_examples = (
        review_candidates[review_candidates["recommended_action"] == "review"]
        .sort_values(
            ["fairness_risk_flag", "threshold_distance", "confidence_score"],
            ascending=[False, True, False],
        )
        .head(DECISION_EXAMPLE_COUNT_PER_ACTION)
    )
    for frame in [approve_examples, reject_examples, review_examples]:
        if not frame.empty:
            slices.append(frame)

    if not slices:
        return []

    example_frame = pd.concat(slices, axis=0).drop_duplicates().copy()
    example_indices = example_frame.index.tolist()
    shap_lookup = compute_optional_shap_explanations(
        explainability_model,
        model_features,
        example_indices,
    )

    example_columns = [
        column
        for column in [
            "raw_source_row_number",
            "raw_source_sheet_name",
            "source_row_number",
            "source_sheet_name",
            "parsed_level",
            "parsed_school",
            "parsed_nationality",
            "parsed_applicant_citizenship",
        ]
        if column in example_frame.columns
    ]
    examples: list[dict[str, Any]] = []
    for index, row in example_frame.iterrows():
        payload = {column: row[column] for column in example_columns}
        payload.update(
            {
                "recommended_action": row["recommended_action"],
                "predicted_probability_awarded": float(row["predicted_probability_awarded"]),
                "confidence_score": float(row["confidence_score"]),
                "fairness_risk_flag": bool(row["fairness_risk_flag"]),
                "action_reason": row["action_reason"],
                "case_explanation": row["case_explanation"],
                "shap_explanation": shap_lookup.get(index),
                "actual_target": row.get("actual_target"),
            }
        )
        examples.append(payload)

    return examples


def summarize_decision_impact(
    decision_recommendations: pd.DataFrame,
    selected_threshold: float,
) -> dict[str, Any]:
    """Translate the decision layer into operational impact terms."""
    if decision_recommendations.empty:
        return {"available": False, "message": "Decision recommendations were not generated."}

    review_mask = decision_recommendations["recommended_action"] == "review"
    approve_mask = decision_recommendations["recommended_action"] == "approve"
    reject_mask = decision_recommendations["recommended_action"] == "reject"

    total_cases = int(len(decision_recommendations))
    review_cases = int(review_mask.sum())
    automated_cases = int((~review_mask).sum())
    automated_share = automated_cases / max(total_cases, 1)

    auto_false_approves = int(
        ((approve_mask) & (decision_recommendations["actual_target"] == 0)).sum()
    )
    auto_false_rejects = int(
        ((reject_mask) & (decision_recommendations["actual_target"] == 1)).sum()
    )

    threshold_only_predictions = (
        decision_recommendations["predicted_probability_awarded"] >= float(selected_threshold)
    ).astype(int)
    threshold_only_false_approves = int(
        ((threshold_only_predictions == 1) & (decision_recommendations["actual_target"] == 0)).sum()
    )
    threshold_only_false_rejects = int(
        ((threshold_only_predictions == 0) & (decision_recommendations["actual_target"] == 1)).sum()
    )

    return {
        "available": True,
        "total_cases": total_cases,
        "manual_review_baseline_cases": total_cases,
        "review_cases_after_model": review_cases,
        "review_reduction_vs_all_manual": int(total_cases - review_cases),
        "automated_cases": automated_cases,
        "automated_share": float(automated_share),
        "auto_approve_cases": int(approve_mask.sum()),
        "auto_reject_cases": int(reject_mask.sum()),
        "auto_false_approves": auto_false_approves,
        "auto_false_rejects": auto_false_rejects,
        "threshold_only_false_approves": threshold_only_false_approves,
        "threshold_only_false_rejects": threshold_only_false_rejects,
        "harmful_auto_errors_avoided_vs_threshold_only": int(
            (threshold_only_false_approves + threshold_only_false_rejects)
            - (auto_false_approves + auto_false_rejects)
        ),
        "note": (
            "The decision layer reduces harmful automatic decisions by routing uncertain or fairness-risk cases "
            "into manual review instead of forcing a binary decision on every applicant."
        ),
    }


def evaluate_decision_boundary_scenarios(
    decision_recommendations: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Compare conservative/current/aggressive decision boundaries."""
    if decision_recommendations.empty:
        return []

    scenarios = [
        {"name": "conservative_auto", "approve_threshold": 0.85, "reject_threshold": 0.15},
        {
            "name": "current_policy",
            "approve_threshold": DECISION_APPROVE_MIN_PROBABILITY,
            "reject_threshold": DECISION_REJECT_MAX_PROBABILITY,
        },
        {"name": "aggressive_auto", "approve_threshold": 0.75, "reject_threshold": 0.25},
    ]

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        approve_mask = (
            (decision_recommendations["predicted_probability_awarded"] >= scenario["approve_threshold"])
            & (~decision_recommendations["fairness_risk_flag"])
        )
        reject_mask = (
            (decision_recommendations["predicted_probability_awarded"] <= scenario["reject_threshold"])
            & (~decision_recommendations["fairness_risk_flag"])
        )
        review_mask = ~(approve_mask | reject_mask)
        rows.append(
            {
                "scenario": scenario["name"],
                "approve_threshold": float(scenario["approve_threshold"]),
                "reject_threshold": float(scenario["reject_threshold"]),
                "approve_cases": int(approve_mask.sum()),
                "reject_cases": int(reject_mask.sum()),
                "review_cases": int(review_mask.sum()),
                "harmful_auto_approves": int(
                    (approve_mask & (decision_recommendations["actual_target"] == 0)).sum()
                ),
                "harmful_auto_rejects": int(
                    (reject_mask & (decision_recommendations["actual_target"] == 1)).sum()
                ),
            }
        )
    return rows


def evaluate_holdout_summary(
    dataset_label: str,
    target_true: pd.Series,
    probabilities: np.ndarray,
    threshold_results: dict[str, ThresholdMetrics],
    selected_policy: str,
) -> dict[str, Any]:
    """Create a unified summary for a validation or test holdout set."""
    if selected_policy not in threshold_results:
        raise KeyError(
            f"Selected threshold policy '{selected_policy}' is unavailable. "
            f"Available policies: {', '.join(threshold_results)}"
        )

    return {
        "dataset_label": dataset_label,
        "probability_metrics": compute_probability_metrics(target_true, probabilities),
        "threshold_results": {
            policy_name: asdict(metrics) for policy_name, metrics in threshold_results.items()
        },
        "selected_policy": selected_policy,
        "selected_threshold": float(threshold_results[selected_policy].threshold),
        "selected_policy_metrics": asdict(threshold_results[selected_policy]),
    }


def print_holdout_summary(holdout_summary: dict[str, Any]) -> None:
    """Print a concise holdout summary with threshold policies."""
    dataset_label = holdout_summary["dataset_label"]
    probability_metrics = holdout_summary["probability_metrics"]
    threshold_results = {
        name: ThresholdMetrics(**metrics)
        for name, metrics in holdout_summary["threshold_results"].items()
    }
    selected_policy = holdout_summary["selected_policy"]

    print_section(f"{dataset_label.title()} Summary")
    print(f"ROC AUC:          {probability_metrics['roc_auc']:.4f}")
    print(f"Average Precision:{probability_metrics['average_precision']:.4f}")
    print(f"Brier Score:      {probability_metrics['brier_score']:.4f}")

    threshold_frame = threshold_results_to_frame(
        threshold_results,
        selected_policy=selected_policy,
    )
    print("\nThreshold policies:")
    print(threshold_frame.round(4).to_string(index=False))

    selected_metrics = threshold_results[selected_policy]
    print("\nSelected policy confusion matrix:")
    print(format_confusion_matrix(selected_metrics.confusion_matrix))


def resolve_suspicious_features(columns: list[str]) -> list[str]:
    """Resolve suspicious features using exact names plus focused keyword rules."""
    resolved = set()
    for column in columns:
        lowered = column.lower()
        if column in SUSPICIOUS_FEATURES:
            resolved.add(column)
            continue
        if any(keyword in lowered for keyword in SUSPICIOUS_FEATURE_KEYWORDS):
            resolved.add(column)
    return sorted(resolved)


def save_curves(
    target_true: pd.Series,
    probabilities: np.ndarray,
    *,
    title_prefix: str,
) -> dict[str, Any]:
    """Save ROC and precision-recall curves when plotting dependencies are available."""
    summary = {
        "requested": bool(GENERATE_PLOTS),
        "available": False,
        "generated": False,
        "roc_curve_path": None,
        "pr_curve_path": None,
        "message": "",
    }

    if not GENERATE_PLOTS:
        summary["message"] = "Plot generation disabled by configuration."
        OPTIONAL_DEPENDENCY_STATUS["matplotlib"] = summary
        return summary

    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        matplotlib_cache = ARTIFACT_DIR / ".matplotlib_cache"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))

        import matplotlib.pyplot as plt
    except Exception as exc:
        summary["message"] = (
            f"matplotlib unavailable, so ROC/PR plots were skipped: {exc}. "
            "Install it with `pip install matplotlib` to enable plots."
        )
        OPTIONAL_DEPENDENCY_STATUS["matplotlib"] = summary
        warnings.warn(
            "matplotlib is not installed; skipping optional ROC/PR plots. "
            f"Reason: {exc}. Install it with `pip install matplotlib` to enable plots.",
            stacklevel=2,
        )
        return summary

    summary["available"] = True

    roc_auc = safe_metric(roc_auc_score, target_true, probabilities)
    average_precision = safe_metric(average_precision_score, target_true, probabilities)
    fpr, tpr, _ = roc_curve(target_true, probabilities)
    precision_curve, recall_curve, _ = precision_recall_curve(target_true, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(
        recall_curve,
        precision_curve,
        linewidth=2,
        label=f"Average Precision = {average_precision:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(PR_CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    summary["generated"] = True
    summary["roc_curve_path"] = str(ROC_CURVE_PATH.resolve())
    summary["pr_curve_path"] = str(PR_CURVE_PATH.resolve())
    summary["message"] = "ROC and PR curves generated successfully."
    OPTIONAL_DEPENDENCY_STATUS["matplotlib"] = summary
    return summary


def to_serializable(value: Any) -> Any:
    """Recursively convert numpy/pandas objects into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def build_report_text(
    *,
    config_summary: dict[str, Any],
    data_metadata: dict[str, Any],
    split_metadata: dict[str, Any],
    normal_result: dict[str, Any],
    leakage_summary: dict[str, Any] | None,
    optional_dependencies: dict[str, Any],
    plot_summary: dict[str, Any],
) -> str:
    """Create a concise audit report."""
    cv_table = pd.DataFrame(normal_result["cv_comparison"])
    validation_table = pd.DataFrame(normal_result["validation_comparison"])
    governance_table = build_governance_table(split_metadata["explicit_model_exclusion_summary"])
    fairness_actions = pd.DataFrame(normal_result["test_fairness_summary"].get("actions", []))
    if not fairness_actions.empty:
        fairness_action_table = fairness_actions[
            [
                "group_column",
                "group_value",
                "sample_size",
                "false_negative_gap",
                "false_positive_gap",
                "recommended_action",
                "action_reason",
            ]
        ].head(20)
    else:
        fairness_action_table = pd.DataFrame(
            [{"note": "No fairness action table was generated for this run."}]
        )
    boundary_scenarios = pd.DataFrame(normal_result.get("decision_boundary_scenarios") or [])
    if boundary_scenarios.empty:
        boundary_scenarios = pd.DataFrame(
            [{"note": "No decision-boundary scenario table was generated for this run."}]
        )
    decision_examples = pd.DataFrame(normal_result.get("decision_examples") or [])
    if decision_examples.empty:
        decision_examples = pd.DataFrame(
            [{"note": "No decision examples were generated for this run."}]
        )
    policy_weight_sensitivity = pd.DataFrame(normal_result.get("policy_weight_sensitivity") or [])
    if policy_weight_sensitivity.empty:
        policy_weight_sensitivity = pd.DataFrame(
            [{"note": "No policy-weight sensitivity table was generated for this run."}]
        )

    lines = [
        "Yasmina Eligibility Model Report",
        "=" * 80,
        "",
        "Configuration",
        "-" * 80,
        json.dumps(to_serializable(config_summary), indent=2),
        "",
        "Data Summary",
        "-" * 80,
        json.dumps(to_serializable(data_metadata), indent=2),
        "",
        "Problem / Solution / Impact",
        "-" * 80,
        (
            "Problem: FAID eligibility labels live in messy operational data, so a model can only "
            "be trusted when the data pipeline, leakage controls, and decision policy are all explicit.\n"
            "Solution: this workflow uses governed feature selection, conservative target filtering, "
            "probability calibration, fairness monitoring, and a three-way decision layer "
            "(approve / reject / review).\n"
            "Impact: the model becomes a decision-support tool for safer automation and triage, "
            "not an opaque replacement for committee judgment."
        ),
        "",
        "Target Governance",
        "-" * 80,
        (
            "Positive labels are treated as explicit award outcomes, denials are negative outcomes, "
            "and non-binary labels such as USAID are excluded rather than forced into the negative class. "
            "This prevents ambiguous or structurally different aid programs from contaminating the binary "
            "eligibility target."
        ),
        "",
        "Target Choice Impact",
        "-" * 80,
        (
            f"Rows dropped for non-binary labels: {data_metadata['rows_dropped_non_binary_target']}. "
            "This exclusion reduces sample size slightly but produces a cleaner and more defensible target definition."
        ),
        "",
        "Split Summary",
        "-" * 80,
        json.dumps(to_serializable(split_metadata), indent=2),
        "",
        "Temporal / Cohort Validation Assessment",
        "-" * 80,
        json.dumps(to_serializable(split_metadata["temporal_split_assessment"]), indent=2),
        "",
        "Leakage and Data Governance",
        "-" * 80,
        (
            "Only information available before the committee decision is allowed into modeling. "
            "Post-decision fields, identifiers, and audit-only fairness columns are excluded from the "
            "model input and retained only for monitoring, reporting, or case routing."
        ),
        "",
        "Governance Table",
        "-" * 80,
        governance_table.to_string(index=False),
        "",
        "Feature Engineering",
        "-" * 80,
        json.dumps(to_serializable(normal_result["feature_engineering"]), indent=2),
        "",
        "Feature Engineering Rationale",
        "-" * 80,
        (
            "Engineered features such as income per person, support per person, tuition burden, "
            "assets-to-income ratio, and debt-to-income ratios are intended to approximate household "
            "financial capacity and financial stress. The goal is to align features with financial-need "
            "theory rather than rely only on raw columns."
        ),
        "",
        "Feature Pruning",
        "-" * 80,
        json.dumps(to_serializable(normal_result["feature_pruning"]), indent=2),
        "",
        "CV Comparison",
        "-" * 80,
        cv_table.round(4).to_string(index=False),
        "",
        "Validation Comparison",
        "-" * 80,
        validation_table.round(4).to_string(index=False),
        "",
        "Model Selection",
        "-" * 80,
        json.dumps(to_serializable(normal_result["selection_summary"]), indent=2),
        "",
        "Threshold Policy Rationale",
        "-" * 80,
        (
            f"Selected threshold policy: {normal_result['selected_threshold_policy']}\n"
            f"Policy rationale: {FAID_POLICY_RATIONALE}\n"
            f"Weight rationale: {FAID_POLICY_WEIGHT_RATIONALE}"
        ),
        "",
        "Threshold Policy Sensitivity",
        "-" * 80,
        policy_weight_sensitivity.to_string(index=False),
        "",
        "Calibration Summary",
        "-" * 80,
        json.dumps(to_serializable(normal_result["calibration_summary"]), indent=2),
        "",
        "Calibration Interpretation",
        "-" * 80,
        (
            "Calibration matters because predicted probabilities are used as decision confidence. "
            "A well-ranked but badly calibrated model can still produce unsafe approve/reject recommendations."
        ),
        "",
        "Repeated CV Stability",
        "-" * 80,
        json.dumps(to_serializable(normal_result["stability_summary"]), indent=2),
        "",
        "Uncertainty Summary",
        "-" * 80,
        json.dumps(to_serializable(normal_result["uncertainty_summary"]), indent=2),
        "",
        "Validation Summary",
        "-" * 80,
        json.dumps(to_serializable(normal_result["validation_summary"]), indent=2),
        "",
        "Validation Fairness Summary",
        "-" * 80,
        json.dumps(to_serializable(normal_result["validation_fairness_summary"]), indent=2),
        "",
        "Test Summary",
        "-" * 80,
        json.dumps(to_serializable(normal_result["test_summary"]), indent=2),
        "",
        "Test Fairness Summary",
        "-" * 80,
        json.dumps(to_serializable(normal_result["test_fairness_summary"]), indent=2),
        "",
        "Fairness Governance Rationale",
        "-" * 80,
        (
            f"Fairness gap threshold rationale: {FAIRNESS_THRESHOLD_RATIONALE}\n"
            f"Minimum subgroup size rationale: {FAIRNESS_MIN_GROUP_SIZE_RATIONALE}\n"
            f"Limitations: {FAIRNESS_LIMITATIONS_NOTE}"
        ),
        "",
        "Fairness Action Table",
        "-" * 80,
        fairness_action_table.to_string(index=False),
        "",
        "Decision Layer Summary",
        "-" * 80,
        json.dumps(to_serializable(normal_result["decision_summary"]), indent=2),
        "",
        "Business Impact Summary",
        "-" * 80,
        json.dumps(to_serializable(normal_result["business_impact_summary"]), indent=2),
        "",
        "Decision Boundary Scenario Table",
        "-" * 80,
        boundary_scenarios.to_string(index=False),
        "",
        "Decision Boundary Rationale",
        "-" * 80,
        DECISION_BOUNDARY_RATIONALE,
        "",
        "Example Decisions",
        "-" * 80,
        decision_examples.to_string(index=False),
        "",
        "Optional Dependencies",
        "-" * 80,
        json.dumps(to_serializable(optional_dependencies), indent=2),
        "",
        "Plot Summary",
        "-" * 80,
        json.dumps(to_serializable(plot_summary), indent=2),
        "",
    ]

    if leakage_summary is not None:
        lines.extend(
            [
                "Leakage Stress Test",
                "-" * 80,
                json.dumps(to_serializable(leakage_summary), indent=2),
                "",
            ]
        )

    lines.extend(
        [
            "Feature Insights",
            "-" * 80,
            LOCAL_EXPLANATION_NOTE,
            "",
            json.dumps(to_serializable(normal_result["feature_insights"]), indent=2),
            "",
        ]
    )

    return "\n".join(lines)


def build_config_summary() -> dict[str, Any]:
    """Capture the key runtime configuration in metadata/reporting."""
    return {
        "model_artifact_type": MODEL_ARTIFACT_TYPE,
        "model_artifact_version": MODEL_ARTIFACT_VERSION,
        "project_root": str(PROJECT_ROOT.resolve()),
        "data_path": str(resolve_data_path(DATA_PATH).resolve()),
        "artifact_dir": str(ARTIFACT_DIR.resolve()),
        "config_env_overrides": {
            "YASMINA_ELIGIBILITY_DATA_PATH": os.environ.get("YASMINA_ELIGIBILITY_DATA_PATH"),
            "YASMINA_ELIGIBILITY_ARTIFACT_DIR": os.environ.get("YASMINA_ELIGIBILITY_ARTIFACT_DIR"),
        },
        "target_column": TARGET_COLUMN,
        "positive_target_label": POSITIVE_TARGET_LABEL,
        "positive_target_labels": list(POSITIVE_TARGET_LABELS),
        "negative_target_labels": list(NEGATIVE_TARGET_LABELS),
        "excluded_target_labels": list(EXCLUDED_TARGET_LABELS),
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "repeated_cv_repeats": REPEATED_CV_REPEATS,
        "random_state": RANDOM_STATE,
        "random_search_enabled": RUN_RANDOM_SEARCH,
        "random_search_iterations": RANDOM_SEARCH_ITERATIONS,
        "calibrate_selected_model": CALIBRATE_SELECTED_MODEL,
        "selected_threshold_policy": SELECTED_THRESHOLD_POLICY,
        "faid_policy_priority_weights": FAID_POLICY_PRIORITY_WEIGHTS,
        "faid_policy_rationale": FAID_POLICY_RATIONALE,
        "faid_policy_weight_rationale": FAID_POLICY_WEIGHT_RATIONALE,
        "min_precision_floor": MIN_PRECISION_FLOOR,
        "min_recall_floor": MIN_RECALL_FLOOR,
        "decision_approve_min_probability": DECISION_APPROVE_MIN_PROBABILITY,
        "decision_reject_max_probability": DECISION_REJECT_MAX_PROBABILITY,
        "decision_review_margin": DECISION_REVIEW_MARGIN,
        "decision_boundary_rationale": DECISION_BOUNDARY_RATIONALE,
        "fairness_gap_threshold": FAIRNESS_GAP_THRESHOLD,
        "fairness_min_group_size": FAIRNESS_MIN_GROUP_SIZE,
        "fairness_threshold_rationale": FAIRNESS_THRESHOLD_RATIONALE,
        "fairness_min_group_size_rationale": FAIRNESS_MIN_GROUP_SIZE_RATIONALE,
        "fairness_limitations_note": FAIRNESS_LIMITATIONS_NOTE,
        "local_explanation_note": LOCAL_EXPLANATION_NOTE,
        "model_selection_cv_roc_auc_tolerance": MODEL_SELECTION_CV_ROC_AUC_TOLERANCE,
        "model_selection_validation_roc_auc_tolerance": MODEL_SELECTION_VALIDATION_ROC_AUC_TOLERANCE,
        "calibration_roc_auc_drop_limit": MODEL_SELECTION_CALIBRATION_ROC_AUC_DROP_LIMIT,
        "calibration_average_precision_drop_limit": MODEL_SELECTION_CALIBRATION_AVG_PRECISION_DROP_LIMIT,
        "calibration_min_brier_improvement": MODEL_SELECTION_CALIBRATION_MIN_BRIER_IMPROVEMENT,
        "run_leakage_stress_test": RUN_LEAKAGE_STRESS_TEST,
        "identifier_columns": sorted(IDENTIFIER_COLUMNS),
        "audit_only_columns": sorted(AUDIT_ONLY_COLUMNS),
        "fairness_audit_columns": list(FAIRNESS_AUDIT_COLUMNS),
        "drop_high_missingness_columns": DROP_HIGH_MISSINGNESS_COLUMNS,
        "high_missingness_threshold": HIGH_MISSINGNESS_THRESHOLD,
        "drop_near_constant_columns": DROP_NEAR_CONSTANT_COLUMNS,
        "near_constant_share_threshold": NEAR_CONSTANT_SHARE_THRESHOLD,
        "drop_high_cardinality_text_columns": DROP_HIGH_CARDINALITY_TEXT_COLUMNS,
        "high_cardinality_min_unique": HIGH_CARDINALITY_MIN_UNIQUE,
        "high_cardinality_unique_ratio": HIGH_CARDINALITY_UNIQUE_RATIO,
        "high_cardinality_avg_length_threshold": HIGH_CARDINALITY_AVG_LENGTH_THRESHOLD,
        "default_workflow_exclusion_columns": sorted(DEFAULT_WORKFLOW_EXCLUSION_COLUMNS),
    }


def build_model_bundle(
    *,
    deployed_model: Any,
    explainability_model: Any | None,
    config_summary: dict[str, Any],
    data_metadata: dict[str, Any],
    split_metadata: dict[str, Any],
    normal_result: dict[str, Any],
) -> dict[str, Any]:
    """Package the deployable model together with the settings required at inference time."""
    test_summary = normal_result["test_summary"] or {}
    calibration_summary = normal_result["calibration_summary"]

    return {
        "artifact_type": MODEL_ARTIFACT_TYPE,
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "estimator": deployed_model,
        "explainability_estimator": explainability_model,
        "selected_threshold_policy": normal_result["selected_threshold_policy"],
        "selected_threshold": test_summary.get("selected_threshold"),
        "selected_model_name": normal_result["selected_candidate"],
        "selected_model_family": normal_result["selected_model_family"],
        "calibration_mode_selected": calibration_summary["selected_mode"],
        "target_definition": {
            "target_column": data_metadata["target_column"],
            "positive_labels": data_metadata["positive_labels"],
            "negative_labels": data_metadata["negative_labels"],
            "excluded_labels": data_metadata["excluded_labels"],
        },
        "feature_definition": {
            "retained_feature_count": normal_result["retained_feature_count"],
            "extra_excluded_columns": normal_result["extra_excluded_columns"],
            "feature_pruning": normal_result["feature_pruning"],
        },
        "training_summary": {
            "rows_used_for_training": data_metadata["rows_used_for_training"],
            "train_rows": split_metadata["train_rows"],
            "validation_rows": split_metadata["validation_rows"],
            "test_rows": split_metadata["test_rows"],
            "selected_validation_roc_auc": normal_result["selection_summary"][
                "selected_model_validation_roc_auc"
            ],
            "test_roc_auc": test_summary.get("probability_metrics", {}).get("roc_auc"),
            "repeated_cv_stability": normal_result["stability_summary"],
        },
        "decision_policy": {
            "selected_threshold_policy": normal_result["selected_threshold_policy"],
            "threshold_policy_rationale": normal_result["threshold_policy_rationale"],
            "policy_weight_sensitivity": normal_result["policy_weight_sensitivity"],
            "decision_summary": normal_result["decision_summary"],
            "decision_boundary_rationale": DECISION_BOUNDARY_RATIONALE,
            "fairness_limitations_note": FAIRNESS_LIMITATIONS_NOTE,
            "business_impact_summary": normal_result["business_impact_summary"],
            "decision_examples": normal_result["decision_examples"],
        },
        "config_snapshot": config_summary,
    }


def summarize_leakage_stress_test(
    normal_result: dict[str, Any],
    stress_result: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact leakage-audit summary across validation and test outputs."""
    selected_policy = normal_result["selected_threshold_policy"]

    normal_validation = normal_result["validation_summary"]
    stress_validation = stress_result["validation_summary"]
    normal_test = normal_result["test_summary"]
    stress_test = stress_result["test_summary"]

    normal_probability_metrics = normal_validation["probability_metrics"]
    stress_probability_metrics = stress_validation["probability_metrics"]
    normal_test_probability_metrics = normal_test["probability_metrics"]
    stress_test_probability_metrics = stress_test["probability_metrics"]

    normal_policy_metrics = normal_validation["threshold_results"][selected_policy]
    stress_policy_metrics = stress_validation["threshold_results"][selected_policy]
    normal_test_policy_metrics = normal_test["threshold_results"][selected_policy]
    stress_test_policy_metrics = stress_test["threshold_results"][selected_policy]

    summary = {
        "run": True,
        "note": (
            "Stress test re-runs the trained workflow without suspicious features to show "
            "how much CV, validation, and final test performance depend on them."
        ),
        "excluded_suspicious_features": stress_result["extra_excluded_columns"],
        "excluded_suspicious_feature_count": len(stress_result["extra_excluded_columns"]),
        "normal_selected_model": normal_result["selected_candidate"],
        "stress_selected_model": stress_result["selected_candidate"],
        "normal_selection_reason": normal_result["selection_summary"]["selection_reason"],
        "stress_selection_reason": stress_result["selection_summary"]["selection_reason"],
        "normal_selected_cv_roc_auc": normal_result["selection_summary"]["selected_model_cv_roc_auc"],
        "stress_selected_cv_roc_auc": stress_result["selection_summary"]["selected_model_cv_roc_auc"],
        "selected_cv_roc_auc_drop": (
            normal_result["selection_summary"]["selected_model_cv_roc_auc"]
            - stress_result["selection_summary"]["selected_model_cv_roc_auc"]
        ),
        "normal_validation_roc_auc": normal_probability_metrics["roc_auc"],
        "stress_validation_roc_auc": stress_probability_metrics["roc_auc"],
        "validation_roc_auc_drop": normal_probability_metrics["roc_auc"] - stress_probability_metrics["roc_auc"],
        "normal_test_roc_auc": normal_test_probability_metrics["roc_auc"],
        "stress_test_roc_auc": stress_test_probability_metrics["roc_auc"],
        "test_roc_auc_drop": normal_test_probability_metrics["roc_auc"] - stress_test_probability_metrics["roc_auc"],
        "normal_validation_brier": normal_probability_metrics["brier_score"],
        "stress_validation_brier": stress_probability_metrics["brier_score"],
        "normal_test_brier": normal_test_probability_metrics["brier_score"],
        "stress_test_brier": stress_test_probability_metrics["brier_score"],
        "normal_selected_policy_f1": normal_policy_metrics["f1"],
        "stress_selected_policy_f1": stress_policy_metrics["f1"],
        "selected_policy_f1_drop": normal_policy_metrics["f1"] - stress_policy_metrics["f1"],
        "normal_test_selected_policy_f1": normal_test_policy_metrics["f1"],
        "stress_test_selected_policy_f1": stress_test_policy_metrics["f1"],
        "test_selected_policy_f1_drop": normal_test_policy_metrics["f1"] - stress_test_policy_metrics["f1"],
        "selected_policy": selected_policy,
        "normal_test_selected_policy_precision": normal_test_policy_metrics["precision"],
        "stress_test_selected_policy_precision": stress_test_policy_metrics["precision"],
        "normal_test_selected_policy_recall": normal_test_policy_metrics["recall"],
        "stress_test_selected_policy_recall": stress_test_policy_metrics["recall"],
    }
    return summary


def run_experiment(
    *,
    experiment_name: str,
    base_train_features: pd.DataFrame,
    base_valid_features: pd.DataFrame,
    base_test_features: pd.DataFrame | None,
    audit_valid_frame: pd.DataFrame | None,
    audit_test_frame: pd.DataFrame | None,
    target_train: pd.Series,
    target_valid: pd.Series,
    target_test: pd.Series | None,
    extra_excluded_columns: list[str] | None = None,
    evaluate_test: bool = True,
    produce_feature_insights: bool = True,
) -> tuple[dict[str, Any], ExperimentArtifacts]:
    """
    Execute the modeling workflow for one feature configuration.

    The normal workflow uses train for fitting/CV, validation for final model and threshold
    selection, and test only once at the end. Leakage stress testing can reuse the same
    function with evaluate_test=False to keep the test untouched.
    """
    extra_excluded_columns = extra_excluded_columns or []

    train_features = base_train_features.drop(columns=extra_excluded_columns, errors="ignore").copy()
    valid_features = base_valid_features.drop(columns=extra_excluded_columns, errors="ignore").copy()
    test_features = (
        base_test_features.drop(columns=extra_excluded_columns, errors="ignore").copy()
        if base_test_features is not None
        else None
    )

    train_features_engineered = build_domain_features(train_features)
    valid_features_engineered = build_domain_features(valid_features)
    test_features_engineered = build_domain_features(test_features) if test_features is not None else None

    engineered_columns = [
        column for column in train_features_engineered.columns if column not in train_features.columns
    ]

    pruning_columns_to_drop, pruning_summary = fit_feature_pruner(train_features_engineered)
    retained_columns = [
        column for column in train_features_engineered.columns if column not in pruning_columns_to_drop
    ]
    if not retained_columns:
        raise RuntimeError("Feature pruning removed every feature column.")

    train_model_features = apply_feature_pruning(
        train_features_engineered,
        pruning_columns_to_drop,
        retained_columns=retained_columns,
    )
    valid_model_features = apply_feature_pruning(
        valid_features_engineered,
        pruning_columns_to_drop,
        retained_columns=retained_columns,
    )
    test_model_features = (
        apply_feature_pruning(
            test_features_engineered,
            pruning_columns_to_drop,
            retained_columns=retained_columns,
        )
        if test_features_engineered is not None
        else None
    )

    preprocessors, numeric_columns, categorical_columns = build_preprocessors(train_model_features)
    class_weight = get_recommended_class_weight(target_train)
    scale_pos_weight = get_recommended_scale_pos_weight(target_train)
    model_specs = get_model_specs(class_weight=class_weight, scale_pos_weight=scale_pos_weight)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print_section(f"{experiment_name} Feature Setup")
    print(f"Extra excluded columns: {len(extra_excluded_columns)}")
    if extra_excluded_columns:
        print(", ".join(extra_excluded_columns))
    print(f"Engineered features added: {len(engineered_columns)}")
    print(f"Pruned columns: {pruning_summary['columns_dropped_total']}")
    print(f"Retained feature columns: {len(retained_columns)}")
    print(f"Numeric columns after pruning: {len(numeric_columns)}")
    print(f"Categorical columns after pruning: {len(categorical_columns)}")
    print(f"Recommended class weight: {class_weight}")

    cv_comparison, base_prototypes = evaluate_models(
        model_specs=model_specs,
        preprocessors=preprocessors,
        features_train=train_model_features,
        target_train=target_train,
        cv=cv,
    )

    tuned_comparison, tuned_prototypes = run_random_search(
        model_specs=model_specs,
        preprocessors=preprocessors,
        features_train=train_model_features,
        target_train=target_train,
        cv=cv,
    )

    comparison_frames = [cv_comparison]
    if not tuned_comparison.empty:
        comparison_frames.append(tuned_comparison)
    full_comparison = sort_model_comparison_table(pd.concat(comparison_frames, ignore_index=True))

    candidate_prototypes = {**base_prototypes, **tuned_prototypes}
    successful_models = set(full_comparison.loc[full_comparison["status"] == "ok", "model"])
    candidate_prototypes = {
        model_name: prototype
        for model_name, prototype in candidate_prototypes.items()
        if model_name in successful_models
    }

    print_section(f"{experiment_name} CV Comparison")
    cv_display_columns = [
        "model",
        "family",
        "source",
        "train_roc_auc_mean",
        "cv_roc_auc_mean",
        "overfit_gap_roc_auc",
        "train_f1_mean",
        "cv_f1_mean",
        "overfit_gap_f1",
        "cv_brier_mean",
        "status",
    ]
    print(full_comparison[cv_display_columns].round(4).to_string(index=False))

    validation_comparison = evaluate_candidates_on_holdout(
        candidate_prototypes=candidate_prototypes,
        comparison_table=full_comparison,
        features_train=train_model_features,
        target_train=target_train,
        features_holdout=valid_model_features,
        target_holdout=target_valid,
        holdout_label="validation",
    )

    print_section(f"{experiment_name} Validation Comparison")
    validation_display_columns = [
        "model",
        "family",
        "source",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "validation_roc_auc",
        "validation_average_precision",
        "validation_brier_score",
        "validation_f1_0.50",
        "overfit_gap_roc_auc",
        "status",
    ]
    print(validation_comparison[validation_display_columns].round(4).to_string(index=False))

    selected_candidate, selection_summary = choose_best_candidate(validation_comparison)
    selected_family = str(
        validation_comparison.loc[validation_comparison["model"] == selected_candidate, "family"].iloc[0]
    )
    selected_prototype = candidate_prototypes[selected_candidate]

    print_section(f"{experiment_name} Selected Model")
    print(f"Selected candidate: {selected_candidate}")
    print(f"Selected family:    {selected_family}")
    print("Selection summary:")
    print(json.dumps(to_serializable(selection_summary), indent=2))

    calibration_summary, _validation_model, validation_probabilities = compare_calibration_modes(
        selected_prototype,
        train_model_features,
        target_train,
        valid_model_features,
        target_valid,
    )

    print_section(f"{experiment_name} Calibration Summary")
    print(json.dumps(to_serializable(calibration_summary), indent=2))

    _, validation_threshold_results = tune_thresholds(
        target_valid,
        validation_probabilities,
        thresholds=THRESHOLD_GRID,
        min_precision_floor=MIN_PRECISION_FLOOR,
        min_recall_floor=MIN_RECALL_FLOOR,
    )
    validation_summary = evaluate_holdout_summary(
        dataset_label="validation",
        target_true=target_valid,
        probabilities=validation_probabilities,
        threshold_results=validation_threshold_results,
        selected_policy=SELECTED_THRESHOLD_POLICY,
    )
    print_holdout_summary(validation_summary)

    stability_summary = evaluate_repeated_cv_stability(
        selected_prototype,
        pd.concat([train_model_features, valid_model_features], axis=0),
        pd.concat([target_train, target_valid], axis=0),
    )
    print_section(f"{experiment_name} Stability Summary")
    print(json.dumps(to_serializable(stability_summary), indent=2))

    fairness_reference_frames = [
        frame for frame in [audit_valid_frame, audit_test_frame] if frame is not None and not frame.empty
    ]
    fairness_reference = (
        pd.concat(fairness_reference_frames, axis=0)
        if fairness_reference_frames
        else pd.DataFrame(index=target_valid.index)
    )
    fairness_group_columns = resolve_fairness_columns(fairness_reference)
    validation_audit = (
        audit_valid_frame.copy()
        if audit_valid_frame is not None and not audit_valid_frame.empty
        else pd.DataFrame(index=target_valid.index)
    )
    validation_fairness_frame, validation_fairness_overall = compute_group_fairness_metrics(
        validation_audit,
        target_valid,
        validation_probabilities,
        validation_summary["selected_threshold"],
        group_columns=fairness_group_columns,
    )
    validation_fairness_actions = derive_fairness_actions(
        validation_fairness_frame,
        validation_fairness_overall,
    )
    validation_fairness_summary = summarize_fairness_assessment(
        validation_fairness_frame,
        validation_fairness_actions,
        validation_fairness_overall,
    )
    print_section(f"{experiment_name} Validation Fairness Summary")
    print(json.dumps(to_serializable(validation_fairness_summary), indent=2))

    artifacts = ExperimentArtifacts()
    test_summary: dict[str, Any] | None = None
    feature_insights: dict[str, Any] | None = None
    test_fairness_summary: dict[str, Any] | None = None
    decision_summary: dict[str, Any] | None = None
    uncertainty_summary: dict[str, Any] | None = None
    decision_examples: list[dict[str, Any]] | None = None
    business_impact_summary: dict[str, Any] | None = None
    decision_boundary_scenarios: list[dict[str, Any]] | None = None
    policy_weight_sensitivity = build_priority_weight_sensitivity(
        selection_target=target_valid,
        selection_probabilities=validation_probabilities,
        evaluation_target=None,
        evaluation_probabilities=None,
    )

    if evaluate_test:
        if test_model_features is None or target_test is None:
            raise ValueError("Test features and target are required when evaluate_test=True.")

        train_valid_features = pd.concat(
            [train_model_features, valid_model_features],
            axis=0,
        )
        train_valid_target = pd.concat([target_train, target_valid], axis=0)

        final_base_model = fit_model_prototype(
            selected_prototype,
            train_valid_features,
            train_valid_target,
        )

        if calibration_summary["selected_mode"] == "calibrated":
            final_deployed_model = fit_calibrated_model(
                selected_prototype,
                train_valid_features,
                train_valid_target,
            )
        else:
            final_deployed_model = final_base_model

        policy_thresholds = {
            policy_name: metrics.threshold
            for policy_name, metrics in validation_threshold_results.items()
        }
        test_probabilities = get_positive_class_probabilities(final_deployed_model, test_model_features)
        test_threshold_results = evaluate_threshold_policies(
            target_test,
            test_probabilities,
            policy_thresholds=policy_thresholds,
        )
        test_summary = evaluate_holdout_summary(
            dataset_label="test",
            target_true=target_test,
            probabilities=test_probabilities,
            threshold_results=test_threshold_results,
            selected_policy=SELECTED_THRESHOLD_POLICY,
        )
        print_holdout_summary(test_summary)
        policy_weight_sensitivity = build_priority_weight_sensitivity(
            selection_target=target_valid,
            selection_probabilities=validation_probabilities,
            evaluation_target=target_test,
            evaluation_probabilities=test_probabilities,
        )

        uncertainty_summary = bootstrap_metric_intervals(
            target_test,
            test_probabilities,
            float(test_summary["selected_threshold"]),
        )
        print_section(f"{experiment_name} Uncertainty Summary")
        print(json.dumps(to_serializable(uncertainty_summary), indent=2))

        test_audit = (
            audit_test_frame.copy()
            if audit_test_frame is not None and not audit_test_frame.empty
            else pd.DataFrame(index=target_test.index)
        )
        test_fairness_frame, test_fairness_overall = compute_group_fairness_metrics(
            test_audit,
            target_test,
            test_probabilities,
            test_summary["selected_threshold"],
            group_columns=fairness_group_columns,
        )
        test_fairness_actions = derive_fairness_actions(
            test_fairness_frame,
            test_fairness_overall,
        )
        test_fairness_summary = summarize_fairness_assessment(
            test_fairness_frame,
            test_fairness_actions,
            test_fairness_overall,
        )
        print_section(f"{experiment_name} Test Fairness Summary")
        print(json.dumps(to_serializable(test_fairness_summary), indent=2))

        if produce_feature_insights:
            feature_insights = print_feature_insights(
                final_base_model,
                train_valid_features,
                train_valid_target,
                top_n=TOP_FEATURES_TO_SHOW,
            )
        else:
            feature_insights = {
                "type": "skipped",
                "reason": "Feature insight generation disabled for this experiment.",
            }

        decision_recommendations, decision_summary = build_decision_recommendations(
            audit_frame=test_audit,
            model_features=test_model_features,
            target_true=target_test,
            probabilities=test_probabilities,
            selected_threshold=float(test_summary["selected_threshold"]),
            fairness_lookup=build_fairness_risk_lookup(validation_fairness_actions),
            explainability_model=final_base_model,
        )
        decision_examples = build_decision_examples(
            decision_recommendations,
            test_model_features,
            final_base_model,
        )
        business_impact_summary = summarize_decision_impact(
            decision_recommendations,
            float(test_summary["selected_threshold"]),
        )
        decision_boundary_scenarios = evaluate_decision_boundary_scenarios(decision_recommendations)

        artifacts = ExperimentArtifacts(
            deployed_model=final_deployed_model,
            explainability_model=final_base_model,
            plot_probabilities=test_probabilities,
            decision_recommendations=decision_recommendations,
        )

    experiment_result = {
        "experiment_name": experiment_name,
        "extra_excluded_columns": extra_excluded_columns,
        "feature_engineering": {
            "created_columns": engineered_columns,
            "created_column_count": len(engineered_columns),
        },
        "feature_pruning": pruning_summary,
        "retained_feature_count": len(retained_columns),
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "class_weight": class_weight,
        "scale_pos_weight": scale_pos_weight,
        "cv_comparison": full_comparison,
        "validation_comparison": validation_comparison,
        "selected_candidate": selected_candidate,
        "selected_model_family": selected_family,
        "selection_summary": selection_summary,
        "selected_threshold_policy": SELECTED_THRESHOLD_POLICY,
        "threshold_policy_rationale": FAID_POLICY_RATIONALE,
        "calibration_summary": calibration_summary,
        "stability_summary": stability_summary,
        "validation_summary": validation_summary,
        "validation_fairness_summary": validation_fairness_summary,
        "test_summary": test_summary,
        "test_fairness_summary": test_fairness_summary,
        "decision_summary": decision_summary,
        "uncertainty_summary": uncertainty_summary,
        "decision_examples": decision_examples,
        "business_impact_summary": business_impact_summary,
        "decision_boundary_scenarios": decision_boundary_scenarios,
        "policy_weight_sensitivity": policy_weight_sensitivity,
        "feature_insights": feature_insights,
    }

    return experiment_result, artifacts


def main() -> None:
    """Run the full modeling workflow end-to-end."""
    try:
        np.random.seed(RANDOM_STATE)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        raw_dataframe = load_data(DATA_PATH)
        labeled_dataframe, binary_target, data_metadata = prepare_target(
            dataframe=raw_dataframe,
            target_column=TARGET_COLUMN,
            positive_labels=POSITIVE_TARGET_LABELS,
            negative_labels=NEGATIVE_TARGET_LABELS,
            excluded_labels=EXCLUDED_TARGET_LABELS,
        )
        audit_frame = build_audit_frame(labeled_dataframe)
        explicit_exclusion_summary = resolve_model_exclusion_summary(
            dataframe=labeled_dataframe,
            target_column=data_metadata["target_column"],
        )
        temporal_split_assessment = assess_temporal_split_feasibility(labeled_dataframe)
        base_feature_frame, excluded_model_input_columns = select_feature_frame(
            dataframe=labeled_dataframe,
            target_column=data_metadata["target_column"],
        )
        baseline_workflow_exclusions = sorted(
            column
            for column in base_feature_frame.columns
            if column in DEFAULT_WORKFLOW_EXCLUSION_COLUMNS
        )

        (
            base_train_features,
            base_valid_features,
            base_test_features,
            target_train,
            target_valid,
            target_test,
        ) = split_dataset(base_feature_frame, binary_target)

        audit_valid_frame = audit_frame.loc[base_valid_features.index].copy()
        audit_test_frame = audit_frame.loc[base_test_features.index].copy()

        split_metadata = {
            "train_rows": int(len(base_train_features)),
            "validation_rows": int(len(base_valid_features)),
            "test_rows": int(len(base_test_features)),
            "train_positive_rate": float(target_train.mean()),
            "validation_positive_rate": float(target_valid.mean()),
            "test_positive_rate": float(target_test.mean()),
            "explicitly_excluded_model_input_columns": excluded_model_input_columns,
            "explicit_model_exclusion_summary": explicit_exclusion_summary,
            "default_workflow_exclusion_columns": baseline_workflow_exclusions,
            "temporal_split_assessment": temporal_split_assessment,
        }

        print_section("Data Summary")
        print(f"Loaded rows: {len(raw_dataframe):,}")
        print(f"Labeled rows used: {len(labeled_dataframe):,}")
        print(f"Rows dropped for missing target: {data_metadata['rows_dropped_missing_target']:,}")
        print(f"Rows dropped for non-binary target labels: {data_metadata['rows_dropped_non_binary_target']:,}")
        print(f"Train rows: {len(base_train_features):,}")
        print(f"Validation rows: {len(base_valid_features):,}")
        print(f"Test rows: {len(base_test_features):,}")
        print(f"Train positive rate: {target_train.mean():.4f}")
        print(f"Validation positive rate: {target_valid.mean():.4f}")
        print(f"Test positive rate: {target_test.mean():.4f}")
        print(f"Explicit model-input exclusions: {len(excluded_model_input_columns)}")
        print(f"Workflow-risk exclusions: {len(baseline_workflow_exclusions)}")

        normal_result, normal_artifacts = run_experiment(
            experiment_name="Normal Workflow",
            base_train_features=base_train_features,
            base_valid_features=base_valid_features,
            base_test_features=base_test_features,
            audit_valid_frame=audit_valid_frame,
            audit_test_frame=audit_test_frame,
            target_train=target_train,
            target_valid=target_valid,
            target_test=target_test,
            extra_excluded_columns=baseline_workflow_exclusions,
            evaluate_test=True,
            produce_feature_insights=True,
        )

        leakage_summary: dict[str, Any] | None = {
            "run": False,
            "note": "Leakage stress test disabled by configuration.",
        }

        if RUN_LEAKAGE_STRESS_TEST:
            suspicious_columns = [
                column
                for column in resolve_suspicious_features(base_feature_frame.columns.tolist())
                if column not in baseline_workflow_exclusions
            ]
            if suspicious_columns:
                stress_result, _ = run_experiment(
                    experiment_name="Leakage Stress Test",
                    base_train_features=base_train_features,
                    base_valid_features=base_valid_features,
                    base_test_features=base_test_features,
                    audit_valid_frame=audit_valid_frame,
                    audit_test_frame=audit_test_frame,
                    target_train=target_train,
                    target_valid=target_valid,
                    target_test=target_test,
                    extra_excluded_columns=suspicious_columns,
                    evaluate_test=True,
                    produce_feature_insights=False,
                )
                leakage_summary = summarize_leakage_stress_test(normal_result, stress_result)

                print_section("Leakage Stress Test Summary")
                print(json.dumps(to_serializable(leakage_summary), indent=2))
            else:
                leakage_summary = {
                    "run": False,
                    "note": "Leakage stress test was requested, but no suspicious features were found.",
                }

        if normal_artifacts.deployed_model is None or normal_artifacts.plot_probabilities is None:
            raise RuntimeError("Normal workflow did not produce a final deployed model.")

        config_summary = build_config_summary()
        model_bundle = build_model_bundle(
            deployed_model=normal_artifacts.deployed_model,
            explainability_model=normal_artifacts.explainability_model,
            config_summary=config_summary,
            data_metadata=data_metadata,
            split_metadata=split_metadata,
            normal_result=normal_result,
        )
        joblib.dump(model_bundle, MODEL_OUTPUT_PATH)
        if normal_artifacts.decision_recommendations is not None:
            normal_artifacts.decision_recommendations.to_csv(
                DECISION_OUTPUT_PATH,
                index=False,
            )
        plot_summary = save_curves(
            target_true=target_test,
            probabilities=normal_artifacts.plot_probabilities,
            title_prefix="Final Test",
        )
        optional_dependencies = dict(OPTIONAL_DEPENDENCY_STATUS)

        metadata_payload = {
            "config": config_summary,
            "data": data_metadata,
            "splits": split_metadata,
            "normal_workflow": normal_result,
            "leakage_stress_test": leakage_summary,
            "optional_dependencies": optional_dependencies,
            "artifacts": {
                "artifact_type": MODEL_ARTIFACT_TYPE,
                "artifact_version": MODEL_ARTIFACT_VERSION,
                "pipeline_path": str(MODEL_OUTPUT_PATH.resolve()),
                "metadata_path": str(METADATA_OUTPUT_PATH.resolve()),
                "report_path": str(REPORT_OUTPUT_PATH.resolve()),
                "decision_output_path": str(DECISION_OUTPUT_PATH.resolve()),
                "roc_curve_path": plot_summary["roc_curve_path"],
                "pr_curve_path": plot_summary["pr_curve_path"],
                "plot_summary": plot_summary,
                "calibration_applied": bool(
                    normal_result["calibration_summary"]["selected_mode"] == "calibrated"
                ),
                "calibration_mode_selected": normal_result["calibration_summary"]["selected_mode"],
                "selected_model_family": normal_result["selected_model_family"],
                "selected_model_name": normal_result["selected_candidate"],
                "selected_threshold_policy": normal_result["selected_threshold_policy"],
                "selected_threshold": normal_result["test_summary"]["selected_threshold"],
                "selection_reason": normal_result["selection_summary"]["selection_reason"],
            },
        }

        with METADATA_OUTPUT_PATH.open("w", encoding="utf-8") as metadata_file:
            json.dump(to_serializable(metadata_payload), metadata_file, indent=2)

        report_text = build_report_text(
            config_summary=config_summary,
            data_metadata=data_metadata,
            split_metadata=split_metadata,
            normal_result=to_serializable(normal_result),
            leakage_summary=to_serializable(leakage_summary),
            optional_dependencies=to_serializable(optional_dependencies),
            plot_summary=to_serializable(plot_summary),
        )
        REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")

        print_section("Optional Dependencies")
        print(json.dumps(to_serializable(optional_dependencies), indent=2))

        print_section("Artifacts Saved")
        print(f"Pipeline: {MODEL_OUTPUT_PATH.resolve()}")
        print(f"Metadata: {METADATA_OUTPUT_PATH.resolve()}")
        print(f"Report:   {REPORT_OUTPUT_PATH.resolve()}")
        print(f"Decisions:{DECISION_OUTPUT_PATH.resolve()}")
        if plot_summary["generated"]:
            print(f"ROC plot: {plot_summary['roc_curve_path']}")
            print(f"PR plot:  {plot_summary['pr_curve_path']}")
        else:
            print(f"Plots:    skipped ({plot_summary['message']})")

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
