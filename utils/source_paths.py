from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get(
        "YASMINA_SOURCE_ROOT",
        ROOT_DIR.parent / "Yasmina-MSBA",
    )
).expanduser()


def _resolve_configured_path(env_var: str) -> Path | None:
    configured = os.environ.get(env_var)
    if not configured:
        return None

    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT_DIR / candidate).resolve()
    return candidate


def _pick_default_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def resolve_artifact_path(env_var: str, *candidates: Path) -> Path:
    configured = _resolve_configured_path(env_var)
    if configured is not None:
        return configured
    return _pick_default_path(*candidates)


SOURCE_ROOT = (
    _resolve_configured_path("YASMINA_SOURCE_ROOT") or DEFAULT_SOURCE_ROOT
)

DEFAULT_DATA_PATH = resolve_artifact_path(
    "YASMINA_DATA_PATH",
    SOURCE_ROOT / "cleaned data" / "faid_cleaned.csv",
    ROOT_DIR / "cleaned data" / "faid_cleaned.csv",
    ROOT_DIR / "faid_cleaned.csv",
)

DEFAULT_DATA_SUMMARY_PATH = resolve_artifact_path(
    "YASMINA_DATA_SUMMARY_PATH",
    SOURCE_ROOT / "cleaned data" / "faid_cleaned_summary.json",
    ROOT_DIR / "cleaned data" / "faid_cleaned_summary.json",
    ROOT_DIR / "faid_cleaned_summary.json",
)

DEFAULT_DATA_DICTIONARY_PATH = resolve_artifact_path(
    "YASMINA_DATA_DICTIONARY_PATH",
    SOURCE_ROOT / "cleaned data" / "faid_cleaned_data_dictionary.json",
    ROOT_DIR / "cleaned data" / "faid_cleaned_data_dictionary.json",
    ROOT_DIR / "faid_cleaned_data_dictionary.json",
)

DEFAULT_REVIEW_REQUIRED_PATH = resolve_artifact_path(
    "YASMINA_REVIEW_REQUIRED_PATH",
    SOURCE_ROOT / "cleaned data" / "faid_cleaned_review_required.csv",
    ROOT_DIR / "cleaned data" / "faid_cleaned_review_required.csv",
    ROOT_DIR / "faid_cleaned_review_required.csv",
)

DEFAULT_ISSUES_PATH = resolve_artifact_path(
    "YASMINA_ISSUES_PATH",
    SOURCE_ROOT / "cleaned data" / "faid_cleaned_issues.csv",
    ROOT_DIR / "cleaned data" / "faid_cleaned_issues.csv",
    ROOT_DIR / "faid_cleaned_issues.csv",
)

DEFAULT_LOANS_PATH = resolve_artifact_path(
    "YASMINA_LOANS_PATH",
    SOURCE_ROOT / "cleaned data" / "faid_cleaned_loans.csv",
    ROOT_DIR / "cleaned data" / "faid_cleaned_loans.csv",
    ROOT_DIR / "faid_cleaned_loans.csv",
)

DEFAULT_PROPERTIES_PATH = resolve_artifact_path(
    "YASMINA_PROPERTIES_PATH",
    SOURCE_ROOT / "cleaned data" / "faid_cleaned_properties.csv",
    ROOT_DIR / "cleaned data" / "faid_cleaned_properties.csv",
    ROOT_DIR / "faid_cleaned_properties.csv",
)

DEFAULT_FINANCIAL_ASSISTANTS_PATH = resolve_artifact_path(
    "YASMINA_FINANCIAL_ASSISTANTS_PATH",
    SOURCE_ROOT / "cleaned data" / "faid_cleaned_financial_assistants.csv",
    ROOT_DIR / "cleaned data" / "faid_cleaned_financial_assistants.csv",
    ROOT_DIR / "faid_cleaned_financial_assistants.csv",
)

DEFAULT_CARS_PATH = resolve_artifact_path(
    "YASMINA_CARS_PATH",
    SOURCE_ROOT / "cleaned data" / "faid_cleaned_cars.csv",
    ROOT_DIR / "cleaned data" / "faid_cleaned_cars.csv",
    ROOT_DIR / "faid_cleaned_cars.csv",
)

DEFAULT_MODEL_PATH = resolve_artifact_path(
    "YASMINA_MODEL_PATH",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_eligibility_notebook"
    / "yasmina_eligibility_pipeline.joblib",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_eligibility"
    / "yasmina_eligibility_pipeline.joblib",
    ROOT_DIR
    / "artifacts"
    / "yasmina_eligibility_notebook"
    / "yasmina_eligibility_pipeline.joblib",
    ROOT_DIR
    / "artifacts"
    / "yasmina_eligibility"
    / "yasmina_eligibility_pipeline.joblib",
)

DEFAULT_MODEL_METADATA_PATH = resolve_artifact_path(
    "YASMINA_MODEL_METADATA_PATH",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_eligibility_notebook"
    / "yasmina_eligibility_metadata.json",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_eligibility"
    / "yasmina_eligibility_metadata.json",
    ROOT_DIR
    / "artifacts"
    / "yasmina_eligibility_notebook"
    / "yasmina_eligibility_metadata.json",
    ROOT_DIR
    / "artifacts"
    / "yasmina_eligibility"
    / "yasmina_eligibility_metadata.json",
)

DEFAULT_ELIGIBILITY_DECISION_PATH = resolve_artifact_path(
    "YASMINA_ELIGIBILITY_DECISION_PATH",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_eligibility_notebook"
    / "yasmina_eligibility_decision_recommendations.csv",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_eligibility"
    / "yasmina_eligibility_decision_recommendations.csv",
    ROOT_DIR
    / "artifacts"
    / "yasmina_eligibility_notebook"
    / "yasmina_eligibility_decision_recommendations.csv",
    ROOT_DIR
    / "artifacts"
    / "yasmina_eligibility"
    / "yasmina_eligibility_decision_recommendations.csv",
)

DEFAULT_MODEL_1_SUMMARY_PATH = resolve_artifact_path(
    "YASMINA_MODEL_1_SUMMARY_PATH",
    ROOT_DIR / "artifacts" / "model_1" / "model_1_summary.json",
)

DEFAULT_PERCENTAGE_METADATA_PATH = resolve_artifact_path(
    "YASMINA_PERCENTAGE_METADATA_PATH",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_metadata.json",
    ROOT_DIR
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_metadata.json",
)

DEFAULT_PERCENTAGE_DECISION_RECOMMENDATIONS_PATH = resolve_artifact_path(
    "YASMINA_PERCENTAGE_DECISION_PATH",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_test_decision_recommendations.csv",
    ROOT_DIR
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_test_decision_recommendations.csv",
)

DEFAULT_PERCENTAGE_TEST_PREDICTIONS_PATH = resolve_artifact_path(
    "YASMINA_PERCENTAGE_TEST_PREDICTIONS_PATH",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_test_predictions.csv",
    ROOT_DIR
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_test_predictions.csv",
)

DEFAULT_PERCENTAGE_VALIDATION_CALIBRATION_BINS_PATH = resolve_artifact_path(
    "YASMINA_PERCENTAGE_VALIDATION_CALIBRATION_BINS_PATH",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_validation_calibration_bins.csv",
    ROOT_DIR
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_validation_calibration_bins.csv",
)

DEFAULT_PERCENTAGE_TEST_CALIBRATION_BINS_PATH = resolve_artifact_path(
    "YASMINA_PERCENTAGE_TEST_CALIBRATION_BINS_PATH",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_test_calibration_bins.csv",
    ROOT_DIR
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_test_calibration_bins.csv",
)

DEFAULT_PERCENTAGE_FAIRNESS_METRICS_PATH = resolve_artifact_path(
    "YASMINA_PERCENTAGE_FAIRNESS_PATH",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_test_fairness_metrics.csv",
    ROOT_DIR
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_test_fairness_metrics.csv",
)

DEFAULT_PERCENTAGE_FAIRNESS_ACTIONS_PATH = resolve_artifact_path(
    "YASMINA_PERCENTAGE_FAIRNESS_ACTIONS_PATH",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_validation_fairness_actions.csv",
    ROOT_DIR
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_validation_fairness_actions.csv",
)

DEFAULT_PERCENTAGE_STAGE1_IMPORTANCE_PATH = resolve_artifact_path(
    "YASMINA_PERCENTAGE_STAGE1_IMPORTANCE_PATH",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_shap_stage1_global.csv",
    ROOT_DIR
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_shap_stage1_global.csv",
)

DEFAULT_PERCENTAGE_STAGE2_IMPORTANCE_PATH = resolve_artifact_path(
    "YASMINA_PERCENTAGE_STAGE2_IMPORTANCE_PATH",
    SOURCE_ROOT
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_shap_stage2_global.csv",
    ROOT_DIR
    / "artifacts"
    / "yasmina_percentage"
    / "yasmina_percentage_shap_stage2_global.csv",
)

DEFAULT_POC_EXPERIMENT_PATH = resolve_artifact_path(
    "YASMINA_POC_EXPERIMENT_PATH",
    SOURCE_ROOT / "faid_models" / "POC" / "purchasing_power_experiment.py",
    ROOT_DIR / "faid_models" / "POC" / "purchasing_power_experiment.py",
)

DEFAULT_POC_NOTEBOOK_PATH = resolve_artifact_path(
    "YASMINA_POC_NOTEBOOK_PATH",
    SOURCE_ROOT
    / "faid_models"
    / "POC"
    / "notebooks"
    / "purchasing_power_experiment_notebook.ipynb",
    ROOT_DIR
    / "faid_models"
    / "POC"
    / "notebooks"
    / "purchasing_power_experiment_notebook.ipynb",
)
