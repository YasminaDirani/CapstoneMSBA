from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.source_paths import DEFAULT_DATA_PATH, ROOT_DIR


LEGACY_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "application_type_raw": ("raw_application_type",),
    "application_type_clean": ("inferred_application_type_clean",),
    "application_track": ("inferred_application_track",),
    "application_has_early_merit": ("inferred_application_has_early_merit",),
    "application_type_inconsistent_flag": ("inferred_application_type_inconsistent_flag",),
    "remaining_cap_50_rule_pct": ("inferred_remaining_cap_50_rule_pct",),
    "level": ("parsed_level",),
    "application_term": ("parsed_application_term",),
    "nationality": ("parsed_nationality",),
    "applicant_citizenship": ("parsed_applicant_citizenship",),
    "applicant_is_employed": ("parsed_applicant_is_employed",),
    "applicant_marital_status": ("parsed_applicant_marital_status",),
    "applicant_plan_to_reside": ("parsed_applicant_plan_to_reside",),
    "applicant_position": ("parsed_applicant_position",),
    "father_annual_income_from_work_status": ("parsed_father_annual_income_from_work_status",),
    "father_employer_name": ("parsed_father_employer_name",),
    "father_employment_sector": ("parsed_father_employment_sector",),
    "father_ever_worked": ("parsed_father_ever_worked",),
    "father_gross_income": ("parsed_father_gross_income",),
    "father_income_document": ("parsed_father_income_document",),
    "father_income_document_status": ("qa_father_income_document_status",),
    "father_income_document_suspicious_flag": ("qa_father_income_document_review_flag",),
    "father_job_info_category": ("parsed_father_job_info_category",),
    "father_job_info_clean": ("parsed_father_job_info_clean",),
    "father_net_income": ("parsed_father_net_income",),
    "father_position": ("parsed_father_position",),
    "father_second_occupation_flag": ("parsed_father_second_occupation_flag",),
    "father_self_employed_name": ("parsed_father_self_employed_name",),
    "father_status": ("parsed_father_status",),
    "father_unemployment_date": ("parsed_father_unemployment_date",),
    "father_work_status_benefits_total": ("parsed_father_work_status_benefits_total",),
    "financial_assistants_count": ("parsed_financial_assistants_count",),
    "financial_assistants_total_est_annual_amount": (
        "parsed_financial_assistants_total_est_annual_amount",
    ),
    "investments_count": ("parsed_investments_count",),
    "investments_total_annual_profit": ("parsed_investments_total_annual_profit",),
    "dependents_count": ("parsed_dependents_count",),
    "loans_count": ("parsed_loans_count",),
    "loans_max_remaining_years": ("parsed_loans_max_remaining_years",),
    "loans_total_amount": ("parsed_loans_total_amount",),
    "loans_total_monthly_payment": ("parsed_loans_total_monthly_payment",),
    "loans_total_remaining_balance": ("parsed_loans_total_remaining_balance",),
    "merit_comment": ("parsed_merit_comment",),
    "merit_hist_pct": ("parsed_merit_hist_pct",),
    "merit_pct": ("parsed_merit_pct",),
    "mother_annual_income_from_work_status": ("parsed_mother_annual_income_from_work_status",),
    "mother_employer_name": ("parsed_mother_employer_name",),
    "mother_employment_sector": ("parsed_mother_employment_sector",),
    "mother_ever_worked": ("parsed_mother_ever_worked",),
    "mother_gross_income": ("parsed_mother_gross_income",),
    "mother_income_document": ("parsed_mother_income_document",),
    "mother_income_document_status": ("qa_mother_income_document_status",),
    "mother_job_info_category": ("parsed_mother_job_info_category",),
    "mother_job_info_clean": ("parsed_mother_job_info_clean",),
    "mother_net_income": ("parsed_mother_net_income",),
    "mother_position": ("parsed_mother_position",),
    "mother_second_occupation_flag": ("parsed_mother_second_occupation_flag",),
    "mother_self_employed_name": ("parsed_mother_self_employed_name",),
    "mother_status": ("parsed_mother_status",),
    "mother_unemployment_date": ("parsed_mother_unemployment_date",),
    "mother_work_status_benefits_total": ("parsed_mother_work_status_benefits_total",),
    "need_comment": ("parsed_need_comment",),
    "need_pct": ("parsed_need_pct",),
    "decision": ("parsed_decision",),
    "bin_status": ("parsed_bin_status",),
    "over_and_above_decision": ("parsed_over_and_above_decision",),
    "over_and_above_percentage_awarded": ("parsed_over_and_above_percentage_awarded",),
    "properties_count": ("parsed_properties_count",),
    "properties_mortgaged_count": ("parsed_properties_mortgaged_count",),
    "properties_planted_income_total": ("parsed_properties_planted_income_total",),
    "properties_rented_income_total": ("parsed_properties_rented_income_total",),
    "properties_total_area": ("parsed_properties_total_area",),
    "properties_total_estimated_value": ("parsed_properties_total_estimated_value",),
    "property_types": ("parsed_property_types",),
    "school": ("parsed_school",),
    "school_was_missing": ("qa_school_was_missing", "inferred_school_filled_for_ops_only"),
    "siblings_at_aub_count": ("parsed_siblings_at_aub_count",),
    "siblings_not_at_aub_count": ("parsed_siblings_not_at_aub_count",),
    "siblings_other_total_financial_assistance": (
        "parsed_siblings_other_total_financial_assistance",
    ),
    "siblings_other_total_tuition": ("parsed_siblings_other_total_tuition",),
    "source_of_income_has_record": ("parsed_source_of_income_has_record",),
    "source_of_income_type": ("parsed_source_of_income_type",),
    "source_of_income_work_status": ("parsed_source_of_income_work_status",),
    "special_family_circumstances_category": ("parsed_special_family_circumstances_category",),
    "special_family_circumstances_detail": ("parsed_special_family_circumstances_detail",),
    "spouse_citizenship": ("parsed_spouse_citizenship",),
    "submission_date": ("parsed_submission_date",),
    "travel_records_category": ("parsed_travel_records_category",),
    "travel_records_clean": ("parsed_travel_records_clean",),
    "travel_records_missing_flag": ("parsed_travel_records_missing_flag",),
    "car_models": ("parsed_car_models",),
    "cars_count": ("parsed_cars_count",),
    "cars_mortgaged_count": ("parsed_cars_mortgaged_count",),
    "certificate_ownership_clean": ("parsed_certificate_ownership_clean",),
    "certificate_ownership_status": ("parsed_certificate_ownership_status",),
    "consent_to_share_information": ("parsed_consent_to_share_information",),
    "faid_missing_documents": ("parsed_faid_missing_documents",),
    "faid_missing_documents_count": ("parsed_faid_missing_documents_count",),
}


def _first_available_series(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
) -> pd.Series | None:
    for column in candidates:
        if column in dataframe.columns:
            return dataframe[column]
    return None


def _coalesce_numeric(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
) -> pd.Series:
    output = pd.Series(np.nan, index=dataframe.index, dtype=float)
    for column in candidates:
        if column not in dataframe.columns:
            continue
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        output = output.where(output.notna(), numeric)
    return output


def _build_parent_income_series(dataframe: pd.DataFrame) -> pd.Series:
    father_income = _coalesce_numeric(
        dataframe,
        ("father_gross_income", "father_net_income"),
    )
    mother_income = _coalesce_numeric(
        dataframe,
        ("mother_gross_income", "mother_net_income"),
    )
    income_frame = pd.concat(
        [father_income.rename("father"), mother_income.rename("mother")],
        axis=1,
    )
    return income_frame.sum(axis=1, min_count=1)


def _sum_numeric_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    existing_columns = [column for column in columns if column in dataframe.columns]
    if not existing_columns:
        return pd.Series(np.nan, index=dataframe.index, dtype=float)

    numeric_frame = dataframe[existing_columns].apply(pd.to_numeric, errors="coerce")
    return numeric_frame.sum(axis=1, min_count=1)


def prepare_dashboard_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add legacy compatibility aliases without removing the enhanced source columns."""
    dashboard_df = dataframe.copy()
    derived_columns: dict[str, pd.Series] = {}

    for alias, candidates in LEGACY_ALIAS_MAP.items():
        if alias in dashboard_df.columns or alias in derived_columns:
            continue
        source_series = _first_available_series(dashboard_df, candidates)
        if source_series is not None:
            derived_columns[alias] = source_series

    if "has_over_and_above" not in dashboard_df.columns:
        if "inferred_has_over_and_above" in dashboard_df.columns:
            derived_columns["has_over_and_above"] = pd.to_numeric(
                dashboard_df["inferred_has_over_and_above"],
                errors="coerce",
            )
        else:
            percentage_awarded = pd.to_numeric(
                dashboard_df.get("over_and_above_percentage_awarded"),
                errors="coerce",
            )
            derived_columns["has_over_and_above"] = percentage_awarded.fillna(0).gt(0).astype(int)

    if "total_parent_income" not in dashboard_df.columns:
        derived_columns["total_parent_income"] = _build_parent_income_series(dashboard_df)

    if "total_siblings" not in dashboard_df.columns:
        derived_columns["total_siblings"] = _sum_numeric_columns(
            dashboard_df,
            ("siblings_at_aub_count", "siblings_not_at_aub_count"),
        )

    if derived_columns:
        dashboard_df = pd.concat(
            [dashboard_df, pd.DataFrame(derived_columns, index=dashboard_df.index)],
            axis=1,
        )

    return dashboard_df


def resolve_project_path(file_path: str | Path) -> Path:
    """Resolve repo-relative inputs while preserving absolute configured paths."""
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


@st.cache_data(show_spinner=False)
def load_csv(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    *,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame and cache the result."""
    return pd.read_csv(resolve_project_path(csv_path), low_memory=low_memory)


@st.cache_data(show_spinner=False)
def load_json(json_path: str | Path) -> dict[str, object]:
    """Load a JSON mapping from disk and cache the parsed payload."""
    with resolve_project_path(json_path).open("r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    return payload if isinstance(payload, dict) else {}


def load_optional_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV when present, otherwise return an empty frame."""
    path = resolve_project_path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    return load_csv(path)


@st.cache_data(show_spinner=False)
def load_dashboard_data(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the dataset and add compatibility fields used by the dashboard."""
    return prepare_dashboard_dataframe(load_csv(csv_path))
