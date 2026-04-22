from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils.chart_insights import (
    award_rate_group_insights,
    correlation_heatmap_insights,
    decision_distribution_insights,
    driver_side_insights,
    feature_importance_insights,
    numeric_decision_comparison_insights,
    school_concentration_insights,
)
from utils.data_loader import DEFAULT_DATA_PATH, load_csv
from utils.decision_analytics import prepare_decision_analytics_frame, summarize_numeric_drivers
from utils.model_explainability import (
    extract_top_feature_drivers,
    extract_top_feature_importances,
)
from utils.prediction_utils import add_prediction_probabilities
from utils.ui import (
    ACCENT_COLOR,
    DANGER_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    add_top_n_flag,
    apply_design_system,
    build_color_condition,
    render_insight_action_panel,
    render_kpi_row,
    render_page_header,
)

try:
    from utils.model_loader import DEFAULT_MODEL_PATH, load_model
except Exception:
    DEFAULT_MODEL_PATH = None
    load_model = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_METADATA_PATH = (
    ROOT_DIR / "artifacts" / "yasmina_eligibility" / "yasmina_eligibility_metadata.json"
)
DEFAULT_MODEL_1_SUMMARY_PATH = ROOT_DIR / "artifacts" / "model_1" / "model_1_summary.json"
LOCAL_MODEL_1_NOTEBOOK_PATH = os.environ.get("YASMINA_MODEL_1_NOTEBOOK_PATH")

PROFILE_METRIC_OPTIONS: dict[str, tuple[str, str, str]] = {
    "Total Parent Income": (
        "total_parent_income",
        "Total Parent Income",
        "Derived from effective father and mother income values when available.",
    ),
    "Dependents Count": (
        "dependents_count",
        "Dependents Count",
        "Higher counts typically indicate a heavier household support burden.",
    ),
    "Total Siblings": (
        "total_siblings",
        "Total Siblings",
        "Computed as siblings at AUB plus siblings not at AUB.",
    ),
    "Estimated Property Value": (
        "properties_total_estimated_value",
        "Estimated Property Value",
        "Property values provide an asset-side view of household capacity.",
    ),
    "Other Siblings Tuition": (
        "siblings_other_total_tuition",
        "Other Siblings Tuition",
        "Captures tuition obligations for siblings enrolled elsewhere.",
    ),
    "Remaining Loan Balance": (
        "loans_total_remaining_balance",
        "Remaining Loan Balance",
        "Represents outstanding liabilities when reported.",
    ),
}

KEY_VARIABLE_SPECS = [
    (
        "total_parent_income",
        "Total Parent Income",
        "Core affordability proxy built from parent income fields.",
    ),
    (
        "properties_total_estimated_value",
        "Estimated Property Value",
        "Asset-side measure of household wealth and fallback capacity.",
    ),
    (
        "dependents_count",
        "Dependents Count",
        "Household burden indicator used in need-oriented interpretation.",
    ),
    (
        "total_siblings",
        "Total Siblings",
        "Captures the broader household education burden.",
    ),
    (
        "siblings_other_total_tuition",
        "Other Siblings Tuition",
        "Measures competing tuition commitments outside AUB.",
    ),
    (
        "loans_total_remaining_balance",
        "Remaining Loan Balance",
        "Represents outstanding liabilities when reported.",
    ),
    (
        "special_family_circumstances_category",
        "Special Circumstances",
        "Adds contextual hardship signals beyond financial quantities.",
    ),
    (
        "school",
        "School",
        "Supports representation and concentration analysis across applicant sources.",
    ),
]


USE_LIMITATION_CUES = (
    "approvals are more common than denials",
    "missing decisions",
    "non-standard",
    "not fully normalized",
    "not evenly distributed",
    "materially shape overall patterns",
    "modest difference rather than a clean split",
    "borderline rather than clearly separable",
    "overlapping information",
    "trade-off",
    "exploratory scores rather than as held-out performance evidence",
    "relatively concentrated rather than evenly spread",
)

DATA_BOUNDARIES = (
    "No causal claims.",
    "No income-based conclusions.",
    "No fairness claims.",
)

MODEL_BOUNDARIES = (
    "No causal claims.",
    "No income-based conclusions.",
    "No fairness claims.",
    "No production prediction.",
)

POPULATION_BOUNDARIES = (
    "No causal claims.",
    "No decision-driver claims.",
    "No fairness claims.",
    "No subgroup ranking claims beyond this sample.",
)

CATALOG_BOUNDARIES = (
    "No causal claims.",
    "No affordability conclusions from missingness alone.",
    "No automatic award or deny decisions from preview scores.",
    "No fairness claims.",
)


def split_chart_guidance(lines: list[str]) -> tuple[list[str], list[str]]:
    """Group chart takeaways into what the dataset supports versus where it is limited."""
    usable_lines: list[str] = []
    limited_lines: list[str] = []

    for line in lines:
        normalized = line.lower()
        if any(cue in normalized for cue in USE_LIMITATION_CUES):
            limited_lines.append(line)
        else:
            usable_lines.append(line)

    if lines and not usable_lines:
        usable_lines.append(
            "Quick descriptive reads inside the cleaned slice shown here."
        )
    if lines and not limited_lines:
        limited_lines.append(
            "Causal, fairness, or policy claims beyond what this chart actually measures."
        )

    return usable_lines, limited_lines


def render_chart_insights(
    lines: list[str],
    *,
    title: str | None = None,
    boundaries: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Render chart guidance as takeaway, capability, and limitation."""
    if not lines:
        return

    usable_lines, limited_lines = split_chart_guidance(lines)
    if title:
        st.markdown(f"**{title}**")

    takeaway = usable_lines[0] if usable_lines else limited_lines[0]
    st.caption(f"Takeaway: {takeaway}")

    remaining_usable = usable_lines[1:] if usable_lines else []
    if remaining_usable:
        st.markdown("**Can Support**")
        for line in remaining_usable:
            st.write(f"- {line}")

    boundary_lines = list(boundaries) if boundaries is not None else limited_lines
    if boundary_lines:
        st.markdown("**Cannot Support**")
        for line in boundary_lines:
            st.write(f"- {line}")


def _as_dict(value: object) -> dict[str, object]:
    """Return a mapping value or an empty dict when the payload is missing."""
    return value if isinstance(value, dict) else {}


def classify_feature_family(column_name: str) -> str:
    """Assign each column to a thesis-friendly feature family."""
    name = column_name.strip().lower()

    if name.startswith("father_"):
        return "Father Profile"
    if name.startswith("mother_"):
        return "Mother Profile"
    if (
        name.startswith("application_")
        or name.startswith("applicant_")
        or name in {"level", "nationality", "school", "school_was_missing", "spouse_citizenship"}
    ):
        return "Application & Demographics"
    if name.startswith("siblings_") or name.startswith("dependents_"):
        return "Household Composition"
    if (
        name.startswith("financial_assistants_")
        or name.startswith("investments_")
        or name.startswith("source_of_income")
    ):
        return "Income, Assistance & Investments"
    if (
        name.startswith("special_family_circumstances_")
        or name.startswith("travel_records_")
        or name.startswith("certificate_ownership_")
    ):
        return "Context & Documentation"
    if name.startswith("loans_"):
        return "Loans & Liabilities"
    if (
        name.startswith("properties_")
        or name.startswith("property_")
        or name.startswith("cars_")
        or name.startswith("car_")
    ):
        return "Assets & Property"
    if (
        name == "decision"
        or name == "bin_status"
        or name.startswith("need_")
        or name.startswith("merit_")
        or name.startswith("submission_")
        or name.startswith("consent_")
        or name.startswith("faid_")
        or name.startswith("over_and_above_")
    ):
        return "Administrative & Outcomes"
    return "Other / Derived"


def classify_dtype_group(series: pd.Series) -> str:
    """Condense pandas dtypes into readable schema buckets."""
    if pd.api.types.is_bool_dtype(series):
        return "Boolean / Flag"
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"
    return "Text / Categorical"


def build_column_catalog(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build a field-level schema catalog with completeness metrics."""
    rows: list[dict[str, object]] = []
    row_count = max(len(dataframe), 1)

    for column in dataframe.columns:
        series = dataframe[column]
        non_null_count = int(series.notna().sum())
        rows.append(
            {
                "column": column,
                "family": classify_feature_family(column),
                "dtype": str(series.dtype),
                "dtype_group": classify_dtype_group(series),
                "non_null_count": non_null_count,
                "non_null_pct": non_null_count / row_count * 100,
                "missing_pct": (row_count - non_null_count) / row_count * 100,
                "distinct_values": int(series.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["family", "missing_pct", "column"],
        ascending=[True, False, True],
        ignore_index=True,
    )


def build_decision_audit_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Audit outcome labels into comparable, missing, and non-standard buckets."""
    if "decision" not in dataframe.columns:
        return pd.DataFrame(columns=["decision_label", "status", "count", "share_pct"])

    display_labels = (
        dataframe["decision"]
        .astype("string")
        .fillna("Missing")
        .str.strip()
        .replace("", "Missing")
    )
    normalized_labels = display_labels.str.lower()

    status = pd.Series("Non-standard outcome", index=dataframe.index, dtype="string")
    status = status.mask(normalized_labels.isin(["awarded", "denied"]), "Comparable outcome")
    status = status.mask(display_labels.eq("Missing"), "Missing outcome")

    audit_frame = (
        pd.DataFrame({"decision_label": display_labels, "status": status})
        .groupby(["decision_label", "status"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    audit_frame["share_pct"] = audit_frame["count"] / max(len(dataframe), 1) * 100
    status_order = {
        "Comparable outcome": 0,
        "Missing outcome": 1,
        "Non-standard outcome": 2,
    }
    audit_frame["status_order"] = audit_frame["status"].map(status_order).fillna(99)
    return audit_frame.sort_values(
        by=["status_order", "count", "decision_label"],
        ascending=[True, False, True],
        ignore_index=True,
    ).drop(columns=["status_order"])


def build_feature_family_summary(column_catalog: pd.DataFrame) -> pd.DataFrame:
    """Summarize the schema by feature family."""
    if column_catalog.empty:
        return pd.DataFrame(columns=["family", "field_count"])

    summary = (
        column_catalog.groupby("family", as_index=False)
        .agg(field_count=("column", "size"))
        .sort_values("field_count", ascending=False, ignore_index=True)
    )
    return summary


def build_missingness_summary(column_catalog: pd.DataFrame, *, top_n: int = 12) -> pd.DataFrame:
    """Return the most incomplete fields for data-quality review."""
    if column_catalog.empty:
        return pd.DataFrame(columns=["column", "missing_pct", "non_null_count"])

    return (
        column_catalog[column_catalog["missing_pct"] > 0]
        .sort_values(["missing_pct", "column"], ascending=[False, True], ignore_index=True)
        .head(top_n)
        .copy()
    )


def build_key_variable_coverage(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Track completeness for variables that matter most to the thesis story."""
    if labeled_df.empty:
        return pd.DataFrame(columns=["variable", "coverage_pct", "available_records", "role"])

    rows: list[dict[str, object]] = []
    for column, label, role in KEY_VARIABLE_SPECS:
        if column not in labeled_df.columns:
            continue

        series = labeled_df[column]
        if pd.api.types.is_numeric_dtype(series):
            available_mask = pd.to_numeric(series, errors="coerce").notna()
        else:
            available_mask = (
                series.astype("string").str.strip().replace("", pd.NA).notna()
            )

        available_records = int(available_mask.sum())
        rows.append(
            {
                "variable": label,
                "coverage_pct": available_records / max(len(labeled_df), 1) * 100,
                "available_records": available_records,
                "role": role,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["coverage_pct", "variable"],
        ascending=[False, True],
        ignore_index=True,
    )


def build_submission_year_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize application volume and comparability by submission year."""
    if "submission_date" not in dataframe.columns:
        return pd.DataFrame(
            columns=[
                "submission_year",
                "applications",
                "comparable_applications",
                "comparable_share_pct",
            ]
        )

    working = dataframe.copy()
    working["submission_year"] = pd.to_datetime(
        working["submission_date"],
        errors="coerce",
    ).dt.year
    working = working.dropna(subset=["submission_year"]).copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "submission_year",
                "applications",
                "comparable_applications",
                "comparable_share_pct",
            ]
        )

    working["submission_year"] = working["submission_year"].astype(int)
    decision_clean = (
        working["decision"].astype("string").fillna("").str.strip().str.lower()
        if "decision" in working.columns
        else pd.Series("", index=working.index, dtype="string")
    )
    working["is_comparable"] = decision_clean.isin(["awarded", "denied"])

    summary = (
        working.groupby("submission_year", as_index=False)
        .agg(
            applications=("submission_year", "size"),
            comparable_applications=("is_comparable", "sum"),
        )
        .sort_values("submission_year", ignore_index=True)
    )
    summary["comparable_share_pct"] = (
        summary["comparable_applications"] / summary["applications"] * 100
    )
    return summary


def build_application_term_summary(dataframe: pd.DataFrame, *, top_n: int = 8) -> pd.DataFrame:
    """Summarize the main academic-term buckets in the raw data."""
    if "application_term" not in dataframe.columns:
        return pd.DataFrame(columns=["application_term", "applications", "share_pct"])

    term_series = (
        dataframe["application_term"]
        .astype("string")
        .fillna("Missing")
        .str.replace(".0", "", regex=False)
        .str.strip()
        .replace("", "Missing")
    )
    summary = (
        term_series.value_counts()
        .head(top_n)
        .rename_axis("application_term")
        .reset_index(name="applications")
    )
    summary["share_pct"] = summary["applications"] / max(len(dataframe), 1) * 100
    return summary


def lookup_variable_coverage(
    key_variable_coverage: pd.DataFrame,
    variable_label: str,
) -> float:
    """Fetch the coverage percentage for one named thesis variable."""
    if key_variable_coverage.empty:
        return float("nan")

    match = key_variable_coverage.loc[
        key_variable_coverage["variable"] == variable_label,
        "coverage_pct",
    ]
    return float(match.iloc[0]) if not match.empty else float("nan")


def build_framing_principles() -> list[dict[str, str]]:
    """State the core reading rules in a principle-first format."""
    return [
        {
            "principle": "Treat one application record as one analytical unit.",
            "why": "This page compares record-level decisions, not household causality.",
        },
        {
            "principle": "Use only standardized awarded/denied outcomes for direct comparisons.",
            "why": "It keeps missing and non-standard labels from distorting rates.",
        },
        {
            "principle": "Separate measurement quality from predictive evidence.",
            "why": "Interesting fields can still be too incomplete for strong inference.",
        },
        {
            "principle": "Treat in-app model scores as exploratory unless saved holdout evidence exists.",
            "why": "Without holdout results, scores help exploration but not validation.",
        },
    ]


def build_submission_year_takeaway(submission_year_summary: pd.DataFrame) -> str:
    """Return one short takeaway for the submission-year chart."""
    if submission_year_summary.empty:
        return "Takeaway unavailable because submission dates are missing."

    major_years = submission_year_summary[submission_year_summary["applications"] >= 100]
    reference_years = major_years if not major_years.empty else submission_year_summary
    first_year = int(reference_years.iloc[0]["submission_year"])
    last_year = int(reference_years.iloc[-1]["submission_year"])
    return (
        f"Reliable time comparison sits in {first_year}-{last_year}; thin edge years should not drive trend claims."
    )


def build_application_term_takeaway(application_term_summary: pd.DataFrame) -> str:
    """Return one short takeaway for the application-term chart."""
    if application_term_summary.empty:
        return "Takeaway unavailable because application terms are missing."

    top_term = application_term_summary.iloc[0]
    return (
        f"{top_term['application_term']} sets the baseline because it holds {float(top_term['share_pct']) / 100:.1%} of all records."
    )


def build_key_variable_takeaway(key_variable_coverage: pd.DataFrame) -> str:
    """Return one short takeaway for the key-variable coverage chart."""
    if key_variable_coverage.empty:
        return "Takeaway unavailable because comparable records are missing."

    weakest = key_variable_coverage.sort_values("coverage_pct").iloc[0]
    strongest = key_variable_coverage.sort_values("coverage_pct", ascending=False).iloc[0]
    return (
        f"Coverage is split: {strongest['variable']} is fully observed, but {weakest['variable']} is only {float(weakest['coverage_pct']) / 100:.1%} complete."
    )


def build_feature_family_takeaway(feature_family_summary: pd.DataFrame) -> str:
    """Return one short takeaway for the feature-family chart."""
    if feature_family_summary.empty:
        return "Takeaway unavailable because the schema summary is missing."

    top_family = feature_family_summary.iloc[0]
    return (
        f"{top_family['family']} is the biggest schema block, so the file sees the process more through profile structure than hard affordability."
    )


def build_missingness_takeaway(column_catalog: pd.DataFrame) -> str:
    """Return one short takeaway for the missingness chart."""
    income_rows = column_catalog[
        column_catalog["column"].str.contains("income", case=False, na=False)
    ]
    income_high_missing = int((income_rows["missing_pct"] >= 90).sum()) if not income_rows.empty else 0
    if income_high_missing >= 3:
        return (
            f"{income_high_missing} income-related fields are 90%+ missing, so direct affordability measurement is unreliable."
        )

    top_missing = column_catalog.sort_values("missing_pct", ascending=False, ignore_index=True).iloc[0]
    return (
        f"Missingness is concentrated, not diffuse: {top_missing['column']} is {float(top_missing['missing_pct']) / 100:.1%} missing."
    )


def build_confusion_matrix_takeaway(confusion_df: pd.DataFrame) -> str:
    """Return one short takeaway for the confusion matrix."""
    if confusion_df.empty:
        return "Takeaway unavailable because the holdout confusion matrix is missing."

    ranked = confusion_df.sort_values("count", ascending=False, ignore_index=True)
    top_cell = ranked.iloc[0]
    return (
        f"The holdout matrix is led by {top_cell['outcome']} ({int(top_cell['count']):,}), so error balance matters more than accuracy alone."
    )


def build_population_guardrail(overview: dict[str, object], total_rows: int) -> dict[str, str]:
    """State the outcome guardrail for the Population & Outcomes tab."""
    cleaned_rows = int(overview.get("rows_standard", 0))
    award_rate = float(overview.get("award_rate", np.nan))
    award_rate_text = f"{award_rate:.1%}" if pd.notna(award_rate) else "unavailable"
    return {
        "subset": (
            f"All outcome charts below use the cleaned awarded/denied subset: {cleaned_rows:,} of {total_rows:,} records."
        ),
        "imbalance": (
            f"That subset is {award_rate_text} awarded, so every subgroup pattern sits on an award-heavy baseline."
        ),
        "rule": "Read these results as population patterns, not as decision drivers.",
    }


def build_school_chart_context(top_schools: pd.DataFrame, total_rows: int) -> str:
    """Add sample-size context for the school chart."""
    if top_schools.empty or total_rows <= 0:
        return "School context is unavailable."

    top_ranked = top_schools.reset_index()
    top_school_name = str(top_ranked.iloc[0, 0])
    top_school_share = float(top_ranked.iloc[0]["Count"]) / total_rows
    top_three_share = float(top_ranked["Count"].head(3).sum()) / total_rows
    return (
        f"{top_school_name} alone supplies {top_school_share:.1%} of the file, and the top three schools supply {top_three_share:.1%}, so school patterns are concentration-sensitive."
    )


def build_level_chart_context(
    level_award_rate: pd.DataFrame,
    overall_award_rate: float,
) -> str:
    """Add sample-size and imbalance context for the academic-level chart."""
    if level_award_rate.empty:
        return "Academic-level context is unavailable."

    ranked = level_award_rate.sort_values("applications", ascending=False, ignore_index=True)
    lead_group = ranked.iloc[0]
    lead_share = float(lead_group["applications"]) / max(int(level_award_rate["applications"].sum()), 1)
    baseline_text = f"{overall_award_rate:.1%}" if pd.notna(overall_award_rate) else "the overall level"
    return (
        f"{lead_group['level']} drives {lead_share:.1%} of the cleaned sample, so smaller levels can move faster than the {baseline_text} baseline."
    )


def build_numeric_comparison_guardrail(metric_label: str, coverage_share: float) -> str:
    """Explain the non-causal and missingness-sensitive reading of the boxplot."""
    if coverage_share < 0.70:
        return (
            f"{metric_label} covers only {coverage_share:.1%} of the cleaned subset, so any awarded-versus-denied gap may partly reflect missingness bias rather than a real population difference."
        )
    return (
        f"{metric_label} is descriptive here, not causal; the boxplot shows association inside the observed subset only."
    )


def build_population_summary(
    dataframe: pd.DataFrame,
    labeled_df: pd.DataFrame,
    overview: dict[str, object],
) -> dict[str, str]:
    """Summarize the main population-pattern message at the bottom of the tab."""
    award_rate = float(overview.get("award_rate", np.nan))
    award_rate_text = f"{award_rate:.1%}" if pd.notna(award_rate) else "unavailable"
    cleaned_rows = int(overview.get("rows_standard", 0))

    top_school_text = "School concentration cannot be estimated."
    if "school" in dataframe.columns:
        top_school_counts = (
            dataframe["school"].fillna("Missing").astype(str).value_counts()
        )
        if not top_school_counts.empty:
            top_school_name = str(top_school_counts.index[0])
            top_school_share = float(top_school_counts.iloc[0]) / max(len(dataframe), 1)
            top_school_text = f"{top_school_name} alone contributes {top_school_share:.1%} of the file."

    dominant_level_text = "Level mix is unavailable."
    if not labeled_df.empty and "level" in labeled_df.columns:
        level_counts = (
            labeled_df["level"].astype("string").fillna("Missing").str.strip().replace("", "Missing").value_counts()
        )
        if not level_counts.empty:
            top_level = str(level_counts.index[0])
            top_level_share = float(level_counts.iloc[0]) / max(len(labeled_df), 1)
            dominant_level_text = f"{top_level} drives {top_level_share:.1%} of the cleaned subset."

    return {
        "distribution": (
            f"Distribution patterns are readable in the cleaned subset of {cleaned_rows:,} records. {top_school_text} {dominant_level_text}"
        ),
        "imbalance": (
            f"The page sits on a {award_rate_text} award rate, so subgroup gaps are measured against an award-heavy baseline."
        ),
        "limitation": (
            "These charts support descriptive pattern reading only; small groups, school concentration, and missing financial data can all distort apparent differences."
        ),
    }


def build_killer_insight(key_variable_coverage: pd.DataFrame) -> str:
    """State the strongest cross-variable reliability insight on the page."""
    if key_variable_coverage.empty:
        return "Financial coverage is too weak to support a strong affordability reading."

    financial_labels = [
        "Total Parent Income",
        "Estimated Property Value",
        "Other Siblings Tuition",
        "Remaining Loan Balance",
    ]
    context_labels = [
        "Dependents Count",
        "Total Siblings",
        "Special Circumstances",
        "School",
    ]

    financial_rows = key_variable_coverage[
        key_variable_coverage["variable"].isin(financial_labels)
    ]
    context_rows = key_variable_coverage[
        key_variable_coverage["variable"].isin(context_labels)
    ]
    if financial_rows.empty or context_rows.empty:
        return "Financial coverage is materially weaker than context coverage, skewing the story away from direct affordability."

    financial_mean = float(financial_rows["coverage_pct"].mean())
    context_mean = float(context_rows["coverage_pct"].mean())
    return (
        f"Financial variables average {financial_mean:.1f}% coverage versus {context_mean:.1f}% for household-context variables, biasing the analysis toward context over true affordability."
    )


def build_business_implication() -> str:
    """State the main business takeaway from the evidence limits."""
    return (
        "Use this dataset to segment cases and flag review priorities, not to automate affordability judgments."
    )


def build_quality_coverage_gap_summary(
    key_variable_coverage: pd.DataFrame,
) -> dict[str, float | str]:
    """Summarize the gap between context coverage and financial coverage."""
    financial_labels = [
        "Total Parent Income",
        "Estimated Property Value",
        "Other Siblings Tuition",
        "Remaining Loan Balance",
    ]
    context_labels = [
        "Dependents Count",
        "Total Siblings",
        "Special Circumstances",
        "School",
    ]

    if key_variable_coverage.empty:
        return {
            "financial_mean": float("nan"),
            "context_mean": float("nan"),
            "parent_income": float("nan"),
            "loan_balance": float("nan"),
            "summary": "Comparable records are unavailable, so the coverage gap cannot be estimated.",
        }

    financial_rows = key_variable_coverage[
        key_variable_coverage["variable"].isin(financial_labels)
    ]
    context_rows = key_variable_coverage[
        key_variable_coverage["variable"].isin(context_labels)
    ]
    financial_mean = (
        float(financial_rows["coverage_pct"].mean()) if not financial_rows.empty else float("nan")
    )
    context_mean = (
        float(context_rows["coverage_pct"].mean()) if not context_rows.empty else float("nan")
    )
    parent_income = lookup_variable_coverage(key_variable_coverage, "Total Parent Income")
    loan_balance = lookup_variable_coverage(key_variable_coverage, "Remaining Loan Balance")
    return {
        "financial_mean": financial_mean,
        "context_mean": context_mean,
        "parent_income": parent_income,
        "loan_balance": loan_balance,
        "summary": (
            f"Context variables average {context_mean:.1f}% coverage, while key financial variables average only {financial_mean:.1f}%."
        ),
    }


def build_data_quality_verdict(
    column_catalog: pd.DataFrame,
    key_variable_coverage: pd.DataFrame,
) -> dict[str, str]:
    """Return a concise verdict for the Data Quality tab."""
    median_completeness = 100 - float(column_catalog["missing_pct"].median())
    severe_missing = int((column_catalog["missing_pct"] >= 80).sum())
    gap_summary = build_quality_coverage_gap_summary(key_variable_coverage)
    parent_income = gap_summary["parent_income"]
    loan_balance = gap_summary["loan_balance"]
    parent_income_text = (
        f"{float(parent_income):.1f}%"
        if pd.notna(parent_income)
        else "unavailable"
    )
    loan_balance_text = (
        f"{float(loan_balance):.1f}%"
        if pd.notna(loan_balance)
        else "unavailable"
    )

    return {
        "overall": (
            f"The file looks clean at {median_completeness:.1f}% median completeness, but that headline is biased."
        ),
        "risk": (
            f"Missingness is not neutral: parent income is {parent_income_text}, loan balance is {loan_balance_text}, and {severe_missing:,} fields are 80%+ missing."
        ),
        "impact": (
            "That bias pulls analysis and models toward context and away from true affordability."
        ),
        "business": (
            "Use the data to support review prioritization, not to make income-based aid judgments automatically."
        ),
    }


def build_modeling_impact_summary(key_variable_coverage: pd.DataFrame) -> str:
    """Explain how uneven missingness biases models."""
    gap_summary = build_quality_coverage_gap_summary(key_variable_coverage)
    if pd.isna(gap_summary["financial_mean"]) or pd.isna(gap_summary["context_mean"]):
        return "Modeling impact cannot be estimated until the comparable subset is available."

    return (
        f"Missingness biases models toward context: context variables average {float(gap_summary['context_mean']):.1f}% coverage, "
        f"while financial variables average only {float(gap_summary['financial_mean']):.1f}%. A naive model will overlearn context signals and underread affordability."
    )


def style_key_variable_coverage_table(coverage_display: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Make weak coverage rows easier to spot in the coverage table."""
    coverage_column = "Coverage in Comparable Subset"

    def highlight_row(row: pd.Series) -> list[str]:
        coverage_value = float(row[coverage_column])
        if coverage_value < 70:
            style = "background-color: #fef2f2;"
        elif coverage_value < 85:
            style = "background-color: #fffbeb;"
        else:
            style = ""
        return [style] * len(row)

    return (
        coverage_display.style
        .format({coverage_column: "{:.1f}%"})
        .apply(highlight_row, axis=1)
    )


def build_schema_summary(
    column_catalog: pd.DataFrame,
    key_variable_coverage: pd.DataFrame,
) -> dict[str, object]:
    """Summarize where the schema is strong versus weak."""
    median_completeness = 100 - float(column_catalog["missing_pct"].median())
    severe_missing = int((column_catalog["missing_pct"] >= 80).sum())
    coverage_gap = build_quality_coverage_gap_summary(key_variable_coverage)
    context_mean = coverage_gap["context_mean"]
    financial_mean = coverage_gap["financial_mean"]

    reliable_fields = (
        key_variable_coverage.loc[key_variable_coverage["coverage_pct"] >= 85, "variable"]
        .head(4)
        .tolist()
    )
    weak_fields = (
        key_variable_coverage.sort_values("coverage_pct", ascending=True, ignore_index=True)
        .loc[lambda frame: frame["coverage_pct"] < 70, "variable"]
        .head(4)
        .tolist()
    )

    if not reliable_fields:
        reliable_fields = (
            column_catalog.sort_values(["missing_pct", "column"], ascending=[True, True], ignore_index=True)
            .head(4)["column"]
            .tolist()
        )
    if not weak_fields:
        weak_fields = (
            column_catalog.sort_values(["missing_pct", "column"], ascending=[False, True], ignore_index=True)
            .head(4)["column"]
            .tolist()
        )

    if pd.notna(context_mean) and pd.notna(financial_mean):
        strong_line = (
            f"Strongest layer: administrative and household-context fields average {float(context_mean):.1f}% coverage."
        )
        weak_line = (
            f"Weakest layer: financial fields average only {float(financial_mean):.1f}% coverage, and {severe_missing:,} fields are 80%+ missing."
        )
    else:
        strong_line = (
            "Strongest layer: the schema is broad enough for field auditing and record-level QA."
        )
        weak_line = (
            f"Weakest layer: {severe_missing:,} fields are 80%+ missing, so raw schema breadth overstates measurement strength."
        )

    return {
        "overall": (
            f"Overall schema coverage looks broad at {median_completeness:.1f}% median completeness, but reliability is uneven."
        ),
        "strong": strong_line,
        "weak": weak_line,
        "reliable_fields": reliable_fields,
        "weak_fields": weak_fields,
    }


def build_schema_catalog_display(column_catalog: pd.DataFrame) -> pd.DataFrame:
    """Prepare a schema table with priority and completeness signals."""
    key_field_map = {column: label for column, label, _ in KEY_VARIABLE_SPECS}
    catalog_display = (
        column_catalog.copy()
        .sort_values(["missing_pct", "column"], ascending=[False, True], ignore_index=True)
    )
    catalog_display["analysis_priority"] = catalog_display["column"].map(
        lambda column: "Key field" if column in key_field_map else "Supporting field"
    )
    catalog_display["schema_status"] = catalog_display["missing_pct"].map(
        lambda value: "⚠️ High missing"
        if value >= 80
        else ("✅ High completeness" if value <= 5 else "Measured")
    )

    return catalog_display.rename(
        columns={
            "schema_status": "Schema Status",
            "analysis_priority": "Analysis Priority",
            "column": "Field",
            "family": "Feature Family",
            "dtype": "Pandas Type",
            "dtype_group": "Type Group",
            "non_null_count": "Non-Null Records",
            "non_null_pct": "Non-Null %",
            "missing_pct": "Missing %",
            "distinct_values": "Distinct Values",
        }
    )[
        [
            "Schema Status",
            "Analysis Priority",
            "Field",
            "Feature Family",
            "Pandas Type",
            "Type Group",
            "Non-Null Records",
            "Non-Null %",
            "Missing %",
            "Distinct Values",
        ]
    ]


def style_schema_catalog_table(catalog_display: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Highlight the strongest and weakest schema rows for fast scanning."""
    missing_column = "Missing %"
    status_column = "Schema Status"
    priority_column = "Analysis Priority"
    field_column = "Field"

    def highlight_row(row: pd.Series) -> list[str]:
        missing_value = float(row[missing_column])
        styles = [""] * len(row)
        if missing_value >= 80:
            styles = ["background-color: #fef2f2;"] * len(row)
        elif missing_value <= 5:
            styles = ["background-color: #ecfdf5;"] * len(row)

        if row[priority_column] == "Key field":
            for column_name in (priority_column, field_column):
                column_index = row.index.get_loc(column_name)
                styles[column_index] += " font-weight: 700;"

        status_index = row.index.get_loc(status_column)
        if missing_value >= 80:
            styles[status_index] += " color: #b91c1c; font-weight: 700;"
        elif missing_value <= 5:
            styles[status_index] += " color: #047857; font-weight: 700;"

        return styles

    return (
        catalog_display.style
        .format({"Non-Null %": "{:.1f}%", "Missing %": "{:.1f}%"})
        .apply(highlight_row, axis=1)
    )


def render_data_quality_verdict(verdict: dict[str, str]) -> None:
    """Render the Data Quality verdict with slightly stronger visual emphasis."""
    verdict_html = f"""
    <div style="
        border-left: 6px solid #dc2626;
        background: linear-gradient(180deg, #fff7ed 0%, #fef2f2 100%);
        border-radius: 14px;
        padding: 0.95rem 1.05rem;
        margin: 0.2rem 0 1rem 0;
        box-shadow: 0 8px 18px rgba(220, 38, 38, 0.06);
    ">
        <div style="
            color: #b91c1c;
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        ">Data Quality Verdict</div>
        <p style="margin: 0 0 0.38rem 0; color: #0f172a;"><strong>Overall:</strong> {html.escape(verdict["overall"])}</p>
        <p style="margin: 0 0 0.38rem 0; color: #0f172a;"><strong>Risk:</strong> {html.escape(verdict["risk"])}</p>
        <p style="margin: 0 0 0.38rem 0; color: #0f172a;"><strong>Impact:</strong> {html.escape(verdict["impact"])}</p>
        <p style="margin: 0; color: #0f172a;"><strong>Business implication:</strong> {html.escape(verdict["business"])}</p>
    </div>
    """
    st.markdown(verdict_html, unsafe_allow_html=True)


def render_final_analytical_verdict(
    summary: dict[str, str],
    killer_insight: str,
    business_implication: str,
) -> None:
    """Render the main conclusion as a visually dominant verdict block."""
    verdict_html = f"""
    <div style="
        border: 2px solid #0f766e;
        background: linear-gradient(180deg, #ecfeff 0%, #f8fafc 100%);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        margin: 0.3rem 0 1rem 0;
        box-shadow: 0 12px 28px rgba(15, 118, 110, 0.08);
    ">
        <div style="
            color: #0f766e;
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        ">Final Analytical Verdict</div>
        <div style="
            color: #0f172a;
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.85rem;
        ">What this dataset can credibly support</div>
        <p style="margin: 0 0 0.45rem 0; color: #0f172a;"><strong>Overall:</strong> {html.escape(summary["overall"])}</p>
        <p style="margin: 0 0 0.45rem 0; color: #0f172a;"><strong>Limitation:</strong> {html.escape(summary["limitation"])}</p>
        <p style="margin: 0 0 0.45rem 0; color: #0f172a;"><strong>Implication:</strong> {html.escape(summary["implication"])}</p>
        <p style="margin: 0 0 0.45rem 0; color: #0f172a;"><strong>Killer insight:</strong> {html.escape(killer_insight)}</p>
        <p style="margin: 0; color: #0f172a;"><strong>Business implication:</strong> {html.escape(business_implication)}</p>
    </div>
    """
    st.markdown(verdict_html, unsafe_allow_html=True)


def build_overview_answer_blocks(
    dataframe: pd.DataFrame,
    overview: dict[str, object],
    key_variable_coverage: pd.DataFrame,
    numeric_drivers: pd.DataFrame,
    submission_year_summary: pd.DataFrame,
) -> list[dict[str, str]]:
    """Summarize the top findings with an explicit interpretation layer."""
    answers: list[dict[str, str]] = []

    comparable_share = int(overview["rows_standard"]) / max(len(dataframe), 1)
    issue_count = int(overview["rows_missing"]) + int(overview["rows_nonstandard"])
    parent_income_coverage = lookup_variable_coverage(key_variable_coverage, "Total Parent Income")
    loan_balance_coverage = lookup_variable_coverage(
        key_variable_coverage,
        "Remaining Loan Balance",
    )
    parent_income_text = f"{parent_income_coverage:.1f}%" if pd.notna(parent_income_coverage) else "not available"
    loan_balance_text = f"{loan_balance_coverage:.1f}%" if pd.notna(loan_balance_coverage) else "not available"

    year_finding = "Submission-date coverage is too thin to support a stable timing read."
    year_why = "Any time trend would be too fragile to treat as a meaningful longitudinal pattern."
    major_years = submission_year_summary[submission_year_summary["applications"] >= 100]
    if len(major_years) >= 2:
        previous_year = major_years.iloc[-2]
        latest_year = major_years.iloc[-1]
        year_finding = (
            f"The usable time pattern is concentrated in {int(previous_year['submission_year'])} and "
            f"{int(latest_year['submission_year'])}, where comparable volume moves from "
            f"{int(previous_year['comparable_applications']):,} to {int(latest_year['comparable_applications']):,} records."
        )
        year_why = (
            "Trend interpretation is strongest in the high-volume years, not in the thin 2026 tail."
        )

    top_coverage = key_variable_coverage["variable"].head(3).tolist()
    weak_coverage = key_variable_coverage.sort_values("coverage_pct").head(2)["variable"].tolist()
    top_drivers = numeric_drivers["metric"].head(2).tolist()
    top_coverage_text = ", ".join(top_coverage) if top_coverage else "the strongest structured household fields"
    weak_coverage_text = ", ".join(weak_coverage) if weak_coverage else "the weaker financial fields"
    driver_text = ", ".join(top_drivers) if top_drivers else "the clearest numeric separators"

    answers.append(
        {
            "title": "Evidence Base",
            "finding": (
                f"{len(dataframe):,} applications across {len(dataframe.columns):,} fields give the study broad descriptive coverage of the aid process."
            ),
            "why": (
                "Strong volume helps segmentation, but it does not make every field trustworthy."
            ),
        }
    )
    answers.append(
        {
            "title": "Comparison Readiness",
            "finding": (
                f"{int(overview['rows_standard']):,} records ({comparable_share:.1%}) form a valid awarded-versus-denied sample, while {issue_count:,} records stay outside that comparison."
            ),
            "why": (
                "Direct rate claims are defensible only inside this cleaned subset."
            ),
        }
    )
    answers.append(
        {
            "title": "Measurement Strength",
            "finding": (
                f"{top_coverage_text} are the strongest-covered thesis variables, and {driver_text} stand out among the comparable numeric fields. "
                f"Coverage drops sharply for Total Parent Income ({parent_income_text}) and Remaining Loan Balance ({loan_balance_text})."
            ),
            "why": (
                f"Structure and context are safer to read than hard affordability claims, especially in {weak_coverage_text}."
            ),
        }
    )
    answers.append(
        {
            "title": "Time Pattern",
            "finding": year_finding,
            "why": year_why,
        }
    )

    return answers


def build_award_rate_context(award_rate: float) -> str:
    """Interpret the award rate as a class-balance signal, not just a percentage."""
    if pd.isna(award_rate):
        return "Unavailable until a standardized awarded/denied subset is prepared."

    imbalance = abs(award_rate - 0.5)
    if imbalance >= 0.10:
        majority_side = "award-heavy" if award_rate > 0.5 else "denial-heavy"
        return (
            f"The comparable sample is {majority_side}, which signals class imbalance "
            "and raises bias risk in pattern or model interpretation."
        )
    if imbalance >= 0.05:
        return "Outcomes lean to one side, so note mild imbalance when reading summary patterns."
    return "Outcomes are fairly balanced, so class-imbalance risk is lower in summary comparisons."


def build_reliability_badge(level: str) -> str:
    """Render a small visual badge for the reliability verdict."""
    palette = {
        "High": ("#ecfdf5", "#047857"),
        "Medium": ("#fffbeb", "#b45309"),
        "Low": ("#fef2f2", "#b91c1c"),
    }
    background, text_color = palette.get(level, ("#f8fafc", "#334155"))
    return (
        '<span style="'
        "display:inline-block;"
        "padding:0.24rem 0.65rem;"
        "border-radius:999px;"
        f"background:{background};"
        f"color:{text_color};"
        "font-size:0.82rem;"
        "font-weight:700;"
        '">'
        f"{level} reliability"
        "</span>"
    )


def build_key_risk_cards(
    overview: dict[str, object],
    key_variable_coverage: pd.DataFrame,
    column_catalog: pd.DataFrame,
) -> list[dict[str, str]]:
    """Highlight the most decision-relevant risks at the top of the page."""
    parent_income_coverage = lookup_variable_coverage(key_variable_coverage, "Total Parent Income")
    award_rate = float(overview.get("award_rate", np.nan))
    high_missing_fields = int((column_catalog["missing_pct"] >= 80).sum())
    median_completeness = 100 - float(column_catalog["missing_pct"].median())

    award_rate_text = f"{award_rate:.1%}" if pd.notna(award_rate) else "N/A"
    parent_income_text = f"{parent_income_coverage:.1f}%" if pd.notna(parent_income_coverage) else "N/A"

    return [
        {
            "title": "Income Coverage Risk",
            "headline": f"Total Parent Income is only {parent_income_text} complete.",
            "impact": "Need-based claims cannot rely on income alone.",
        },
        {
            "title": "Outcome Imbalance Risk",
            "headline": f"The valid comparison sample is {award_rate_text} awarded.",
            "impact": "Patterns and models will naturally lean toward awards.",
        },
        {
            "title": "Missing-Data Concentration",
            "headline": f"Median completeness is {median_completeness:.1f}%, but {high_missing_fields:,} fields are at least 80% missing.",
            "impact": "Overall completeness looks better than field-level reliability.",
        },
    ]


def build_study_readiness_summary(
    dataframe: pd.DataFrame,
    overview: dict[str, object],
    key_variable_coverage: pd.DataFrame,
    column_catalog: pd.DataFrame,
    submission_year_summary: pd.DataFrame,
    metadata_payload: dict[str, object] | None,
    *,
    metadata_error: str | None = None,
) -> dict[str, str]:
    """Tie the major strengths and limits into three short decision-ready lines."""
    comparable_share = int(overview.get("rows_standard", 0)) / max(len(dataframe), 1)
    award_rate = float(overview.get("award_rate", np.nan))
    median_completeness = 100 - float(column_catalog["missing_pct"].median())
    high_missing_fields = int((column_catalog["missing_pct"] >= 80).sum())
    parent_income_coverage = lookup_variable_coverage(key_variable_coverage, "Total Parent Income")
    parent_income_text = (
        f"{parent_income_coverage:.1f}%"
        if pd.notna(parent_income_coverage)
        else "not available"
    )
    award_rate_text = f"{award_rate:.1%}" if pd.notna(award_rate) else "unavailable"

    if submission_year_summary.empty:
        timing_note = "Timing stability is still hard to test."
    else:
        major_years = submission_year_summary[submission_year_summary["applications"] >= 100]
        if len(major_years) >= 2:
            first_year = int(major_years.iloc[0]["submission_year"])
            last_year = int(major_years.iloc[-1]["submission_year"])
            lowest_comparable_share = float(major_years["comparable_share_pct"].min()) / 100
            timing_note = (
                f"Timing is strongest in {first_year}-{last_year}, where comparability stays at or above {lowest_comparable_share:.1%}."
            )
        else:
            timing_note = "There are not enough high-volume years for a strong trend claim."

    if metadata_error:
        model_note = "Saved holdout evidence could not be read."
    elif not metadata_payload:
        model_note = "No saved holdout evidence yet."
    else:
        holdout_overview = build_holdout_overview_frame(metadata_payload)
        if holdout_overview.empty:
            model_note = "Saved metadata still lacks usable holdout validation."
        else:
            model_note = "Holdout evidence exists."

    return {
        "overall": (
            f"Ready for descriptive comparison: {comparable_share:.1%} of records are usable and median completeness is {median_completeness:.1f}%."
        ),
        "limitation": (
            f"Main limit: Parent income is only {parent_income_text} complete, outcomes are {award_rate_text} awarded, and {high_missing_fields:,} fields are 80%+ missing."
        ),
        "implication": (
            f"Use it for directional segment insight, not causal, fairness, or production claims. {timing_note} {model_note}"
        ),
    }


def build_data_reliability_verdicts(
    dataframe: pd.DataFrame,
    overview: dict[str, object],
    key_variable_coverage: pd.DataFrame,
    submission_year_summary: pd.DataFrame,
    metadata_payload: dict[str, object] | None,
    *,
    metadata_error: str | None = None,
) -> list[dict[str, str]]:
    """Summarize which evidence areas are ready for strong conclusions."""
    verdicts: list[dict[str, str]] = []

    total_rows = max(int(overview.get("rows_total", len(dataframe))), 1)
    comparable_share = int(overview.get("rows_standard", 0)) / total_rows
    issue_share = (
        int(overview.get("rows_missing", 0)) + int(overview.get("rows_nonstandard", 0))
    ) / total_rows
    if comparable_share >= 0.80:
        outcome_level = "High"
    elif comparable_share >= 0.55:
        outcome_level = "Medium"
    else:
        outcome_level = "Low"
    verdicts.append(
        {
            "area": "Outcome Labels",
            "level": outcome_level,
            "reason": (
                f"{comparable_share:.1%} of records have standardized awarded/denied outcomes; "
                f"{issue_share:.1%} remain missing or non-standard."
            ),
            "reliable_for": "Comparing awarded versus denied patterns inside the cleaned subset.",
            "limited_for": "Treating the full raw dataset as if every outcome label were directly comparable.",
        }
    )

    if key_variable_coverage.empty:
        variable_level = "Low"
        variable_reason = (
            "Key thesis variables cannot yet be profiled on the comparable subset, so measurement reliability is weak."
        )
    else:
        median_coverage = float(key_variable_coverage["coverage_pct"].median()) / 100
        low_coverage_count = int((key_variable_coverage["coverage_pct"] < 70).sum())
        weakest_variables = key_variable_coverage.sort_values("coverage_pct").head(2)["variable"].tolist()
        weakest_text = ", ".join(weakest_variables)
        if median_coverage >= 0.85 and low_coverage_count <= 1:
            variable_level = "High"
        elif median_coverage >= 0.65:
            variable_level = "Medium"
        else:
            variable_level = "Low"
        variable_reason = (
            f"Median key-variable coverage is {median_coverage:.1%}; weakest coverage is in {weakest_text}."
        )
    verdicts.append(
        {
            "area": "Key Variables",
            "level": variable_level,
            "reason": variable_reason,
            "reliable_for": "Profiling the best-covered financial and household drivers.",
            "limited_for": "Making strong claims from weak-covered fields without extra validation.",
        }
    )

    if submission_year_summary.empty:
        timing_level = "Low"
        timing_reason = (
            "Submission dates are too sparse or missing to test whether observed patterns stay stable across time."
        )
    else:
        total_years = int(len(submission_year_summary))
        major_years = submission_year_summary[submission_year_summary["applications"] >= 100]
        reference_years = major_years if not major_years.empty else submission_year_summary
        average_comparable_share = float(reference_years["comparable_share_pct"].mean()) / 100
        if len(major_years) >= 3 and average_comparable_share >= 0.70:
            timing_level = "High"
        elif total_years >= 2 and average_comparable_share >= 0.45:
            timing_level = "Medium"
        else:
            timing_level = "Low"
        timing_reason = (
            f"{total_years} submission years are visible; the main dated years average "
            f"{average_comparable_share:.1%} comparable outcomes."
        )
    verdicts.append(
        {
            "area": "Time Coverage",
            "level": timing_level,
            "reason": timing_reason,
            "reliable_for": "Reading broad timing patterns at a descriptive level.",
            "limited_for": "Claiming strong longitudinal stability when dated coverage is thin or uneven.",
        }
    )

    if metadata_error:
        model_level = "Low"
        model_reason = "Saved holdout metadata could not be read, so predictive reliability cannot be audited."
    elif not metadata_payload:
        model_level = "Low"
        model_reason = (
            "No saved validation or test summary is available, so only exploratory full-sample scoring can be shown."
        )
    else:
        holdout_overview = build_holdout_overview_frame(metadata_payload)
        if holdout_overview.empty:
            model_level = "Low"
            model_reason = "Saved metadata is present, but it does not include validation or test holdout metrics."
        else:
            test_rows = holdout_overview.loc[holdout_overview["dataset"] == "Test"]
            has_test_summary = not test_rows.empty
            selected_row = test_rows.iloc[0] if has_test_summary else holdout_overview.iloc[0]
            split_label = str(selected_row["dataset"])
            roc_auc = float(selected_row["roc_auc"]) if pd.notna(selected_row["roc_auc"]) else float("nan")
            f1 = float(selected_row["f1"]) if pd.notna(selected_row["f1"]) else float("nan")
            precision = (
                float(selected_row["precision"]) if pd.notna(selected_row["precision"]) else float("nan")
            )
            recall = float(selected_row["recall"]) if pd.notna(selected_row["recall"]) else float("nan")

            if (
                pd.notna(roc_auc)
                and pd.notna(f1)
                and pd.notna(precision)
                and pd.notna(recall)
                and roc_auc >= 0.80
                and f1 >= 0.70
                and precision >= 0.70
                and recall >= 0.60
            ):
                model_level = "High" if has_test_summary else "Medium"
            elif pd.notna(roc_auc) and pd.notna(f1) and roc_auc >= 0.70 and f1 >= 0.55:
                model_level = "Medium"
            else:
                model_level = "Low"

            if pd.notna(roc_auc) and pd.notna(f1) and pd.notna(precision) and pd.notna(recall):
                model_reason = (
                    f"{split_label} holdout metrics show ROC AUC {roc_auc:.3f}, F1 {f1:.3f}, "
                    f"precision {precision:.1%}, and recall {recall:.1%}."
                )
            else:
                model_reason = (
                    f"{split_label} holdout evidence is present, but the saved metrics are incomplete."
                )
            if not has_test_summary:
                model_reason += " Confidence stays capped because no saved test split is available."
    verdicts.append(
        {
            "area": "Model Evidence",
            "level": model_level,
            "reason": model_reason,
            "reliable_for": "Using saved holdout metrics as directional evidence of ranking quality.",
            "limited_for": "Operational automation or fairness claims without deeper subgroup validation.",
        }
    )

    return verdicts


def build_holdout_overview_frame(metadata_payload: dict[str, object]) -> pd.DataFrame:
    """Extract validation and test metrics from saved training metadata."""
    normal_workflow = _as_dict(metadata_payload.get("normal_workflow"))
    rows: list[dict[str, object]] = []

    for summary_key, display_label in (
        ("validation_summary", "Validation"),
        ("test_summary", "Test"),
    ):
        holdout_summary = _as_dict(normal_workflow.get(summary_key))
        if not holdout_summary:
            continue

        probability_metrics = _as_dict(holdout_summary.get("probability_metrics"))
        selected_metrics = _as_dict(holdout_summary.get("selected_policy_metrics"))

        rows.append(
            {
                "dataset": display_label,
                "policy": holdout_summary.get("selected_policy"),
                "threshold": holdout_summary.get("selected_threshold"),
                "roc_auc": probability_metrics.get("roc_auc"),
                "average_precision": probability_metrics.get("average_precision"),
                "brier_score": probability_metrics.get("brier_score"),
                "accuracy": selected_metrics.get("accuracy"),
                "precision": selected_metrics.get("precision"),
                "recall": selected_metrics.get("recall"),
                "f1": selected_metrics.get("f1"),
            }
        )

    return pd.DataFrame(rows)


def build_threshold_policy_frame(holdout_summary: dict[str, object]) -> pd.DataFrame:
    """Tabulate threshold-policy candidates from a saved holdout summary."""
    threshold_results = _as_dict(holdout_summary.get("threshold_results"))
    selected_policy = holdout_summary.get("selected_policy")

    rows: list[dict[str, object]] = []
    for policy_name, metrics_object in threshold_results.items():
        metrics = _as_dict(metrics_object)
        rows.append(
            {
                "policy": policy_name,
                "selected": "Yes" if policy_name == selected_policy else "",
                "threshold": metrics.get("threshold"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
            }
        )

    if not rows:
        return pd.DataFrame()

    policy_frame = pd.DataFrame(rows)
    policy_frame["selected_order"] = policy_frame["selected"].eq("Yes").map(
        {True: 0, False: 1}
    )
    return policy_frame.sort_values(
        by=["selected_order", "f1", "recall", "policy"],
        ascending=[True, False, False, True],
        ignore_index=True,
    ).drop(columns=["selected_order"])


def build_confusion_matrix_frame(confusion_matrix: object) -> pd.DataFrame:
    """Convert a 2x2 matrix into a chart-ready long format."""
    if not isinstance(confusion_matrix, list) or len(confusion_matrix) != 2:
        return pd.DataFrame()
    if any(not isinstance(row, list) or len(row) != 2 for row in confusion_matrix):
        return pd.DataFrame()

    true_negative, false_positive = confusion_matrix[0]
    false_negative, true_positive = confusion_matrix[1]
    return pd.DataFrame(
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


def build_model_comparison_verdict(
    metadata_payload: dict[str, object] | None,
    model_1_summary: dict[str, object],
) -> dict[str, str]:
    """Summarize the comparison between the current pipeline and Model 1."""
    if not metadata_payload or not model_1_summary:
        return {}

    holdout_overview = build_holdout_overview_frame(metadata_payload)
    if holdout_overview.empty:
        return {}

    test_row = holdout_overview.loc[holdout_overview["dataset"] == "Test"]
    selected_row = test_row.iloc[0] if not test_row.empty else holdout_overview.iloc[0]
    artifacts = _as_dict(metadata_payload.get("artifacts"))
    current_model_name = str(
        artifacts.get("selected_model_name")
        or _as_dict(metadata_payload.get("normal_workflow")).get("selected_candidate")
        or "Current model"
    )
    current_auc = float(selected_row["roc_auc"]) if pd.notna(selected_row["roc_auc"]) else float("nan")
    model_1_auc = float(model_1_summary.get("best_model_auc", np.nan))
    model_1_name = str(model_1_summary.get("best_model_name") or "Model 1 benchmark")

    if pd.notna(current_auc) and pd.notna(model_1_auc):
        auc_gap = current_auc - model_1_auc
        if auc_gap >= 0.015:
            overall = (
                f"{current_model_name} beats {model_1_name} on ROC AUC ({current_auc:.3f} test vs {model_1_auc:.3f} validation)."
            )
        elif auc_gap <= -0.015:
            overall = (
                f"{current_model_name} trails {model_1_name} on raw ROC AUC ({current_auc:.3f} test vs {model_1_auc:.3f} validation), but it is evaluated more rigorously."
            )
        else:
            overall = (
                f"{current_model_name} and {model_1_name} land in the same AUC range ({current_auc:.3f} vs {model_1_auc:.3f})."
            )
    else:
        overall = (
            f"{current_model_name} is the stronger evidence pipeline because it keeps separate validation and test stages, while {model_1_name} is a single validation benchmark."
        )

    return {
        "overall": overall,
        "guardrail": (
            "This is directional, not apples-to-apples: Model 1 reports a single validation-split ROC AUC, while the current pipeline uses cross-validation, model selection, threshold tuning, and a held-out test."
        ),
        "implication": (
            "Use the current pipeline as the thesis-grade evidence base and keep Model 1 as the earlier baseline benchmark."
        ),
    }


def build_model_comparison_frame(
    metadata_payload: dict[str, object] | None,
    model_1_summary: dict[str, object],
) -> pd.DataFrame:
    """Create a side-by-side benchmark table for the model evidence tab."""
    if not metadata_payload or not model_1_summary:
        return pd.DataFrame()

    holdout_overview = build_holdout_overview_frame(metadata_payload)
    if holdout_overview.empty:
        return pd.DataFrame()

    test_row = holdout_overview.loc[holdout_overview["dataset"] == "Test"]
    selected_row = test_row.iloc[0] if not test_row.empty else holdout_overview.iloc[0]

    artifacts = _as_dict(metadata_payload.get("artifacts"))
    normal_workflow = _as_dict(metadata_payload.get("normal_workflow"))
    feature_engineering = _as_dict(normal_workflow.get("feature_engineering"))
    feature_pruning = _as_dict(normal_workflow.get("feature_pruning"))

    current_model_name = str(
        artifacts.get("selected_model_name")
        or normal_workflow.get("selected_candidate")
        or "Current model"
    )
    current_threshold_policy = str(
        artifacts.get("selected_threshold_policy")
        or normal_workflow.get("selected_threshold_policy")
        or "Saved policy"
    )
    current_threshold = (
        f"{float(selected_row['threshold']):.2f}"
        if pd.notna(selected_row["threshold"])
        else "N/A"
    )
    current_auc = (
        f"{float(selected_row['roc_auc']):.3f} ({str(selected_row['dataset']).lower()})"
        if pd.notna(selected_row["roc_auc"])
        else "Not available"
    )
    current_ap = (
        f"{float(selected_row['average_precision']):.3f}"
        if pd.notna(selected_row["average_precision"])
        else "Not available"
    )
    current_f1 = (
        f"{float(selected_row['f1']):.3f}"
        if pd.notna(selected_row["f1"])
        else "Not available"
    )
    created_count = int(feature_engineering.get("created_column_count", 0))
    pruned_count = int(feature_pruning.get("columns_dropped_total", 0))

    model_1_best_name = str(model_1_summary.get("best_model_name") or "Model 1 benchmark")
    model_1_best_auc = model_1_summary.get("best_model_auc")
    model_1_auc_text = (
        f"{float(model_1_best_auc):.3f} (validation)"
        if pd.notna(model_1_best_auc)
        else "Not available"
    )
    train_share = model_1_summary.get("train_share")
    validation_share = model_1_summary.get("validation_share")
    split_text = "Single validation split"
    if pd.notna(train_share) and pd.notna(validation_share):
        split_text = f"Single {float(train_share):.0%}/{float(validation_share):.0%} train/validation split"

    numeric_feature_count = model_1_summary.get("numeric_feature_count")
    categorical_feature_count = model_1_summary.get("categorical_feature_count")
    feature_strategy_text = "Manual numeric features plus one-hot encoded categories"
    if numeric_feature_count is not None and categorical_feature_count is not None:
        feature_strategy_text = (
            f"{int(numeric_feature_count)} numeric fields + {int(categorical_feature_count)} categorical fields with one-hot encoding"
        )

    return pd.DataFrame(
        [
            {
                "Comparison Point": "Best saved model",
                "Current model": current_model_name,
                "Model 1": model_1_best_name,
            },
            {
                "Comparison Point": "Primary ROC AUC evidence",
                "Current model": current_auc,
                "Model 1": model_1_auc_text,
            },
            {
                "Comparison Point": "Average Precision",
                "Current model": current_ap,
                "Model 1": "Not reported in notebook",
            },
            {
                "Comparison Point": "F1 at chosen threshold",
                "Current model": current_f1,
                "Model 1": "Not reported in notebook",
            },
            {
                "Comparison Point": "Evaluation design",
                "Current model": "5-fold CV + validation + held-out test",
                "Model 1": split_text,
            },
            {
                "Comparison Point": "Threshold policy",
                "Current model": f"{current_threshold_policy} @ {current_threshold}",
                "Model 1": "AUC-only comparison in saved output",
            },
            {
                "Comparison Point": "Feature strategy",
                "Current model": f"{created_count} engineered features, {pruned_count} pruned columns",
                "Model 1": feature_strategy_text,
            },
            {
                "Comparison Point": "Leakage controls",
                "Current model": "Explicit post-decision field exclusions and pruning rules",
                "Model 1": "No explicit leakage audit shown in saved notebook output",
            },
        ]
    )


def render_model_comparison_verdict(verdict: dict[str, str]) -> None:
    """Render a concise comparison verdict box for the Model Evidence tab."""
    if not verdict:
        return

    verdict_html = f"""
    <div style="
        border-left: 6px solid #0f766e;
        background: linear-gradient(180deg, #f0fdfa 0%, #f8fafc 100%);
        border-radius: 14px;
        padding: 0.95rem 1.05rem;
        margin: 0.2rem 0 1rem 0;
        box-shadow: 0 8px 18px rgba(15, 118, 110, 0.06);
    ">
        <div style="
            color: #0f766e;
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        ">Current Model vs Model 1</div>
        <p style="margin: 0 0 0.38rem 0; color: #0f172a;"><strong>Overall:</strong> {html.escape(verdict["overall"])}</p>
        <p style="margin: 0 0 0.38rem 0; color: #0f172a;"><strong>Guardrail:</strong> {html.escape(verdict["guardrail"])}</p>
        <p style="margin: 0; color: #0f172a;"><strong>Implication:</strong> {html.escape(verdict["implication"])}</p>
    </div>
    """
    st.markdown(verdict_html, unsafe_allow_html=True)


def build_saved_feature_importance_frame(
    metadata_payload: dict[str, object] | None,
    *,
    top_n: int = 15,
) -> tuple[pd.DataFrame, dict[str, str]] | None:
    """Recover saved explainability output when the loaded model has no native importances."""
    if not metadata_payload:
        return None

    normal_workflow = _as_dict(metadata_payload.get("normal_workflow"))
    feature_insights = _as_dict(normal_workflow.get("feature_insights"))
    if not feature_insights:
        return None

    insight_type = str(feature_insights.get("type", "")).strip().lower()

    top_features = feature_insights.get("top_features")
    if not isinstance(top_features, list):
        top_features = feature_insights.get("top_permutation_features")

    if isinstance(top_features, list) and top_features:
        rows: list[dict[str, object]] = []
        for row in top_features[:top_n]:
            if not isinstance(row, dict):
                continue
            importance_value = row.get("importance")
            if importance_value is None:
                importance_value = row.get("importance_mean")
            if importance_value is None:
                continue
            numeric_value = float(importance_value)
            rows.append(
                {
                    "feature": str(row.get("feature", "Unknown feature")),
                    "importance": numeric_value,
                    "signed_value": numeric_value,
                    "direction": "Positive",
                }
            )

        if rows:
            caption = "Bars show saved feature-importance results from the training run."
            value_label = "Importance"
            if insight_type == "permutation_importance" or "importance_mean" in top_features[0]:
                caption = (
                    "Bars show saved permutation importance scores from the training run, which are more appropriate for models like HistGradientBoosting."
                )
                value_label = "Permutation Importance"

            return pd.DataFrame(rows), {
                "value_label": value_label,
                "caption": caption,
            }

    top_positive = feature_insights.get("top_positive")
    top_negative = feature_insights.get("top_negative")
    if isinstance(top_positive, list) or isinstance(top_negative, list):
        rows = []
        for direction, records in (("Positive", top_positive), ("Negative", top_negative)):
            if not isinstance(records, list):
                continue
            for row in records[:top_n]:
                if not isinstance(row, dict) or "coefficient" not in row:
                    continue
                coefficient = float(row["coefficient"])
                rows.append(
                    {
                        "feature": str(row.get("feature", "Unknown feature")),
                        "importance": abs(coefficient),
                        "signed_value": coefficient,
                        "direction": direction,
                    }
                )
        if rows:
            importance_df = pd.DataFrame(rows).sort_values(
                by="importance",
                ascending=False,
                ignore_index=True,
            ).head(top_n)
            return importance_df, {
                "value_label": "Absolute Coefficient",
                "caption": "Bars show saved coefficient magnitudes from the training run.",
            }

    return None


@st.cache_data(show_spinner=False)
def load_json(json_path: str | Path) -> dict[str, object]:
    """Load JSON metadata from disk and cache it between reruns."""
    path = Path(json_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    with path.open("r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(show_spinner=False)
def load_model_1_summary(
    summary_path: str | Path = DEFAULT_MODEL_1_SUMMARY_PATH,
    notebook_path: str | Path | None = LOCAL_MODEL_1_NOTEBOOK_PATH,
) -> dict[str, object]:
    """Load the packaged Model 1 summary, or parse the local notebook as a fallback."""
    summary_candidate = Path(summary_path)
    if not summary_candidate.is_absolute():
        summary_candidate = ROOT_DIR / summary_candidate
    if summary_candidate.exists():
        with summary_candidate.open("r", encoding="utf-8") as summary_file:
            summary_payload = json.load(summary_file)
        return summary_payload if isinstance(summary_payload, dict) else {}

    if not notebook_path:
        return {}

    path = Path(notebook_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as notebook_file:
        notebook_payload = json.load(notebook_file)

    train_share = float("nan")
    validation_share = float("nan")
    numeric_feature_count: int | None = None
    categorical_feature_count: int | None = None
    auc_by_model: dict[str, float] = {}
    top_features: list[dict[str, object]] = []

    for cell in notebook_payload.get("cells", []):
        source = "".join(cell.get("source", []))

        split_match = re.search(
            r"train_test_split\(X,\s*y,\s*test_size=([0-9.]+),\s*random_state=42\)",
            source,
        )
        if split_match and pd.isna(validation_share):
            validation_share = float(split_match.group(1))
            train_share = 1.0 - validation_share

        numeric_match = re.search(r"numerical_features\s*=\s*\[(.*?)\]", source, re.S)
        if numeric_match and numeric_feature_count is None:
            numeric_feature_count = len(re.findall(r"""['"][^'"]+['"]""", numeric_match.group(1)))

        categorical_match = re.search(r"categorical_features\s*=\s*\[(.*?)\]", source, re.S)
        if categorical_match and categorical_feature_count is None:
            categorical_feature_count = len(
                re.findall(r"""['"][^'"]+['"]""", categorical_match.group(1))
            )

        for output in cell.get("outputs", []):
            text_blocks: list[str] = []
            if "text" in output:
                text_blocks.append("".join(output["text"]))
            data = output.get("data", {})
            if isinstance(data, dict) and "text/plain" in data:
                text_blocks.append("".join(data["text/plain"]))

            for text_block in text_blocks:
                for model_name, auc_text in re.findall(
                    r"([A-Za-z][A-Za-z\- ]+) AUC: ([0-9.]+)",
                    text_block,
                ):
                    auc_by_model[model_name.strip()] = float(auc_text)

                if not top_features:
                    for line in text_block.splitlines():
                        feature_match = re.match(
                            r"\s*\d+\s+(.+?)\s+([0-9]*\.[0-9]+)\s*$",
                            line,
                        )
                        if feature_match:
                            top_features.append(
                                {
                                    "feature": feature_match.group(1).strip(),
                                    "importance": float(feature_match.group(2)),
                                }
                            )

    if not auc_by_model and not top_features and pd.isna(validation_share):
        return {}

    best_model_name = max(auc_by_model, key=auc_by_model.get) if auc_by_model else None
    best_model_auc = auc_by_model.get(best_model_name) if best_model_name else float("nan")

    return {
        "path": str(path),
        "train_share": train_share,
        "validation_share": validation_share,
        "auc_by_model": auc_by_model,
        "best_model_name": best_model_name,
        "best_model_auc": best_model_auc,
        "numeric_feature_count": numeric_feature_count,
        "categorical_feature_count": categorical_feature_count,
        "top_features": top_features[:5],
    }

apply_design_system()

if not DEFAULT_DATA_PATH.exists():
    st.error(f"CSV file not found: {DEFAULT_DATA_PATH}")
else:
    df = load_csv(DEFAULT_DATA_PATH)
    preview_df = df.copy()
    column_catalog = build_column_catalog(df)
    feature_family_summary = build_feature_family_summary(column_catalog)
    missingness_summary = build_missingness_summary(column_catalog)
    decision_audit = build_decision_audit_frame(df)

    try:
        labeled_df, overview = prepare_decision_analytics_frame(df)
    except Exception as exc:
        labeled_df = pd.DataFrame()
        overview = {
            "rows_total": int(len(df)),
            "rows_standard": 0,
            "rows_awarded": 0,
            "rows_denied": 0,
            "rows_missing": 0,
            "rows_nonstandard": 0,
            "award_rate": float("nan"),
        }
        st.warning(f"Comparable decision subset could not be prepared: {exc}")

    key_variable_coverage = build_key_variable_coverage(labeled_df)
    numeric_drivers = summarize_numeric_drivers(labeled_df) if not labeled_df.empty else pd.DataFrame()
    submission_year_summary = build_submission_year_summary(df)
    application_term_summary = build_application_term_summary(df)
    overview_answers = build_overview_answer_blocks(
        df,
        overview,
        key_variable_coverage,
        numeric_drivers,
        submission_year_summary,
    )
    numeric_columns = int((column_catalog["dtype_group"] == "Numeric").sum())
    text_columns = int((column_catalog["dtype_group"] == "Text / Categorical").sum())
    flag_columns = int((column_catalog["dtype_group"] == "Boolean / Flag").sum())
    comparable_share = overview["rows_standard"] / max(len(df), 1)
    issue_count = int(overview["rows_missing"]) + int(overview["rows_nonstandard"])
    has_model_metadata_file = DEFAULT_MODEL_METADATA_PATH.exists()
    metadata_payload: dict[str, object] | None = None
    metadata_error: str | None = None
    if has_model_metadata_file:
        try:
            metadata_payload = load_json(DEFAULT_MODEL_METADATA_PATH)
        except Exception as exc:
            metadata_error = f"Unable to read model metadata: {exc}"
    try:
        model_1_summary = load_model_1_summary()
    except Exception:
        model_1_summary = {}
    reliability_verdicts = build_data_reliability_verdicts(
        df,
        overview,
        key_variable_coverage,
        submission_year_summary,
        metadata_payload,
        metadata_error=metadata_error,
    )
    framing_principles = build_framing_principles()
    key_risk_cards = build_key_risk_cards(overview, key_variable_coverage, column_catalog)
    killer_insight = build_killer_insight(key_variable_coverage)
    business_implication = build_business_implication()
    study_readiness_summary = build_study_readiness_summary(
        df,
        overview,
        key_variable_coverage,
        column_catalog,
        submission_year_summary,
        metadata_payload,
        metadata_error=metadata_error,
    )
    data_quality_verdict = build_data_quality_verdict(column_catalog, key_variable_coverage)
    quality_coverage_gap = build_quality_coverage_gap_summary(key_variable_coverage)
    modeling_impact_summary = build_modeling_impact_summary(key_variable_coverage)
    population_guardrail = build_population_guardrail(overview, len(df))
    population_summary = build_population_summary(df, labeled_df, overview)
    schema_summary = build_schema_summary(column_catalog, key_variable_coverage)
    schema_catalog_display = build_schema_catalog_display(column_catalog)
    model_comparison_verdict = build_model_comparison_verdict(metadata_payload, model_1_summary)
    model_comparison_frame = build_model_comparison_frame(metadata_payload, model_1_summary)
    median_completeness = 100 - float(column_catalog["missing_pct"].median())
    high_missing_fields = int((column_catalog["missing_pct"] >= 80).sum())
    parent_income_coverage = lookup_variable_coverage(key_variable_coverage, "Total Parent Income")
    award_rate_value = float(overview["award_rate"]) if not labeled_df.empty else float("nan")

    render_page_header(
        title="Data Overview",
        description="Defines what conclusions in this project are reliable, limited, and safe to carry forward into decisions.",
        takeaway=study_readiness_summary["overall"],
        kicker="Evidence Readiness Workspace",
        pills=[
            (f"{int(overview['rows_standard']):,} valid comparisons", "primary"),
            ("Patterns, not causal rules", "warning"),
            ("Audit weak affordability fields", "danger"),
        ],
    )
    render_kpi_row(
        [
            {
                "label": "Valid Comparison Sample",
                "value": f"{int(overview['rows_standard']):,}",
                "note": f"{comparable_share:.1%} of the file supports direct awarded-versus-denied comparison.",
                "tone": "primary",
            },
            {
                "label": "Outcome Mix",
                "value": f"{award_rate_value:.1%}" if pd.notna(award_rate_value) else "N/A",
                "note": "Award-heavy baseline adds imbalance risk to every downstream pattern.",
                "tone": "warning",
            },
            {
                "label": "Typical Completeness",
                "value": f"{median_completeness:.1f}%",
                "note": "Headline completeness looks solid, but key financial fields are uneven.",
                "tone": "secondary",
            },
            {
                "label": "Fields at Severe Missingness",
                "value": f"{high_missing_fields:,}",
                "note": (
                    f"Parent income covers only {parent_income_coverage:.1f}% of comparable rows."
                    if pd.notna(parent_income_coverage)
                    else "Affordability fields remain materially weaker than context fields."
                ),
                "tone": "danger",
            },
        ]
    )
    render_insight_action_panel(
        insight=killer_insight,
        implication=business_implication,
    )

    overview_tab, quality_tab, population_tab, model_tab, preview_tab = st.tabs(
        [
            "Study Framing",
            "Data Quality",
            "Population & Outcomes",
            "Model Evidence",
            "Catalog & Preview",
        ]
    )

    with overview_tab:
        st.subheader("Framing Principles")
        principle_cols = st.columns(2)
        for index, principle in enumerate(framing_principles):
            with principle_cols[index % 2]:
                with st.container(border=True):
                    st.markdown(f"**Principle:** {principle['principle']}")
                    st.markdown(f"**Why:** {principle['why']}")

        st.subheader("Executive Summary")
        answer_cols = st.columns(2)
        for index, card in enumerate(overview_answers):
            with answer_cols[index % 2]:
                with st.container(border=True):
                    st.markdown(f"**{card['title']}**")
                    st.write(card["finding"])
                    st.markdown(f"**Why:** {card['why']}")

        metric_cols = st.columns(6)
        award_rate_value = float(overview["award_rate"]) if labeled_df.empty is False else float("nan")
        metric_cards = [
            (
                "Evidence Base Size",
                f"{len(df):,}",
                f"{len(df):,} rows -> enough for segmentation.",
            ),
            (
                "Observed Feature Breadth",
                f"{len(df.columns):,}",
                f"{len(df.columns):,} fields -> broad but uneven measurement.",
            ),
            (
                "Valid Comparison Sample",
                f"{int(overview['rows_standard']):,}",
                f"{int(overview['rows_standard']):,} clean rows -> direct comparison is credible.",
            ),
            (
                "Analysis Coverage",
                f"{comparable_share:.1%}",
                f"{comparable_share:.1%} retained -> low label loss.",
            ),
            (
                "Outcome Mix",
                f"{award_rate_value:.1%}" if pd.notna(award_rate_value) else "N/A",
                (
                    f"{award_rate_value:.1%} awarded -> imbalance risk."
                    if pd.notna(award_rate_value)
                    else "Outcome mix unavailable."
                ),
            ),
            (
                "Label Exclusions",
                f"{issue_count:,}",
                f"{issue_count:,} dropped -> exclude from rate claims.",
            ),
        ]
        for column, (label, value, caption) in zip(metric_cols, metric_cards):
            with column:
                with st.container(border=True):
                    st.metric(label, value)
                    st.caption(caption)

        st.subheader("Key Risks to Carry Forward")
        risk_cols = st.columns(3)
        for column, risk_card in zip(risk_cols, key_risk_cards):
            with column:
                with st.container(border=True):
                    st.caption("High-priority risk")
                    st.markdown(f"**{risk_card['title']}**")
                    st.write(risk_card["headline"])
                    st.markdown(f"**Why:** {risk_card['impact']}")

        st.subheader("Data Reliability Verdict")
        st.caption(
            "High / Medium / Low ratings summarize which evidence areas are ready for stronger conclusions and which still limit the project."
        )
        verdict_cols = st.columns(2)
        for index, verdict in enumerate(reliability_verdicts):
            with verdict_cols[index % 2]:
                with st.container(border=True):
                    st.markdown(f"**{verdict['area']}**")
                    st.markdown(build_reliability_badge(verdict["level"]), unsafe_allow_html=True)
                    st.write(verdict["reason"])
                    st.markdown(f"**Reliable for:** {verdict['reliable_for']}")
                    st.markdown(f"**Not reliable for:** {verdict['limited_for']}")

        render_final_analytical_verdict(
            study_readiness_summary,
            killer_insight,
            business_implication,
        )

        st.markdown("**Schema Snapshot**")
        schema_cols = st.columns(3)
        with schema_cols[0]:
            st.metric("Numeric Fields", f"{numeric_columns:,}")
        with schema_cols[1]:
            st.metric("Text / Categorical Fields", f"{text_columns:,}")
        with schema_cols[2]:
            st.metric("Boolean / Flag Fields", f"{flag_columns:,}")

        timing_cols = st.columns(2)
        with timing_cols[0]:
            st.markdown("**Application Volume by Submission Year**")
            if submission_year_summary.empty:
                st.info("No valid submission dates are available.")
            else:
                year_chart_df = add_top_n_flag(
                    submission_year_summary,
                    value_column="applications",
                    top_n=3,
                )
                year_chart = (
                    alt.Chart(year_chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("submission_year:O", title="Submission Year"),
                        y=alt.Y("applications:Q", title="Applications"),
                        color=build_color_condition(),
                        tooltip=[
                            alt.Tooltip("submission_year:O", title="Year"),
                            alt.Tooltip("applications:Q", title="Applications"),
                            alt.Tooltip(
                                "comparable_applications:Q",
                                title="Comparable Outcomes",
                            ),
                            alt.Tooltip(
                                "comparable_share_pct:Q",
                                title="Comparable Share (%)",
                                format=".1f",
                            ),
                        ],
                    )
                    .properties(height=280)
                )
                st.altair_chart(year_chart, width="stretch")
                st.caption(
                    "Higher bar = more applications in that submission year. Tooltip shows how much of that year is safe for awarded-versus-denied comparison."
                )
                st.caption(f"Takeaway: {build_submission_year_takeaway(submission_year_summary)}")

        with timing_cols[1]:
            st.markdown("**Application Term Mix**")
            if application_term_summary.empty:
                st.info("No application-term field is available.")
            else:
                term_chart_df = add_top_n_flag(
                    application_term_summary,
                    value_column="applications",
                    top_n=3,
                )
                term_chart = (
                    alt.Chart(term_chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("applications:Q", title="Applications"),
                        y=alt.Y(
                            "application_term:N",
                            title=None,
                            sort=alt.SortField(field="applications", order="descending"),
                        ),
                        color=build_color_condition(highlight_color=SECONDARY_COLOR),
                        tooltip=[
                            alt.Tooltip("application_term:N", title="Term"),
                            alt.Tooltip("applications:Q", title="Applications"),
                            alt.Tooltip("share_pct:Q", title="Share (%)", format=".1f"),
                        ],
                    )
                    .properties(height=280)
                )
                st.altair_chart(term_chart, width="stretch")
                st.caption(
                    "This shows where the raw application volume is concentrated across academic-term codes."
                )
                st.caption(f"Takeaway: {build_application_term_takeaway(application_term_summary)}")

        summary_chart_cols = st.columns(2)
        with summary_chart_cols[0]:
            st.markdown("**Outcome Readiness**")
            if decision_audit.empty:
                st.info("No decision field is available to audit.")
            else:
                status_summary = (
                    decision_audit.groupby("status", as_index=False)
                    .agg(count=("count", "sum"))
                    .sort_values(
                        by="status",
                        key=lambda series: series.map(
                            {
                                "Comparable outcome": 0,
                                "Missing outcome": 1,
                                "Non-standard outcome": 2,
                            }
                        ),
                    )
                )
                readiness_chart = (
                    alt.Chart(status_summary)
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", title="Records"),
                        y=alt.Y("status:N", title=None),
                        color=alt.Color(
                            "status:N",
                            title=None,
                            scale=alt.Scale(
                                domain=[
                                    "Comparable outcome",
                                    "Missing outcome",
                                    "Non-standard outcome",
                                ],
                                range=[PRIMARY_COLOR, ACCENT_COLOR, DANGER_COLOR],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("status:N", title="Status"),
                            alt.Tooltip("count:Q", title="Records"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(readiness_chart, width="stretch")
                decision_counts = (
                    decision_audit[["decision_label", "count"]]
                    .rename(columns={"decision_label": "Decision", "count": "Count"})
                    .set_index("Decision")
                )
                render_chart_insights(
                    decision_distribution_insights(decision_counts),
                    boundaries=DATA_BOUNDARIES,
                )
                st.caption(
                    "Comparable outcomes are the records that can safely enter awarded-versus-denied analysis."
                )

        with summary_chart_cols[1]:
            st.markdown("**Coverage of Key Analysis Variables**")
            if key_variable_coverage.empty:
                st.info("Comparable records are unavailable, so key-variable coverage cannot be shown.")
            else:
                coverage_chart_df = add_top_n_flag(
                    key_variable_coverage,
                    value_column="coverage_pct",
                    top_n=3,
                )
                coverage_chart = (
                    alt.Chart(coverage_chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "coverage_pct:Q",
                            title="Coverage in Comparable Subset (%)",
                            scale=alt.Scale(domain=[0, 100]),
                        ),
                        y=alt.Y(
                            "variable:N",
                            title=None,
                            sort=alt.SortField(field="coverage_pct", order="descending"),
                        ),
                        color=build_color_condition(highlight_color=SECONDARY_COLOR),
                        tooltip=[
                            alt.Tooltip("variable:N", title="Variable"),
                            alt.Tooltip("available_records:Q", title="Available Records"),
                            alt.Tooltip("coverage_pct:Q", title="Coverage (%)", format=".1f"),
                            alt.Tooltip("role:N", title="Why It Matters"),
                        ],
                    )
                    .properties(height=280)
                )
                st.altair_chart(coverage_chart, width="stretch")
                st.caption(
                    "Longer bar = stronger measurement coverage. This is one of the fastest ways to see which variables are safe to rely on."
                )
                st.caption(f"Takeaway: {build_key_variable_takeaway(key_variable_coverage)}")

        if not decision_audit.empty:
            decision_audit_display = decision_audit.copy()
            decision_audit_display["share_pct"] = decision_audit_display["share_pct"].map(
                lambda value: f"{value:.1f}%"
            )
            st.markdown("**Decision Label Audit**")
            st.dataframe(
                decision_audit_display.rename(
                    columns={
                        "decision_label": "Decision Label",
                        "status": "Audit Status",
                        "count": "Records",
                        "share_pct": "Share of Dataset",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        st.markdown("**Feature Family Composition**")
        if feature_family_summary.empty:
            st.info("No schema summary is available.")
        else:
            family_chart_df = add_top_n_flag(
                feature_family_summary,
                value_column="field_count",
                top_n=3,
            )
            family_chart = (
                alt.Chart(family_chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("field_count:Q", title="Field Count"),
                    y=alt.Y(
                        "family:N",
                        title=None,
                        sort=alt.SortField(field="field_count", order="descending"),
                    ),
                    color=build_color_condition(),
                    tooltip=[
                        alt.Tooltip("family:N", title="Feature Family"),
                        alt.Tooltip("field_count:Q", title="Fields"),
                    ],
                )
                .properties(height=240)
            )
            st.altair_chart(family_chart, width="stretch")
            st.caption(
                "This shows where the dataset is most detailed: application data, family economics, documentation, or administrative outcome fields."
            )
            st.caption(f"Takeaway: {build_feature_family_takeaway(feature_family_summary)}")

    with quality_tab:
        st.subheader("Data Quality and Measurement Readiness")
        st.write(
            "The goal here is to show whether the dataset is strong enough for defensible business analytics, "
            "and where incomplete measurement could weaken inference."
        )

        render_data_quality_verdict(data_quality_verdict)

        with st.container(border=True):
            st.markdown("**Impact on Modeling**")
            st.write(modeling_impact_summary)

        quality_metric_cols = st.columns(3)
        with quality_metric_cols[0]:
            with st.container(border=True):
                st.metric(
                    "Typical Field Readiness",
                    f"{(100 - float(column_catalog['missing_pct'].median())):.1f}%",
                )
                st.caption("Looks strong at file level, but hides weak affordability fields.")
        with quality_metric_cols[1]:
            with st.container(border=True):
                st.metric(
                    "Fields With Gaps",
                    f"{int((column_catalog['missing_pct'] > 0).sum()):,}",
                )
                st.caption("Missingness touches most of the schema, not just a few edge columns.")
        with quality_metric_cols[2]:
            with st.container(border=True):
                st.metric(
                    "Fields at Severe Missingness",
                    f"{int((column_catalog['missing_pct'] >= 80).sum()):,}",
                )
                st.caption("These fields are weak candidates for robust modeling or strong claims.")
        st.caption(
            "High overall completeness but uneven across key variables, especially income and liability measures."
        )

        quality_cols = st.columns(2)
        with quality_cols[0]:
            st.markdown("**Top Missing Fields**")
            if missingness_summary.empty:
                st.info("No missing values were detected.")
            else:
                missingness_chart_df = add_top_n_flag(
                    missingness_summary,
                    value_column="missing_pct",
                    top_n=3,
                )
                missingness_chart = (
                    alt.Chart(missingness_chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("missing_pct:Q", title="Missingness (%)", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y(
                            "column:N",
                            title=None,
                            sort=alt.SortField(field="missing_pct", order="descending"),
                        ),
                        color=build_color_condition(highlight_color=DANGER_COLOR),
                        tooltip=[
                            alt.Tooltip("column:N", title="Field"),
                            alt.Tooltip("missing_pct:Q", title="Missingness (%)", format=".1f"),
                            alt.Tooltip("non_null_count:Q", title="Non-Null Records"),
                        ],
                    )
                    .properties(height=340)
                )
                st.altair_chart(missingness_chart, width="stretch")
                st.caption(
                    "High-missingness fields may still be useful descriptively, but they are weaker candidates for robust comparative modeling."
                )
                st.caption(f"Takeaway: {build_missingness_takeaway(column_catalog)}")

        with quality_cols[1]:
            st.markdown("**Coverage of Key Analysis Variables**")
            if key_variable_coverage.empty:
                st.info("Comparable outcome records are unavailable, so key-variable coverage cannot be summarized.")
            else:
                st.write(
                    f"{quality_coverage_gap['summary']} Parent income falls to {float(quality_coverage_gap['parent_income']):.1f}% and loan balance to {float(quality_coverage_gap['loan_balance']):.1f}%, so context variables are far more reliable than financial ones."
                )
                coverage_display = key_variable_coverage.copy()
                coverage_display["coverage_status"] = coverage_display["coverage_pct"].map(
                    lambda value: "Weak" if value < 70 else ("Moderate" if value < 85 else "Strong")
                )
                coverage_display = coverage_display.rename(
                    columns={
                        "variable": "Variable",
                        "coverage_pct": "Coverage in Comparable Subset",
                        "available_records": "Available Records",
                        "role": "Why It Matters",
                        "coverage_status": "Coverage Status",
                    }
                )
                st.dataframe(
                    style_key_variable_coverage_table(coverage_display),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Coverage is measured on the awarded-versus-denied subset because that is the population used for comparative decision analysis."
                )

        st.markdown("**Column Catalog**")
        catalog_display = column_catalog.copy()
        catalog_display["non_null_pct"] = catalog_display["non_null_pct"].map(
            lambda value: f"{value:.1f}%"
        )
        catalog_display["missing_pct"] = catalog_display["missing_pct"].map(
            lambda value: f"{value:.1f}%"
        )
        st.dataframe(
            catalog_display.rename(
                columns={
                    "column": "Field",
                    "family": "Feature Family",
                    "dtype": "Pandas Type",
                    "dtype_group": "Type Group",
                    "non_null_count": "Non-Null Records",
                    "non_null_pct": "Non-Null %",
                    "missing_pct": "Missing %",
                    "distinct_values": "Distinct Values",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with population_tab:
        st.subheader("Population Mix and Outcome Structure")
        st.caption(
            "Outcome-conditioned comparisons below use only the standardized awarded-versus-denied subset."
        )
        with st.container(border=True):
            st.markdown("### Outcome Guardrail")
            st.markdown(f"**Subset:** {population_guardrail['subset']}")
            st.markdown(f"**Imbalance:** {population_guardrail['imbalance']}")
            st.markdown(f"**Rule:** {population_guardrail['rule']}")

        population_cols = st.columns(2)
        with population_cols[0]:
            st.markdown("**Top Applicant Schools**")
            if "school" not in df.columns:
                st.info("The dataset does not contain a school field.")
            else:
                top_schools = (
                    df["school"]
                    .fillna("Missing")
                    .astype(str)
                    .value_counts()
                    .head(10)
                    .rename_axis("School")
                    .to_frame("Count")
                )
                school_chart_data = top_schools.reset_index()
                school_chart_data["share_pct"] = school_chart_data["Count"] / len(df) * 100
                school_chart_data = add_top_n_flag(
                    school_chart_data,
                    value_column="Count",
                    top_n=3,
                )
                school_chart = (
                    alt.Chart(school_chart_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("Count:Q", title="Applications"),
                        y=alt.Y(
                            "School:N",
                            title=None,
                            sort=alt.SortField(field="Count", order="descending"),
                        ),
                        color=build_color_condition(),
                        tooltip=[
                            alt.Tooltip("School:N", title="School"),
                            alt.Tooltip("Count:Q", title="Applications"),
                            alt.Tooltip("share_pct:Q", title="Share (%)", format=".1f"),
                        ],
                    )
                    .properties(height=360)
                )
                st.altair_chart(school_chart, width="stretch")
                render_chart_insights(
                    school_concentration_insights(top_schools, len(df)),
                    boundaries=POPULATION_BOUNDARIES,
                )
                st.caption(build_school_chart_context(top_schools, len(df)))

        with population_cols[1]:
            st.markdown("**Award Rate by Academic Level**")
            if labeled_df.empty or "level" not in labeled_df.columns:
                st.info("Comparable records with academic level are unavailable.")
            else:
                level_df = labeled_df[["level", "award_flag"]].copy()
                level_df["level"] = level_df["level"].astype("string").fillna("Missing").str.strip()
                level_df["level"] = level_df["level"].replace("", "Missing")
                level_award_rate = (
                    level_df.groupby("level", as_index=False)["award_flag"]
                    .agg(applications="size", award_rate="mean")
                    .sort_values("award_rate", ascending=False, ignore_index=True)
                )
                level_award_rate["award_rate_pct"] = level_award_rate["award_rate"] * 100
                level_chart = (
                    alt.Chart(level_award_rate)
                    .mark_bar()
                    .encode(
                        x=alt.X("level:N", title="Level", sort=None),
                        y=alt.Y(
                            "award_rate_pct:Q",
                            title="Award Rate (%)",
                            scale=alt.Scale(domain=[0, 100]),
                        ),
                        color=alt.Color(
                            "award_rate_pct:Q",
                            title="Award Rate (%)",
                            scale=alt.Scale(scheme="teals"),
                        ),
                        tooltip=[
                            alt.Tooltip("level:N", title="Level"),
                            alt.Tooltip("applications:Q", title="Applications"),
                            alt.Tooltip("award_rate_pct:Q", title="Award Rate (%)", format=".1f"),
                        ],
                    )
                    .properties(height=360)
                )
                st.altair_chart(level_chart, width="stretch")
                render_chart_insights(
                    award_rate_group_insights(level_award_rate, group_column="level"),
                    boundaries=POPULATION_BOUNDARIES,
                )
                st.caption(
                    "Unlike the earlier dashboard version, this chart excludes missing and non-standard outcome labels from the denominator."
                )
                st.caption(
                    build_level_chart_context(
                        level_award_rate,
                        float(overview["award_rate"]) if pd.notna(overview["award_rate"]) else float("nan"),
                    )
                )

        st.markdown("**Outcome-Conditioned Numeric Comparison**")
        if labeled_df.empty:
            st.info("Comparable awarded-versus-denied records are unavailable for numeric comparisons.")
        else:
            selected_metric_label = st.selectbox(
                "Select a numeric feature to profile against the decision outcome",
                options=list(PROFILE_METRIC_OPTIONS.keys()),
            )
            metric_column, metric_axis_label, metric_note = PROFILE_METRIC_OPTIONS[
                selected_metric_label
            ]

            if metric_column not in labeled_df.columns:
                st.info(f"{selected_metric_label} is not available in the labeled subset.")
            else:
                comparison_df = labeled_df[["decision", metric_column]].copy()
                comparison_df[metric_column] = pd.to_numeric(
                    comparison_df[metric_column],
                    errors="coerce",
                )
                comparison_df = comparison_df.dropna(subset=[metric_column])

                if comparison_df.empty:
                    st.info("No valid records are available for the selected comparison.")
                else:
                    coverage_share = len(comparison_df) / max(len(labeled_df), 1)
                    awarded_values = comparison_df.loc[
                        comparison_df["decision"].astype("string").str.strip().str.lower()
                        == "awarded",
                        metric_column,
                    ]
                    denied_values = comparison_df.loc[
                        comparison_df["decision"].astype("string").str.strip().str.lower()
                        == "denied",
                        metric_column,
                    ]

                    comparison_metric_cols = st.columns(3)
                    with comparison_metric_cols[0]:
                        st.metric(
                            "Coverage in Comparable Subset",
                            f"{len(comparison_df) / max(len(labeled_df), 1):.1%}",
                        )
                    with comparison_metric_cols[1]:
                        st.metric(
                            "Awarded Median",
                            f"{float(awarded_values.median()):,.0f}"
                            if not awarded_values.empty
                            else "N/A",
                        )
                    with comparison_metric_cols[2]:
                        st.metric(
                            "Denied Median",
                            f"{float(denied_values.median()):,.0f}"
                            if not denied_values.empty
                            else "N/A",
                        )

                    comparison_df["decision_label"] = (
                        comparison_df["decision"].astype("string").str.strip().str.title()
                    )
                    numeric_chart = (
                        alt.Chart(comparison_df)
                        .mark_boxplot(size=45)
                        .encode(
                            x=alt.X("decision_label:N", title="Decision"),
                            y=alt.Y(f"{metric_column}:Q", title=metric_axis_label),
                            color=alt.Color(
                                "decision_label:N",
                                legend=None,
                                scale=alt.Scale(
                                    domain=["Awarded", "Denied"],
                                    range=[PRIMARY_COLOR, DANGER_COLOR],
                                ),
                            ),
                        )
                        .properties(height=360)
                    )
                    st.altair_chart(numeric_chart, width="stretch")
                    render_chart_insights(
                        numeric_decision_comparison_insights(
                            comparison_df,
                            metric_column,
                            metric_axis_label,
                        ),
                        boundaries=POPULATION_BOUNDARIES,
                    )
                    st.caption(build_numeric_comparison_guardrail(metric_axis_label, coverage_share))
                    st.caption(metric_note)

        st.markdown("**Correlation Structure Among Numeric Fields**")
        numeric_df = df.select_dtypes(include="number")
        numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]

        if numeric_df.shape[1] < 2:
            st.info("At least two non-constant numeric features are needed to compute correlations.")
        else:
            max_heatmap_features = 15
            feature_note = "The heatmap includes all non-constant numeric features."

            if numeric_df.shape[1] > max_heatmap_features:
                selected_features = (
                    numeric_df.var(numeric_only=True)
                    .sort_values(ascending=False)
                    .head(max_heatmap_features)
                    .index
                    .tolist()
                )
                numeric_df = numeric_df[selected_features]
                feature_note = (
                    "Showing the 15 highest-variance numeric fields to keep the correlation view readable."
                )

            correlation_df = numeric_df.corr(numeric_only=True)
            heatmap_df = (
                correlation_df.reset_index()
                .melt(id_vars="index", var_name="feature_y", value_name="correlation")
                .rename(columns={"index": "feature_x"})
            )

            heatmap = (
                alt.Chart(heatmap_df)
                .mark_rect()
                .encode(
                    x=alt.X(
                        "feature_x:N",
                        title=None,
                        sort=correlation_df.columns.tolist(),
                        axis=alt.Axis(labelAngle=-45),
                    ),
                    y=alt.Y(
                        "feature_y:N",
                        title=None,
                        sort=correlation_df.index.tolist(),
                    ),
                    color=alt.Color(
                        "correlation:Q",
                        title="Correlation",
                        scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
                    ),
                    tooltip=[
                        alt.Tooltip("feature_x:N", title="Feature 1"),
                        alt.Tooltip("feature_y:N", title="Feature 2"),
                        alt.Tooltip("correlation:Q", title="Correlation", format=".2f"),
                    ],
                )
                .properties(height=520)
            )
            st.altair_chart(heatmap, width="stretch")
            render_chart_insights(
                correlation_heatmap_insights(correlation_df),
                boundaries=POPULATION_BOUNDARIES,
            )
            st.caption(
                f"{feature_note} Constant numeric columns are excluded from the correlation matrix."
            )
            st.caption(
                "Correlations here can come from form structure, derived variables, or shared missingness, not just real-world relationships."
            )

        st.markdown("**Population Summary**")
        st.markdown(f"**Distribution:** {population_summary['distribution']}")
        st.markdown(f"**Imbalance:** {population_summary['imbalance']}")
        st.markdown(f"**Limitation:** {population_summary['limitation']}")

    with model_tab:
        st.subheader("Predictive Model Evidence")
        st.write(
            "This section separates holdout evaluation from exploratory full-sample scoring. "
            "That distinction is important for thesis credibility."
        )

        if metadata_error:
            st.error(metadata_error)
        elif not has_model_metadata_file:
            st.info(
                f"No saved holdout metadata was found at `{DEFAULT_MODEL_METADATA_PATH}`. "
                "The page will avoid reporting thesis-style performance metrics until that file exists."
            )

        if metadata_payload:
            holdout_overview = build_holdout_overview_frame(metadata_payload)
            normal_workflow = _as_dict(metadata_payload.get("normal_workflow"))
            test_summary = _as_dict(normal_workflow.get("test_summary"))

            if holdout_overview.empty:
                st.info("The metadata file is present, but it does not contain holdout summaries.")
            else:
                test_row = holdout_overview.loc[
                    holdout_overview["dataset"] == "Test"
                ]
                selected_row = (
                    test_row.iloc[0]
                    if not test_row.empty
                    else holdout_overview.iloc[0]
                )

                performance_cols = st.columns(4)
                with performance_cols[0]:
                    st.metric("Holdout ROC AUC", f"{float(selected_row['roc_auc']):.3f}")
                with performance_cols[1]:
                    st.metric(
                        "Average Precision",
                        f"{float(selected_row['average_precision']):.3f}",
                    )
                with performance_cols[2]:
                    st.metric("Selected F1", f"{float(selected_row['f1']):.3f}")
                with performance_cols[3]:
                    st.metric("Selected Threshold", f"{float(selected_row['threshold']):.2f}")

                policy_cols = st.columns(3)
                with policy_cols[0]:
                    st.metric("Accuracy", f"{float(selected_row['accuracy']):.1%}")
                with policy_cols[1]:
                    st.metric("Precision", f"{float(selected_row['precision']):.1%}")
                with policy_cols[2]:
                    st.metric("Recall", f"{float(selected_row['recall']):.1%}")

                holdout_display = holdout_overview.copy()
                for percentage_column in ("accuracy", "precision", "recall", "f1"):
                    holdout_display[percentage_column] = holdout_display[percentage_column].map(
                        lambda value: f"{float(value):.1%}" if pd.notna(value) else "N/A"
                    )
                for score_column in ("roc_auc", "average_precision", "brier_score", "threshold"):
                    holdout_display[score_column] = holdout_display[score_column].map(
                        lambda value: f"{float(value):.3f}" if pd.notna(value) else "N/A"
                    )

                st.markdown("**Validation vs Test Summary**")
                st.dataframe(
                    holdout_display.rename(
                        columns={
                            "dataset": "Holdout Split",
                            "policy": "Selected Policy",
                            "threshold": "Threshold",
                            "roc_auc": "ROC AUC",
                            "average_precision": "Average Precision",
                            "brier_score": "Brier Score",
                            "accuracy": "Accuracy",
                            "precision": "Precision",
                            "recall": "Recall",
                            "f1": "F1",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

                threshold_policy_frame = build_threshold_policy_frame(test_summary)
                if not threshold_policy_frame.empty:
                    threshold_display = threshold_policy_frame.copy()
                    for percentage_column in ("accuracy", "precision", "recall", "f1"):
                        threshold_display[percentage_column] = threshold_display[
                            percentage_column
                        ].map(
                            lambda value: f"{float(value):.1%}" if pd.notna(value) else "N/A"
                        )
                    threshold_display["threshold"] = threshold_display["threshold"].map(
                        lambda value: f"{float(value):.2f}" if pd.notna(value) else "N/A"
                    )
                    st.markdown("**Test-Set Threshold Policy Comparison**")
                    st.dataframe(
                        threshold_display.rename(
                            columns={
                                "policy": "Policy",
                                "selected": "Selected",
                                "threshold": "Threshold",
                                "accuracy": "Accuracy",
                                "precision": "Precision",
                                "recall": "Recall",
                                "f1": "F1",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                selected_policy_metrics = _as_dict(test_summary.get("selected_policy_metrics"))
                confusion_df = build_confusion_matrix_frame(
                    selected_policy_metrics.get("confusion_matrix")
                )
                if not confusion_df.empty:
                    st.markdown("**Holdout Confusion Matrix**")
                    confusion_heatmap = (
                        alt.Chart(confusion_df)
                        .mark_rect()
                        .encode(
                            x=alt.X(
                                "predicted_label:N",
                                title="Predicted Label",
                                sort=["Denied", "Awarded"],
                            ),
                            y=alt.Y(
                                "actual_label:N",
                                title="Actual Label",
                                sort=["Denied", "Awarded"],
                            ),
                            color=alt.Color(
                                "count:Q",
                                title="Count",
                                scale=alt.Scale(scheme="teals"),
                            ),
                            tooltip=[
                                alt.Tooltip("actual_label:N", title="Actual"),
                                alt.Tooltip("predicted_label:N", title="Predicted"),
                                alt.Tooltip("outcome:N", title="Outcome"),
                                alt.Tooltip("count:Q", title="Count"),
                            ],
                        )
                        .properties(height=320)
                    )
                    confusion_text = (
                        alt.Chart(confusion_df)
                        .mark_text(fontSize=18, fontWeight="bold")
                        .encode(
                            x=alt.X("predicted_label:N", sort=["Denied", "Awarded"]),
                            y=alt.Y("actual_label:N", sort=["Denied", "Awarded"]),
                            text=alt.Text("count:Q"),
                            color=alt.value("white"),
                        )
                    )
                    st.altair_chart(
                        alt.layer(confusion_heatmap, confusion_text),
                        width="stretch",
                    )
                    st.caption(
                        "This confusion matrix comes from the saved holdout summary rather than from rescoring the full dataset in the app."
                    )
                    st.caption(f"Takeaway: {build_confusion_matrix_takeaway(confusion_df)}")

        st.markdown("**Current Model vs Model 1**")
        if not model_1_summary:
            st.info(
                "The packaged Model 1 benchmark summary is unavailable, so the baseline comparison cannot be shown."
            )
        elif not metadata_payload:
            st.info(
                "Current-model metadata is still unavailable, so this section can only compare methodology once the training artifacts are generated."
            )
        else:
            render_model_comparison_verdict(model_comparison_verdict)

            comparison_cols = st.columns(4)
            holdout_overview = build_holdout_overview_frame(metadata_payload)
            comparison_row = holdout_overview.loc[holdout_overview["dataset"] == "Test"]
            comparison_row = comparison_row.iloc[0] if not comparison_row.empty else holdout_overview.iloc[0]
            current_auc = float(comparison_row["roc_auc"]) if pd.notna(comparison_row["roc_auc"]) else float("nan")
            model_1_auc = float(model_1_summary.get("best_model_auc", np.nan))

            with comparison_cols[0]:
                st.metric(
                    "Current Test ROC AUC",
                    f"{current_auc:.3f}" if pd.notna(current_auc) else "N/A",
                )
            with comparison_cols[1]:
                st.metric(
                    "Model 1 Best Validation AUC",
                    f"{model_1_auc:.3f}" if pd.notna(model_1_auc) else "N/A",
                )
            with comparison_cols[2]:
                st.metric(
                    "AUC Gap",
                    f"{(current_auc - model_1_auc):+.3f}"
                    if pd.notna(current_auc) and pd.notna(model_1_auc)
                    else "N/A",
                )
            with comparison_cols[3]:
                st.metric(
                    "Selected Current Model",
                    str(
                        _as_dict(metadata_payload.get("artifacts")).get("selected_model_name")
                        or _as_dict(metadata_payload.get("normal_workflow")).get("selected_candidate")
                        or "N/A"
                    ),
                )

            st.dataframe(model_comparison_frame, width="stretch", hide_index=True)
            st.caption(
                "Read this as a benchmark comparison, not a controlled bake-off: Model 1 reports single-split validation AUC, while the current model reports a stricter holdout result."
            )

        st.markdown("**Exploratory Full-Sample Scoring**")
        if DEFAULT_MODEL_PATH is None or load_model is None:
            st.info(
                "Model-loading dependencies are unavailable, so exploratory scoring cannot be shown yet."
            )
        elif not Path(DEFAULT_MODEL_PATH).exists():
            st.info(
                f"Add a trained model at `{DEFAULT_MODEL_PATH}` to score the current dataset."
            )
        else:
            try:
                model = load_model(DEFAULT_MODEL_PATH)
                preview_df = add_prediction_probabilities(df, model)
            except Exception as exc:
                st.error(f"Unable to compute prediction probabilities: {exc}")
            else:
                probability_series = pd.to_numeric(
                    preview_df["predicted_award_probability"],
                    errors="coerce",
                ).dropna()
                high_confidence_share = float(
                    ((probability_series >= 0.80) | (probability_series <= 0.20)).mean()
                )
                mid_band_share = float(
                    ((probability_series >= 0.40) & (probability_series <= 0.60)).mean()
                )

                scoring_cols = st.columns(3)
                with scoring_cols[0]:
                    st.metric(
                        "Average Predicted Award Probability",
                        f"{probability_series.mean():.1%}",
                    )
                with scoring_cols[1]:
                    st.metric("High-Confidence Tail Share", f"{high_confidence_share:.1%}")
                with scoring_cols[2]:
                    st.metric("Mid-Band Share (0.40-0.60)", f"{mid_band_share:.1%}")

                probability_chart = (
                    alt.Chart(preview_df)
                    .mark_bar(color=PRIMARY_COLOR)
                    .encode(
                        x=alt.X(
                            "predicted_award_probability:Q",
                            title="Predicted Award Probability",
                            bin=alt.Bin(maxbins=20),
                        ),
                        y=alt.Y("count():Q", title="Applications"),
                        tooltip=[alt.Tooltip("count():Q", title="Applications")],
                    )
                    .properties(height=320)
                )
                st.altair_chart(probability_chart, width="stretch")
                render_chart_insights(
                    [
                        f"{high_confidence_share:.1%} of scored applications fall into the high-confidence tails, which is useful for prioritization and queue design.",
                        "These probabilities are generated on the current dataset inside the app, so they should be interpreted as exploratory scores rather than as held-out performance evidence.",
                    ],
                    title="How to read this chart",
                    boundaries=MODEL_BOUNDARIES,
                )

                feature_importance_df: pd.DataFrame | None = None
                importance_metadata: dict[str, str] | None = None
                importance_source = "live_model"
                try:
                    feature_importance_df, importance_metadata = extract_top_feature_importances(
                        model,
                        top_n=15,
                    )
                except Exception as exc:
                    saved_importance_bundle = build_saved_feature_importance_frame(
                        metadata_payload,
                        top_n=15,
                    )
                    if saved_importance_bundle is None:
                        st.info(f"Feature importances are unavailable: {exc}")
                    else:
                        feature_importance_df, importance_metadata = saved_importance_bundle
                        importance_source = "saved_metadata"

                if feature_importance_df is not None and importance_metadata is not None:
                    st.markdown("**Top Feature Importances**")
                    importance_chart = (
                        alt.Chart(feature_importance_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "importance:Q",
                                title=importance_metadata["value_label"],
                            ),
                            y=alt.Y(
                                "feature:N",
                                title=None,
                                sort=alt.SortField(field="importance", order="descending"),
                            ),
                            color=alt.Color(
                                "direction:N",
                                title="Direction",
                                scale=alt.Scale(
                                    domain=["Positive", "Negative"],
                                    range=[PRIMARY_COLOR, DANGER_COLOR],
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip("feature:N", title="Feature"),
                                alt.Tooltip(
                                    "importance:Q",
                                    title=importance_metadata["value_label"],
                                    format=".4f",
                                ),
                                alt.Tooltip("signed_value:Q", title="Signed Value", format=".4f"),
                            ],
                        )
                        .properties(height=420)
                    )
                    st.altair_chart(importance_chart, width="stretch")
                    render_chart_insights(
                        feature_importance_insights(feature_importance_df),
                        boundaries=MODEL_BOUNDARIES,
                    )
                    st.caption(importance_metadata["caption"])
                    if importance_source == "saved_metadata":
                        st.caption(
                            "Native importances are not exposed by the loaded model, so this chart falls back to the saved training-run explainability output."
                        )

                try:
                    positive_drivers_df, negative_drivers_df, driver_metadata = (
                        extract_top_feature_drivers(model, top_n=10)
                    )
                except Exception as exc:
                    if isinstance(exc, AttributeError) and "signed coefficients" in str(exc).lower():
                        st.caption(
                            "Directional coefficient drivers are only available for linear models, so they are not shown for the selected tree-based model."
                        )
                    else:
                        st.info(f"Top positive and negative coefficient drivers are unavailable: {exc}")
                else:
                    st.markdown("**Directional Coefficient Drivers**")
                    driver_cols = st.columns(2)

                    with driver_cols[0]:
                        st.markdown("**Top Positive Drivers**")
                        if positive_drivers_df.empty:
                            st.info("No positive coefficient drivers were found.")
                        else:
                            positive_chart = (
                                alt.Chart(positive_drivers_df)
                                .mark_bar(color=PRIMARY_COLOR)
                                .encode(
                                    x=alt.X(
                                        "coefficient:Q",
                                        title=driver_metadata["value_label"],
                                    ),
                                    y=alt.Y(
                                        "feature:N",
                                        title=None,
                                        sort=alt.SortField(
                                            field="coefficient",
                                            order="descending",
                                        ),
                                    ),
                                    tooltip=[
                                        alt.Tooltip("feature:N", title="Feature"),
                                        alt.Tooltip(
                                            "coefficient:Q",
                                            title=driver_metadata["value_label"],
                                            format=".4f",
                                        ),
                                    ],
                                )
                                .properties(height=320)
                            )
                            st.altair_chart(positive_chart, width="stretch")
                            render_chart_insights(
                                driver_side_insights(positive_drivers_df, side="positive"),
                                boundaries=MODEL_BOUNDARIES,
                            )

                    with driver_cols[1]:
                        st.markdown("**Top Negative Drivers**")
                        if negative_drivers_df.empty:
                            st.info("No negative coefficient drivers were found.")
                        else:
                            negative_chart = (
                                alt.Chart(negative_drivers_df)
                                .mark_bar(color=DANGER_COLOR)
                                .encode(
                                    x=alt.X(
                                        "coefficient:Q",
                                        title=driver_metadata["value_label"],
                                    ),
                                    y=alt.Y(
                                        "feature:N",
                                        title=None,
                                        sort=alt.SortField(
                                            field="coefficient",
                                            order="ascending",
                                        ),
                                    ),
                                    tooltip=[
                                        alt.Tooltip("feature:N", title="Feature"),
                                        alt.Tooltip(
                                            "coefficient:Q",
                                            title=driver_metadata["value_label"],
                                            format=".4f",
                                        ),
                                    ],
                                )
                                .properties(height=320)
                            )
                            st.altair_chart(negative_chart, width="stretch")
                            render_chart_insights(
                                driver_side_insights(negative_drivers_df, side="negative"),
                                boundaries=MODEL_BOUNDARIES,
                            )

                    st.caption(driver_metadata["caption"])

    with preview_tab:
        st.subheader("Field Catalog and Record Preview")
        st.write(
            "Use this section when writing the methodology or appendix: it provides both the schema inventory and a direct preview of the underlying records."
        )

        with st.container(border=True):
            st.markdown("### Schema Summary")
            st.markdown(f"**Overall:** {schema_summary['overall']}")
            st.markdown(f"**Strong:** {schema_summary['strong']}")
            st.markdown(f"**Weak:** {schema_summary['weak']}")

        render_chart_insights(
            [
                "This schema is strong for structure and household context, but materially weaker for direct affordability measurement.",
                "It can support schema auditing, field-level QA, and measurement-readiness checks before modeling or thesis write-up.",
                "The preview can also show the exploratory ranking score when the saved model is available.",
            ],
            title="How to read this tab",
            boundaries=CATALOG_BOUNDARIES,
        )

        schema_cols = st.columns(2)
        with schema_cols[0]:
            with st.container(border=True):
                st.markdown("**Reliable Fields**")
                for field in schema_summary["reliable_fields"]:
                    st.write(f"- {field}")
        with schema_cols[1]:
            with st.container(border=True):
                st.markdown("**High-Missingness Fields**")
                for field in schema_summary["weak_fields"]:
                    st.write(f"- {field}")

        st.markdown("**Field Catalog**")
        st.dataframe(
            style_schema_catalog_table(schema_catalog_display),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "The catalog is sorted by missingness so the weakest fields surface first, while `✅` and `⚠️` flags mark the strongest and weakest completeness bands."
        )

        st.markdown("**Record Preview**")
        preview_row_count = st.slider(
            "Rows to display",
            min_value=5,
            max_value=40,
            value=15,
            step=5,
        )
        preview_display = preview_df.copy()
        if "predicted_award_probability" in preview_display.columns:
            ordered_columns = ["predicted_award_probability"] + [
                column
                for column in preview_display.columns
                if column != "predicted_award_probability"
            ]
            preview_display = preview_display[ordered_columns]
        st.dataframe(preview_display.head(preview_row_count), width="stretch")

        if "predicted_award_probability" in preview_display.columns:
            render_chart_insights(
                [
                    "predicted_award_probability is a ranking score: higher values mean a record looks more award-like to the model than lower-scored records in this dataset.",
                    "It can support sorting and review prioritization when paired with human review.",
                ],
                title="How to read the preview",
                boundaries=CATALOG_BOUNDARIES,
            )
            st.caption(
                "Rows stay in dataset order for inspection; the score column is shown first for convenience, not because the table is sorted by it."
            )
        else:
            render_chart_insights(
                [
                    "This preview is a schema check on raw records, not a modeled ranking view.",
                    "It can support appendix examples and field-level QA of how values are stored.",
                ],
                title="How to read the preview",
                boundaries=CATALOG_BOUNDARIES,
            )
