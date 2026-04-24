from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


STATUS_RELIABLE = "Reliable"
STATUS_CAUTION = "Caution"
STATUS_AUDIT = "Audit"
STATUS_HIGH_RISK = "High Risk"
STATUS_NOT_PRODUCTION = "Not Production Ready"
STATUS_MONITOR_ONLY = "Monitor Only"

STATUS_TONE = {
    STATUS_RELIABLE: "primary",
    STATUS_CAUTION: "warning",
    STATUS_AUDIT: "audit",
    STATUS_HIGH_RISK: "danger",
    STATUS_NOT_PRODUCTION: "secondary",
    STATUS_MONITOR_ONLY: "secondary",
    "Audit / Risk": "audit",
    "Not production-ready": "secondary",
}

SEVERITY_RANK = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


@dataclass(frozen=True)
class FeatureSpec:
    column: str
    label: str
    why_it_matters: str


KEY_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("decision", "Decision Label", "Defines the comparable awarded/denied evidence base."),
    FeatureSpec("qa_quality_score", "QA Quality Score", "Controls whether records can support confident routing."),
    FeatureSpec("school", "School", "Supports representation, concentration, and missing-school review."),
    FeatureSpec("total_parent_income", "Parent Income", "Useful affordability signal, but incomplete and currency-sensitive."),
    FeatureSpec("properties_total_estimated_value", "Property Value", "Structured asset signal with stronger coverage than income."),
    FeatureSpec("cars_count", "Cars / Assets", "Adds asset-side context without relying only on income."),
    FeatureSpec("loans_total_remaining_balance", "Remaining Loan Balance", "Represents liabilities when reported."),
    FeatureSpec("financial_assistants_total_est_annual_amount", "External Assistance", "Captures outside support when reported."),
    FeatureSpec("dependents_count", "Dependents", "Shows household burden and support obligations."),
    FeatureSpec("total_siblings", "Siblings", "Adds education-burden context."),
)


def status_tone(status: str) -> str:
    return STATUS_TONE.get(str(status), STATUS_TONE.get(str(status).strip().title(), "secondary"))


def format_pct(value: float | int | None, *, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def _string_not_blank(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().ne("")


def _numeric_series(dataframe: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(default, index=dataframe.index, dtype=float)
    return pd.to_numeric(dataframe[column], errors="coerce").fillna(default)


def _severity_series(dataframe: pd.DataFrame) -> pd.Series:
    if "qa_max_severity" not in dataframe.columns:
        return pd.Series("none", index=dataframe.index, dtype="string")
    return (
        dataframe["qa_max_severity"]
        .astype("string")
        .fillna("none")
        .str.strip()
        .str.lower()
        .replace("", "none")
    )


def get_reliability_label(
    coverage: float,
    issue_rate: float,
    sample_size: int,
) -> str:
    """Classify evidence reliability from coverage, issue concentration, and sample size."""
    if sample_size < 30:
        return STATUS_NOT_PRODUCTION
    if coverage < 0.50 or issue_rate >= 0.25:
        return STATUS_AUDIT
    if coverage < 0.80 or issue_rate >= 0.10 or sample_size < 100:
        return STATUS_CAUTION
    return STATUS_RELIABLE


def quality_adjusted_reliability_label(row: pd.Series) -> str:
    severity = str(row.get("qa_max_severity", "none")).strip().lower() or "none"
    issue_count = float(pd.to_numeric(row.get("qa_issue_count", 0), errors="coerce") or 0)
    quality_score = float(pd.to_numeric(row.get("qa_quality_score", 100), errors="coerce") or 100)
    requires_review = float(pd.to_numeric(row.get("qa_requires_review", 0), errors="coerce") or 0) > 0
    school_missing = float(pd.to_numeric(row.get("qa_school_was_missing", 0), errors="coerce") or 0) > 0

    if severity == "high" or quality_score < 70 or issue_count >= 4:
        return STATUS_HIGH_RISK
    if severity == "medium" or requires_review or school_missing or quality_score < 85 or issue_count >= 2:
        return STATUS_CAUTION
    return STATUS_RELIABLE


def high_confidence_mask(dataframe: pd.DataFrame) -> pd.Series:
    severity = _severity_series(dataframe)
    quality_score = _numeric_series(dataframe, "qa_quality_score", default=100)
    issue_count = _numeric_series(dataframe, "qa_issue_count", default=0)
    requires_review = _numeric_series(dataframe, "qa_requires_review", default=0)
    school_missing = _numeric_series(dataframe, "qa_school_was_missing", default=0)
    return (
        severity.isin(["none", "low"])
        & quality_score.ge(85)
        & issue_count.le(1)
        & requires_review.eq(0)
        & school_missing.eq(0)
    )


def apply_high_confidence_mode(
    dataframe: pd.DataFrame,
    *,
    high_confidence_only: bool,
) -> pd.DataFrame:
    if dataframe.empty or not high_confidence_only:
        return dataframe.copy()
    return dataframe.loc[high_confidence_mask(dataframe)].copy()


def build_qa_summary(
    dataframe: pd.DataFrame,
    review_dataframe: pd.DataFrame,
    issues_dataframe: pd.DataFrame,
) -> dict[str, float | int | str]:
    total_records = int(len(dataframe))
    severity = _severity_series(dataframe)
    review_required = _numeric_series(dataframe, "qa_requires_review", default=0).gt(0)
    high_severity = severity.eq("high")
    school_missing = _numeric_series(dataframe, "qa_school_was_missing", default=0).gt(0)
    issue_count = int(len(issues_dataframe)) if issues_dataframe is not None else int(_numeric_series(dataframe, "qa_issue_count").sum())
    review_cases = int(review_required.sum()) if "qa_requires_review" in dataframe.columns else int(len(review_dataframe))
    return {
        "total_records": total_records,
        "review_required_cases": review_cases,
        "review_required_rate": review_cases / max(total_records, 1),
        "high_severity_cases": int(high_severity.sum()),
        "high_severity_rate": float(high_severity.mean()) if total_records else 0.0,
        "missing_school_cases": int(school_missing.sum()),
        "missing_school_rate": float(school_missing.mean()) if total_records else 0.0,
        "issue_rows": issue_count,
        "avg_quality_score": float(_numeric_series(dataframe, "qa_quality_score", default=np.nan).mean()),
        "high_confidence_cases": int(high_confidence_mask(dataframe).sum()),
    }


def build_feature_reliability_frame(
    dataframe: pd.DataFrame,
    issues_dataframe: pd.DataFrame | None = None,
    *,
    feature_specs: Iterable[FeatureSpec] = KEY_FEATURES,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    issue_field_counts: dict[str, int] = {}
    if issues_dataframe is not None and not issues_dataframe.empty and "field_name" in issues_dataframe.columns:
        issue_field_counts = (
            issues_dataframe["field_name"].astype("string").fillna("").str.strip().value_counts().to_dict()
        )

    for spec in feature_specs:
        if spec.column in dataframe.columns:
            if dataframe[spec.column].dtype == "O" or str(dataframe[spec.column].dtype).startswith("string"):
                available = _string_not_blank(dataframe[spec.column])
            else:
                available = dataframe[spec.column].notna()
        else:
            available = pd.Series(False, index=dataframe.index)
        sample_size = int(available.sum())
        coverage = sample_size / max(len(dataframe), 1)
        issue_rows = int(issue_field_counts.get(spec.column, 0))
        issue_rate = issue_rows / max(sample_size, 1)
        status = get_reliability_label(coverage, issue_rate, sample_size)
        if spec.column == "total_parent_income" and status == STATUS_RELIABLE:
            status = STATUS_CAUTION
        rows.append(
            {
                "metric": spec.label,
                "column": spec.column,
                "sample_size": sample_size,
                "coverage": coverage,
                "issue_rows": issue_rows,
                "issue_rate": issue_rate,
                "status": status,
                "reason": spec.why_it_matters,
            }
        )
    return pd.DataFrame(rows)


def build_issue_code_frame(issues_dataframe: pd.DataFrame, *, top_n: int = 12) -> pd.DataFrame:
    if issues_dataframe.empty or "issue_code" not in issues_dataframe.columns:
        return pd.DataFrame(columns=["issue_code", "issue_rows", "share"])
    counts = (
        issues_dataframe["issue_code"]
        .astype("string")
        .fillna("unknown")
        .str.strip()
        .replace("", "unknown")
        .value_counts()
        .head(top_n)
        .rename_axis("issue_code")
        .reset_index(name="issue_rows")
    )
    counts["share"] = counts["issue_rows"] / max(len(issues_dataframe), 1)
    return counts


def build_severity_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    severity = _severity_series(dataframe)
    order = ["high", "medium", "low", "none"]
    frame = severity.value_counts().rename_axis("severity").reset_index(name="cases")
    frame["severity"] = pd.Categorical(frame["severity"], categories=order, ordered=True)
    frame = frame.sort_values("severity").reset_index(drop=True)
    frame["share"] = frame["cases"] / max(frame["cases"].sum(), 1)
    return frame


def build_fairness_rules_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule": "Elevated subgroup risk",
                "trigger": "Validation fairness action is force_manual_review.",
                "action": "Block auto decision and route to committee.",
                "rationale": "Subgroup behavior differs enough that automation should not decide alone.",
            },
            {
                "rule": "Moderate subgroup risk",
                "trigger": "Fairness action tightens thresholds for the subgroup.",
                "action": "Allow automation only under stricter thresholds.",
                "rationale": "Automation remains possible, but confidence requirements increase.",
            },
            {
                "rule": "Small subgroup sample",
                "trigger": "Subgroup sample size is below 30.",
                "action": "Monitor only; do not infer fairness stability.",
                "rationale": "Very small groups produce unstable error estimates.",
            },
            {
                "rule": "Fairness flag on case",
                "trigger": "Case-level fairness_risk_flag is true.",
                "action": "Force manual review.",
                "rationale": "Guardrail flags override otherwise high model confidence.",
            },
        ]
    )


def summarize_fairness_actions(actions_dataframe: pd.DataFrame) -> dict[str, int]:
    if actions_dataframe.empty or "recommended_action" not in actions_dataframe.columns:
        return {"manual_review_groups": 0, "tightened_groups": 0, "monitor_only_groups": 0}
    actions = actions_dataframe["recommended_action"].astype("string").fillna("monitor_only")
    return {
        "manual_review_groups": int(actions.eq("force_manual_review").sum()),
        "tightened_groups": int(actions.str.contains("tighten", case=False, na=False).sum()),
        "monitor_only_groups": int(actions.eq("monitor_only").sum()),
    }


def build_decision_action_frame(
    decision_dataframe: pd.DataFrame,
    review_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if decision_dataframe.empty:
        return pd.DataFrame()

    working = decision_dataframe.copy()
    working["raw_source_row_number"] = pd.to_numeric(
        working["raw_source_row_number"],
        errors="coerce",
    )

    review_columns = [
        "raw_source_row_number",
        "qa_max_severity",
        "qa_issue_count",
        "qa_quality_score",
        "qa_requires_review",
        "qa_school_was_missing",
        "qa_risk_categories",
    ]
    if not review_dataframe.empty:
        review_lookup = review_dataframe.copy()
        review_lookup["raw_source_row_number"] = pd.to_numeric(
            review_lookup["raw_source_row_number"],
            errors="coerce",
        )
        available_columns = [column for column in review_columns if column in review_lookup.columns]
        working = working.merge(
            review_lookup[available_columns].drop_duplicates("raw_source_row_number"),
            on="raw_source_row_number",
            how="left",
        )

    for column, default in (
        ("qa_max_severity", "none"),
        ("qa_issue_count", 0),
        ("qa_quality_score", 100),
        ("qa_requires_review", 0),
        ("qa_school_was_missing", 0),
        ("qa_risk_categories", ""),
    ):
        if column not in working.columns:
            working[column] = default
        else:
            working[column] = working[column].fillna(default)

    working["data_confidence"] = working.apply(quality_adjusted_reliability_label, axis=1)
    if "fairness_risk_flag" in working.columns:
        working["fairness_risk_flag"] = (
            working["fairness_risk_flag"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
    else:
        working["fairness_risk_flag"] = False
    working["model_score"] = pd.to_numeric(
        working.get("predicted_probability_awarded"),
        errors="coerce",
    )

    final_actions: list[str] = []
    reasons: list[str] = []
    for _, row in working.iterrows():
        saved_action = str(row.get("recommended_action", "review")).strip().lower()
        data_confidence = str(row.get("data_confidence", STATUS_CAUTION))
        fairness_risk = bool(row.get("fairness_risk_flag", False))
        qa_reason = str(row.get("qa_risk_categories", "")).strip()

        if fairness_risk:
            final_actions.append("Block automation due to data/fairness risk")
            reasons.append("Fairness guardrail overrides the model recommendation.")
        elif data_confidence in {STATUS_AUDIT, STATUS_HIGH_RISK, "Audit / Risk"}:
            final_actions.append("Deep manual review")
            reasons.append("QA severity, issue count, or quality score makes automation unsafe.")
        elif data_confidence == STATUS_CAUTION or saved_action == "review":
            final_actions.append("Committee review")
            reasons.append("Model confidence or data confidence is not strong enough for auto-routing.")
        elif saved_action in {"approve", "auto_award", "award"}:
            final_actions.append("Auto approve")
            reasons.append("High model confidence and acceptable data confidence support auto-routing.")
        elif saved_action in {"reject", "deny", "auto_zero"}:
            final_actions.append("Auto reject / likely deny")
            reasons.append("Low model/award signal under the configured decision policy.")
        else:
            final_actions.append("Committee review")
            reasons.append("No safe auto-action rule matched.")

        if qa_reason and qa_reason not in {"nan", "None"}:
            reasons[-1] = f"{reasons[-1]} Risk categories: {qa_reason}."

    working["final_action"] = final_actions
    working["decision_support_reason"] = reasons
    return working


def build_business_impact_summary(decision_actions: pd.DataFrame) -> dict[str, int | float]:
    if decision_actions.empty or "final_action" not in decision_actions.columns:
        return {
            "cases": 0,
            "automation_cases": 0,
            "automation_rate": 0.0,
            "review_cases": 0,
            "protected_cases": 0,
        }
    actions = decision_actions["final_action"].astype("string")
    auto_mask = actions.isin(["Auto approve", "Auto reject / likely deny"])
    protected_mask = actions.isin(["Deep manual review", "Block automation due to data/fairness risk"])
    return {
        "cases": int(len(decision_actions)),
        "automation_cases": int(auto_mask.sum()),
        "automation_rate": float(auto_mask.mean()),
        "review_cases": int((~auto_mask).sum()),
        "protected_cases": int(protected_mask.sum()),
    }
