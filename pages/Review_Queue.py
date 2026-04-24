from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils.chart_components import make_horizontal_bar, render_chart_card
from utils.data_loader import load_csv, load_optional_csv, prepare_dashboard_dataframe
from utils.prediction_utils import add_prediction_probabilities
from utils.source_paths import (
    DEFAULT_CARS_PATH,
    DEFAULT_FINANCIAL_ASSISTANTS_PATH,
    DEFAULT_ISSUES_PATH,
    DEFAULT_LOANS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROPERTIES_PATH,
    DEFAULT_REVIEW_REQUIRED_PATH,
)
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
)

try:
    from utils.model_loader import load_model
except Exception:
    load_model = None


SEVERITY_RANK = {
    "high": 4,
    "medium": 3,
    "low": 2,
    "none": 1,
}


def split_token_series(series: pd.Series) -> list[str]:
    tokens: set[str] = set()
    for value in series.astype("string").fillna(""):
        for token in str(value).split(";"):
            cleaned = token.strip()
            if cleaned:
                tokens.add(cleaned)
    return sorted(tokens)


def build_child_record_counts(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "raw_source_sheet_name",
                "raw_source_row_number",
                f"{prefix}_records",
            ]
        )

    counts = (
        frame.groupby(["source_sheet_name", "source_row_number"], as_index=False)
        .size()
        .rename(
            columns={
                "source_sheet_name": "raw_source_sheet_name",
                "source_row_number": "raw_source_row_number",
                "size": f"{prefix}_records",
            }
        )
    )
    return counts


def prepare_review_queue(
    raw_review_df: pd.DataFrame,
    *,
    issues_df: pd.DataFrame,
    loans_df: pd.DataFrame,
    properties_df: pd.DataFrame,
    assistants_df: pd.DataFrame,
    cars_df: pd.DataFrame,
) -> pd.DataFrame:
    review_df = prepare_dashboard_dataframe(raw_review_df).copy()
    review_df["raw_source_row_number"] = pd.to_numeric(
        review_df["raw_source_row_number"],
        errors="coerce",
    )
    review_df["severity_clean"] = (
        review_df["qa_max_severity"]
        .astype("string")
        .fillna("none")
        .str.strip()
        .str.lower()
        .replace("", "none")
    )
    review_df["severity_rank"] = review_df["severity_clean"].map(SEVERITY_RANK).fillna(0)
    review_df["qa_issue_count"] = pd.to_numeric(review_df["qa_issue_count"], errors="coerce").fillna(0)
    review_df["qa_quality_score"] = pd.to_numeric(
        review_df["qa_quality_score"],
        errors="coerce",
    ).fillna(0)
    review_df["qa_requires_review"] = pd.to_numeric(
        review_df["qa_requires_review"],
        errors="coerce",
    ).fillna(0)
    review_df["qa_school_was_missing"] = pd.to_numeric(
        review_df.get("qa_school_was_missing"),
        errors="coerce",
    ).fillna(0)
    review_df["risk_category_count"] = (
        review_df["qa_risk_categories"]
        .astype("string")
        .fillna("")
        .map(lambda value: sum(bool(token.strip()) for token in str(value).split(";")))
    )
    review_df["review_priority_score"] = (
        review_df["severity_rank"] * 100
        + review_df["qa_issue_count"] * 10
        + (100 - review_df["qa_quality_score"]).clip(lower=0)
        + review_df["risk_category_count"] * 6
        + review_df["qa_school_was_missing"] * 15
    )

    review_df["school_display"] = (
        review_df["parsed_school"]
        .astype("string")
        .fillna("Missing school")
        .str.strip()
        .replace("", "Missing school")
    )
    review_df["decision_display"] = (
        review_df["parsed_decision"]
        .astype("string")
        .fillna("Missing decision")
        .str.strip()
        .replace("", "Missing decision")
    )
    review_df["level_display"] = (
        review_df["parsed_level"]
        .astype("string")
        .fillna("Missing level")
        .str.strip()
        .replace("", "Missing level")
    )
    review_df["track_display"] = (
        review_df["inferred_application_track"]
        .astype("string")
        .fillna("Missing track")
        .str.strip()
        .replace("", "Missing track")
    )

    issue_counts = (
        issues_df.groupby("source_row_number", as_index=False)
        .size()
        .rename(columns={"source_row_number": "raw_source_row_number", "size": "logged_issue_rows"})
        if not issues_df.empty
        else pd.DataFrame(columns=["raw_source_row_number", "logged_issue_rows"])
    )
    if not issue_counts.empty:
        issue_counts["raw_source_row_number"] = pd.to_numeric(
            issue_counts["raw_source_row_number"],
            errors="coerce",
        )

    review_df = review_df.merge(
        issue_counts,
        on="raw_source_row_number",
        how="left",
    )

    for prefix, frame in (
        ("loan", loans_df),
        ("property", properties_df),
        ("assistant", assistants_df),
        ("car", cars_df),
    ):
        review_df = review_df.merge(
            build_child_record_counts(frame, prefix),
            on=["raw_source_sheet_name", "raw_source_row_number"],
            how="left",
        )

    count_columns = [
        "logged_issue_rows",
        "loan_records",
        "property_records",
        "assistant_records",
        "car_records",
    ]
    for column in count_columns:
        review_df[column] = pd.to_numeric(review_df[column], errors="coerce").fillna(0).astype(int)

    return review_df.sort_values(
        by=[
            "review_priority_score",
            "severity_rank",
            "qa_issue_count",
            "raw_source_row_number",
        ],
        ascending=[False, False, False, True],
        ignore_index=True,
    )


def add_optional_model_scores(review_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if load_model is None or DEFAULT_MODEL_PATH is None or not Path(DEFAULT_MODEL_PATH).exists():
        return review_df, None

    try:
        model = load_model(DEFAULT_MODEL_PATH)
        scored_df = add_prediction_probabilities(review_df, model)
    except Exception as exc:
        return review_df, f"Eligibility score unavailable: {exc}"

    return scored_df, None


def filter_review_queue(
    review_df: pd.DataFrame,
    *,
    severity_filter: str,
    selected_levels: list[str],
    selected_tracks: list[str],
    selected_risks: list[str],
    search_text: str,
    missing_school_only: bool,
) -> pd.DataFrame:
    filtered = review_df.copy()

    if severity_filter != "All":
        filtered = filtered[filtered["severity_clean"] == severity_filter.lower()]
    if selected_levels:
        filtered = filtered[filtered["level_display"].isin(selected_levels)]
    if selected_tracks:
        filtered = filtered[filtered["track_display"].isin(selected_tracks)]
    if selected_risks:
        risk_mask = filtered["qa_risk_categories"].astype("string").fillna("").map(
            lambda value: any(
                selected_token in {token.strip() for token in str(value).split(";") if token.strip()}
                for selected_token in selected_risks
            )
        )
        filtered = filtered[risk_mask]
    if missing_school_only:
        filtered = filtered[filtered["qa_school_was_missing"].eq(1)]
    if search_text.strip():
        needle = search_text.strip().lower()
        search_frame = pd.DataFrame(
            {
                "row": filtered["raw_source_row_number"].astype("Int64").astype("string"),
                "school": filtered["school_display"].astype("string"),
                "summary": filtered["qa_issue_summary"].astype("string").fillna(""),
            }
        ).fillna("")
        mask = (
            search_frame["row"].str.lower().str.contains(needle, regex=False)
            | search_frame["school"].str.lower().str.contains(needle, regex=False)
            | search_frame["summary"].str.lower().str.contains(needle, regex=False)
        )
        filtered = filtered[mask]

    return filtered.reset_index(drop=True)


def build_severity_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["severity", "applications"])
    order = ["high", "medium", "low", "none"]
    distribution = (
        frame["severity_clean"]
        .value_counts()
        .rename_axis("severity")
        .reset_index(name="applications")
    )
    distribution["severity_order"] = distribution["severity"].map({value: index for index, value in enumerate(order)})
    return distribution.sort_values("severity_order").drop(columns=["severity_order"])


def build_issue_code_distribution(
    issues_df: pd.DataFrame,
    *,
    case_rows: pd.Series,
    top_n: int = 12,
) -> pd.DataFrame:
    if issues_df.empty or case_rows.empty:
        return pd.DataFrame(columns=["issue_code", "issues"])

    working = issues_df.copy()
    working["source_row_number"] = pd.to_numeric(working["source_row_number"], errors="coerce")
    working = working[working["source_row_number"].isin(case_rows.dropna().astype(float))]
    if working.empty:
        return pd.DataFrame(columns=["issue_code", "issues"])

    return (
        working["issue_code"]
        .astype("string")
        .fillna("unknown")
        .value_counts()
        .rename_axis("issue_code")
        .reset_index(name="issues")
        .head(top_n)
    )


def build_risk_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["risk_category", "applications"])

    rows: list[dict[str, object]] = []
    for value in frame["qa_risk_categories"].astype("string").fillna(""):
        for token in str(value).split(";"):
            cleaned = token.strip()
            if cleaned:
                rows.append({"risk_category": cleaned})

    if not rows:
        return pd.DataFrame(columns=["risk_category", "applications"])

    return (
        pd.DataFrame(rows)["risk_category"]
        .value_counts()
        .rename_axis("risk_category")
        .reset_index(name="applications")
    )


def build_case_options(frame: pd.DataFrame) -> list[tuple[str, int]]:
    options: list[tuple[str, int]] = []
    for index, row in frame.head(200).iterrows():
        label = (
            f"#{int(row['raw_source_row_number'])} | "
            f"{row['level_display']} | "
            f"{row['school_display']} | "
            f"{str(row['severity_clean']).title()} severity"
        )
        options.append((label, index))
    return options


def format_value(value: object, *, pct: bool = False, money: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if pct:
        return f"{float(value):.1f}%"
    if money:
        return f"{float(value):,.0f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{float(value):,.2f}"
    return str(value)


def build_case_snapshot(case_row: pd.Series) -> pd.DataFrame:
    fields = [
        ("Source Row", format_value(case_row.get("raw_source_row_number"))),
        ("Level", format_value(case_row.get("level_display"))),
        ("Application Track", format_value(case_row.get("track_display"))),
        ("Application Term", format_value(case_row.get("parsed_application_term"))),
        ("School", format_value(case_row.get("school_display"))),
        ("Decision", format_value(case_row.get("decision_display"))),
        ("Nationality", format_value(case_row.get("parsed_nationality"))),
        ("Submission Date", format_value(case_row.get("parsed_submission_date"))),
        ("Total Parent Income", format_value(case_row.get("total_parent_income"), money=True)),
        ("Need Percentage", format_value(case_row.get("parsed_need_pct"), pct=True)),
        ("Estimated Property Value", format_value(case_row.get("properties_total_estimated_value"), money=True)),
        ("Remaining Loan Balance", format_value(case_row.get("loans_total_remaining_balance"), money=True)),
        ("Other Siblings Tuition", format_value(case_row.get("siblings_other_total_tuition"), money=True)),
        ("Dependents Count", format_value(case_row.get("dependents_count"))),
        ("Total Siblings", format_value(case_row.get("total_siblings"))),
    ]
    return pd.DataFrame(fields, columns=["Field", "Value"]).astype("string")


def build_case_model_snapshot(case_row: pd.Series) -> pd.DataFrame:
    rows = [
        ("Review Priority Score", format_value(case_row.get("review_priority_score"))),
        ("QA Severity", str(case_row.get("severity_clean", "none")).title()),
        ("Logged Issues", format_value(case_row.get("qa_issue_count"))),
        ("Quality Score", format_value(case_row.get("qa_quality_score"))),
        ("Risk Categories", case_row.get("qa_risk_categories") or "N/A"),
        ("School Missing", "Yes" if pd.to_numeric(case_row.get("qa_school_was_missing"), errors="coerce") == 1 else "No"),
    ]
    if "predicted_award_probability" in case_row.index:
        rows.extend(
            [
                (
                    "Eligibility Probability",
                    format_value(case_row.get("predicted_award_probability", np.nan) * 100, pct=True),
                ),
                ("Model Label", case_row.get("predicted_award_label") or "N/A"),
                (
                    "Threshold Used",
                    format_value(case_row.get("prediction_threshold_used", np.nan) * 100, pct=True),
                ),
            ]
        )
    return pd.DataFrame(rows, columns=["Field", "Value"]).astype("string")


def filter_child_records(frame: pd.DataFrame, case_row: pd.Series) -> pd.DataFrame:
    if frame.empty:
        return frame

    filtered = frame.copy()
    if "source_sheet_name" in filtered.columns and "raw_source_sheet_name" in case_row.index:
        filtered = filtered[
            filtered["source_sheet_name"].astype("string")
            == str(case_row.get("raw_source_sheet_name"))
        ]
    if "source_row_number" in filtered.columns:
        filtered = filtered[
            pd.to_numeric(filtered["source_row_number"], errors="coerce")
            == pd.to_numeric(case_row.get("raw_source_row_number"), errors="coerce")
        ]
    return filtered.reset_index(drop=True)


def render_child_table(title: str, frame: pd.DataFrame, columns: list[str]) -> None:
    st.markdown(f"**{title}**")
    if frame.empty:
        st.info(f"No {title.lower()} found for this case.")
        return

    available_columns = [column for column in columns if column in frame.columns]
    st.dataframe(frame[available_columns], width="stretch", hide_index=True)


apply_design_system()

if not DEFAULT_REVIEW_REQUIRED_PATH.exists():
    st.error(f"Review-required dataset not found: {DEFAULT_REVIEW_REQUIRED_PATH}")
else:
    raw_review_df = load_csv(DEFAULT_REVIEW_REQUIRED_PATH)
    issues_df = load_optional_csv(DEFAULT_ISSUES_PATH)
    loans_df = load_optional_csv(DEFAULT_LOANS_PATH)
    properties_df = load_optional_csv(DEFAULT_PROPERTIES_PATH)
    assistants_df = load_optional_csv(DEFAULT_FINANCIAL_ASSISTANTS_PATH)
    cars_df = load_optional_csv(DEFAULT_CARS_PATH)

    review_df = prepare_review_queue(
        raw_review_df,
        issues_df=issues_df,
        loans_df=loans_df,
        properties_df=properties_df,
        assistants_df=assistants_df,
        cars_df=cars_df,
    )
    review_df, scoring_note = add_optional_model_scores(review_df)

    severity_options = ["All", "High", "Medium", "Low", "None"]
    level_options = sorted(review_df["level_display"].dropna().unique().tolist())
    track_options = sorted(review_df["track_display"].dropna().unique().tolist())
    risk_options = split_token_series(review_df["qa_risk_categories"])

    render_decision_journey("Action")
    render_page_header(
        title="Review Queue",
        description=(
            "Operational review workspace for the cleaned-data pipeline: rank risky cases, "
            "inspect exact issue drivers, and drill into the structured household records behind each application."
        ),
        takeaway=(
            f"{len(review_df):,} cases require review in the cleaned export, with "
            f"{int(review_df['severity_clean'].eq('high').sum()):,} already flagged as high severity."
        ),
        kicker="Operations Review Workspace",
        pills=[
            (f"{len(review_df):,} review-required cases", "danger"),
            (f"{int(review_df['qa_issue_count'].sum()):,} logged issue flags", "warning"),
            (f"{int(review_df['qa_school_was_missing'].sum()):,} missing-school cases", "primary"),
        ],
    )
    render_kpi_row(
        [
            {
                "label": "High Severity Cases",
                "value": f"{int(review_df['severity_clean'].eq('high').sum()):,}",
                "note": "Rows where the cleaning pipeline found the strongest unresolved concerns.",
                "tone": "danger",
            },
            {
                "label": "Average Quality Score",
                "value": f"{float(review_df['qa_quality_score'].mean()):.1f}",
                "note": "Lower scores indicate more parsing ambiguity or unresolved QA issues.",
                "tone": "secondary",
            },
            {
                "label": "Average Issues Per Case",
                "value": f"{float(review_df['qa_issue_count'].mean()):.1f}",
                "note": "Based on the exported review-required subset only.",
                "tone": "warning",
            },
            {
                "label": "Cases With Child Records",
                "value": f"{int((review_df[['loan_records', 'property_records', 'assistant_records', 'car_records']].sum(axis=1) > 0).sum()):,}",
                "note": "These cases have supporting structured records you can inspect in detail.",
                "tone": "primary",
            },
        ]
    )
    render_insight_action_panel(
        insight=(
            "This page is strongest when used as a reviewer triage workspace: start with severity and issue-count concentration, "
            "then open the exact case details before making any interpretation."
        ),
        implication=(
            "Use the queue to prioritize human review, not to replace it. The exported QA fields tell you where the data still needs caution."
        ),
    )

    st.markdown("**Queue Filters**")
    filter_cols = st.columns(5)
    with filter_cols[0]:
        severity_filter = st.selectbox(
            "Severity",
            severity_options,
            index=0,
        )
    with filter_cols[1]:
        selected_levels = st.multiselect(
            "Academic Level",
            options=level_options,
            default=level_options,
        )
    with filter_cols[2]:
        selected_tracks = st.multiselect(
            "Application Track",
            options=track_options,
            default=track_options,
        )
    with filter_cols[3]:
        selected_risks = st.multiselect(
            "Risk Category",
            options=risk_options,
            default=[],
        )
    with filter_cols[4]:
        search_text = st.text_input(
            "Search row or school",
            value="",
            placeholder="e.g. 401 or Choueifat",
        )

    missing_school_only = st.checkbox("Only cases with missing school", value=False)

    filtered_queue = filter_review_queue(
        review_df,
        severity_filter=severity_filter,
        selected_levels=selected_levels,
        selected_tracks=selected_tracks,
        selected_risks=selected_risks,
        search_text=search_text,
        missing_school_only=missing_school_only,
    )

    severity_distribution = build_severity_distribution(filtered_queue)
    issue_code_distribution = build_issue_code_distribution(
        issues_df,
        case_rows=filtered_queue["raw_source_row_number"],
    )
    risk_distribution = build_risk_distribution(filtered_queue)

    chart_cols = st.columns(3)
    with chart_cols[0]:
        if severity_distribution.empty:
            render_chart_card(
                title="Severity Mix",
                warning="No cases match the current filters.",
                status="Caution",
            )
        else:
            render_chart_card(
                title="Severity Mix",
                subtitle="Filtered review queue by QA severity.",
                takeaway="High-severity cases should be protected from automation first.",
                chart=make_horizontal_bar(
                    severity_distribution,
                    x="applications",
                    y="severity",
                    color_field="severity",
                    color_scale=alt.Scale(
                        domain=["high", "medium", "low", "none"],
                        range=[DANGER_COLOR, ACCENT_COLOR, SECONDARY_COLOR, PRIMARY_COLOR],
                    ),
                    x_title="Cases",
                    height=280,
                ),
                caption="The chart reflects the current filters, so it can be used for operational triage.",
                status="Caution",
                sample_size=int(severity_distribution["applications"].sum()),
            )
    with chart_cols[1]:
        if issue_code_distribution.empty:
            render_chart_card(
                title="Top Issue Codes",
                warning="No issue-code rows match the current filters.",
                status="Caution",
            )
        else:
            render_chart_card(
                title="Top Issue Codes",
                subtitle="Review burden by QA issue type.",
                takeaway="Top issue codes point to the fastest process improvements.",
                chart=make_horizontal_bar(
                    issue_code_distribution,
                    x="issues",
                    y="issue_code",
                    color=PRIMARY_COLOR,
                    x_title="Issue rows",
                    height=280,
                ),
                caption="Use this to decide which cleaning or collection fixes would reduce queue burden.",
                status="Caution",
                sample_size=int(issue_code_distribution["issues"].sum()),
            )
    with chart_cols[2]:
        if risk_distribution.empty:
            render_chart_card(
                title="Risk Categories",
                warning="No risk categories are present in the current filter slice.",
                status="Reliable",
            )
        else:
            render_chart_card(
                title="Risk Categories",
                subtitle="Operational triage reasons in the current queue.",
                takeaway="Risk categories explain why cases stay in human review.",
                chart=make_horizontal_bar(
                    risk_distribution.head(10),
                    x="applications",
                    y="risk_category",
                    color=SECONDARY_COLOR,
                    x_title="Cases",
                    height=280,
                ),
                caption="This is a queue-management chart, not a model-performance chart.",
                status="Caution",
                sample_size=int(risk_distribution["applications"].sum()),
            )

    st.markdown("**Ranked Queue**")
    if scoring_note:
        st.caption(scoring_note)
    st.caption(
        "Review priority combines exported QA severity, issue volume, missing-school flags, risk-category count, and lower quality scores."
    )
    queue_display_columns = [
        "review_priority_score",
        "raw_source_row_number",
        "level_display",
        "track_display",
        "school_display",
        "decision_display",
        "severity_clean",
        "qa_issue_count",
        "qa_quality_score",
        "logged_issue_rows",
        "loan_records",
        "property_records",
        "assistant_records",
        "car_records",
    ]
    if "predicted_award_probability" in filtered_queue.columns:
        queue_display_columns.append("predicted_award_probability")

    queue_display = filtered_queue[queue_display_columns].copy()
    queue_display = queue_display.rename(
        columns={
            "review_priority_score": "Priority",
            "raw_source_row_number": "Source Row",
            "level_display": "Level",
            "track_display": "Track",
            "school_display": "School",
            "decision_display": "Decision",
            "severity_clean": "Severity",
            "qa_issue_count": "QA Issues",
            "qa_quality_score": "Quality Score",
            "logged_issue_rows": "Issue Rows",
            "loan_records": "Loans",
            "property_records": "Properties",
            "assistant_records": "Assistants",
            "car_records": "Cars",
            "predicted_award_probability": "Eligibility Probability",
        }
    )
    if "Eligibility Probability" in queue_display.columns:
        queue_display["Eligibility Probability"] = pd.to_numeric(
            queue_display["Eligibility Probability"],
            errors="coerce",
        ).map(lambda value: f"{value:.1%}" if pd.notna(value) else "N/A")
    st.dataframe(queue_display, width="stretch", hide_index=True)

    st.markdown("**Case Drill-Down**")
    if filtered_queue.empty:
        st.info("No cases match the current filters.")
    else:
        case_options = build_case_options(filtered_queue)
        selected_label = st.selectbox(
            "Select a case",
            options=[label for label, _ in case_options],
        )
        selected_index = dict(case_options)[selected_label]
        case_row = filtered_queue.loc[selected_index]

        case_cols = st.columns(4)
        with case_cols[0]:
            st.metric("Severity", str(case_row["severity_clean"]).title())
        with case_cols[1]:
            st.metric("QA Issues", f"{int(case_row['qa_issue_count']):,}")
        with case_cols[2]:
            st.metric("Quality Score", f"{float(case_row['qa_quality_score']):.0f}")
        with case_cols[3]:
            if "predicted_award_probability" in case_row.index:
                st.metric(
                    "Eligibility Probability",
                    f"{float(case_row['predicted_award_probability']):.1%}",
                )
            else:
                st.metric("Eligibility Probability", "N/A")

        summary_tab, issues_tab, records_tab, raw_tab = st.tabs(
            ["Summary", "Issues", "Structured Records", "Raw Detail"]
        )

        with summary_tab:
            summary_cols = st.columns(2)
            with summary_cols[0]:
                st.markdown("**Application Snapshot**")
                st.dataframe(
                    build_case_snapshot(case_row),
                    width="stretch",
                    hide_index=True,
                )
            with summary_cols[1]:
                st.markdown("**Review Snapshot**")
                st.dataframe(
                    build_case_model_snapshot(case_row),
                    width="stretch",
                    hide_index=True,
                )
            if pd.notna(case_row.get("qa_issue_summary")) and str(case_row.get("qa_issue_summary")).strip():
                st.markdown("**Issue Summary**")
                st.caption(str(case_row.get("qa_issue_summary")).strip())

        with issues_tab:
            case_issues = issues_df.copy()
            if not case_issues.empty:
                case_issues["source_row_number"] = pd.to_numeric(
                    case_issues["source_row_number"],
                    errors="coerce",
                )
                case_issues = case_issues[
                    case_issues["source_row_number"]
                    == pd.to_numeric(case_row.get("raw_source_row_number"), errors="coerce")
                ]
            if case_issues.empty:
                st.info("No row-level issues were exported for this case.")
            else:
                st.dataframe(
                    case_issues[
                        [
                            column
                            for column in [
                                "field_name",
                                "issue_code",
                                "detail",
                                "raw_value",
                            ]
                            if column in case_issues.columns
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

        with records_tab:
            loan_records = filter_child_records(loans_df, case_row)
            property_records = filter_child_records(properties_df, case_row)
            assistant_records = filter_child_records(assistants_df, case_row)
            car_records = filter_child_records(cars_df, case_row)

            record_tabs = st.tabs(["Loans", "Properties", "Assistants", "Cars"])
            with record_tabs[0]:
                render_child_table(
                    "Loans",
                    loan_records,
                    [
                        "loan_reason",
                        "loan_source",
                        "loan_total_amount",
                        "loan_monthly_payment",
                        "loan_remaining_balance",
                        "loan_remaining_years",
                        "loan_review_flag",
                        "loan_review_reasons",
                    ],
                )
            with record_tabs[1]:
                render_child_table(
                    "Properties",
                    property_records,
                    [
                        "property_type",
                        "property_location",
                        "property_area",
                        "property_estimated_present_value",
                        "property_mortgaged",
                        "property_rented_income",
                        "property_planted_income",
                        "property_review_flag",
                        "property_review_reasons",
                    ],
                )
            with record_tabs[2]:
                render_child_table(
                    "Assistants",
                    assistant_records,
                    [
                        "assistant_subject",
                        "assistant_amount_text",
                        "assistant_estimated_annual_amount",
                        "assistant_currency_signal",
                        "assistant_review_flag",
                        "assistant_review_reasons",
                    ],
                )
            with record_tabs[3]:
                render_child_table(
                    "Cars",
                    car_records,
                    [
                        "car_type",
                        "car_model",
                        "car_mortgaged",
                        "car_review_flag",
                        "car_review_reasons",
                    ],
                )

        with raw_tab:
            raw_fields = [
                ("Special Circumstances Detail", case_row.get("parsed_special_family_circumstances_detail")),
                ("Need Comment", case_row.get("parsed_need_comment")),
                ("Raw Father Work Status", case_row.get("raw_father_work_status")),
                ("Raw Mother Work Status", case_row.get("raw_mother_work_status")),
                ("Raw Financial Assistants", case_row.get("raw_financial_assistants")),
                ("Raw Loans", case_row.get("raw_loans")),
                ("Raw Properties", case_row.get("raw_properties")),
                ("Raw Cars", case_row.get("raw_cars")),
            ]
            any_detail = False
            for title, value in raw_fields:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    continue
                text_value = str(value).strip()
                if not text_value:
                    continue
                any_detail = True
                with st.expander(title, expanded=False):
                    st.text(text_value)
            if not any_detail:
                st.info("No additional raw detail blocks were exported for this case.")
