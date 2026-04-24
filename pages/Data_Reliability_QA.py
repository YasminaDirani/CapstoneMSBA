from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from utils.chart_components import (
    build_financial_signal_frame,
    build_issue_cooccurrence_frame,
    build_missingness_frame,
    make_coverage_reliability_chart,
    make_financial_signal_chart,
    make_issue_cooccurrence_heatmap,
    make_issue_pareto_chart,
    make_missingness_heatmap,
    make_horizontal_bar,
    render_chart_card,
    render_table_card,
)
from utils.data_loader import load_csv, prepare_dashboard_dataframe
from utils.decision_support import (
    STATUS_AUDIT,
    STATUS_CAUTION,
    STATUS_HIGH_RISK,
    STATUS_RELIABLE,
    apply_high_confidence_mode,
    build_feature_reliability_frame,
    build_issue_code_frame,
    build_qa_summary,
    build_severity_frame,
    format_pct,
    status_tone,
)
from utils.source_paths import DEFAULT_DATA_PATH, DEFAULT_ISSUES_PATH, DEFAULT_REVIEW_REQUIRED_PATH
from utils.ui import (
    ACCENT_COLOR,
    DANGER_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    render_decision_journey,
    render_metric_card,
    render_page_header,
    render_takeaway_box,
)


@st.cache_data(show_spinner=False)
def load_qa_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = prepare_dashboard_dataframe(load_csv(DEFAULT_DATA_PATH))
    review = prepare_dashboard_dataframe(load_csv(DEFAULT_REVIEW_REQUIRED_PATH))
    issues = load_csv(DEFAULT_ISSUES_PATH)
    return data, review, issues


dataframe, review_dataframe, issues_dataframe = load_qa_inputs()

mode = st.radio(
    "Confidence mode",
    ["All data mode", "High-confidence only mode"],
    horizontal=True,
)
working_dataframe = apply_high_confidence_mode(
    dataframe,
    high_confidence_only=mode == "High-confidence only mode",
)

qa_summary = build_qa_summary(working_dataframe, review_dataframe, issues_dataframe)
feature_reliability = build_feature_reliability_frame(working_dataframe, issues_dataframe)
issue_codes = build_issue_code_frame(issues_dataframe)
severity_mix = build_severity_frame(working_dataframe)

render_decision_journey("Data Quality")
render_page_header(
    title="Data Reliability & QA",
    description=(
        "The QA workspace turns cleaning issues into decision governance. It shows what can be trusted, "
        "what needs committee caution, and what should block automation."
    ),
    takeaway=(
        "QA is not an appendix: it is the control layer that decides whether model evidence can be used."
    ),
    kicker="Step 1: Data Quality",
    pills=[
        ("Reliable", "primary"),
        ("Caution", "warning"),
        ("Audit", "audit"),
        ("High Risk", "danger"),
        ("Not Production Ready", "secondary"),
    ],
)

render_takeaway_box(
    "Most records are usable for descriptive evidence, but QA review flags still define the automation boundary.",
    status=STATUS_CAUTION,
)

metric_columns = st.columns(4)
with metric_columns[0]:
    render_metric_card(
        label="Total Records",
        value=f"{int(qa_summary['total_records']):,}",
        interpretation="Records in the active confidence mode.",
        status=STATUS_RELIABLE,
        sample_size=int(qa_summary["total_records"]),
        coverage="100.0%",
    )
with metric_columns[1]:
    render_metric_card(
        label="Review Required",
        value=f"{int(qa_summary['review_required_cases']):,}",
        interpretation="Rows where QA says human review should stay active.",
        status=STATUS_CAUTION,
        sample_size=int(qa_summary["review_required_cases"]),
        coverage=format_pct(float(qa_summary["review_required_rate"])),
    )
with metric_columns[2]:
    render_metric_card(
        label="High Severity",
        value=f"{int(qa_summary['high_severity_cases']):,}",
        interpretation="Cases with severe QA issues that should block automation.",
        status=STATUS_HIGH_RISK if int(qa_summary["high_severity_cases"]) else STATUS_RELIABLE,
        sample_size=int(qa_summary["high_severity_cases"]),
        coverage=format_pct(float(qa_summary["high_severity_rate"])),
    )
with metric_columns[3]:
    render_metric_card(
        label="Issue Rows",
        value=f"{int(qa_summary['issue_rows']):,}",
        interpretation="Logged issue rows across exported QA checks.",
        status=STATUS_CAUTION,
        sample_size=int(qa_summary["issue_rows"]),
        coverage=f"avg quality {float(qa_summary['avg_quality_score']):.0f}",
    )

chart_columns = st.columns(2)
with chart_columns[0]:
    if severity_mix.empty:
        render_chart_card(
            title="QA Severity Mix",
            status=STATUS_CAUTION,
            warning="No severity fields are available.",
        )
    else:
        severity_chart = make_horizontal_bar(
            severity_mix,
            x="cases",
            y="severity",
            color_field="severity",
            color_scale=alt.Scale(
                domain=["high", "medium", "low", "none"],
                range=[DANGER_COLOR, ACCENT_COLOR, SECONDARY_COLOR, PRIMARY_COLOR],
            ),
            x_title="Cases",
            tooltip=[
                alt.Tooltip("severity:N", title="Severity"),
                alt.Tooltip("cases:Q", title="Cases"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
            height=300,
        )
        render_chart_card(
            title="QA Severity Mix",
            subtitle="Stacked risk categories converted into review pressure.",
            takeaway="High and medium severities should be read as review pressure, not just data-cleaning noise.",
            chart=severity_chart,
            caption="Horizontal bars keep severity labels readable; high severity is treated as a high-risk automation blocker.",
            status=STATUS_CAUTION,
            sample_size=len(working_dataframe),
        )

with chart_columns[1]:
    if issue_codes.empty:
        render_chart_card(
            title="Which Data Issues Drive Most Review Burden?",
            status=STATUS_CAUTION,
            warning="No issue-code export is available.",
        )
    else:
        render_chart_card(
            title="Which Data Issues Drive Most Review Burden?",
            subtitle="Pareto view of issue concentration.",
            takeaway="Issue-code concentration identifies the operational fixes that would most improve model trust.",
            chart=make_issue_pareto_chart(issue_codes),
            caption="Bars show issue rows; the red line shows cumulative share so FAID can see whether a few fixes remove most review burden.",
            status=STATUS_CAUTION,
            sample_size=len(issues_dataframe),
        )

qa_visual_columns = st.columns(2)
missingness_fields = [
    ("Decision label", "decision"),
    ("School/context", "school"),
    ("Parent income", "total_parent_income"),
    ("Property/assets", "properties_total_estimated_value"),
    ("Cars/assets", "cars_count"),
    ("Loans/liabilities", "loans_total_remaining_balance"),
    ("External assistance", "financial_assistants_total_est_annual_amount"),
    ("Household burden", "total_siblings"),
]
missingness_group = "inferred_application_track" if "inferred_application_track" in working_dataframe.columns else "parsed_level"
missingness_frame = build_missingness_frame(
    working_dataframe,
    missingness_fields,
    group_by=missingness_group,
)
cooccurrence_frame = build_issue_cooccurrence_frame(issues_dataframe)
with qa_visual_columns[0]:
    render_chart_card(
        title="Missingness Heatmap",
        subtitle=f"Field-group coverage by {missingness_group.replace('_', ' ')}.",
        takeaway="Missingness is a business pattern, not just a technical defect.",
        chart=make_missingness_heatmap(missingness_frame),
        caption="Darker cells indicate higher missingness; concentrated gaps should be fixed before relying on subgroup or affordability comparisons.",
        status=STATUS_CAUTION,
        sample_size=len(working_dataframe),
    )
with qa_visual_columns[1]:
    render_chart_card(
        title="Issue Co-occurrence Heatmap",
        subtitle="Pairs of QA issue types appearing on the same source row.",
        takeaway="Co-occurring issues identify cases where one cleaning fix may not be enough.",
        chart=make_issue_cooccurrence_heatmap(cooccurrence_frame),
        caption="Cells count cases where both issue types appear. Dense clusters should be reviewed before automation.",
        status=STATUS_AUDIT if not cooccurrence_frame.empty else STATUS_CAUTION,
        sample_size=int(cooccurrence_frame["cases"].sum()) if not cooccurrence_frame.empty else None,
        warning="No issue co-occurrence chart is shown because required issue fields are missing." if cooccurrence_frame.empty else None,
    )

st.subheader("Metric Reliability Layer")
render_takeaway_box(
    "High coverage does not always mean high reliability if QA flags are concentrated.",
    status=STATUS_RELIABLE,
)
render_chart_card(
    title="Coverage vs Reliability",
    subtitle="Key evidence metrics by coverage, issue concentration, and sample size.",
    takeaway="Metrics move from use-ready to audit-only when coverage falls, sample size is weak, or issue concentration rises.",
    chart=make_coverage_reliability_chart(feature_reliability),
    caption="Bubble size is sample size; color is reliability status. The chart prevents high-volume fields from being overtrusted when issue rates are high.",
    status=STATUS_RELIABLE,
    sample_size=len(working_dataframe),
)

financial_fields = [
    ("Parent income", "total_parent_income"),
    ("Father income", "father_gross_income"),
    ("Mother income", "mother_gross_income"),
    ("Property value", "properties_total_estimated_value"),
    ("Loan balance", "loans_total_remaining_balance"),
    ("External assistance", "financial_assistants_total_est_annual_amount"),
    ("Sibling tuition", "siblings_other_total_tuition"),
    ("Cars/assets", "cars_count"),
]
financial_signal_frame = build_financial_signal_frame(working_dataframe, issues_dataframe, financial_fields)
render_chart_card(
    title="Financial Signal Reliability",
    subtitle="Coverage, QA issue rate, and awarded-vs-denied medians for affordability signals.",
    takeaway="Property and structured asset signals are stronger than raw income alone; income remains useful but cautious.",
    chart=make_financial_signal_chart(financial_signal_frame),
    caption="Bubble size is available sample size. Financial variables should not be interpreted without coverage and QA context.",
    status=STATUS_CAUTION,
    sample_size=len(working_dataframe),
    warning="Currency ambiguity and missingness mean income should remain a support signal, not a standalone decision rule.",
)

display_frame = feature_reliability.copy()
display_frame["coverage"] = display_frame["coverage"].map(format_pct)
display_frame["issue_rate"] = display_frame["issue_rate"].map(format_pct)
display_frame["status_tone"] = display_frame["status"].map(status_tone)
render_table_card(
    title="Metric Reliability Table",
    dataframe=display_frame[
        ["metric", "sample_size", "coverage", "issue_rows", "issue_rate", "status", "reason"]
    ],
    takeaway="Every decision-facing metric is labeled before it is reused elsewhere.",
    caption="Statuses are configurable assumptions based on coverage, issue concentration, and sample size.",
    status=STATUS_RELIABLE,
)

with st.expander("Business impact of data issues"):
    st.write(
        "Data quality determines routing: high-severity and low-confidence cases should become committee or deep manual review, "
        "while reliable cases can safely move to model evidence and decision-action review."
    )
    st.write(
        f"Missing-school cases account for {format_pct(float(qa_summary['missing_school_rate']))} "
        "of records in the active mode, which directly affects subgroup and representation analysis."
    )
