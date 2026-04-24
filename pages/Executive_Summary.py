from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.chart_components import render_decision_grid, render_system_flow, render_table_card
from utils.data_loader import load_csv, load_json
from utils.decision_labels import normalize_decision_series
from utils.decision_support import (
    STATUS_AUDIT,
    STATUS_CAUTION,
    STATUS_HIGH_RISK,
    STATUS_NOT_PRODUCTION,
    STATUS_RELIABLE,
    apply_high_confidence_mode,
    build_feature_reliability_frame,
    build_qa_summary,
    format_pct,
    high_confidence_mask,
)
from utils.source_paths import (
    DEFAULT_DATA_PATH,
    DEFAULT_ISSUES_PATH,
    DEFAULT_MODEL_METADATA_PATH,
    DEFAULT_PERCENTAGE_METADATA_PATH,
    DEFAULT_REVIEW_REQUIRED_PATH,
)
from utils.ui import (
    render_decision_journey,
    render_insight_action_panel,
    render_metric_card,
    render_page_header,
    render_takeaway_box,
)


@st.cache_data(show_spinner=False)
def load_executive_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    data = load_csv(DEFAULT_DATA_PATH)
    review = load_csv(DEFAULT_REVIEW_REQUIRED_PATH)
    issues = load_csv(DEFAULT_ISSUES_PATH)
    eligibility_metadata = load_json(DEFAULT_MODEL_METADATA_PATH)
    percentage_metadata = load_json(DEFAULT_PERCENTAGE_METADATA_PATH)
    return data, review, issues, eligibility_metadata, percentage_metadata


dataframe, review_dataframe, issues_dataframe, eligibility_metadata, percentage_metadata = load_executive_inputs()

mode = st.radio(
    "Confidence mode",
    ["All data mode", "High-confidence only mode"],
    horizontal=True,
)
high_confidence_only = mode == "High-confidence only mode"
working_dataframe = apply_high_confidence_mode(
    dataframe,
    high_confidence_only=high_confidence_only,
)

render_decision_journey("Data Quality")
render_page_header(
    title="Start Here / Executive Summary",
    description=(
        "A governed readout of what the FAID data and models can support today, "
        "where the evidence is risky, and what should happen next."
    ),
    takeaway=(
        "Use the system for decision support, committee prioritization, and consistency review; "
        "do not position it as a fully automated eligibility engine."
    ),
    kicker="Decision Intelligence Platform",
    pills=[
        ("Data Quality -> Evidence -> Model -> Decision -> Action", "primary"),
        ("Decision support only", "warning"),
        ("High-confidence mode available", "secondary"),
    ],
)

normalized_decision = normalize_decision_series(working_dataframe.get("parsed_decision", working_dataframe.get("decision")))
valid_comparison_count = int(normalized_decision.notna().sum())
qa_summary = build_qa_summary(working_dataframe, review_dataframe, issues_dataframe)
feature_reliability = build_feature_reliability_frame(working_dataframe, issues_dataframe)
reliable_feature_count = int(feature_reliability["status"].eq(STATUS_RELIABLE).sum())
audit_feature_count = int(feature_reliability["status"].eq(STATUS_AUDIT).sum())

normal_workflow = eligibility_metadata.get("normal_workflow", {}) if isinstance(eligibility_metadata, dict) else {}
test_summary = normal_workflow.get("test_summary", {}) if isinstance(normal_workflow, dict) else {}
test_metrics = test_summary.get("selected_policy_metrics", {}) if isinstance(test_summary, dict) else {}
model_auc = test_summary.get("probability_metrics", {}).get("roc_auc") if isinstance(test_summary, dict) else None
recall = test_metrics.get("recall") if isinstance(test_metrics, dict) else None
percentage_summary = percentage_metadata.get("decision_summary", {}) if isinstance(percentage_metadata, dict) else {}
automation_rate = percentage_summary.get("automation_rate")


def _feature_status(metric: str, fallback: str = STATUS_CAUTION) -> str:
    matches = feature_reliability.loc[feature_reliability["metric"].eq(metric), "status"]
    return str(matches.iloc[0]) if not matches.empty else fallback

render_takeaway_box(
    "The evidence base is broad, but the reliable operating zone is narrower than the full dataset.",
    status=STATUS_CAUTION,
)

metric_columns = st.columns(4)
with metric_columns[0]:
    render_metric_card(
        label="Dataset",
        value=f"{len(working_dataframe):,}",
        interpretation="Application records in the active confidence mode.",
        status=STATUS_RELIABLE,
        sample_size=len(working_dataframe),
        coverage=format_pct(len(working_dataframe) / max(len(dataframe), 1)),
    )
with metric_columns[1]:
    render_metric_card(
        label="Comparable Outcomes",
        value=f"{valid_comparison_count:,}",
        interpretation="Rows that can support awarded-versus-denied comparisons.",
        status=STATUS_RELIABLE if valid_comparison_count / max(len(working_dataframe), 1) >= 0.9 else STATUS_CAUTION,
        sample_size=valid_comparison_count,
        coverage=format_pct(valid_comparison_count / max(len(working_dataframe), 1)),
    )
with metric_columns[2]:
    render_metric_card(
        label="QA Review Load",
        value=f"{int(qa_summary['review_required_cases']):,}",
        interpretation="Cases that should remain under human review pressure.",
        status=STATUS_CAUTION,
        sample_size=int(qa_summary["review_required_cases"]),
        coverage=format_pct(float(qa_summary["review_required_rate"])),
    )
with metric_columns[3]:
    render_metric_card(
        label="Model Support",
        value=f"AUC {model_auc:.3f}" if model_auc is not None else "Saved",
        interpretation="Holdout evidence supports ranking and review prioritization.",
        status=STATUS_CAUTION,
        sample_size=valid_comparison_count,
        coverage=f"recall {format_pct(recall, digits=0)}" if recall is not None else "holdout saved",
    )

render_insight_action_panel(
    insight=(
        f"{reliable_feature_count} key evidence areas are reliable, while {audit_feature_count} "
        "remain audit-risk or not production-ready."
    ),
    implication=(
        "FAID should use the app to prioritize review, surface exceptions, and protect risky cases from automation."
    ),
)

st.subheader("Executive Visual Summary")
render_system_flow(
    [
        ("Raw Data", "Original FAID application exports and source-row identifiers.", STATUS_CAUTION),
        ("Cleaned Data", "Structured application, family, asset, income, and QA fields.", STATUS_RELIABLE),
        ("QA / Reliability", "Coverage, issue concentration, and high-confidence filtering.", STATUS_CAUTION),
        ("Model Score", "Eligibility and aid-percentage models support ranking, not final judgment.", STATUS_CAUTION),
        ("Fairness Guardrails", "Subgroup risk can tighten thresholds or block automation.", STATUS_AUDIT),
        ("Recommended Action", "Human-in-the-loop routing for committee and manual review.", STATUS_RELIABLE),
    ]
)

trust_grid = pd.DataFrame(
    [
        {
            "Area": "Outcome labels",
            "Status": _feature_status("Decision Label", STATUS_RELIABLE),
            "Reason": "Comparable awarded/denied outcomes are available for evidence and model evaluation.",
            "Action": "Use for descriptive evidence and model validation.",
        },
        {
            "Area": "Income reliability",
            "Status": _feature_status("Parent Income", STATUS_CAUTION),
            "Reason": "Income is useful but has missingness and currency ambiguity.",
            "Action": "Use with affordability context, not as a standalone rule.",
        },
        {
            "Area": "Property/assets reliability",
            "Status": _feature_status("Property Value", STATUS_CAUTION),
            "Reason": "Structured asset fields provide stronger non-income affordability evidence.",
            "Action": "Use as supporting evidence in review.",
        },
        {
            "Area": "Model ranking",
            "Status": STATUS_CAUTION,
            "Reason": "Holdout performance supports prioritization, but thresholds are policy assumptions.",
            "Action": "Use for routing and ranking with QA guardrails.",
        },
        {
            "Area": "Aid percentage model",
            "Status": STATUS_CAUTION,
            "Reason": "Useful for conservative routing under uncertainty controls.",
            "Action": "Keep human review for uncertain or fairness-risk cases.",
        },
        {
            "Area": "Fairness guardrails",
            "Status": STATUS_AUDIT,
            "Reason": "Subgroup results can be small or unstable, so risk must override automation.",
            "Action": "Monitor, tighten, or force manual review depending on trigger.",
        },
        {
            "Area": "Automation readiness",
            "Status": STATUS_HIGH_RISK,
            "Reason": "Safe automation is intentionally narrow and blocked by QA/fairness risk.",
            "Action": "Automate only low-risk routing recommendations.",
        },
        {
            "Area": "Production readiness",
            "Status": STATUS_NOT_PRODUCTION,
            "Reason": "The system is thesis-ready decision support, not an approved eligibility engine.",
            "Action": "Improve data collection and governance before production claims.",
        },
    ]
)
render_table_card(
    title="Trust Status Grid",
    dataframe=trust_grid,
    takeaway="The platform is strongest when it separates evidence, risk, and action instead of collapsing them into one score.",
    caption="Statuses are meant to guide governance conversations; they are not final FAID policy decisions.",
    status=STATUS_CAUTION,
)

render_decision_grid(
    [
        ("Use Now", "QA triage, evidence summaries, model ranking, and committee prioritization.", STATUS_RELIABLE),
        ("Use With Caution", "Income, affordability proxy, and percentage recommendations under guardrails.", STATUS_CAUTION),
        ("Audit Only", "Small subgroup fairness reads, free-text narratives, and weak-coverage fields.", STATUS_AUDIT),
        ("Do Not Automate", "Final FAID judgment, severe QA cases, and fairness-risk cases.", STATUS_HIGH_RISK),
    ]
)

st.subheader("Executive Answers")
render_takeaway_box(
    "The dataset contains cleaned financial-aid application records, QA flags, structured child records, and saved model evidence.",
    status=STATUS_RELIABLE,
    label="What is the dataset?",
)
render_takeaway_box(
    "Outcome labels, household context, and structured QA fields are the strongest evidence areas.",
    status=STATUS_RELIABLE,
    label="What is reliable?",
)
render_takeaway_box(
    "Parent income, loan balance, school gaps, and raw free-text affordability details require caution.",
    status=STATUS_CAUTION,
    label="What is risky?",
)
render_takeaway_box(
    "The eligibility model supports ranking and governed routing, while the percentage model supports conservative automation only under guardrails.",
    status=STATUS_CAUTION,
    label="What does the model support?",
)
render_takeaway_box(
    "Next step: run the Decision Engine workflow, keep high-risk cases in committee review, and improve weak affordability fields before any production automation.",
    status=STATUS_AUDIT,
    label="What should FAID do next?",
)

with st.expander("Reliability details"):
    st.dataframe(
        feature_reliability.assign(
            coverage=lambda frame: frame["coverage"].map(lambda value: format_pct(value)),
            issue_rate=lambda frame: frame["issue_rate"].map(lambda value: format_pct(value)),
        ),
        width="stretch",
        hide_index=True,
    )

with st.expander("Mode impact"):
    st.write(
        f"High-confidence mode keeps {int(high_confidence_mask(dataframe).sum()):,} "
        f"of {len(dataframe):,} records. It excludes high-severity, review-required, missing-school, "
        "or low-quality rows from headline decision reads."
    )
    if automation_rate is not None:
        st.write(f"Saved percentage-model policy automation rate: {format_pct(float(automation_rate))}.")
