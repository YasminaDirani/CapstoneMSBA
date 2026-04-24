from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.chart_components import render_decision_grid, render_table_card
from utils.data_loader import load_csv, load_json, prepare_dashboard_dataframe
from utils.decision_support import (
    STATUS_AUDIT,
    STATUS_CAUTION,
    STATUS_HIGH_RISK,
    STATUS_NOT_PRODUCTION,
    STATUS_RELIABLE,
    build_business_impact_summary,
    build_decision_action_frame,
    build_feature_reliability_frame,
    format_pct,
)
from utils.source_paths import (
    DEFAULT_DATA_PATH,
    DEFAULT_ELIGIBILITY_DECISION_PATH,
    DEFAULT_ISSUES_PATH,
    DEFAULT_MODEL_METADATA_PATH,
    DEFAULT_PERCENTAGE_METADATA_PATH,
    DEFAULT_REVIEW_REQUIRED_PATH,
)
from utils.ui import (
    render_decision_journey,
    render_metric_card,
    render_page_header,
    render_takeaway_box,
)


@st.cache_data(show_spinner=False)
def load_final_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    data = prepare_dashboard_dataframe(load_csv(DEFAULT_DATA_PATH))
    review = prepare_dashboard_dataframe(load_csv(DEFAULT_REVIEW_REQUIRED_PATH))
    issues = load_csv(DEFAULT_ISSUES_PATH)
    decisions = load_csv(DEFAULT_ELIGIBILITY_DECISION_PATH)
    eligibility_metadata = load_json(DEFAULT_MODEL_METADATA_PATH)
    percentage_metadata = load_json(DEFAULT_PERCENTAGE_METADATA_PATH)
    return data, review, issues, decisions, eligibility_metadata, percentage_metadata


dataframe, review_dataframe, issues_dataframe, decision_dataframe, eligibility_metadata, percentage_metadata = load_final_inputs()
feature_reliability = build_feature_reliability_frame(dataframe, issues_dataframe)
decision_actions = build_decision_action_frame(decision_dataframe, review_dataframe)
impact = build_business_impact_summary(decision_actions)

normal_workflow = eligibility_metadata.get("normal_workflow", {}) if isinstance(eligibility_metadata, dict) else {}
test_summary = normal_workflow.get("test_summary", {}) if isinstance(normal_workflow, dict) else {}
probability_metrics = test_summary.get("probability_metrics", {}) if isinstance(test_summary, dict) else {}
percentage_summary = percentage_metadata.get("decision_summary", {}) if isinstance(percentage_metadata, dict) else {}

render_decision_journey("Action")
render_page_header(
    title="Final Recommendation",
    description=(
        "A thesis-ready conclusion that separates what FAID can use now from what must remain "
        "review-only, audit-only, or outside automation."
    ),
    takeaway=(
        "This project should be positioned as a governed decision-support system for financial aid review, "
        "not a fully automated eligibility engine."
    ),
    kicker="Step 5: Action",
    pills=[
        ("Use now: prioritization", "primary"),
        ("Review-only: affordability evidence", "warning"),
        ("Do not automate: risky cases", "danger"),
    ],
)

render_takeaway_box(
    "The strongest value is safer prioritization, transparent evidence, and consistency review under messy real-world data.",
    status=STATUS_RELIABLE,
)

metric_columns = st.columns(4)
with metric_columns[0]:
    render_metric_card(
        label="Evidence Base",
        value=f"{len(dataframe):,}",
        interpretation="Cleaned application records packaged for deployment.",
        status=STATUS_RELIABLE,
        sample_size=len(dataframe),
        coverage="portfolio",
    )
with metric_columns[1]:
    render_metric_card(
        label="Model Ranking",
        value=f"AUC {float(probability_metrics.get('roc_auc', 0)):.3f}",
        interpretation="Supports directional ranking and review prioritization.",
        status=STATUS_CAUTION,
        sample_size=len(decision_dataframe),
        coverage="saved holdout",
    )
with metric_columns[2]:
    render_metric_card(
        label="Guarded Automation",
        value=format_pct(float(impact["automation_rate"])),
        interpretation="Automation remains narrow after QA and fairness guardrails.",
        status=STATUS_CAUTION,
        sample_size=int(impact["automation_cases"]),
        coverage=f"{int(impact['cases']):,} artifact cases",
    )
with metric_columns[3]:
    render_metric_card(
        label="Protected Review",
        value=f"{int(impact['protected_cases']):,}",
        interpretation="High-risk cases explicitly protected from auto-routing.",
        status=STATUS_HIGH_RISK if int(impact["protected_cases"]) else STATUS_RELIABLE,
        sample_size=int(impact["protected_cases"]),
        coverage="guardrails",
    )

st.subheader("Final Recommendation Graphic")
render_decision_grid(
    [
        ("Use Now", "Evidence readiness, QA triage, review queue prioritization, and model ranking.", STATUS_RELIABLE),
        ("Use With Caution", "Financial affordability signals and aid-percentage suggestions under human review.", STATUS_CAUTION),
        ("Audit Only", "Fairness small-sample reads, severe data anomalies, and unvalidated proxy features.", STATUS_AUDIT),
        ("Do Not Automate", "Final approval or denial, high-risk QA cases, and fairness-risk cases.", STATUS_HIGH_RISK),
    ]
)

recommendation_columns = st.columns(2)
with recommendation_columns[0]:
    st.subheader("What Can Be Used Now")
    render_takeaway_box(
        "Use descriptive comparisons, QA triage, model ranking, and committee prioritization.",
        status=STATUS_RELIABLE,
    )
    st.write("- Evidence readiness and data-quality summaries.")
    st.write("- Review queue prioritization from QA severity and issue concentration.")
    st.write("- Eligibility probability as a ranking signal, not a final award rule.")
    st.write("- Fairness guardrails that block or tighten automation.")

    st.subheader("What Should Only Support Review")
    render_takeaway_box(
        "Affordability and wealth signals should guide reviewer attention, not independently decide outcomes.",
        status=STATUS_CAUTION,
    )
    st.write("- Parent income, because coverage and currency ambiguity remain material.")
    st.write("- Wealth / affordability proxy combining property, cars, loans, external assistance, and household size.")
    st.write("- Purchasing-power adjustment, because it is a sensitivity experiment rather than a production fix.")

with recommendation_columns[1]:
    st.subheader("What Must Stay Audit-Only")
    render_takeaway_box(
        "Weak-coverage fields and fairness-risk groups should inform governance, not direct automation.",
        status=STATUS_AUDIT,
    )
    st.write("- Small subgroup fairness results.")
    st.write("- Raw free-text hardship narratives.")
    st.write("- Severe QA cases and missing-school records.")
    st.write("- Any case where fairness guardrails trigger manual review.")

    st.subheader("What Should Not Be Automated")
    render_takeaway_box(
        "Do not automate final FAID judgment, borderline cases, severe QA cases, or fairness-risk cases.",
        status=STATUS_NOT_PRODUCTION,
    )
    st.write("- Final eligibility approval or denial without committee oversight.")
    st.write("- Need-based conclusions from income alone.")
    st.write("- Production claims before data collection and subgroup validation improve.")

st.subheader("Data To Improve Next")
render_takeaway_box(
    "Improving a small number of weak fields would raise the ceiling of every model and dashboard page.",
    status=STATUS_CAUTION,
)
weak_features = feature_reliability[feature_reliability["status"].isin([STATUS_AUDIT, STATUS_NOT_PRODUCTION, STATUS_CAUTION])].copy()
weak_features["coverage"] = weak_features["coverage"].map(format_pct)
weak_features["issue_rate"] = weak_features["issue_rate"].map(format_pct)
st.dataframe(
    weak_features[
        [
            "metric",
            "sample_size",
            "coverage",
            "issue_rows",
            "issue_rate",
            "status",
            "reason",
        ]
    ],
    width="stretch",
    hide_index=True,
)

st.subheader("Wealth / Affordability Proxy")
render_takeaway_box(
    "Property and structured asset signals are stronger than raw income alone; the combined proxy is exploratory until validated.",
    status=STATUS_CAUTION,
)
proxy_components = pd.DataFrame(
    [
        {
            "Component": "Income",
            "Fields": "Parent income and income-document confidence",
            "Use": "Useful but incomplete; currency ambiguity requires caution.",
            "Status": "Caution",
        },
        {
            "Component": "Property",
            "Fields": "Estimated property value, rented/planted income",
            "Use": "Structured asset-side capacity signal.",
            "Status": "Reliable",
        },
        {
            "Component": "Cars / Assets",
            "Fields": "Cars count and mortgaged flags",
            "Use": "Contextual asset indicator, not a standalone wealth measure.",
            "Status": "Caution",
        },
        {
            "Component": "Loans",
            "Fields": "Remaining balance, monthly payment, years",
            "Use": "Liability pressure when reported.",
            "Status": STATUS_AUDIT,
        },
        {
            "Component": "External Assistance",
            "Fields": "Financial assistants and estimated annual support",
            "Use": "Offsets affordability burden when structured values exist.",
            "Status": "Caution",
        },
        {
            "Component": "Household Size",
            "Fields": "Dependents and siblings",
            "Use": "Strong contextual burden signal.",
            "Status": "Reliable",
        },
    ]
)
render_table_card(
    title="Wealth / Affordability Proxy Components",
    dataframe=proxy_components,
    takeaway="The proxy is exploratory until validated against improved income, asset, and household-burden collection.",
    caption="Property and structured asset fields strengthen affordability context, but no single component should decide eligibility.",
    status=STATUS_CAUTION,
)

st.subheader("Final Thesis Conclusion")
render_takeaway_box(
    (
        "This project should be positioned as a governed decision-support system for financial aid review, "
        "not a fully automated eligibility engine. The strongest value is safer prioritization, transparent evidence, "
        "and consistency review under messy real-world data."
    ),
    status=STATUS_RELIABLE,
)

with st.expander("Supporting model and business evidence"):
    st.write("Eligibility model probability metrics")
    st.json(probability_metrics)
    st.write("Percentage-model decision summary")
    st.json(percentage_summary)
