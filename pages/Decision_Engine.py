from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from utils.chart_components import (
    ACTION_COLORS,
    build_routing_flow_frame,
    make_decision_priority_matrix,
    make_fairness_bubble_chart,
    make_horizontal_bar,
    make_routing_flow_chart,
    render_chart_card,
    render_table_card,
)
from utils.data_loader import load_csv, load_json
from utils.decision_support import (
    STATUS_AUDIT,
    STATUS_CAUTION,
    STATUS_HIGH_RISK,
    STATUS_NOT_PRODUCTION,
    STATUS_RELIABLE,
    build_business_impact_summary,
    build_decision_action_frame,
    build_fairness_rules_frame,
    format_pct,
)
from utils.source_paths import (
    DEFAULT_ELIGIBILITY_DECISION_PATH,
    DEFAULT_PERCENTAGE_FAIRNESS_ACTIONS_PATH,
    DEFAULT_PERCENTAGE_METADATA_PATH,
    DEFAULT_REVIEW_REQUIRED_PATH,
)
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
def load_decision_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    decisions = load_csv(DEFAULT_ELIGIBILITY_DECISION_PATH)
    review = load_csv(DEFAULT_REVIEW_REQUIRED_PATH)
    fairness_actions = load_csv(DEFAULT_PERCENTAGE_FAIRNESS_ACTIONS_PATH)
    percentage_metadata = load_json(DEFAULT_PERCENTAGE_METADATA_PATH)
    return decisions, review, fairness_actions, percentage_metadata


decision_dataframe, review_dataframe, fairness_actions, percentage_metadata = load_decision_inputs()
decision_actions = build_decision_action_frame(decision_dataframe, review_dataframe)

mode = st.radio(
    "Confidence mode",
    ["All data mode", "High-confidence only mode"],
    horizontal=True,
)
if mode == "High-confidence only mode" and not decision_actions.empty:
    decision_actions = decision_actions[decision_actions["data_confidence"].eq(STATUS_RELIABLE)].copy()

impact = build_business_impact_summary(decision_actions)
decision_policy = percentage_metadata.get("decision_policy", {}) if isinstance(percentage_metadata, dict) else {}
decision_summary = percentage_metadata.get("decision_summary", {}) if isinstance(percentage_metadata, dict) else {}

render_decision_journey("Decision")
render_page_header(
    title="Decision Engine",
    description=(
        "A governed routing layer that combines model probability, data confidence, QA severity, "
        "fairness risk, and configurable thresholds into decision-support actions."
    ),
    takeaway=(
        "This page recommends review routing and automation eligibility; it is not a final FAID judgment."
    ),
    kicker="Step 4: Decision",
    pills=[
        ("Model score + QA + fairness", "primary"),
        ("Configurable assumptions", "warning"),
        ("Human review preserved", "secondary"),
    ],
)

render_takeaway_box(
    "Automation is allowed only when model confidence, data confidence, and fairness guardrails all agree.",
    status=STATUS_CAUTION,
)

metric_columns = st.columns(4)
with metric_columns[0]:
    render_metric_card(
        label="Cases Evaluated",
        value=f"{int(impact['cases']):,}",
        interpretation="Saved decision-recommendation cases in the active mode.",
        status=STATUS_RELIABLE,
        sample_size=int(impact["cases"]),
        coverage="decision artifact",
    )
with metric_columns[1]:
    render_metric_card(
        label="Automation Rate",
        value=format_pct(float(impact["automation_rate"])),
        interpretation="Share routed to auto approve or likely deny after guardrails.",
        status=STATUS_CAUTION,
        sample_size=int(impact["automation_cases"]),
        coverage=f"{int(impact['automation_cases']):,} cases",
    )
with metric_columns[2]:
    render_metric_card(
        label="Review Routed",
        value=f"{int(impact['review_cases']):,}",
        interpretation="Cases protected for committee or manual review.",
        status=STATUS_RELIABLE,
        sample_size=int(impact["review_cases"]),
        coverage=format_pct(int(impact["review_cases"]) / max(int(impact["cases"]), 1)),
    )
with metric_columns[3]:
    render_metric_card(
        label="Protected High Risk",
        value=f"{int(impact['protected_cases']):,}",
        interpretation="Cases blocked from automation by QA or fairness risk.",
        status=STATUS_HIGH_RISK if int(impact["protected_cases"]) else STATUS_RELIABLE,
        sample_size=int(impact["protected_cases"]),
        coverage="guardrail-protected",
    )

chart_columns = st.columns(2)
with chart_columns[0]:
    if decision_actions.empty:
        render_chart_card(
            title="Final Action Buckets",
            status=STATUS_CAUTION,
            warning="No decision recommendations are available.",
        )
    else:
        action_counts = (
            decision_actions["final_action"]
            .value_counts()
            .rename_axis("final_action")
            .reset_index(name="cases")
        )
        action_chart = make_horizontal_bar(
            action_counts,
            x="cases",
            y="final_action",
            color_field="final_action",
            color_scale=alt.Scale(domain=list(ACTION_COLORS), range=list(ACTION_COLORS.values())),
            x_title="Cases",
            tooltip=[
                alt.Tooltip("final_action:N", title="Action"),
                alt.Tooltip("cases:Q", title="Cases"),
            ],
            height=310,
        )
        render_chart_card(
            title="Final Action Buckets",
            subtitle="Guarded routing after model score, QA, and fairness checks.",
            takeaway="The decision engine prioritizes safe routing over maximum automation.",
            chart=action_chart,
            caption="Each bucket is a recommended review action, not a final FAID eligibility judgment.",
            status=STATUS_RELIABLE,
            sample_size=len(decision_actions),
        )

with chart_columns[1]:
    render_table_card(
        title="Fairness Guardrail Rules",
        dataframe=build_fairness_rules_frame(),
        takeaway="Fairness risk overrides automation, and small subgroup samples stay monitor-only.",
        caption="These rules are configurable governance assumptions and should be approved before operational use.",
        status=STATUS_AUDIT,
    )

decision_visual_columns = st.columns(2)
with decision_visual_columns[0]:
    render_chart_card(
        title="Review-Priority Matrix",
        subtitle="Model confidence crossed with QA/data risk.",
        takeaway="High-confidence cases still go to review when data risk is high.",
        chart=make_decision_priority_matrix(decision_actions),
        caption="Bubble size is case volume; color is the final recommended action after guardrails.",
        status=STATUS_CAUTION,
        sample_size=len(decision_actions),
    )
with decision_visual_columns[1]:
    flow_frame = build_routing_flow_frame(decision_actions)
    render_chart_card(
        title="Decision Routing Flow",
        subtitle="Altair fallback for Sankey-style routing.",
        takeaway="Guardrails visibly move cases from model evidence into review-oriented action buckets.",
        chart=make_routing_flow_chart(flow_frame),
        caption="Each vertical bar is normalized to show how the portfolio is filtered by confidence, data quality, fairness, and action.",
        status=STATUS_CAUTION,
        sample_size=len(decision_actions),
    )

if not fairness_actions.empty:
    render_chart_card(
        title="Fairness Bubble Chart",
        subtitle="Subgroup sample size vs error gap.",
        takeaway="Larger bubbles with elevated gaps are the strongest governance review candidates.",
        chart=make_fairness_bubble_chart(fairness_actions),
        caption="Recommended action colors distinguish monitor-only, tightened thresholds, and forced manual review.",
        status=STATUS_AUDIT,
        sample_size=len(fairness_actions),
    )

st.subheader("Business Impact Summary")
render_takeaway_box(
    "The business value is operational: fewer routine reviews, safer prioritization, and explicit protection against harmful misses.",
    status=STATUS_RELIABLE,
)
business_columns = st.columns(4)
with business_columns[0]:
    render_metric_card(
        label="Manual Review Reduction",
        value=f"{int(impact['automation_cases']):,}",
        interpretation="Cases that could avoid routine manual review under guardrails.",
        status=STATUS_CAUTION,
        sample_size=int(impact["automation_cases"]),
        coverage=format_pct(float(impact["automation_rate"])),
    )
with business_columns[1]:
    render_metric_card(
        label="Fairness Review Cases",
        value=f"{int(decision_summary.get('fairness_review_cases', 0)):,}",
        interpretation="Percentage-model cases routed away from automation by fairness policy.",
        status=STATUS_AUDIT,
        sample_size=int(decision_summary.get("fairness_review_cases", 0)),
        coverage="saved policy",
    )
with business_columns[2]:
    render_metric_card(
        label="False-Negative Protection",
        value="Primary",
        interpretation="Threshold logic emphasizes avoiding missed need over maximizing automation.",
        status=STATUS_RELIABLE,
        sample_size=int(impact["cases"]),
        coverage="policy assumption",
    )
with business_columns[3]:
    render_metric_card(
        label="Review Workload",
        value=f"{int(impact['review_cases']):,}",
        interpretation="Expected committee queue after guardrails in this artifact.",
        status=STATUS_CAUTION,
        sample_size=int(impact["review_cases"]),
        coverage="operational impact",
    )

st.subheader("Case-Level Decision Support")
render_takeaway_box(
    "Every case shows model score, data confidence, risk flags, final action, and the reason for routing.",
    status=STATUS_CAUTION,
)
if decision_actions.empty:
    st.info("No case-level recommendations are available.")
else:
    action_filter = st.multiselect(
        "Action filter",
        options=sorted(decision_actions["final_action"].dropna().unique().tolist()),
        default=sorted(decision_actions["final_action"].dropna().unique().tolist()),
    )
    filtered = decision_actions[decision_actions["final_action"].isin(action_filter)].copy()
    display = filtered[
        [
            "raw_source_row_number",
            "parsed_school",
            "parsed_level",
            "model_score",
            "data_confidence",
            "qa_max_severity",
            "qa_issue_count",
            "fairness_risk_flag",
            "final_action",
            "decision_support_reason",
        ]
    ].rename(
        columns={
            "raw_source_row_number": "Source Row",
            "parsed_school": "School",
            "parsed_level": "Level",
            "model_score": "Model Score",
            "data_confidence": "Data Confidence",
            "qa_max_severity": "QA Severity",
            "qa_issue_count": "QA Issues",
            "fairness_risk_flag": "Fairness Risk",
            "final_action": "Recommended Action",
            "decision_support_reason": "Reason",
        }
    )
    display["Model Score"] = pd.to_numeric(display["Model Score"], errors="coerce").map(
        lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
    )
    st.dataframe(display, width="stretch", hide_index=True)

with st.expander("Threshold assumptions"):
    st.write(
        "Decision thresholds are configurable assumptions inherited from the saved model artifacts. "
        "They should be approved by FAID before operational use."
    )
    st.json(decision_policy)
    st.write(
        "Current routing labels: Auto approve, Auto reject / likely deny, Committee review, "
        "Deep manual review, and Block automation due to data/fairness risk."
    )
