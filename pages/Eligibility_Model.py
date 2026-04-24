from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from utils.chart_components import (
    build_grouped_feature_importance,
    make_feature_importance_chart,
    make_grouped_importance_chart,
    make_threshold_tradeoff_curve,
    render_chart_card,
    render_table_card,
)
from utils.data_loader import load_csv, load_json, load_optional_csv
from utils.source_paths import (
    DEFAULT_ELIGIBILITY_DECISION_PATH,
    DEFAULT_MODEL_METADATA_PATH,
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


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def build_calibration_frame(decision_frame: pd.DataFrame) -> pd.DataFrame:
    required = {"actual_target", "predicted_probability_awarded"}
    if decision_frame.empty or not required.issubset(decision_frame.columns):
        return pd.DataFrame(
            columns=[
                "mean_predicted_probability",
                "observed_award_rate",
                "cases",
                "bin_label",
            ]
        )

    frame = decision_frame.copy()
    frame["actual_target"] = pd.to_numeric(frame["actual_target"], errors="coerce")
    frame["predicted_probability_awarded"] = pd.to_numeric(
        frame["predicted_probability_awarded"],
        errors="coerce",
    )
    frame = frame.dropna(subset=["actual_target", "predicted_probability_awarded"])
    if frame.empty:
        return pd.DataFrame()

    bin_count = min(10, max(int(frame["predicted_probability_awarded"].nunique()), 3))
    frame["probability_bin"] = pd.qcut(
        frame["predicted_probability_awarded"],
        q=bin_count,
        duplicates="drop",
    )
    calibration = (
        frame.groupby("probability_bin", observed=False)
        .agg(
            cases=("actual_target", "size"),
            mean_predicted_probability=("predicted_probability_awarded", "mean"),
            observed_award_rate=("actual_target", "mean"),
        )
        .reset_index()
    )
    calibration["bin_label"] = calibration["probability_bin"].astype("string")
    return calibration


def build_policy_sensitivity_frame(metadata_payload: dict[str, object]) -> pd.DataFrame:
    normal_workflow = _as_dict(metadata_payload.get("normal_workflow"))
    frame = pd.DataFrame(normal_workflow.get("policy_weight_sensitivity", []))
    if frame.empty:
        return frame
    if "threshold" not in frame.columns and "selected_threshold" in frame.columns:
        frame["threshold"] = frame["selected_threshold"]
    for column in [
        "threshold",
        "selected_threshold",
        "missed_deserving_students",
        "extra_review_burden",
        "students_shortlisted",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("threshold", ignore_index=True)


def build_boundary_scenarios_frame(metadata_payload: dict[str, object]) -> pd.DataFrame:
    normal_workflow = _as_dict(metadata_payload.get("normal_workflow"))
    frame = pd.DataFrame(normal_workflow.get("decision_boundary_scenarios", []))
    if frame.empty:
        return frame
    for column in [
        "approve_cases",
        "reject_cases",
        "review_cases",
        "approve_threshold",
        "reject_threshold",
        "harmful_auto_approves",
        "harmful_auto_rejects",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_action_distribution(decision_frame: pd.DataFrame) -> pd.DataFrame:
    if decision_frame.empty or "recommended_action" not in decision_frame.columns:
        return pd.DataFrame(columns=["action", "cases", "share_pct"])

    label_map = {
        "approve": "Auto approve",
        "reject": "Auto reject",
        "review": "Committee review",
    }
    counts = (
        decision_frame["recommended_action"]
        .astype("string")
        .fillna("unknown")
        .map(lambda value: label_map.get(str(value), str(value).replace("_", " ").title()))
        .value_counts(dropna=False)
        .rename_axis("action")
        .reset_index(name="cases")
    )
    counts["share_pct"] = counts["cases"] / max(int(counts["cases"].sum()), 1) * 100
    return counts


def build_impact_highlights(
    metadata_payload: dict[str, object],
    sensitivity_frame: pd.DataFrame,
) -> pd.DataFrame:
    normal_workflow = _as_dict(metadata_payload.get("normal_workflow"))
    impact_summary = _as_dict(normal_workflow.get("business_impact_summary"))
    selected_threshold = float(
        _as_dict(normal_workflow.get("test_summary")).get("selected_threshold", 0.0)
    )

    reference_threshold = 0.50
    reference_row = pd.Series(dtype="float64")
    selected_row = pd.Series(dtype="float64")
    if not sensitivity_frame.empty and "threshold" in sensitivity_frame.columns:
        reference_row = sensitivity_frame.iloc[
            (sensitivity_frame["threshold"] - reference_threshold).abs().argmin()
        ]
        selected_row = sensitivity_frame.iloc[
            (sensitivity_frame["threshold"] - selected_threshold).abs().argmin()
        ]

    highlights: list[dict[str, object]] = []
    if not reference_row.empty and not selected_row.empty:
        highlights.append(
            {
                "impact": "Missed deserving students",
                "baseline": int(reference_row.get("missed_deserving_students", 0)),
                "with_governed_policy": int(selected_row.get("missed_deserving_students", 0)),
                "improvement": int(reference_row.get("missed_deserving_students", 0))
                - int(selected_row.get("missed_deserving_students", 0)),
            }
        )
    if impact_summary:
        highlights.extend(
            [
                {
                    "impact": "Manual reviews",
                    "baseline": int(impact_summary.get("manual_review_baseline_cases", 0)),
                    "with_governed_policy": int(impact_summary.get("review_cases_after_model", 0)),
                    "improvement": int(impact_summary.get("manual_review_baseline_cases", 0))
                    - int(impact_summary.get("review_cases_after_model", 0)),
                },
                {
                    "impact": "Safe automation",
                    "baseline": 0,
                    "with_governed_policy": int(impact_summary.get("automated_cases", 0)),
                    "improvement": int(impact_summary.get("automated_cases", 0)),
                },
            ]
        )

    return pd.DataFrame(highlights)


def build_saved_driver_frame(metadata_payload: dict[str, object]) -> pd.DataFrame:
    normal_workflow = _as_dict(metadata_payload.get("normal_workflow"))
    feature_insights = _as_dict(normal_workflow.get("feature_insights"))
    driver_frame = pd.DataFrame(feature_insights.get("top_features", []))
    if driver_frame.empty:
        return driver_frame

    driver_frame["importance"] = pd.to_numeric(driver_frame["importance"], errors="coerce")
    driver_frame["feature"] = driver_frame["feature"].astype("string")
    return driver_frame.dropna(subset=["importance", "feature"]).sort_values(
        "importance",
        ascending=False,
        ignore_index=True,
    )


def build_boundary_stack_frame(boundary_frame: pd.DataFrame) -> pd.DataFrame:
    if boundary_frame.empty:
        return pd.DataFrame(columns=["scenario", "decision_type", "cases"])

    melted = boundary_frame.melt(
        id_vars=["scenario"],
        value_vars=["approve_cases", "reject_cases", "review_cases"],
        var_name="decision_type",
        value_name="cases",
    )
    label_map = {
        "approve_cases": "Approve",
        "reject_cases": "Reject",
        "review_cases": "Review",
    }
    melted["decision_type"] = melted["decision_type"].map(label_map)
    return melted


def build_case_examples(decision_frame: pd.DataFrame) -> pd.DataFrame:
    if decision_frame.empty:
        return pd.DataFrame()

    columns = [
        "parsed_level",
        "parsed_school",
        "predicted_probability_awarded",
        "confidence_score",
        "recommended_action",
        "action_reason",
        "case_explanation",
    ]
    available_columns = [column for column in columns if column in decision_frame.columns]
    return decision_frame[available_columns].head(12).copy()


apply_design_system()

if not DEFAULT_MODEL_METADATA_PATH.exists():
    st.error(f"Eligibility-model metadata not found: {DEFAULT_MODEL_METADATA_PATH}")
else:
    metadata_payload = load_json(DEFAULT_MODEL_METADATA_PATH)
    decision_frame = load_optional_csv(DEFAULT_ELIGIBILITY_DECISION_PATH)

    normal_workflow = _as_dict(metadata_payload.get("normal_workflow"))
    test_summary = _as_dict(normal_workflow.get("test_summary"))
    probability_metrics = _as_dict(test_summary.get("probability_metrics"))
    calibration_summary = _as_dict(normal_workflow.get("calibration_summary"))
    business_impact_summary = _as_dict(normal_workflow.get("business_impact_summary"))
    decision_summary = _as_dict(normal_workflow.get("decision_summary"))
    threshold_rationale = str(normal_workflow.get("threshold_policy_rationale", ""))
    feature_insights = _as_dict(normal_workflow.get("feature_insights"))

    selected_threshold = float(test_summary.get("selected_threshold", 0.0))
    selected_candidate = str(normal_workflow.get("selected_candidate", "Eligibility model"))
    automation_share = float(business_impact_summary.get("automated_share", 0.0))
    review_reduction = float(business_impact_summary.get("review_reduction_vs_all_manual", 0.0))
    brier_score = float(probability_metrics.get("brier_score", 0.0))
    avg_precision = float(probability_metrics.get("average_precision", 0.0))
    fairness_risk_count = int(decision_summary.get("fairness_risk_count", 0))

    calibration_frame = build_calibration_frame(decision_frame)
    policy_sensitivity_frame = build_policy_sensitivity_frame(metadata_payload)
    boundary_frame = build_boundary_scenarios_frame(metadata_payload)
    impact_highlights = build_impact_highlights(metadata_payload, policy_sensitivity_frame)
    saved_driver_frame = build_saved_driver_frame(metadata_payload)
    action_distribution = build_action_distribution(decision_frame)
    case_examples = build_case_examples(decision_frame)

    render_decision_journey("Model")
    render_page_header(
        title="Eligibility Model",
        description=(
            "Notebook-aligned diagnostics for the refreshed eligibility model, including "
            "calibration, threshold sensitivity, governed decision boundaries, and saved drivers."
        ),
        takeaway=(
            f"{selected_candidate} uses a selected threshold of {selected_threshold:.3f} and currently "
            f"automates {automation_share:.1%} of holdout cases under the governed policy."
        ),
        kicker="Model Diagnostics Workspace",
        pills=[
            (selected_candidate.replace("_", " "), "primary"),
            (f"{selected_threshold:.3f} selected threshold", "warning"),
            (f"{automation_share:.1%} automated", "secondary"),
            (f"{fairness_risk_count} fairness-risk cases", "danger"),
        ],
    )
    render_kpi_row(
        [
            {
                "label": "Brier Score",
                "value": f"{brier_score:.3f}",
                "note": "Lower is better for probability quality on the saved holdout split.",
                "tone": "primary",
            },
            {
                "label": "Avg Precision",
                "value": f"{avg_precision:.3f}",
                "note": "Precision-recall summary for the saved holdout probabilities.",
                "tone": "secondary",
            },
            {
                "label": "Review Reduction",
                "value": f"{review_reduction:.1%}",
                "note": "Manual-review reduction versus an all-manual process.",
                "tone": "warning",
            },
            {
                "label": "Auto Decisions",
                "value": str(int(business_impact_summary.get("automated_cases", 0))),
                "note": "Governed auto-approve plus auto-reject cases on the saved holdout.",
                "tone": "danger",
            },
        ]
    )
    render_insight_action_panel(
        insight=(
            "This page mirrors the notebook's core argument: probability quality matters because "
            "the model is used for routing, not only ranking."
        ),
        implication=(
            "Use calibration and threshold sensitivity together. A good classifier is not enough if "
            "the chosen threshold misses deserving students or pushes too many cases into review."
        ),
    )

    summary_tab, calibration_tab, policy_tab, drivers_tab = st.tabs(
        ["Overview", "Calibration", "Threshold Policy", "Drivers"]
    )

    with summary_tab:
        st.subheader("Governed Holdout Impact")
        if impact_highlights.empty:
            st.info("Impact highlights are not available in the current artifact set.")
        else:
            render_table_card(
                title="Governed Holdout Impact",
                dataframe=impact_highlights,
                takeaway="The model value is governed routing, not unrestricted automation.",
                caption="These rows are exported from the notebook artifact and preserve the original calculations.",
                status="Caution",
            )

        overview_cols = st.columns(2)
        with overview_cols[0]:
            if action_distribution.empty:
                st.info("Decision-routing output is not available yet.")
            else:
                routing_chart = (
                    alt.Chart(action_distribution)
                    .mark_arc(innerRadius=55, outerRadius=95)
                    .encode(
                        theta=alt.Theta("cases:Q"),
                        color=alt.Color(
                            "action:N",
                            scale=alt.Scale(range=CHART_NEUTRALS[: len(action_distribution)]),
                            legend=alt.Legend(orient="bottom"),
                        ),
                        tooltip=[
                            alt.Tooltip("action:N", title="Action"),
                            alt.Tooltip("cases:Q", title="Cases"),
                            alt.Tooltip("share_pct:Q", title="Share", format=".1f"),
                        ],
                    )
                    .properties(height=310, title="Approve / Reject / Review Mix")
                )
                render_chart_card(
                    title="Approve / Reject / Review Mix",
                    subtitle="Saved holdout routing under the governed policy.",
                    takeaway="Automation is useful only when review boundaries and fairness constraints stay visible.",
                    chart=routing_chart,
                    caption="The donut is retained for composition, but final action routing is governed by the Decision Engine page.",
                    status="Caution",
                    sample_size=int(action_distribution["cases"].sum()),
                )
        with overview_cols[1]:
            render_chart_card(
                title="Threshold Rule",
                takeaway=threshold_rationale.strip() or "Thresholds are configurable policy assumptions.",
                caption=(
                    f"Calibration mode: {str(calibration_summary.get('selected_mode', 'unknown'))}. "
                    f"Approve threshold: {float(decision_summary.get('approve_threshold', 0.0)):.3f}; "
                    f"reject threshold: {float(decision_summary.get('reject_threshold', 0.0)):.3f}; "
                    f"review margin: {float(decision_summary.get('review_margin', 0.0)):.3f}."
                ),
                status="Caution",
            )

        if not case_examples.empty:
            with st.expander("Notebook-Style Case Examples", expanded=False):
                st.dataframe(case_examples, width="stretch", hide_index=True)

    with calibration_tab:
        st.subheader("Calibration Curve On The Saved Holdout")
        if calibration_frame.empty:
            st.info("Calibration data is not available in the current decision export.")
        else:
            perfect_line = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
            calibration_chart = alt.layer(
                alt.Chart(perfect_line)
                .mark_line(strokeDash=[6, 6], color="#111827")
                .encode(x="x:Q", y="y:Q"),
                alt.Chart(calibration_frame)
                .mark_line(point=True, color=PRIMARY_COLOR, strokeWidth=3)
                .encode(
                    x=alt.X(
                        "mean_predicted_probability:Q",
                        title="Mean predicted probability",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    y=alt.Y(
                        "observed_award_rate:Q",
                        title="Observed award rate",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    size=alt.Size("cases:Q", legend=None),
                    tooltip=[
                        alt.Tooltip("bin_label:N", title="Quantile bin"),
                        alt.Tooltip("cases:Q", title="Cases"),
                        alt.Tooltip(
                            "mean_predicted_probability:Q",
                            title="Mean predicted",
                            format=".3f",
                        ),
                        alt.Tooltip(
                            "observed_award_rate:Q",
                            title="Observed award rate",
                            format=".3f",
                        ),
                    ],
                ),
            ).properties(height=360)
            render_chart_card(
                title="Calibration Curve On The Saved Holdout",
                subtitle=f"Brier score {brier_score:.3f}.",
                takeaway="Calibration checks whether predicted probabilities behave like real review probabilities.",
                chart=calibration_chart,
                caption="The dashed line is perfect calibration. Bubble size reflects bin sample size; small bins should be interpreted cautiously.",
                status="Caution",
                sample_size=int(calibration_frame["cases"].sum()),
            )
            render_table_card(
                title="Calibration Bins",
                dataframe=calibration_frame[
                    ["bin_label", "cases", "mean_predicted_probability", "observed_award_rate"]
                ],
                takeaway="Never interpret a rate without its denominator.",
                status="Caution",
            )

    with policy_tab:
        st.subheader("Threshold Sensitivity And Boundary Scenarios")
        policy_cols = st.columns(2)
        with policy_cols[0]:
            if policy_sensitivity_frame.empty:
                st.info("Threshold-sensitivity output is not available yet.")
            else:
                threshold_rule = pd.DataFrame({"selected_threshold": [selected_threshold]})
                sensitivity_chart = alt.layer(
                    alt.Chart(policy_sensitivity_frame)
                    .mark_line(point=True, color=DANGER_COLOR, strokeWidth=3)
                    .encode(
                        x=alt.X("threshold:Q", title="Threshold"),
                        y=alt.Y(
                            "missed_deserving_students:Q",
                            title="Missed deserving students",
                        ),
                        tooltip=[
                            alt.Tooltip("threshold:Q", title="Threshold", format=".3f"),
                            alt.Tooltip(
                                "missed_deserving_students:Q",
                                title="Missed deserving",
                            ),
                            alt.Tooltip("precision:Q", title="Precision", format=".3f"),
                            alt.Tooltip("recall:Q", title="Recall", format=".3f"),
                        ],
                    ),
                    alt.Chart(threshold_rule)
                    .mark_rule(color="#111827", strokeDash=[6, 6])
                    .encode(x="selected_threshold:Q"),
                ).properties(height=340, title="Threshold sensitivity: missed deserving students")
                render_chart_card(
                    title="Threshold Sensitivity",
                    subtitle="Missed deserving students versus threshold.",
                    takeaway="Selected thresholds should be policy-aware, not metric-only.",
                    chart=sensitivity_chart,
                    caption="The dashed rule is the selected threshold; false-negative protection is the key policy concern.",
                    status="Caution",
                    sample_size=len(policy_sensitivity_frame),
                )

                render_chart_card(
                    title="Threshold Trade-off Curve",
                    subtitle="Precision, recall, F1, and normalized automation pressure.",
                    takeaway="A safer threshold balances review load against false-negative risk.",
                    chart=make_threshold_tradeoff_curve(
                        policy_sensitivity_frame,
                        selected_threshold=selected_threshold,
                    ),
                    caption="Automation rate is normalized within the available threshold export so it can be read on the same scale.",
                    status="Caution",
                    sample_size=len(policy_sensitivity_frame),
                )

        with policy_cols[1]:
            boundary_stack_frame = build_boundary_stack_frame(boundary_frame)
            if boundary_stack_frame.empty:
                st.info("Decision-boundary scenarios are not available yet.")
            else:
                boundary_chart = (
                    alt.Chart(boundary_stack_frame)
                    .mark_bar()
                    .encode(
                        x=alt.X("scenario:N", title=None),
                        y=alt.Y("cases:Q", title="Cases"),
                        color=alt.Color(
                            "decision_type:N",
                            scale=alt.Scale(
                                domain=["Approve", "Reject", "Review"],
                                range=[SECONDARY_COLOR, DANGER_COLOR, ACCENT_COLOR],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("scenario:N", title="Scenario"),
                            alt.Tooltip("decision_type:N", title="Decision"),
                            alt.Tooltip("cases:Q", title="Cases"),
                        ],
                    )
                    .properties(height=340, title="Safe automation vs review under different boundaries")
                )
                render_chart_card(
                    title="Safe Automation vs Review Boundaries",
                    subtitle="Decision-boundary scenarios from the notebook artifact.",
                    takeaway="Boundary choices change review volume and harmful auto-action exposure.",
                    chart=boundary_chart,
                    caption="Approve, reject, and review counts should be reviewed together before adopting a policy.",
                    status="Caution",
                    sample_size=int(boundary_stack_frame["cases"].sum()),
                )

        if not boundary_frame.empty:
            render_table_card(
                title="Decision Boundary Table",
                dataframe=boundary_frame,
                takeaway="Thresholds are configurable assumptions, not immutable model facts.",
                status="Caution",
            )

    with drivers_tab:
        st.subheader("Saved Global Drivers")
        if saved_driver_frame.empty:
            st.info("Saved feature-importance output is not available yet.")
        else:
            render_chart_card(
                title="Raw Top Feature Importance",
                subtitle="Human-readable labels layered over exported raw feature names.",
                takeaway="Feature importance explains model behavior, not causal policy rules.",
                chart=make_feature_importance_chart(saved_driver_frame, top_n=15),
                caption="The chart preserves the exported ranking while making field names easier to read.",
                status="Caution",
                sample_size=len(saved_driver_frame),
            )
            grouped_driver_frame = build_grouped_feature_importance(saved_driver_frame)
            render_chart_card(
                title="Grouped Feature Importance",
                subtitle="Raw drivers rolled into thesis-friendly evidence groups.",
                takeaway="Grouped importance makes the model easier to explain to non-technical reviewers.",
                chart=make_grouped_importance_chart(grouped_driver_frame),
                caption="Grouping is an interpretability layer only; it does not change model training or outputs.",
                status="Reliable",
                sample_size=len(saved_driver_frame),
            )
            if feature_insights:
                st.caption(
                    "This matches the notebook's saved interpretability fallback: when richer explainability "
                    "is unavailable, the dashboard surfaces the exported global importance ranking."
                )
