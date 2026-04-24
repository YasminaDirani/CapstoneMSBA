from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from utils.chart_components import (
    build_grouped_feature_importance,
    make_fairness_bubble_chart,
    make_feature_importance_chart,
    make_grouped_importance_chart,
    render_chart_card,
    render_table_card,
)
from utils.data_loader import load_csv, load_json, load_optional_csv
from utils.source_paths import (
    DEFAULT_PERCENTAGE_DECISION_RECOMMENDATIONS_PATH,
    DEFAULT_PERCENTAGE_FAIRNESS_ACTIONS_PATH,
    DEFAULT_PERCENTAGE_FAIRNESS_METRICS_PATH,
    DEFAULT_PERCENTAGE_METADATA_PATH,
    DEFAULT_PERCENTAGE_STAGE1_IMPORTANCE_PATH,
    DEFAULT_PERCENTAGE_STAGE2_IMPORTANCE_PATH,
    DEFAULT_PERCENTAGE_TEST_CALIBRATION_BINS_PATH,
    DEFAULT_PERCENTAGE_TEST_PREDICTIONS_PATH,
    DEFAULT_PERCENTAGE_VALIDATION_CALIBRATION_BINS_PATH,
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


def build_action_distribution(decision_frame: pd.DataFrame) -> pd.DataFrame:
    if decision_frame.empty or "recommended_action" not in decision_frame.columns:
        return pd.DataFrame(columns=["action", "applications", "share_pct"])

    label_map = {
        "auto_zero": "Auto zero",
        "auto_award": "Auto award",
        "review": "Manual review",
    }
    counts = (
        decision_frame["recommended_action"]
        .astype("string")
        .fillna("unknown")
        .map(lambda value: label_map.get(str(value), str(value).replace("_", " ").title()))
        .value_counts(dropna=False)
        .rename_axis("action")
        .reset_index(name="applications")
    )
    counts["share_pct"] = counts["applications"] / max(len(decision_frame), 1) * 100
    return counts


def select_probability_bins(
    calibration_frame: pd.DataFrame,
    probability_mode: str,
) -> pd.DataFrame:
    if calibration_frame.empty or "probability_mode" not in calibration_frame.columns:
        return pd.DataFrame()
    return calibration_frame[
        calibration_frame["probability_mode"].astype("string").eq(str(probability_mode))
    ].copy()


def build_policy_frame(policy: dict[str, object]) -> pd.DataFrame:
    rows = [
        ("Auto-zero probability cap", policy.get("auto_zero_probability_max")),
        ("Auto-award probability floor", policy.get("auto_award_probability_min")),
        ("Max prediction interval width", policy.get("max_interval_width")),
        ("Minimum confidence", policy.get("min_confidence")),
    ]
    return pd.DataFrame(rows, columns=["Policy Setting", "Value"])


def build_candidate_frame(metadata_payload: dict[str, object]) -> pd.DataFrame:
    candidate_frame = pd.DataFrame(metadata_payload.get("test_candidate_comparison", []))
    if candidate_frame.empty:
        return candidate_frame

    keep_columns = [
        "model_name",
        "mae",
        "rmse",
        "r2",
        "within_5_pct_points",
        "within_10_pct_points",
        "binary_nonzero_f1",
    ]
    available_columns = [column for column in keep_columns if column in candidate_frame.columns]
    candidate_frame = candidate_frame[available_columns].copy()
    return candidate_frame.sort_values(by="mae", ascending=True, ignore_index=True)


def build_fairness_action_frame(action_frame: pd.DataFrame) -> pd.DataFrame:
    if action_frame.empty or "recommended_action" not in action_frame.columns:
        return pd.DataFrame()

    action_frame = action_frame.copy()
    actionable = action_frame[
        ~action_frame["recommended_action"].isin(["monitor_only", "monitor_only_small_sample"])
    ].copy()
    if actionable.empty:
        return actionable

    keep_columns = [
        "group_column",
        "group_value",
        "sample_size",
        "recommended_action",
        "reason",
        "mae",
        "bias",
    ]
    available_columns = [column for column in keep_columns if column in actionable.columns]
    return actionable[available_columns].sort_values(
        by=["sample_size", "recommended_action"],
        ascending=[False, True],
        ignore_index=True,
    )


def build_fairness_gap_frame(action_frame: pd.DataFrame) -> pd.DataFrame:
    if action_frame.empty or "mae_gap" not in action_frame.columns:
        return pd.DataFrame()

    frame = action_frame.copy()
    frame["sample_size"] = pd.to_numeric(frame["sample_size"], errors="coerce")
    frame["mae_gap"] = pd.to_numeric(frame["mae_gap"], errors="coerce")
    frame = frame.dropna(subset=["sample_size", "mae_gap"])
    if frame.empty:
        return frame

    frame["label"] = (
        frame["group_column"].astype("string")
        + "="
        + frame["group_value"].astype("string")
    )
    frame["abs_mae_gap"] = frame["mae_gap"].abs()
    return frame.sort_values("abs_mae_gap", ascending=False, ignore_index=True).head(8)


def build_importance_chart(
    importance_frame: pd.DataFrame,
    *,
    title: str,
    color: str,
) -> alt.Chart:
    top_frame = importance_frame.head(12).copy()
    return (
        alt.Chart(top_frame)
        .mark_bar(color=color)
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y(
                "feature:N",
                title=None,
                sort=alt.SortField(field="importance", order="descending"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".4f"),
            ],
        )
        .properties(height=360, title=title)
    )


def build_calibration_curve_chart(
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> alt.Chart:
    perfect_line = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    chart = alt.Chart(perfect_line).mark_line(
        strokeDash=[6, 6],
        color="#111827",
    ).encode(x="x:Q", y="y:Q")

    if not validation_frame.empty:
        chart += (
            alt.Chart(validation_frame)
            .mark_line(point=True, color=PRIMARY_COLOR, strokeWidth=3)
            .encode(
                x=alt.X(
                    "mean_predicted_probability:Q",
                    title="Predicted probability",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y(
                    "empirical_nonzero_rate:Q",
                    title="Observed nonzero-aid rate",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=[
                    alt.Tooltip("bin_label:N", title="Bin"),
                    alt.Tooltip("sample_size:Q", title="Sample"),
                    alt.Tooltip(
                        "mean_predicted_probability:Q",
                        title="Mean predicted",
                        format=".3f",
                    ),
                    alt.Tooltip(
                        "empirical_nonzero_rate:Q",
                        title="Observed rate",
                        format=".3f",
                    ),
                ],
            )
        )

    if not test_frame.empty:
        chart += (
            alt.Chart(test_frame)
            .mark_line(point=alt.OverlayMarkDef(shape="square"), color=SECONDARY_COLOR, strokeWidth=3)
            .encode(
                x="mean_predicted_probability:Q",
                y="empirical_nonzero_rate:Q",
                tooltip=[
                    alt.Tooltip("bin_label:N", title="Bin"),
                    alt.Tooltip("sample_size:Q", title="Sample"),
                    alt.Tooltip(
                        "mean_predicted_probability:Q",
                        title="Mean predicted",
                        format=".3f",
                    ),
                    alt.Tooltip(
                        "empirical_nonzero_rate:Q",
                        title="Observed rate",
                        format=".3f",
                    ),
                ],
            )
        )

    return chart.properties(height=320, title="Calibration Plot")


def build_error_distribution_chart(prediction_frame: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(prediction_frame)
        .mark_bar(color=PRIMARY_COLOR, opacity=0.85)
        .encode(
            x=alt.X("absolute_error:Q", bin=alt.Bin(maxbins=20), title="Absolute error (percentage points)"),
            y=alt.Y("count():Q", title="Cases"),
            tooltip=[alt.Tooltip("count():Q", title="Cases")],
        )
        .properties(height=320, title="Error Distribution")
    )


def build_fairness_gap_chart(fairness_gap_frame: pd.DataFrame) -> alt.Chart:
    color_scale = alt.Scale(
        domain=[
            "force_manual_review",
            "tighten_automation_thresholds",
            "monitor_only",
            "monitor_only_small_sample",
        ],
        range=["#E45756", "#F58518", "#72B7B2", "#B279A2"],
    )
    return (
        alt.Chart(fairness_gap_frame)
        .mark_bar()
        .encode(
            x=alt.X("mae_gap:Q", title="MAE gap vs overall"),
            y=alt.Y(
                "label:N",
                title=None,
                sort=alt.SortField(field="abs_mae_gap", order="descending"),
            ),
            color=alt.Color("recommended_action:N", scale=color_scale, title="Intervention"),
            tooltip=[
                alt.Tooltip("label:N", title="Group"),
                alt.Tooltip("sample_size:Q", title="Sample"),
                alt.Tooltip("mae_gap:Q", title="MAE gap", format=".2f"),
                alt.Tooltip("recommended_action:N", title="Action"),
            ],
        )
        .properties(height=320, title="Fairness Gap Chart")
    )


def build_automation_chart(action_distribution: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(action_distribution)
        .mark_arc(innerRadius=55, outerRadius=95)
        .encode(
            theta=alt.Theta("applications:Q"),
            color=alt.Color(
                "action:N",
                scale=alt.Scale(range=CHART_NEUTRALS[: len(action_distribution)]),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("action:N", title="Action"),
                alt.Tooltip("applications:Q", title="Applications"),
                alt.Tooltip("share_pct:Q", title="Share", format=".1f"),
            ],
        )
        .properties(height=320, title="Automation vs Review")
    )


apply_design_system()

if not DEFAULT_PERCENTAGE_METADATA_PATH.exists():
    st.error(f"Percentage-model metadata not found: {DEFAULT_PERCENTAGE_METADATA_PATH}")
else:
    metadata_payload = load_json(DEFAULT_PERCENTAGE_METADATA_PATH)
    decision_frame = load_optional_csv(DEFAULT_PERCENTAGE_DECISION_RECOMMENDATIONS_PATH)
    test_prediction_frame = load_optional_csv(DEFAULT_PERCENTAGE_TEST_PREDICTIONS_PATH)
    validation_calibration_bins = load_optional_csv(
        DEFAULT_PERCENTAGE_VALIDATION_CALIBRATION_BINS_PATH
    )
    test_calibration_bins = load_optional_csv(DEFAULT_PERCENTAGE_TEST_CALIBRATION_BINS_PATH)
    fairness_metrics_frame = load_optional_csv(DEFAULT_PERCENTAGE_FAIRNESS_METRICS_PATH)
    fairness_actions_frame = load_optional_csv(DEFAULT_PERCENTAGE_FAIRNESS_ACTIONS_PATH)
    stage1_importance_frame = load_optional_csv(DEFAULT_PERCENTAGE_STAGE1_IMPORTANCE_PATH)
    stage2_importance_frame = load_optional_csv(DEFAULT_PERCENTAGE_STAGE2_IMPORTANCE_PATH)

    problem_framing = _as_dict(metadata_payload.get("problem_framing"))
    final_test_metrics = _as_dict(metadata_payload.get("final_test_metrics"))
    decision_policy = _as_dict(metadata_payload.get("decision_policy"))
    decision_summary = _as_dict(metadata_payload.get("decision_summary"))
    fairness_summary = _as_dict(metadata_payload.get("fairness_test_summary"))
    calibration_summary = _as_dict(metadata_payload.get("calibration_test_summary"))
    temporal_validation = _as_dict(metadata_payload.get("temporal_validation"))

    automation_rate = float(decision_summary.get("automation_rate", 0.0))
    within_10 = float(final_test_metrics.get("within_10_pct_points", 0.0))
    mae = float(final_test_metrics.get("mae", 0.0))
    flagged_groups = int(fairness_summary.get("flagged_group_count", 0))
    tightened_groups = int(fairness_summary.get("tightened_group_count", 0))
    selected_model_name = str(problem_framing.get("selected_model_name", "Two-stage model"))
    selected_mode = str(calibration_summary.get("selected_mode", "unknown"))
    validation_mode = str(
        _as_dict(metadata_payload.get("calibration_validation_summary")).get(
            "selected_mode",
            selected_mode,
        )
    )
    validation_curve_frame = select_probability_bins(validation_calibration_bins, validation_mode)
    test_curve_frame = select_probability_bins(test_calibration_bins, selected_mode)
    notebook_fairness_gap_frame = build_fairness_gap_frame(fairness_actions_frame)
    notebook_action_distribution = build_action_distribution(decision_frame)

    render_decision_journey("Model")
    render_page_header(
        title="Aid Percentage Model",
        description=(
            "Tracks the refreshed two-stage award-percentage model, its routing policy, "
            "and the fairness guardrails exported from the MSBA artifact pipeline."
        ),
        takeaway=(
            f"{selected_model_name} keeps overall test MAE at {mae:.1f} percentage points "
            f"while limiting automation to {automation_rate:.1%} of cases."
        ),
        kicker="Model Portfolio Workspace",
        pills=[
            (selected_model_name.replace("_", " "), "primary"),
            (f"{automation_rate:.1%} automation rate", "warning"),
            (f"{flagged_groups} flagged fairness groups", "danger"),
            ("Temporal validation available" if temporal_validation.get("available") else "No temporal validation", "primary"),
        ],
    )
    render_kpi_row(
        [
            {
                "label": "Test MAE",
                "value": f"{mae:.1f} pts",
                "note": "Average absolute error on the held-out percentage test split.",
                "tone": "primary",
            },
            {
                "label": "Within 10 Points",
                "value": f"{within_10:.1%}",
                "note": "Share of held-out predictions that land within 10 percentage points.",
                "tone": "secondary",
            },
            {
                "label": "Automation Rate",
                "value": f"{automation_rate:.1%}",
                "note": (
                    f"{int(decision_summary.get('auto_zero_cases', 0))} auto-zero and "
                    f"{int(decision_summary.get('auto_award_cases', 0))} auto-award cases."
                ),
                "tone": "warning",
            },
            {
                "label": "Fairness Interventions",
                "value": f"{flagged_groups + tightened_groups}",
                "note": (
                    f"{flagged_groups} forced-review groups and "
                    f"{tightened_groups} tightened-threshold groups on test monitoring."
                ),
                "tone": "danger",
            },
        ]
    )
    render_insight_action_panel(
        insight=(
            "The target is zero-inflated, so the selected model first predicts whether any aid is likely, "
            "then estimates the positive award size instead of forcing one regression rule across all cases."
        ),
        implication=(
            "Use the decision-routing layer for queue design and manual-review prioritization, not as a blanket automation rule."
        ),
    )

    summary_tab, diagnostics_tab, routing_tab, fairness_tab, drivers_tab = st.tabs(
        ["Performance", "Notebook Diagnostics", "Routing", "Fairness", "Drivers"]
    )

    with summary_tab:
        st.subheader("Model Summary")
        st.write(
            str(problem_framing.get("selection_rationale", "Selection rationale is unavailable."))
        )
        st.caption(
            f"Calibration mode kept for deployment: {selected_mode}. "
            f"{str(calibration_summary.get('selection_reason', ''))}"
        )

        candidate_frame = build_candidate_frame(metadata_payload)
        if candidate_frame.empty:
            st.info("Candidate-comparison output is not available yet.")
        else:
            comparison_chart = (
                alt.Chart(candidate_frame)
                .mark_bar(color=PRIMARY_COLOR)
                .encode(
                    x=alt.X("mae:Q", title="Test MAE"),
                    y=alt.Y("model_name:N", title=None, sort=alt.SortField(field="mae", order="ascending")),
                    tooltip=[
                        alt.Tooltip("model_name:N", title="Model"),
                        alt.Tooltip("mae:Q", title="MAE", format=".2f"),
                        alt.Tooltip("rmse:Q", title="RMSE", format=".2f"),
                        alt.Tooltip("r2:Q", title="R-squared", format=".3f"),
                        alt.Tooltip("within_10_pct_points:Q", title="Within 10 pts", format=".1%"),
                    ],
                )
                .properties(height=320)
            )
            render_chart_card(
                title="Candidate Model Comparison",
                subtitle="Sorted by lower test MAE.",
                takeaway="Model selection should balance error, calibration, and routing safety.",
                chart=comparison_chart,
                caption="Lower MAE is better, but routing policy and fairness guardrails decide operational use.",
                status="Caution",
                sample_size=len(candidate_frame),
            )
            render_table_card(
                title="Candidate Metrics Table",
                dataframe=candidate_frame,
                takeaway="The table preserves the notebook-exported metric details.",
                status="Reliable",
            )

        if temporal_validation:
            temporal_metrics = _as_dict(temporal_validation.get("final_test_metrics"))
            st.markdown("**Temporal Validation**")
            if temporal_validation.get("available"):
                st.caption(str(temporal_validation.get("note", "")))
                temporal_cols = st.columns(3)
                with temporal_cols[0]:
                    st.metric(
                        "Temporal MAE",
                        f"{float(temporal_metrics.get('mae', 0.0)):.1f} pts",
                    )
                with temporal_cols[1]:
                    st.metric(
                        "Temporal Within 10",
                        f"{float(temporal_metrics.get('within_10_pct_points', 0.0)):.1%}",
                    )
                with temporal_cols[2]:
                    st.metric(
                        "Temporal Binary F1",
                        f"{float(temporal_metrics.get('binary_nonzero_f1', 0.0)):.3f}",
                    )
            else:
                st.info("Temporal validation is not available in the current artifact set.")

        business_summary = metadata_payload.get("business_summary")
        if isinstance(business_summary, str) and business_summary.strip():
            with st.expander("Business Summary", expanded=False):
                st.text(business_summary.strip())

    with diagnostics_tab:
        st.subheader("Notebook Diagnostic Panel")
        st.caption(
            "This reproduces the notebook's core 2x2 visual panel: calibration, holdout error spread, fairness gap severity, and automation composition."
        )

        top_row = st.columns(2)
        with top_row[0]:
            if validation_curve_frame.empty and test_curve_frame.empty:
                st.info("Calibration-bin exports are not available yet.")
            else:
                render_chart_card(
                    title="Calibration Curve",
                    subtitle="Validation and test calibration bins.",
                    takeaway="Calibration matters because award-percentage predictions become review-routing evidence.",
                    chart=build_calibration_curve_chart(validation_curve_frame, test_curve_frame),
                    caption="The diagonal is perfect calibration; small bins should be read cautiously.",
                    status="Caution",
                    sample_size=int(len(validation_curve_frame) + len(test_curve_frame)),
                )
        with top_row[1]:
            if test_prediction_frame.empty or "absolute_error" not in test_prediction_frame.columns:
                st.info("Holdout prediction errors are not available yet.")
            else:
                render_chart_card(
                    title="Holdout Error Distribution",
                    subtitle="Absolute prediction error on test cases.",
                    takeaway="Large errors should feed manual review and uncertainty messaging.",
                    chart=build_error_distribution_chart(test_prediction_frame),
                    caption="This chart shows model error spread, not reviewer error or causal policy effects.",
                    status="Caution",
                    sample_size=len(test_prediction_frame),
                )

        bottom_row = st.columns(2)
        with bottom_row[0]:
            if notebook_fairness_gap_frame.empty:
                st.info("Fairness-gap chart data is not available yet.")
            else:
                render_chart_card(
                    title="Fairness Gap Severity",
                    subtitle="Largest subgroup MAE gaps from the notebook action export.",
                    takeaway="Elevated subgroup gaps should trigger tighter automation thresholds or manual review.",
                    chart=build_fairness_gap_chart(notebook_fairness_gap_frame),
                    caption="This is a monitoring and guardrail view, not a final fairness certification.",
                    status="Audit",
                    sample_size=len(notebook_fairness_gap_frame),
                )
        with bottom_row[1]:
            if notebook_action_distribution.empty:
                st.info("Automation-composition data is not available yet.")
            else:
                render_chart_card(
                    title="Automation vs Review",
                    subtitle="Routing composition from saved recommendations.",
                    takeaway="Automation remains intentionally limited under uncertainty and fairness controls.",
                    chart=build_automation_chart(notebook_action_distribution),
                    caption="The composition chart shows operating posture; final judgment remains human-in-the-loop.",
                    status="Caution",
                    sample_size=int(notebook_action_distribution["applications"].sum()),
                )

    with routing_tab:
        st.subheader("Decision Routing")
        policy_frame = build_policy_frame(decision_policy)
        render_table_card(
            title="Decision Policy",
            dataframe=policy_frame,
            takeaway="Policy settings are configurable assumptions, not hidden model facts.",
            status="Caution",
        )

        action_distribution = build_action_distribution(decision_frame)
        if action_distribution.empty:
            st.info("Decision-routing exports are not available yet.")
        else:
            routing_chart = (
                alt.Chart(action_distribution)
                .mark_bar()
                .encode(
                    x=alt.X("action:N", title=None, sort="-y"),
                    y=alt.Y("applications:Q", title="Applications"),
                    color=alt.Color(
                        "action:N",
                        title="Action",
                        scale=alt.Scale(
                            domain=action_distribution["action"].tolist(),
                            range=CHART_NEUTRALS[: len(action_distribution)],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("action:N", title="Action"),
                        alt.Tooltip("applications:Q", title="Applications"),
                        alt.Tooltip("share_pct:Q", title="Share", format=".1f"),
                    ],
                )
                .properties(height=320)
            )
            render_chart_card(
                title="Decision Routing Distribution",
                subtitle="Recommended actions for the aid-percentage model.",
                takeaway="Review routing protects uncertain and fairness-risk cases from automated action.",
                chart=routing_chart,
                caption="Use this to discuss workload implications rather than final award decisions.",
                status="Caution",
                sample_size=int(action_distribution["applications"].sum()),
            )
            render_table_card(
                title="Routing Examples",
                dataframe=decision_frame[
                    [
                        column
                        for column in [
                            "parsed_level",
                            "inferred_application_track",
                            "actual_need_pct",
                            "predicted_need_pct",
                            "predicted_nonzero_probability",
                            "prediction_interval_width",
                            "recommended_action",
                            "action_reason",
                        ]
                        if column in decision_frame.columns
                    ]
                ].head(25),
                takeaway="Case-level examples keep routing explainable and auditable.",
                status="Caution",
            )

    with fairness_tab:
        st.subheader("Fairness Monitoring")
        if fairness_metrics_frame.empty:
            st.info("Fairness metrics are not available yet.")
        else:
            top_metrics = fairness_metrics_frame.sort_values(
                by="mae",
                ascending=False,
                ignore_index=True,
            ).head(15)
            fairness_chart = (
                alt.Chart(top_metrics)
                .mark_circle(size=120, color=ACCENT_COLOR)
                .encode(
                    x=alt.X("sample_size:Q", title="Sample Size"),
                    y=alt.Y("mae:Q", title="MAE"),
                    tooltip=[
                        alt.Tooltip("group_column:N", title="Dimension"),
                        alt.Tooltip("group_value:N", title="Group"),
                        alt.Tooltip("sample_size:Q", title="Sample"),
                        alt.Tooltip("mae:Q", title="MAE", format=".2f"),
                        alt.Tooltip("bias:Q", title="Bias", format=".2f"),
                    ],
                )
                .properties(height=320)
            )
            render_chart_card(
                title="Fairness Monitoring Bubble Chart",
                subtitle="Subgroup sample size vs MAE.",
                takeaway="Large groups with high error should be prioritized for governance review.",
                chart=fairness_chart,
                caption="Bubble charts make sample size visible so small groups are not overinterpreted.",
                status="Audit",
                sample_size=len(top_metrics),
            )

        actionable_fairness = build_fairness_action_frame(fairness_actions_frame)
        if actionable_fairness.empty:
            st.info("No fairness-triggered interventions were exported beyond monitoring-only rows.")
        else:
            render_table_card(
                title="Actionable Fairness Interventions",
                dataframe=actionable_fairness,
                takeaway="Fairness monitoring changes routing by forcing review or tightening automation thresholds.",
                status="Audit",
            )
        if not fairness_actions_frame.empty:
            render_chart_card(
                title="Fairness Guardrail Actions",
                subtitle="Sample size vs subgroup gap using recommended action colors.",
                takeaway="Guardrails are enforceable because risky subgroups change the routing decision.",
                chart=make_fairness_bubble_chart(fairness_actions_frame),
                caption="Monitor-only groups are gray; manual-review and tightened-threshold actions use stronger colors.",
                status="Audit",
                sample_size=len(fairness_actions_frame),
            )

    with drivers_tab:
        st.subheader("Global Drivers")
        driver_cols = st.columns(2)
        with driver_cols[0]:
            if stage1_importance_frame.empty:
                st.info("Stage-1 importance output is not available yet.")
            else:
                render_chart_card(
                    title="Stage 1 Raw Drivers",
                    subtitle="Probability of any aid.",
                    takeaway="Feature importance explains model behavior, not causal policy rules.",
                    chart=make_feature_importance_chart(stage1_importance_frame, top_n=14),
                    caption="The chart keeps raw feature rankings while making names more readable.",
                    status="Caution",
                    sample_size=len(stage1_importance_frame),
                )
                render_chart_card(
                    title="Stage 1 Grouped Drivers",
                    subtitle="Probability drivers by evidence family.",
                    takeaway="Grouped drivers show what pushes eligibility likelihood.",
                    chart=make_grouped_importance_chart(build_grouped_feature_importance(stage1_importance_frame)),
                    status="Reliable",
                    sample_size=len(stage1_importance_frame),
                )
        with driver_cols[1]:
            if stage2_importance_frame.empty:
                st.info("Stage-2 importance output is not available yet.")
            else:
                render_chart_card(
                    title="Stage 2 Raw Drivers",
                    subtitle="Positive award amount.",
                    takeaway="Stage 2 can lean on different evidence than Stage 1.",
                    chart=make_feature_importance_chart(stage2_importance_frame, top_n=14),
                    caption="Compare this with Stage 1 to separate eligibility from amount behavior.",
                    status="Caution",
                    sample_size=len(stage2_importance_frame),
                )
                render_chart_card(
                    title="Stage 2 Grouped Drivers",
                    subtitle="Amount drivers by evidence family.",
                    takeaway="Grouped drivers show what pushes award size once aid is likely.",
                    chart=make_grouped_importance_chart(build_grouped_feature_importance(stage2_importance_frame)),
                    status="Reliable",
                    sample_size=len(stage2_importance_frame),
                )
        st.caption(
            "These charts show saved global importance artifacts from the percentage-model pipeline. "
            "Use them as directional evidence of what the model leans on, not as causal explanations."
        )
