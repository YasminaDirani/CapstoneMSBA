from __future__ import annotations

import altair as alt
import html
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from utils.chart_insights import (
    driver_side_insights,
    feature_importance_insights,
    group_pattern_chart_insights,
)
from utils.data_loader import DEFAULT_DATA_PATH, load_csv
from utils.decision_analytics import (
    GROUP_DIMENSIONS,
    build_inconsistency_review_observations,
    build_review_candidate_tables,
    find_counterintuitive_metrics,
    find_extreme_segments,
    find_similar_profile_review_pairs,
    prepare_decision_analytics_frame,
    summarize_group_patterns,
    summarize_mixed_profile_groups,
    summarize_numeric_drivers,
)
from utils.model_explainability import (
    extract_top_feature_drivers,
    extract_top_feature_importances,
)
from utils.prediction_utils import add_prediction_probabilities
from utils.ui import (
    ACCENT_COLOR,
    BACKGROUND_COLOR,
    BORDER_COLOR,
    CHART_NEUTRALS,
    DANGER_COLOR,
    MUTED_TEXT_COLOR,
    PRIMARY_COLOR,
    PRIMARY_DARK,
    SECONDARY_COLOR,
    TEXT_COLOR,
    add_top_n_flag,
    apply_design_system,
    build_color_condition,
    collapse_small_categories,
    render_insight_action_panel,
    render_kpi_row,
    render_page_header,
)

try:
    from utils.model_loader import DEFAULT_MODEL_PATH, load_model
except Exception:
    DEFAULT_MODEL_PATH = None
    load_model = None


NEGATIVE_COLOR = DANGER_COLOR
WARNING_COLOR = ACCENT_COLOR
SURFACE_COLOR = BACKGROUND_COLOR
TEXT_MUTED = MUTED_TEXT_COLOR
TEXT_DARK = TEXT_COLOR
NEUTRAL_PIE_COLORS = CHART_NEUTRALS
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_METADATA_PATH = (
    ROOT_DIR / "artifacts" / "yasmina_eligibility" / "yasmina_eligibility_metadata.json"
)
FACTOR_PRIORITY_COLORS = {
    "Primary": PRIMARY_COLOR,
    "Secondary": SECONDARY_COLOR,
    "Audit": WARNING_COLOR,
}
FACTOR_THEME_KEYWORDS = {
    "total_parent_income": ("income",),
    "properties_total_estimated_value": ("property", "properties"),
    "dependents_count": ("dependents",),
    "siblings_other_total_tuition": ("sibling", "siblings", "tuition"),
    "loans_total_remaining_balance": ("loan", "balance"),
    "total_siblings": ("sibling", "siblings"),
    "financial_assistants_total_est_annual_amount": ("assistant", "assist", "support"),
}


def _as_dict(value: object) -> dict[str, object]:
    """Return a mapping value or an empty dict when the payload is missing."""
    return value if isinstance(value, dict) else {}


@st.cache_data(show_spinner=False)
def load_json(json_path: str | Path) -> dict[str, object]:
    """Load JSON metadata from disk and cache it between reruns."""
    path = Path(json_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    with path.open("r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    return payload if isinstance(payload, dict) else {}


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

    return None


def prettify_model_feature_label(feature_name: str) -> str:
    """Turn raw feature identifiers into readable short labels."""
    return str(feature_name).replace("_", " ").strip()


def render_chart_insights(lines: list[str], *, title: str = "Insight and Implication") -> None:
    """Render concise insights with implication-forward labels."""
    if not lines:
        return
    if title.lower().startswith("how to read") or title.lower().startswith("what this suggests"):
        title = "Insight and Implication"
    st.markdown(f"**{title}**")
    labels = ["Insight", "Implication", "Focus"]
    for index, line in enumerate(lines):
        label = labels[index] if index < len(labels) else "Implication"
        st.write(f"- **{label}:** {line}")


def apply_page_styling() -> None:
    """Apply the shared project-wide design system."""
    apply_design_system()


def render_explainer_card(
    title: str,
    body: str,
    *,
    bullets: list[str] | None = None,
) -> None:
    """Render a compact explainer instead of a large card."""
    st.markdown(f"**{title}**")
    st.caption(body)
    if bullets:
        for line in bullets:
            st.caption(f"- {line}")


def render_chart_guide(
    *,
    what: str,
    simple_rule: str,
    positive_signal: str,
    negative_signal: str,
    watch_for: str | None = None,
    positive_label: str = "More / Higher",
    negative_label: str = "Less / Lower",
) -> None:
    """Render a compact chart-reading note without overwhelming the page."""
    st.caption(f"Quick read: {simple_rule}")
    if watch_for:
        st.caption(f"Audit focus: {watch_for}")


def render_page_intro_panel(
    *,
    overview: dict[str, object],
    top_stable_factor: str,
    most_uneven_dimension: str,
    strongest_high_segment: str,
    strongest_low_segment: str,
    model_ready: bool,
) -> None:
    """Render a clearer top-of-page overview with guidance and reading rules."""
    award_rate = float(overview.get("award_rate", np.nan))
    hero_pills = [
        f"{int(overview.get('rows_standard', 0)):,} comparable applications",
        (
            f"{award_rate:.1%} historical award rate"
            if pd.notna(award_rate)
            else "Award rate unavailable"
        ),
        f"Top stable factor: {top_stable_factor}",
        f"Most uneven dimension: {most_uneven_dimension}",
    ]
    hero_pills_html = "".join(
        f'<span class="flow-pill">{html.escape(text)}</span>' for text in hero_pills
    )

    review_text = "Consistency Review is the main validation layer for borderline or suspicious cases."
    guide_cards = [
        (
            "Focus clean signals",
            f"Start with {top_stable_factor} and the other stable financial factors before you explain any contextual exceptions.",
        ),
        (
            "Explain the gap",
            f"Use Segments to see why {most_uneven_dimension} and the highest-lift groups move furthest away from baseline.",
        ),
        (
            "Audit the exceptions",
            f"Review {strongest_low_segment} at the low end and {strongest_high_segment} at the high end when the pattern needs a policy explanation.",
        ),
    ]
    guide_cards_html = "".join(
        (
            '<div class="guide-card">'
            f'<p class="guide-card-title">{html.escape(title)}</p>'
            f'<p class="guide-card-body">{html.escape(body)}</p>'
            "</div>"
        )
        for title, body in guide_cards
    )

    legend_items = [
        (PRIMARY_COLOR, "Award-supporting pattern"),
        (NEGATIVE_COLOR, "Lower-award or higher-concern pattern"),
        (WARNING_COLOR, "Important but audit"),
        (MUTED_TEXT_COLOR, "Bigger gap or bigger mark = stronger signal"),
    ]
    legend_html = "".join(
        (
            '<span class="legend-pill">'
            f'<span class="legend-dot" style="background:{color};"></span>'
            f"{html.escape(label)}"
            "</span>"
        )
        for color, label in legend_items
    )

    st.markdown(
        f"""
        <section class="insight-hero">
            <div class="hero-kicker">Decision Review Workspace</div>
            <h2 class="hero-title">Focus the strongest signals, then audit the exceptions</h2>
            <p class="hero-body">
                Use this page to separate clean need-aligned evidence from patterns that still need policy explanation, audit, or case-level review.
            </p>
            <div class="hero-flow">{hero_pills_html}</div>
            <div class="guide-grid">{guide_cards_html}</div>
            <div class="legend-heading">What to focus on</div>
            <div class="legend-strip">{legend_html}</div>
            <div class="page-summary-note">
                Recommended flow: Start Here -> Factors -> Segments -> Consistency.
                {html.escape(review_text)}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def build_executive_takeaway(
    factor_scorecard: pd.DataFrame,
    dimension_summary: pd.DataFrame,
    counterintuitive_metrics: pd.DataFrame,
) -> str:
    """State the strongest whole-page conclusion in one sentence."""
    stable_factors = factor_scorecard[
        (~factor_scorecard["counterintuitive"]) & (factor_scorecard["coverage_pct"] >= 20)
    ].reset_index(drop=True)
    stable_labels = (
        stable_factors["metric"].head(2).tolist()
        if not stable_factors.empty
        else factor_scorecard["metric"].head(2).tolist()
    )
    stable_text = format_label_list(stable_labels, max_items=2)
    top_dimension = (
        str(dimension_summary.iloc[0]["dimension"])
        if not dimension_summary.empty
        else "contextual segmentation"
    )
    audit_metric = (
        "Dependents Count"
        if not counterintuitive_metrics.empty
        and counterintuitive_metrics["metric"].astype(str).eq("Dependents Count").any()
        else str(counterintuitive_metrics.iloc[0]["metric"])
        if not counterintuitive_metrics.empty
        else "the weakest aligned signals"
    )
    return (
        f"Historical decisions are most defensibly aligned with {stable_text}, but {top_dimension} materially shifts outcomes and {audit_metric} is too directionally suspicious to use as a clean rule."
    )


def build_audit_risk_lines(
    counterintuitive_metrics: pd.DataFrame,
    dimension_summary: pd.DataFrame,
    review_summary: dict[str, object] | None,
) -> list[str]:
    """Highlight the clearest audit risks at the top of the page."""
    lines: list[str] = []

    dependents_row = counterintuitive_metrics.loc[
        counterintuitive_metrics["metric"].astype(str) == "Dependents Count"
    ]
    first_counter = (
        dependents_row.iloc[0]
        if not dependents_row.empty
        else counterintuitive_metrics.iloc[0]
        if not counterintuitive_metrics.empty
        else None
    )
    if first_counter is not None:
        lines.append(
            f"{first_counter['metric']} is statistically strong but points the wrong way ({first_counter['awarded_median']:,.0f} awarded vs {first_counter['denied_median']:,.0f} denied), so it should be audited before it influences policy or scoring."
        )

    school_row = dimension_summary.loc[
        dimension_summary["dimension"].astype(str) == "School"
    ]
    top_dimension_row = school_row.iloc[0] if not school_row.empty else (
        dimension_summary.iloc[0] if not dimension_summary.empty else None
    )
    if top_dimension_row is not None:
        lines.append(
            f"{top_dimension_row['dimension']} shows the widest spread in award rates ({top_dimension_row['spread_pp']:.1f} pp), so those differences need an explicit policy explanation and periodic review."
        )

    if review_summary and int(review_summary.get("low_confidence_awards_count", 0)) > 0:
        lines.append(
            f"{int(review_summary['low_confidence_awards_count']):,} historical awards fall into the model's low-confidence bucket, so the current process still leaves a visible review queue."
        )

    return lines[:3]


def build_business_implication_lines(
    stable_factors: pd.DataFrame,
    dimension_summary: pd.DataFrame,
    low_segments: pd.DataFrame,
    review_summary: dict[str, object] | None,
) -> list[str]:
    """Translate the analysis into concrete business implications."""
    lines: list[str] = []

    stable_labels = stable_factors["metric"].head(2).tolist()
    if stable_labels:
        weakest_segment = low_segments.iloc[0] if not low_segments.empty else None
        if weakest_segment is not None:
            lines.append(
                f"Focus committee calibration on {format_label_list(stable_labels, max_items=2)} inside low-award segments like {weakest_segment['group']} ({weakest_segment['dimension']})."
            )
        else:
            lines.append(
                f"Keep {format_label_list(stable_labels, max_items=2)} at the center of manual review because they are the cleanest need-aligned signals on the page."
            )

    top_dimension = (
        str(dimension_summary.iloc[0]["dimension"])
        if not dimension_summary.empty
        else None
    )
    if top_dimension:
        lines.append(
            f"If {top_dimension.lower()} differences are intentional, document the rationale clearly; if not, this page shows where reviewer guidance or rules should be tightened."
        )

    if review_summary and int(review_summary.get("high_priority_count", 0)) > 0:
        lines.append(
            f"Use the {int(review_summary['high_priority_count']):,}-case high-priority review queue for targeted re-checks instead of broadening one-factor rules."
        )

    return lines[:2]


def render_executive_takeaway_panel(
    takeaway: str,
    audit_risks: list[str],
    business_implications: list[str],
) -> None:
    """Render the top-line decision story with risks and actions."""
    audit_html = "".join(f"<li>{html.escape(line)}</li>" for line in audit_risks)
    business_html = "".join(f"<li>{html.escape(line)}</li>" for line in business_implications)
    st.markdown(
        f"""
        <section style="
            border-left: 6px solid {PRIMARY_COLOR};
            background: linear-gradient(180deg, #f0fdfa 0%, #f8fafc 100%);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin: 0.1rem 0 1rem 0;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        ">
            <div style="
                color: {PRIMARY_COLOR};
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin-bottom: 0.3rem;
            ">Executive Takeaway</div>
            <p style="
                color: {TEXT_DARK};
                font-size: 1.08rem;
                line-height: 1.5;
                font-weight: 700;
                margin: 0 0 0.9rem 0;
            ">{html.escape(takeaway)}</p>
            <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem;">
                <div style="background: rgba(255,255,255,0.82); border: 1px solid #e2e8f0; border-radius: 14px; padding: 0.85rem 0.95rem;">
                    <div style="color: {NEGATIVE_COLOR}; font-weight: 800; margin-bottom: 0.35rem;">Audit Risks</div>
                    <ul style="margin: 0; padding-left: 1rem; color: {TEXT_MUTED}; line-height: 1.5;">{audit_html}</ul>
                </div>
                <div style="background: rgba(255,255,255,0.82); border: 1px solid #e2e8f0; border-radius: 14px; padding: 0.85rem 0.95rem;">
                    <div style="color: {PRIMARY_COLOR}; font-weight: 800; margin-bottom: 0.35rem;">Business Implications</div>
                    <ul style="margin: 0; padding-left: 1rem; color: {TEXT_MUTED}; line-height: 1.5;">{business_html}</ul>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def build_share_breakdown(
    dataframe: pd.DataFrame,
    label_column: str,
    value_column: str,
    *,
    top_n: int | None = None,
    other_label: str = "Other",
) -> pd.DataFrame:
    """Convert a numeric ranking into a pie-friendly percentage breakdown."""
    if dataframe.empty or label_column not in dataframe.columns or value_column not in dataframe.columns:
        return pd.DataFrame(columns=["label", "value", "share_pct", "share_label"])

    working = dataframe[[label_column, value_column]].copy()
    working[label_column] = (
        working[label_column].astype("string").fillna("Missing").str.strip().replace("", "Missing")
    )
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce").fillna(0.0)
    working = (
        working.groupby(label_column, as_index=False)[value_column]
        .sum()
        .rename(columns={label_column: "label", value_column: "value"})
        .sort_values("value", ascending=False, ignore_index=True)
    )
    working = working[working["value"] > 0].reset_index(drop=True)
    if working.empty:
        return pd.DataFrame(columns=["label", "value", "share_pct", "share_label"])

    if top_n is not None and top_n > 0 and len(working) > top_n:
        top = working.head(top_n).copy()
        other_value = float(working.iloc[top_n:]["value"].sum())
        if other_value > 0:
            top.loc[len(top)] = {"label": other_label, "value": other_value}
        working = top.reset_index(drop=True)

    total = float(working["value"].sum())
    working["share_pct"] = np.where(total > 0, working["value"] / total * 100.0, 0.0)
    working["share_label"] = working["share_pct"].map(lambda value: f"{value:.1f}%")
    return working


def build_factor_influence_share(
    factor_scorecard: pd.DataFrame,
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """Show how much each factor contributes to the displayed evidence pool."""
    return build_share_breakdown(
        factor_scorecard.rename(columns={"metric": "label", "evidence_score": "value"}),
        "label",
        "value",
        top_n=top_n,
        other_label="Other factors",
    )


def build_factor_status_share(
    factor_scorecard: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how evidence is split between clean and audit-worthy factors."""
    if factor_scorecard.empty:
        return pd.DataFrame(columns=["label", "value", "share_pct", "share_label"])

    status_frame = factor_scorecard.groupby("factor_status", as_index=False).agg(
        evidence_score=("evidence_score", "sum")
    )
    return build_share_breakdown(
        status_frame.rename(columns={"factor_status": "label", "evidence_score": "value"}),
        "label",
        "value",
    )


def build_factor_count_share(
    factor_scorecard: pd.DataFrame,
) -> pd.DataFrame:
    """Show how many tracked factors fall into each interpretation bucket."""
    if factor_scorecard.empty:
        return pd.DataFrame(columns=["label", "value", "share_pct", "share_label"])

    count_frame = factor_scorecard.groupby("factor_status", as_index=False).agg(
        factor_count=("metric", "size")
    )
    return build_share_breakdown(
        count_frame.rename(columns={"factor_status": "label", "factor_count": "value"}),
        "label",
        "value",
    )


def build_factor_priority_share(
    factor_priority_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Show how evidence is split across decision-use priority tiers."""
    if factor_priority_frame.empty:
        return pd.DataFrame(columns=["label", "value", "share_pct", "share_label"])

    priority_frame = factor_priority_frame.groupby("priority", as_index=False).agg(
        evidence_score=("evidence_score", "sum")
    )
    return build_share_breakdown(
        priority_frame.rename(columns={"priority": "label", "evidence_score": "value"}),
        "label",
        "value",
    )


def build_factor_priority_count_share(
    factor_priority_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Show how many factors land in each use tier."""
    if factor_priority_frame.empty:
        return pd.DataFrame(columns=["label", "value", "share_pct", "share_label"])

    count_frame = factor_priority_frame.groupby("priority", as_index=False).agg(
        factor_count=("metric", "size")
    )
    return build_share_breakdown(
        count_frame.rename(columns={"priority": "label", "factor_count": "value"}),
        "label",
        "value",
    )


def build_dimension_influence_share(
    dimension_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Convert dimension-level spread into a compact influence split."""
    return build_share_breakdown(
        dimension_summary.rename(columns={"dimension": "label", "spread_pp": "value"}),
        "label",
        "value",
    )


def build_group_application_share(
    group_patterns: pd.DataFrame,
    *,
    top_n: int = 6,
) -> pd.DataFrame:
    """Summarize which groups make up most of the currently selected segment view."""
    return build_share_breakdown(
        group_patterns.rename(columns={"group": "label", "applications": "value"}),
        "label",
        "value",
        top_n=top_n,
        other_label="Other groups",
    )


def build_model_influence_share(
    feature_importance_df: pd.DataFrame,
    *,
    top_n: int = 6,
) -> pd.DataFrame:
    """Summarize model importance concentration as a donut chart."""
    return build_share_breakdown(
        feature_importance_df.rename(columns={"feature": "label", "importance": "value"}),
        "label",
        "value",
        top_n=top_n,
        other_label="Other features",
    )


def render_donut_chart(
    dataframe: pd.DataFrame,
    *,
    title: str,
    empty_message: str,
    value_title: str,
    summary_subject: str,
    subtitle: str | None = None,
    color_domain: list[str] | None = None,
    color_range: list[str] | None = None,
    value_format: str = ".3f",
    label_threshold_pct: float = 8.0,
) -> None:
    """Render a compact donut chart with percentage labels."""
    st.markdown(f"**{title}**")
    if dataframe.empty:
        st.info(empty_message)
        return

    chart_frame = collapse_small_categories(
        dataframe,
        label_column="label",
        value_column="value",
        max_categories=5,
    )

    if color_domain and color_range:
        color_encoding = alt.Color(
            "label:N",
            title=None,
            legend=alt.Legend(orient="bottom"),
            scale=alt.Scale(domain=color_domain, range=color_range),
        )
    else:
        color_encoding = alt.Color(
            "label:N",
            title=None,
            legend=alt.Legend(orient="bottom"),
            scale=alt.Scale(range=NEUTRAL_PIE_COLORS),
        )

    base = alt.Chart(chart_frame).encode(
        theta=alt.Theta("value:Q"),
        color=color_encoding,
        order=alt.Order("value:Q", sort="descending"),
        tooltip=[
            alt.Tooltip("label:N", title="Category"),
            alt.Tooltip("value:Q", title=value_title, format=value_format),
            alt.Tooltip("share_pct:Q", title="Share (%)", format=".1f"),
        ],
    )
    donut = base.mark_arc(
        innerRadius=58,
        outerRadius=94,
        stroke="#ffffff",
        strokeWidth=1.5,
    )
    labels = (
        base.transform_filter(alt.datum.share_pct >= label_threshold_pct)
        .mark_text(radius=76, color="#ffffff", fontSize=11, fontWeight="bold")
        .encode(text="share_label:N")
    )
    st.altair_chart(alt.layer(donut, labels).properties(height=280), width="stretch")
    if subtitle:
        st.caption(subtitle)

    top_slice = chart_frame.iloc[0]
    st.caption(
        f"Top slice: {top_slice['label']} at {top_slice['share_pct']:.1f}% of displayed {summary_subject}."
    )


def build_factor_scorecard(
    numeric_drivers: pd.DataFrame,
    counterintuitive_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine signal strength, coverage, and direction into one factor scorecard."""
    if numeric_drivers.empty:
        return pd.DataFrame(
            columns=[
                "metric_column",
                "metric",
                "awarded_median",
                "denied_median",
                "median_gap",
                "separation_score",
                "coverage_pct",
                "direction",
                "counterintuitive",
                "factor_status",
                "coverage_share",
                "evidence_score",
                "relative_gap_pct",
                "absolute_gap",
            ]
        )

    counterintuitive_set = set(counterintuitive_metrics["metric_column"].tolist())
    factor_frame = numeric_drivers.copy()
    factor_frame["counterintuitive"] = factor_frame["metric_column"].isin(counterintuitive_set)
    factor_frame["factor_status"] = np.where(
        factor_frame["counterintuitive"],
        "Important but audit",
        "Need-aligned",
    )
    factor_frame["coverage_share"] = factor_frame["coverage_pct"] / 100.0
    factor_frame["evidence_score"] = (
        factor_frame["separation_score"].fillna(0.0) * factor_frame["coverage_share"]
    )
    factor_frame["absolute_gap"] = factor_frame["median_gap"].abs()
    nonzero_denominator = factor_frame["denied_median"].abs() > 1e-9
    factor_frame["relative_gap_pct"] = np.where(
        nonzero_denominator,
        factor_frame["median_gap"] / factor_frame["denied_median"].abs() * 100.0,
        np.where(factor_frame["absolute_gap"] <= 1e-9, 0.0, np.nan),
    )
    return factor_frame.sort_values(
        by=["evidence_score", "separation_score", "coverage_pct"],
        ascending=[False, False, False],
        ignore_index=True,
    )


def match_factor_to_model_features(
    metric_column: str,
    model_feature_names: list[str],
) -> list[str]:
    """Find model features that map to the same broad factor theme."""
    if not model_feature_names:
        return []

    keywords = FACTOR_THEME_KEYWORDS.get(
        str(metric_column),
        tuple(
            token
            for token in str(metric_column).lower().split("_")
            if token and len(token) >= 4
        ),
    )
    matches: list[str] = []
    for feature_name in model_feature_names:
        lowered = str(feature_name).lower()
        if any(keyword in lowered for keyword in keywords):
            pretty_name = prettify_model_feature_label(str(feature_name))
            if pretty_name not in matches:
                matches.append(pretty_name)
    return matches[:2]


def build_factor_priority_reason(row: pd.Series) -> str:
    """Explain why the factor landed in its current priority tier."""
    coverage = float(row["coverage_pct"])
    separation = float(row["separation_score"])
    if str(row["priority"]) == "Primary":
        return f"High separation ({separation:.2f}) + good coverage ({coverage:.1f}%)."
    if str(row["priority"]) == "Audit":
        if coverage >= 60:
            return (
                f"Strong signal ({separation:.2f}) + broad coverage ({coverage:.1f}%), "
                "but the direction is contradictory."
            )
        return (
            f"Contradictory direction with only partial coverage ({coverage:.1f}%), "
            "so it cannot be trusted as a rule."
        )
    if coverage < 20:
        return f"Low coverage ({coverage:.1f}%) makes this too thin for direct use."
    return f"Usable coverage ({coverage:.1f}%), but weak separation ({separation:.2f})."


def build_factor_action_label(row: pd.Series) -> str:
    """Assign a short action label to each factor."""
    if str(row["priority"]) == "Primary":
        return "Use"
    if str(row["priority"]) == "Audit":
        return "Audit"
    if float(row["coverage_pct"]) < 20:
        return "Ignore"
    return "Support"


def build_factor_action_note(row: pd.Series) -> str:
    """Turn factor evidence into a direct recommendation."""
    action_label = str(row["action_label"])
    if action_label == "Use":
        return "Use directly in reviewer guidance and case explanations."
    if action_label == "Audit":
        return "Review only; do not turn this into a policy or scoring rule."
    if action_label == "Ignore":
        return "Ignore in broad rules until coverage improves."
    return "Use only as supporting context; it should not carry decision weight."


def build_factor_model_alignment(
    row: pd.Series,
    model_feature_names: list[str],
) -> tuple[str, str]:
    """Describe whether the current model reinforces this factor story."""
    matches = match_factor_to_model_features(str(row["metric_column"]), model_feature_names)
    priority = str(row["priority"])
    if not model_feature_names:
        return "Not aligned", "Saved model importance is unavailable."
    if matches:
        match_text = format_label_list(matches, max_items=2)
        if priority == "Primary":
            return "Aligned", f"Model also surfaces {match_text}."
        return "Partial", f"Model sees related features ({match_text}), but this factor still needs caution."
    return "Not aligned", "This factor theme is not prominent in saved model importance."


def build_factor_priority_frame(
    factor_scorecard: pd.DataFrame,
    model_feature_names: list[str],
) -> pd.DataFrame:
    """Classify factors into direct-use, caution, and audit tiers."""
    if factor_scorecard.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "metric_column",
                "priority",
                "priority_level",
                "priority_reason",
                "recommended_action",
                "model_behavior",
            ]
        )

    frame = factor_scorecard.copy()
    primary_mask = (
        (~frame["counterintuitive"])
        & (frame["coverage_pct"].fillna(0.0) >= 25.0)
        & (frame["separation_score"].fillna(0.0) >= 0.25)
    )
    frame["priority"] = np.select(
        [frame["counterintuitive"], primary_mask],
        ["Audit", "Primary"],
        default="Secondary",
    )
    frame["priority_level"] = frame["priority"].map(
        {"Primary": "High", "Secondary": "Medium", "Audit": "Audit"}
    )
    frame["priority_reason"] = frame.apply(build_factor_priority_reason, axis=1)
    frame["action_label"] = frame.apply(build_factor_action_label, axis=1)
    frame["recommended_action"] = frame.apply(build_factor_action_note, axis=1)
    model_alignment = frame.apply(
        lambda row: build_factor_model_alignment(row, model_feature_names),
        axis=1,
    )
    frame["model_alignment"] = model_alignment.map(lambda value: value[0])
    frame["model_alignment_note"] = model_alignment.map(lambda value: value[1])
    frame["model_behavior"] = frame["model_alignment"] + ": " + frame["model_alignment_note"]
    frame["is_key_contradiction"] = frame["metric"].astype(str).eq("Dependents Count")
    frame["priority_order"] = frame["priority"].map({"Primary": 0, "Secondary": 1, "Audit": 2})
    return frame.sort_values(
        by=["priority_order", "is_key_contradiction", "evidence_score", "coverage_pct"],
        ascending=[True, False, False, False],
        ignore_index=True,
    )


def build_factor_executive_takeaway(
    factor_priority_frame: pd.DataFrame,
    model_feature_names: list[str],
) -> str:
    """Summarize the factors tab in one directive sentence."""
    if factor_priority_frame.empty:
        return "Factor evidence is not available yet."

    primary_factors = factor_priority_frame.loc[
        factor_priority_frame["priority"].eq("Primary"),
        "metric",
    ].head(2).tolist()
    audit_factors = factor_priority_frame.loc[
        factor_priority_frame["priority"].eq("Audit"),
        "metric",
    ].head(1).tolist()
    primary_text = (
        format_label_list(primary_factors, max_items=2)
        if primary_factors
        else "no factor is currently strong enough for direct use"
    )
    audit_text = audit_factors[0] if audit_factors else "the contradictory signals"
    total_factors = len(factor_priority_frame)
    reliable_count = int(factor_priority_frame["priority"].eq("Primary").sum())
    model_link = (
        " The saved model also leans on related financial themes."
        if any(
            match_factor_to_model_features(metric_column, model_feature_names)
            for metric_column in factor_priority_frame.loc[
                factor_priority_frame["priority"].eq("Primary"),
                "metric_column",
            ].tolist()
        )
        else ""
    )
    return (
        f"Only {reliable_count} of {total_factors} tracked factors are strong enough to use directly: {primary_text} drive the clearest decision split, while {audit_text} is strong on paper but too contradictory to trust as a rule.{model_link}"
    )


def build_factor_decision_summary(factor_priority_frame: pd.DataFrame) -> str:
    """State the factor tab's operating rule in one line."""
    if factor_priority_frame.empty:
        return "Decision summary: factor evidence is not available yet."

    use_factors = factor_priority_frame.loc[
        factor_priority_frame["action_label"].eq("Use"),
        "metric",
    ].head(2).tolist()
    support_factors = factor_priority_frame.loc[
        factor_priority_frame["action_label"].eq("Support"),
        "metric",
    ].head(2).tolist()
    ignore_factors = factor_priority_frame.loc[
        factor_priority_frame["action_label"].eq("Ignore"),
        "metric",
    ].head(1).tolist()
    audit_factor = factor_priority_frame.loc[
        factor_priority_frame["is_key_contradiction"],
        "metric",
    ].head(1).tolist()

    use_text = (
        format_label_list(use_factors, max_items=2)
        if use_factors
        else "no factor"
    )
    support_text = (
        format_label_list(support_factors, max_items=2)
        if support_factors
        else "other context only"
    )
    ignore_text = f", ignore thin signals like {ignore_factors[0]}" if ignore_factors else ""
    audit_text = audit_factor[0] if audit_factor else "contradictions"
    return (
        f"Decision summary: use {use_text}, support {support_text}{ignore_text}, and audit contradictions led by {audit_text}."
    )


def render_factor_executive_panel(takeaway: str) -> None:
    """Render a top-line takeaway for the factors tab."""
    st.markdown(
        f"""
        <section style="
            border-left: 6px solid {PRIMARY_COLOR};
            background: linear-gradient(180deg, #f0fdfa 0%, #ffffff 100%);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin: 0.1rem 0 1rem 0;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        ">
            <div style="
                color: {PRIMARY_COLOR};
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin-bottom: 0.3rem;
            ">Executive Takeaway</div>
            <p style="
                color: {TEXT_DARK};
                font-size: 1.02rem;
                line-height: 1.55;
                font-weight: 700;
                margin: 0;
            ">{html.escape(takeaway)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def build_factor_decision_table(
    factor_priority_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create a prioritized factor table for reviewer and policy decisions."""
    if factor_priority_frame.empty:
        return pd.DataFrame(
            columns=[
                "Priority Level",
                "Action",
                "Factor",
                "Why This Tier",
                "Observed Pattern",
                "Model Link",
            ]
        )

    display = factor_priority_frame.copy()
    display["Factor"] = np.where(
        display["is_key_contradiction"],
        "⚠ Dependents Count",
        display["metric"],
    )
    display["Observed Pattern"] = display.apply(
        lambda row: (
            f"{row['direction']} "
            f"({float(row['awarded_median']):,.0f} vs {float(row['denied_median']):,.0f}; "
            f"{float(row['coverage_pct']):.1f}% coverage)"
        ),
        axis=1,
    )
    display["Model Link"] = (
        display["model_alignment"].astype(str) + ": " + display["model_alignment_note"].astype(str)
    )
    return display.rename(
        columns={
            "priority_level": "Priority Level",
            "action_label": "Action",
            "priority_reason": "Why This Tier",
        }
    )[
        ["Priority Level", "Action", "Factor", "Why This Tier", "Observed Pattern", "Model Link"]
    ]


def style_factor_decision_table(factor_decision_table: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Highlight the key contradiction row in the decision table."""
    def highlight_row(row: pd.Series) -> list[str]:
        if str(row.get("Factor", "")).startswith("⚠"):
            return ["background-color: #fff7ed; font-weight: 700; color: #9a3412;"] * len(row)
        if str(row.get("Action", "")) == "Use":
            return ["background-color: #f0fdfa;"] * len(row)
        return [""] * len(row)

    return factor_decision_table.style.apply(highlight_row, axis=1)


def style_appendix_scorecard(appendix_scorecard: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Color-code the appendix scorecard so the action tier is immediately visible."""

    def highlight_row(row: pd.Series) -> list[str]:
        factor_text = str(row.get("Factor", ""))
        action = str(row.get("Action", ""))

        if factor_text.startswith("⚠"):
            return ["background-color: #fff7ed; color: #9a3412; font-weight: 700;"] * len(row)
        if action == "Use":
            return ["background-color: #ecfdf5; color: #166534;"] * len(row)
        if action == "Support":
            return ["background-color: #fffbeb; color: #92400e;"] * len(row)
        if action == "Audit":
            return ["background-color: #fef2f2; color: #b91c1c; font-weight: 700;"] * len(row)
        if action == "Ignore":
            return ["background-color: #f8fafc; color: #475569;"] * len(row)
        return [""] * len(row)

    return appendix_scorecard.style.apply(highlight_row, axis=1)



def build_segment_universe(
    labeled_df: pd.DataFrame,
    *,
    min_group_size: int,
) -> pd.DataFrame:
    """Compile every supported grouped segment view into one analytic frame."""
    frames: list[pd.DataFrame] = []

    for dimension_label, group_column in GROUP_DIMENSIONS.items():
        group_patterns = summarize_group_patterns(
            labeled_df,
            group_column,
            min_group_size=min_group_size,
        )
        if group_patterns.empty:
            continue

        group_patterns = group_patterns.copy()
        group_patterns["dimension"] = dimension_label
        group_patterns["segment_label"] = (
            group_patterns["group"].astype(str) + " (" + dimension_label + ")"
        )
        group_patterns["abs_lift_pp"] = group_patterns["lift_pp"].abs()
        group_patterns["lift_direction"] = np.where(
            group_patterns["lift_pp"] >= 0,
            "Above baseline",
            "Below baseline",
        )
        frames.append(group_patterns)

    if not frames:
        return pd.DataFrame(
            columns=[
                "dimension",
                "group",
                "applications",
                "awarded",
                "award_rate",
                "award_rate_pct",
                "lift_pp",
                "share_of_sample_pct",
                "segment_label",
                "abs_lift_pp",
                "lift_direction",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def build_dimension_summary(segment_universe: pd.DataFrame) -> pd.DataFrame:
    """Summarize how much decision outcomes vary within each grouping dimension."""
    if segment_universe.empty:
        return pd.DataFrame(
            columns=[
                "dimension",
                "groups",
                "avg_abs_lift_pp",
                "max_positive_lift_pp",
                "max_negative_lift_pp",
                "spread_pp",
            ]
        )

    summary = (
        segment_universe.groupby("dimension", as_index=False)
        .agg(
            groups=("group", "size"),
            avg_abs_lift_pp=("abs_lift_pp", "mean"),
            max_positive_lift_pp=("lift_pp", "max"),
            max_negative_lift_pp=("lift_pp", "min"),
        )
        .sort_values("avg_abs_lift_pp", ascending=False, ignore_index=True)
    )
    summary["spread_pp"] = (
        summary["max_positive_lift_pp"] - summary["max_negative_lift_pp"]
    )
    return summary


def build_segment_extreme_frame(segment_universe: pd.DataFrame, *, top_n: int = 8) -> pd.DataFrame:
    """Return the most positive and most negative segments across all dimensions."""
    if segment_universe.empty:
        return pd.DataFrame(
            columns=[
                "dimension",
                "group",
                "applications",
                "award_rate_pct",
                "lift_pp",
                "share_of_sample_pct",
                "segment_label",
                "abs_lift_pp",
                "impact_score",
                "action_tag",
            ]
        )

    working = segment_universe.copy()
    working["impact_score"] = working["abs_lift_pp"] * working["applications"]
    working["action_tag"] = np.where(
        working["lift_pp"] < 0,
        "Review priority",
        "Benchmark",
    )
    working["action_order"] = np.where(working["lift_pp"] < 0, 0, 1)

    positive = working[working["lift_pp"] > 0].sort_values(
        ["impact_score", "applications", "lift_pp"],
        ascending=[False, False, False],
        ignore_index=True,
    ).head(top_n)
    negative = working[working["lift_pp"] < 0].sort_values(
        ["impact_score", "applications", "abs_lift_pp"],
        ascending=[False, False, False],
        ignore_index=True,
    ).head(top_n)

    extremes = (
        pd.concat([positive, negative], ignore_index=True)
        .drop_duplicates(subset=["segment_label"])
        .sort_values(["action_order", "impact_score"], ascending=[True, False], ignore_index=True)
    )
    return extremes.drop(columns=["action_order"])


def build_segment_driver_text(
    labeled_df: pd.DataFrame,
    *,
    dimension: str,
    group: str,
    fallback_factor_labels: list[str],
) -> str:
    """Explain which numeric signals most likely drive a segment's internal split."""
    group_column = GROUP_DIMENSIONS.get(str(dimension))
    if not group_column or group_column not in labeled_df.columns:
        return format_label_list(fallback_factor_labels, max_items=2)

    group_values = labeled_df[group_column].astype("string").fillna("Missing").str.strip()
    group_values = group_values.mask(group_values.eq(""), "Missing")
    segment_slice = labeled_df.loc[group_values.eq(str(group))].copy()
    if len(segment_slice) < 40:
        return format_label_list(fallback_factor_labels, max_items=2)

    min_non_null = max(10, min(20, len(segment_slice) // 6))
    numeric_drivers = summarize_numeric_drivers(
        segment_slice,
        min_non_null_per_group=min_non_null,
    )
    if numeric_drivers.empty:
        return format_label_list(fallback_factor_labels, max_items=2)

    counterintuitive_metrics = find_counterintuitive_metrics(numeric_drivers)
    factor_scorecard = build_factor_scorecard(numeric_drivers, counterintuitive_metrics)
    stable_factors = factor_scorecard[
        (~factor_scorecard["counterintuitive"]) & (factor_scorecard["coverage_pct"] >= 20)
    ]
    chosen_factors = stable_factors["metric"].head(2).tolist()
    if not chosen_factors:
        chosen_factors = factor_scorecard["metric"].head(2).tolist()
    if not chosen_factors:
        chosen_factors = fallback_factor_labels
    return format_label_list(chosen_factors, max_items=2)


def build_segment_driver_lookup(
    labeled_df: pd.DataFrame,
    segment_frame: pd.DataFrame,
    fallback_factor_labels: list[str],
) -> dict[tuple[str, str], str]:
    """Attach short factor cues to the segments we expect users to inspect."""
    lookup: dict[tuple[str, str], str] = {}
    if segment_frame.empty:
        return lookup

    for _, row in segment_frame.iterrows():
        key = (str(row["dimension"]), str(row["group"]))
        if key in lookup:
            continue
        lookup[key] = build_segment_driver_text(
            labeled_df,
            dimension=str(row["dimension"]),
            group=str(row["group"]),
            fallback_factor_labels=fallback_factor_labels,
        )
    return lookup


def build_segment_priority_table(
    segment_extremes: pd.DataFrame,
    driver_lookup: dict[tuple[str, str], str],
) -> pd.DataFrame:
    """Create an action-first table for the most important segments."""
    if segment_extremes.empty:
        return pd.DataFrame(
            columns=[
                "Action",
                "Segment",
                "Size and Lift",
                "Driven by",
            ]
        )

    display = segment_extremes.copy()
    display["Action"] = display["action_tag"]
    display["Segment"] = display["group"].astype(str) + " (" + display["dimension"].astype(str) + ")"
    display["Size and Lift"] = display.apply(
        lambda row: (
            f"n={int(row['applications']):,} | lift {float(row['lift_pp']):+.1f} pp "
            f"| impact {float(row['impact_score']):,.0f}"
        ),
        axis=1,
    )
    display["Driven by"] = display.apply(
        lambda row: driver_lookup.get((str(row["dimension"]), str(row["group"])), "Needs factor review"),
        axis=1,
    )
    return display[["Action", "Segment", "Size and Lift", "Driven by"]]


def style_segment_priority_table(priority_table: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Highlight review rows and benchmarks in the segment action table."""
    def highlight_row(row: pd.Series) -> list[str]:
        action = str(row.get("Action", ""))
        if action == "Review priority":
            return ["background-color: #fef2f2; color: #991b1b; font-weight: 700;"] * len(row)
        if action == "Benchmark":
            return ["background-color: #f0fdfa; color: #115e59;"] * len(row)
        return [""] * len(row)

    return priority_table.style.apply(highlight_row, axis=1)


def build_segment_decision_summary(segment_extremes: pd.DataFrame) -> str:
    """State the top review and monitoring segments in one line."""
    if segment_extremes.empty:
        return "Focus review on: not enough segment evidence yet."

    negative = segment_extremes[segment_extremes["lift_pp"] < 0].head(2)
    positive = segment_extremes[segment_extremes["lift_pp"] > 0].head(1)
    review_text = format_label_list(
        [
            f"{row['group']} ({row['dimension']})"
            for _, row in negative.iterrows()
        ],
        max_items=2,
    )
    if review_text == "N/A":
        review_text = "no negative segment is large enough yet"
    monitor_text = format_label_list(
        [
            f"{row['group']} ({row['dimension']})"
            for _, row in positive.iterrows()
        ],
        max_items=1,
    )
    if monitor_text == "N/A":
        monitor_text = "no positive benchmark yet"
    return f"Focus review on: {review_text} | Monitor: {monitor_text}"


def build_segment_scatter_frame(
    segment_universe: pd.DataFrame,
    *,
    min_group_size: int,
) -> pd.DataFrame:
    """Prepare the all-segments scatter with action-oriented styling."""
    if segment_universe.empty:
        return pd.DataFrame()

    frame = segment_universe.copy()
    low_sample_threshold = max(min_group_size * 1.5, 60)
    frame["opacity_level"] = np.where(frame["applications"] < low_sample_threshold, 0.22, 0.82)

    label_indices = frame["applications"].nlargest(5).index
    frame["label_text"] = ""
    frame.loc[label_indices, "label_text"] = frame.loc[label_indices, "group"].astype(str)
    return frame


def build_segment_annotation_frame(segment_universe: pd.DataFrame) -> pd.DataFrame:
    """Create lightweight quadrant labels for the all-segments scatter."""
    if segment_universe.empty:
        return pd.DataFrame(columns=["x", "y", "label"])

    max_share = max(float(segment_universe["share_of_sample_pct"].max()), 1.0)
    min_lift = float(segment_universe["lift_pp"].min())
    max_lift = float(segment_universe["lift_pp"].max())
    near_zero = max(float(segment_universe["abs_lift_pp"].quantile(0.15)), 1.0)

    return pd.DataFrame(
        [
            {
                "x": max_share * 0.72,
                "y": min_lift * 0.82 if min_lift < 0 else -near_zero,
                "label": "High impact (review)",
            },
            {
                "x": max_share * 0.18,
                "y": near_zero if max_lift > 0 else 0.0,
                "label": "Low impact (ignore)",
            },
        ]
    )


def build_selected_dimension_summary(
    group_patterns: pd.DataFrame,
    driver_lookup: dict[tuple[str, str], str],
    *,
    dimension: str,
) -> str:
    """Summarize the most important positive and negative group inside one dimension."""
    if group_patterns.empty:
        return "No eligible groups met the current threshold."

    sorted_groups = group_patterns.copy()
    sorted_groups["impact_score"] = sorted_groups["abs_lift_pp"] * sorted_groups["applications"]
    review_rows = sorted_groups[sorted_groups["lift_pp"] < 0].sort_values(
        ["impact_score", "applications"],
        ascending=[False, False],
    )
    benchmark_rows = sorted_groups[sorted_groups["lift_pp"] > 0].sort_values(
        ["impact_score", "applications"],
        ascending=[False, False],
    )

    parts: list[str] = []
    if not review_rows.empty:
        review_row = review_rows.iloc[0]
        review_driver = driver_lookup.get((dimension, str(review_row["group"])), "Needs factor review")
        parts.append(
            f"Review first: {review_row['group']} at n={int(review_row['applications']):,} and {float(review_row['lift_pp']):+.1f} pp. Driven by: {review_driver}."
        )
    if not benchmark_rows.empty:
        benchmark_row = benchmark_rows.iloc[0]
        benchmark_driver = driver_lookup.get((dimension, str(benchmark_row["group"])), "Needs factor review")
        parts.append(
            f"Benchmark: {benchmark_row['group']} at n={int(benchmark_row['applications']):,} and {float(benchmark_row['lift_pp']):+.1f} pp. Driven by: {benchmark_driver}."
        )
    return " ".join(parts)


def build_segment_risk_lines(
    segment_extremes: pd.DataFrame,
    driver_lookup: dict[tuple[str, str], str],
    *,
    top_n: int = 3,
) -> list[str]:
    """Call out the groups most likely to need fairness or consistency review."""
    if segment_extremes.empty:
        return []

    risk_frame = segment_extremes[segment_extremes["lift_pp"] < 0].head(top_n)
    lines: list[str] = []
    for _, row in risk_frame.iterrows():
        driven_by = driver_lookup.get((str(row["dimension"]), str(row["group"])), "Needs factor review")
        lines.append(
            f"{row['group']} ({row['dimension']}) at n={int(row['applications']):,} and {float(row['lift_pp']):+.1f} pp. Driven by: {driven_by}."
        )
    return lines


def build_factor_distribution_frame(
    labeled_df: pd.DataFrame,
    metric_column: str,
) -> pd.DataFrame:
    """Prepare one metric for an awarded-vs-denied distribution chart."""
    if metric_column not in labeled_df.columns:
        return pd.DataFrame()

    frame = labeled_df[["decision", metric_column]].copy()
    frame[metric_column] = pd.to_numeric(frame[metric_column], errors="coerce")
    frame = frame.dropna(subset=[metric_column])
    if frame.empty:
        return frame

    frame["decision_label"] = (
        frame["decision"].astype("string").str.strip().str.lower().map(
            {"awarded": "Awarded", "denied": "Denied"}
        )
    )
    return frame.dropna(subset=["decision_label"])


def build_review_reason_summary(similar_profile_pairs: pd.DataFrame) -> pd.DataFrame:
    """Count why matched cases were flagged for review."""
    if similar_profile_pairs.empty or "review_note" not in similar_profile_pairs.columns:
        return pd.DataFrame(
            columns=["review_note", "reason_label", "reason_type", "count", "share_pct", "share_label"]
        )

    summary = (
        similar_profile_pairs["review_note"]
        .value_counts()
        .rename_axis("review_note")
        .reset_index(name="count")
    )
    summary["reason_label"] = summary["review_note"].map(shorten_review_reason)
    summary["reason_type"] = summary["review_note"].map(classify_review_reason_type)
    total = float(summary["count"].sum())
    summary["share_pct"] = np.where(total > 0, summary["count"] / total * 100.0, 0.0)
    summary["share_label"] = summary["share_pct"].map(lambda value: f"{value:.1f}%")
    return summary


def shorten_review_reason(review_note: object) -> str:
    """Turn long review notes into short chart/table labels."""
    note = str(review_note)
    if note.startswith("Application type differs"):
        return "Application Type"
    if note.startswith("Special-circumstance coding differs"):
        return "Special Circumstances"
    return "Near-identical financial profile"


def classify_review_reason_type(review_note: object) -> str:
    """Separate contextual review triggers from financial ones."""
    short_label = shorten_review_reason(review_note)
    if short_label in {"Application Type", "Special Circumstances"}:
        return "Context / categorical"
    return "Financial similarity"


def build_consistency_action_summary(
    similar_profile_summary: dict[str, object],
    mixed_profile_summary: dict[str, object],
) -> str:
    """State the consistency tab's immediate decision in one line."""
    return (
        f"Action: Review {int(similar_profile_summary.get('review_pair_count', 0)):,} closest matched pairs "
        f"and audit {int(mixed_profile_summary.get('mixed_group_count', 0)):,} mixed-profile groups for consistency."
    )


def build_consistency_why_lines(
    similar_profile_summary: dict[str, object],
) -> list[str]:
    """Keep only the two strongest reasons this tab matters."""
    applications_with_match = int(similar_profile_summary.get("applications_with_match", 0))
    review_pair_count = int(similar_profile_summary.get("review_pair_count", 0))
    return [
        f"{applications_with_match:,} denied applications have a similar awarded match, so review is needed.",
        f"{review_pair_count:,} closest pairs are the strongest case-level inconsistency signals.",
    ]


def build_consistency_reason_headline(review_reason_summary: pd.DataFrame) -> str:
    """Summarize the top inconsistency drivers in one sentence."""
    if review_reason_summary.empty:
        return "Top drivers of inconsistency are not available yet."

    top_rows = review_reason_summary.head(3)
    fragments = [
        f"{row['reason_label']} ({row['share_pct']:.1f}%, n={int(row['count'])})"
        for _, row in top_rows.iterrows()
    ]
    return "Top drivers of inconsistency: " + "; ".join(fragments) + "."


def build_consistency_key_insight(review_reason_summary: pd.DataFrame) -> str:
    """State the biggest thesis-level punchline from matched-pair reasons."""
    if review_reason_summary.empty:
        return "Insight: inconsistency drivers are not available yet."

    context_share = float(
        review_reason_summary.loc[
            review_reason_summary["reason_type"].eq("Context / categorical"),
            "share_pct",
        ].sum()
    )
    if context_share >= 50:
        return (
            "Insight: Inconsistencies are driven more by categorical/context differences "
            "than financial variables."
        )
    return (
        "Insight: Financial similarity still leaves visible decision mismatches, so case-level review remains necessary."
    )


def build_matched_pair_risk_level(
    pair_frame: pd.DataFrame,
    review_threshold: float,
) -> pd.Series:
    """Assign a simple risk tier to matched pairs for table styling."""
    if pair_frame.empty:
        return pd.Series(dtype="string")

    working = pair_frame.copy()
    working["distance_rank"] = working["distance"].rank(method="first", ascending=True)
    high_risk_mask = working["distance_rank"] <= 5
    moderate_cutoff = min(
        float(review_threshold),
        float(working["distance"].quantile(0.5)),
    )
    moderate_mask = (~high_risk_mask) & (working["distance"] <= moderate_cutoff)
    risk_level = np.select(
        [high_risk_mask, moderate_mask],
        ["High-risk mismatch", "Moderate review"],
        default="Review",
    )
    return pd.Series(risk_level, index=pair_frame.index, dtype="string")


def style_matched_pair_table(pair_display: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Highlight the most similar matched pairs and the review tiers."""
    def highlight_row(row: pd.Series) -> list[str]:
        risk_level = str(row.get("Risk Level", ""))
        if risk_level == "High-risk mismatch":
            return ["background-color: #fef2f2; color: #991b1b; font-weight: 700;"] * len(row)
        if risk_level == "Moderate review":
            return ["background-color: #fefce8; color: #854d0e;"] * len(row)
        return [""] * len(row)

    return pair_display.style.apply(highlight_row, axis=1)


def build_mixed_group_long(mixed_profile_groups: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    """Convert mixed-outcome groups into a long frame for stacked bars."""
    if mixed_profile_groups.empty:
        return pd.DataFrame()

    working = mixed_profile_groups.head(top_n).reset_index(drop=True).copy()
    working["group_id"] = [f"G{index + 1}" for index in range(len(working))]
    working["profile"] = working.apply(
        lambda row: (
            f"Level={row['Level']}; Income={row['Parent Income']}; "
            f"Property={row['Property Value']}; Dependents={row['Dependents']}; "
            f"Siblings={row['Siblings']}"
        ),
        axis=1,
    )

    long_frame = (
        working[
            ["group_id", "profile", "award_rate_pct", "review_score", "awarded", "denied"]
        ]
        .melt(
            id_vars=["group_id", "profile", "award_rate_pct", "review_score"],
            value_vars=["awarded", "denied"],
            var_name="decision_label",
            value_name="count",
        )
    )
    long_frame["decision_label"] = long_frame["decision_label"].map(
        {"awarded": "Awarded", "denied": "Denied"}
    )
    return long_frame


def build_scored_decision_frame(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Filter scored rows to comparable awarded-vs-denied outcomes."""
    required_columns = {"decision", "predicted_award_probability"}
    if not required_columns.issubset(scored_df.columns):
        return pd.DataFrame()

    frame = scored_df.copy()
    frame["decision_clean"] = frame["decision"].astype("string").str.strip().str.lower()
    frame["predicted_award_probability"] = pd.to_numeric(
        frame["predicted_award_probability"],
        errors="coerce",
    )
    frame = frame[
        frame["decision_clean"].isin(["awarded", "denied"])
        & frame["predicted_award_probability"].notna()
    ].copy()
    if frame.empty:
        return frame

    frame["decision_label"] = frame["decision_clean"].map(
        {"awarded": "Awarded", "denied": "Denied"}
    )
    return frame


def build_count_by_category(
    dataframe: pd.DataFrame,
    category_column: str,
    *,
    top_n: int = 8,
) -> pd.DataFrame:
    """Count records by a categorical field for bar-chart display."""
    if dataframe.empty or category_column not in dataframe.columns:
        return pd.DataFrame(columns=[category_column, "count"])

    counts = (
        dataframe[category_column]
        .astype("string")
        .fillna("Missing")
        .str.strip()
        .replace("", "Missing")
        .value_counts()
        .head(top_n)
        .rename_axis(category_column)
        .reset_index(name="count")
    )
    return counts


def format_label_list(values: list[str], *, max_items: int = 3) -> str:
    """Join short label lists into readable plain English."""
    cleaned = [str(value).strip() for value in values if pd.notna(value) and str(value).strip()]
    cleaned = cleaned[:max_items]
    if not cleaned:
        return "N/A"
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def build_submission_year_summary(
    labeled_df: pd.DataFrame,
    *,
    min_applications: int = 100,
) -> pd.DataFrame:
    """Summarize award rates by submission year."""
    if "submission_date" not in labeled_df.columns:
        return pd.DataFrame(
            columns=[
                "submission_year",
                "applications",
                "awarded",
                "award_rate",
                "award_rate_pct",
                "is_major_year",
            ]
        )

    years = pd.to_datetime(labeled_df["submission_date"], errors="coerce").dt.year
    summary = (
        labeled_df.assign(submission_year=years)
        .dropna(subset=["submission_year"])
        .groupby("submission_year", as_index=False)["award_flag"]
        .agg(applications="size", awarded="sum", award_rate="mean")
    )
    if summary.empty:
        return summary

    summary["submission_year"] = summary["submission_year"].astype(int)
    summary["award_rate_pct"] = summary["award_rate"] * 100
    summary["is_major_year"] = summary["applications"] >= min_applications
    return summary.sort_values("submission_year", ignore_index=True)


def build_year_group_summary(
    labeled_df: pd.DataFrame,
    group_column: str,
    *,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """Summarize award rates by submission year within a grouping variable."""
    if "submission_date" not in labeled_df.columns or group_column not in labeled_df.columns:
        return pd.DataFrame(
            columns=["submission_year", "group", "applications", "award_rate", "award_rate_pct"]
        )

    years = pd.to_datetime(labeled_df["submission_date"], errors="coerce").dt.year
    group_values = labeled_df[group_column].astype("string").fillna("Missing").str.strip()
    group_values = group_values.mask(group_values.eq(""), "Missing")

    summary = (
        labeled_df.assign(submission_year=years, group=group_values)
        .dropna(subset=["submission_year"])
        .groupby(["submission_year", "group"], as_index=False)["award_flag"]
        .agg(applications="size", award_rate="mean")
    )
    summary = summary[summary["applications"] >= min_group_size].copy()
    if summary.empty:
        return summary

    summary["submission_year"] = summary["submission_year"].astype(int)
    summary["award_rate_pct"] = summary["award_rate"] * 100
    return summary.sort_values(
        ["submission_year", "award_rate_pct"],
        ascending=[True, False],
        ignore_index=True,
    )


def build_key_question_answers(
    *,
    factor_scorecard: pd.DataFrame,
    stable_factors: pd.DataFrame,
    dimension_summary: pd.DataFrame,
    high_segments: pd.DataFrame,
    low_segments: pd.DataFrame,
    counterintuitive_metrics: pd.DataFrame,
    similar_profile_summary: dict[str, object],
    mixed_profile_summary: dict[str, object],
    year_summary: pd.DataFrame,
    level_year_summary: pd.DataFrame,
) -> list[dict[str, str]]:
    """Answer the core thesis questions with direct priorities and actions."""
    answers: list[dict[str, str]] = []
    driver_source = stable_factors if not stable_factors.empty else factor_scorecard
    top_factor_labels = driver_source["metric"].head(3).tolist()
    top_dimension_labels = dimension_summary["dimension"].head(3).tolist()
    primary_factors_text = (
        format_label_list(top_factor_labels[:2], max_items=2)
        if top_factor_labels
        else "the cleanest financial need signals"
    )
    top_context_text = (
        format_label_list(top_dimension_labels[:2], max_items=2)
        if top_dimension_labels
        else "the biggest contextual gaps"
    )
    audit_metric = (
        str(counterintuitive_metrics.iloc[0]["metric"])
        if not counterintuitive_metrics.empty
        else None
    )
    school_row = dimension_summary.loc[dimension_summary["dimension"].astype(str) == "School"]
    audit_dimension = (
        str(school_row.iloc[0]["dimension"])
        if not school_row.empty
        else str(dimension_summary.iloc[0]["dimension"])
        if not dimension_summary.empty
        else "contextual group differences"
    )
    support_factor = top_factor_labels[2] if len(top_factor_labels) >= 3 else None

    top_segment_bits: list[str] = []
    for _, row in high_segments.head(3).iterrows():
        top_segment_bits.append(
            f"{row['group']} in {row['dimension']} ({row['award_rate_pct']:.1f}% awarded)"
        )

    low_segment_bits: list[str] = []
    for _, row in low_segments.head(2).iterrows():
        low_segment_bits.append(
            f"{row['group']} in {row['dimension']} ({row['award_rate_pct']:.1f}% awarded)"
        )

    answers.append(
        {
            "question": "What factors most influence financial aid decisions?",
            "executive": (
                f"Put {primary_factors_text} at the center of every explanation; other signals are secondary or audit-only."
            ),
            "primary": (
                f"Use {primary_factors_text} first because awarded cases separate most cleanly from denied cases on those need-aligned signals."
            ),
            "secondary": (
                f"Use {support_factor} and contextual differences like {top_context_text} only after the financial explanation is already clear."
                if support_factor
                else f"Use contextual differences like {top_context_text} only after the financial explanation is already clear."
            ),
            "risk": (
                f"Keep {audit_metric} out of scoring and policy for now; it looks important statistically, but its direction is suspicious."
                if audit_metric
                else "Keep any unstable or counterintuitive signal out of scoring until its direction is explained."
            ),
            "next_step": (
                "Rewrite reviewer guidance so the first written rationale starts with financial need signals and flags any exception separately."
            ),
        }
    )

    answers.append(
        {
            "question": "Which variables matter the most (income? family size? school type?)",
            "executive": (
                f"Financial variables should lead; {audit_dimension.lower()} differences need explanation, not blind trust."
            ),
            "primary": (
                f"Make {primary_factors_text} the primary decision lens when comparing awarded and denied applications."
            ),
            "secondary": (
                f"Treat {top_context_text} as contextual review dimensions because they create the biggest outcome spread across groups."
            ),
            "risk": (
                f"Do not rely on family-size signals alone; {audit_metric} does not behave like a clean need rule in this file."
                if audit_metric
                else "Do not rely on thin or unstable demographic signals as a primary rule."
            ),
            "next_step": (
                "Update the review rubric so financial variables drive the first pass and school or context gaps trigger explanation or escalation."
            ),
        }
    )

    answers.append(
        {
            "question": "Are there patterns (e.g., certain profiles always get higher aid)?",
            "executive": (
                "Yes. There are repeatable high- and low-award patterns, but they should drive audit priority rather than automatic treatment."
            ),
            "primary": (
                f"Start with the strongest high-award patterns: {format_label_list(top_segment_bits)}."
                if top_segment_bits
                else "Start with the strongest high-award groups because they show where the historical process most consistently leaned positive."
            ),
            "secondary": (
                f"Use low-award patterns like {format_label_list(low_segment_bits, max_items=2)} as the sharper consistency checkpoints."
                if low_segment_bits
                else "Use the weakest-award groups as the sharper consistency checkpoints."
            ),
            "risk": (
                "These patterns come from historical outcomes inside an award-heavy sample, so turning them into rules would hard-code past imbalance."
            ),
            "next_step": (
                "Create a targeted audit for the highest-lift and lowest-lift groups and require a written rationale for why they differ from baseline."
            ),
        }
    )

    answers.append(
        {
            "question": "Are there biases or inconsistencies?",
            "executive": (
                "Yes. The file shows enough structured inconsistency to require active bias and consistency audit now."
            ),
            "primary": (
                f"Audit matched cases first: {int(similar_profile_summary.get('applications_with_match', 0)):,} denied applications have at least one similar awarded match and {int(similar_profile_summary.get('review_pair_count', 0)):,} sit in the closest review band."
            ),
            "secondary": (
                f"Then inspect repeated group gaps in {audit_dimension}: {int(mixed_profile_summary.get('mixed_group_count', 0)):,} profile buckets contain both awards and denials."
            ),
            "risk": (
                f"This is not causal proof of intent, but it is strong evidence that standards may not be landing evenly, especially where {audit_dimension.lower()} gaps stay wide and {audit_metric} behaves abnormally."
                if audit_metric
                else f"This is not causal proof of intent, but it is strong evidence that standards may not be landing evenly, especially where {audit_dimension.lower()} gaps stay wide."
            ),
            "next_step": (
                f"Start the audit with {audit_dimension} and matched awarded-vs-denied pairs, then escalate any repeated gap that cannot be tied back to documented policy."
            ),
        }
    )

    major_years = year_summary[year_summary["is_major_year"]] if not year_summary.empty else pd.DataFrame()
    year_trend_text = "There is not enough year data yet."
    if len(major_years) >= 2:
        previous_year = major_years.iloc[-2]
        latest_year = major_years.iloc[-1]
        delta_pp = latest_year["award_rate_pct"] - previous_year["award_rate_pct"]
        year_trend_text = (
            f"By submission year, the award rate moved from {int(previous_year['submission_year'])} "
            f"({previous_year['award_rate_pct']:.1f}%) to {int(latest_year['submission_year'])} "
            f"({latest_year['award_rate_pct']:.1f}%), a {delta_pp:+.1f} point change."
        )

    undergraduate_rows = level_year_summary[level_year_summary["group"].eq("Undergraduate")]
    level_trend_text = ""
    if len(undergraduate_rows) >= 2:
        previous_level = undergraduate_rows.iloc[-2]
        latest_level = undergraduate_rows.iloc[-1]
        level_trend_text = (
            f" Undergraduate award rate also moved from {previous_level['award_rate_pct']:.1f}% "
            f"in {int(previous_level['submission_year'])} to {latest_level['award_rate_pct']:.1f}% "
            f"in {int(latest_level['submission_year'])}."
        )

    answers.append(
        {
            "question": "What trends do you see across years / groups?",
            "executive": (
                "Year and group differences look structural enough to monitor, not dismiss as noise."
            ),
            "primary": year_trend_text,
            "secondary": (
                f"{level_trend_text.strip()} Across groups, graduate applicants and hardship-coded cases tend to sit above baseline, while several freshman and low-award school pathways sit below it."
            ).strip(),
            "risk": (
                "Do not normalize recurring year or group gaps without explanation; they can reflect budget shifts, intake mix, or review drift rather than a stable rule."
            ),
            "next_step": (
                "Review policy changes, budget pressure, and reviewer guidance in the years where rates moved most, then test whether the same groups stayed advantaged or disadvantaged."
            ),
        }
    )

    return answers


def build_questions_executive_answer(
    stable_factors: pd.DataFrame,
    dimension_summary: pd.DataFrame,
    counterintuitive_metrics: pd.DataFrame,
) -> str:
    """Summarize the questions tab in one directive line."""
    primary_labels = stable_factors["metric"].head(2).tolist()
    primary_text = (
        format_label_list(primary_labels, max_items=2)
        if primary_labels
        else "the cleanest financial need signals"
    )
    school_row = dimension_summary.loc[dimension_summary["dimension"].astype(str) == "School"]
    audit_dimension = (
        str(school_row.iloc[0]["dimension"])
        if not school_row.empty
        else str(dimension_summary.iloc[0]["dimension"])
        if not dimension_summary.empty
        else "contextual group gaps"
    )
    risk_metric = (
        "Dependents Count"
        if not counterintuitive_metrics.empty
        and counterintuitive_metrics["metric"].astype(str).eq("Dependents Count").any()
        else str(counterintuitive_metrics.iloc[0]["metric"])
        if not counterintuitive_metrics.empty
        else "unstable signals"
    )
    return (
        f"Use {primary_text} as the primary explanation, treat {audit_dimension} as an audit lens, and keep {risk_metric} out of automatic rules until it is explained."
    )


def build_questions_next_steps(
    stable_factors: pd.DataFrame,
    dimension_summary: pd.DataFrame,
    counterintuitive_metrics: pd.DataFrame,
    similar_profile_summary: dict[str, object],
) -> list[str]:
    """List the next actions implied by the questions tab."""
    steps: list[str] = []

    primary_labels = stable_factors["metric"].head(2).tolist()
    if primary_labels:
        steps.append(
            f"Keep {format_label_list(primary_labels, max_items=2)} as the first explanation reviewers must document."
        )

    school_row = dimension_summary.loc[dimension_summary["dimension"].astype(str) == "School"]
    audit_dimension = (
        str(school_row.iloc[0]["dimension"])
        if not school_row.empty
        else str(dimension_summary.iloc[0]["dimension"])
        if not dimension_summary.empty
        else None
    )
    if audit_dimension:
        steps.append(
            f"Run a targeted consistency audit on {audit_dimension} and the {int(similar_profile_summary.get('review_pair_count', 0)):,} closest matched awarded-vs-denied comparisons."
        )

    risk_metric = (
        "Dependents Count"
        if not counterintuitive_metrics.empty
        and counterintuitive_metrics["metric"].astype(str).eq("Dependents Count").any()
        else str(counterintuitive_metrics.iloc[0]["metric"])
        if not counterintuitive_metrics.empty
        else None
    )
    if risk_metric:
        steps.append(
            f"Quarantine {risk_metric} from scorecards or automation until the direction is justified with policy or cleaner data."
        )

    return steps[:3]


def render_questions_executive_panel(executive_answer: str) -> None:
    """Render a compact top-line answer for the questions tab."""
    st.markdown(
        f"""
        <section style="
            border-left: 6px solid {PRIMARY_COLOR};
            background: linear-gradient(180deg, #f0fdfa 0%, #ffffff 100%);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin: 0.15rem 0 1rem 0;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        ">
            <div style="
                color: {PRIMARY_COLOR};
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin-bottom: 0.3rem;
            ">Executive Answer</div>
            <p style="
                color: {TEXT_DARK};
                font-size: 1.02rem;
                line-height: 1.55;
                font-weight: 700;
                margin: 0;
            ">{html.escape(executive_answer)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_question_answer_card(answer: dict[str, str]) -> None:
    """Render one question with directive priorities and next action."""
    with st.container(border=True):
        st.markdown(f"**{answer['question']}**")
        st.write(answer["executive"])

        priority_cols = st.columns(3)
        priority_sections = [
            ("Primary", answer["primary"]),
            ("Secondary", answer["secondary"]),
            ("Risk", answer["risk"]),
        ]
        for column, (label, body) in zip(priority_cols, priority_sections):
            with column:
                st.markdown(f"**{label}**")
                st.write(body)

        st.markdown("**What to do next**")
        st.write(answer["next_step"])


def render_questions_next_steps(steps: list[str]) -> None:
    """Render the closing action section for the questions tab."""
    if not steps:
        return

    step_items_html = "".join(f"<li>{html.escape(step)}</li>" for step in steps)
    st.markdown(
        f"""
        <section style="
            border: 1px solid #d1fae5;
            background: #f8fafc;
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin-top: 1rem;
        ">
            <div style="
                color: {TEXT_DARK};
                font-size: 1rem;
                font-weight: 800;
                margin-bottom: 0.4rem;
            ">What to do next</div>
            <ul style="margin: 0; padding-left: 1rem; color: {TEXT_MUTED}; line-height: 1.55;">{step_items_html}</ul>
        </section>
        """,
        unsafe_allow_html=True,
    )


def build_factor_tldr(
    factor_scorecard: pd.DataFrame,
    dimension_summary: pd.DataFrame,
    counterintuitive_metrics: pd.DataFrame,
    high_segments: pd.DataFrame,
    low_segments: pd.DataFrame,
    review_summary: dict[str, object] | None,
    model_factor_names: list[str],
) -> list[str]:
    """Produce a concise set of recommendation-style priorities."""
    bullets: list[str] = []

    stable_factors = factor_scorecard[
        (~factor_scorecard["counterintuitive"]) & (factor_scorecard["coverage_pct"] >= 20)
    ].reset_index(drop=True)
    audit_factors = factor_scorecard[factor_scorecard["counterintuitive"]].reset_index(drop=True)

    if len(stable_factors) >= 2:
        bullets.append(
            f"Focus the main explanation and review rubric on {format_label_list(stable_factors['metric'].head(2).tolist(), max_items=2)}; they are the cleanest need-aligned signals with usable coverage."
        )
    elif not stable_factors.empty:
        top_factor = stable_factors.iloc[0]
        bullets.append(
            f"Focus the main explanation on {top_factor['metric']}; it is the cleanest stable factor on the page."
        )

    if not dimension_summary.empty:
        top_dimension = dimension_summary.iloc[0]
        bullets.append(
            f"Prioritize policy explanation for {top_dimension['dimension']}; it creates the widest internal award-rate spread on the page at {top_dimension['spread_pp']:.1f} percentage points."
        )

    dependents_risk = counterintuitive_metrics.loc[
        counterintuitive_metrics["metric"].astype(str) == "Dependents Count"
    ]
    chosen_risk = (
        dependents_risk.iloc[0]
        if not dependents_risk.empty
        else audit_factors.iloc[0]
        if not audit_factors.empty
        else None
    )
    if chosen_risk is not None:
        bullets.append(
            f"Audit {chosen_risk['metric']} before using it operationally; it is statistically strong but directionally suspicious."
        )

    if not low_segments.empty:
        weakest_segment = low_segments.iloc[0]
        bullets.append(
            f"Use {weakest_segment['group']} in {weakest_segment['dimension']} as a first fairness or consistency checkpoint because it sits {weakest_segment['lift_pp']:+.1f} percentage points below baseline."
        )

    if review_summary and int(review_summary.get("high_priority_count", 0)) > 0:
        bullets.append(
            f"Route the {int(review_summary['high_priority_count']):,} high-priority review cases through manual re-check rather than widening blanket rules."
        )
    elif model_factor_names:
        bullets.append(
            f"When the trained model is available, its top factors still cluster around {', '.join(model_factor_names[:3])}, which reinforces the main review focus."
        )

    return bullets


def format_count_table(
    dataframe: pd.DataFrame,
    *,
    probability_columns: list[str] | None = None,
    income_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Format common model-review tables for display."""
    probability_columns = probability_columns or []
    income_columns = income_columns or []
    display_df = dataframe.copy()

    for column in probability_columns:
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce").map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
            )

    for column in income_columns:
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce").map(
                lambda value: f"{value:,.0f}" if pd.notna(value) else "N/A"
            )

    return display_df


apply_page_styling()

if not DEFAULT_DATA_PATH.exists():
    st.error(f"CSV file not found: {DEFAULT_DATA_PATH}")
else:
    raw_df = load_csv(DEFAULT_DATA_PATH)

    try:
        labeled_df, overview = prepare_decision_analytics_frame(raw_df)
    except Exception as exc:
        st.error(f"Unable to prepare decision analytics: {exc}")
    else:
        numeric_drivers = summarize_numeric_drivers(labeled_df)
        counterintuitive_metrics = find_counterintuitive_metrics(numeric_drivers)
        factor_scorecard = build_factor_scorecard(numeric_drivers, counterintuitive_metrics)
        high_segments, low_segments = find_extreme_segments(labeled_df, top_n=10)

        similar_profile_pairs, similar_profile_summary = find_similar_profile_review_pairs(
            labeled_df,
            top_n=200,
        )
        mixed_profile_groups, mixed_profile_summary = summarize_mixed_profile_groups(
            labeled_df,
            top_n=20,
        )
        inconsistency_review_observations = build_inconsistency_review_observations(
            similar_profile_pairs,
            similar_profile_summary,
            mixed_profile_groups,
            mixed_profile_summary,
        )

        feature_importance_df = pd.DataFrame()
        importance_metadata: dict[str, str] = {}
        positive_drivers_df = pd.DataFrame()
        negative_drivers_df = pd.DataFrame()
        driver_metadata: dict[str, str] = {}
        metadata_payload: dict[str, object] | None = None
        scored_df = None
        scored_decision_df = pd.DataFrame()
        review_candidates_df = pd.DataFrame()
        low_confidence_awards_df = pd.DataFrame()
        review_summary = None
        model_factor_names: list[str] = []

        if DEFAULT_MODEL_PATH is not None and load_model is not None:
            try:
                model = load_model(DEFAULT_MODEL_PATH)
                scored_df = add_prediction_probabilities(raw_df, model)
                scored_decision_df = build_scored_decision_frame(scored_df)
                review_candidates_df, low_confidence_awards_df, review_summary = (
                    build_review_candidate_tables(scored_df)
                )
                feature_importance_df, importance_metadata = extract_top_feature_importances(
                    model,
                    top_n=15,
                )
                model_factor_names = feature_importance_df["feature"].head(5).tolist()
                try:
                    positive_drivers_df, negative_drivers_df, driver_metadata = (
                        extract_top_feature_drivers(model, top_n=10)
                    )
                except Exception:
                    positive_drivers_df = pd.DataFrame()
                    negative_drivers_df = pd.DataFrame()
                    driver_metadata = {}
            except Exception:
                scored_df = None
                scored_decision_df = pd.DataFrame()
                review_candidates_df = pd.DataFrame()
                low_confidence_awards_df = pd.DataFrame()
                review_summary = None
                model_factor_names = []
                feature_importance_df = pd.DataFrame()
                positive_drivers_df = pd.DataFrame()
                negative_drivers_df = pd.DataFrame()

        if feature_importance_df.empty and DEFAULT_MODEL_METADATA_PATH.exists():
            try:
                metadata_payload = load_json(DEFAULT_MODEL_METADATA_PATH)
            except Exception:
                metadata_payload = None
            else:
                saved_importance_bundle = build_saved_feature_importance_frame(
                    metadata_payload,
                    top_n=15,
                )
                if saved_importance_bundle is not None:
                    feature_importance_df, importance_metadata = saved_importance_bundle
                    model_factor_names = feature_importance_df["feature"].head(10).tolist()

        stable_factors = factor_scorecard[
            (~factor_scorecard["counterintuitive"]) & (factor_scorecard["coverage_pct"] >= 20)
        ].reset_index(drop=True)
        factor_priority_frame = build_factor_priority_frame(
            factor_scorecard,
            model_factor_names,
        )
        default_segment_universe = build_segment_universe(labeled_df, min_group_size=40)
        default_dimension_summary = build_dimension_summary(default_segment_universe)
        submission_year_summary = build_submission_year_summary(labeled_df)
        level_year_summary = build_year_group_summary(labeled_df, "level", min_group_size=30)
        key_question_answers = build_key_question_answers(
            factor_scorecard=factor_scorecard,
            stable_factors=stable_factors,
            dimension_summary=default_dimension_summary,
            high_segments=high_segments,
            low_segments=low_segments,
            counterintuitive_metrics=counterintuitive_metrics,
            similar_profile_summary=similar_profile_summary,
            mixed_profile_summary=mixed_profile_summary,
            year_summary=submission_year_summary,
            level_year_summary=level_year_summary,
        )
        questions_executive_answer = build_questions_executive_answer(
            stable_factors,
            default_dimension_summary,
            counterintuitive_metrics,
        )
        questions_next_steps = build_questions_next_steps(
            stable_factors,
            default_dimension_summary,
            counterintuitive_metrics,
            similar_profile_summary,
        )
        executive_takeaway = build_executive_takeaway(
            factor_scorecard,
            default_dimension_summary,
            counterintuitive_metrics,
        )
        executive_audit_risks = build_audit_risk_lines(
            counterintuitive_metrics,
            default_dimension_summary,
            review_summary,
        )
        business_implication_lines = build_business_implication_lines(
            stable_factors,
            default_dimension_summary,
            low_segments,
            review_summary,
        )
        tldr_lines = build_factor_tldr(
            factor_scorecard,
            default_dimension_summary,
            counterintuitive_metrics,
            high_segments,
            low_segments,
            review_summary,
            model_factor_names,
        )
        factor_influence_share = build_factor_influence_share(factor_scorecard)
        factor_priority_share = build_factor_priority_share(factor_priority_frame)
        factor_priority_count_share = build_factor_priority_count_share(factor_priority_frame)
        default_dimension_influence_share = build_dimension_influence_share(
            default_dimension_summary
        )
        model_influence_share = build_model_influence_share(feature_importance_df)
        factor_executive_takeaway = build_factor_executive_takeaway(
            factor_priority_frame,
            model_factor_names,
        )
        factor_decision_summary = build_factor_decision_summary(factor_priority_frame)
        factor_decision_table = build_factor_decision_table(factor_priority_frame)
        top_stable_factor = (
            stable_factors.iloc[0]["metric"]
            if not stable_factors.empty
            else factor_scorecard.iloc[0]["metric"]
            if not factor_scorecard.empty
            else "N/A"
        )
        strongest_high_segment = (
            high_segments.iloc[0]["group"] if not high_segments.empty else "N/A"
        )
        strongest_low_segment = (
            low_segments.iloc[0]["group"] if not low_segments.empty else "N/A"
        )
        most_uneven_dimension = (
            default_dimension_summary.iloc[0]["dimension"]
            if not default_dimension_summary.empty
            else "N/A"
        )
        primary_count = int(factor_priority_frame["priority"].eq("Primary").sum())
        audit_count = int(factor_priority_frame["priority"].eq("Audit").sum())
        total_factor_count = len(factor_priority_frame)
        top_dimension_spread = (
            float(default_dimension_summary.iloc[0]["spread_pp"])
            if not default_dimension_summary.empty
            else float("nan")
        )
        review_pair_count = int(similar_profile_summary.get("review_pair_count", 0))
        cleaned_rows = int(overview.get("rows_standard", 0))

        page_pills: list[tuple[str, str]] = [
            (f"{cleaned_rows:,} cleaned decisions", "primary"),
            ("Patterns over policy rules", "warning"),
        ]
        if review_pair_count > 0:
            page_pills.append((f"{review_pair_count:,} close-match reviews", "danger"))

        render_page_header(
            title="Decision Insights",
            description="Turns historical decision patterns into focus areas, audit triggers, and governance review priorities.",
            takeaway=executive_takeaway,
            kicker="Decision Review Workspace",
            pills=page_pills,
        )
        render_kpi_row(
            [
                {
                    "label": "Reliable Direct-Use Factors",
                    "value": f"{primary_count}/{total_factor_count}" if total_factor_count else "0/0",
                    "note": "Use these as the core explanation in manual review.",
                    "tone": "primary",
                },
                {
                    "label": "Top Governance Dimension",
                    "value": most_uneven_dimension,
                    "note": (
                        f"{top_dimension_spread:.1f} pp internal spread needs policy explanation."
                        if pd.notna(top_dimension_spread)
                        else "Largest contextual spread to monitor."
                    ),
                    "tone": "warning",
                },
                {
                    "label": "Closest Matched Pairs",
                    "value": f"{review_pair_count:,}",
                    "note": "These are the strongest case-level consistency checks.",
                    "tone": "danger",
                },
                {
                    "label": "Audit-Only Signals",
                    "value": f"{audit_count}/{total_factor_count}" if total_factor_count else "0/0",
                    "note": "Keep contradictory factors out of scoring and policy rules.",
                    "tone": "secondary",
                },
            ]
        )
        render_insight_action_panel(
            insight=(
                executive_audit_risks[0]
                if executive_audit_risks
                else f"{most_uneven_dimension} drives the biggest internal decision spread on the page."
            ),
            implication=(
                business_implication_lines[0]
                if business_implication_lines
                else "Keep financial need variables first, then route contradictions to manual audit."
            ),
        )

        section_options = [
            "Start Here",
            "Questions",
            "Factors",
            "Segments",
            "Consistency",
            "Model",
            "Appendix",
        ]
        selected_section = st.radio(
            "Decision insights section",
            options=section_options,
            horizontal=True,
            label_visibility="collapsed",
        )

        if selected_section == "Questions":
            st.subheader("Your Questions, Answered")
            st.caption("Direct answers first. Evidence comes right after.")
            render_questions_executive_panel(questions_executive_answer)

            for answer in key_question_answers:
                render_question_answer_card(answer)

            major_year_summary = submission_year_summary[
                submission_year_summary["is_major_year"]
            ].copy()
            if not major_year_summary.empty or not level_year_summary.empty:
                trend_cols = st.columns(2)

                with trend_cols[0]:
                    st.markdown("**Award Rate by Submission Year**")
                    if major_year_summary.empty:
                        st.info("Not enough year data for a stable trend view.")
                    else:
                        year_chart = (
                            alt.Chart(major_year_summary)
                            .mark_bar(color=PRIMARY_COLOR)
                            .encode(
                                x=alt.X("submission_year:O", title="Submission Year"),
                                y=alt.Y("award_rate_pct:Q", title="Award Rate (%)"),
                                tooltip=[
                                    alt.Tooltip("submission_year:O", title="Year"),
                                    alt.Tooltip("applications:Q", title="Applications"),
                                    alt.Tooltip("award_rate_pct:Q", title="Award Rate (%)", format=".1f"),
                                ],
                        )
                            .properties(height=260)
                        )
                        st.altair_chart(year_chart, width="stretch")
                        st.caption(
                            "Takeaway: year-to-year movement is large enough to control for timing before comparing decisions."
                        )

                with trend_cols[1]:
                    st.markdown("**Award Rate by Year and Academic Level**")
                    if level_year_summary.empty:
                        st.info("Not enough level-by-year data yet.")
                    else:
                        level_chart = (
                            alt.Chart(level_year_summary)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("submission_year:O", title="Submission Year"),
                                y=alt.Y("award_rate_pct:Q", title="Award Rate (%)"),
                                color=alt.Color("group:N", title="Academic Level"),
                                tooltip=[
                                    alt.Tooltip("submission_year:O", title="Year"),
                                    alt.Tooltip("group:N", title="Level"),
                                    alt.Tooltip("applications:Q", title="Applications"),
                                    alt.Tooltip("award_rate_pct:Q", title="Award Rate (%)", format=".1f"),
                                ],
                        )
                            .properties(height=260)
                        )
                        st.altair_chart(level_chart, width="stretch")
                        st.caption(
                            "Takeaway: if academic-level lines diverge by year, reviewer standards or applicant mix are not landing evenly across cohorts."
                        )

            render_questions_next_steps(questions_next_steps)

        if selected_section == "Start Here":
            st.subheader("Start Here: What Matters Most?")
            st.caption("Use this tab for the shortest path from evidence to decision focus.")

            if tldr_lines:
                st.markdown("**What Should Be Focused On**")
                for line in tldr_lines:
                    st.write(f"- {line}")

            tldr_donut_cols = st.columns(2)
            with tldr_donut_cols[0]:
                render_donut_chart(
                    factor_influence_share,
                    title="Factor Influence Share",
                    empty_message="No factor influence breakdown is available yet.",
                    value_title="Evidence Score",
                    summary_subject="factor influence",
                    subtitle="Each slice shows how much of the displayed factor evidence comes from that variable.",
                )
            with tldr_donut_cols[1]:
                render_donut_chart(
                    default_dimension_influence_share,
                    title="Dimension Influence Share",
                    empty_message="No dimension influence breakdown is available yet.",
                    value_title="Spread (pp)",
                    summary_subject="dimension influence",
                    subtitle="Each slice shows how much of the visible group-level variation comes from that dimension.",
                    value_format=".1f",
                )

            tldr_chart_cols = st.columns(2)

            with tldr_chart_cols[0]:
                st.markdown("**Top Factors by Combined Evidence**")
                if factor_scorecard.empty:
                    st.info("No factor scorecard is available yet.")
                else:
                    top_factor_chart_df = factor_scorecard.head(7).copy()
                    factor_chart = (
                        alt.Chart(top_factor_chart_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("evidence_score:Q", title="Evidence Score"),
                            y=alt.Y(
                                "metric:N",
                                title=None,
                                sort=alt.SortField(field="evidence_score", order="descending"),
                            ),
                            color=alt.Color(
                                "factor_status:N",
                                title=None,
                                scale=alt.Scale(
                                    domain=["Need-aligned", "Important but audit"],
                                    range=[PRIMARY_COLOR, WARNING_COLOR],
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip("metric:N", title="Factor"),
                                alt.Tooltip("separation_score:Q", title="Signal Strength", format=".2f"),
                                alt.Tooltip("coverage_pct:Q", title="Coverage (%)", format=".1f"),
                                alt.Tooltip("evidence_score:Q", title="Evidence Score", format=".3f"),
                                alt.Tooltip("factor_status:N", title="Interpretation"),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(factor_chart, width="stretch")
                    render_chart_guide(
                        what="This ranks the strongest numeric factors after combining two things: how much the factor separates awarded and denied cases, and how often that factor is actually filled in.",
                        simple_rule="Longer bars mean stronger overall evidence. Green means a cleaner need-based story. Orange means important but suspicious.",
                        positive_signal="Long green bars near the top mean the factor is both useful and believable.",
                        negative_signal="Short bars mean weak evidence. Orange bars mean the factor may matter statistically but should be audited before trusting it.",
                        watch_for="A factor can look powerful but still be weak if only a small share of applications reported it.",
                    )
                    render_chart_insights(
                        [
                            "Total Parent Income and Estimated Property Value deserve the most weight because they combine real separation with usable coverage.",
                            "Dependents Count leads the raw ranking, but its direction is suspicious enough that it should be audited before it shapes policy or scoring.",
                        ]
                    )

            with tldr_chart_cols[1]:
                st.markdown("**Which Dimensions Move Outcomes Most?**")
                if default_dimension_summary.empty:
                    st.info("No grouped segment summary is available.")
                else:
                    dimension_chart = (
                        alt.Chart(default_dimension_summary)
                        .mark_bar(color=PRIMARY_COLOR)
                        .encode(
                            x=alt.X("spread_pp:Q", title="Within-Dimension Spread (percentage points)"),
                            y=alt.Y(
                                "dimension:N",
                                title=None,
                                sort=alt.SortField(field="spread_pp", order="descending"),
                            ),
                            tooltip=[
                                alt.Tooltip("dimension:N", title="Dimension"),
                                alt.Tooltip("groups:Q", title="Eligible Groups"),
                                alt.Tooltip("avg_abs_lift_pp:Q", title="Average Absolute Lift", format=".1f"),
                                alt.Tooltip("max_positive_lift_pp:Q", title="Best Lift", format="+.1f"),
                                alt.Tooltip("max_negative_lift_pp:Q", title="Worst Lift", format="+.1f"),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(dimension_chart, width="stretch")
                    render_chart_guide(
                        what="This compares broad dimensions such as school, nationality, or special circumstances to see which one changes award outcomes the most inside that dimension.",
                        simple_rule="Longer bars mean that dimension creates bigger differences in award rates.",
                        positive_signal="A long bar means that dimension really changes who gets aid in the historical data.",
                        negative_signal="A short bar means that dimension is more stable and changes outcomes less.",
                        watch_for="This shows what historically moved decisions, not what should move decisions.",
                    )
                    render_chart_insights(
                        [
                            "School, nationality, and special circumstances create the widest outcome gaps, so contextual segmentation is materially shaping decisions alongside need.",
                            "School should be treated as an audit dimension, not just a descriptive one, because large school-based gaps need a clear policy rationale.",
                        ]
                    )

        if selected_section == "Factors":
            st.subheader("Factor Analysis")
            st.caption(
                "Use this tab to decide which numeric factors should lead explanation, which only support context, and which should stay in audit."
            )
            render_factor_executive_panel(factor_executive_takeaway)
            st.markdown(f"**{factor_decision_summary}**")

            primary_count = int(factor_priority_frame["priority"].eq("Primary").sum())
            secondary_count = int(factor_priority_frame["priority"].eq("Secondary").sum())
            audit_count = int(factor_priority_frame["priority"].eq("Audit").sum())
            total_factor_count = len(factor_priority_frame)
            model_aligned_count = (
                int(
                    factor_priority_frame["model_alignment"].eq("Aligned").sum()
                )
                if not factor_priority_frame.empty
                else 0
            )

            factor_metric_cols = st.columns(4)
            with factor_metric_cols[0]:
                st.metric("Reliable for Direct Use", f"{primary_count}/{total_factor_count}")
                st.caption(
                    "Only the top financial signals are strong enough to drive decisions directly; the rest need caution or audit."
                )
            with factor_metric_cols[1]:
                st.metric("Secondary With Caution", f"{secondary_count}/{total_factor_count}")
                st.caption(
                    "These factors can support a case explanation, but they are too weak or incomplete to lead the rule."
                )
            with factor_metric_cols[2]:
                st.metric("Audit-Only Factors", f"{audit_count}/{total_factor_count}")
                st.caption(
                    "Counterintuitive factors should trigger review, not policy."
                )
            with factor_metric_cols[3]:
                st.metric(
                    "Primary Themes Reinforced",
                    f"{model_aligned_count}"
                    if model_factor_names
                    else "N/A",
                )
                st.caption(
                    "Saved model behavior is used only as a cross-check on the factor story, not as proof of causation."
                )

            st.markdown("**Priority Hierarchy**")
            priority_cols = st.columns(3)
            priority_specs = [
                (
                    "Primary",
                    "Use",
                    "Lead with these in reviewer guidance and decision explanations.",
                ),
                (
                    "Secondary",
                    "Support",
                    "Keep these as supporting context only.",
                ),
                (
                    "Audit",
                    "Audit",
                    "Treat these as warning flags, not rules.",
                ),
            ]
            for column, (priority_label, action_label, summary_text) in zip(priority_cols, priority_specs):
                with column:
                    with st.container(border=True):
                        st.markdown(f"**{priority_label}**")
                        st.caption(summary_text)
                        priority_rows = factor_priority_frame[
                            factor_priority_frame["priority"].eq(priority_label)
                        ]
                        if priority_rows.empty:
                            st.write("No factors currently sit in this tier.")
                        else:
                            for _, row in priority_rows.iterrows():
                                factor_label = (
                                    f"⚠ {row['metric']}"
                                    if bool(row.get("is_key_contradiction", False))
                                    else str(row["metric"])
                                )
                                st.write(f"{factor_label} ({action_label})")
                                st.caption(row["priority_reason"])

            st.markdown("**Evidence Charts**")
            factor_donut_cols = st.columns(3)
            with factor_donut_cols[0]:
                render_donut_chart(
                    factor_influence_share,
                    title="Evidence Share by Factor",
                    empty_message="No factor evidence split is available yet.",
                    value_title="Evidence Score",
                    summary_subject="factor evidence",
                    subtitle="This shows where the strongest factor evidence is concentrated.",
                )
            with factor_donut_cols[1]:
                render_donut_chart(
                    factor_priority_share,
                    title="Evidence by Priority",
                    empty_message="No factor-priority split is available yet.",
                    value_title="Evidence Score",
                    summary_subject="factor evidence",
                    subtitle="This shows whether the evidence pool is actually reliable or mostly caution and audit material.",
                    color_domain=["Primary", "Secondary", "Audit"],
                    color_range=[
                        FACTOR_PRIORITY_COLORS["Primary"],
                        FACTOR_PRIORITY_COLORS["Secondary"],
                        FACTOR_PRIORITY_COLORS["Audit"],
                    ],
                )
            with factor_donut_cols[2]:
                render_donut_chart(
                    factor_priority_count_share,
                    title="Factor Count by Priority",
                    empty_message="No factor-count split is available yet.",
                    value_title="Factor Count",
                    summary_subject="tracked factors",
                    subtitle="This shows how many tracked factors are actually usable versus cautionary.",
                    color_domain=["Primary", "Secondary", "Audit"],
                    color_range=[
                        FACTOR_PRIORITY_COLORS["Primary"],
                        FACTOR_PRIORITY_COLORS["Secondary"],
                        FACTOR_PRIORITY_COLORS["Audit"],
                    ],
                    value_format=".0f",
                    label_threshold_pct=0.0,
                )
            render_chart_insights(
                [
                    f"Usable evidence is concentrated in {format_label_list(factor_priority_frame.loc[factor_priority_frame['priority'].eq('Primary'), 'metric'].head(2).tolist(), max_items=2) if primary_count else 'no factor yet'}, not spread evenly across all tracked factors.",
                    f"Only {primary_count} of {total_factor_count} factors are reliable enough to use directly; the rest belong in caution or audit.",
                    (
                        "Saved model behavior also leans on related income and property themes, which supports keeping the core explanation financial first."
                        if model_factor_names
                        else "Model-importance reinforcement is unavailable, so this tab rests entirely on historical factor separation."
                    ),
                ],
                title="What the evidence mix means",
            )

            factor_chart_cols = st.columns(2)

            with factor_chart_cols[0]:
                st.markdown("**Signal vs Coverage**")
                if factor_priority_frame.empty:
                    st.info("No factor scorecard is available yet.")
                else:
                    scatter = (
                        alt.Chart(factor_priority_frame)
                        .mark_circle(opacity=0.88)
                        .encode(
                            x=alt.X("coverage_pct:Q", title="Coverage in Comparable Subset (%)"),
                            y=alt.Y("separation_score:Q", title="Signal Strength"),
                            size=alt.Size(
                                "evidence_score:Q",
                                title="Evidence Score",
                                scale=alt.Scale(range=[120, 1500]),
                            ),
                            color=alt.Color(
                                "priority:N",
                                title=None,
                                scale=alt.Scale(
                                    domain=["Primary", "Secondary", "Audit"],
                                    range=[
                                        FACTOR_PRIORITY_COLORS["Primary"],
                                        FACTOR_PRIORITY_COLORS["Secondary"],
                                        FACTOR_PRIORITY_COLORS["Audit"],
                                    ],
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip("metric:N", title="Factor"),
                                alt.Tooltip("priority:N", title="Priority"),
                                alt.Tooltip("coverage_pct:Q", title="Coverage (%)", format=".1f"),
                                alt.Tooltip("separation_score:Q", title="Signal Strength", format=".2f"),
                                alt.Tooltip("recommended_action:N", title="Action"),
                                alt.Tooltip("direction:N", title="Observed Direction"),
                            ],
                        )
                        .properties(height=360)
                    )
                    scatter_text = (
                        alt.Chart(factor_priority_frame)
                        .mark_text(align="left", dx=8, dy=-4, fontSize=11)
                        .encode(
                            x="coverage_pct:Q",
                            y="separation_score:Q",
                            text="metric:N",
                        )
                    )
                    coverage_rule = (
                        alt.Chart(pd.DataFrame({"x": [25]}))
                        .mark_rule(color="#94a3b8", strokeDash=[5, 4])
                        .encode(x="x:Q")
                    )
                    signal_rule = (
                        alt.Chart(pd.DataFrame({"y": [0.25]}))
                        .mark_rule(color="#cbd5e1", strokeDash=[5, 4])
                        .encode(y="y:Q")
                    )
                    st.altair_chart(
                        alt.layer(coverage_rule, signal_rule, scatter, scatter_text),
                        width="stretch",
                    )
                    st.caption(
                        "Action map: upper-right green factors are use candidates, blue factors stay supporting only, and orange factors stay in audit even when the raw signal looks strong."
                    )
                    render_chart_insights(
                        [
                            "Use only the upper-right primary factors for direct reviewer guidance because they are both strong and sufficiently observed.",
                            "Ignore or de-prioritize thin or weak secondary factors for broad rules; they can support context but should not decide outcomes.",
                            "Dependents Count is the clearest audit case: it is statistically strong, fully observed, and still points the wrong way.",
                        ],
                        title="Action",
                    )

            with factor_chart_cols[1]:
                st.markdown("**Median Gap by Factor**")
                if factor_priority_frame.empty:
                    st.info("No relative median gaps are available.")
                else:
                    relative_gap_df = factor_priority_frame.dropna(subset=["relative_gap_pct"]).copy()
                    relative_gap_chart = (
                        alt.Chart(relative_gap_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "relative_gap_pct:Q",
                                title="Awarded vs Denied Median Gap (%)",
                            ),
                            y=alt.Y(
                                "metric:N",
                                title=None,
                                sort=alt.SortField(field="relative_gap_pct", order="ascending"),
                            ),
                            color=alt.Color(
                                "priority:N",
                                title=None,
                                scale=alt.Scale(
                                    domain=["Primary", "Secondary", "Audit"],
                                    range=[
                                        FACTOR_PRIORITY_COLORS["Primary"],
                                        FACTOR_PRIORITY_COLORS["Secondary"],
                                        FACTOR_PRIORITY_COLORS["Audit"],
                                    ],
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip("metric:N", title="Factor"),
                                alt.Tooltip("priority:N", title="Priority"),
                                alt.Tooltip("awarded_median:Q", title="Awarded Median", format=",.0f"),
                                alt.Tooltip("denied_median:Q", title="Denied Median", format=",.0f"),
                                alt.Tooltip("relative_gap_pct:Q", title="Relative Gap (%)", format="+.1f"),
                            ],
                        )
                        .properties(height=360)
                    )
                    baseline = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#9ca3af").encode(
                        x="x:Q"
                    )
                    st.altair_chart(alt.layer(relative_gap_chart, baseline), width="stretch")
                    st.caption(
                        "Interpretation: left of zero means awarded cases had lower typical values. That supports need for income and asset variables; on burden variables, the same left-side pattern can signal contradiction instead."
                    )
                    render_chart_insights(
                        [
                            "Income and property sit on the need-aligned side of the chart, meaning awarded cases report materially lower typical values on those affordability proxies.",
                            "For burden variables such as dependents or sibling tuition, a left-side bar is the wrong direction, so the gap is real but not trustworthy as policy logic.",
                            "Read direction before strength: a large gap only helps if the story still makes sense.",
                        ],
                        title="Explanation",
                    )

            with st.expander("Open detailed factor profile"):
                st.caption(
                    "Use this only when you want the full distribution behind one factor's median gap."
                )
                available_factor_options = factor_priority_frame["metric"].tolist()
                selected_factor_label = (
                    st.selectbox(
                        "Profile one factor in detail",
                        options=available_factor_options,
                    )
                    if available_factor_options
                    else None
                )

                if selected_factor_label is None:
                    st.info("No factor is available for profiling.")
                else:
                    selected_factor_row = factor_priority_frame.loc[
                        factor_priority_frame["metric"] == selected_factor_label
                    ].iloc[0]
                    factor_distribution_df = build_factor_distribution_frame(
                        labeled_df,
                        selected_factor_row["metric_column"],
                    )

                    if factor_distribution_df.empty:
                        st.info("The selected factor does not have enough valid records for distribution analysis.")
                    else:
                        distribution_chart = (
                            alt.Chart(factor_distribution_df)
                            .mark_boxplot(size=45)
                            .encode(
                                x=alt.X("decision_label:N", title="Decision"),
                                y=alt.Y(
                                    f"{selected_factor_row['metric_column']}:Q",
                                    title=selected_factor_label,
                                ),
                                color=alt.Color(
                                    "decision_label:N",
                                    legend=None,
                                    scale=alt.Scale(
                                        domain=["Awarded", "Denied"],
                                        range=[PRIMARY_COLOR, NEGATIVE_COLOR],
                                    ),
                                ),
                            )
                            .properties(height=360)
                        )
                        st.altair_chart(distribution_chart, width="stretch")
                        render_chart_insights(
                            [
                                f"{selected_factor_label} has {selected_factor_row['coverage_pct']:.1f}% coverage and a separation score of {selected_factor_row['separation_score']:.2f}, so the observed split is measurable.",
                                f"Its current use tier is {selected_factor_row['priority'].lower()}: {selected_factor_row['recommended_action'].lower()}",
                            ],
                            title="Detailed read",
                        )

            st.markdown("**Counterintuitive Factors**")
            st.warning("Counterintuitive factors are review triggers, not policy rules.")
            if counterintuitive_metrics.empty:
                st.info("No counterintuitive factor directions were detected in the strongest signals.")
            else:
                key_contradiction = factor_priority_frame.loc[
                    factor_priority_frame["is_key_contradiction"]
                ]
                if not key_contradiction.empty:
                    contradiction_row = key_contradiction.iloc[0]
                    st.markdown(
                        (
                            f"**⚠ Key contradiction: {contradiction_row['metric']}** is one of the strongest raw signals "
                            f"({contradiction_row['separation_score']:.2f}) with {contradiction_row['coverage_pct']:.1f}% coverage, "
                            "but it moves in the wrong direction for a simple need-based story."
                        )
                    )
                counter_chart_df = factor_priority_frame[
                    factor_priority_frame["priority"].eq("Audit")
                ].copy()
                counter_chart_df["contradiction_flag"] = np.where(
                    counter_chart_df["is_key_contradiction"],
                    "Key contradiction",
                    "Audit signal",
                )
                counter_chart = (
                    alt.Chart(counter_chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("evidence_score:Q", title="Evidence Score"),
                        y=alt.Y(
                            "metric:N",
                            title=None,
                            sort=alt.SortField(field="evidence_score", order="descending"),
                        ),
                        color=alt.Color(
                            "contradiction_flag:N",
                            title=None,
                            scale=alt.Scale(
                                domain=["Key contradiction", "Audit signal"],
                                range=[NEGATIVE_COLOR, WARNING_COLOR],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("metric:N", title="Factor"),
                            alt.Tooltip("coverage_pct:Q", title="Coverage (%)", format=".1f"),
                            alt.Tooltip("separation_score:Q", title="Signal Strength", format=".2f"),
                            alt.Tooltip("direction:N", title="Observed Direction"),
                            alt.Tooltip("recommended_action:N", title="Action"),
                        ],
                    )
                    .properties(height=240)
                )
                st.altair_chart(counter_chart, width="stretch")
                render_chart_insights(
                    [
                        "Dependents Count is the clearest contradiction: it is the strongest raw separator on the page, but awarded cases show fewer dependents than denied cases.",
                        f"{format_label_list(counter_chart_df['metric'].tolist(), max_items=3)} should be treated as audit prompts that force a case review, not as factors to reward or penalize automatically.",
                        "If a factor contradicts the need story, escalate it for policy or data-quality review before it touches automation.",
                    ],
                    title="What the contradiction means",
                )

            st.markdown("**Prioritized Decision Table**")
            st.caption("Sorted from direct-use factors to review-only triggers.")
            if factor_decision_table.empty:
                st.info("No factor decision table is available yet.")
            else:
                st.dataframe(
                    style_factor_decision_table(factor_decision_table),
                    width="stretch",
                    hide_index=True,
                )

        if selected_section == "Segments":
            st.subheader("Segment Analysis")
            st.caption(
                "Use this tab to decide exactly which groups need review first, which groups only need monitoring, and which small pockets can be ignored for now."
            )

            control_cols = st.columns([2, 1])
            with control_cols[0]:
                selected_dimension_label = st.selectbox(
                    "Compare award patterns by",
                    options=list(GROUP_DIMENSIONS.keys()),
                )
            with control_cols[1]:
                min_group_size = st.slider(
                    "Minimum applications per group",
                    min_value=20,
                    max_value=150,
                    value=40,
                    step=5,
                )

            segment_universe = build_segment_universe(
                labeled_df,
                min_group_size=min_group_size,
            )
            dimension_summary = build_dimension_summary(segment_universe)
            segment_extremes = build_segment_extreme_frame(segment_universe)

            if segment_universe.empty:
                st.info(
                    "No groups met the current minimum sample-size threshold. Lower the threshold to explore smaller segments."
                )
            else:
                fallback_factor_labels = (
                    factor_priority_frame.loc[
                        factor_priority_frame["action_label"].isin(["Use", "Support"]),
                        "metric",
                    ]
                    .head(2)
                    .tolist()
                )
                if not fallback_factor_labels:
                    fallback_factor_labels = factor_scorecard["metric"].head(2).tolist()

                segment_driver_lookup = build_segment_driver_lookup(
                    labeled_df,
                    segment_extremes,
                    fallback_factor_labels,
                )
                segment_decision_summary = build_segment_decision_summary(segment_extremes)
                segment_risk_lines = build_segment_risk_lines(
                    segment_extremes,
                    segment_driver_lookup,
                )
                segment_priority_table = build_segment_priority_table(
                    segment_extremes,
                    segment_driver_lookup,
                )

                segment_metric_cols = st.columns(4)
                with segment_metric_cols[0]:
                    st.metric("Eligible Segments", f"{len(segment_universe):,}")
                with segment_metric_cols[1]:
                    st.metric(
                        "Dimensions With Variation",
                        f"{segment_universe['dimension'].nunique():,}",
                    )
                with segment_metric_cols[2]:
                    st.metric(
                        "Largest Positive Lift",
                        f"{segment_universe['lift_pp'].max():+.1f} pp",
                    )
                with segment_metric_cols[3]:
                    st.metric(
                        "Largest Negative Lift",
                        f"{segment_universe['lift_pp'].min():+.1f} pp",
                    )

                st.markdown(f"**{segment_decision_summary}**")
                if segment_risk_lines:
                    st.warning("Potential fairness / consistency review segments:")
                    for line in segment_risk_lines:
                        st.write(f"- {line}")

                selected_group_column = GROUP_DIMENSIONS[selected_dimension_label]
                group_patterns = summarize_group_patterns(
                    labeled_df,
                    selected_group_column,
                    min_group_size=min_group_size,
                )
                selected_group_share = build_group_application_share(group_patterns)
                segment_scatter_frame = build_segment_scatter_frame(
                    segment_universe,
                    min_group_size=min_group_size,
                )
                segment_annotation_frame = build_segment_annotation_frame(segment_universe)

                segment_donut_cols = st.columns(2)
                with segment_donut_cols[0]:
                    render_donut_chart(
                        build_dimension_influence_share(dimension_summary),
                        title="Dimension Share of Variation",
                        empty_message="No dimension-level split is available.",
                        value_title="Spread (pp)",
                        summary_subject="group variation",
                        subtitle="This percentage split shows which broad dimensions explain the most visible difference in award rates.",
                        value_format=".1f",
                    )
                with segment_donut_cols[1]:
                    render_donut_chart(
                        selected_group_share,
                        title=f"{selected_dimension_label} Sample Share",
                        empty_message="No group-share view is available for the current filter.",
                        value_title="Applications",
                        summary_subject="applications in this dimension",
                        subtitle="This shows which groups dominate the currently selected dimension before you interpret the lift charts.",
                        value_format=".0f",
                    )

                segment_chart_cols = st.columns(2)

                with segment_chart_cols[0]:
                    st.markdown("**All Segments: Lift vs Sample Share**")
                    bubble_chart = (
                        alt.Chart(segment_scatter_frame)
                        .mark_circle()
                        .encode(
                            x=alt.X(
                                "share_of_sample_pct:Q",
                                title="Share of Comparable Sample (%)",
                            ),
                            y=alt.Y(
                                "lift_pp:Q",
                                title="Award Rate Lift vs Overall (pp)",
                            ),
                            size=alt.Size(
                                "applications:Q",
                                title="Applications",
                                scale=alt.Scale(range=[100, 1400]),
                            ),
                            color=alt.Color(
                                "lift_direction:N",
                                title=None,
                                scale=alt.Scale(
                                    domain=["Above baseline", "Below baseline"],
                                    range=[PRIMARY_COLOR, NEGATIVE_COLOR],
                                ),
                            ),
                            opacity=alt.Opacity(
                                "opacity_level:Q",
                                scale=None,
                                legend=None,
                            ),
                            tooltip=[
                                alt.Tooltip("dimension:N", title="Dimension"),
                                alt.Tooltip("group:N", title="Group"),
                                alt.Tooltip("applications:Q", title="Applications"),
                                alt.Tooltip("award_rate_pct:Q", title="Award Rate (%)", format=".1f"),
                                alt.Tooltip("lift_pp:Q", title="Lift (pp)", format="+.1f"),
                                alt.Tooltip("share_of_sample_pct:Q", title="Sample Share (%)", format=".1f"),
                            ],
                        )
                        .properties(height=380)
                    )
                    baseline = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                        color="#9ca3af",
                        strokeDash=[6, 6],
                    ).encode(y="y:Q")
                    scatter_labels = (
                        alt.Chart(
                            segment_scatter_frame[segment_scatter_frame["label_text"].astype(str) != ""]
                        )
                        .mark_text(align="left", dx=8, dy=-4, fontSize=11, color=TEXT_DARK)
                        .encode(
                            x="share_of_sample_pct:Q",
                            y="lift_pp:Q",
                            text="label_text:N",
                        )
                    )
                    quadrant_labels = (
                        alt.Chart(segment_annotation_frame)
                        .mark_text(fontSize=12, fontWeight="bold", color=TEXT_MUTED)
                        .encode(
                            x="x:Q",
                            y="y:Q",
                            text="label:N",
                        )
                    )
                    st.altair_chart(
                        alt.layer(baseline, bubble_chart, scatter_labels, quadrant_labels),
                        width="stretch",
                    )
                    st.caption(
                        "Small groups are intentionally faded so the chart emphasizes segments that move decisions at scale."
                    )
                    render_chart_insights(
                        [
                            "The segments that matter most are the large negative groups sitting far below zero, because they combine real volume with materially worse outcomes.",
                            "Action: review the bottom-right groups first and ignore tiny near-zero pockets unless another tab shows a separate risk.",
                        ]
                    )

                with segment_chart_cols[1]:
                    st.markdown("**Highest-Impact Segments Across The Dataset**")
                    if segment_extremes.empty:
                        st.info("No extreme segment frame is available.")
                    else:
                        extreme_chart_frame = segment_extremes.copy()
                        extreme_chart_frame["display_label"] = np.where(
                            extreme_chart_frame["action_tag"].eq("Review priority"),
                            "Review | " + extreme_chart_frame["segment_label"],
                            "Benchmark | " + extreme_chart_frame["segment_label"],
                        )
                        extreme_chart_frame["driven_by"] = extreme_chart_frame.apply(
                            lambda row: segment_driver_lookup.get(
                                (str(row["dimension"]), str(row["group"])),
                                "Needs factor review",
                            ),
                            axis=1,
                        )
                        extremes_chart = (
                            alt.Chart(extreme_chart_frame)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "lift_pp:Q",
                                    title="Award Rate Lift vs Overall (pp)",
                                ),
                                y=alt.Y(
                                    "display_label:N",
                                    title=None,
                                    sort=extreme_chart_frame["display_label"].tolist(),
                                ),
                                color=alt.Color(
                                    "action_tag:N",
                                    title=None,
                                    scale=alt.Scale(
                                        domain=["Review priority", "Benchmark"],
                                        range=[NEGATIVE_COLOR, PRIMARY_COLOR],
                                    ),
                                ),
                                tooltip=[
                                    alt.Tooltip("action_tag:N", title="Action"),
                                    alt.Tooltip("dimension:N", title="Dimension"),
                                    alt.Tooltip("group:N", title="Group"),
                                    alt.Tooltip("applications:Q", title="Applications"),
                                    alt.Tooltip("award_rate_pct:Q", title="Award Rate (%)", format=".1f"),
                                    alt.Tooltip("lift_pp:Q", title="Lift (pp)", format="+.1f"),
                                    alt.Tooltip("impact_score:Q", title="Impact x Size", format=".0f"),
                                    alt.Tooltip("driven_by:N", title="Driven by"),
                                ],
                            )
                            .properties(height=380)
                        )
                        st.altair_chart(extremes_chart, width="stretch")
                        render_chart_insights(
                            [
                                "These rows are sorted by impact x size, so the top review bars are the segments most worth investigating first, not just the noisiest extremes.",
                                "Action: treat red rows as review priorities and green rows as benchmarks that still need a policy explanation.",
                            ]
                        )

                st.markdown("**Segment Action List**")
                st.caption("This turns the biggest segment gaps into a short investigation queue.")
                st.dataframe(
                    style_segment_priority_table(segment_priority_table),
                    width="stretch",
                    hide_index=True,
                )

                st.markdown("**Which Dimensions Are Most Uneven?**")
                if dimension_summary.empty:
                    st.info("No dimension summary is available.")
                else:
                    dimension_spread_chart = (
                        alt.Chart(dimension_summary)
                        .mark_bar(color=PRIMARY_COLOR)
                        .encode(
                            x=alt.X("spread_pp:Q", title="Within-Dimension Spread (pp)"),
                            y=alt.Y(
                                "dimension:N",
                                title=None,
                                sort=alt.SortField(field="spread_pp", order="descending"),
                            ),
                            tooltip=[
                                alt.Tooltip("dimension:N", title="Dimension"),
                                alt.Tooltip("groups:Q", title="Eligible Groups"),
                                alt.Tooltip("avg_abs_lift_pp:Q", title="Average Absolute Lift", format=".1f"),
                                alt.Tooltip("spread_pp:Q", title="Spread (pp)", format=".1f"),
                            ],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(dimension_spread_chart, width="stretch")
                    render_chart_insights(
                        [
                            "Long-spread dimensions are where committee standards are landing least evenly across subgroups.",
                            "Action: put the widest dimension gaps, especially school, into recurring governance or fairness review.",
                        ]
                    )

                st.markdown(f"**Detailed View: {selected_dimension_label}**")
                if group_patterns.empty:
                    st.info(
                        "No groups met the current minimum sample-size threshold for the selected dimension."
                    )
                else:
                    group_patterns = group_patterns.copy()
                    group_patterns["dimension"] = selected_dimension_label
                    group_patterns["segment_label"] = group_patterns["group"].astype(str)
                    group_patterns["abs_lift_pp"] = group_patterns["lift_pp"].abs()
                    group_patterns["lift_direction"] = np.where(
                        group_patterns["lift_pp"] >= 0,
                        "Above baseline",
                        "Below baseline",
                    )
                    if len(group_patterns) <= 12:
                        chart_groups = group_patterns.sort_values("lift_pp", ascending=False)
                    else:
                        chart_groups = (
                            pd.concat(
                                [group_patterns.head(6), group_patterns.tail(6)],
                                ignore_index=True,
                            )
                            .drop_duplicates(subset=["group"])
                            .sort_values("lift_pp", ascending=False)
                        )

                    selected_lookup_frame = pd.concat(
                        [
                            group_patterns[group_patterns["lift_pp"] < 0]
                            .sort_values(["applications", "abs_lift_pp"], ascending=[False, False])
                            .head(3),
                            group_patterns[group_patterns["lift_pp"] > 0]
                            .sort_values(["applications", "abs_lift_pp"], ascending=[False, False])
                            .head(2),
                        ],
                        ignore_index=True,
                    ).drop_duplicates(subset=["group"])
                    selected_driver_lookup = build_segment_driver_lookup(
                        labeled_df,
                        selected_lookup_frame,
                        fallback_factor_labels,
                    )
                    selected_dimension_summary = build_selected_dimension_summary(
                        group_patterns,
                        selected_driver_lookup,
                        dimension=selected_dimension_label,
                    )
                    st.caption(selected_dimension_summary)

                    detail_cols = st.columns(2)
                    with detail_cols[0]:
                        comparison_chart = (
                            alt.Chart(chart_groups)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "lift_pp:Q",
                                    title="Award Rate Lift vs Overall (percentage points)",
                                ),
                                y=alt.Y(
                                    "group:N",
                                    title=None,
                                    sort=alt.SortField(field="lift_pp", order="descending"),
                                ),
                                color=alt.Color(
                                    "lift_direction:N",
                                    title=None,
                                    scale=alt.Scale(
                                        domain=["Above baseline", "Below baseline"],
                                        range=[PRIMARY_COLOR, NEGATIVE_COLOR],
                                    ),
                                ),
                                tooltip=[
                                    alt.Tooltip("group:N", title="Group"),
                                    alt.Tooltip("applications:Q", title="Applications"),
                                    alt.Tooltip("award_rate_pct:Q", title="Award Rate (%)", format=".1f"),
                                    alt.Tooltip("lift_pp:Q", title="Lift (pp)", format="+.1f"),
                                ],
                            )
                            .properties(height=360)
                        )
                        st.altair_chart(comparison_chart, width="stretch")
                        render_chart_insights(
                            [
                                f"In {selected_dimension_label}, the lowest bars are the review queue and the highest bars are the benchmark groups the committee should be able to explain.",
                                "Action: investigate the lowest bars first, then use the highest bars as calibration checks rather than automatic approvals.",
                            ]
                        )

                    with detail_cols[1]:
                        scatter_chart = (
                            alt.Chart(group_patterns)
                            .mark_circle(opacity=0.85)
                            .encode(
                                x=alt.X("applications:Q", title="Applications"),
                                y=alt.Y("lift_pp:Q", title="Award Rate Lift vs Overall (pp)"),
                                size=alt.Size(
                                    "abs_lift_pp:Q",
                                    title="Absolute Lift (pp)",
                                    scale=alt.Scale(range=[100, 1000]),
                                ),
                                color=alt.Color(
                                    "lift_direction:N",
                                    title=None,
                                    scale=alt.Scale(
                                        domain=["Above baseline", "Below baseline"],
                                        range=[PRIMARY_COLOR, NEGATIVE_COLOR],
                                    ),
                                ),
                                tooltip=[
                                    alt.Tooltip("group:N", title="Group"),
                                    alt.Tooltip("applications:Q", title="Applications"),
                                    alt.Tooltip("lift_pp:Q", title="Lift (pp)", format="+.1f"),
                                    alt.Tooltip("award_rate_pct:Q", title="Award Rate (%)", format=".1f"),
                                ],
                            )
                            .properties(height=360)
                        )
                        overall_line = alt.Chart(pd.DataFrame({"lift_pp": [0]})).mark_rule(
                            color="#9ca3af",
                            strokeDash=[6, 6],
                        ).encode(
                            y="lift_pp:Q"
                        )
                        st.altair_chart(alt.layer(scatter_chart, overall_line), width="stretch")
                        render_chart_insights(
                            [
                                "This view combines sample size and lift directly, so the biggest circles below zero are the groups with the clearest operational downside.",
                                "Action: use this chart to separate high-volume review segments from small outliers you can safely deprioritize.",
                            ]
                        )

        if selected_section == "Consistency":
            st.subheader("Consistency Review")
            st.caption(
                "Use this tab to isolate the strongest case-level mismatches and the biggest repeated profile inconsistencies."
            )
            consistency_action_summary = build_consistency_action_summary(
                similar_profile_summary,
                mixed_profile_summary,
            )
            consistency_why_lines = build_consistency_why_lines(similar_profile_summary)
            review_reason_summary = build_review_reason_summary(similar_profile_pairs)
            consistency_reason_headline = build_consistency_reason_headline(review_reason_summary)
            consistency_key_insight = build_consistency_key_insight(review_reason_summary)
            st.markdown(f"**{consistency_action_summary}**")
            st.markdown("**Why this matters**")
            for line in consistency_why_lines:
                st.write(f"- {line}")
            st.info(
                "Matched differences indicate inconsistency, not necessarily bias; further review is required."
            )

            consistency_metric_cols = st.columns(4)
            with consistency_metric_cols[0]:
                st.metric(
                    "Denied Cases With Awarded Match",
                    f"{int(similar_profile_summary.get('applications_with_match', 0)):,}",
                )
            with consistency_metric_cols[1]:
                st.metric(
                    "Closest Review Pairs",
                    f"{int(similar_profile_summary.get('review_pair_count', 0)):,}",
                )
            with consistency_metric_cols[2]:
                st.metric(
                    "Matched Comparison Pool",
                    f"{int(similar_profile_summary.get('display_pair_count', len(similar_profile_pairs))):,}",
                )
            with consistency_metric_cols[3]:
                st.metric(
                    "Mixed Profile Groups",
                    f"{int(mixed_profile_summary.get('mixed_group_count', 0)):,}",
                )

            consistency_chart_cols = st.columns(2)

            with consistency_chart_cols[0]:
                st.markdown("**Distribution of Profile Distances**")
                if similar_profile_pairs.empty:
                    st.info("No similar-profile review pairs are available.")
                else:
                    review_threshold = float(similar_profile_summary.get("review_threshold", np.nan))
                    hist_peak = int(np.histogram(similar_profile_pairs["distance"], bins=20)[0].max())
                    distance_hist = (
                        alt.Chart(similar_profile_pairs)
                        .mark_bar(color=PRIMARY_COLOR)
                        .encode(
                            x=alt.X(
                                "distance:Q",
                                title="Profile Distance",
                                bin=alt.Bin(maxbins=20),
                            ),
                            y=alt.Y("count():Q", title="Matched Pairs"),
                            tooltip=[alt.Tooltip("count():Q", title="Pairs")],
                        )
                        .properties(height=320)
                    )
                    threshold_line = alt.Chart(
                        pd.DataFrame(
                            {
                                "threshold": [review_threshold]
                            }
                        )
                    ).mark_rule(color=NEGATIVE_COLOR, strokeDash=[6, 6]).encode(x="threshold:Q")
                    threshold_label = alt.Chart(
                        pd.DataFrame(
                            {
                                "threshold": [review_threshold],
                                "pairs": [max(1, hist_peak * 0.92)],
                                "label": [f"Review cutoff (distance <= {review_threshold:.3f})"],
                            }
                        )
                    ).mark_text(
                        align="left",
                        dx=8,
                        dy=-8,
                        color=NEGATIVE_COLOR,
                        fontSize=11,
                        fontWeight="bold",
                    ).encode(
                        x="threshold:Q",
                        y="pairs:Q",
                        text="label:N",
                    )
                    st.altair_chart(
                        alt.layer(distance_hist, threshold_line, threshold_label),
                        width="stretch",
                    )
                    render_chart_insights(
                        [
                            "Pairs left of the cutoff are the strongest apples-to-apples mismatches because the denied and awarded applications look genuinely alike.",
                            "Action: use the cutoff to define the manual-review queue instead of reviewing the full comparison pool.",
                        ]
                    )

            with consistency_chart_cols[1]:
                st.markdown("**Top Drivers of Inconsistency**")
                if review_reason_summary.empty:
                    st.info("No review-note summary is available.")
                else:
                    top_reason_summary = review_reason_summary.head(3).copy()
                    reason_chart = (
                        alt.Chart(top_reason_summary)
                        .mark_bar(color=WARNING_COLOR)
                        .encode(
                            x=alt.X("count:Q", title="Pairs"),
                            y=alt.Y(
                                "reason_label:N",
                                title=None,
                                sort=alt.SortField(field="count", order="descending"),
                            ),
                            tooltip=[
                                alt.Tooltip("reason_label:N", title="Reason"),
                                alt.Tooltip("count:Q", title="Pairs"),
                                alt.Tooltip("share_pct:Q", title="Share (%)", format=".1f"),
                            ],
                        )
                        .properties(height=320)
                    )
                    reason_labels = (
                        alt.Chart(top_reason_summary)
                        .mark_text(
                            align="left",
                            dx=8,
                            color=TEXT_DARK,
                            fontSize=11,
                            fontWeight="bold",
                        )
                        .encode(
                            x="count:Q",
                            y=alt.Y(
                                "reason_label:N",
                                sort=alt.SortField(field="count", order="descending"),
                            ),
                            text="share_label:N",
                        )
                    )
                    st.altair_chart(alt.layer(reason_chart, reason_labels), width="stretch")
                    st.caption(consistency_reason_headline)
                    st.markdown(f"**{consistency_key_insight}**")
                    render_chart_insights(
                        [
                            "Application type and special-circumstance differences dominate the matched-pair review queue.",
                            "Action: clean up contextual rules before changing financial thresholds, because the biggest mismatches are not primarily financial.",
                        ],
                        title="Insight and Action",
                    )

            if not similar_profile_pairs.empty:
                st.markdown("**Match Quality: Shared Features vs Distance**")
                pair_scatter_frame = similar_profile_pairs.copy()
                pair_scatter_frame["reason_label"] = pair_scatter_frame["review_note"].map(
                    shorten_review_reason
                )
                pair_scatter = (
                    alt.Chart(pair_scatter_frame)
                    .mark_circle(opacity=0.85)
                    .encode(
                        x=alt.X("shared_features:Q", title="Shared Financial Features"),
                        y=alt.Y("distance:Q", title="Profile Distance"),
                        color=alt.Color("reason_label:N", title="Review Note"),
                        tooltip=[
                            alt.Tooltip("denied_record_id:Q", title="Denied Record"),
                            alt.Tooltip("awarded_record_id:Q", title="Awarded Match"),
                            alt.Tooltip("shared_features:Q", title="Shared Features"),
                            alt.Tooltip("distance:Q", title="Distance", format=".3f"),
                            alt.Tooltip("reason_label:N", title="Reason"),
                        ],
                    )
                    .properties(height=320)
                )
                scatter_threshold = alt.Chart(
                    pd.DataFrame(
                        {"threshold": [float(similar_profile_summary.get("review_threshold", np.nan))]}
                    )
                ).mark_rule(color=NEGATIVE_COLOR, strokeDash=[6, 6]).encode(y="threshold:Q")
                st.altair_chart(alt.layer(pair_scatter, scatter_threshold), width="stretch")
                render_chart_insights(
                    [
                        "Bottom-right pairs are the strongest matched-case evidence because they combine more shared features with lower distance.",
                        "Action: prioritize the lowest-distance pairs inside the high-shared-feature area when assigning manual review.",
                    ],
                    title="Insight and Action",
                )

            st.markdown("**Mixed Profile Groups**")
            mixed_group_long = build_mixed_group_long(mixed_profile_groups, top_n=10)
            if mixed_group_long.empty:
                st.info("No mixed profile groups are available.")
            else:
                mixed_group_chart = (
                    alt.Chart(mixed_group_long)
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", title="Applications"),
                        y=alt.Y("group_id:N", title="Profile Group"),
                        color=alt.Color(
                            "decision_label:N",
                            title=None,
                            scale=alt.Scale(
                                domain=["Awarded", "Denied"],
                                range=[PRIMARY_COLOR, NEGATIVE_COLOR],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("group_id:N", title="Group"),
                            alt.Tooltip("profile:N", title="Profile"),
                            alt.Tooltip("decision_label:N", title="Decision"),
                            alt.Tooltip("count:Q", title="Applications"),
                            alt.Tooltip("award_rate_pct:Q", title="Award Rate (%)", format=".1f"),
                            alt.Tooltip("review_score:Q", title="Review Score"),
                        ],
                    )
                    .properties(height=340)
                )
                st.altair_chart(mixed_group_chart, width="stretch")
                render_chart_insights(
                    [
                        "Mixed-profile groups show that inconsistency is not just pair-level noise; some similar profile buckets still contain both awards and denials.",
                        "Action: audit the most balanced, highest-volume mixed groups before expanding or defending any policy rule.",
                    ],
                    title="Insight and Action",
                )

            if not similar_profile_pairs.empty:
                pair_display = similar_profile_pairs.copy()
                pair_display["Risk Level"] = build_matched_pair_risk_level(
                    similar_profile_pairs,
                    float(similar_profile_summary.get("review_threshold", np.nan)),
                )
                pair_display["distance"] = pair_display["distance"].map(
                    lambda value: f"{value:.3f}"
                )
                pair_display["shared_features"] = pair_display["shared_features"].map(
                    lambda value: f"{int(value)}"
                )
                pair_display["review_note"] = pair_display["review_note"].map(shorten_review_reason)
                numeric_pair_columns = [
                    "denied_total_parent_income",
                    "awarded_total_parent_income",
                    "denied_properties_total_estimated_value",
                    "awarded_properties_total_estimated_value",
                    "denied_dependents_count",
                    "awarded_dependents_count",
                    "denied_total_siblings",
                    "awarded_total_siblings",
                ]
                for column in numeric_pair_columns:
                    if column in pair_display.columns:
                        pair_display[column] = pd.to_numeric(
                            pair_display[column],
                            errors="coerce",
                        ).map(lambda value: f"{value:,.0f}" if pd.notna(value) else "N/A")

                with st.expander("Open matched-case review table"):
                    st.caption("Top 5 most similar pairs are highlighted in red.")
                    matched_case_display = pair_display[
                        [
                            "Risk Level",
                            "denied_record_id",
                            "awarded_record_id",
                            "level",
                            "distance",
                            "shared_features",
                            "denied_school",
                            "awarded_school",
                            "denied_total_parent_income",
                            "awarded_total_parent_income",
                            "denied_properties_total_estimated_value",
                            "awarded_properties_total_estimated_value",
                            "review_note",
                        ]
                    ].rename(
                        columns={
                            "denied_record_id": "Denied Record ID",
                            "awarded_record_id": "Awarded Match ID",
                            "level": "Level",
                            "distance": "Profile Distance",
                            "shared_features": "Shared Features",
                            "denied_school": "Denied School",
                            "awarded_school": "Awarded School",
                            "denied_total_parent_income": "Denied Parent Income",
                            "awarded_total_parent_income": "Awarded Parent Income",
                            "denied_properties_total_estimated_value": "Denied Property Value",
                            "awarded_properties_total_estimated_value": "Awarded Property Value",
                            "review_note": "Why Review",
                        }
                    )
                    st.dataframe(
                        style_matched_pair_table(matched_case_display),
                        width="stretch",
                        hide_index=True,
                    )

            if not mixed_profile_groups.empty:
                mixed_group_display = mixed_profile_groups.copy()
                mixed_group_display["award_rate_pct"] = mixed_group_display[
                    "award_rate_pct"
                ].map(lambda value: f"{value:.1f}")
                mixed_group_display["balance_gap_pp"] = mixed_group_display[
                    "balance_gap_pp"
                ].map(lambda value: f"{value:.1f}")
                with st.expander("Open mixed-profile group table"):
                    st.dataframe(
                        mixed_group_display[
                            [
                                "Level",
                                "Parent Income",
                                "Property Value",
                                "Dependents",
                                "Siblings",
                                "Other Siblings Tuition",
                                "applications",
                                "awarded",
                                "denied",
                                "award_rate_pct",
                                "review_score",
                            ]
                        ].rename(
                            columns={
                                "applications": "Applications",
                                "awarded": "Awarded",
                                "denied": "Denied",
                                "award_rate_pct": "Award Rate (%)",
                                "review_score": "Review Score",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

        if selected_section == "Model":
            st.subheader("Model Readback")
            st.caption(
                "Use this tab to check whether saved model behavior supports the same decision story. Read it as ranking support, not as an automation rule."
            )

            if feature_importance_df.empty and positive_drivers_df.empty and negative_drivers_df.empty:
                st.info("Saved model evidence is not available yet.")
            else:
                top_model_feature = (
                    prettify_model_feature_label(str(feature_importance_df.iloc[0]["feature"]))
                    if not feature_importance_df.empty
                    else "No saved top feature"
                )
                high_priority_count = int(review_summary.get("high_priority_count", 0)) if review_summary else 0
                low_confidence_count = (
                    int(review_summary.get("low_confidence_awards_count", 0))
                    if review_summary
                    else 0
                )
                aligned_factor_count = (
                    int(factor_priority_frame["model_alignment"].eq("Aligned").sum())
                    if not factor_priority_frame.empty
                    else 0
                )
                model_takeaway = (
                    f"Saved model behavior stays closest to {top_stable_factor.lower()}-style evidence, which supports keeping the explanation financial first."
                    if not feature_importance_df.empty
                    else "Saved model artifacts are too thin to draw a strong reinforcement story."
                )
                st.markdown(f"**{model_takeaway}**")

                render_kpi_row(
                    [
                        {
                            "label": "Top Saved Feature",
                            "value": top_model_feature,
                            "note": "Highest-ranked signal in saved model importance.",
                            "tone": "primary",
                        },
                        {
                            "label": "Aligned Direct-Use Factors",
                            "value": (
                                f"{aligned_factor_count}/{primary_count}"
                                if primary_count
                                else "0/0"
                            ),
                            "note": "How much the model agrees with the cleanest factor story.",
                            "tone": "secondary",
                        },
                        {
                            "label": "High-Priority Review Queue",
                            "value": f"{high_priority_count:,}",
                            "note": "Cases the model rates highly despite non-award history.",
                            "tone": "warning",
                        },
                        {
                            "label": "Low-Confidence Awards",
                            "value": f"{low_confidence_count:,}",
                            "note": "Awards that still look atypical under the saved model.",
                            "tone": "danger",
                        },
                    ]
                )

                model_cols = st.columns(2)
                with model_cols[0]:
                    render_donut_chart(
                        model_influence_share,
                        title="Model Influence Share",
                        empty_message="No saved model influence breakdown is available yet.",
                        value_title="Importance",
                        summary_subject="saved model influence",
                        subtitle="This shows where the model places most of its visible weight.",
                    )
                with model_cols[1]:
                    st.markdown("**Top Feature Importances**")
                    if feature_importance_df.empty:
                        st.info("Feature importances are unavailable.")
                    else:
                        importance_chart_df = feature_importance_df.head(8).copy()
                        importance_chart_df = add_top_n_flag(
                            importance_chart_df,
                            value_column="importance",
                            top_n=3,
                        )
                        importance_chart = (
                            alt.Chart(importance_chart_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("importance:Q", title=importance_metadata.get("value_label", "Importance")),
                                y=alt.Y(
                                    "feature:N",
                                    title=None,
                                    sort=alt.SortField(field="importance", order="descending"),
                                ),
                                color=build_color_condition(),
                                tooltip=[
                                    alt.Tooltip("feature:N", title="Feature"),
                                    alt.Tooltip("importance:Q", title="Importance", format=".3f"),
                                ],
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(importance_chart, width="stretch")
                        if importance_metadata.get("caption"):
                            st.caption(importance_metadata["caption"])
                        st.caption(
                            "Takeaway: the model still concentrates on a small set of features, so agreement with the factor tab matters more than raw complexity."
                        )

                if not review_candidates_df.empty or not low_confidence_awards_df.empty:
                    review_cols = st.columns(2)
                    with review_cols[0]:
                        st.markdown("**Highest-Support Review Candidates**")
                        if review_candidates_df.empty:
                            st.info("No saved review-candidate table is available.")
                        else:
                            st.dataframe(
                                format_count_table(
                                    review_candidates_df.head(8),
                                    probability_columns=["predicted_award_probability"],
                                    income_columns=["total_parent_income"],
                                ),
                                width="stretch",
                                hide_index=True,
                            )
                    with review_cols[1]:
                        st.markdown("**Low-Confidence Awards**")
                        if low_confidence_awards_df.empty:
                            st.info("No low-confidence award table is available.")
                        else:
                            st.dataframe(
                                format_count_table(
                                    low_confidence_awards_df.head(8),
                                    probability_columns=["predicted_award_probability"],
                                    income_columns=["total_parent_income"],
                                ),
                                width="stretch",
                                hide_index=True,
                            )

        if selected_section == "Appendix":
            st.subheader("Appendix and Final Recommendation")

            st.markdown(
                f"""
                <section style="
                    border: 1px solid rgba(15, 118, 110, 0.18);
                    border-left: 7px solid {PRIMARY_COLOR};
                    background: linear-gradient(180deg, #f0fdfa 0%, #ffffff 100%);
                    border-radius: 20px;
                    padding: 1.05rem 1.15rem;
                    margin: 0.15rem 0 1rem 0;
                    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
                ">
                    <div style="
                        color: {PRIMARY_COLOR};
                        font-size: 0.78rem;
                        font-weight: 800;
                        letter-spacing: 0.08em;
                        text-transform: uppercase;
                        margin-bottom: 0.35rem;
                    ">Final Recommendation</div>
                    <p style="
                        color: {TEXT_DARK};
                        font-size: 1.02rem;
                        line-height: 1.55;
                        font-weight: 700;
                        margin: 0 0 0.55rem 0;
                    ">Use income + property as primary drivers, treat dependents/sibling-related variables as audit triggers, and prioritize school-based disparities for governance review.</p>
                    <p style="
                        color: {TEXT_MUTED};
                        font-size: 0.93rem;
                        line-height: 1.55;
                        margin: 0;
                    ">These findings reflect historical patterns and should not be interpreted as causal or normative policy rules.</p>
                </section>
                """,
                unsafe_allow_html=True,
            )

            primary_appendix_factors = factor_priority_frame.loc[
                factor_priority_frame["action_label"].eq("Use"),
                "metric",
            ].head(2).tolist()
            support_appendix_factors = factor_priority_frame.loc[
                factor_priority_frame["action_label"].eq("Support"),
                "metric",
            ].head(2).tolist()
            audit_appendix_factors = factor_priority_frame.loc[
                factor_priority_frame["action_label"].eq("Audit"),
                "metric",
            ].head(2).tolist()
            top_dimension_name = (
                str(default_dimension_summary.iloc[0]["dimension"])
                if not default_dimension_summary.empty
                else "school-based segments"
            )

            st.markdown("**Decision Drivers**")
            if primary_appendix_factors:
                st.write(
                    f"- Use {format_label_list(primary_appendix_factors, max_items=2)} directly; they combine the cleanest separation, usable coverage, and the strongest alignment with historical award behavior."
                )
            if support_appendix_factors or audit_appendix_factors:
                support_text = (
                    format_label_list(support_appendix_factors, max_items=2)
                    if support_appendix_factors
                    else "the remaining weaker signals"
                )
                audit_text = (
                    format_label_list(audit_appendix_factors, max_items=2)
                    if audit_appendix_factors
                    else "contradictory household-context signals"
                )
                st.write(
                    f"- Use {support_text} only as support, and keep {audit_text} in audit mode because the direction is too weak or too contradictory for policy rules."
                )

            st.markdown("**Governance Risks**")
            governance_line = (
                f"Prioritize {top_dimension_name.lower()} for governance review; it creates the widest internal outcome spread on the page, and contradictory cases still require manual consistency audit."
            )
            st.write(f"- {governance_line}")

            if not factor_decision_table.empty:
                st.markdown("**Decision Driver Scorecard**")
                st.dataframe(
                    style_appendix_scorecard(factor_decision_table),
                    width="stretch",
                    hide_index=True,
                )

            appendix_cols = st.columns(2)
            with appendix_cols[0]:
                st.markdown("**Strongest High-Award Segments**")
                if high_segments.empty:
                    st.info("No high-award segments met the minimum sample threshold.")
                else:
                    st.caption(
                        "High-award segments reflect strong need signals or exceptional-case patterns, not guaranteed eligibility rules."
                    )
                    high_segment_display = high_segments.copy()
                    high_segment_display["award_rate_pct"] = high_segment_display[
                        "award_rate_pct"
                    ].map(lambda value: f"{value:.1f}")
                    high_segment_display["lift_pp"] = high_segment_display["lift_pp"].map(
                        lambda value: f"{value:+.1f}"
                    )
                    st.dataframe(
                        high_segment_display[
                            ["dimension", "group", "applications", "award_rate_pct", "lift_pp"]
                        ].rename(
                            columns={
                                "dimension": "Dimension",
                                "group": "Segment",
                                "applications": "Applications",
                                "award_rate_pct": "Award Rate (%)",
                                "lift_pp": "Lift (pp)",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )
            with appendix_cols[1]:
                st.markdown("**Strongest Low-Award Segments**")
                if low_segments.empty:
                    st.info("No low-award segments met the minimum sample threshold.")
                else:
                    st.caption(
                        "Low-award segments indicate review targets and governance questions, not automatic unfairness."
                    )
                    low_segment_display = low_segments.copy()
                    low_segment_display["award_rate_pct"] = low_segment_display[
                        "award_rate_pct"
                    ].map(lambda value: f"{value:.1f}")
                    low_segment_display["lift_pp"] = low_segment_display["lift_pp"].map(
                        lambda value: f"{value:+.1f}"
                    )
                    st.dataframe(
                        low_segment_display[
                            ["dimension", "group", "applications", "award_rate_pct", "lift_pp"]
                        ].rename(
                            columns={
                                "dimension": "Dimension",
                                "group": "Segment",
                                "applications": "Applications",
                                "award_rate_pct": "Award Rate (%)",
                                "lift_pp": "Lift (pp)",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

            st.markdown("**Overall Conclusion**")
            st.write("- Financial need signals are consistent and dominant, with income and property providing the cleanest basis for explanation.")
            st.write(
                f"- Contextual factors introduce variability and risk, especially when {top_dimension_name.lower()} creates wide internal outcome gaps or household-count signals point in a contradictory direction."
            )
            st.write(
                "- Audit mechanisms are required before operational use, because contradictory factors and matched-case inconsistencies indicate review needs that a static rule would miss."
            )
