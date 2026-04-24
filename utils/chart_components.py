from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, Sequence

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils.decision_labels import normalize_decision_series
from utils.decision_support import (
    STATUS_AUDIT,
    STATUS_CAUTION,
    STATUS_HIGH_RISK,
    STATUS_MONITOR_ONLY,
    STATUS_NOT_PRODUCTION,
    STATUS_RELIABLE,
    get_reliability_label,
    status_tone,
)
from utils.ui import (
    ACCENT_COLOR,
    CARD_COLOR,
    DANGER_COLOR,
    MUTED_TEXT_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    TEXT_COLOR,
    badge_color_for_status,
)


STATUS_COLORS = {
    STATUS_RELIABLE: PRIMARY_COLOR,
    STATUS_CAUTION: ACCENT_COLOR,
    STATUS_AUDIT: "#F97316",
    STATUS_HIGH_RISK: DANGER_COLOR,
    STATUS_NOT_PRODUCTION: "#64748B",
    STATUS_MONITOR_ONLY: "#64748B",
    "Audit / Risk": "#F97316",
    "Not production-ready": "#64748B",
}

ACTION_COLORS = {
    "Auto approve": PRIMARY_COLOR,
    "Auto reject / likely deny": SECONDARY_COLOR,
    "Committee review": ACCENT_COLOR,
    "Deep manual review": "#F97316",
    "Block automation due to data/fairness risk": DANGER_COLOR,
    "Manual review": ACCENT_COLOR,
    "Auto award": PRIMARY_COLOR,
    "Auto zero": SECONDARY_COLOR,
    "force_manual_review": "#F97316",
    "tighten_automation_thresholds": ACCENT_COLOR,
    "monitor_only": "#64748B",
    "monitor_only_small_sample": "#94A3B8",
}


@contextmanager
def _bordered_container():
    try:
        container = st.container(border=True)
    except TypeError:
        container = st.container()
    with container:
        yield


def _safe_pct(value: float | int | None, *, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    numeric = float(value)
    if numeric > 1:
        return f"{numeric:.{digits}f}%"
    return f"{numeric * 100:.{digits}f}%"


def compact_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    numeric = float(value)
    sign = "-" if numeric < 0 else ""
    numeric = abs(numeric)
    if numeric >= 1_000_000:
        return f"{sign}{numeric / 1_000_000:.1f}M"
    if numeric >= 1_000:
        return f"{sign}{numeric / 1_000:.1f}K"
    if numeric.is_integer():
        return f"{sign}{int(numeric):,}"
    return f"{sign}{numeric:,.1f}"


def readable_feature_name(feature: object) -> str:
    text = str(feature).replace("parsed_", "").replace("derived_", "").replace("inferred_", "")
    text = text.replace("qa_", "QA ").replace("_", " ")
    return " ".join(text.split()).title()


def assign_feature_group(feature: object) -> str:
    text = str(feature).lower()
    if any(token in text for token in ["income", "father", "mother", "gross", "net"]):
        return "Parent income"
    if any(token in text for token in ["propert", "asset", "car", "investment"]):
        return "Property/assets"
    if any(token in text for token in ["loan", "liabil", "debt", "balance"]):
        return "Loans/liabilities"
    if any(token in text for token in ["sibling", "depend", "household", "tuition"]):
        return "Household structure"
    if any(token in text for token in ["application", "track", "term", "year", "submission", "level"]):
        return "Application context"
    if any(token in text for token in ["school", "aub"]):
        return "School/context"
    if any(token in text for token in ["nationality", "citizenship"]):
        return "Nationality/citizenship"
    if any(token in text for token in ["qa", "risk", "quality", "issue", "missing"]):
        return "QA/data quality"
    if "missing" in text or "isna" in text or "indicator" in text:
        return "Missingness indicators"
    return "Other"


def _empty_chart_message(message: str = "No chartable data available.", *, height: int = 240) -> alt.Chart:
    return (
        alt.Chart(pd.DataFrame({"message": [message]}))
        .mark_text(color=MUTED_TEXT_COLOR, fontSize=14, fontWeight=600)
        .encode(text="message:N")
        .properties(height=height)
    )


def render_chart_card(
    *,
    title: str,
    chart: alt.Chart | alt.LayerChart | alt.VConcatChart | alt.HConcatChart | None = None,
    renderer: Callable[[], None] | None = None,
    subtitle: str | None = None,
    takeaway: str | None = None,
    caption: str | None = None,
    status: str | None = None,
    sample_size: int | None = None,
    coverage: float | str | None = None,
    warning: str | None = None,
    height: int | None = None,
) -> None:
    """Render a chart with a consistent executive card, metadata, and caption."""
    meta_parts: list[str] = []
    if sample_size is not None:
        meta_parts.append(f"n={sample_size:,}")
    if coverage is not None:
        coverage_text = coverage if isinstance(coverage, str) else _safe_pct(float(coverage))
        meta_parts.append(f"coverage={coverage_text}")
    meta_text = " | ".join(meta_parts)
    with _bordered_container():
        header_columns = st.columns([0.72, 0.28])
        with header_columns[0]:
            st.markdown(f"#### {title}")
            if subtitle:
                st.caption(subtitle)
        with header_columns[1]:
            if status:
                st.badge(status, color=badge_color_for_status(status))
            if meta_text:
                st.caption(meta_text)
        if warning:
            st.warning(warning)
        if takeaway:
            st.info(f"Takeaway: {takeaway}")
        if chart is not None:
            if height is not None:
                chart = chart.properties(height=height)
            st.altair_chart(chart, width="stretch")
        if renderer is not None:
            renderer()
        if caption:
            st.caption(caption)


def render_table_card(
    *,
    title: str,
    dataframe: pd.DataFrame,
    subtitle: str | None = None,
    takeaway: str | None = None,
    caption: str | None = None,
    status: str | None = None,
    sample_size: int | None = None,
) -> None:
    render_chart_card(
        title=title,
        subtitle=subtitle,
        takeaway=takeaway,
        caption=caption,
        status=status,
        sample_size=sample_size if sample_size is not None else len(dataframe),
        renderer=lambda: st.dataframe(dataframe, width="stretch", hide_index=True),
    )


def render_system_flow(nodes: Sequence[tuple[str, str, str]]) -> None:
    for start in range(0, len(nodes), 3):
        row = nodes[start : start + 3]
        columns = st.columns(len(row))
        for offset, (title, note, status) in enumerate(row):
            with columns[offset]:
                with st.container(border=True):
                    st.caption(f"Step {start + offset + 1}")
                    st.markdown(f"**{title}**")
                    st.write(note)
                    st.badge(status, color=badge_color_for_status(status))


def render_decision_grid(items: Sequence[tuple[str, str, str]]) -> None:
    columns = st.columns(2)
    for index, (title, note, status) in enumerate(items):
        with columns[index % 2]:
            render_chart_card(
                title=title,
                takeaway=note,
                status=status,
                caption="Positioning guidance for thesis and operational discussion.",
            )


def make_horizontal_bar(
    dataframe: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str | None = None,
    color: str | None = None,
    color_field: str | None = None,
    color_scale: alt.Scale | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    label_format: str = ",.0f",
    tooltip: list[alt.Tooltip] | None = None,
    height: int = 320,
) -> alt.LayerChart:
    frame = dataframe.copy()
    if frame.empty or x not in frame.columns or y not in frame.columns:
        return _empty_chart_message()
    frame[x] = pd.to_numeric(frame[x], errors="coerce").fillna(0)
    if frame[x].max() <= 0:
        return _empty_chart_message("No positive values available for this chart.")
    color_encoding = alt.value(color or PRIMARY_COLOR)
    if color_field:
        color_encoding = alt.Color(
            f"{color_field}:N",
            scale=color_scale,
            legend=alt.Legend(orient="bottom"),
        )
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title),
            y=alt.Y(
                f"{y}:N",
                title=y_title,
                sort=alt.SortField(field=x, order="descending"),
                axis=alt.Axis(labelLimit=280),
            ),
            color=color_encoding,
            tooltip=tooltip
            or [
                alt.Tooltip(f"{y}:N", title=y_title or y),
                alt.Tooltip(f"{x}:Q", title=x_title or x, format=label_format),
            ],
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(align="left", baseline="middle", dx=5, color=TEXT_COLOR, fontWeight=700)
        .encode(
            x=alt.X(f"{x}:Q"),
            y=alt.Y(f"{y}:N", sort=alt.SortField(field=x, order="descending")),
            text=alt.Text(f"{x}:Q", format=label_format),
        )
    )
    chart = alt.layer(bars, labels).properties(height=height)
    if title:
        chart = chart.properties(title=title)
    return chart


def build_pareto_frame(issue_frame: pd.DataFrame, *, count_column: str = "issue_rows") -> pd.DataFrame:
    if issue_frame.empty or count_column not in issue_frame.columns:
        return pd.DataFrame()
    frame = issue_frame.copy().sort_values(count_column, ascending=False, ignore_index=True)
    frame["cumulative_share"] = frame[count_column].cumsum() / max(frame[count_column].sum(), 1)
    return frame


def make_issue_pareto_chart(issue_frame: pd.DataFrame) -> alt.LayerChart:
    frame = build_pareto_frame(issue_frame)
    if frame.empty:
        return alt.Chart(pd.DataFrame()).mark_bar()
    bars = (
        alt.Chart(frame)
        .mark_bar(color=PRIMARY_COLOR, cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("issue_code:N", title=None, sort=None, axis=alt.Axis(labelAngle=-25, labelLimit=130)),
            y=alt.Y("issue_rows:Q", title="Issue rows"),
            tooltip=[
                alt.Tooltip("issue_code:N", title="Issue"),
                alt.Tooltip("issue_rows:Q", title="Issue rows"),
                alt.Tooltip("cumulative_share:Q", title="Cumulative share", format=".1%"),
            ],
        )
    )
    line = (
        alt.Chart(frame)
        .mark_line(point=True, color=DANGER_COLOR, strokeWidth=3)
        .encode(
            x=alt.X("issue_code:N", sort=None),
            y=alt.Y(
                "cumulative_share:Q",
                title="Cumulative share",
                axis=alt.Axis(format="%"),
            ),
        )
    )
    return alt.layer(bars, line).resolve_scale(y="independent").properties(height=330)


def make_coverage_reliability_chart(feature_frame: pd.DataFrame) -> alt.Chart:
    if feature_frame.empty:
        return alt.Chart(pd.DataFrame()).mark_point()
    frame = feature_frame.copy()
    frame["coverage_pct"] = frame["coverage"] * 100
    frame["issue_rate_pct"] = frame["issue_rate"] * 100
    return (
        alt.Chart(frame)
        .mark_circle(opacity=0.88, stroke=CARD_COLOR, strokeWidth=1.2)
        .encode(
            x=alt.X("coverage_pct:Q", title="Coverage %", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("issue_rate_pct:Q", title="Issue rate %"),
            size=alt.Size("sample_size:Q", title="Sample size", scale=alt.Scale(range=[90, 620])),
            color=alt.Color(
                "status:N",
                title="Reliability",
                scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values())),
            ),
            tooltip=[
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("sample_size:Q", title="Sample"),
                alt.Tooltip("coverage_pct:Q", title="Coverage", format=".1f"),
                alt.Tooltip("issue_rate_pct:Q", title="Issue rate", format=".1f"),
                alt.Tooltip("status:N", title="Status"),
            ],
        )
        .properties(height=330)
    )


def build_missingness_frame(
    dataframe: pd.DataFrame,
    fields: Sequence[tuple[str, str]],
    *,
    group_by: str | None = None,
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()
    if not group_by or group_by not in dataframe.columns:
        working = dataframe.copy()
        working["_group"] = "Portfolio"
        group_by = "_group"
    else:
        working = dataframe.copy()
        working[group_by] = working[group_by].astype("string").fillna("Missing").replace("", "Missing")

    rows: list[dict[str, object]] = []
    for group_value, group_frame in working.groupby(group_by, dropna=False):
        for field_group, column in fields:
            if column not in group_frame.columns:
                continue
            series = group_frame[column]
            if series.dtype == "O" or str(series.dtype).startswith("string"):
                missing = series.astype("string").fillna("").str.strip().eq("")
            else:
                missing = series.isna()
            rows.append(
                {
                    "group": str(group_value),
                    "field_group": field_group,
                    "field": column,
                    "records": int(len(group_frame)),
                    "missing_rate": float(missing.mean()) if len(group_frame) else 0.0,
                    "coverage": 1.0 - float(missing.mean()) if len(group_frame) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def make_missingness_heatmap(missingness_frame: pd.DataFrame) -> alt.Chart:
    if missingness_frame.empty:
        return alt.Chart(pd.DataFrame()).mark_rect()
    frame = missingness_frame.copy()
    frame["missing_pct"] = frame["missing_rate"] * 100
    return (
        alt.Chart(frame)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X("group:N", title=None, axis=alt.Axis(labelAngle=-25, labelLimit=110)),
            y=alt.Y("field_group:N", title=None),
            color=alt.Color(
                "missing_pct:Q",
                title="Missing %",
                scale=alt.Scale(range=["#ECFDF5", "#FFFBEB", "#FED7AA", "#DC2626"]),
            ),
            tooltip=[
                alt.Tooltip("field_group:N", title="Field group"),
                alt.Tooltip("group:N", title="Segment"),
                alt.Tooltip("records:Q", title="Records"),
                alt.Tooltip("missing_pct:Q", title="Missing %", format=".1f"),
            ],
        )
        .properties(height=320)
    )


def build_issue_cooccurrence_frame(
    issues_dataframe: pd.DataFrame,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    required = {"source_row_number", "issue_code"}
    if issues_dataframe.empty or not required.issubset(issues_dataframe.columns):
        return pd.DataFrame()
    top_codes = (
        issues_dataframe["issue_code"]
        .astype("string")
        .fillna("unknown")
        .str.strip()
        .value_counts()
        .head(top_n)
        .index
    )
    working = issues_dataframe[
        issues_dataframe["issue_code"].astype("string").isin(top_codes)
    ].copy()
    if working.empty:
        return pd.DataFrame()
    matrix = pd.crosstab(working["source_row_number"], working["issue_code"].astype("string")).gt(0).astype(int)
    cooccurrence = matrix.T.dot(matrix)
    cooccurrence.index = cooccurrence.index.rename("issue_a")
    cooccurrence.columns = cooccurrence.columns.rename("issue_b")
    rows = cooccurrence.stack().rename("cases").reset_index()
    rows.columns = ["issue_a", "issue_b", "cases"]
    rows = rows[rows["issue_a"].ne(rows["issue_b"])].copy()
    return rows.sort_values("cases", ascending=False, ignore_index=True)


def make_issue_cooccurrence_heatmap(cooccurrence_frame: pd.DataFrame) -> alt.Chart:
    if cooccurrence_frame.empty:
        return alt.Chart(pd.DataFrame()).mark_rect()
    return (
        alt.Chart(cooccurrence_frame)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X("issue_a:N", title=None, axis=alt.Axis(labelAngle=-35, labelLimit=120)),
            y=alt.Y("issue_b:N", title=None, axis=alt.Axis(labelLimit=160)),
            color=alt.Color("cases:Q", title="Cases", scale=alt.Scale(scheme="oranges")),
            tooltip=[
                alt.Tooltip("issue_a:N", title="Issue A"),
                alt.Tooltip("issue_b:N", title="Issue B"),
                alt.Tooltip("cases:Q", title="Cases"),
            ],
        )
        .properties(height=340)
    )


def build_financial_signal_frame(
    dataframe: pd.DataFrame,
    issues_dataframe: pd.DataFrame | None,
    fields: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()
    issue_counts: dict[str, int] = {}
    if issues_dataframe is not None and not issues_dataframe.empty and "field_name" in issues_dataframe.columns:
        issue_counts = issues_dataframe["field_name"].astype("string").value_counts().to_dict()

    outcome = normalize_decision_series(dataframe.get("parsed_decision", dataframe.get("decision")))
    awarded = outcome.eq("awarded")
    denied = outcome.eq("denied")
    rows: list[dict[str, object]] = []
    for label, column in fields:
        if column not in dataframe.columns:
            continue
        values = pd.to_numeric(dataframe[column], errors="coerce")
        sample_size = int(values.notna().sum())
        coverage = sample_size / max(len(dataframe), 1)
        issue_rows = int(issue_counts.get(column, 0))
        issue_rate = issue_rows / max(sample_size, 1)
        awarded_median = float(values[awarded].median()) if values[awarded].notna().any() else np.nan
        denied_median = float(values[denied].median()) if values[denied].notna().any() else np.nan
        separation = awarded_median - denied_median if pd.notna(awarded_median) and pd.notna(denied_median) else np.nan
        rows.append(
            {
                "signal": label,
                "column": column,
                "coverage": coverage,
                "missingness": 1.0 - coverage,
                "issue_rate": issue_rate,
                "sample_size": sample_size,
                "median_awarded": awarded_median,
                "median_denied": denied_median,
                "separation": separation,
                "status": get_reliability_label(coverage, issue_rate, sample_size),
            }
        )
    return pd.DataFrame(rows)


def make_financial_signal_chart(financial_frame: pd.DataFrame) -> alt.Chart:
    if financial_frame.empty:
        return alt.Chart(pd.DataFrame()).mark_bar()
    frame = financial_frame.copy()
    frame["coverage_pct"] = frame["coverage"] * 100
    frame["issue_rate_pct"] = frame["issue_rate"] * 100
    return (
        alt.Chart(frame)
        .mark_circle(opacity=0.9, stroke=CARD_COLOR, strokeWidth=1.2)
        .encode(
            x=alt.X("coverage_pct:Q", title="Coverage %", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("signal:N", title=None, sort=alt.SortField(field="coverage_pct", order="descending")),
            size=alt.Size("sample_size:Q", title="Sample size", scale=alt.Scale(range=[90, 620])),
            color=alt.Color(
                "status:N",
                title="Status",
                scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values())),
            ),
            tooltip=[
                alt.Tooltip("signal:N", title="Signal"),
                alt.Tooltip("sample_size:Q", title="Sample"),
                alt.Tooltip("coverage_pct:Q", title="Coverage %", format=".1f"),
                alt.Tooltip("issue_rate_pct:Q", title="Issue rate %", format=".1f"),
                alt.Tooltip("median_awarded:Q", title="Median awarded", format=",.0f"),
                alt.Tooltip("median_denied:Q", title="Median denied", format=",.0f"),
                alt.Tooltip("status:N", title="Status"),
            ],
        )
        .properties(height=330)
    )


def build_grouped_feature_importance(importance_frame: pd.DataFrame) -> pd.DataFrame:
    if importance_frame.empty or "feature" not in importance_frame.columns or "importance" not in importance_frame.columns:
        return pd.DataFrame()
    frame = importance_frame.copy()
    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").fillna(0)
    frame = frame[frame["importance"].gt(0)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["feature_label"] = frame["feature"].map(readable_feature_name)
    frame["feature_group"] = frame["feature"].map(assign_feature_group)
    grouped = (
        frame.groupby("feature_group", as_index=False)
        .agg(total_importance=("importance", "sum"), feature_count=("feature", "count"))
        .sort_values("total_importance", ascending=False, ignore_index=True)
    )
    return grouped


def make_feature_importance_chart(
    importance_frame: pd.DataFrame,
    *,
    top_n: int = 14,
    title: str | None = None,
) -> alt.LayerChart:
    if importance_frame.empty:
        return _empty_chart_message("No feature-importance data available.")
    frame = importance_frame.copy()
    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").fillna(0)
    frame = frame[frame["importance"].gt(0)].copy()
    if frame.empty:
        return _empty_chart_message("No positive feature importance available.")
    frame["feature_label"] = frame["feature"].map(readable_feature_name)
    top_frame = frame.sort_values("importance", ascending=False).head(top_n)
    return make_horizontal_bar(
        top_frame,
        x="importance",
        y="feature_label",
        title=title,
        color=PRIMARY_COLOR,
        x_title="Importance",
        label_format=".3f",
        tooltip=[
            alt.Tooltip("feature_label:N", title="Feature"),
            alt.Tooltip("feature:N", title="Raw name"),
            alt.Tooltip("importance:Q", title="Importance", format=".4f"),
        ],
        height=max(300, top_n * 28),
    )


def make_grouped_importance_chart(grouped_frame: pd.DataFrame) -> alt.LayerChart:
    if grouped_frame.empty:
        return _empty_chart_message("No grouped feature importance available.")
    grouped_frame = grouped_frame.copy()
    grouped_frame["total_importance"] = pd.to_numeric(grouped_frame["total_importance"], errors="coerce").fillna(0)
    grouped_frame = grouped_frame[grouped_frame["total_importance"].gt(0)].copy()
    if grouped_frame.empty:
        return _empty_chart_message("No positive grouped feature importance available.")
    return make_horizontal_bar(
        grouped_frame,
        x="total_importance",
        y="feature_group",
        color=SECONDARY_COLOR,
        x_title="Total importance",
        label_format=".3f",
        tooltip=[
            alt.Tooltip("feature_group:N", title="Feature group"),
            alt.Tooltip("total_importance:Q", title="Total importance", format=".4f"),
            alt.Tooltip("feature_count:Q", title="Features"),
        ],
        height=max(280, len(grouped_frame) * 32),
    )


def make_importance_reliability_quadrant(
    grouped_importance: pd.DataFrame,
    feature_reliability: pd.DataFrame,
) -> alt.LayerChart:
    if grouped_importance.empty:
        return alt.Chart(pd.DataFrame()).mark_circle()
    reliability_lookup = {
        "Parent income": "Parent Income",
        "Property/assets": "Property Value",
        "Loans/liabilities": "Remaining Loan Balance",
        "Household structure": "Dependents",
        "Application context": "Decision Label",
        "School/context": "School",
        "QA/data quality": "QA Quality Score",
    }
    frame = grouped_importance.copy()
    rel = feature_reliability.set_index("metric") if not feature_reliability.empty else pd.DataFrame()

    def lookup(metric: str, column: str, default: object) -> object:
        if rel.empty or metric not in rel.index or column not in rel.columns:
            return default
        return rel.loc[metric, column]

    frame["reliability_metric"] = frame["feature_group"].map(reliability_lookup).fillna("Decision Label")
    frame["coverage"] = frame["reliability_metric"].map(lambda metric: lookup(metric, "coverage", 0.5))
    frame["status"] = frame["reliability_metric"].map(lambda metric: lookup(metric, "status", STATUS_CAUTION))
    frame["coverage_pct"] = pd.to_numeric(frame["coverage"], errors="coerce").fillna(0.5) * 100
    points = (
        alt.Chart(frame)
        .mark_circle(opacity=0.9, stroke=CARD_COLOR, strokeWidth=1.2)
        .encode(
            x=alt.X("coverage_pct:Q", title="Reliability coverage %", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("total_importance:Q", title="Grouped model importance"),
            size=alt.Size("feature_count:Q", title="Feature count", scale=alt.Scale(range=[120, 650])),
            color=alt.Color(
                "status:N",
                title="Use status",
                scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values())),
            ),
            tooltip=[
                alt.Tooltip("feature_group:N", title="Group"),
                alt.Tooltip("total_importance:Q", title="Importance", format=".4f"),
                alt.Tooltip("coverage_pct:Q", title="Coverage %", format=".1f"),
                alt.Tooltip("status:N", title="Status"),
            ],
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(align="left", dx=9, color=TEXT_COLOR, fontSize=11)
        .encode(x="coverage_pct:Q", y="total_importance:Q", text="feature_group:N")
    )
    caution_line = alt.Chart(pd.DataFrame({"coverage_pct": [80]})).mark_rule(
        color=MUTED_TEXT_COLOR,
        strokeDash=[5, 5],
    ).encode(x="coverage_pct:Q")
    return alt.layer(points, labels, caution_line).properties(height=350)


def make_fairness_bubble_chart(fairness_frame: pd.DataFrame) -> alt.Chart:
    if fairness_frame.empty or "sample_size" not in fairness_frame.columns:
        return alt.Chart(pd.DataFrame()).mark_circle()
    frame = fairness_frame.copy()
    frame["sample_size"] = pd.to_numeric(frame["sample_size"], errors="coerce").fillna(0)
    gap_column = "mae_gap" if "mae_gap" in frame.columns else "bias_gap"
    frame[gap_column] = pd.to_numeric(frame[gap_column], errors="coerce").fillna(0)
    frame["abs_gap"] = frame[gap_column].abs()
    frame["label"] = frame.get("group_column", "").astype("string") + "=" + frame.get("group_value", "").astype("string")
    return (
        alt.Chart(frame)
        .mark_circle(opacity=0.84, stroke=CARD_COLOR, strokeWidth=1.2)
        .encode(
            x=alt.X("sample_size:Q", title="Subgroup sample size"),
            y=alt.Y(f"{gap_column}:Q", title=f"{gap_column.replace('_', ' ').title()} vs overall"),
            size=alt.Size("sample_size:Q", legend=None, scale=alt.Scale(range=[80, 720])),
            color=alt.Color(
                "recommended_action:N",
                title="Guardrail action",
                scale=alt.Scale(domain=list(ACTION_COLORS), range=list(ACTION_COLORS.values())),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Subgroup"),
                alt.Tooltip("sample_size:Q", title="Sample"),
                alt.Tooltip(f"{gap_column}:Q", title="Gap", format=".3f"),
                alt.Tooltip("recommended_action:N", title="Action"),
                alt.Tooltip("reason:N", title="Reason"),
            ],
        )
        .properties(height=340)
    )


def make_decision_priority_matrix(decision_actions: pd.DataFrame) -> alt.Chart:
    if decision_actions.empty or "model_score" not in decision_actions.columns:
        return alt.Chart(pd.DataFrame()).mark_circle()
    frame = decision_actions.copy()
    frame["model_score"] = pd.to_numeric(frame["model_score"], errors="coerce")
    frame["qa_issue_count"] = pd.to_numeric(frame.get("qa_issue_count", 0), errors="coerce").fillna(0)
    severity = frame.get("qa_max_severity", "none").astype("string").str.lower()
    severity_score = severity.map({"none": 0, "low": 1, "medium": 2, "high": 3}).fillna(0)
    frame["risk_score"] = frame["qa_issue_count"].clip(0, 5) + severity_score * 1.5
    frame["confidence_band"] = np.where(frame["model_score"].ge(0.8) | frame["model_score"].le(0.2), "High model confidence", "Lower model confidence")
    frame["risk_band"] = np.where(frame["risk_score"].ge(3), "High QA/data risk", "Lower QA/data risk")
    grouped = (
        frame.groupby(["confidence_band", "risk_band", "final_action"], as_index=False)
        .size()
        .rename(columns={"size": "cases"})
    )
    return (
        alt.Chart(grouped)
        .mark_circle(opacity=0.88, stroke=CARD_COLOR, strokeWidth=1.2)
        .encode(
            x=alt.X("confidence_band:N", title=None),
            y=alt.Y("risk_band:N", title=None),
            size=alt.Size("cases:Q", title="Cases", scale=alt.Scale(range=[180, 1200])),
            color=alt.Color(
                "final_action:N",
                title="Recommended action",
                scale=alt.Scale(domain=list(ACTION_COLORS), range=list(ACTION_COLORS.values())),
            ),
            tooltip=[
                alt.Tooltip("confidence_band:N", title="Model confidence"),
                alt.Tooltip("risk_band:N", title="QA risk"),
                alt.Tooltip("final_action:N", title="Action"),
                alt.Tooltip("cases:Q", title="Cases"),
            ],
        )
        .properties(height=320)
    )


def build_routing_flow_frame(decision_actions: pd.DataFrame) -> pd.DataFrame:
    if decision_actions.empty:
        return pd.DataFrame()
    frame = decision_actions.copy()
    frame["model_score"] = pd.to_numeric(frame.get("model_score"), errors="coerce")
    frame["Model confidence"] = np.where(
        frame["model_score"].ge(0.8) | frame["model_score"].le(0.2),
        "High confidence",
        "Needs review band",
    )
    frame["Data guardrail"] = np.where(
        frame.get("data_confidence", STATUS_CAUTION).astype("string").isin([STATUS_RELIABLE]),
        "Reliable data",
        "QA caution/risk",
    )
    frame["Fairness guardrail"] = np.where(
        frame.get("fairness_risk_flag", False).astype(bool),
        "Fairness risk",
        "No fairness flag",
    )
    stages = [
        ("All applications", pd.Series("All cases", index=frame.index)),
        ("Model confidence", frame["Model confidence"]),
        ("Data guardrail", frame["Data guardrail"]),
        ("Fairness guardrail", frame["Fairness guardrail"]),
        ("Recommended action", frame["final_action"]),
    ]
    rows: list[pd.DataFrame] = []
    for order, (stage, series) in enumerate(stages):
        counts = series.value_counts().rename_axis("segment").reset_index(name="cases")
        counts["stage"] = stage
        counts["stage_order"] = order
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def make_routing_flow_chart(flow_frame: pd.DataFrame) -> alt.Chart:
    if flow_frame.empty:
        return alt.Chart(pd.DataFrame()).mark_bar()
    return (
        alt.Chart(flow_frame)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("stage:N", title=None, sort=alt.SortField("stage_order")),
            y=alt.Y("cases:Q", title="Cases", stack="normalize", axis=alt.Axis(format="%")),
            color=alt.Color(
                "segment:N",
                title="Segment",
                scale=alt.Scale(range=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, "#F97316", DANGER_COLOR, "#64748B"]),
            ),
            tooltip=[
                alt.Tooltip("stage:N", title="Stage"),
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("cases:Q", title="Cases"),
            ],
        )
        .properties(height=330)
    )


def make_confusion_matrix_chart(matrix_frame: pd.DataFrame) -> alt.LayerChart:
    if matrix_frame.empty or not {"actual", "predicted", "cases"}.issubset(matrix_frame.columns):
        return alt.Chart(pd.DataFrame()).mark_rect()
    frame = matrix_frame.copy()
    frame["cases"] = pd.to_numeric(frame["cases"], errors="coerce").fillna(0)
    frame["row_total"] = frame.groupby("actual")["cases"].transform("sum")
    frame["col_total"] = frame.groupby("predicted")["cases"].transform("sum")
    frame["row_pct"] = frame["cases"] / frame["row_total"].replace(0, np.nan)
    frame["col_pct"] = frame["cases"] / frame["col_total"].replace(0, np.nan)
    frame["label"] = frame.apply(
        lambda row: f"{int(row['cases']):,}\\nrow {row['row_pct']:.0%}\\ncol {row['col_pct']:.0%}",
        axis=1,
    )
    frame["error_type"] = np.where(
        frame["actual"].astype(str).str.lower().str.contains("awarded")
        & frame["predicted"].astype(str).str.lower().str.contains("denied|reject|zero"),
        "False negative risk",
        "Other cell",
    )
    heatmap = (
        alt.Chart(frame)
        .mark_rect(cornerRadius=5)
        .encode(
            x=alt.X("predicted:N", title="Predicted"),
            y=alt.Y("actual:N", title="Actual"),
            color=alt.Color(
                "cases:Q",
                title="Cases",
                scale=alt.Scale(range=["#F8FAFC", "#A7F3D0", PRIMARY_COLOR]),
            ),
            stroke=alt.condition(
                alt.datum.error_type == "False negative risk",
                alt.value(DANGER_COLOR),
                alt.value(CARD_COLOR),
            ),
            strokeWidth=alt.condition(alt.datum.error_type == "False negative risk", alt.value(3), alt.value(1)),
            tooltip=[
                alt.Tooltip("actual:N", title="Actual"),
                alt.Tooltip("predicted:N", title="Predicted"),
                alt.Tooltip("cases:Q", title="Cases"),
                alt.Tooltip("row_pct:Q", title="Row %", format=".1%"),
                alt.Tooltip("col_pct:Q", title="Column %", format=".1%"),
                alt.Tooltip("error_type:N", title="Consequence"),
            ],
        )
    )
    labels = alt.Chart(frame).mark_text(color=TEXT_COLOR, fontWeight=700, lineBreak="\n").encode(
        x="predicted:N",
        y="actual:N",
        text="label:N",
    )
    return alt.layer(heatmap, labels).properties(height=280)


def make_threshold_tradeoff_curve(
    threshold_frame: pd.DataFrame,
    *,
    selected_threshold: float | None = None,
) -> alt.LayerChart:
    if threshold_frame.empty or not {"threshold", "selected_threshold"}.intersection(threshold_frame.columns):
        return alt.Chart(pd.DataFrame()).mark_line()
    frame = threshold_frame.copy()
    if "threshold" not in frame.columns and "selected_threshold" in frame.columns:
        frame = frame.rename(columns={"selected_threshold": "threshold"})
    if "threshold" in frame.columns and isinstance(frame["threshold"], pd.DataFrame):
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    metric_columns = [column for column in ["precision", "recall"] if column in frame.columns]
    if "true_positives" in frame.columns and "false_positives" in frame.columns:
        total = pd.to_numeric(frame["true_positives"], errors="coerce").fillna(0) + pd.to_numeric(frame["false_positives"], errors="coerce").fillna(0)
        frame["automation_rate"] = total / max(float(total.max()), 1.0)
        metric_columns.append("automation_rate")
    if {"precision", "recall"}.issubset(frame.columns):
        precision = pd.to_numeric(frame["precision"], errors="coerce")
        recall = pd.to_numeric(frame["recall"], errors="coerce")
        frame["f1"] = 2 * precision * recall / (precision + recall).replace(0, np.nan)
        metric_columns.append("f1")
    long_frame = frame.melt(
        id_vars=["threshold"],
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    ).dropna()
    lines = (
        alt.Chart(long_frame)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("threshold:Q", title="Threshold"),
            y=alt.Y("value:Q", title="Metric value", axis=alt.Axis(format="%")),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=[
                alt.Tooltip("threshold:Q", title="Threshold", format=".3f"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=".1%"),
            ],
        )
    )
    if selected_threshold is None:
        return lines.properties(height=330)
    rule = alt.Chart(pd.DataFrame({"threshold": [selected_threshold]})).mark_rule(
        color=TEXT_COLOR,
        strokeDash=[6, 6],
    ).encode(x="threshold:Q")
    return alt.layer(lines, rule).properties(height=330)
