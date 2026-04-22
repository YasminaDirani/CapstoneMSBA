from __future__ import annotations

import html
from typing import Iterable, Sequence

import altair as alt
import pandas as pd
import streamlit as st

PRIMARY_COLOR = "#1F7A6C"
PRIMARY_DARK = "#165E55"
SECONDARY_COLOR = "#2CA58D"
ACCENT_COLOR = "#F59E0B"
DANGER_COLOR = "#DC2626"
BACKGROUND_COLOR = "#F8FAFC"
CARD_COLOR = "#FFFFFF"
TEXT_COLOR = "#1F2937"
MUTED_TEXT_COLOR = "#64748B"
BORDER_COLOR = "#D7E3E0"
SUCCESS_TINT = "#ECFDF5"
ACCENT_TINT = "#FFFBEB"
DANGER_TINT = "#FEF2F2"
INFO_TINT = "#EEF6F5"
CHART_NEUTRALS = [
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    ACCENT_COLOR,
    DANGER_COLOR,
    "#94A3B8",
]
_ALTAIR_THEME_NAME = "yasmina_executive"


def _register_altair_theme() -> None:
    """Enable a calm, readable default chart theme for every page."""

    def theme() -> dict[str, object]:
        return {
            "config": {
                "background": CARD_COLOR,
                "view": {"stroke": None},
                "font": "Avenir Next, Avenir, Segoe UI, sans-serif",
                "title": {
                    "font": "Avenir Next, Avenir, Segoe UI, sans-serif",
                    "fontSize": 16,
                    "fontWeight": 700,
                    "color": TEXT_COLOR,
                    "anchor": "start",
                },
                "axis": {
                    "domain": False,
                    "grid": False,
                    "labelColor": MUTED_TEXT_COLOR,
                    "labelFontSize": 11,
                    "labelFontWeight": 500,
                    "titleColor": TEXT_COLOR,
                    "titleFontSize": 12,
                    "titleFontWeight": 700,
                    "tickColor": BORDER_COLOR,
                },
                "legend": {
                    "labelColor": MUTED_TEXT_COLOR,
                    "labelFontSize": 11,
                    "titleColor": TEXT_COLOR,
                    "titleFontSize": 12,
                    "titleFontWeight": 700,
                    "orient": "bottom",
                },
                "range": {"category": CHART_NEUTRALS},
                "bar": {"cornerRadiusTopLeft": 5, "cornerRadiusTopRight": 5},
                "line": {"strokeWidth": 3},
                "point": {"filled": True, "size": 72},
                "arc": {"stroke": CARD_COLOR, "strokeWidth": 1.25},
                "text": {"font": "Avenir Next, Avenir, Segoe UI, sans-serif"},
            }
        }

    try:
        alt.themes.register(_ALTAIR_THEME_NAME, theme)
    except ValueError:
        pass
    alt.themes.enable(_ALTAIR_THEME_NAME)


def apply_design_system(*, max_width: int = 1460, extra_css: str = "") -> None:
    """Inject the shared design system and chart defaults."""
    _register_altair_theme()
    st.markdown(
        f"""
        <style>
        :root {{
            --primary: {PRIMARY_COLOR};
            --primary-dark: {PRIMARY_DARK};
            --secondary: {SECONDARY_COLOR};
            --accent: {ACCENT_COLOR};
            --danger: {DANGER_COLOR};
            --bg: {BACKGROUND_COLOR};
            --card: {CARD_COLOR};
            --text: {TEXT_COLOR};
            --muted: {MUTED_TEXT_COLOR};
            --border: {BORDER_COLOR};
            --success-tint: {SUCCESS_TINT};
            --accent-tint: {ACCENT_TINT};
            --danger-tint: {DANGER_TINT};
            --info-tint: {INFO_TINT};
        }}
        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(44, 165, 141, 0.08), transparent 26%),
                linear-gradient(180deg, var(--bg) 0%, #ffffff 14%, #ffffff 100%);
            color: var(--text);
            font-family: "Avenir Next", Avenir, "Segoe UI", sans-serif;
        }}
        [data-testid="stMainBlockContainer"],
        .block-container {{
            width: min({max_width}px, calc(100vw - 2.6rem)) !important;
            max-width: min({max_width}px, calc(100vw - 2.6rem)) !important;
            padding-top: 1.25rem !important;
            padding-bottom: 4rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }}
        @media (max-width: 900px) {{
            [data-testid="stMainBlockContainer"],
            .block-container {{
                width: calc(100vw - 0.8rem) !important;
                max-width: calc(100vw - 0.8rem) !important;
                padding-left: 0.25rem !important;
                padding-right: 0.25rem !important;
            }}
        }}
        h1, h2, h3, h4 {{
            color: var(--text);
            letter-spacing: -0.02em;
        }}
        h1 {{
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }}
        h2 {{
            font-size: 1.4rem;
            font-weight: 750;
            margin-top: 1.25rem;
        }}
        h3 {{
            font-size: 1.08rem;
            font-weight: 720;
        }}
        p, li, div, span, label {{
            color: var(--text);
        }}
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stCaptionContainer"] {{
            line-height: 1.55;
        }}
        div[data-testid="stCaptionContainer"] {{
            color: var(--muted);
            font-size: 0.92rem;
        }}
        div[data-baseweb="tab-list"] {{
            gap: 0.45rem;
            margin: 1.1rem 0 0.75rem 0;
            flex-wrap: wrap;
        }}
        button[data-baseweb="tab"] {{
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--border);
            border-radius: 999px;
            color: var(--muted);
            font-weight: 650;
            padding: 0.46rem 0.92rem;
            transition: all 0.18s ease;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }}
        button[data-baseweb="tab"]:hover {{
            border-color: rgba(31, 122, 108, 0.35);
            color: var(--primary);
            transform: translateY(-1px);
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            border-color: var(--primary);
            color: #ffffff;
            box-shadow: 0 8px 18px rgba(31, 122, 108, 0.16);
        }}
        div[data-baseweb="tab-highlight"] {{
            display: none;
        }}
        div[data-baseweb="tab-panel"] {{
            padding-top: 0.25rem;
        }}
        div[data-testid="stMetric"] {{
            background: linear-gradient(180deg, #ffffff 0%, #fbfdfd 100%);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.95rem 1rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
            min-height: 126px;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.08);
            border-color: rgba(31, 122, 108, 0.2);
        }}
        div[data-testid="stMetricLabel"] {{
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        div[data-testid="stMetricValue"] {{
            color: var(--text);
            font-size: 1.8rem;
            font-weight: 780;
            letter-spacing: -0.03em;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
            background: var(--card);
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-color: var(--border) !important;
            border-radius: 14px !important;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.06);
        }}
        div[data-testid="stAlert"] {{
            border-radius: 14px;
            border: 1px solid var(--border);
        }}
        .ui-page-header {{
            background:
                radial-gradient(circle at top right, rgba(44, 165, 141, 0.09), transparent 28%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.98) 100%);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1.2rem 1.25rem;
            margin: 0.1rem 0 1.05rem 0;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.05);
        }}
        .ui-kicker {{
            color: var(--primary);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }}
        .ui-page-title {{
            color: var(--text);
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1.08;
            letter-spacing: -0.03em;
            margin: 0;
        }}
        .ui-page-description {{
            color: var(--muted);
            font-size: 1rem;
            max-width: 74ch;
            margin: 0.45rem 0 0 0;
        }}
        .ui-executive-box {{
            border-left: 6px solid var(--primary);
            background: linear-gradient(180deg, rgba(240, 253, 250, 0.98) 0%, #ffffff 100%);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin: 0.95rem 0 0 0;
        }}
        .ui-executive-box.warning {{
            border-left-color: var(--accent);
            background: linear-gradient(180deg, rgba(255, 251, 235, 0.98) 0%, #ffffff 100%);
        }}
        .ui-executive-label {{
            color: var(--primary);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 0.28rem;
        }}
        .ui-executive-box.warning .ui-executive-label {{
            color: var(--accent);
        }}
        .ui-executive-body {{
            color: var(--text);
            font-size: 1rem;
            line-height: 1.5;
            font-weight: 700;
            margin: 0;
        }}
        .ui-pill-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.8rem;
        }}
        .ui-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.34rem 0.7rem;
            font-size: 0.83rem;
            font-weight: 650;
            background: rgba(248, 250, 252, 0.95);
            border: 1px solid var(--border);
            color: var(--muted);
        }}
        .ui-pill.primary {{
            color: var(--primary);
            background: rgba(236, 253, 245, 0.92);
        }}
        .ui-pill.warning {{
            color: #9A6700;
            background: rgba(255, 251, 235, 0.96);
        }}
        .ui-pill.danger {{
            color: #B42318;
            background: rgba(254, 242, 242, 0.96);
        }}
        .ui-kpi-card {{
            background: linear-gradient(180deg, #ffffff 0%, #fbfdfd 100%);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.95rem 1rem;
            min-height: 136px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }}
        .ui-kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.08);
        }}
        .ui-kpi-card.primary {{
            border-top: 4px solid var(--primary);
        }}
        .ui-kpi-card.secondary {{
            border-top: 4px solid var(--secondary);
        }}
        .ui-kpi-card.warning {{
            border-top: 4px solid var(--accent);
        }}
        .ui-kpi-card.danger {{
            border-top: 4px solid var(--danger);
        }}
        .ui-kpi-label {{
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.42rem;
        }}
        .ui-kpi-value {{
            color: var(--text);
            font-size: 1.9rem;
            line-height: 1;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.55rem;
        }}
        .ui-kpi-note {{
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0;
        }}
        .ui-insight-panel {{
            border: 1px solid var(--border);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.96);
            padding: 0.95rem 1rem;
            margin: 0.85rem 0 0.2rem 0;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
        }}
        .ui-insight-title {{
            color: var(--text);
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }}
        .ui-insight-body {{
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
            margin: 0;
        }}
        {extra_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    *,
    title: str,
    description: str,
    takeaway: str | None = None,
    kicker: str = "Decision-Grade Analytics",
    takeaway_label: str = "Executive Takeaway",
    pills: Sequence[tuple[str, str]] | None = None,
) -> None:
    """Render a consistent page title, purpose line, and executive takeaway."""
    pills = pills or []
    pill_html = "".join(
        (
            f'<span class="ui-pill {html.escape(tone)}">{html.escape(text)}</span>'
        )
        for text, tone in pills
    )
    takeaway_html = ""
    if takeaway:
        takeaway_html = (
            '<div class="ui-executive-box">'
            f'<div class="ui-executive-label">{html.escape(takeaway_label)}</div>'
            f'<p class="ui-executive-body">{html.escape(takeaway)}</p>'
            "</div>"
        )
    st.markdown(
        f"""
        <section class="ui-page-header">
            <div class="ui-kicker">{html.escape(kicker)}</div>
            <h1 class="ui-page-title">{html.escape(title)}</h1>
            <p class="ui-page-description">{html.escape(description)}</p>
            {takeaway_html}
            <div class="ui-pill-row">{pill_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(cards: Sequence[dict[str, str]]) -> None:
    """Render up to four consistent KPI cards."""
    if not cards:
        return
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        tone = str(card.get("tone", "primary"))
        with column:
            st.markdown(
                f"""
                <section class="ui-kpi-card {html.escape(tone)}">
                    <div class="ui-kpi-label">{html.escape(str(card.get("label", "")))}</div>
                    <div class="ui-kpi-value">{html.escape(str(card.get("value", "")))}</div>
                    <p class="ui-kpi-note">{html.escape(str(card.get("note", "")))}</p>
                </section>
                """,
                unsafe_allow_html=True,
            )


def render_insight_action_panel(
    *,
    insight: str,
    implication: str,
    insight_label: str = "So What",
    implication_label: str = "What To Do",
) -> None:
    """Render a short insight plus action box below top-level KPIs."""
    columns = st.columns(2)
    with columns[0]:
        st.markdown(
            f"""
            <section class="ui-insight-panel">
                <div class="ui-insight-title">{html.escape(insight_label)}</div>
                <p class="ui-insight-body">{html.escape(insight)}</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            f"""
            <section class="ui-insight-panel">
                <div class="ui-insight-title">{html.escape(implication_label)}</div>
                <p class="ui-insight-body">{html.escape(implication)}</p>
            </section>
            """,
            unsafe_allow_html=True,
        )


def add_top_n_flag(
    dataframe: pd.DataFrame,
    *,
    value_column: str,
    top_n: int = 3,
    ascending: bool = False,
    flag_column: str = "is_top",
) -> pd.DataFrame:
    """Mark the strongest rows so charts can highlight top groups cleanly."""
    if dataframe.empty or value_column not in dataframe.columns:
        return dataframe.copy()
    working = dataframe.copy()
    order = working[value_column].rank(method="first", ascending=ascending)
    working[flag_column] = order <= top_n
    return working


def collapse_small_categories(
    dataframe: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    max_categories: int = 5,
    other_label: str = "Other",
) -> pd.DataFrame:
    """Group small slices so donut charts stay readable."""
    if dataframe.empty or len(dataframe) <= max_categories:
        return dataframe.copy()

    working = dataframe.copy().sort_values(value_column, ascending=False, ignore_index=True)
    head = working.head(max_categories - 1).copy()
    tail = working.iloc[max_categories - 1 :].copy()
    if tail.empty:
        return working

    numeric_tail = pd.to_numeric(tail[value_column], errors="coerce").fillna(0)
    other_value = float(numeric_tail.sum())
    other_row: dict[str, object] = {column: tail.iloc[0][column] for column in tail.columns}
    other_row[label_column] = other_label
    other_row[value_column] = other_value
    collapsed = pd.concat([head, pd.DataFrame([other_row])], ignore_index=True)

    total = float(pd.to_numeric(collapsed[value_column], errors="coerce").fillna(0).sum())
    if "share_pct" in collapsed.columns:
        collapsed["share_pct"] = (
            pd.to_numeric(collapsed[value_column], errors="coerce").fillna(0) / total * 100.0
            if total > 0
            else 0.0
        )
    if "share_label" in collapsed.columns:
        collapsed["share_label"] = collapsed["share_pct"].map(lambda value: f"{value:.1f}%")
    return collapsed


def build_color_condition(
    flag_column: str = "is_top",
    *,
    highlight_color: str = PRIMARY_COLOR,
    muted_color: str = "#D6DEE4",
) -> alt.Color:
    """Return a simple highlight-vs-muted color rule for bar charts."""
    return alt.condition(
        f"datum.{flag_column}",
        alt.value(highlight_color),
        alt.value(muted_color),
    )
