from __future__ import annotations

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
        .ui-pill.audit {{
            color: #B45309;
            background: rgba(255, 247, 237, 0.96);
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
        .ui-kpi-card.audit {{
            border-top: 4px solid #F97316;
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
        .ui-status-badge {{
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.22rem 0.58rem;
            border: 1px solid var(--border);
            font-size: 0.78rem;
            font-weight: 760;
            line-height: 1.1;
            white-space: nowrap;
        }}
        .ui-status-badge.primary {{
            color: var(--primary);
            background: var(--success-tint);
            border-color: rgba(31, 122, 108, 0.24);
        }}
        .ui-status-badge.warning {{
            color: #9A6700;
            background: var(--accent-tint);
            border-color: rgba(245, 158, 11, 0.28);
        }}
        .ui-status-badge.audit {{
            color: #B45309;
            background: #FFF7ED;
            border-color: rgba(249, 115, 22, 0.28);
        }}
        .ui-status-badge::before {{
            content: "";
            width: 0.42rem;
            height: 0.42rem;
            border-radius: 999px;
            background: currentColor;
            margin-right: 0.34rem;
            opacity: 0.82;
        }}
        .ui-status-badge.danger {{
            color: #B42318;
            background: var(--danger-tint);
            border-color: rgba(220, 38, 38, 0.24);
        }}
        .ui-status-badge.secondary {{
            color: var(--muted);
            background: var(--info-tint);
            border-color: var(--border);
        }}
        .ui-takeaway {{
            border: 1px solid var(--border);
            border-left: 5px solid var(--primary);
            border-radius: 12px;
            background: #ffffff;
            padding: 0.85rem 0.95rem;
            margin: 0.85rem 0 1rem 0;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
        }}
        .ui-takeaway.warning {{
            border-left-color: var(--accent);
        }}
        .ui-takeaway.audit {{
            border-left-color: #F97316;
            background: linear-gradient(180deg, #ffffff 0%, #fffaf5 100%);
        }}
        .ui-takeaway.danger {{
            border-left-color: var(--danger);
        }}
        .ui-takeaway.secondary {{
            border-left-color: var(--muted);
        }}
        .ui-takeaway-label {{
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.22rem;
        }}
        .ui-takeaway-body {{
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 650;
            line-height: 1.5;
            margin: 0;
        }}
        .ui-metric-card {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.9rem 0.95rem;
            min-height: 154px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
        }}
        .ui-metric-card.primary {{
            border-top: 4px solid var(--primary);
        }}
        .ui-metric-card.warning {{
            border-top: 4px solid var(--accent);
        }}
        .ui-metric-card.audit {{
            border-top: 4px solid #F97316;
        }}
        .ui-metric-card.danger {{
            border-top: 4px solid var(--danger);
        }}
        .ui-metric-card.secondary {{
            border-top: 4px solid var(--muted);
        }}
        .ui-metric-topline {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.55rem;
            margin-bottom: 0.42rem;
        }}
        .ui-metric-label {{
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .ui-metric-value {{
            color: var(--text);
            font-size: 1.8rem;
            line-height: 1.05;
            font-weight: 820;
            margin-bottom: 0.48rem;
        }}
        .ui-metric-note {{
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.42;
            margin: 0 0 0.6rem 0;
        }}
        .ui-metric-meta {{
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.35;
            border-top: 1px solid rgba(215, 227, 224, 0.75);
            padding-top: 0.45rem;
        }}
        .ui-step-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0 0 0.95rem 0;
        }}
        .ui-step {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: #ffffff;
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 720;
            padding: 0.32rem 0.65rem;
        }}
        .ui-step.active {{
            color: #ffffff;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            border-color: var(--primary);
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
        .ui-chart-head {{
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            align-items: flex-start;
            margin-bottom: 0.55rem;
        }}
        .ui-chart-title {{
            color: var(--text);
            font-size: 1rem;
            font-weight: 780;
            line-height: 1.25;
            margin: 0;
        }}
        .ui-chart-subtitle {{
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.4;
            margin: 0.18rem 0 0 0;
        }}
        .ui-chart-meta {{
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
            text-align: right;
        }}
        .ui-chart-takeaway {{
            color: var(--text);
            background: #F8FAFC;
            border: 1px solid rgba(215, 227, 224, 0.75);
            border-left: 4px solid var(--primary);
            border-radius: 10px;
            font-size: 0.9rem;
            font-weight: 650;
            line-height: 1.45;
            padding: 0.64rem 0.72rem;
            margin: 0.55rem 0 0.8rem 0;
        }}
        .ui-chart-caption {{
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.45;
            margin: 0.68rem 0 0 0;
        }}
        .ui-warning-inline {{
            color: #9A3412;
            background: #FFF7ED;
            border: 1px solid rgba(249, 115, 22, 0.24);
            border-radius: 10px;
            font-size: 0.86rem;
            font-weight: 650;
            line-height: 1.4;
            padding: 0.58rem 0.7rem;
            margin: 0.55rem 0 0.75rem 0;
        }}
        .ui-flow-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
            gap: 0.65rem;
            margin: 0.75rem 0 1rem 0;
        }}
        .ui-flow-node {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.75rem 0.8rem;
            min-height: 96px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        }}
        .ui-flow-step {{
            color: var(--primary);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }}
        .ui-flow-title {{
            color: var(--text);
            font-size: 0.96rem;
            font-weight: 780;
            line-height: 1.25;
            margin-bottom: 0.25rem;
        }}
        .ui-flow-note {{
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.35;
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
    st.caption(kicker)
    st.title(title)
    st.write(description)
    if takeaway:
        st.info(f"**{takeaway_label}:** {takeaway}")
    if pills:
        columns = st.columns(min(len(pills), 6))
        for index, (text, tone) in enumerate(pills):
            with columns[index % len(columns)]:
                st.badge(text, color=badge_color_for_status(tone))


def render_kpi_row(cards: Sequence[dict[str, str]]) -> None:
    """Render up to four consistent KPI cards."""
    if not cards:
        return
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            with st.container(border=True):
                st.caption(str(card.get("label", "")).upper())
                st.markdown(f"### {card.get('value', '')}")
                note = str(card.get("note", ""))
                if note:
                    st.write(note)


def _tone_for_status(status: str) -> str:
    normalized = str(status).strip().lower()
    if normalized == "reliable":
        return "primary"
    if normalized == "caution":
        return "warning"
    if normalized in {"audit", "audit / risk"}:
        return "audit"
    if normalized == "high risk":
        return "danger"
    if normalized in {"not production-ready", "not production ready", "monitor only"}:
        return "secondary"
    return "secondary"


def badge_color_for_status(status: str) -> str:
    """Map decision statuses or legacy CSS tones to native Streamlit badge colors."""
    normalized = str(status).strip().lower()
    if normalized in {"reliable", "primary", "success", "green"}:
        return "green"
    if normalized in {"caution", "warning", "yellow"}:
        return "yellow"
    if normalized in {"audit", "audit / risk", "orange"}:
        return "orange"
    if normalized in {"high risk", "danger", "red"}:
        return "red"
    if normalized in {"not production-ready", "not production ready", "monitor only", "secondary", "gray", "grey"}:
        return "gray"
    return "gray"


def render_reliability_badge(status: str, *, reason: str | None = None) -> None:
    """Render a compact status badge with an optional captioned reason."""
    st.badge(str(status), color=badge_color_for_status(status))
    if reason:
        st.caption(reason)


def render_takeaway_box(
    takeaway: str,
    *,
    status: str = "Reliable",
    label: str = "Takeaway",
) -> None:
    """Render the required section-level takeaway line."""
    with st.container(border=True):
        st.badge(status, color=badge_color_for_status(status))
        st.markdown(f"**{label}:** {takeaway}")


def render_metric_card(
    *,
    label: str,
    value: str,
    interpretation: str,
    status: str,
    sample_size: int | None = None,
    coverage: str | None = None,
    reason: str | None = None,
) -> None:
    """Render a decision-grade metric card with status, coverage, and sample size."""
    meta_parts = []
    if sample_size is not None:
        meta_parts.append(f"n={sample_size:,}")
    if coverage is not None:
        meta_parts.append(f"coverage={coverage}")
    if reason:
        meta_parts.append(reason)
    meta_text = " | ".join(meta_parts)
    with st.container(border=True):
        top_columns = st.columns([0.62, 0.38])
        with top_columns[0]:
            st.caption(label.upper())
        with top_columns[1]:
            st.badge(status, color=badge_color_for_status(status))
        st.markdown(f"### {value}")
        st.write(interpretation)
        if meta_text:
            st.caption(meta_text)


def render_decision_journey(active_step: str) -> None:
    """Show the Data Quality -> Evidence -> Model -> Decision -> Action journey."""
    steps = ["Data Quality", "Evidence", "Model", "Decision", "Action"]
    active_index = steps.index(active_step) + 1 if active_step in steps else None
    journey = " -> ".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    st.caption(f"Decision journey: {journey}")
    if active_index is not None:
        st.badge(f"Current step {active_index}: {active_step}", color="primary")


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
        with st.container(border=True):
            st.markdown(f"**{insight_label}**")
            st.write(insight)
    with columns[1]:
        with st.container(border=True):
            st.markdown(f"**{implication_label}**")
            st.write(implication)


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
