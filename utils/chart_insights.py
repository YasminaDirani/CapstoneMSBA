from __future__ import annotations

import numpy as np
import pandas as pd

from utils.decision_labels import COMPARABLE_DECISION_LABELS, normalize_decision_series


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _fmt_number(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    magnitude = abs(float(value))
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def decision_distribution_insights(decision_counts: pd.DataFrame) -> list[str]:
    """Summarize the shape and cleanliness of decision labels."""
    if decision_counts.empty or "Count" not in decision_counts.columns:
        return []

    counts = decision_counts["Count"].copy()
    counts.index = counts.index.astype(str)
    normalized_index = counts.index.str.strip().str.lower()

    awarded = int(counts[normalized_index.isin(COMPARABLE_DECISION_LABELS - {"denied"})].sum())
    denied = int(counts[normalized_index == "denied"].sum())
    missing = int(counts[normalized_index == "missing"].sum())
    nonstandard = int(counts.sum() - awarded - denied - missing)
    standard_total = awarded + denied

    insights: list[str] = []
    if standard_total > 0:
        award_rate = awarded / standard_total
        insights.append(
            f"Awarded cases make up {award_rate:.1%} of standard decisions, so approvals are more common than denials in the labeled subset."
        )
    if missing > 0 or nonstandard > 0:
        insights.append(
            f"There are {missing:,} missing decisions and {nonstandard:,} non-standard decision labels, which suggests the outcome field is not fully normalized yet."
        )
    return insights


def school_concentration_insights(top_schools: pd.DataFrame, total_rows: int) -> list[str]:
    """Highlight concentration in school representation."""
    if top_schools.empty or "Count" not in top_schools.columns or total_rows <= 0:
        return []

    ranked = top_schools.reset_index().rename(columns={top_schools.index.name or "index": "school"})
    top_school = ranked.iloc[0]
    top_three_share = ranked["Count"].head(3).sum() / total_rows
    top_school_share = top_school["Count"] / total_rows

    insights = [
        f"{top_school['school']} contributes the largest visible share of applications at {top_school_share:.1%} of the full dataset, so school representation is not evenly distributed."
    ]
    if top_three_share >= 0.25:
        insights.append(
            f"The top three schools account for {top_three_share:.1%} of all records, which means institution mix could materially shape overall patterns."
        )
    else:
        insights.append(
            f"No single school completely dominates the sample, but the leading schools still account for {top_three_share:.1%} of all records."
        )
    return insights


def numeric_decision_comparison_insights(
    dataframe: pd.DataFrame,
    value_column: str,
    metric_label: str,
    *,
    decision_column: str = "decision",
    positive_label: str = "awarded",
    negative_label: str = "denied",
) -> list[str]:
    """Compare a numeric field between awarded and denied applications."""
    if value_column not in dataframe.columns or decision_column not in dataframe.columns:
        return []

    decision_clean = normalize_decision_series(dataframe[decision_column])
    values = pd.to_numeric(dataframe[value_column], errors="coerce")
    valid_mask = decision_clean.isin([positive_label, negative_label]) & values.notna()

    if int(valid_mask.sum()) == 0:
        return []

    comparison = pd.DataFrame(
        {
            "decision": decision_clean.loc[valid_mask],
            "value": values.loc[valid_mask],
        }
    )
    awarded = comparison.loc[comparison["decision"] == positive_label, "value"]
    denied = comparison.loc[comparison["decision"] == negative_label, "value"]
    if awarded.empty or denied.empty:
        return []

    awarded_median = float(awarded.median())
    denied_median = float(denied.median())
    awarded_mean = float(awarded.mean())
    denied_mean = float(denied.mean())
    coverage = len(comparison) / max(int(decision_clean.isin([positive_label, negative_label]).sum()), 1)

    insights: list[str] = []
    if awarded_median == denied_median:
        mean_gap = awarded_mean - denied_mean
        if abs(mean_gap) < 1e-9:
            insights.append(
                f"{metric_label} shows almost no separation between awarded and denied applications, so it does not look like a strong stand-alone discriminator here."
            )
        else:
            direction = "higher" if mean_gap > 0 else "lower"
            insights.append(
                f"The medians are the same, but the average {metric_label.lower()} is {direction} for awarded applications, which suggests only a modest difference rather than a clean split."
            )
    else:
        direction = "lower" if awarded_median < denied_median else "higher"
        insights.append(
            f"Awarded applications show a {direction} observed median {metric_label.lower()} in the cleaned subset ({_fmt_number(awarded_median)} vs {_fmt_number(denied_median)}), but that gap is descriptive rather than causal."
        )

    if coverage < 0.5:
        insights.append(
            f"This comparison uses only {coverage:.0%} of standard awarded-versus-denied records, so missing data may be shaping the pattern."
        )
    return insights


def award_rate_group_insights(
    grouped_df: pd.DataFrame,
    *,
    group_column: str,
    rate_column: str = "award_rate_pct",
    applications_column: str | None = "applications",
) -> list[str]:
    """Summarize strongest and weakest segments in a grouped award-rate view."""
    if grouped_df.empty or group_column not in grouped_df.columns or rate_column not in grouped_df.columns:
        return []

    ranked = grouped_df.sort_values(rate_column, ascending=False, ignore_index=True)
    highest = ranked.iloc[0]
    lowest = ranked.iloc[-1]
    spread = float(highest[rate_column] - lowest[rate_column])

    insights = [
        f"{highest[group_column]} has the highest observed award rate at {highest[rate_column]:.1f}%, while {lowest[group_column]} is lowest at {lowest[rate_column]:.1f}%, a spread of {spread:.1f} percentage points."
    ]
    if applications_column and applications_column in ranked.columns:
        largest_group = ranked.sort_values(applications_column, ascending=False, ignore_index=True).iloc[0]
        insights.append(
            f"{largest_group[group_column]} contributes the largest volume of applications, so shifts in that group would move overall results more than smaller segments at the extremes."
        )
    return insights


def probability_distribution_insights(
    dataframe: pd.DataFrame,
    *,
    probability_column: str = "predicted_award_probability",
    decision_column: str = "decision",
) -> list[str]:
    """Summarize how decisive and how separated predicted probabilities look."""
    if probability_column not in dataframe.columns:
        return []

    probabilities = pd.to_numeric(dataframe[probability_column], errors="coerce").dropna()
    if probabilities.empty:
        return []

    high_share = float((probabilities >= 0.8).mean())
    low_share = float((probabilities <= 0.2).mean())
    middle_share = float(((probabilities >= 0.4) & (probabilities <= 0.6)).mean())

    insights: list[str] = []
    if high_share + low_share >= 0.6:
        insights.append(
            f"{high_share + low_share:.1%} of scores sit in the high-confidence tails, so the model is making relatively decisive calls for much of the dataset."
        )
    else:
        insights.append(
            f"{middle_share:.1%} of scores fall between 0.4 and 0.6, which suggests many cases are borderline rather than clearly separable."
        )

    if decision_column in dataframe.columns:
        decision_clean = normalize_decision_series(dataframe[decision_column])
        valid_mask = decision_clean.isin(["awarded", "denied"]) & dataframe[probability_column].notna()
        if int(valid_mask.sum()) > 0:
            comparison = dataframe.loc[valid_mask, [probability_column]].copy()
            comparison["decision"] = decision_clean.loc[valid_mask]
            awarded_mean = float(
                pd.to_numeric(
                    comparison.loc[comparison["decision"] == "awarded", probability_column],
                    errors="coerce",
                ).mean()
            )
            denied_mean = float(
                pd.to_numeric(
                    comparison.loc[comparison["decision"] == "denied", probability_column],
                    errors="coerce",
                ).mean()
            )
            gap = awarded_mean - denied_mean
            insights.append(
                f"Average predicted award probability is {awarded_mean:.1%} for awarded cases versus {denied_mean:.1%} for denied cases, a separation of {gap:.1%}."
            )
    return insights


def feature_importance_insights(feature_importance_df: pd.DataFrame) -> list[str]:
    """Summarize concentration and mix in top feature importances."""
    if feature_importance_df.empty:
        return []

    ranked = feature_importance_df.sort_values("importance", ascending=False, ignore_index=True)
    top_feature = ranked.iloc[0]
    total_importance = float(ranked["importance"].sum())
    top_three_share = (
        float(ranked["importance"].head(3).sum() / total_importance)
        if total_importance > 0
        else 0.0
    )

    insights = [
        f"{top_feature['feature']} is the strongest modeled signal in the displayed set."
    ]
    if top_three_share >= 0.5:
        insights.append(
            f"The top three features account for {top_three_share:.0%} of the displayed signal, so the model appears relatively concentrated rather than evenly spread."
        )
    else:
        insights.append(
            f"The top three features account for {top_three_share:.0%} of the displayed signal, so model influence is distributed across several variables."
        )
    return insights


def driver_side_insights(driver_df: pd.DataFrame, *, side: str) -> list[str]:
    """Summarize the strongest directional coefficient drivers."""
    if driver_df.empty:
        return []

    top_driver = driver_df.iloc[0]
    coefficient = float(top_driver["coefficient"])
    side_phrase = "upward" if side == "positive" else "downward"
    insights = [
        f"{top_driver['feature']} is the clearest {side_phrase} driver in the displayed set, with a coefficient of {_fmt_number(coefficient)}."
    ]
    if len(driver_df) > 1:
        runner_up = driver_df.iloc[1]
        if abs(coefficient) >= abs(float(runner_up["coefficient"])) * 1.5:
            insights.append(
                f"It stands noticeably ahead of {runner_up['feature']}, which suggests the directional signal is concentrated rather than evenly shared."
            )
        else:
            insights.append(
                f"It is relatively close to {runner_up['feature']}, so this side of the decision signal is spread across multiple features."
            )
    return insights


def roc_curve_insights(roc_curve_df: pd.DataFrame, roc_auc: float) -> list[str]:
    """Summarize ranking strength from ROC behavior."""
    if roc_curve_df.empty:
        return []

    if roc_auc >= 0.85:
        quality = "strong"
    elif roc_auc >= 0.75:
        quality = "useful"
    elif roc_auc >= 0.65:
        quality = "modest"
    else:
        quality = "weak"

    point = roc_curve_df.iloc[(roc_curve_df["false_positive_rate"] - 0.2).abs().argsort()[:1]].iloc[0]
    insights = [
        f"The model's ranking ability looks {quality} with an ROC AUC of {roc_auc:.3f}."
    ]
    insights.append(
        f"At roughly a 20% false positive rate, the curve captures about {float(point['true_positive_rate']):.1%} of awarded cases, which gives a practical sense of the trade-off."
    )
    return insights


def confusion_matrix_insights(
    confusion_summary: dict[str, int],
    *,
    precision: float | None,
    recall: float | None,
    threshold: float,
) -> list[str]:
    """Summarize threshold-specific error trade-offs."""
    true_positive = int(confusion_summary.get("true_positive", 0))
    false_positive = int(confusion_summary.get("false_positive", 0))
    false_negative = int(confusion_summary.get("false_negative", 0))
    true_negative = int(confusion_summary.get("true_negative", 0))

    total = true_positive + false_positive + false_negative + true_negative
    if total == 0:
        return []

    insights: list[str] = []
    if precision is not None and recall is not None:
        insights.append(
            f"At a threshold of {threshold:.2f}, precision is {precision:.1%} and recall is {recall:.1%}, so the cutoff is balancing hit rate against over-approval risk."
        )
    if false_positive > false_negative:
        insights.append(
            "False positives outnumber false negatives at this threshold, which means the model is leaning more toward over-predicting awards than missing them."
        )
    elif false_negative > false_positive:
        insights.append(
            "False negatives outnumber false positives at this threshold, which means the model is acting more conservatively and may miss some award-worthy cases."
        )
    else:
        insights.append(
            "False positives and false negatives are evenly balanced at this threshold, so the current cutoff is not obviously biased toward one error type."
        )
    return insights


def correlation_heatmap_insights(correlation_df: pd.DataFrame) -> list[str]:
    """Summarize strongest numeric relationships in the displayed matrix."""
    if correlation_df.empty or len(correlation_df.columns) < 2:
        return []

    stacked = (
        correlation_df.where(~pd.DataFrame(
            np.eye(len(correlation_df), dtype=bool),
            index=correlation_df.index,
            columns=correlation_df.columns,
        ))
        .stack()
        .reset_index()
    )
    if stacked.empty:
        return []
    stacked.columns = ["feature_a", "feature_b", "correlation"]
    stacked["pair_key"] = stacked.apply(
        lambda row: tuple(sorted((row["feature_a"], row["feature_b"]))),
        axis=1,
    )
    stacked = stacked.drop_duplicates(subset=["pair_key"])

    strongest_positive = stacked.sort_values("correlation", ascending=False).iloc[0]
    strongest_negative = stacked.sort_values("correlation", ascending=True).iloc[0]

    insights = [
        f"The strongest positive relationship in the displayed set is between {strongest_positive['feature_a']} and {strongest_positive['feature_b']} (r = {float(strongest_positive['correlation']):.2f}), which suggests those fields may be carrying overlapping information."
    ]
    if float(strongest_negative["correlation"]) <= -0.30:
        insights.append(
            f"The strongest negative relationship is between {strongest_negative['feature_a']} and {strongest_negative['feature_b']} (r = {float(strongest_negative['correlation']):.2f}), indicating a meaningful trade-off between those measures."
        )
    else:
        insights.append(
            "There are no strongly negative relationships in the displayed matrix, so most numeric fields move independently rather than as clear opposites."
        )
    return insights


def group_pattern_chart_insights(
    group_patterns: pd.DataFrame,
    *,
    dimension_label: str,
) -> list[str]:
    """Summarize the strongest and weakest groups in a segment comparison chart."""
    if group_patterns.empty:
        return []

    ranked = group_patterns.sort_values("lift_pp", ascending=False, ignore_index=True)
    strongest = ranked.iloc[0]
    weakest = ranked.iloc[-1]
    spread = float(strongest["award_rate_pct"] - weakest["award_rate_pct"])

    insights = [
        f"Within {dimension_label.lower()}, {strongest['group']} is furthest above baseline at {strongest['award_rate_pct']:.1f}% awarded, while {weakest['group']} is furthest below at {weakest['award_rate_pct']:.1f}%."
    ]
    insights.append(
        f"The gap between those groups is {spread:.1f} percentage points, which suggests this dimension is materially associated with award outcomes rather than just adding noise."
    )
    largest_group = ranked.sort_values("applications", ascending=False, ignore_index=True).iloc[0]
    insights.append(
        f"{largest_group['group']} has the largest sample within this view, so it deserves extra attention when interpreting the overall pattern."
    )
    return insights
