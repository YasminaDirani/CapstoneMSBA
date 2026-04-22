from __future__ import annotations

import numpy as np
import pandas as pd


POSITIVE_DECISION_LABEL = "awarded"
NEGATIVE_DECISION_LABEL = "denied"

GROUP_DIMENSIONS = {
    "Academic Level": "level",
    "Application Track": "application_track",
    "Application Type": "application_type_clean",
    "School": "school",
    "Nationality": "nationality",
    "Special Circumstances": "special_family_circumstances_category",
}

NUMERIC_METRICS = {
    "total_parent_income": "Total Parent Income",
    "dependents_count": "Dependents Count",
    "properties_total_estimated_value": "Estimated Property Value",
    "siblings_other_total_tuition": "Other Siblings Tuition",
    "loans_total_remaining_balance": "Remaining Loan Balance",
    "total_siblings": "Total Siblings",
    "financial_assistants_total_est_annual_amount": "External Assistant Support",
}

EXPECTED_DRIVER_DIRECTIONS = {
    "total_parent_income": "lower",
    "properties_total_estimated_value": "lower",
    "dependents_count": "higher",
    "total_siblings": "higher",
    "loans_total_remaining_balance": "higher",
    "siblings_other_total_tuition": "higher",
    "financial_assistants_total_est_annual_amount": "lower",
}

INCONSISTENCY_SIMILARITY_FEATURES = [
    "total_parent_income",
    "properties_total_estimated_value",
    "dependents_count",
    "total_siblings",
    "siblings_other_total_tuition",
]

INCONSISTENCY_PROFILE_FEATURES = {
    "total_parent_income": "Parent Income",
    "properties_total_estimated_value": "Property Value",
    "dependents_count": "Dependents",
    "total_siblings": "Siblings",
    "siblings_other_total_tuition": "Other Siblings Tuition",
}


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize text labels for safe comparisons."""
    return series.astype("string").str.strip().str.lower()


def coalesce_numeric(
    dataframe: pd.DataFrame,
    primary_column: str,
    fallback_column: str,
) -> pd.Series:
    """Prefer the primary numeric column and fill missing values from a fallback column."""
    primary = (
        pd.to_numeric(dataframe[primary_column], errors="coerce")
        if primary_column in dataframe.columns
        else pd.Series(np.nan, index=dataframe.index, dtype=float)
    )
    fallback = (
        pd.to_numeric(dataframe[fallback_column], errors="coerce")
        if fallback_column in dataframe.columns
        else pd.Series(np.nan, index=dataframe.index, dtype=float)
    )
    return primary.where(primary.notna(), fallback)


def sum_numeric_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    *,
    min_count: int = 1,
) -> pd.Series:
    """Sum numeric columns while preserving missingness when all inputs are missing."""
    existing_columns = [column for column in columns if column in dataframe.columns]
    if not existing_columns:
        return pd.Series(np.nan, index=dataframe.index, dtype=float)

    numeric_frame = dataframe[existing_columns].apply(pd.to_numeric, errors="coerce")
    return numeric_frame.sum(axis=1, min_count=min_count)


def prepare_decision_analytics_frame(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a comparable awarded-vs-denied subset with derived household metrics."""
    if "decision" not in dataframe.columns:
        raise KeyError("The dataset does not contain a 'decision' column.")

    decision_clean = normalize_text(dataframe["decision"])
    missing_mask = decision_clean.isna() | decision_clean.eq("")
    standard_mask = decision_clean.isin([POSITIVE_DECISION_LABEL, NEGATIVE_DECISION_LABEL])
    nonstandard_mask = (~missing_mask) & (~standard_mask)

    labeled_df = dataframe.loc[standard_mask].copy()
    labeled_df["record_id"] = labeled_df.index.astype(int)
    labeled_df["decision_clean"] = decision_clean.loc[standard_mask]
    labeled_df["award_flag"] = labeled_df["decision_clean"].eq(POSITIVE_DECISION_LABEL)

    labeled_df["father_income_effective"] = coalesce_numeric(
        labeled_df,
        "father_gross_income",
        "father_net_income",
    )
    labeled_df["mother_income_effective"] = coalesce_numeric(
        labeled_df,
        "mother_gross_income",
        "mother_net_income",
    )
    labeled_df["total_parent_income"] = (
        labeled_df[["father_income_effective", "mother_income_effective"]]
        .sum(axis=1, min_count=1)
    )
    labeled_df["total_siblings"] = sum_numeric_columns(
        labeled_df,
        ["siblings_at_aub_count", "siblings_not_at_aub_count"],
    )

    nonstandard_labels = (
        decision_clean.loc[nonstandard_mask].value_counts().sort_values(ascending=False).to_dict()
    )
    overview = {
        "rows_total": int(len(dataframe)),
        "rows_standard": int(standard_mask.sum()),
        "rows_awarded": int(labeled_df["award_flag"].sum()),
        "rows_denied": int((~labeled_df["award_flag"]).sum()),
        "rows_missing": int(missing_mask.sum()),
        "rows_nonstandard": int(nonstandard_mask.sum()),
        "award_rate": float(labeled_df["award_flag"].mean()) if not labeled_df.empty else float("nan"),
        "nonstandard_labels": nonstandard_labels,
    }
    return labeled_df, overview


def summarize_numeric_drivers(
    labeled_df: pd.DataFrame,
    *,
    metric_map: dict[str, str] | None = None,
    min_non_null_per_group: int = 30,
) -> pd.DataFrame:
    """Compare robust numeric summaries for awarded and denied applications."""
    metric_map = metric_map or NUMERIC_METRICS
    rows: list[dict[str, float | str]] = []

    for metric_column, metric_label in metric_map.items():
        if metric_column not in labeled_df.columns:
            continue

        values = pd.to_numeric(labeled_df[metric_column], errors="coerce")
        awarded_values = values[labeled_df["award_flag"]].dropna()
        denied_values = values[~labeled_df["award_flag"]].dropna()

        if len(awarded_values) < min_non_null_per_group or len(denied_values) < min_non_null_per_group:
            continue

        overall_values = values.dropna()
        overall_iqr = overall_values.quantile(0.75) - overall_values.quantile(0.25)
        awarded_median = float(awarded_values.median())
        denied_median = float(denied_values.median())
        median_gap = awarded_median - denied_median
        separation_score = (
            abs(median_gap) / float(overall_iqr)
            if pd.notna(overall_iqr) and float(overall_iqr) != 0.0
            else float("nan")
        )
        coverage_pct = float(values.notna().mean() * 100)

        rows.append(
            {
                "metric_column": metric_column,
                "metric": metric_label,
                "awarded_median": awarded_median,
                "denied_median": denied_median,
                "median_gap": median_gap,
                "separation_score": separation_score,
                "coverage_pct": coverage_pct,
                "direction": (
                    "Higher among awarded"
                    if median_gap > 0
                    else "Lower among awarded"
                    if median_gap < 0
                    else "No median gap"
                ),
            }
        )

    if not rows:
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
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["separation_score", "coverage_pct"],
        ascending=[False, False],
        ignore_index=True,
    )


def summarize_group_patterns(
    labeled_df: pd.DataFrame,
    group_column: str,
    *,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """Summarize award-rate patterns for one grouping dimension."""
    if group_column not in labeled_df.columns:
        return pd.DataFrame(
            columns=[
                "group",
                "applications",
                "awarded",
                "award_rate",
                "award_rate_pct",
                "lift_pp",
                "share_of_sample_pct",
            ]
        )

    group_values = labeled_df[group_column].astype("string").fillna("Missing").str.strip()
    group_values = group_values.mask(group_values.eq(""), "Missing")
    grouped = (
        labeled_df.assign(group=group_values)
        .groupby("group", as_index=False)["award_flag"]
        .agg(applications="size", awarded="sum", award_rate="mean")
    )
    grouped = grouped[grouped["applications"] >= min_group_size].copy()
    if grouped.empty:
        return grouped

    overall_award_rate = float(labeled_df["award_flag"].mean())
    grouped["award_rate_pct"] = grouped["award_rate"] * 100
    grouped["lift_pp"] = (grouped["award_rate"] - overall_award_rate) * 100
    grouped["share_of_sample_pct"] = grouped["applications"] / len(labeled_df) * 100
    grouped = grouped.sort_values(
        by=["lift_pp", "applications"],
        ascending=[False, False],
        ignore_index=True,
    )
    return grouped


def find_extreme_segments(
    labeled_df: pd.DataFrame,
    *,
    group_dimensions: dict[str, str] | None = None,
    min_group_size: int = 30,
    top_n: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find the strongest high-award and low-award segments across curated dimensions."""
    group_dimensions = group_dimensions or GROUP_DIMENSIONS
    all_segments: list[pd.DataFrame] = []

    for dimension_label, group_column in group_dimensions.items():
        summary = summarize_group_patterns(
            labeled_df,
            group_column,
            min_group_size=min_group_size,
        )
        if summary.empty:
            continue

        summary = summary.copy()
        summary["dimension"] = dimension_label
        all_segments.append(summary)

    if not all_segments:
        empty = pd.DataFrame(
            columns=[
                "dimension",
                "group",
                "applications",
                "award_rate",
                "award_rate_pct",
                "lift_pp",
                "share_of_sample_pct",
            ]
        )
        return empty, empty

    combined = pd.concat(all_segments, ignore_index=True)
    high_segments = combined.sort_values(
        by=["award_rate", "applications"],
        ascending=[False, False],
        ignore_index=True,
    ).head(top_n)
    low_segments = combined.sort_values(
        by=["award_rate", "applications"],
        ascending=[True, False],
        ignore_index=True,
    ).head(top_n)
    return high_segments, low_segments


def build_key_findings(
    overview: dict[str, object],
    high_segments: pd.DataFrame,
    low_segments: pd.DataFrame,
    numeric_drivers: pd.DataFrame,
) -> list[str]:
    """Generate concise narrative findings for the insights page."""
    findings: list[str] = []

    rows_standard = int(overview["rows_standard"])
    rows_total = int(overview["rows_total"])
    award_rate = float(overview["award_rate"])
    findings.append(
        f"The comparable awarded-vs-denied subset covers {rows_standard:,} of {rows_total:,} applications, "
        f"with an overall award rate of {award_rate:.1%}."
    )

    rows_missing = int(overview["rows_missing"])
    rows_nonstandard = int(overview["rows_nonstandard"])
    if rows_missing or rows_nonstandard:
        nonstandard_labels = overview.get("nonstandard_labels", {})
        label_summary = ", ".join(
            f"{label} ({count})" for label, count in list(nonstandard_labels.items())[:3]
        )
        findings.append(
            f"{rows_missing:,} decisions are missing and {rows_nonstandard:,} are non-standard; "
            f"those non-standard outcomes are excluded from the comparison set"
            f"{f' ({label_summary})' if label_summary else ''}."
        )

    if not high_segments.empty:
        strongest_high = high_segments.iloc[0]
        findings.append(
            f"The strongest high-award segment is {strongest_high['group']} in {strongest_high['dimension']}, "
            f"at {strongest_high['award_rate_pct']:.1f}% awarded "
            f"({strongest_high['lift_pp']:+.1f} percentage points vs overall, "
            f"{int(strongest_high['applications']):,} applications)."
        )

    if not low_segments.empty:
        strongest_low = low_segments.iloc[0]
        findings.append(
            f"The weakest segment is {strongest_low['group']} in {strongest_low['dimension']}, "
            f"at {strongest_low['award_rate_pct']:.1f}% awarded "
            f"({strongest_low['lift_pp']:+.1f} percentage points vs overall, "
            f"{int(strongest_low['applications']):,} applications)."
        )

    if len(numeric_drivers) >= 2:
        first_driver = numeric_drivers.iloc[0]
        second_driver = numeric_drivers.iloc[1]
        findings.append(
            f"The largest median contrasts are in {first_driver['metric']} "
            f"({first_driver['awarded_median']:,.0f} for awarded vs {first_driver['denied_median']:,.0f} for denied) "
            f"and {second_driver['metric']} "
            f"({second_driver['awarded_median']:,.0f} vs {second_driver['denied_median']:,.0f})."
        )

    return findings


def find_counterintuitive_metrics(
    numeric_drivers: pd.DataFrame,
    *,
    expected_directions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Flag metrics whose observed direction conflicts with a simple need-based expectation."""
    expected_directions = expected_directions or EXPECTED_DRIVER_DIRECTIONS
    if numeric_drivers.empty:
        return pd.DataFrame(
            columns=[
                "metric_column",
                "metric",
                "expected_pattern",
                "observed_pattern",
                "awarded_median",
                "denied_median",
                "review_note",
            ]
        )

    rows: list[dict[str, object]] = []
    for _, row in numeric_drivers.iterrows():
        metric_column = row["metric_column"]
        expected = expected_directions.get(metric_column)
        if expected is None:
            continue

        median_gap = float(row["median_gap"])
        if median_gap > 0:
            observed = "higher"
        elif median_gap < 0:
            observed = "lower"
        else:
            observed = "flat"

        if observed == "flat" or observed == expected:
            continue

        rows.append(
            {
                "metric_column": metric_column,
                "metric": row["metric"],
                "expected_pattern": f"{expected.title()} among awarded",
                "observed_pattern": f"{observed.title()} among awarded",
                "awarded_median": float(row["awarded_median"]),
                "denied_median": float(row["denied_median"]),
                "review_note": (
                    "Historical direction conflicts with a simple need-based expectation and should be reviewed."
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "metric_column",
                "metric",
                "expected_pattern",
                "observed_pattern",
                "awarded_median",
                "denied_median",
                "review_note",
            ]
        )

    return pd.DataFrame(rows)


def build_factor_review_table(
    numeric_drivers: pd.DataFrame,
    counterintuitive_metrics: pd.DataFrame,
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """Translate the strongest historical signals into factor-level review guidance."""
    if numeric_drivers.empty:
        return pd.DataFrame(
            columns=["factor", "historical_pattern", "decision_implication"]
        )

    counterintuitive_set = set(counterintuitive_metrics["metric_column"].tolist())
    rows: list[dict[str, str]] = []

    for _, row in numeric_drivers.head(top_n).iterrows():
        factor = str(row["metric"])
        awarded_median = float(row["awarded_median"])
        denied_median = float(row["denied_median"])
        direction = "higher" if awarded_median > denied_median else "lower"

        historical_pattern = (
            f"Awarded median is {direction} ({awarded_median:,.0f} vs {denied_median:,.0f})."
        )
        if row["metric_column"] in counterintuitive_set:
            implication = (
                "Do not rely on this factor blindly; the observed direction is counterintuitive and should trigger policy or data review."
            )
        else:
            implication = (
                "Treat this as a meaningful historical separator when prioritizing manual review."
            )

        rows.append(
            {
                "factor": factor,
                "historical_pattern": historical_pattern,
                "decision_implication": implication,
            }
        )

    return pd.DataFrame(rows)


def build_review_candidate_tables(
    scored_df: pd.DataFrame,
    *,
    high_threshold: float = 0.75,
    low_threshold: float = 0.25,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build case-level review queues from model probabilities."""
    if "predicted_award_probability" not in scored_df.columns:
        raise KeyError("The dataset does not contain predicted award probabilities.")

    working_df = scored_df.copy()
    working_df["decision_clean"] = normalize_text(working_df["decision"])
    working_df["decision_label"] = (
        working_df["decision"].astype("string").fillna("Missing").str.strip().replace("", "Missing")
    )
    working_df["total_parent_income"] = (
        pd.DataFrame(
            {
                "father": coalesce_numeric(working_df, "father_gross_income", "father_net_income"),
                "mother": coalesce_numeric(working_df, "mother_gross_income", "mother_net_income"),
            }
        ).sum(axis=1, min_count=1)
    )

    high_priority = (
        working_df[
            (~working_df["decision_clean"].eq(POSITIVE_DECISION_LABEL))
            & (pd.to_numeric(working_df["predicted_award_probability"], errors="coerce") >= high_threshold)
        ]
        .sort_values("predicted_award_probability", ascending=False)
        .head(top_n)
        .copy()
    )
    low_confidence_awards = (
        working_df[
            working_df["decision_clean"].eq(POSITIVE_DECISION_LABEL)
            & (pd.to_numeric(working_df["predicted_award_probability"], errors="coerce") <= low_threshold)
        ]
        .sort_values("predicted_award_probability", ascending=True)
        .head(top_n)
        .copy()
    )

    summary = {
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "high_priority_count": int(
            (
                (~working_df["decision_clean"].eq(POSITIVE_DECISION_LABEL))
                & (pd.to_numeric(working_df["predicted_award_probability"], errors="coerce") >= high_threshold)
            ).sum()
        ),
        "low_confidence_awards_count": int(
            (
                working_df["decision_clean"].eq(POSITIVE_DECISION_LABEL)
                & (pd.to_numeric(working_df["predicted_award_probability"], errors="coerce") <= low_threshold)
            ).sum()
        ),
    }
    return high_priority, low_confidence_awards, summary


def _safe_context_value(row: pd.Series, column: str, default: str = "Unknown") -> str:
    """Return a readable context value for case review tables."""
    if column not in row.index:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    value_str = str(value).strip()
    return value_str if value_str else default


def _make_quantile_band(
    series: pd.Series,
    labels: list[str] | None = None,
) -> pd.Series:
    """Create coarse quantile bands with a graceful fallback for tied values."""
    labels = labels or ["Low", "Mid-Low", "Mid-High", "High"]
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty:
        return pd.Series("Unknown", index=series.index, dtype="string")
    if non_null.nunique() == 1:
        return numeric.map(lambda value: "Mid" if pd.notna(value) else "Unknown").astype("string")

    try:
        return pd.qcut(numeric, q=4, labels=labels, duplicates="drop").astype("string").fillna("Unknown")
    except ValueError:
        ranked = numeric.rank(method="first")
        unique_bins = min(4, int(non_null.nunique()))
        return (
            pd.qcut(
                ranked,
                q=unique_bins,
                labels=labels[:unique_bins],
                duplicates="drop",
            )
            .astype("string")
            .fillna("Unknown")
        )


def _make_count_band(series: pd.Series) -> pd.Series:
    """Create simple count-oriented buckets for household-size features."""
    numeric = pd.to_numeric(series, errors="coerce")
    return (
        pd.cut(
            numeric,
            bins=[-1, 0, 1, 2, np.inf],
            labels=["0", "1", "2", "3+"],
        )
        .astype("string")
        .fillna("Unknown")
    )


def find_similar_profile_review_pairs(
    labeled_df: pd.DataFrame,
    *,
    top_n: int = 10,
    min_shared_features: int = 4,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Find denied applications that closely resemble awarded applications on core financial features."""
    available_features = [
        column for column in INCONSISTENCY_SIMILARITY_FEATURES if column in labeled_df.columns
    ]
    if len(available_features) < min_shared_features:
        empty = pd.DataFrame()
        return empty, {
            "review_pair_count": 0,
            "review_threshold": float("nan"),
            "applications_with_match": 0,
            "feature_count": len(available_features),
            "features_used": available_features,
        }

    working_df = labeled_df.copy()
    if "record_id" not in working_df.columns:
        working_df["record_id"] = working_df.index.astype(int)

    for column in available_features:
        numeric = pd.to_numeric(working_df[column], errors="coerce")
        lower_bound = float(numeric.quantile(0.01)) if numeric.notna().any() else np.nan
        upper_bound = float(numeric.quantile(0.99)) if numeric.notna().any() else np.nan
        working_df[column] = numeric.clip(lower=lower_bound, upper=upper_bound)

    feature_frame = working_df[available_features]
    medians = feature_frame.median()
    iqr = (feature_frame.quantile(0.75) - feature_frame.quantile(0.25)).replace(0, 1)
    scaled = (feature_frame.fillna(medians) - medians) / iqr
    masks = feature_frame.notna().to_numpy()
    values = scaled.to_numpy(dtype=float)
    decisions = working_df["decision_clean"].to_numpy()
    levels = working_df["level"].astype("string").fillna("Missing").to_numpy()

    denied_indexes = np.where(decisions == NEGATIVE_DECISION_LABEL)[0]
    awarded_indexes = np.where(decisions == POSITIVE_DECISION_LABEL)[0]
    pair_rows: list[dict[str, object]] = []

    for denied_idx in denied_indexes:
        target_pool = awarded_indexes[levels[awarded_indexes] == levels[denied_idx]]
        if len(target_pool) == 0:
            target_pool = awarded_indexes
        if len(target_pool) == 0:
            continue

        differences = np.abs(values[target_pool] - values[denied_idx])
        overlap = masks[target_pool] & masks[denied_idx]
        shared_features = overlap.sum(axis=1)
        valid_matches = shared_features >= min_shared_features
        if not valid_matches.any():
            continue

        overlap_ratio = shared_features / len(available_features)
        base_distance = np.where(
            valid_matches,
            (differences * overlap).sum(axis=1) / shared_features,
            np.inf,
        )
        distance = np.where(
            valid_matches,
            base_distance + (1.0 - overlap_ratio) * 0.25,
            np.inf,
        )
        best_position = int(np.argmin(distance))
        best_distance = float(distance[best_position])
        if not np.isfinite(best_distance):
            continue

        awarded_idx = int(target_pool[best_position])
        denied_row = working_df.iloc[denied_idx]
        awarded_row = working_df.iloc[awarded_idx]

        if (
            _safe_context_value(denied_row, "application_type_clean")
            != _safe_context_value(awarded_row, "application_type_clean")
        ):
            review_note = "Application type differs, so verify whether policy rules explain the divergent decision."
        elif (
            _safe_context_value(denied_row, "special_family_circumstances_category")
            != _safe_context_value(awarded_row, "special_family_circumstances_category")
        ):
            review_note = "Special-circumstance coding differs, so confirm whether contextual hardship explains the decision split."
        else:
            review_note = "Displayed financial profile is very similar, so this pair deserves consistency review."

        pair_rows.append(
            {
                "denied_record_id": int(denied_row["record_id"]),
                "awarded_record_id": int(awarded_row["record_id"]),
                "level": _safe_context_value(denied_row, "level"),
                "distance": best_distance,
                "shared_features": int(shared_features[best_position]),
                "overlap_ratio": float(overlap_ratio[best_position]),
                "denied_school": _safe_context_value(denied_row, "school"),
                "awarded_school": _safe_context_value(awarded_row, "school"),
                "denied_application_type": _safe_context_value(denied_row, "application_type_clean"),
                "awarded_application_type": _safe_context_value(awarded_row, "application_type_clean"),
                "denied_special_circumstances": _safe_context_value(
                    denied_row,
                    "special_family_circumstances_category",
                ),
                "awarded_special_circumstances": _safe_context_value(
                    awarded_row,
                    "special_family_circumstances_category",
                ),
                "denied_total_parent_income": pd.to_numeric(
                    denied_row.get("total_parent_income"),
                    errors="coerce",
                ),
                "awarded_total_parent_income": pd.to_numeric(
                    awarded_row.get("total_parent_income"),
                    errors="coerce",
                ),
                "denied_properties_total_estimated_value": pd.to_numeric(
                    denied_row.get("properties_total_estimated_value"),
                    errors="coerce",
                ),
                "awarded_properties_total_estimated_value": pd.to_numeric(
                    awarded_row.get("properties_total_estimated_value"),
                    errors="coerce",
                ),
                "denied_dependents_count": pd.to_numeric(
                    denied_row.get("dependents_count"),
                    errors="coerce",
                ),
                "awarded_dependents_count": pd.to_numeric(
                    awarded_row.get("dependents_count"),
                    errors="coerce",
                ),
                "denied_total_siblings": pd.to_numeric(
                    denied_row.get("total_siblings"),
                    errors="coerce",
                ),
                "awarded_total_siblings": pd.to_numeric(
                    awarded_row.get("total_siblings"),
                    errors="coerce",
                ),
                "review_note": review_note,
            }
        )

    if not pair_rows:
        return pd.DataFrame(), {
            "review_pair_count": 0,
            "review_threshold": float("nan"),
            "applications_with_match": 0,
            "feature_count": len(available_features),
            "features_used": available_features,
        }

    pair_df = pd.DataFrame(pair_rows).sort_values(
        by=["distance", "shared_features"],
        ascending=[True, False],
        ignore_index=True,
    )
    matched_denied_count = int(pair_df["denied_record_id"].nunique())
    pair_df = pair_df.drop_duplicates(subset=["awarded_record_id"], keep="first").reset_index(drop=True)

    review_threshold = (
        float(pair_df["distance"].quantile(0.15))
        if len(pair_df) >= 10
        else float(pair_df["distance"].max())
    )
    flagged_pairs = pair_df[pair_df["distance"] <= review_threshold].copy()
    display_pairs = flagged_pairs.head(top_n) if not flagged_pairs.empty else pair_df.head(top_n)

    summary = {
        "review_pair_count": int(len(flagged_pairs)),
        "review_threshold": review_threshold,
        "applications_with_match": matched_denied_count,
        "display_pair_count": int(len(pair_df)),
        "feature_count": len(available_features),
        "features_used": available_features,
        "median_distance": float(pair_df["distance"].median()),
    }
    return display_pairs, summary


def summarize_mixed_profile_groups(
    labeled_df: pd.DataFrame,
    *,
    min_group_size: int = 6,
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Summarize groups with similar financial buckets but mixed decisions."""
    working_df = labeled_df.copy()
    band_columns: dict[str, str] = {}

    for feature_column, feature_label in INCONSISTENCY_PROFILE_FEATURES.items():
        if feature_column not in working_df.columns:
            continue

        if feature_column in {"dependents_count", "total_siblings"}:
            working_df[f"{feature_column}_band"] = _make_count_band(working_df[feature_column])
        else:
            working_df[f"{feature_column}_band"] = _make_quantile_band(working_df[feature_column])
        band_columns[feature_column] = f"{feature_column}_band"

    if len(band_columns) < 3 or "level" not in working_df.columns:
        return pd.DataFrame(), {
            "mixed_group_count": 0,
            "applications_in_mixed_groups": 0,
            "balanced_group_count": 0,
        }

    working_df["level_band"] = working_df["level"].astype("string").fillna("Missing").str.strip()
    working_df["level_band"] = working_df["level_band"].replace("", "Missing")
    grouped_columns = ["level_band"] + list(band_columns.values())
    working_df["known_feature_count"] = sum(
        working_df[column].ne("Unknown") for column in band_columns.values()
    )

    group_summary = (
        working_df.groupby(grouped_columns, as_index=False)
        .agg(
            applications=("award_flag", "size"),
            awarded=("award_flag", "sum"),
            known_feature_count=("known_feature_count", "median"),
        )
    )
    group_summary["denied"] = group_summary["applications"] - group_summary["awarded"]
    group_summary["award_rate"] = group_summary["awarded"] / group_summary["applications"]
    group_summary["award_rate_pct"] = group_summary["award_rate"] * 100
    group_summary["review_score"] = group_summary[["awarded", "denied"]].min(axis=1)
    group_summary["balance_gap_pp"] = (group_summary["award_rate"] - 0.5).abs() * 100

    mixed_groups = group_summary[
        (group_summary["applications"] >= min_group_size)
        & (group_summary["awarded"] > 0)
        & (group_summary["denied"] > 0)
        & (group_summary["known_feature_count"] >= 3)
    ].copy()
    if mixed_groups.empty:
        return mixed_groups, {
            "mixed_group_count": 0,
            "applications_in_mixed_groups": 0,
            "balanced_group_count": 0,
        }

    renamed_columns = {
        "level_band": "Level",
        "total_parent_income_band": "Parent Income",
        "properties_total_estimated_value_band": "Property Value",
        "dependents_count_band": "Dependents",
        "total_siblings_band": "Siblings",
        "siblings_other_total_tuition_band": "Other Siblings Tuition",
    }
    mixed_groups = mixed_groups.rename(columns=renamed_columns).sort_values(
        by=["review_score", "balance_gap_pp", "applications"],
        ascending=[False, True, False],
        ignore_index=True,
    )

    summary = {
        "mixed_group_count": int(len(mixed_groups)),
        "applications_in_mixed_groups": int(mixed_groups["applications"].sum()),
        "balanced_group_count": int((mixed_groups["balance_gap_pp"] <= 15).sum()),
    }
    return mixed_groups.head(top_n), summary


def build_inconsistency_review_observations(
    review_pairs: pd.DataFrame,
    review_pair_summary: dict[str, object],
    mixed_groups: pd.DataFrame,
    mixed_group_summary: dict[str, object],
) -> list[str]:
    """Turn inconsistency analytics into plain-English review observations."""
    observations: list[str] = []

    if int(review_pair_summary.get("applications_with_match", 0)) > 0:
        observations.append(
            f"{int(review_pair_summary['applications_with_match']):,} denied applications have at least one comparable awarded match on four or more shared financial features, which suggests that some denials deserve secondary review."
        )
    if int(review_pair_summary.get("review_pair_count", 0)) > 0:
        observations.append(
            f"{int(review_pair_summary['review_pair_count']):,} of those pairs fall within the closest similarity band (distance <= {float(review_pair_summary['review_threshold']):.3f}), making them the sharpest case-level inconsistency signals."
        )
    if int(mixed_group_summary.get("mixed_group_count", 0)) > 0:
        observations.append(
            f"{int(mixed_group_summary['mixed_group_count']):,} financial profile groups contain both awarded and denied outcomes, so policy application is not fully uniform even within similar buckets."
        )
    if not mixed_groups.empty:
        top_group = mixed_groups.iloc[0]
        observations.append(
            f"The most review-worthy mixed profile currently shown has {int(top_group['applications']):,} applications split {int(top_group['awarded'])} awarded to {int(top_group['denied'])} denied, which is a meaningful divergence within one profile bucket."
        )
    return observations


def build_decision_guidance(
    overview: dict[str, object],
    numeric_drivers: pd.DataFrame,
    high_segments: pd.DataFrame,
    low_segments: pd.DataFrame,
    counterintuitive_metrics: pd.DataFrame,
    *,
    review_summary: dict[str, object] | None = None,
    model_factors: list[str] | None = None,
) -> dict[str, list[str]]:
    """Translate analytics outputs into decision-support answers."""
    who_should_receive_aid: list[str] = []
    factors_that_matter: list[str] = []
    inconsistency_flags: list[str] = []
    counterintuitive_set = set(counterintuitive_metrics["metric_column"].tolist())
    stable_drivers = numeric_drivers[
        ~numeric_drivers["metric_column"].isin(counterintuitive_set)
    ].reset_index(drop=True)
    driver_source = stable_drivers if not stable_drivers.empty else numeric_drivers

    if len(driver_source) >= 2:
        top_factor = driver_source.iloc[0]
        second_factor = driver_source.iloc[1]
        who_should_receive_aid.append(
            f"Historically, aid recipients look most different on {top_factor['metric'].lower()} and {second_factor['metric'].lower()}, so applicants with those same need-like profiles deserve closer consideration."
        )
    if not high_segments.empty:
        top_segment = high_segments.iloc[0]
        who_should_receive_aid.append(
            f"{top_segment['group']} within {top_segment['dimension']} has the strongest historical award rate at {top_segment['award_rate_pct']:.1f}%, making it a useful benchmark for high-priority profiles."
        )
    if review_summary and int(review_summary.get("high_priority_count", 0)) > 0:
        who_should_receive_aid.append(
            f"The model flags {int(review_summary['high_priority_count']):,} non-awarded or unresolved cases above the high-support threshold, so those applications are good candidates for manual re-check."
        )

    if not driver_source.empty:
        top_factor = driver_source.iloc[0]
        factors_that_matter.append(
            f"{top_factor['metric']} is the strongest historical separator, with awarded applications centered at {top_factor['awarded_median']:,.0f} versus {top_factor['denied_median']:,.0f} for denied applications."
        )
    if len(driver_source) > 1:
        second_factor = driver_source.iloc[1]
        factors_that_matter.append(
            f"{second_factor['metric']} is the next strongest signal, so it should remain visible in any scoring rubric or manual review checklist."
        )
    if model_factors:
        factors_that_matter.append(
            f"The fitted model also concentrates on {', '.join(model_factors[:3])}, which reinforces that these variables matter even after joint modeling."
        )

    if int(overview.get("rows_missing", 0)) > 0 or int(overview.get("rows_nonstandard", 0)) > 0:
        inconsistency_flags.append(
            f"There are {int(overview.get('rows_missing', 0)):,} missing decisions and {int(overview.get('rows_nonstandard', 0)):,} non-standard outcomes, so the decision pipeline is not fully clean or standardized."
        )
    if not counterintuitive_metrics.empty:
        first_issue = counterintuitive_metrics.iloc[0]
        inconsistency_flags.append(
            f"{first_issue['metric']} behaves counter to a simple need-based expectation, so that field should be audited for coding issues, policy exceptions, or hidden confounders."
        )
    if not low_segments.empty:
        weakest_segment = low_segments.iloc[0]
        inconsistency_flags.append(
            f"{weakest_segment['group']} in {weakest_segment['dimension']} sits well below the baseline award rate at {weakest_segment['award_rate_pct']:.1f}%, which warrants a fairness or consistency review."
        )
    if review_summary and int(review_summary.get("low_confidence_awards_count", 0)) > 0:
        inconsistency_flags.append(
            f"The model sees {int(review_summary['low_confidence_awards_count']):,} awarded cases below the low-confidence threshold, so there are historical approvals that do not align well with the broader pattern."
        )

    return {
        "who_should_receive_aid": who_should_receive_aid,
        "factors_that_matter": factors_that_matter,
        "inconsistency_flags": inconsistency_flags,
    }
