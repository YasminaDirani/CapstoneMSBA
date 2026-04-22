"""Utility modules for the Streamlit app."""

__all__ = [
    "load_csv",
    "load_model",
    "add_prediction_probabilities",
    "build_roc_curve_data",
    "build_confusion_matrix_data",
    "extract_top_feature_importances",
    "extract_top_feature_drivers",
    "prepare_decision_analytics_frame",
    "summarize_numeric_drivers",
    "summarize_group_patterns",
    "find_extreme_segments",
    "build_key_findings",
    "find_counterintuitive_metrics",
    "build_factor_review_table",
    "build_review_candidate_tables",
    "build_decision_guidance",
    "find_similar_profile_review_pairs",
    "summarize_mixed_profile_groups",
    "build_inconsistency_review_observations",
]


def __getattr__(name: str):
    if name == "load_csv":
        from utils.data_loader import load_csv

        return load_csv

    if name == "load_model":
        from utils.model_loader import load_model

        return load_model

    if name == "add_prediction_probabilities":
        from utils.prediction_utils import add_prediction_probabilities

        return add_prediction_probabilities

    if name == "build_roc_curve_data":
        from utils.prediction_utils import build_roc_curve_data

        return build_roc_curve_data

    if name == "build_confusion_matrix_data":
        from utils.prediction_utils import build_confusion_matrix_data

        return build_confusion_matrix_data

    if name == "extract_top_feature_importances":
        from utils.model_explainability import extract_top_feature_importances

        return extract_top_feature_importances

    if name == "extract_top_feature_drivers":
        from utils.model_explainability import extract_top_feature_drivers

        return extract_top_feature_drivers

    if name == "prepare_decision_analytics_frame":
        from utils.decision_analytics import prepare_decision_analytics_frame

        return prepare_decision_analytics_frame

    if name == "summarize_numeric_drivers":
        from utils.decision_analytics import summarize_numeric_drivers

        return summarize_numeric_drivers

    if name == "summarize_group_patterns":
        from utils.decision_analytics import summarize_group_patterns

        return summarize_group_patterns

    if name == "find_extreme_segments":
        from utils.decision_analytics import find_extreme_segments

        return find_extreme_segments

    if name == "build_key_findings":
        from utils.decision_analytics import build_key_findings

        return build_key_findings

    if name == "find_counterintuitive_metrics":
        from utils.decision_analytics import find_counterintuitive_metrics

        return find_counterintuitive_metrics

    if name == "build_factor_review_table":
        from utils.decision_analytics import build_factor_review_table

        return build_factor_review_table

    if name == "build_review_candidate_tables":
        from utils.decision_analytics import build_review_candidate_tables

        return build_review_candidate_tables

    if name == "build_decision_guidance":
        from utils.decision_analytics import build_decision_guidance

        return build_decision_guidance

    if name == "find_similar_profile_review_pairs":
        from utils.decision_analytics import find_similar_profile_review_pairs

        return find_similar_profile_review_pairs

    if name == "summarize_mixed_profile_groups":
        from utils.decision_analytics import summarize_mixed_profile_groups

        return summarize_mixed_profile_groups

    if name == "build_inconsistency_review_observations":
        from utils.decision_analytics import build_inconsistency_review_observations

        return build_inconsistency_review_observations

    raise AttributeError(f"module 'utils' has no attribute {name!r}")
