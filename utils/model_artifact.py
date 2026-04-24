from __future__ import annotations

from typing import Any


MODEL_ARTIFACT_TYPE = "yasmina_eligibility_bundle"


def is_model_bundle(model: Any) -> bool:
    """Return True when the loaded object looks like a packaged model bundle."""
    return isinstance(model, dict) and (
        model.get("artifact_type") == MODEL_ARTIFACT_TYPE or "estimator" in model
    )


def resolve_prediction_estimator(model: Any) -> Any:
    """Return the deployed estimator used for scoring probabilities."""
    if is_model_bundle(model):
        return model.get("estimator")
    return model


def resolve_explainability_estimator(model: Any) -> Any:
    """Return the estimator best suited for feature importance and driver charts."""
    if is_model_bundle(model):
        return model.get("explainability_estimator") or model.get("estimator")
    return model


def resolve_selected_threshold(model: Any, default: float = 0.50) -> float:
    """Recover the deployment threshold from a model bundle when available."""
    if is_model_bundle(model):
        threshold = model.get("selected_threshold")
        if threshold is not None:
            try:
                return float(threshold)
            except (TypeError, ValueError):
                return float(default)
    return float(default)


def resolve_model_metadata(model: Any) -> dict[str, Any]:
    """Expose non-estimator bundle fields for downstream reporting when available."""
    if is_model_bundle(model):
        return {
            key: value
            for key, value in model.items()
            if key not in {"estimator", "explainability_estimator"}
        }
    return {}
