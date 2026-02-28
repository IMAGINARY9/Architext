"""Compatibility shim — re-exports from src.tasks.orchestration.recommendations.

All implementations live in src/tasks/orchestration/recommendations.py.
Import from ``src.tasks.orchestration`` or ``src.tasks`` for the public API.
"""
from src.tasks.orchestration.recommendations import *  # noqa: F401,F403
from src.tasks.orchestration.recommendations import (  # explicit re-exports for type checkers
    TaskRecommendation,
    ScoringWeights,
    ScoringWeightsStore,
    RecommendationConfig,
    TaskRecommendationEngine,
    get_recommendation_engine,
    get_task_recommendations,
    get_scoring_weights,
    update_scoring_weights,
    reset_scoring_weights,
    get_weight_presets,
    apply_weight_preset,
)

__all__ = [
    "TaskRecommendation",
    "ScoringWeights",
    "ScoringWeightsStore",
    "RecommendationConfig",
    "TaskRecommendationEngine",
    "get_recommendation_engine",
    "get_task_recommendations",
    "get_scoring_weights",
    "update_scoring_weights",
    "reset_scoring_weights",
    "get_weight_presets",
    "apply_weight_preset",
]
