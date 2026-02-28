"""Task recommendation and scoring weight endpoints.

Extracted from src/api/tasks.py — handles recommendation queries and
scoring weight CRUD (presets, updates, resets).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException

from src.task_registry import TASK_CATEGORIES


def build_recommendations_router() -> APIRouter:
    """Create an APIRouter with recommendation and scoring-weight endpoints."""
    router = APIRouter()

    # ===== Recommendations =====

    @router.get("/tasks/recommendations")
    async def get_recommendations(
        limit: int = 5,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get task recommendations based on execution history.

        Args:
            limit: Maximum number of recommendations (default: 5)
            category: Prefer tasks from this category
        """
        from src.tasks.orchestration.recommendations import get_task_recommendations

        recommendations = get_task_recommendations(
            limit=limit,
            prefer_category=category,
        )

        return {
            "recommendations": recommendations,
            "count": len(recommendations),
        }

    @router.get("/tasks/recommendations/quick-scan")
    async def get_quick_scan_recommendations() -> Dict[str, Any]:
        """Get recommendations for a quick codebase scan."""
        from src.tasks.orchestration.recommendations import get_recommendation_engine

        engine = get_recommendation_engine()
        recommendations = engine.get_quick_scan_recommendation()

        return {
            "recommendations": [r.to_dict() for r in recommendations],
            "description": "Essential tasks for a quick codebase overview",
        }

    @router.get("/tasks/recommendations/category/{category}")
    async def get_category_recommendations(
        category: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Get recommendations for tasks in a specific category."""
        from src.tasks.orchestration.recommendations import get_recommendation_engine

        if category not in TASK_CATEGORIES:
            raise HTTPException(
                status_code=404,
                detail=f"Category not found: {category}. Available: {list(TASK_CATEGORIES.keys())}"
            )

        engine = get_recommendation_engine()
        recommendations = engine.get_category_recommendations(category, limit)

        return {
            "category": category,
            "recommendations": [r.to_dict() for r in recommendations],
        }

    @router.get("/tasks/recommendations/related/{task_name}")
    async def get_related_recommendations(
        task_name: str,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """Get recommendations for tasks related to a given task."""
        from src.tasks.orchestration.recommendations import get_recommendation_engine
        from src.task_registry import TASK_REGISTRY

        if task_name not in TASK_REGISTRY:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {task_name}"
            )

        engine = get_recommendation_engine()
        recommendations = engine.get_related_recommendations(task_name, limit)

        return {
            "task_name": task_name,
            "related_recommendations": [r.to_dict() for r in recommendations],
        }

    # ===== Scoring Weights =====

    @router.get("/tasks/recommendations/weights")
    async def get_scoring_weights() -> Dict[str, Any]:
        """Get current scoring weights for task recommendations."""
        from src.tasks.orchestration.recommendations import get_scoring_weights as get_weights

        return {
            "weights": get_weights(),
            "description": "Current scoring weights used for task recommendations",
        }

    @router.put("/tasks/recommendations/weights")
    async def update_scoring_weights(
        request: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        """Update scoring weights for task recommendations.

        Body should contain weight names and values, e.g.:
        {"never_run_boost": 40.0, "stale_boost": 25.0}

        Available weights:
        - never_run_boost: Boost for tasks never executed (default: 30.0)
        - stale_boost: Boost for tasks not run recently (default: 20.0)
        - very_stale_boost: Additional boost for very old tasks (default: 25.0)
        - high_success_rate_boost: Boost for reliable tasks (default: 10.0)
        - low_success_rate_penalty: Penalty for failing tasks (default: -15.0)
        - fast_execution_boost: Boost for quick tasks (default: 5.0)
        - category_coverage_boost: Boost for underrepresented categories (default: 15.0)
        - dependency_boost: Boost for tasks with dependencies ready (default: 10.0)
        """
        from src.tasks.orchestration.recommendations import update_scoring_weights as update_weights

        # Filter to only float values
        weight_updates = {k: float(v) for k, v in request.items() if isinstance(v, (int, float))}

        if not weight_updates:
            raise HTTPException(
                status_code=400,
                detail="No valid weight updates provided"
            )

        updated = update_weights(**weight_updates)

        return {
            "weights": updated,
            "updated_fields": list(weight_updates.keys()),
        }

    @router.post("/tasks/recommendations/weights/reset")
    async def reset_scoring_weights() -> Dict[str, Any]:
        """Reset scoring weights to default values."""
        from src.tasks.orchestration.recommendations import reset_scoring_weights as reset_weights

        defaults = reset_weights()

        return {
            "weights": defaults,
            "message": "Scoring weights reset to defaults",
        }

    @router.get("/tasks/recommendations/weights/presets")
    async def get_weight_presets() -> Dict[str, Any]:
        """Get available scoring weight presets.

        Presets provide pre-configured weight combinations for different use cases:
        - default: Balanced recommendations
        - aggressive: Prioritize coverage and freshness
        - conservative: Prioritize reliability and speed
        - reliability-focused: Strongly prefer tasks that succeed
        - coverage-focused: Strongly prefer running all tasks
        """
        from src.tasks.orchestration.recommendations import get_weight_presets as get_presets

        presets = get_presets()

        return {
            "presets": presets,
            "available": list(presets.keys()),
        }

    @router.post("/tasks/recommendations/weights/presets/{preset_name}")
    async def apply_weight_preset(preset_name: str) -> Dict[str, Any]:
        """Apply a scoring weight preset.

        Available presets: default, aggressive, conservative,
        reliability-focused, coverage-focused
        """
        from src.tasks.orchestration.recommendations import apply_weight_preset as apply_preset

        result = apply_preset(preset_name)

        if result is None:
            from src.tasks.orchestration.recommendations import get_weight_presets as get_presets
            available = list(get_presets().keys())
            raise HTTPException(
                status_code=404,
                detail=f"Preset not found: {preset_name}. Available: {available}"
            )

        return {
            "preset": preset_name,
            "weights": result,
            "message": f"Applied preset: {preset_name}",
        }

    return router
