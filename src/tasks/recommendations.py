"""Task recommendation engine based on execution history.

This module provides intelligent task recommendations based on:
- Historical execution patterns
- Task success rates
- Time since last execution
- Related task analysis
- Customizable scoring weights (configurable and persistent)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.tasks.history import get_task_history, TaskExecution, TaskAnalytics
from src.task_registry import TASK_REGISTRY, TASK_CATEGORIES, TASK_DEPENDENCIES


# Default weights file location
WEIGHTS_DIR = Path.home() / ".architext" / "config"
WEIGHTS_FILE = WEIGHTS_DIR / "scoring_weights.json"


@dataclass
class TaskRecommendation:
    """A recommended task with reasoning."""
    task_name: str
    score: float  # 0-100, higher is more recommended
    reasons: List[str]
    category: Optional[str] = None
    last_run: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_name": self.task_name,
            "score": round(self.score, 1),
            "reasons": self.reasons,
            "category": self.category,
            "last_run": self.last_run.isoformat() if self.last_run else None,
        }


@dataclass
class ScoringWeights:
    """
    Customizable scoring weights for recommendations.
    
    All weights can be adjusted to tune the recommendation algorithm.
    Higher weights increase the influence of that factor on the final score.
    """
    # Time-based weights (how much to boost based on recency)
    never_run_boost: float = 30.0  # Boost for tasks never executed
    stale_boost: float = 20.0  # Boost for tasks not run recently
    very_stale_boost: float = 25.0  # Additional boost for very old tasks
    
    # Performance-based weights
    high_success_rate_boost: float = 10.0  # Boost for reliable tasks
    low_success_rate_penalty: float = -15.0  # Penalty for failing tasks
    fast_execution_boost: float = 5.0  # Boost for quick tasks
    
    # Coverage weights
    category_coverage_boost: float = 15.0  # Boost for underrepresented categories
    dependency_boost: float = 10.0  # Boost for tasks with dependencies ready
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringWeights":
        """Create from dictionary."""
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    def update(self, **kwargs: float) -> "ScoringWeights":
        """Update weights and return new instance."""
        current = self.to_dict()
        current.update(kwargs)
        return ScoringWeights.from_dict(current)


@dataclass
class RecommendationConfig:
    """Configuration for the recommendation engine."""
    # Time-based thresholds
    stale_threshold_hours: int = 24  # Consider task stale after this
    very_stale_threshold_hours: int = 168  # One week
    
    # Performance thresholds
    success_rate_high: float = 90.0
    success_rate_low: float = 50.0
    fast_execution_seconds: float = 5.0
    
    # Scoring weights (customizable)
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stale_threshold_hours": self.stale_threshold_hours,
            "very_stale_threshold_hours": self.very_stale_threshold_hours,
            "success_rate_high": self.success_rate_high,
            "success_rate_low": self.success_rate_low,
            "fast_execution_seconds": self.fast_execution_seconds,
            "weights": self.weights.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecommendationConfig":
        """Create from dictionary."""
        weights_data = data.pop("weights", {})
        weights = ScoringWeights.from_dict(weights_data) if weights_data else ScoringWeights()
        return cls(weights=weights, **{k: v for k, v in data.items() if k != "weights"})


class ScoringWeightsStore:
    """
    Persistent storage for scoring weights.
    
    Allows users to save and load custom weight configurations.
    """
    
    def __init__(self, path: Optional[Path] = None):
        """Initialize the store."""
        self.path = path or WEIGHTS_FILE
        self._ensure_dir()
    
    def _ensure_dir(self) -> None:
        """Ensure the storage directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def save(self, weights: ScoringWeights) -> None:
        """Save weights to disk."""
        with open(self.path, "w") as f:
            json.dump(weights.to_dict(), f, indent=2)
    
    def load(self) -> ScoringWeights:
        """Load weights from disk, or return defaults."""
        if not self.path.exists():
            return ScoringWeights()
        try:
            with open(self.path) as f:
                data = json.load(f)
            return ScoringWeights.from_dict(data)
        except Exception:
            return ScoringWeights()
    
    def reset(self) -> ScoringWeights:
        """Reset to default weights."""
        defaults = ScoringWeights()
        self.save(defaults)
        return defaults
    
    def update(self, **kwargs: float) -> ScoringWeights:
        """Update specific weights and save."""
        current = self.load()
        updated = current.update(**kwargs)
        self.save(updated)
        return updated
    
    def get_presets(self) -> Dict[str, ScoringWeights]:
        """Get predefined weight presets."""
        return {
            "default": ScoringWeights(),
            "aggressive": ScoringWeights(
                never_run_boost=50.0,
                stale_boost=30.0,
                very_stale_boost=35.0,
                high_success_rate_boost=5.0,
                low_success_rate_penalty=-25.0,
                fast_execution_boost=3.0,
                category_coverage_boost=20.0,
            ),
            "conservative": ScoringWeights(
                never_run_boost=15.0,
                stale_boost=10.0,
                very_stale_boost=15.0,
                high_success_rate_boost=20.0,
                low_success_rate_penalty=-5.0,
                fast_execution_boost=10.0,
                category_coverage_boost=10.0,
            ),
            "reliability-focused": ScoringWeights(
                never_run_boost=10.0,
                stale_boost=5.0,
                very_stale_boost=10.0,
                high_success_rate_boost=30.0,
                low_success_rate_penalty=-30.0,
                fast_execution_boost=15.0,
                category_coverage_boost=5.0,
            ),
            "coverage-focused": ScoringWeights(
                never_run_boost=40.0,
                stale_boost=25.0,
                very_stale_boost=30.0,
                high_success_rate_boost=5.0,
                low_success_rate_penalty=-10.0,
                fast_execution_boost=2.0,
                category_coverage_boost=25.0,
            ),
        }
    
    def apply_preset(self, preset_name: str) -> Optional[ScoringWeights]:
        """Apply a predefined preset."""
        presets = self.get_presets()
        if preset_name not in presets:
            return None
        preset = presets[preset_name]
        self.save(preset)
        return preset


class TaskRecommendationEngine:
    """
    Engine for generating task recommendations.
    
    Uses execution history, task relationships, and time-based heuristics
    to suggest which tasks should be run next.
    
    Supports customizable scoring weights that can be persisted and loaded.
    """
    
    def __init__(
        self,
        config: Optional[RecommendationConfig] = None,
        weights: Optional[ScoringWeights] = None,
    ):
        """
        Initialize the recommendation engine.
        
        Args:
            config: Full configuration (optional)
            weights: Scoring weights (optional, loaded from disk if not provided)
        """
        self._weights_store = ScoringWeightsStore()
        
        if config:
            self.config = config
        else:
            # Load weights from disk if not provided
            loaded_weights = weights or self._weights_store.load()
            self.config = RecommendationConfig(weights=loaded_weights)
        
        self._history = get_task_history()
        self._task_to_category = self._build_task_category_map()
    
    def _build_task_category_map(self) -> Dict[str, str]:
        """Build mapping from task name to category."""
        mapping = {}
        for category, tasks in TASK_CATEGORIES.items():
            for task in tasks:
                mapping[task] = category
        return mapping
    
    def get_weights(self) -> ScoringWeights:
        """Get current scoring weights."""
        return self.config.weights
    
    def set_weights(self, weights: ScoringWeights, persist: bool = True) -> None:
        """
        Set scoring weights.
        
        Args:
            weights: New weights to use
            persist: Whether to save to disk
        """
        self.config.weights = weights
        if persist:
            self._weights_store.save(weights)
    
    def update_weights(self, persist: bool = True, **kwargs: float) -> ScoringWeights:
        """
        Update specific weights.
        
        Args:
            persist: Whether to save to disk
            **kwargs: Weight values to update
            
        Returns:
            Updated weights
        """
        current = self.config.weights
        updated = current.update(**kwargs)
        self.config.weights = updated
        if persist:
            self._weights_store.save(updated)
        return updated
    
    def reset_weights(self, persist: bool = True) -> ScoringWeights:
        """Reset weights to defaults."""
        defaults = ScoringWeights()
        self.config.weights = defaults
        if persist:
            self._weights_store.save(defaults)
        return defaults
    
    def apply_preset(self, preset_name: str, persist: bool = True) -> Optional[ScoringWeights]:
        """
        Apply a predefined weight preset.
        
        Available presets: default, aggressive, conservative, 
        reliability-focused, coverage-focused
        """
        preset = self._weights_store.apply_preset(preset_name) if persist else None
        if preset is None:
            presets = self._weights_store.get_presets()
            if preset_name not in presets:
                return None
            preset = presets[preset_name]
        self.config.weights = preset
        return preset
    
    def get_presets(self) -> Dict[str, Dict[str, float]]:
        """Get available weight presets."""
        return {name: w.to_dict() for name, w in self._weights_store.get_presets().items()}
    
    def get_recommendations(
        self,
        limit: int = 5,
        exclude_tasks: Optional[Set[str]] = None,
        prefer_category: Optional[str] = None,
    ) -> List[TaskRecommendation]:
        """
        Get task recommendations sorted by score.
        
        Args:
            limit: Maximum number of recommendations
            exclude_tasks: Tasks to exclude from recommendations
            prefer_category: Boost tasks in this category
            
        Returns:
            List of TaskRecommendation sorted by score (descending)
        """
        exclude_tasks = exclude_tasks or set()
        recommendations = []
        
        # Get history and analytics for all tasks
        all_history = self._history.get_history()
        
        for task_name in TASK_REGISTRY:
            if task_name in exclude_tasks:
                continue
            
            recommendation = self._score_task(
                task_name,
                all_history,
                prefer_category,
            )
            recommendations.append(recommendation)
        
        # Sort by score descending
        recommendations.sort(key=lambda r: r.score, reverse=True)
        
        return recommendations[:limit]
    
    def _score_task(
        self,
        task_name: str,
        all_history: List[TaskExecution],
        prefer_category: Optional[str],
    ) -> TaskRecommendation:
        """Calculate recommendation score for a task."""
        config = self.config
        weights = config.weights
        score = 50.0  # Base score
        reasons: List[str] = []
        
        # Get task-specific history
        task_history = [h for h in all_history if h.task_name == task_name]
        
        # Get category
        category = self._task_to_category.get(task_name)
        
        # Last run time
        last_run: Optional[datetime] = None
        if task_history:
            last_run = max(h.started_at for h in task_history)
        
        # Never run boost
        if not task_history:
            score += weights.never_run_boost
            reasons.append("Never been executed")
        else:
            # Time since last run
            hours_since = self._hours_since(last_run)
            
            if hours_since > config.very_stale_threshold_hours:
                score += weights.very_stale_boost
                days = int(hours_since / 24)
                reasons.append(f"Not run in {days} days")
            elif hours_since > config.stale_threshold_hours:
                score += weights.stale_boost
                reasons.append(f"Not run in {int(hours_since)} hours")
        
        # Success rate analysis
        if task_history:
            analytics = self._history.get_analytics(task_name)
            if analytics:
                if analytics.success_rate >= config.success_rate_high:
                    score += weights.high_success_rate_boost
                    reasons.append(f"High reliability ({analytics.success_rate:.0f}% success)")
                elif analytics.success_rate < config.success_rate_low:
                    score += weights.low_success_rate_penalty
                    reasons.append(f"Low reliability ({analytics.success_rate:.0f}% success)")
                
                # Fast execution boost
                if analytics.average_duration_seconds < config.fast_execution_seconds:
                    score += weights.fast_execution_boost
                    reasons.append("Quick to execute")
        
        # Category preference boost
        if prefer_category and category == prefer_category:
            score += weights.category_coverage_boost
            reasons.append(f"Matches preferred category: {prefer_category}")
        
        # Category coverage - boost tasks in underrepresented categories
        if category:
            category_tasks = TASK_CATEGORIES.get(category, [])
            category_runs = sum(
                1 for h in all_history
                if h.task_name in category_tasks
            )
            if category_runs == 0:
                score += weights.category_coverage_boost / 2
                reasons.append(f"Category '{category}' not recently analyzed")
        
        return TaskRecommendation(
            task_name=task_name,
            score=max(0, min(100, score)),  # Clamp to 0-100
            reasons=reasons if reasons else ["Standard recommendation"],
            category=category,
            last_run=last_run,
        )
    
    @staticmethod
    def _hours_since(dt: Optional[datetime]) -> float:
        """Calculate hours since a datetime."""
        if dt is None:
            return float("inf")
        delta = datetime.now() - dt
        return delta.total_seconds() / 3600
    
    def get_quick_scan_recommendation(self) -> List[TaskRecommendation]:
        """
        Get recommendations for a quick codebase scan.
        
        Returns fast, essential tasks for a quick overview.
        """
        quick_tasks = {"analyze-structure", "tech-stack", "health-score"}
        recommendations = []
        
        for task_name in quick_tasks:
            if task_name in TASK_REGISTRY:
                rec = self._score_task(task_name, [], None)
                rec.score = 100.0  # Max score for quick scan
                rec.reasons = ["Essential for quick overview"]
                recommendations.append(rec)
        
        return recommendations
    
    def get_category_recommendations(
        self,
        category: str,
        limit: int = 5,
    ) -> List[TaskRecommendation]:
        """
        Get recommendations for a specific category.
        
        Args:
            category: Category name (quality, security, architecture, etc.)
            limit: Maximum recommendations
            
        Returns:
            Recommendations for tasks in the category
        """
        if category not in TASK_CATEGORIES:
            return []
        
        category_tasks = set(TASK_CATEGORIES[category])
        return self.get_recommendations(
            limit=limit,
            exclude_tasks=set(TASK_REGISTRY.keys()) - category_tasks,
            prefer_category=category,
        )
    
    def get_related_recommendations(
        self,
        task_name: str,
        limit: int = 3,
    ) -> List[TaskRecommendation]:
        """
        Get recommendations for tasks related to a given task.
        
        Uses task dependencies and category membership.
        """
        related: Set[str] = set()
        
        # Add tasks in the same category
        category = self._task_to_category.get(task_name)
        if category:
            related.update(TASK_CATEGORIES.get(category, []))
        
        # Add dependent tasks
        for task, deps in TASK_DEPENDENCIES.items():
            if task_name in deps:
                related.add(task)
        
        # Add tasks that this task depends on
        related.update(TASK_DEPENDENCIES.get(task_name, set()))
        
        # Remove the original task
        related.discard(task_name)
        
        if not related:
            return []
        
        # Get recommendations for related tasks only
        return self.get_recommendations(
            limit=limit,
            exclude_tasks=set(TASK_REGISTRY.keys()) - related,
        )


# Singleton instance
_recommendation_engine: Optional[TaskRecommendationEngine] = None


def get_recommendation_engine() -> TaskRecommendationEngine:
    """Get the singleton recommendation engine instance."""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = TaskRecommendationEngine()
    return _recommendation_engine


def get_task_recommendations(
    limit: int = 5,
    exclude_tasks: Optional[Set[str]] = None,
    prefer_category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get task recommendations as dictionaries.
    
    Convenience function for API usage.
    """
    engine = get_recommendation_engine()
    recommendations = engine.get_recommendations(
        limit=limit,
        exclude_tasks=exclude_tasks,
        prefer_category=prefer_category,
    )
    return [r.to_dict() for r in recommendations]


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


# Scoring weights helper functions
def get_scoring_weights() -> Dict[str, float]:
    """Get current scoring weights as dictionary."""
    engine = get_recommendation_engine()
    return engine.get_weights().to_dict()


def update_scoring_weights(**kwargs: float) -> Dict[str, float]:
    """Update scoring weights and return new values."""
    engine = get_recommendation_engine()
    updated = engine.update_weights(**kwargs)
    return updated.to_dict()


def reset_scoring_weights() -> Dict[str, float]:
    """Reset scoring weights to defaults."""
    engine = get_recommendation_engine()
    defaults = engine.reset_weights()
    return defaults.to_dict()


def get_weight_presets() -> Dict[str, Dict[str, float]]:
    """Get available weight presets."""
    engine = get_recommendation_engine()
    return engine.get_presets()


def apply_weight_preset(preset_name: str) -> Optional[Dict[str, float]]:
    """Apply a weight preset."""
    engine = get_recommendation_engine()
    result = engine.apply_preset(preset_name)
    return result.to_dict() if result else None
