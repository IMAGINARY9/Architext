"""Task recommendation engine based on execution history.

This module provides intelligent task recommendations based on:
- Historical execution patterns
- Task success rates
- Time since last execution
- Related task analysis
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from src.tasks.history import get_task_history, TaskExecution, TaskAnalytics
from src.task_registry import TASK_REGISTRY, TASK_CATEGORIES, TASK_DEPENDENCIES


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
class RecommendationConfig:
    """Configuration for the recommendation engine."""
    # Time-based weights
    stale_threshold_hours: int = 24  # Consider task stale after this
    very_stale_threshold_hours: int = 168  # One week
    
    # Score weights
    never_run_boost: float = 30.0
    stale_boost: float = 20.0
    very_stale_boost: float = 25.0
    high_success_rate_boost: float = 10.0
    low_success_rate_penalty: float = -15.0
    fast_execution_boost: float = 5.0
    category_coverage_boost: float = 15.0
    
    # Thresholds
    success_rate_high: float = 90.0
    success_rate_low: float = 50.0
    fast_execution_seconds: float = 5.0


class TaskRecommendationEngine:
    """
    Engine for generating task recommendations.
    
    Uses execution history, task relationships, and time-based heuristics
    to suggest which tasks should be run next.
    """
    
    def __init__(self, config: Optional[RecommendationConfig] = None):
        """Initialize the recommendation engine."""
        self.config = config or RecommendationConfig()
        self._history = get_task_history()
        self._task_to_category = self._build_task_category_map()
    
    def _build_task_category_map(self) -> Dict[str, str]:
        """Build mapping from task name to category."""
        mapping = {}
        for category, tasks in TASK_CATEGORIES.items():
            for task in tasks:
                mapping[task] = category
        return mapping
    
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
            score += config.never_run_boost
            reasons.append("Never been executed")
        else:
            # Time since last run
            hours_since = self._hours_since(last_run)
            
            if hours_since > config.very_stale_threshold_hours:
                score += config.very_stale_boost
                days = int(hours_since / 24)
                reasons.append(f"Not run in {days} days")
            elif hours_since > config.stale_threshold_hours:
                score += config.stale_boost
                reasons.append(f"Not run in {int(hours_since)} hours")
        
        # Success rate analysis
        if task_history:
            analytics = self._history.get_analytics(task_name)
            if analytics:
                if analytics.success_rate >= config.success_rate_high:
                    score += config.high_success_rate_boost
                    reasons.append(f"High reliability ({analytics.success_rate:.0f}% success)")
                elif analytics.success_rate < config.success_rate_low:
                    score += config.low_success_rate_penalty
                    reasons.append(f"Low reliability ({analytics.success_rate:.0f}% success)")
                
                # Fast execution boost
                if analytics.average_duration_seconds < config.fast_execution_seconds:
                    score += config.fast_execution_boost
                    reasons.append("Quick to execute")
        
        # Category preference boost
        if prefer_category and category == prefer_category:
            score += config.category_coverage_boost
            reasons.append(f"Matches preferred category: {prefer_category}")
        
        # Category coverage - boost tasks in underrepresented categories
        if category:
            category_tasks = TASK_CATEGORIES.get(category, [])
            category_runs = sum(
                1 for h in all_history
                if h.task_name in category_tasks
            )
            if category_runs == 0:
                score += config.category_coverage_boost / 2
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
    "RecommendationConfig",
    "TaskRecommendationEngine",
    "get_recommendation_engine",
    "get_task_recommendations",
]
