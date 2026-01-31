"""Tests for task recommendation engine."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.tasks.recommendations import (
    TaskRecommendation,
    RecommendationConfig,
    TaskRecommendationEngine,
    get_recommendation_engine,
    get_task_recommendations,
)


class TestTaskRecommendation:
    """Tests for TaskRecommendation dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        rec = TaskRecommendation(
            task_name="analyze-structure",
            score=85.5,
            reasons=["Never been executed", "Essential task"],
            category="analysis",
            last_run=datetime(2024, 1, 15, 10, 30),
        )
        
        d = rec.to_dict()
        
        assert d["task_name"] == "analyze-structure"
        assert d["score"] == 85.5
        assert len(d["reasons"]) == 2
        assert d["category"] == "analysis"
        assert d["last_run"] == "2024-01-15T10:30:00"
    
    def test_to_dict_without_last_run(self):
        """Test conversion when last_run is None."""
        rec = TaskRecommendation(
            task_name="test-task",
            score=50.0,
            reasons=["Default"],
        )
        
        d = rec.to_dict()
        
        assert d["last_run"] is None


class TestRecommendationConfig:
    """Tests for RecommendationConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = RecommendationConfig()
        
        assert config.stale_threshold_hours == 24
        assert config.very_stale_threshold_hours == 168
        # Scoring weights are now in a separate ScoringWeights object
        assert config.weights is not None
        assert config.weights.never_run_boost == 30.0
        assert config.success_rate_high == 90.0
    
    def test_custom_values(self):
        """Test custom configuration."""
        from src.tasks.recommendations import ScoringWeights
        custom_weights = ScoringWeights(never_run_boost=50.0)
        config = RecommendationConfig(
            stale_threshold_hours=12,
            weights=custom_weights,
        )
        
        assert config.stale_threshold_hours == 12
        assert config.weights.never_run_boost == 50.0


class TestTaskRecommendationEngine:
    """Tests for TaskRecommendationEngine."""
    
    @patch("src.tasks.recommendations.get_task_history")
    def test_get_recommendations_returns_list(self, mock_history):
        """Test that recommendations returns a list."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        engine = TaskRecommendationEngine()
        recommendations = engine.get_recommendations(limit=5)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
    
    @patch("src.tasks.recommendations.get_task_history")
    def test_recommendations_sorted_by_score(self, mock_history):
        """Test that recommendations are sorted by score descending."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        engine = TaskRecommendationEngine()
        recommendations = engine.get_recommendations(limit=10)
        
        scores = [r.score for r in recommendations]
        assert scores == sorted(scores, reverse=True)
    
    @patch("src.tasks.recommendations.get_task_history")
    def test_never_run_tasks_boosted(self, mock_history):
        """Test that tasks never run get a boost."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []  # No history
        mock_history.return_value = mock_history_instance
        
        engine = TaskRecommendationEngine()
        recommendations = engine.get_recommendations()
        
        # All tasks should have "Never been executed" reason
        for rec in recommendations:
            assert "Never been executed" in rec.reasons
    
    @patch("src.tasks.recommendations.get_task_history")
    def test_exclude_tasks(self, mock_history):
        """Test that excluded tasks are not recommended."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        engine = TaskRecommendationEngine()
        recommendations = engine.get_recommendations(
            exclude_tasks={"analyze-structure", "tech-stack"}
        )
        
        task_names = [r.task_name for r in recommendations]
        assert "analyze-structure" not in task_names
        assert "tech-stack" not in task_names
    
    @patch("src.tasks.recommendations.get_task_history")
    def test_prefer_category_boosts_tasks(self, mock_history):
        """Test that preferred category tasks get a boost."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        engine = TaskRecommendationEngine()
        recommendations = engine.get_recommendations(
            prefer_category="quality",
            limit=10,
        )
        
        # At least one task should mention category preference
        quality_tasks = [r for r in recommendations if r.category == "quality"]
        if quality_tasks:
            # Check that category preference reason exists
            assert any(
                "preferred category" in " ".join(r.reasons).lower()
                for r in quality_tasks
            )
    
    @patch("src.tasks.recommendations.get_task_history")
    def test_get_quick_scan_recommendation(self, mock_history):
        """Test quick scan recommendations."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        engine = TaskRecommendationEngine()
        recommendations = engine.get_quick_scan_recommendation()
        
        assert len(recommendations) > 0
        # All should have max score
        for rec in recommendations:
            assert rec.score == 100.0
            assert "Essential" in rec.reasons[0] or "quick" in rec.reasons[0].lower()
    
    @patch("src.tasks.recommendations.get_task_history")
    def test_get_category_recommendations(self, mock_history):
        """Test category-specific recommendations."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        engine = TaskRecommendationEngine()
        recommendations = engine.get_category_recommendations("security", limit=3)
        
        # All should be in security category
        for rec in recommendations:
            assert rec.category == "security"
    
    @patch("src.tasks.recommendations.get_task_history")
    def test_get_category_recommendations_invalid_category(self, mock_history):
        """Test that invalid category returns empty list."""
        mock_history_instance = MagicMock()
        mock_history.return_value = mock_history_instance
        
        engine = TaskRecommendationEngine()
        recommendations = engine.get_category_recommendations("nonexistent")
        
        assert recommendations == []
    
    @patch("src.tasks.recommendations.get_task_history")
    def test_get_related_recommendations(self, mock_history):
        """Test related task recommendations."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        engine = TaskRecommendationEngine()
        recommendations = engine.get_related_recommendations("health-score", limit=3)
        
        # Should not include the original task
        task_names = [r.task_name for r in recommendations]
        assert "health-score" not in task_names


class TestSingletonAndConvenienceFunctions:
    """Tests for singleton and convenience functions."""
    
    @patch("src.tasks.recommendations.get_task_history")
    def test_get_recommendation_engine_singleton(self, mock_history):
        """Test that singleton returns same instance."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        # Reset singleton
        import src.tasks.recommendations as rec_module
        rec_module._recommendation_engine = None
        
        engine1 = get_recommendation_engine()
        engine2 = get_recommendation_engine()
        
        assert engine1 is engine2
    
    @patch("src.tasks.recommendations.get_task_history")
    def test_get_task_recommendations_returns_dicts(self, mock_history):
        """Test that convenience function returns dictionaries."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        recommendations = get_task_recommendations(limit=3)
        
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert isinstance(rec, dict)
            assert "task_name" in rec
            assert "score" in rec
            assert "reasons" in rec
