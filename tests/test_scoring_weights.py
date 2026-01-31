"""Tests for customizable scoring weights in the recommendation engine."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.tasks.recommendations import (
    ScoringWeights,
    ScoringWeightsStore,
    RecommendationConfig,
    TaskRecommendationEngine,
    get_scoring_weights,
    update_scoring_weights,
    reset_scoring_weights,
    get_weight_presets,
    apply_weight_preset,
)


class TestScoringWeights:
    """Tests for the ScoringWeights dataclass."""
    
    def test_default_weights(self):
        """Test default weight values."""
        weights = ScoringWeights()
        
        assert weights.never_run_boost == 30.0
        assert weights.stale_boost == 20.0
        assert weights.very_stale_boost == 25.0
        assert weights.high_success_rate_boost == 10.0
        assert weights.low_success_rate_penalty == -15.0
        assert weights.fast_execution_boost == 5.0
        assert weights.category_coverage_boost == 15.0
        assert weights.dependency_boost == 10.0
    
    def test_custom_weights(self):
        """Test creating weights with custom values."""
        weights = ScoringWeights(
            never_run_boost=50.0,
            stale_boost=30.0,
        )
        
        assert weights.never_run_boost == 50.0
        assert weights.stale_boost == 30.0
        # Other values should be defaults
        assert weights.very_stale_boost == 25.0
    
    def test_to_dict(self):
        """Test converting weights to dictionary."""
        weights = ScoringWeights()
        result = weights.to_dict()
        
        assert isinstance(result, dict)
        assert "never_run_boost" in result
        assert result["never_run_boost"] == 30.0
    
    def test_from_dict(self):
        """Test creating weights from dictionary."""
        data = {
            "never_run_boost": 40.0,
            "stale_boost": 25.0,
            "invalid_field": "ignored",
        }
        
        weights = ScoringWeights.from_dict(data)
        
        assert weights.never_run_boost == 40.0
        assert weights.stale_boost == 25.0
        # Invalid fields should be ignored
    
    def test_update(self):
        """Test updating specific weights."""
        weights = ScoringWeights()
        updated = weights.update(never_run_boost=45.0, stale_boost=22.0)
        
        # Original should be unchanged
        assert weights.never_run_boost == 30.0
        
        # Updated should have new values
        assert updated.never_run_boost == 45.0
        assert updated.stale_boost == 22.0


class TestScoringWeightsStore:
    """Tests for the ScoringWeightsStore persistence."""
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading weights."""
        path = tmp_path / "weights.json"
        store = ScoringWeightsStore(path)
        
        weights = ScoringWeights(never_run_boost=42.0)
        store.save(weights)
        
        loaded = store.load()
        assert loaded.never_run_boost == 42.0
    
    def test_load_default_when_missing(self, tmp_path):
        """Test loading returns defaults when file doesn't exist."""
        path = tmp_path / "nonexistent.json"
        store = ScoringWeightsStore(path)
        
        loaded = store.load()
        assert loaded.never_run_boost == 30.0  # Default value
    
    def test_reset(self, tmp_path):
        """Test resetting weights to defaults."""
        path = tmp_path / "weights.json"
        store = ScoringWeightsStore(path)
        
        # Save custom weights
        store.save(ScoringWeights(never_run_boost=99.0))
        
        # Reset
        reset = store.reset()
        
        assert reset.never_run_boost == 30.0
        
        # Verify persisted
        loaded = store.load()
        assert loaded.never_run_boost == 30.0
    
    def test_update(self, tmp_path):
        """Test updating specific weights."""
        path = tmp_path / "weights.json"
        store = ScoringWeightsStore(path)
        
        updated = store.update(never_run_boost=55.0)
        
        assert updated.never_run_boost == 55.0
        
        # Verify persisted
        loaded = store.load()
        assert loaded.never_run_boost == 55.0
    
    def test_get_presets(self, tmp_path):
        """Test getting predefined presets."""
        path = tmp_path / "weights.json"
        store = ScoringWeightsStore(path)
        
        presets = store.get_presets()
        
        assert "default" in presets
        assert "aggressive" in presets
        assert "conservative" in presets
        assert "reliability-focused" in presets
        assert "coverage-focused" in presets
        
        # Verify presets have different values
        assert presets["aggressive"].never_run_boost > presets["default"].never_run_boost
    
    def test_apply_preset(self, tmp_path):
        """Test applying a preset."""
        path = tmp_path / "weights.json"
        store = ScoringWeightsStore(path)
        
        result = store.apply_preset("aggressive")
        
        assert result is not None
        assert result.never_run_boost == 50.0
        
        # Verify persisted
        loaded = store.load()
        assert loaded.never_run_boost == 50.0
    
    def test_apply_invalid_preset(self, tmp_path):
        """Test applying an invalid preset returns None."""
        path = tmp_path / "weights.json"
        store = ScoringWeightsStore(path)
        
        result = store.apply_preset("invalid_preset")
        assert result is None


class TestRecommendationConfig:
    """Tests for RecommendationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RecommendationConfig()
        
        assert config.stale_threshold_hours == 24
        assert config.very_stale_threshold_hours == 168
        assert config.success_rate_high == 90.0
        assert isinstance(config.weights, ScoringWeights)
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = RecommendationConfig()
        result = config.to_dict()
        
        assert "stale_threshold_hours" in result
        assert "weights" in result
        assert isinstance(result["weights"], dict)
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "stale_threshold_hours": 48,
            "weights": {"never_run_boost": 40.0},
        }
        
        config = RecommendationConfig.from_dict(data)
        
        assert config.stale_threshold_hours == 48
        assert config.weights.never_run_boost == 40.0


class TestTaskRecommendationEngineWeights:
    """Tests for TaskRecommendationEngine scoring weight support."""
    
    @patch("src.tasks.recommendations.get_task_history")
    @patch("src.tasks.recommendations.TASK_REGISTRY", {"task-a": lambda: None, "task-b": lambda: None})
    @patch("src.tasks.recommendations.TASK_CATEGORIES", {"cat1": ["task-a", "task-b"]})
    def test_get_weights(self, mock_history):
        """Test getting current weights."""
        mock_history.return_value = MagicMock(get_history=lambda: [], get_analytics=lambda x: None)
        
        engine = TaskRecommendationEngine()
        weights = engine.get_weights()
        
        assert isinstance(weights, ScoringWeights)
    
    @patch("src.tasks.recommendations.get_task_history")
    @patch("src.tasks.recommendations.TASK_REGISTRY", {"task-a": lambda: None})
    @patch("src.tasks.recommendations.TASK_CATEGORIES", {"cat1": ["task-a"]})
    def test_set_weights(self, mock_history, tmp_path):
        """Test setting weights."""
        mock_history.return_value = MagicMock(get_history=lambda: [], get_analytics=lambda x: None)
        
        # Use temp path for weights file
        with patch("src.tasks.recommendations.WEIGHTS_FILE", tmp_path / "weights.json"):
            engine = TaskRecommendationEngine()
            
            new_weights = ScoringWeights(never_run_boost=60.0)
            engine.set_weights(new_weights, persist=False)
            
            assert engine.get_weights().never_run_boost == 60.0
    
    @patch("src.tasks.recommendations.get_task_history")
    @patch("src.tasks.recommendations.TASK_REGISTRY", {"task-a": lambda: None})
    @patch("src.tasks.recommendations.TASK_CATEGORIES", {"cat1": ["task-a"]})
    def test_update_weights(self, mock_history, tmp_path):
        """Test updating specific weights."""
        mock_history.return_value = MagicMock(get_history=lambda: [], get_analytics=lambda x: None)
        
        with patch("src.tasks.recommendations.WEIGHTS_FILE", tmp_path / "weights.json"):
            engine = TaskRecommendationEngine()
            
            updated = engine.update_weights(persist=False, never_run_boost=45.0)
            
            assert updated.never_run_boost == 45.0
            assert engine.get_weights().never_run_boost == 45.0
    
    @patch("src.tasks.recommendations.get_task_history")
    @patch("src.tasks.recommendations.TASK_REGISTRY", {"task-a": lambda: None})
    @patch("src.tasks.recommendations.TASK_CATEGORIES", {"cat1": ["task-a"]})
    def test_apply_preset(self, mock_history, tmp_path):
        """Test applying a preset."""
        mock_history.return_value = MagicMock(get_history=lambda: [], get_analytics=lambda x: None)
        
        with patch("src.tasks.recommendations.WEIGHTS_FILE", tmp_path / "weights.json"):
            engine = TaskRecommendationEngine()
            
            result = engine.apply_preset("aggressive", persist=False)
            
            assert result is not None
            assert result.never_run_boost == 50.0
    
    @patch("src.tasks.recommendations.get_task_history")
    @patch("src.tasks.recommendations.TASK_REGISTRY", {"task-a": lambda: None})
    @patch("src.tasks.recommendations.TASK_CATEGORIES", {"cat1": ["task-a"]})
    def test_get_presets(self, mock_history, tmp_path):
        """Test getting preset list."""
        mock_history.return_value = MagicMock(get_history=lambda: [], get_analytics=lambda x: None)
        
        with patch("src.tasks.recommendations.WEIGHTS_FILE", tmp_path / "weights.json"):
            engine = TaskRecommendationEngine()
            
            presets = engine.get_presets()
            
            assert "default" in presets
            assert "aggressive" in presets
            assert isinstance(presets["default"], dict)


class TestWeightEffectOnScoring:
    """Tests verifying weights actually affect scoring."""
    
    @patch("src.tasks.recommendations.get_task_history")
    @patch("src.tasks.recommendations.TASK_REGISTRY", {"new-task": lambda: None})
    @patch("src.tasks.recommendations.TASK_CATEGORIES", {"cat1": ["new-task"]})
    def test_never_run_boost_affects_score(self, mock_history, tmp_path):
        """Test that never_run_boost affects score for new tasks."""
        mock_history.return_value = MagicMock(
            get_history=lambda: [],  # Empty history
            get_analytics=lambda x: None,
        )
        
        with patch("src.tasks.recommendations.WEIGHTS_FILE", tmp_path / "weights.json"):
            # Low boost
            engine1 = TaskRecommendationEngine(weights=ScoringWeights(never_run_boost=10.0))
            recs1 = engine1.get_recommendations(limit=1)
            
            # High boost
            engine2 = TaskRecommendationEngine(weights=ScoringWeights(never_run_boost=50.0))
            recs2 = engine2.get_recommendations(limit=1)
            
            # Higher boost should result in higher score
            assert recs2[0].score > recs1[0].score


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    @patch("src.tasks.recommendations._recommendation_engine", None)
    @patch("src.tasks.recommendations.get_task_history")
    @patch("src.tasks.recommendations.TASK_REGISTRY", {"task-a": lambda: None})
    @patch("src.tasks.recommendations.TASK_CATEGORIES", {"cat1": ["task-a"]})
    def test_get_scoring_weights_func(self, mock_history, tmp_path):
        """Test get_scoring_weights function."""
        mock_history.return_value = MagicMock(get_history=lambda: [], get_analytics=lambda x: None)
        
        with patch("src.tasks.recommendations.WEIGHTS_FILE", tmp_path / "weights.json"):
            result = get_scoring_weights()
            
            assert isinstance(result, dict)
            assert "never_run_boost" in result
    
    @patch("src.tasks.recommendations._recommendation_engine", None)
    @patch("src.tasks.recommendations.get_task_history")
    @patch("src.tasks.recommendations.TASK_REGISTRY", {"task-a": lambda: None})
    @patch("src.tasks.recommendations.TASK_CATEGORIES", {"cat1": ["task-a"]})
    def test_get_weight_presets_func(self, mock_history, tmp_path):
        """Test get_weight_presets function."""
        mock_history.return_value = MagicMock(get_history=lambda: [], get_analytics=lambda x: None)
        
        with patch("src.tasks.recommendations.WEIGHTS_FILE", tmp_path / "weights.json"):
            result = get_weight_presets()
            
            assert isinstance(result, dict)
            assert "default" in result
            assert "aggressive" in result
