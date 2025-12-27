"""
Tests for Prediction Pipeline

Ensures prediction pipeline works correctly with various inputs.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.pipeline.predict_pipeline import PredictionPipeline, PredictionPipelineConfig


class TestPredictionPipeline:
    """Tests for prediction pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create prediction pipeline instance."""
        # Check if models exist
        models_dir = Path("artifacts/models")
        if not models_dir.exists() or not list(models_dir.glob("*_model.joblib")):
            pytest.skip("No trained models found - run training pipeline first")
        
        return PredictionPipeline()
    
    @pytest.fixture
    def valid_store_product(self):
        """Valid store-product combination for testing."""
        return {"store_id": 1, "product_id": 1, "horizon": 7}
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes successfully."""
        assert pipeline is not None
        assert pipeline.model is not None, "Model should be loaded"
        assert pipeline.scaler is not None, "Scaler should be loaded"
    
    def test_single_prediction(self, pipeline, valid_store_product):
        """Test single prediction returns expected format."""
        result = pipeline.predict(**valid_store_product)
        
        assert "store_id" in result
        assert "product_id" in result
        assert "predicted_quantity" in result
        assert "model_used" in result
        assert result["predicted_quantity"] >= 0, "Prediction should be non-negative"
    
    def test_prediction_different_horizons(self, pipeline):
        """Test predictions work for all valid horizons."""
        store_id, product_id = 1, 1
        
        for horizon in [7, 14, 30]:
            result = pipeline.predict(store_id, product_id, horizon)
            assert result["horizon_days"] == horizon
            assert result["predicted_quantity"] >= 0
    
    def test_batch_predictions(self, pipeline):
        """Test batch prediction functionality."""
        requests = [
            {"store_id": 1, "product_id": 1, "horizon": 7},
            {"store_id": 1, "product_id": 2, "horizon": 14},
            {"store_id": 2, "product_id": 1, "horizon": 30},
        ]
        
        results = pipeline.predict_batch(requests)
        
        assert len(results) == 3
        assert all("predicted_quantity" in r for r in results)
    
    def test_prediction_consistency(self, pipeline):
        """Test same input gives consistent predictions."""
        store_id, product_id, horizon = 1, 1, 7
        
        result1 = pipeline.predict(store_id, product_id, horizon)
        result2 = pipeline.predict(store_id, product_id, horizon)
        
        # Same inputs should give same output
        assert result1["predicted_quantity"] == result2["predicted_quantity"]
    
    def test_prediction_with_custom_features(self, pipeline):
        """Test prediction with custom feature dict."""
        features = {
            "store_id": 5,
            "product_id": 100,
            "sales_lag_7": 150,
            "rolling_mean_7": 145,
            "day_of_week": 5,
            "is_weekend": True,
        }
        
        result = pipeline.predict(5, 100, 7, features=features)
        assert result["predicted_quantity"] >= 0
    
    def test_invalid_store_id_handling(self, pipeline):
        """Test pipeline handles invalid store IDs gracefully."""
        # Very large store ID should still work (uses default features)
        result = pipeline.predict(store_id=9999, product_id=1, horizon=7)
        assert "predicted_quantity" in result
    
    def test_boundary_values(self, pipeline):
        """Test edge cases and boundary values."""
        # Minimum values
        result = pipeline.predict(1, 1, 7)
        assert result["predicted_quantity"] >= 0
        
        # Maximum configured values
        result = pipeline.predict(50, 1000, 30)
        assert result["predicted_quantity"] >= 0


class TestPredictionPipelineConfig:
    """Tests for prediction pipeline configuration."""
    
    def test_config_initialization(self):
        """Test config initializes with defaults."""
        config = PredictionPipelineConfig()
        
        assert config.models_dir == "artifacts/models"
        assert config.default_model == "lightgbm"
        assert len(config.forecast_horizons) == 3
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PredictionPipelineConfig(
            default_model="xgboost",
            forecast_horizons=[7, 30]
        )
        
        assert config.default_model == "xgboost"
        assert config.forecast_horizons == [7, 30]
