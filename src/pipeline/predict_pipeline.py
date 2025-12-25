"""
Prediction Pipeline

Pipeline for generating demand forecasts using trained models.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.exception import PredictionException
from src.logger import get_logger
from src.utils import load_config, load_object

logger = get_logger(__name__)


@dataclass
class PredictionPipelineConfig:
    """Configuration for prediction pipeline."""
    models_dir: str = "artifacts/models"
    default_model: str = "lightgbm"  # lowercase
    forecast_horizons: List[int] = None
    
    def __post_init__(self):
        if self.forecast_horizons is None:
            self.forecast_horizons = [7, 14, 30]


class PredictionPipeline:
    """
    Pipeline for generating demand forecasts.
    
    Loads trained models and generates predictions for specified
    store-product combinations and forecast horizons.
    """
    
    def __init__(self, config: PredictionPipelineConfig = None):
        """
        Initialize prediction pipeline.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config or PredictionPipelineConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Load model and preprocessing artifacts
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load trained model and preprocessing artifacts."""
        try:
            models_dir = Path(self.config.models_dir)
            
            # Load model
            model_path = models_dir / f"{self.config.default_model}_model.joblib"
            if model_path.exists():
                self.model = load_object(str(model_path))
                self.logger.info(f"Loaded model: {self.config.default_model}")
            else:
                # Try to find any model
                model_files = list(models_dir.glob("*_model.joblib"))
                if model_files:
                    self.model = load_object(str(model_files[0]))
                    self.logger.info(f"Loaded model: {model_files[0].stem}")
                else:
                    self.logger.warning("No trained model found")
            
            # Load scaler
            scaler_path = models_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = load_object(str(scaler_path))
            
            # Load label encoders
            encoders_path = models_dir / "label_encoders.joblib"
            if encoders_path.exists():
                self.label_encoders = load_object(str(encoders_path))
            
            # Load feature names
            import json
            features_path = models_dir / "feature_names.json"
            if features_path.exists():
                with open(features_path, "r") as f:
                    self.feature_names = json.load(f)
                    
        except Exception as e:
            self.logger.error(f"Failed to load artifacts: {e}")
    
    def predict(
        self,
        store_id: int,
        product_id: int,
        horizon: int = 7,
        features: Dict = None,
    ) -> Dict:
        """
        Generate demand forecast for a store-product combination.
        
        Args:
            store_id: Store ID.
            product_id: Product ID.
            horizon: Forecast horizon in days.
            features: Optional feature dictionary.
            
        Returns:
            Dictionary with prediction results.
        """
        try:
            if self.model is None:
                raise PredictionException("No model loaded")
            
            # Create feature vector
            if features is None:
                features = self._create_default_features(store_id, product_id)
            
            # Prepare features in correct order
            X = self._prepare_features(features)
            
            # Scale features
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Predict
            prediction = self.model.predict(X)[0]
            
            # Ensure non-negative
            prediction = max(0, prediction)
            
            result = {
                "store_id": store_id,
                "product_id": product_id,
                "horizon_days": horizon,
                "predicted_quantity": round(prediction, 2),
                "model_used": self.config.default_model,
            }
            
            self.logger.debug(f"Prediction: {result}")
            
            return result
            
        except Exception as e:
            raise PredictionException(f"Prediction failed: {e}", sys)
    
    def predict_batch(
        self,
        requests: List[Dict],
    ) -> List[Dict]:
        """
        Generate predictions for multiple store-product combinations.
        
        Args:
            requests: List of dictionaries with store_id, product_id, horizon.
            
        Returns:
            List of prediction results.
        """
        results = []
        
        for req in requests:
            try:
                result = self.predict(
                    store_id=req.get("store_id"),
                    product_id=req.get("product_id"),
                    horizon=req.get("horizon", 7),
                    features=req.get("features"),
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch prediction error: {e}")
                results.append({
                    "store_id": req.get("store_id"),
                    "product_id": req.get("product_id"),
                    "error": str(e),
                })
        
        return results
    
    def _create_default_features(
        self,
        store_id: int,
        product_id: int
    ) -> Dict:
        """Create default feature values for prediction."""
        return {
            "store_id": store_id,
            "product_id": product_id,
            "sales_lag_7": 100,
            "sales_lag_14": 95,
            "sales_lag_30": 90,
            "qty_lag_7": 50,
            "qty_lag_14": 48,
            "qty_lag_30": 45,
            "rolling_mean_7": 100,
            "rolling_std_7": 15,
            "rolling_mean_14": 98,
            "rolling_std_14": 18,
            "rolling_mean_30": 95,
            "rolling_std_30": 20,
            "rolling_min_7": 70,
            "rolling_max_7": 130,
            "day_of_week": 3,
            "week_of_year": 25,
            "month": 6,
            "quarter": 2,
            "is_weekend": False,
            "is_holiday": False,
            "days_to_holiday": 30,
            "store_size_sqft": 50000,
            "product_price": 5.0,
            "is_perishable": False,
            "temp_high_f": 75,
            "temp_low_f": 55,
            "precipitation_inches": 0,
            "has_promotion": False,
            "promotion_discount": 0,
            "store_avg_daily_sales_30d": 5000,
            "product_avg_daily_sales_30d": 500,
        }
    
    def _prepare_features(self, features: Dict) -> np.ndarray:
        """Prepare feature vector for prediction."""
        if self.feature_names is None:
            self.feature_names = list(features.keys())
        
        # Build feature vector in correct order
        X = []
        for col in self.feature_names:
            value = features.get(col, 0)
            
            # Encode categorical if needed
            if self.label_encoders and col in self.label_encoders:
                try:
                    value = self.label_encoders[col].transform([str(value)])[0]
                except ValueError:
                    value = 0  # Unknown category
            
            # Convert boolean
            if isinstance(value, bool):
                value = int(value)
            
            X.append(value)
        
        # Return as DataFrame with feature names to avoid LightGBM warning
        return pd.DataFrame([X], columns=self.feature_names)


if __name__ == "__main__":
    # Test prediction pipeline
    pipeline = PredictionPipeline()
    
    # Single prediction
    result = pipeline.predict(store_id=1, product_id=1, horizon=7)
    print(f"Single prediction: {result}")
    
    # Batch predictions
    batch_requests = [
        {"store_id": 1, "product_id": 1, "horizon": 7},
        {"store_id": 1, "product_id": 2, "horizon": 14},
        {"store_id": 2, "product_id": 1, "horizon": 30},
    ]
    results = pipeline.predict_batch(batch_requests)
    print(f"\nBatch predictions: {results}")
