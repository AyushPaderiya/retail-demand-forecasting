"""
Model Trainer Component

Trains and evaluates multiple ML models for demand forecasting.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.exception import ModelTrainingException
from src.logger import get_logger
from src.utils import calculate_metrics, load_config, save_object

logger = get_logger(__name__)


@dataclass
class ModelTrainerConfig:
    """Configuration for model training."""
    models_dir: str = "artifacts/models"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 3  # Matches config.yaml


class ModelTrainer:
    """
    Trains multiple ML models and selects the best one.
    
    Models include:
    - Linear Regression, Ridge, Lasso
    - Random Forest
    - XGBoost, LightGBM, CatBoost (when available)
    """
    
    def __init__(self, config: ModelTrainerConfig = None):
        """
        Initialize model trainer.
        
        Args:
            config: Model training configuration.
        """
        self.config = config or ModelTrainerConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Load app config
        app_config = load_config()
        model_config = app_config.get("modeling", {})
        
        self.test_size = model_config.get("test_size", self.config.test_size)
        self.random_state = model_config.get("random_state", self.config.random_state)
        self.cv_folds = model_config.get("cv_folds", self.config.cv_folds)
        
        # Ensure models directory exists
        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize encoders and scalers
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        
        # Store training results
        self.training_results: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Any = None
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training using TIME-BASED split.
        
        Unlike random splitting, this method preserves temporal order to prevent
        data leakage in time-series forecasting. The test set contains only
        data from the most recent time period.
        
        Args:
            df: Feature DataFrame (must contain 'date' column).
            feature_cols: List of feature column names.
            target_col: Target column name.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        try:
            # Get available features
            available_features = [c for c in feature_cols if c in df.columns]
            missing_features = set(feature_cols) - set(available_features)
            
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
            
            # CRITICAL: Sort by date for proper time-series split
            if "date" in df.columns:
                df = df.sort_values("date").reset_index(drop=True)
                self.logger.info("Time-based split: Data sorted by date (no shuffle)")
            else:
                self.logger.warning(
                    "No 'date' column found - using index order. "
                    "Ensure data is pre-sorted chronologically!"
                )
            
            X = df[available_features].copy()
            y = df[target_col].values
            
            # Encode categorical columns
            categorical_cols = X.select_dtypes(include=["object", "bool"]).columns
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
            
            # Handle NaN values
            X = X.fillna(0)
            
            # TIME-BASED SPLIT: Use chronological order, no shuffling
            # Last (test_size * 100)% of data becomes test set
            n_samples = len(X)
            split_idx = int(n_samples * (1 - self.test_size))
            
            X_train = X.values[:split_idx]
            X_test = X.values[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            
            # Log date ranges if available
            if "date" in df.columns:
                train_dates = df["date"].iloc[:split_idx]
                test_dates = df["date"].iloc[split_idx:]
                self.logger.info(
                    f"Train period: {train_dates.min()} to {train_dates.max()}"
                )
                self.logger.info(
                    f"Test period:  {test_dates.min()} to {test_dates.max()}"
                )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            self.logger.info(
                f"Data prepared: {X_train.shape[0]:,} train, "
                f"{X_test.shape[0]:,} test samples, "
                f"{X_train.shape[1]} features"
            )
            
            # Store feature names
            self.feature_names = available_features
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise ModelTrainingException(f"Data preparation failed: {e}", sys)
    
    def _gpu_available(self) -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            # Try checking via subprocess
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi"], 
                    capture_output=True, 
                    text=True
                )
                return result.returncode == 0
            except Exception:
                return False
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of models to train.
        
        Returns:
            Dictionary of model name to model instance.
        """
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0, random_state=self.random_state),
            "Lasso": Lasso(alpha=0.1, random_state=self.random_state),
        }
        
        # Load model configs from config file
        app_config = load_config()
        model_configs = app_config.get("modeling", {}).get("models", [])
        
        # Add RandomForest with config params
        rf_config = next((m for m in model_configs if m["name"] == "RandomForest"), {})
        if rf_config.get("enabled", True):
            rf_params = rf_config.get("params", {})
            models["RandomForest"] = RandomForestRegressor(
                n_estimators=rf_params.get("n_estimators", 50),
                max_depth=rf_params.get("max_depth", 8),
                min_samples_leaf=rf_params.get("min_samples_leaf", 5),
                random_state=rf_params.get("random_state", self.random_state),
                n_jobs=-1
            )
        
        # Try to add XGBoost (GPU enabled if available)
        try:
            import xgboost as xgb
            xgb_config = next((m for m in model_configs if m["name"] == "XGBoost"), {})
            if xgb_config.get("enabled", True):
                xgb_params = xgb_config.get("params", {})
                models["XGBoost"] = xgb.XGBRegressor(
                    n_estimators=xgb_params.get("n_estimators", 100),
                    max_depth=xgb_params.get("max_depth", 6),
                    learning_rate=xgb_params.get("learning_rate", 0.1),
                    random_state=xgb_params.get("random_state", self.random_state),
                    n_jobs=-1,
                    tree_method="hist",
                    device="cuda" if self._gpu_available() else "cpu",
                )
        except ImportError:
            self.logger.warning("XGBoost not available")
        
        # Try to add LightGBM (GPU enabled if available)
        try:
            import lightgbm as lgb
            lgb_config = next((m for m in model_configs if m["name"] == "LightGBM"), {})
            if lgb_config.get("enabled", True):
                lgb_cfg_params = lgb_config.get("params", {})
                lgb_params = {
                    "n_estimators": lgb_cfg_params.get("n_estimators", 100),
                    "max_depth": lgb_cfg_params.get("max_depth", 6),
                    "learning_rate": lgb_cfg_params.get("learning_rate", 0.1),
                    "random_state": lgb_cfg_params.get("random_state", self.random_state),
                    "n_jobs": -1,
                    "verbose": -1,
                }
                models["LightGBM"] = lgb.LGBMRegressor(**lgb_params)
        except ImportError:
            self.logger.warning("LightGBM not available")
        
        # Try to add CatBoost (GPU enabled if available)
        try:
            from catboost import CatBoostRegressor
            cat_config = next((m for m in model_configs if m["name"] == "CatBoost"), {})
            if cat_config.get("enabled", True):
                cat_params = cat_config.get("params", {})
                models["CatBoost"] = CatBoostRegressor(
                    iterations=cat_params.get("iterations", 100),
                    depth=cat_params.get("depth", 6),
                    learning_rate=cat_params.get("learning_rate", 0.1),
                    random_state=cat_params.get("random_state", self.random_state),
                    verbose=cat_params.get("verbose", False),
                    task_type="GPU" if self._gpu_available() else "CPU",
                )
        except ImportError:
            self.logger.warning("CatBoost not available")
        
        return models
    
    def train_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict]:
        """
        Train all models and evaluate performance.
        
        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training target.
            y_test: Test target.
            
        Returns:
            Dictionary of model results.
        """
        try:
            models = self.get_models()
            results = {}
            
            self.logger.info(f"Training {len(models)} models...")
            
            for name, model in models.items():
                self.logger.info(f"Training {name}...")
                
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    train_metrics = calculate_metrics(y_train, y_train_pred)
                    test_metrics = calculate_metrics(y_test, y_test_pred)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=self.cv_folds,
                        scoring="neg_root_mean_squared_error"
                    )
                    cv_rmse = -cv_scores.mean()
                    
                    results[name] = {
                        "model": model,
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "cv_rmse": round(cv_rmse, 4),
                        "cv_std": round(cv_scores.std(), 4),
                    }
                    
                    self.logger.info(
                        f"  {name} - Test RMSE: {test_metrics['rmse']:.4f}, "
                        f"R2: {test_metrics['r2']:.4f}, "
                        f"CV RMSE: {cv_rmse:.4f}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to train {name}: {e}")
                    continue
            
            self.training_results = results
            return results
            
        except Exception as e:
            raise ModelTrainingException(f"Model training failed: {e}", sys)
    
    def select_best_model(
        self,
        metric: str = "rmse"
    ) -> Tuple[str, Any]:
        """
        Select the best model based on test metrics.
        
        Args:
            metric: Metric to use for selection (rmse, mae, r2).
            
        Returns:
            Tuple of (best_model_name, best_model).
        """
        if not self.training_results:
            raise ModelTrainingException("No training results available")
        
        best_name = None
        best_score = float("inf") if metric in ["rmse", "mae"] else float("-inf")
        
        for name, result in self.training_results.items():
            score = result["test_metrics"].get(metric, float("inf"))
            
            if metric in ["rmse", "mae"]:
                if score < best_score:
                    best_score = score
                    best_name = name
            else:  # Higher is better (r2)
                if score > best_score:
                    best_score = score
                    best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.training_results[best_name]["model"]
        
        self.logger.info(
            f"Best model: {best_name} with {metric.upper()}: {best_score:.4f}"
        )
        
        return best_name, self.best_model
    
    def save_models(self) -> Dict[str, str]:
        """
        Save all trained models.
        
        Returns:
            Dictionary of model name to file path.
        """
        saved_paths = {}
        
        for name, result in self.training_results.items():
            model_path = Path(self.config.models_dir) / f"{name.lower()}_model.joblib"
            save_object(str(model_path), result["model"])
            saved_paths[name] = str(model_path)
        
        # Save scaler and encoders
        if self.scaler:
            scaler_path = Path(self.config.models_dir) / "scaler.joblib"
            save_object(str(scaler_path), self.scaler)
            saved_paths["scaler"] = str(scaler_path)
        
        if self.label_encoders:
            encoders_path = Path(self.config.models_dir) / "label_encoders.joblib"
            save_object(str(encoders_path), self.label_encoders)
            saved_paths["label_encoders"] = str(encoders_path)
        
        # Save feature names
        if hasattr(self, "feature_names"):
            import json
            features_path = Path(self.config.models_dir) / "feature_names.json"
            with open(features_path, "w") as f:
                json.dump(self.feature_names, f)
            saved_paths["feature_names"] = str(features_path)
        
        self.logger.info(f"Saved {len(saved_paths)} artifacts to {self.config.models_dir}")
        
        return saved_paths
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary of all model results.
        
        Returns:
            DataFrame with model comparison.
        """
        rows = []
        for name, result in self.training_results.items():
            row = {
                "Model": name,
                "Train_RMSE": result["train_metrics"]["rmse"],
                "Test_RMSE": result["test_metrics"]["rmse"],
                "Train_MAE": result["train_metrics"]["mae"],
                "Test_MAE": result["test_metrics"]["mae"],
                "Train_R2": result["train_metrics"]["r2"],
                "Test_R2": result["test_metrics"]["r2"],
                "CV_RMSE": result["cv_rmse"],
                "CV_Std": result["cv_std"],
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values("Test_RMSE")
        
        return df


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.feature_engineering import FeatureEngineering
    
    # Test model training
    ingestion = DataIngestion()
    data = ingestion.load_all_data()
    
    # Load sample of sales
    sales = ingestion.load_sales(nrows=100000)
    daily_sales = ingestion.create_aggregated_sales(sales)
    
    # Create features
    fe = FeatureEngineering()
    features_df = fe.create_features(
        daily_sales,
        data["calendar"],
        data["stores"],
        data["products"],
        data["weather"],
        data["promotions"]
    )
    
    # Get feature columns
    feature_cols, target_col = fe.get_feature_columns()
    
    # Train models
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        features_df, feature_cols, target_col
    )
    
    results = trainer.train_models(X_train, X_test, y_train, y_test)
    best_name, best_model = trainer.select_best_model()
    
    print("\nModel Comparison:")
    print(trainer.get_results_summary().to_string())
    
    # Save models
    trainer.save_models()
