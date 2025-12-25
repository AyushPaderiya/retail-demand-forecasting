"""
Training Pipeline

End-to-end pipeline for training demand forecasting models.
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_config

logger = get_logger(__name__)


@dataclass
class TrainingPipelineConfig:
    """Configuration for training pipeline."""
    sample_size: Optional[int] = None  # None for full data
    save_features: bool = True
    features_path: str = "artifacts/data/features.csv"


class TrainingPipeline:
    """
    End-to-end training pipeline.
    
    Steps:
    1. Data ingestion
    2. Feature engineering
    3. Model training
    4. Model selection
    5. Save artifacts
    """
    
    def __init__(self, config: TrainingPipelineConfig = None):
        """
        Initialize training pipeline.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config or TrainingPipelineConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize components
        self.data_ingestion = DataIngestion()
        self.feature_engineering = FeatureEngineering()
        self.model_trainer = ModelTrainer()
    
    def run(self) -> Dict:
        """
        Execute the full training pipeline.
        
        Returns:
            Dictionary with pipeline results.
        """
        start_time = datetime.now()
        results = {}
        
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING TRAINING PIPELINE")
            self.logger.info("=" * 60)
            
            # Step 1: Data Ingestion
            self.logger.info("\n--- STEP 1: Data Ingestion ---")
            data = self.data_ingestion.load_all_data()
            
            # Load sales with optional sampling
            if self.config.sample_size:
                self.logger.info(f"Loading {self.config.sample_size:,} sales samples")
                sales_df = self.data_ingestion.load_sales(nrows=self.config.sample_size)
            else:
                self.logger.info("Loading full sales data (this may take a while)...")
                sales_df = self.data_ingestion.load_sales()
            
            results["total_sales"] = len(sales_df)
            
            # Aggregate to daily
            daily_sales = self.data_ingestion.create_aggregated_sales(sales_df)
            results["daily_aggregations"] = len(daily_sales)
            
            # Free memory
            del sales_df
            
            # Step 2: Feature Engineering
            self.logger.info("\n--- STEP 2: Feature Engineering ---")
            features_df = self.feature_engineering.create_features(
                daily_sales=daily_sales,
                calendar=data["calendar"],
                stores=data["stores"],
                products=data["products"],
                weather=data.get("weather"),
                promotions=data.get("promotions"),
            )
            
            results["feature_rows"] = len(features_df)
            results["feature_columns"] = len(features_df.columns)
            
            # Save features if configured
            if self.config.save_features:
                features_path = Path(self.config.features_path)
                features_path.parent.mkdir(parents=True, exist_ok=True)
                features_df.to_csv(features_path, index=False)
                self.logger.info(f"Features saved to {features_path}")
            
            # Step 3: Model Training
            self.logger.info("\n--- STEP 3: Model Training ---")
            feature_cols, target_col = self.feature_engineering.get_feature_columns()
            
            X_train, X_test, y_train, y_test = self.model_trainer.prepare_data(
                features_df, feature_cols, target_col
            )
            
            training_results = self.model_trainer.train_models(
                X_train, X_test, y_train, y_test
            )
            
            results["models_trained"] = len(training_results)
            
            # Step 4: Model Selection
            self.logger.info("\n--- STEP 4: Model Selection ---")
            best_name, best_model = self.model_trainer.select_best_model()
            
            results["best_model"] = best_name
            results["best_model_rmse"] = training_results[best_name]["test_metrics"]["rmse"]
            results["best_model_r2"] = training_results[best_name]["test_metrics"]["r2"]
            
            # Step 5: Save Artifacts
            self.logger.info("\n--- STEP 5: Saving Artifacts ---")
            saved_paths = self.model_trainer.save_models()
            
            results["saved_artifacts"] = len(saved_paths)
            
            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            results["duration_seconds"] = round(duration, 2)
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("TRAINING PIPELINE COMPLETE")
            self.logger.info("=" * 60)
            self.logger.info(f"Duration: {duration:.2f}s")
            self.logger.info(f"Best Model: {best_name}")
            self.logger.info(f"Test RMSE: {results['best_model_rmse']:.4f}")
            self.logger.info(f"Test R2: {results['best_model_r2']:.4f}")
            
            # Print model comparison
            print("\nModel Comparison:")
            print(self.model_trainer.get_results_summary().to_string())
            
            return results
            
        except Exception as e:
            raise CustomException(f"Training pipeline failed: {e}", sys)


def main():
    """Main entry point for training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of sales rows to sample (default: all)"
    )
    parser.add_argument(
        "--no-save-features",
        action="store_true",
        help="Don't save engineered features"
    )
    
    args = parser.parse_args()
    
    config = TrainingPipelineConfig(
        sample_size=args.sample_size,
        save_features=not args.no_save_features,
    )
    
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
    
    print("\nPipeline Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
