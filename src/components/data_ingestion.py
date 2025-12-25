"""
Data Ingestion Component

Handles loading data from CSV files or database for ML training.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.exception import DataIngestionException
from src.logger import get_logger
from src.utils import load_config

logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion."""
    data_dir: str = "artifacts/data"
    train_data_path: str = "artifacts/data/train.csv"
    test_data_path: str = "artifacts/data/test.csv"


class DataIngestion:
    """
    Handles data ingestion from various sources.
    
    Loads raw data, performs initial cleaning, and splits into train/test.
    """
    
    def __init__(self, config: DataIngestionConfig = None):
        """
        Initialize data ingestion.
        
        Args:
            config: Data ingestion configuration.
        """
        self.config = config or DataIngestionConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Load app config
        self.app_config = load_config()
    
    def load_stores(self) -> pd.DataFrame:
        """Load stores data."""
        path = Path(self.config.data_dir) / "stores.csv"
        return self._load_csv(path, "stores")
    
    def load_products(self) -> pd.DataFrame:
        """Load products data."""
        path = Path(self.config.data_dir) / "products.csv"
        return self._load_csv(path, "products")
    
    def load_calendar(self) -> pd.DataFrame:
        """Load calendar data."""
        path = Path(self.config.data_dir) / "calendar.csv"
        df = self._load_csv(path, "calendar")
        df["date"] = pd.to_datetime(df["date"])
        return df
    
    def load_promotions(self) -> pd.DataFrame:
        """Load promotions data."""
        path = Path(self.config.data_dir) / "promotions.csv"
        df = self._load_csv(path, "promotions")
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        return df
    
    def load_weather(self) -> pd.DataFrame:
        """Load weather data."""
        path = Path(self.config.data_dir) / "weather.csv"
        df = self._load_csv(path, "weather")
        df["date"] = pd.to_datetime(df["date"])
        return df
    
    def load_customers(self) -> pd.DataFrame:
        """Load customers data."""
        path = Path(self.config.data_dir) / "customers.csv"
        df = self._load_csv(path, "customers")
        df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"])
        return df
    
    def load_sales(
        self, 
        chunksize: int = None,
        nrows: int = None
    ) -> pd.DataFrame:
        """
        Load sales transactions data.
        
        Args:
            chunksize: If provided, read in chunks (returns iterator).
            nrows: Maximum number of rows to read.
            
        Returns:
            Sales DataFrame or iterator of DataFrames.
        """
        path = Path(self.config.data_dir) / "sales_transactions.csv"
        
        if chunksize:
            return pd.read_csv(path, chunksize=chunksize)
        
        df = self._load_csv(path, "sales", nrows=nrows)
        df["date"] = pd.to_datetime(df["date"])
        return df
    
    def load_all_data(self) -> dict:
        """
        Load all datasets.
        
        Returns:
            Dictionary containing all DataFrames.
        """
        self.logger.info("Loading all datasets...")
        
        data = {
            "stores": self.load_stores(),
            "products": self.load_products(),
            "calendar": self.load_calendar(),
            "promotions": self.load_promotions(),
            "weather": self.load_weather(),
            "customers": self.load_customers(),
        }
        
        self.logger.info("All dimension tables loaded")
        return data
    
    def create_aggregated_sales(
        self,
        sales_df: pd.DataFrame = None,
        nrows: int = None
    ) -> pd.DataFrame:
        """
        Create daily aggregated sales by store-product.
        
        Args:
            sales_df: Pre-loaded sales DataFrame (optional).
            nrows: Max rows to load if sales_df not provided.
            
        Returns:
            Aggregated daily sales DataFrame.
        """
        try:
            if sales_df is None:
                sales_df = self.load_sales(nrows=nrows)
            
            self.logger.info(f"Aggregating {len(sales_df):,} sales transactions...")
            
            # Aggregate to daily level
            daily_sales = sales_df.groupby(
                ["date", "store_id", "product_id"]
            ).agg({
                "quantity": "sum",
                "total_amount": "sum",
                "transaction_id": "count",
                "discount_applied": "sum",
            }).reset_index()
            
            daily_sales.columns = [
                "date", "store_id", "product_id",
                "daily_quantity", "daily_revenue",
                "transaction_count", "discounted_transactions"
            ]
            
            self.logger.info(
                f"Created {len(daily_sales):,} daily store-product aggregations"
            )
            
            return daily_sales
            
        except Exception as e:
            raise DataIngestionException(f"Failed to aggregate sales: {e}", sys)
    
    def _load_csv(
        self, 
        path: Path, 
        name: str,
        nrows: int = None
    ) -> pd.DataFrame:
        """
        Load a CSV file with error handling.
        
        Args:
            path: Path to CSV file.
            name: Dataset name for logging.
            nrows: Max rows to load.
            
        Returns:
            Loaded DataFrame.
        """
        try:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            df = pd.read_csv(path, nrows=nrows)
            self.logger.info(f"Loaded {name}: {len(df):,} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            raise DataIngestionException(f"Failed to load {name}: {e}", sys)


if __name__ == "__main__":
    # Test data ingestion
    ingestion = DataIngestion()
    data = ingestion.load_all_data()
    
    for name, df in data.items():
        print(f"{name}: {df.shape}")
