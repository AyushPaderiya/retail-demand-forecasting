"""
Feature Engineering Component

Creates ML features from raw data for demand forecasting.
"""

import sys
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from src.exception import FeatureEngineeringException
from src.logger import get_logger
from src.utils import load_config

logger = get_logger(__name__)


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""
    lag_features: List[int] = None
    rolling_windows: List[int] = None
    
    def __post_init__(self):
        if self.lag_features is None:
            self.lag_features = [7, 14, 30]
        if self.rolling_windows is None:
            self.rolling_windows = [7, 14, 30]


class FeatureEngineering:
    """
    Creates features for demand forecasting models.
    
    Features include:
    - Lag features (past sales)
    - Rolling statistics (mean, std)
    - Date/time features
    - Store and product aggregates
    - Weather features
    - Promotion indicators
    """
    
    def __init__(self, config: FeatureEngineeringConfig = None):
        """
        Initialize feature engineering.
        
        Args:
            config: Feature engineering configuration.
        """
        self.config = config or FeatureEngineeringConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Load app config
        app_config = load_config()
        fe_config = app_config.get("feature_engineering", {})
        
        self.lag_features = fe_config.get("lag_features", self.config.lag_features)
        self.rolling_windows = fe_config.get("rolling_windows", self.config.rolling_windows)
    
    def create_features(
        self,
        daily_sales: pd.DataFrame,
        calendar: pd.DataFrame,
        stores: pd.DataFrame,
        products: pd.DataFrame,
        weather: pd.DataFrame = None,
        promotions: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Create all features for ML training.
        
        Args:
            daily_sales: Daily aggregated sales by store-product.
            calendar: Calendar dimension table.
            stores: Stores dimension table.
            products: Products dimension table.
            weather: Weather data (optional).
            promotions: Promotions data (optional).
            
        Returns:
            DataFrame with all engineered features.
        """
        try:
            self.logger.info("Starting feature engineering...")
            
            # Start with daily sales
            df = daily_sales.copy()
            df["date"] = pd.to_datetime(df["date"])
            
            # Sort for time-series operations
            df = df.sort_values(["store_id", "product_id", "date"])
            
            # Create lag features
            self.logger.info("Creating lag features...")
            df = self._create_lag_features(df)
            
            # Create rolling features
            self.logger.info("Creating rolling features...")
            df = self._create_rolling_features(df)
            
            # Merge calendar features
            self.logger.info("Adding calendar features...")
            df = self._add_calendar_features(df, calendar)
            
            # Merge store features
            self.logger.info("Adding store features...")
            df = self._add_store_features(df, stores)
            
            # Merge product features
            self.logger.info("Adding product features...")
            df = self._add_product_features(df, products)
            
            # Add weather features
            if weather is not None:
                self.logger.info("Adding weather features...")
                df = self._add_weather_features(df, weather)
            
            # Add promotion features
            if promotions is not None:
                self.logger.info("Adding promotion features...")
                df = self._add_promotion_features(df, promotions)
            
            # Create entity aggregates
            self.logger.info("Creating entity aggregates...")
            df = self._create_entity_aggregates(df)
            
            # Drop rows with NaN from lag/rolling (initial period)
            initial_rows = len(df)
            df = df.dropna(subset=[f"sales_lag_{self.lag_features[0]}"])
            self.logger.info(
                f"Dropped {initial_rows - len(df):,} rows with NaN lag features"
            )
            
            self.logger.info(f"Feature engineering complete: {len(df):,} rows, {len(df.columns)} features")
            
            return df
            
        except Exception as e:
            raise FeatureEngineeringException(f"Feature engineering failed: {e}", sys)
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for sales."""
        group_cols = ["store_id", "product_id"]
        
        for lag in self.lag_features:
            df[f"sales_lag_{lag}"] = df.groupby(group_cols)["daily_revenue"].shift(lag)
            df[f"qty_lag_{lag}"] = df.groupby(group_cols)["daily_quantity"].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features."""
        group_cols = ["store_id", "product_id"]
        
        for window in self.rolling_windows:
            # Rolling mean
            df[f"rolling_mean_{window}"] = df.groupby(group_cols)["daily_revenue"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            
            # Rolling std
            df[f"rolling_std_{window}"] = df.groupby(group_cols)["daily_revenue"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
            
            # Rolling min/max for 7 days
            if window == 7:
                df[f"rolling_min_{window}"] = df.groupby(group_cols)["daily_revenue"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).min()
                )
                df[f"rolling_max_{window}"] = df.groupby(group_cols)["daily_revenue"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).max()
                )
        
        return df
    
    def _add_calendar_features(
        self,
        df: pd.DataFrame,
        calendar: pd.DataFrame
    ) -> pd.DataFrame:
        """Add calendar/date features."""
        calendar = calendar.copy()
        calendar["date"] = pd.to_datetime(calendar["date"])
        
        # Select relevant columns
        calendar_cols = [
            "date", "day_of_week", "week_of_year", "month", "quarter",
            "is_weekend", "is_holiday", "days_to_holiday", "season"
        ]
        calendar_subset = calendar[[c for c in calendar_cols if c in calendar.columns]]
        
        df = df.merge(calendar_subset, on="date", how="left")
        return df
    
    def _add_store_features(
        self,
        df: pd.DataFrame,
        stores: pd.DataFrame
    ) -> pd.DataFrame:
        """Add store features."""
        store_cols = ["store_id", "region", "store_format", "size_sqft"]
        stores_subset = stores[[c for c in store_cols if c in stores.columns]]
        
        df = df.merge(stores_subset, on="store_id", how="left")
        
        # Rename for clarity
        df = df.rename(columns={
            "region": "store_region",
            "size_sqft": "store_size_sqft"
        })
        
        return df
    
    def _add_product_features(
        self,
        df: pd.DataFrame,
        products: pd.DataFrame
    ) -> pd.DataFrame:
        """Add product features."""
        product_cols = [
            "product_id", "category", "subcategory",
            "unit_price", "is_perishable"
        ]
        products_subset = products[[c for c in product_cols if c in products.columns]]
        
        df = df.merge(products_subset, on="product_id", how="left")
        
        # Rename for clarity
        df = df.rename(columns={
            "category": "product_category",
            "subcategory": "product_subcategory",
            "unit_price": "product_price"
        })
        
        return df
    
    def _add_weather_features(
        self,
        df: pd.DataFrame,
        weather: pd.DataFrame
    ) -> pd.DataFrame:
        """Add weather features."""
        weather = weather.copy()
        weather["date"] = pd.to_datetime(weather["date"])
        
        weather_cols = [
            "date", "store_id", "temp_high_f", "temp_low_f",
            "precipitation_inches", "conditions"
        ]
        weather_subset = weather[[c for c in weather_cols if c in weather.columns]]
        
        df = df.merge(weather_subset, on=["date", "store_id"], how="left")
        
        # Rename for clarity
        df = df.rename(columns={"conditions": "weather_conditions"})
        
        return df
    
    def _add_promotion_features(
        self,
        df: pd.DataFrame,
        promotions: pd.DataFrame
    ) -> pd.DataFrame:
        """Add promotion indicator features."""
        promotions = promotions.copy()
        promotions["start_date"] = pd.to_datetime(promotions["start_date"])
        promotions["end_date"] = pd.to_datetime(promotions["end_date"])
        
        # Create a date range for each promotion
        promo_records = []
        for _, row in promotions.iterrows():
            dates = pd.date_range(row["start_date"], row["end_date"])
            for date in dates:
                promo_records.append({
                    "date": date,
                    "store_id": row["store_id"],
                    "product_id": row["product_id"],
                    "has_promotion": True,
                    "promotion_discount": row["discount_percent"]
                })
        
        if promo_records:
            promo_df = pd.DataFrame(promo_records)
            # Keep max discount if multiple promotions
            promo_df = promo_df.groupby(
                ["date", "store_id", "product_id"]
            ).agg({
                "has_promotion": "max",
                "promotion_discount": "max"
            }).reset_index()
            
            df = df.merge(promo_df, on=["date", "store_id", "product_id"], how="left")
        
        # Fill NaN with no promotion (handle case where columns may not exist)
        if "has_promotion" not in df.columns:
            df["has_promotion"] = False
        else:
            df["has_promotion"] = df["has_promotion"].fillna(False)
        
        if "promotion_discount" not in df.columns:
            df["promotion_discount"] = 0
        else:
            df["promotion_discount"] = df["promotion_discount"].fillna(0)
        
        return df
    
    def _create_entity_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store and product level aggregates."""
        # Store average daily sales (30-day rolling)
        df["store_avg_daily_sales_30d"] = df.groupby("store_id")["daily_revenue"].transform(
            lambda x: x.shift(1).rolling(30, min_periods=1).mean()
        )
        
        # Product average daily sales (30-day rolling)
        df["product_avg_daily_sales_30d"] = df.groupby("product_id")["daily_revenue"].transform(
            lambda x: x.shift(1).rolling(30, min_periods=1).mean()
        )
        
        return df
    
    def get_feature_columns(self) -> Tuple[List[str], str]:
        """
        Get feature column names and target column.
        
        Returns:
            Tuple of (feature_columns, target_column).
        """
        feature_cols = [
            # Lag features
            *[f"sales_lag_{lag}" for lag in self.lag_features],
            *[f"qty_lag_{lag}" for lag in self.lag_features],
            
            # Rolling features
            *[f"rolling_mean_{w}" for w in self.rolling_windows],
            *[f"rolling_std_{w}" for w in self.rolling_windows],
            "rolling_min_7", "rolling_max_7",
            
            # Calendar features
            "day_of_week", "week_of_year", "month", "quarter",
            "is_weekend", "is_holiday", "days_to_holiday",
            
            # Store features
            "store_size_sqft",
            
            # Product features
            "product_price", "is_perishable",
            
            # Weather features
            "temp_high_f", "temp_low_f", "precipitation_inches",
            
            # Promotion features
            "has_promotion", "promotion_discount",
            
            # Entity aggregates
            "store_avg_daily_sales_30d", "product_avg_daily_sales_30d",
        ]
        
        target_col = "daily_quantity"
        
        return feature_cols, target_col


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    
    # Test feature engineering
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
    
    print(f"Features shape: {features_df.shape}")
    print(f"Columns: {features_df.columns.tolist()}")
