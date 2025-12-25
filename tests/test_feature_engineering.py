"""
Unit Tests for Feature Engineering

Tests for feature engineering components.
"""

import pytest
import pandas as pd
import numpy as np

from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringConfig


class TestFeatureEngineering:
    """Tests for FeatureEngineering component."""
    
    def test_init(self):
        """Test feature engineering initialization."""
        fe = FeatureEngineering()
        
        assert fe.lag_features is not None
        assert fe.rolling_windows is not None
        assert 7 in fe.lag_features
        assert 30 in fe.rolling_windows
    
    def test_create_lag_features(self, sample_daily_sales):
        """Test lag feature creation."""
        fe = FeatureEngineering()
        
        df = fe._create_lag_features(sample_daily_sales)
        
        assert "sales_lag_7" in df.columns
        assert "sales_lag_14" in df.columns
        assert "sales_lag_30" in df.columns
        assert "qty_lag_7" in df.columns
    
    def test_create_rolling_features(self, sample_daily_sales):
        """Test rolling feature creation."""
        fe = FeatureEngineering()
        
        df = sample_daily_sales.copy()
        df = fe._create_rolling_features(df)
        
        assert "rolling_mean_7" in df.columns
        assert "rolling_std_7" in df.columns
        assert "rolling_mean_30" in df.columns
    
    def test_add_calendar_features(self, sample_daily_sales, sample_calendar):
        """Test calendar feature addition."""
        fe = FeatureEngineering()
        
        df = fe._add_calendar_features(sample_daily_sales, sample_calendar)
        
        assert "day_of_week" in df.columns
        assert "is_weekend" in df.columns
        assert "is_holiday" in df.columns
    
    def test_add_store_features(self, sample_daily_sales, sample_stores):
        """Test store feature addition."""
        fe = FeatureEngineering()
        
        df = fe._add_store_features(sample_daily_sales, sample_stores)
        
        assert "store_region" in df.columns
        assert "store_format" in df.columns
    
    def test_add_product_features(self, sample_daily_sales, sample_products):
        """Test product feature addition."""
        fe = FeatureEngineering()
        
        df = fe._add_product_features(sample_daily_sales, sample_products)
        
        assert "product_category" in df.columns
        assert "product_price" in df.columns
    
    def test_get_feature_columns(self):
        """Test feature column listing."""
        fe = FeatureEngineering()
        
        feature_cols, target_col = fe.get_feature_columns()
        
        assert isinstance(feature_cols, list)
        assert len(feature_cols) > 0
        assert target_col == "daily_quantity"
        assert "sales_lag_7" in feature_cols
        assert "rolling_mean_7" in feature_cols
    
    def test_full_feature_creation(
        self, 
        sample_daily_sales, 
        sample_calendar, 
        sample_stores, 
        sample_products
    ):
        """Test full feature engineering pipeline."""
        fe = FeatureEngineering()
        
        # Create features without weather/promotions
        df = fe.create_features(
            daily_sales=sample_daily_sales,
            calendar=sample_calendar,
            stores=sample_stores,
            products=sample_products,
            weather=None,
            promotions=None,
        )
        
        # Should have features after dropping NaN
        assert len(df) > 0
        assert "sales_lag_7" in df.columns
        assert "rolling_mean_7" in df.columns
        assert "daily_quantity" in df.columns


class TestFeatureEngineeringConfig:
    """Tests for FeatureEngineeringConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = FeatureEngineeringConfig()
        
        assert config.lag_features == [7, 14, 30]
        assert config.rolling_windows == [7, 14, 30]
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = FeatureEngineeringConfig(
            lag_features=[1, 3, 7],
            rolling_windows=[3, 7, 14]
        )
        
        assert config.lag_features == [1, 3, 7]
        assert config.rolling_windows == [3, 7, 14]
