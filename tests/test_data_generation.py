"""
Unit Tests for Data Generation

Tests for synthetic data generators.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_generation.config import DataGenerationConfig
from src.data_generation.stores_generator import StoresGenerator
from src.data_generation.products_generator import ProductsGenerator
from src.data_generation.calendar_generator import CalendarGenerator
from src.data_generation.customers_generator import CustomersGenerator


class TestDataGenerationConfig:
    """Tests for DataGenerationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DataGenerationConfig()
        
        assert config.seed == 42
        assert config.stores.count == 50
        assert config.products.count == 1000
        assert config.sales.max_total_rows == 5000000
    
    def test_date_range(self):
        """Test date range calculation."""
        config = DataGenerationConfig(
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        
        start, end = config.get_date_range()
        assert start.year == 2024
        assert start.month == 1
        assert start.day == 1
        assert config.get_num_days() == 31


class TestStoresGenerator:
    """Tests for StoresGenerator."""
    
    def test_generate_stores(self, tmp_path):
        """Test store generation."""
        config = DataGenerationConfig(output_dir=str(tmp_path))
        config.stores.count = 10
        
        generator = StoresGenerator(config)
        df = generator.generate()
        
        assert len(df) == 10
        assert "store_id" in df.columns
        assert "store_name" in df.columns
        assert "region" in df.columns
        assert df["store_id"].nunique() == 10
    
    def test_regions_distribution(self, tmp_path):
        """Test that regions are distributed across stores."""
        config = DataGenerationConfig(output_dir=str(tmp_path))
        config.stores.count = 50
        
        generator = StoresGenerator(config)
        df = generator.generate()
        
        # All configured regions should be represented
        for region in config.stores.regions:
            assert region in df["region"].values


class TestProductsGenerator:
    """Tests for ProductsGenerator."""
    
    def test_generate_products(self, tmp_path):
        """Test product generation."""
        config = DataGenerationConfig(output_dir=str(tmp_path))
        config.products.count = 100
        
        generator = ProductsGenerator(config)
        df = generator.generate()
        
        assert len(df) == 100
        assert "product_id" in df.columns
        assert "category" in df.columns
        assert "unit_price" in df.columns
        assert df["unit_price"].min() > 0
    
    def test_price_cost_relationship(self, tmp_path):
        """Test that price is always greater than cost."""
        config = DataGenerationConfig(output_dir=str(tmp_path))
        config.products.count = 50
        
        generator = ProductsGenerator(config)
        df = generator.generate()
        
        assert (df["unit_price"] > df["unit_cost"]).all()


class TestCalendarGenerator:
    """Tests for CalendarGenerator."""
    
    def test_generate_calendar(self, tmp_path):
        """Test calendar generation."""
        config = DataGenerationConfig(
            output_dir=str(tmp_path),
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        
        generator = CalendarGenerator(config)
        df = generator.generate()
        
        assert len(df) == 31
        assert "date" in df.columns
        assert "is_weekend" in df.columns
        assert "is_holiday" in df.columns
    
    def test_weekend_detection(self, tmp_path):
        """Test weekend detection."""
        config = DataGenerationConfig(
            output_dir=str(tmp_path),
            start_date="2024-01-01",
            end_date="2024-01-07"
        )
        
        generator = CalendarGenerator(config)
        df = generator.generate()
        
        # Jan 6-7, 2024 are Saturday-Sunday
        weekend_count = df["is_weekend"].sum()
        assert weekend_count == 2


class TestCustomersGenerator:
    """Tests for CustomersGenerator."""
    
    def test_generate_customers(self, tmp_path):
        """Test customer generation."""
        config = DataGenerationConfig(output_dir=str(tmp_path))
        config.customers.count = 100
        
        generator = CustomersGenerator(config)
        df = generator.generate()
        
        assert len(df) == 100
        assert "customer_id" in df.columns
        assert "segment" in df.columns
        assert "lifetime_value" in df.columns
    
    def test_segment_distribution(self, tmp_path):
        """Test that all segments are represented."""
        config = DataGenerationConfig(output_dir=str(tmp_path))
        config.customers.count = 500
        
        generator = CustomersGenerator(config)
        df = generator.generate()
        
        for segment in config.customers.segments:
            assert segment in df["segment"].values
