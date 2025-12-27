"""
Data Quality Tests

Tests for data integrity and quality across all datasets.
Ensures pipeline reliability and catches data issues early.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestDataQuality:
    """Tests for data quality and integrity."""
    
    @pytest.fixture
    def data_dir(self):
        """Get the data directory path."""
        return Path(__file__).parent.parent / "artifacts" / "data"
    
    @pytest.fixture
    def load_stores(self, data_dir):
        """Load stores dataset if exists."""
        path = data_dir / "stores.csv"
        if path.exists():
            return pd.read_csv(path)
        pytest.skip("stores.csv not found - run data generation first")
    
    @pytest.fixture
    def load_products(self, data_dir):
        """Load products dataset if exists."""
        path = data_dir / "products.csv"
        if path.exists():
            return pd.read_csv(path)
        pytest.skip("products.csv not found - run data generation first")
    
    @pytest.fixture
    def load_calendar(self, data_dir):
        """Load calendar dataset if exists."""
        path = data_dir / "calendar.csv"
        if path.exists():
            return pd.read_csv(path)
        pytest.skip("calendar.csv not found - run data generation first")
    
    @pytest.fixture
    def load_sales_sample(self, data_dir):
        """Load a sample of sales data if exists."""
        path = data_dir / "sales_transactions.csv"
        if path.exists():
            return pd.read_csv(path, nrows=100000)
        pytest.skip("sales_transactions.csv not found - run data generation first")
    
    # =========================================================================
    # NULL VALUE TESTS
    # =========================================================================
    
    def test_stores_no_null_critical_columns(self, load_stores):
        """Critical store columns should have no null values."""
        critical_cols = ["store_id", "store_name", "region", "store_format"]
        for col in critical_cols:
            if col in load_stores.columns:
                null_count = load_stores[col].isnull().sum()
                assert null_count == 0, f"Column '{col}' has {null_count} null values"
    
    def test_products_no_null_critical_columns(self, load_products):
        """Critical product columns should have no null values."""
        critical_cols = ["product_id", "product_name", "category", "unit_price"]
        for col in critical_cols:
            if col in load_products.columns:
                null_count = load_products[col].isnull().sum()
                assert null_count == 0, f"Column '{col}' has {null_count} null values"
    
    def test_sales_no_null_critical_columns(self, load_sales_sample):
        """Critical sales columns should have no null values."""
        critical_cols = ["transaction_id", "date", "store_id", "product_id", "quantity"]
        for col in critical_cols:
            if col in load_sales_sample.columns:
                null_count = load_sales_sample[col].isnull().sum()
                assert null_count == 0, f"Column '{col}' has {null_count} null values"
    
    # =========================================================================
    # NEGATIVE VALUE TESTS
    # =========================================================================
    
    def test_no_negative_quantities(self, load_sales_sample):
        """Quantities should never be negative."""
        if "quantity" in load_sales_sample.columns:
            negative_count = (load_sales_sample["quantity"] < 0).sum()
            assert negative_count == 0, f"Found {negative_count} negative quantities"
    
    def test_no_negative_prices(self, load_products):
        """Prices should never be negative."""
        if "unit_price" in load_products.columns:
            negative_count = (load_products["unit_price"] < 0).sum()
            assert negative_count == 0, f"Found {negative_count} negative prices"
    
    def test_no_negative_total_amounts(self, load_sales_sample):
        """Total amounts should never be negative."""
        if "total_amount" in load_sales_sample.columns:
            negative_count = (load_sales_sample["total_amount"] < 0).sum()
            assert negative_count == 0, f"Found {negative_count} negative amounts"
    
    def test_no_negative_store_sizes(self, load_stores):
        """Store sizes should never be negative."""
        if "size_sqft" in load_stores.columns:
            negative_count = (load_stores["size_sqft"] < 0).sum()
            assert negative_count == 0, f"Found {negative_count} negative store sizes"
    
    # =========================================================================
    # PRIMARY KEY UNIQUENESS TESTS
    # =========================================================================
    
    def test_store_id_unique(self, load_stores):
        """Store IDs should be unique."""
        duplicate_count = load_stores["store_id"].duplicated().sum()
        assert duplicate_count == 0, f"Found {duplicate_count} duplicate store IDs"
    
    def test_product_id_unique(self, load_products):
        """Product IDs should be unique."""
        duplicate_count = load_products["product_id"].duplicated().sum()
        assert duplicate_count == 0, f"Found {duplicate_count} duplicate product IDs"
    
    def test_transaction_id_unique(self, load_sales_sample):
        """Transaction IDs should be unique."""
        duplicate_count = load_sales_sample["transaction_id"].duplicated().sum()
        assert duplicate_count == 0, f"Found {duplicate_count} duplicate transaction IDs"
    
    # =========================================================================
    # REFERENTIAL INTEGRITY TESTS
    # =========================================================================
    
    def test_sales_store_ids_exist(self, load_sales_sample, load_stores):
        """All store IDs in sales should exist in stores table."""
        sales_stores = set(load_sales_sample["store_id"].unique())
        valid_stores = set(load_stores["store_id"].unique())
        invalid_stores = sales_stores - valid_stores
        assert len(invalid_stores) == 0, f"Invalid store IDs in sales: {invalid_stores}"
    
    def test_sales_product_ids_exist(self, load_sales_sample, load_products):
        """All product IDs in sales should exist in products table."""
        sales_products = set(load_sales_sample["product_id"].unique())
        valid_products = set(load_products["product_id"].unique())
        invalid_products = sales_products - valid_products
        assert len(invalid_products) == 0, f"Invalid product IDs in sales: {invalid_products}"
    
    # =========================================================================
    # DATE VALIDITY TESTS
    # =========================================================================
    
    def test_calendar_date_range_reasonable(self, load_calendar):
        """Calendar dates should be within reasonable range."""
        load_calendar["date"] = pd.to_datetime(load_calendar["date"])
        min_date = load_calendar["date"].min()
        max_date = load_calendar["date"].max()
        
        # Expect data between 2010 and 2030
        assert min_date.year >= 2010, f"Date too old: {min_date}"
        assert max_date.year <= 2030, f"Date too far in future: {max_date}"
    
    def test_sales_dates_within_calendar(self, load_sales_sample, load_calendar):
        """All sales dates should exist in calendar."""
        load_sales_sample["date"] = pd.to_datetime(load_sales_sample["date"])
        load_calendar["date"] = pd.to_datetime(load_calendar["date"])
        
        sales_dates = set(load_sales_sample["date"].dt.date.unique())
        calendar_dates = set(load_calendar["date"].dt.date.unique())
        
        invalid_dates = sales_dates - calendar_dates
        assert len(invalid_dates) == 0, f"Sales contain dates not in calendar: {list(invalid_dates)[:5]}"
    
    # =========================================================================
    # STATISTICAL DISTRIBUTION TESTS
    # =========================================================================
    
    def test_quantity_distribution_reasonable(self, load_sales_sample):
        """Quantity distribution should be within reasonable bounds."""
        if "quantity" in load_sales_sample.columns:
            q99 = load_sales_sample["quantity"].quantile(0.99)
            # 99th percentile should be less than 1000 units (reasonable for retail)
            assert q99 < 1000, f"99th percentile quantity ({q99}) seems unreasonably high"
    
    def test_price_distribution_reasonable(self, load_products):
        """Price distribution should be within reasonable bounds."""
        if "unit_price" in load_products.columns:
            max_price = load_products["unit_price"].max()
            min_price = load_products["unit_price"].min()
            
            assert min_price > 0, f"Minimum price ({min_price}) should be positive"
            assert max_price < 10000, f"Maximum price ({max_price}) seems unreasonably high"
    
    def test_store_size_distribution_reasonable(self, load_stores):
        """Store sizes should be within reasonable bounds."""
        if "size_sqft" in load_stores.columns:
            min_size = load_stores["size_sqft"].min()
            max_size = load_stores["size_sqft"].max()
            
            assert min_size >= 500, f"Store size ({min_size}) too small"
            assert max_size <= 500000, f"Store size ({max_size}) unreasonably large"


class TestDataCompleteness:
    """Tests for data completeness and coverage."""
    
    @pytest.fixture
    def data_dir(self):
        """Get the data directory path."""
        return Path(__file__).parent.parent / "artifacts" / "data"
    
    def test_required_files_exist(self, data_dir):
        """All required data files should exist."""
        required_files = [
            "stores.csv",
            "products.csv",
            "calendar.csv",
            "sales_transactions.csv",
        ]
        
        missing_files = []
        for file in required_files:
            if not (data_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            pytest.skip(f"Missing data files (run data generation first): {missing_files}")
    
    def test_stores_have_sufficient_count(self, data_dir):
        """Should have a reasonable number of stores."""
        path = data_dir / "stores.csv"
        if not path.exists():
            pytest.skip("stores.csv not found")
        
        df = pd.read_csv(path)
        assert len(df) >= 5, f"Only {len(df)} stores - expected at least 5"
    
    def test_products_have_sufficient_count(self, data_dir):
        """Should have a reasonable number of products."""
        path = data_dir / "products.csv"
        if not path.exists():
            pytest.skip("products.csv not found")
        
        df = pd.read_csv(path)
        assert len(df) >= 10, f"Only {len(df)} products - expected at least 10"
    
    def test_calendar_has_sufficient_coverage(self, data_dir):
        """Calendar should cover at least 1 year."""
        path = data_dir / "calendar.csv"
        if not path.exists():
            pytest.skip("calendar.csv not found")
        
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        date_range = (df["date"].max() - df["date"].min()).days
        
        assert date_range >= 365, f"Calendar only covers {date_range} days - need at least 1 year"
