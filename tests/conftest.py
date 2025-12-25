"""
Pytest Configuration

Shared fixtures for testing.
"""

import os
import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_stores():
    """Sample stores DataFrame for testing."""
    return pd.DataFrame({
        "store_id": [1, 2, 3],
        "store_name": ["Store A", "Store B", "Store C"],
        "region": ["North", "South", "East"],
        "store_format": ["Supermarket", "Hypermarket", "Express"],
        "size_sqft": [50000, 80000, 15000],
        "open_date": ["2020-01-01", "2019-06-15", "2021-03-01"],
    })


@pytest.fixture
def sample_products():
    """Sample products DataFrame for testing."""
    return pd.DataFrame({
        "product_id": [1, 2, 3],
        "product_name": ["Product A", "Product B", "Product C"],
        "category": ["Beverages", "Dairy", "Snacks"],
        "subcategory": ["Soft Drinks", "Milk", "Chips"],
        "brand": ["Brand X", "Brand Y", "Brand Z"],
        "unit_price": [2.99, 4.99, 3.49],
        "unit_cost": [1.50, 2.50, 1.75],
        "is_perishable": [False, True, False],
    })


@pytest.fixture
def sample_calendar():
    """Sample calendar DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "date": dates,
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "day_of_week": dates.dayofweek,
        "week_of_year": dates.isocalendar().week,
        "quarter": dates.quarter,
        "is_weekend": dates.dayofweek >= 5,
        "is_holiday": False,
        "days_to_holiday": 30,
        "season": "Winter",
    })


@pytest.fixture
def sample_sales():
    """Sample sales DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100
    
    return pd.DataFrame({
        "transaction_id": range(1, n_rows + 1),
        "date": pd.date_range("2024-01-01", periods=n_rows, freq="D"),
        "store_id": np.random.randint(1, 4, n_rows),
        "product_id": np.random.randint(1, 4, n_rows),
        "quantity": np.random.randint(1, 10, n_rows),
        "unit_price": np.random.uniform(1, 10, n_rows).round(2),
        "total_amount": np.random.uniform(1, 50, n_rows).round(2),
        "discount_applied": np.random.choice([True, False], n_rows),
    })


@pytest.fixture
def sample_daily_sales():
    """Sample aggregated daily sales for testing."""
    np.random.seed(42)
    
    # Create store-product-date combinations
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    stores = [1, 2]
    products = [1, 2, 3]
    
    records = []
    for date in dates:
        for store in stores:
            for product in products:
                records.append({
                    "date": date,
                    "store_id": store,
                    "product_id": product,
                    "daily_quantity": np.random.randint(10, 100),
                    "daily_revenue": round(np.random.uniform(50, 500), 2),
                })
    
    return pd.DataFrame(records)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory with sample files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
