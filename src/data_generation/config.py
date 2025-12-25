"""
Data Generation Configuration

Centralized configuration for synthetic data generation.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, Any
from pathlib import Path

import yaml


@dataclass
class StoreConfig:
    """Configuration for store data generation."""
    count: int = 50
    regions: List[str] = field(default_factory=lambda: [
        "North", "South", "East", "West", "Central"
    ])
    formats: List[str] = field(default_factory=lambda: [
        "Supermarket", "Hypermarket", "Express", "Online"
    ])


@dataclass
class ProductConfig:
    """Configuration for product data generation."""
    count: int = 1000
    categories: int = 50
    subcategories: int = 200
    brands: int = 100


@dataclass
class SalesConfig:
    """Configuration for sales data generation."""
    avg_transactions_per_store_per_day: int = 90
    max_total_rows: int = 5000000
    quantity_range: tuple = (1, 10)


@dataclass
class CustomerConfig:
    """Configuration for customer data generation."""
    count: int = 50000
    segments: List[str] = field(default_factory=lambda: [
        "Regular", "Premium", "VIP", "New", "Inactive"
    ])


@dataclass
class PromotionConfig:
    """Configuration for promotion data generation."""
    avg_promotions_per_month: int = 200
    discount_range: tuple = (5, 50)


@dataclass
class DataGenerationConfig:
    """Master configuration for data generation."""
    seed: int = 42
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    chunk_size: int = 50000
    output_dir: str = "artifacts/data"
    
    stores: StoreConfig = field(default_factory=StoreConfig)
    products: ProductConfig = field(default_factory=ProductConfig)
    sales: SalesConfig = field(default_factory=SalesConfig)
    customers: CustomerConfig = field(default_factory=CustomerConfig)
    promotions: PromotionConfig = field(default_factory=PromotionConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str = None) -> "DataGenerationConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. Defaults to project config.yaml.
            
        Returns:
            DataGenerationConfig instance.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        
        dg_config = yaml_config.get("data_generation", {})
        
        return cls(
            seed=dg_config.get("seed", 42),
            start_date=dg_config.get("start_date", "2022-01-01"),
            end_date=dg_config.get("end_date", "2024-12-31"),
            chunk_size=dg_config.get("chunk_size", 50000),
            output_dir=dg_config.get("output_dir", "artifacts/data"),
            stores=StoreConfig(
                count=dg_config.get("stores", {}).get("count", 50),
                regions=dg_config.get("stores", {}).get("regions", [
                    "North", "South", "East", "West", "Central"
                ]),
                formats=dg_config.get("stores", {}).get("formats", [
                    "Supermarket", "Hypermarket", "Express", "Online"
                ]),
            ),
            products=ProductConfig(
                count=dg_config.get("products", {}).get("count", 1000),
                categories=dg_config.get("products", {}).get("categories", 50),
                subcategories=dg_config.get("products", {}).get("subcategories", 200),
                brands=dg_config.get("products", {}).get("brands", 100),
            ),
            sales=SalesConfig(
                avg_transactions_per_store_per_day=dg_config.get("sales", {}).get(
                    "avg_transactions_per_store_per_day", 90
                ),
                max_total_rows=dg_config.get("sales", {}).get("max_total_rows", 5000000),
                quantity_range=tuple(dg_config.get("sales", {}).get("quantity_range", [1, 10])),
            ),
            customers=CustomerConfig(
                count=dg_config.get("customers", {}).get("count", 50000),
                segments=dg_config.get("customers", {}).get("segments", [
                    "Regular", "Premium", "VIP", "New", "Inactive"
                ]),
            ),
            promotions=PromotionConfig(
                avg_promotions_per_month=dg_config.get("promotions", {}).get(
                    "avg_promotions_per_month", 200
                ),
                discount_range=tuple(dg_config.get("promotions", {}).get("discount_range", [5, 50])),
            ),
        )
    
    def get_date_range(self) -> tuple:
        """Get start and end dates as date objects."""
        from datetime import datetime
        start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        return start, end
    
    def get_num_days(self) -> int:
        """Get total number of days in the date range."""
        start, end = self.get_date_range()
        return (end - start).days + 1
