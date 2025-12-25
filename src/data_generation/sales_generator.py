"""
Sales Transactions Data Generator

Generates millions of synthetic sales transactions with realistic patterns.
Uses chunked writing for memory efficiency.
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.data_generation.base_generator import BaseGenerator
from src.data_generation.config import DataGenerationConfig
from src.exception import DataGenerationException


class SalesGenerator(BaseGenerator):
    """
    Generator for synthetic sales transaction data.
    
    Implements:
    - Realistic demand patterns (seasonality, trends, day-of-week effects)
    - Promotion impact on sales
    - Weather-influenced demand
    - Memory-efficient chunked generation
    """
    
    @property
    def filename(self) -> str:
        return "sales_transactions.csv"
    
    def __init__(
        self,
        config: DataGenerationConfig = None,
        stores_df: pd.DataFrame = None,
        products_df: pd.DataFrame = None,
        calendar_df: pd.DataFrame = None,
        promotions_df: pd.DataFrame = None,
        customers_df: pd.DataFrame = None,
    ):
        """
        Initialize the sales generator.
        
        Args:
            config: Data generation configuration.
            stores_df: Pre-generated stores DataFrame.
            products_df: Pre-generated products DataFrame.
            calendar_df: Pre-generated calendar DataFrame.
            promotions_df: Pre-generated promotions DataFrame.
            customers_df: Pre-generated customers DataFrame.
        """
        super().__init__(config)
        self.stores_df = stores_df
        self.products_df = products_df
        self.calendar_df = calendar_df
        self.promotions_df = promotions_df
        self.customers_df = customers_df
        
        # Build lookup structures
        self._build_lookups()
    
    def _build_lookups(self):
        """Build lookup dictionaries for efficient access."""
        # Store IDs
        if self.stores_df is not None:
            self.store_ids = self.stores_df["store_id"].tolist()
        else:
            self.store_ids = list(range(1, self.config.stores.count + 1))
        
        # Product info
        if self.products_df is not None:
            self.product_ids = self.products_df["product_id"].tolist()
            self.product_prices = dict(zip(
                self.products_df["product_id"],
                self.products_df["unit_price"]
            ))
            self.product_categories = dict(zip(
                self.products_df["product_id"],
                self.products_df["category"]
            ))
        else:
            self.product_ids = list(range(1, self.config.products.count + 1))
            self.product_prices = {pid: 5.0 for pid in self.product_ids}
            self.product_categories = {pid: "General" for pid in self.product_ids}
        
        # Customer IDs
        if self.customers_df is not None:
            self.customer_ids = self.customers_df["customer_id"].tolist()
        else:
            self.customer_ids = list(range(1, self.config.customers.count + 1))
        
        # Promotion lookup: (date, store_id, product_id) -> discount
        self.promo_lookup: Dict[Tuple[str, int, int], float] = {}
        if self.promotions_df is not None and not self.promotions_df.empty:
            self._build_promo_lookup()
        
        # Calendar lookup
        self.calendar_lookup: Dict[str, Dict] = {}
        if self.calendar_df is not None:
            for _, row in self.calendar_df.iterrows():
                self.calendar_lookup[row["date"]] = row.to_dict()
    
    def _build_promo_lookup(self):
        """Build promotion lookup for quick access."""
        for _, promo in self.promotions_df.iterrows():
            start = datetime.strptime(promo["start_date"], "%Y-%m-%d").date()
            end = datetime.strptime(promo["end_date"], "%Y-%m-%d").date()
            
            current = start
            while current <= end:
                key = (current.strftime("%Y-%m-%d"), promo["store_id"], promo["product_id"])
                self.promo_lookup[key] = promo["discount_percent"] / 100.0
                current += timedelta(days=1)
    
    def generate(self) -> pd.DataFrame:
        """
        Generate sales data using chunked approach.
        
        Returns:
            DataFrame with sales transactions.
        """
        # Use chunked generation for memory efficiency
        start_date, end_date = self.config.get_date_range()
        num_days = self.config.get_num_days()
        
        avg_daily = self.config.sales.avg_transactions_per_store_per_day
        estimated_total = num_days * len(self.store_ids) * avg_daily
        max_rows = self.config.sales.max_total_rows
        
        # Scale down if needed
        if estimated_total > max_rows:
            scale_factor = max_rows / estimated_total
            avg_daily = int(avg_daily * scale_factor)
            estimated_total = num_days * len(self.store_ids) * avg_daily
        
        self.logger.info(
            f"Generating ~{estimated_total:,} sales transactions "
            f"(~{avg_daily} per store per day)"
        )
        
        # Generate using chunked approach and save
        generator = self._generate_chunks(start_date, end_date, avg_daily)
        self.save_chunked(generator, estimated_total)
        
        # Return empty DataFrame as data is already saved
        return pd.DataFrame()
    
    def run(self) -> str:
        """Execute generation with chunked saving."""
        self.logger.info(f"Starting generation for {self.filename}")
        self.generate()
        self.logger.info(f"Completed generation for {self.filename}")
        return str(self.output_path)
    
    def _generate_chunks(
        self,
        start_date: datetime,
        end_date: datetime,
        avg_daily_per_store: int
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Generate sales data in chunks.
        
        Yields:
            DataFrames of sales transactions.
        """
        chunk_size = self.config.chunk_size
        transaction_id = 1
        buffer = []
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Get calendar info
            cal_info = self.calendar_lookup.get(date_str, {})
            is_weekend = cal_info.get("is_weekend", current_date.weekday() >= 5)
            is_holiday = cal_info.get("is_holiday", False)
            day_of_week = cal_info.get("day_of_week", current_date.weekday())
            month = current_date.month
            
            # Calculate demand multipliers
            base_multiplier = self._get_demand_multiplier(
                month, day_of_week, is_weekend, is_holiday
            )
            
            for store_id in self.store_ids:
                # Vary transactions per store
                store_multiplier = float(self.random_normal(1.0, 0.2))
                store_multiplier = max(0.5, min(1.5, store_multiplier))
                
                num_transactions = int(
                    avg_daily_per_store * base_multiplier * store_multiplier
                )
                
                # Select products for this store today
                # Not all products sold every day - randomly select subset
                num_products_today = int(self.random_int(
                    len(self.product_ids) // 10,
                    len(self.product_ids) // 3
                ))
                products_today = list(self.random_choice(
                    self.product_ids,
                    size=num_products_today
                ))
                
                for _ in range(num_transactions):
                    product_id = int(self.random_choice(products_today))
                    
                    # Base price
                    base_price = self.product_prices.get(product_id, 5.0)
                    
                    # Check for promotion
                    promo_key = (date_str, store_id, product_id)
                    discount = self.promo_lookup.get(promo_key, 0.0)
                    
                    # Apply discount
                    unit_price = base_price * (1 - discount)
                    unit_price = round(unit_price, 2)
                    
                    # Quantity (promotions increase quantity)
                    q_min, q_max = self.config.sales.quantity_range
                    if discount > 0:
                        q_max = min(q_max * 2, 20)  # Higher qty during promos
                    
                    quantity = int(self.random_int(q_min, q_max))
                    
                    # Total amount
                    total_amount = round(unit_price * quantity, 2)
                    
                    # Customer (some transactions are anonymous)
                    customer_id = None
                    if self.random_float(0, 1) < 0.7:  # 70% have customer ID
                        customer_id = int(self.random_choice(self.customer_ids))
                    
                    buffer.append({
                        "transaction_id": transaction_id,
                        "date": date_str,
                        "store_id": store_id,
                        "product_id": product_id,
                        "customer_id": customer_id,
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "discount_applied": discount > 0,
                        "discount_percent": int(discount * 100) if discount > 0 else 0,
                        "total_amount": total_amount,
                    })
                    
                    transaction_id += 1
                    
                    # Yield chunk when buffer is full
                    if len(buffer) >= chunk_size:
                        yield pd.DataFrame(buffer)
                        buffer = []
            
            current_date += timedelta(days=1)
        
        # Yield remaining records
        if buffer:
            yield pd.DataFrame(buffer)
    
    def _get_demand_multiplier(
        self,
        month: int,
        day_of_week: int,
        is_weekend: bool,
        is_holiday: bool
    ) -> float:
        """
        Calculate demand multiplier based on temporal factors.
        
        Args:
            month: Month (1-12).
            day_of_week: Day of week (0=Monday, 6=Sunday).
            is_weekend: Whether it's a weekend.
            is_holiday: Whether it's a holiday.
            
        Returns:
            Demand multiplier.
        """
        multiplier = 1.0
        
        # Seasonal effect
        seasonal = {
            1: 0.9,   # January - post-holiday lull
            2: 0.95,
            3: 1.0,
            4: 1.0,
            5: 1.05,
            6: 1.1,   # Summer
            7: 1.15,
            8: 1.1,
            9: 1.0,
            10: 1.05,
            11: 1.2,  # Pre-holiday shopping
            12: 1.4,  # Holiday season peak
        }
        multiplier *= seasonal.get(month, 1.0)
        
        # Day of week effect
        dow_effect = {
            0: 0.9,   # Monday
            1: 0.95,
            2: 1.0,
            3: 1.0,
            4: 1.2,   # Friday
            5: 1.3,   # Saturday
            6: 1.1,   # Sunday
        }
        multiplier *= dow_effect.get(day_of_week, 1.0)
        
        # Holiday boost
        if is_holiday:
            multiplier *= 1.5
        
        # Add some random noise
        noise = float(self.random_normal(1.0, 0.1))
        multiplier *= max(0.7, min(1.3, noise))
        
        return multiplier


if __name__ == "__main__":
    generator = SalesGenerator()
    generator.run()
