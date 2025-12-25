"""
Promotions Data Generator

Generates synthetic promotion/discount campaigns.
"""

import sys
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from src.data_generation.base_generator import BaseGenerator
from src.data_generation.config import DataGenerationConfig
from src.exception import DataGenerationException


class PromotionsGenerator(BaseGenerator):
    """Generator for synthetic promotion data."""
    
    PROMOTION_TYPES = [
        "Percentage Off",
        "Buy One Get One",
        "Bundle Deal",
        "Flash Sale",
        "Clearance",
        "Seasonal",
        "Holiday Special",
        "Member Exclusive",
    ]
    
    @property
    def filename(self) -> str:
        return "promotions.csv"
    
    def __init__(
        self, 
        config: DataGenerationConfig = None,
        stores_df: pd.DataFrame = None,
        products_df: pd.DataFrame = None
    ):
        """
        Initialize the promotions generator.
        
        Args:
            config: Data generation configuration.
            stores_df: Pre-generated stores DataFrame.
            products_df: Pre-generated products DataFrame.
        """
        super().__init__(config)
        self.stores_df = stores_df
        self.products_df = products_df
    
    def generate(self) -> pd.DataFrame:
        """
        Generate promotions data.
        
        Returns:
            DataFrame with promotion information.
        """
        try:
            start_date, end_date = self.config.get_date_range()
            num_months = (end_date.year - start_date.year) * 12 + \
                         (end_date.month - start_date.month) + 1
            
            avg_per_month = self.config.promotions.avg_promotions_per_month
            total_promotions = num_months * avg_per_month
            
            self.logger.info(f"Generating approximately {total_promotions} promotions")
            
            # Get store and product IDs
            if self.stores_df is not None:
                store_ids = self.stores_df["store_id"].tolist()
            else:
                store_ids = list(range(1, self.config.stores.count + 1))
            
            if self.products_df is not None:
                product_ids = self.products_df["product_id"].tolist()
            else:
                product_ids = list(range(1, self.config.products.count + 1))
            
            promotions = []
            promo_id = 1
            
            # Generate promotions month by month
            current_date = start_date
            while current_date <= end_date:
                # Vary number of promotions per month
                num_promos = int(self.random_normal(avg_per_month, avg_per_month * 0.2))
                num_promos = max(50, num_promos)
                
                for _ in range(num_promos):
                    # Random start within the month
                    day_offset = int(self.random_int(0, 27))
                    promo_start = current_date.replace(day=1) + timedelta(days=day_offset)
                    
                    if promo_start > end_date:
                        break
                    
                    # Duration: 1-14 days typically
                    duration = int(self.random_choice(
                        [1, 3, 5, 7, 10, 14],
                        p=[0.1, 0.15, 0.2, 0.3, 0.15, 0.1]
                    ))
                    promo_end = promo_start + timedelta(days=duration - 1)
                    promo_end = min(promo_end, end_date)
                    
                    # Select promotion scope
                    scope = self.random_choice(
                        ["single_store", "regional", "chain_wide"],
                        p=[0.3, 0.4, 0.3]
                    )
                    
                    if scope == "single_store":
                        selected_stores = [self.random_choice(store_ids)]
                    elif scope == "regional":
                        # Select a subset of stores
                        n_stores = int(self.random_int(5, len(store_ids) // 2))
                        selected_stores = list(self.random_choice(store_ids, size=n_stores))
                    else:
                        selected_stores = store_ids
                    
                    # Select product(s) for promotion
                    n_products = int(self.random_choice(
                        [1, 3, 5, 10],
                        p=[0.4, 0.3, 0.2, 0.1]
                    ))
                    selected_products = list(self.random_choice(
                        product_ids, 
                        size=min(n_products, len(product_ids))
                    ))
                    
                    # Discount percentage
                    discount_min, discount_max = self.config.promotions.discount_range
                    discount = int(self.random_choice(
                        [5, 10, 15, 20, 25, 30, 40, 50],
                        p=[0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
                    ))
                    discount = max(discount_min, min(discount_max, discount))
                    
                    # Promotion type
                    promo_type = self.random_choice(self.PROMOTION_TYPES)
                    
                    # Create promotion records (one per store-product combination)
                    for store_id in selected_stores:
                        for product_id in selected_products:
                            promotions.append({
                                "promotion_id": promo_id,
                                "store_id": int(store_id),
                                "product_id": int(product_id),
                                "promotion_type": promo_type,
                                "discount_percent": discount,
                                "start_date": promo_start.strftime("%Y-%m-%d"),
                                "end_date": promo_end.strftime("%Y-%m-%d"),
                                "is_featured": bool(self.random_choice([True, False], p=[0.2, 0.8])),
                            })
                            promo_id += 1
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            
            df = pd.DataFrame(promotions)
            
            self.logger.info(
                f"Generated {len(df)} promotion records covering "
                f"{df['promotion_id'].nunique() if not df.empty else 0} unique campaigns"
            )
            
            return df
            
        except Exception as e:
            raise DataGenerationException(f"Failed to generate promotions: {e}", sys)


if __name__ == "__main__":
    generator = PromotionsGenerator()
    generator.run()
