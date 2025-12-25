"""
Customers Data Generator

Generates synthetic customer data with segments and attributes.
"""

import sys
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from src.data_generation.base_generator import BaseGenerator
from src.data_generation.config import DataGenerationConfig
from src.exception import DataGenerationException


class CustomersGenerator(BaseGenerator):
    """Generator for synthetic customer data."""
    
    @property
    def filename(self) -> str:
        return "customers.csv"
    
    def generate(self) -> pd.DataFrame:
        """
        Generate customers data.
        
        Returns:
            DataFrame with customer information.
        """
        try:
            num_customers = self.config.customers.count
            segments = self.config.customers.segments
            
            self.logger.info(f"Generating {num_customers:,} customers")
            
            # Segment distribution (realistic retail distribution)
            segment_weights = {
                "Regular": 0.40,
                "Premium": 0.25,
                "VIP": 0.05,
                "New": 0.20,
                "Inactive": 0.10,
            }
            
            # Ensure all configured segments have weights
            for seg in segments:
                if seg not in segment_weights:
                    segment_weights[seg] = 0.1
            
            # Normalize weights for configured segments
            total_weight = sum(segment_weights.get(s, 0.1) for s in segments)
            segment_probs = [segment_weights.get(s, 0.1) / total_weight for s in segments]
            
            # Generate customer attributes
            customer_segments = list(self.random_choice(
                segments, size=num_customers, p=segment_probs
            ))
            
            # Generate first purchase dates
            start_date, end_date = self.config.get_date_range()
            date_range_days = (end_date - start_date).days
            
            first_purchase_dates = []
            for segment in customer_segments:
                if segment == "New":
                    # New customers joined in last 90 days
                    days_ago = int(self.random_int(0, 90))
                else:
                    # Other customers joined throughout the period
                    days_ago = int(self.random_int(0, date_range_days))
                
                purchase_date = end_date - timedelta(days=days_ago)
                first_purchase_dates.append(purchase_date.strftime("%Y-%m-%d"))
            
            # Generate lifetime values based on segment
            lifetime_values = []
            for segment in customer_segments:
                lvt_params = {
                    "VIP": (5000, 2000),
                    "Premium": (1500, 500),
                    "Regular": (500, 200),
                    "New": (100, 50),
                    "Inactive": (200, 100),
                }
                mean, std = lvt_params.get(segment, (300, 150))
                ltv = float(self.random_normal(mean, std))
                ltv = max(10, ltv)  # Minimum LTV
                lifetime_values.append(round(ltv, 2))
            
            # Generate purchase frequency (purchases per month)
            purchase_frequencies = []
            for segment in customer_segments:
                freq_params = {
                    "VIP": (12, 3),
                    "Premium": (6, 2),
                    "Regular": (3, 1),
                    "New": (1, 0.5),
                    "Inactive": (0.2, 0.1),
                }
                mean, std = freq_params.get(segment, (2, 1))
                freq = float(self.random_normal(mean, std))
                freq = max(0.1, freq)
                purchase_frequencies.append(round(freq, 1))
            
            # Generate average basket size
            basket_sizes = []
            for segment in customer_segments:
                basket_params = {
                    "VIP": (150, 50),
                    "Premium": (80, 25),
                    "Regular": (45, 15),
                    "New": (35, 10),
                    "Inactive": (30, 10),
                }
                mean, std = basket_params.get(segment, (40, 15))
                basket = float(self.random_normal(mean, std))
                basket = max(10, basket)
                basket_sizes.append(round(basket, 2))
            
            # Generate customer age groups
            age_groups = list(self.random_choice(
                ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
                size=num_customers,
                p=[0.12, 0.25, 0.23, 0.18, 0.14, 0.08]
            ))
            
            # Generate preferred shopping channels
            channels = list(self.random_choice(
                ["In-Store", "Online", "Both"],
                size=num_customers,
                p=[0.45, 0.25, 0.30]
            ))
            
            # Email opt-in
            email_optin = list(self.random_choice(
                [True, False],
                size=num_customers,
                p=[0.65, 0.35]
            ))
            
            df = pd.DataFrame({
                "customer_id": list(range(1, num_customers + 1)),
                "segment": customer_segments,
                "first_purchase_date": first_purchase_dates,
                "lifetime_value": lifetime_values,
                "purchase_frequency_monthly": purchase_frequencies,
                "avg_basket_size": basket_sizes,
                "age_group": age_groups,
                "preferred_channel": channels,
                "email_opt_in": email_optin,
            })
            
            # Log segment distribution
            segment_counts = df["segment"].value_counts()
            self.logger.info(f"Customer segment distribution:\n{segment_counts.to_string()}")
            
            return df
            
        except Exception as e:
            raise DataGenerationException(f"Failed to generate customers: {e}", sys)


if __name__ == "__main__":
    generator = CustomersGenerator()
    generator.run()
