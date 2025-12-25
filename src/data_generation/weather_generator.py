"""
Weather Data Generator

Generates synthetic weather data for each store location.
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_generation.base_generator import BaseGenerator
from src.data_generation.config import DataGenerationConfig
from src.exception import DataGenerationException


class WeatherGenerator(BaseGenerator):
    """Generator for synthetic weather data by store location."""
    
    CONDITIONS = [
        "Clear",
        "Partly Cloudy",
        "Cloudy",
        "Light Rain",
        "Heavy Rain",
        "Thunderstorm",
        "Snow",
        "Fog",
    ]
    
    # Temperature patterns by region and month (in Fahrenheit)
    REGION_TEMPS = {
        "North": {
            1: (15, 30), 2: (18, 35), 3: (28, 45), 4: (38, 55),
            5: (48, 68), 6: (58, 78), 7: (63, 85), 8: (60, 82),
            9: (50, 72), 10: (40, 58), 11: (28, 45), 12: (18, 32),
        },
        "South": {
            1: (40, 60), 2: (42, 65), 3: (50, 72), 4: (58, 80),
            5: (65, 88), 6: (72, 95), 7: (75, 98), 8: (74, 97),
            9: (68, 92), 10: (58, 82), 11: (48, 70), 12: (42, 62),
        },
        "East": {
            1: (25, 40), 2: (28, 44), 3: (35, 52), 4: (45, 62),
            5: (55, 72), 6: (65, 82), 7: (70, 88), 8: (68, 85),
            9: (60, 78), 10: (48, 65), 11: (38, 52), 12: (28, 42),
        },
        "West": {
            1: (42, 58), 2: (44, 62), 3: (46, 65), 4: (50, 70),
            5: (54, 75), 6: (58, 82), 7: (62, 88), 8: (62, 88),
            9: (58, 85), 10: (52, 75), 11: (46, 65), 12: (42, 58),
        },
        "Central": {
            1: (20, 38), 2: (24, 45), 3: (34, 55), 4: (45, 68),
            5: (55, 78), 6: (65, 88), 7: (70, 95), 8: (68, 92),
            9: (58, 82), 10: (45, 68), 11: (32, 52), 12: (22, 40),
        },
    }
    
    @property
    def filename(self) -> str:
        return "weather.csv"
    
    def __init__(
        self,
        config: DataGenerationConfig = None,
        stores_df: pd.DataFrame = None,
    ):
        """
        Initialize the weather generator.
        
        Args:
            config: Data generation configuration.
            stores_df: Pre-generated stores DataFrame.
        """
        super().__init__(config)
        self.stores_df = stores_df
    
    def generate(self) -> pd.DataFrame:
        """
        Generate weather data.
        
        Returns:
            DataFrame with weather information.
        """
        try:
            start_date, end_date = self.config.get_date_range()
            num_days = self.config.get_num_days()
            
            # Get store regions
            if self.stores_df is not None:
                store_regions = dict(zip(
                    self.stores_df["store_id"],
                    self.stores_df["region"]
                ))
                store_ids = self.stores_df["store_id"].tolist()
            else:
                store_ids = list(range(1, self.config.stores.count + 1))
                regions = self.config.stores.regions
                store_regions = {sid: regions[sid % len(regions)] for sid in store_ids}
            
            total_records = num_days * len(store_ids)
            self.logger.info(
                f"Generating {total_records:,} weather records "
                f"({num_days} days × {len(store_ids)} stores)"
            )
            
            weather_records = []
            weather_id = 1
            
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                month = current_date.month
                
                for store_id in store_ids:
                    region = store_regions.get(store_id, "Central")
                    
                    # Get base temperature range for region and month
                    temp_range = self.REGION_TEMPS.get(region, {}).get(
                        month, (40, 70)
                    )
                    
                    # Generate temperatures with some randomness
                    temp_low = int(self.random_normal(temp_range[0], 5))
                    temp_high = int(self.random_normal(temp_range[1], 5))
                    
                    # Ensure high > low
                    if temp_high <= temp_low:
                        temp_high = temp_low + 10
                    
                    # Generate weather condition based on temperature and region
                    condition, precipitation = self._generate_condition(
                        region, month, temp_low
                    )
                    
                    # Humidity
                    base_humidity = 50
                    if "Rain" in condition or "Snow" in condition:
                        base_humidity = 80
                    humidity = int(self.random_normal(base_humidity, 15))
                    humidity = max(20, min(100, humidity))
                    
                    # Wind speed
                    wind_speed = float(self.random_float(0, 25))
                    wind_speed = round(wind_speed, 1)
                    
                    weather_records.append({
                        "weather_id": weather_id,
                        "date": date_str,
                        "store_id": store_id,
                        "temp_high_f": temp_high,
                        "temp_low_f": temp_low,
                        "precipitation_inches": precipitation,
                        "humidity_percent": humidity,
                        "wind_speed_mph": wind_speed,
                        "conditions": condition,
                    })
                    
                    weather_id += 1
                
                current_date += timedelta(days=1)
            
            df = pd.DataFrame(weather_records)
            
            self.logger.info(
                f"Generated {len(df):,} weather records with "
                f"{df['conditions'].nunique()} unique conditions"
            )
            
            return df
            
        except Exception as e:
            raise DataGenerationException(f"Failed to generate weather: {e}", sys)
    
    def _generate_condition(
        self,
        region: str,
        month: int,
        temp_low: int
    ) -> tuple:
        """
        Generate weather condition and precipitation.
        
        Args:
            region: Store region.
            month: Month of year.
            temp_low: Low temperature.
            
        Returns:
            Tuple of (condition, precipitation).
        """
        # Base probabilities
        probs = {
            "Clear": 0.35,
            "Partly Cloudy": 0.25,
            "Cloudy": 0.15,
            "Light Rain": 0.10,
            "Heavy Rain": 0.05,
            "Thunderstorm": 0.03,
            "Snow": 0.05,
            "Fog": 0.02,
        }
        
        # Adjust for winter
        if month in [12, 1, 2]:
            if temp_low < 32:
                probs["Snow"] = 0.20
                probs["Clear"] = 0.25
                probs["Light Rain"] = 0.05
                probs["Heavy Rain"] = 0.02
        
        # Adjust for summer storms
        if month in [6, 7, 8]:
            probs["Thunderstorm"] = 0.08
            probs["Snow"] = 0.0
        
        # Adjust for rainy regions
        if region in ["East", "South"]:
            probs["Light Rain"] += 0.05
            probs["Clear"] -= 0.05
        
        # Normalize probabilities
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        # Select condition
        conditions = list(probs.keys())
        probabilities = list(probs.values())
        condition = self.random_choice(conditions, p=probabilities)
        
        # Generate precipitation
        precipitation = 0.0
        if condition == "Light Rain":
            precipitation = float(self.random_float(0.01, 0.25))
        elif condition == "Heavy Rain":
            precipitation = float(self.random_float(0.25, 2.0))
        elif condition == "Thunderstorm":
            precipitation = float(self.random_float(0.5, 3.0))
        elif condition == "Snow":
            precipitation = float(self.random_float(0.1, 1.0))  # Water equivalent
        
        return condition, round(precipitation, 2)


if __name__ == "__main__":
    generator = WeatherGenerator()
    generator.run()
