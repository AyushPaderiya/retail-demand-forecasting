"""
Stores Data Generator

Generates synthetic store data with realistic attributes.
"""

import sys
from typing import List

import pandas as pd

from src.data_generation.base_generator import BaseGenerator
from src.data_generation.config import DataGenerationConfig
from src.exception import DataGenerationException


class StoresGenerator(BaseGenerator):
    """Generator for synthetic store data."""
    
    # Store name prefixes and suffixes for realistic names
    NAME_PREFIXES = [
        "Metro", "City", "Central", "Grand", "Prime", "Value", "Fresh",
        "Quick", "Super", "Mega", "Urban", "Local", "Express", "Daily",
        "Family", "Smart", "Green", "Golden", "Royal", "Elite"
    ]
    
    NAME_SUFFIXES = [
        "Mart", "Store", "Market", "Plaza", "Center", "Outlet", "Shop",
        "Hub", "Point", "Zone", "Square", "Corner", "Place", "Stop"
    ]
    
    @property
    def filename(self) -> str:
        return "stores.csv"
    
    def generate(self) -> pd.DataFrame:
        """
        Generate stores data.
        
        Returns:
            DataFrame with store information.
        """
        try:
            num_stores = self.config.stores.count
            self.logger.info(f"Generating {num_stores} stores")
            
            # Generate store IDs
            store_ids = list(range(1, num_stores + 1))
            
            # Generate store names
            names = self._generate_store_names(num_stores)
            
            # Assign regions (balanced distribution)
            regions = self._assign_regions(num_stores)
            
            # Assign store formats
            formats = self._assign_formats(num_stores)
            
            # Generate store sizes based on format
            sizes = self._generate_sizes(formats)
            
            # Generate open dates (stores opened between 2010 and start_date)
            open_dates = self._generate_open_dates(num_stores)
            
            # Generate latitude/longitude for each region
            lat, lon = self._generate_coordinates(regions)
            
            df = pd.DataFrame({
                "store_id": store_ids,
                "store_name": names,
                "region": regions,
                "store_format": formats,
                "size_sqft": sizes,
                "open_date": open_dates,
                "latitude": lat,
                "longitude": lon,
                "manager_count": self.random_int(2, 8, num_stores),
                "employee_count": self.random_int(20, 200, num_stores),
            })
            
            self.logger.info(f"Generated {len(df)} stores across {len(set(regions))} regions")
            return df
            
        except Exception as e:
            raise DataGenerationException(f"Failed to generate stores: {e}", sys)
    
    def _generate_store_names(self, count: int) -> List[str]:
        """Generate unique store names."""
        names = []
        used_names = set()
        
        for i in range(count):
            while True:
                prefix = self.random_choice(self.NAME_PREFIXES)
                suffix = self.random_choice(self.NAME_SUFFIXES)
                name = f"{prefix} {suffix} #{i + 1}"
                
                if name not in used_names:
                    used_names.add(name)
                    names.append(name)
                    break
        
        return names
    
    def _assign_regions(self, count: int) -> List[str]:
        """Assign regions with balanced distribution."""
        regions = self.config.stores.regions
        base_count = count // len(regions)
        remainder = count % len(regions)
        
        result = []
        for i, region in enumerate(regions):
            n = base_count + (1 if i < remainder else 0)
            result.extend([region] * n)
        
        self.rng.shuffle(result)
        return result
    
    def _assign_formats(self, count: int) -> List[str]:
        """Assign store formats with weighted distribution."""
        formats = self.config.stores.formats
        # Weights: Supermarket most common, Online least common
        weights = [0.4, 0.25, 0.25, 0.1]
        
        if len(formats) != len(weights):
            weights = [1/len(formats)] * len(formats)
        
        return list(self.random_choice(formats, size=count, p=weights))
    
    def _generate_sizes(self, formats: List[str]) -> List[int]:
        """Generate store sizes based on format."""
        size_ranges = {
            "Hypermarket": (80000, 150000),
            "Supermarket": (30000, 80000),
            "Express": (5000, 15000),
            "Online": (10000, 30000),  # Warehouse size
        }
        
        sizes = []
        for fmt in formats:
            low, high = size_ranges.get(fmt, (20000, 50000))
            size = int(self.random_int(low, high))
            sizes.append(size)
        
        return sizes
    
    def _generate_open_dates(self, count: int) -> List[str]:
        """Generate store opening dates."""
        from datetime import datetime, timedelta
        
        start_date, _ = self.config.get_date_range()
        earliest_open = datetime(2010, 1, 1).date()
        
        date_range = (start_date - earliest_open).days
        
        dates = []
        for _ in range(count):
            days_offset = int(self.random_int(0, date_range))
            open_date = earliest_open + timedelta(days=days_offset)
            dates.append(open_date.strftime("%Y-%m-%d"))
        
        return dates
    
    def _generate_coordinates(self, regions: List[str]) -> tuple:
        """Generate lat/lon coordinates based on region."""
        # Approximate US region centers
        region_coords = {
            "North": (45.0, -93.0),
            "South": (32.0, -95.0),
            "East": (40.0, -74.0),
            "West": (37.0, -122.0),
            "Central": (39.0, -98.0),
        }
        
        lats = []
        lons = []
        
        for region in regions:
            base_lat, base_lon = region_coords.get(region, (39.0, -98.0))
            # Add some randomness (±2 degrees)
            lat = base_lat + float(self.random_normal(0, 1))
            lon = base_lon + float(self.random_normal(0, 1))
            lats.append(round(lat, 4))
            lons.append(round(lon, 4))
        
        return lats, lons


if __name__ == "__main__":
    generator = StoresGenerator()
    generator.run()
