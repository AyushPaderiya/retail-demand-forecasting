"""
Products Data Generator

Generates synthetic product catalog with realistic attributes.
"""

import sys
from typing import Dict, List, Tuple

import pandas as pd

from src.data_generation.base_generator import BaseGenerator
from src.data_generation.config import DataGenerationConfig
from src.exception import DataGenerationException


class ProductsGenerator(BaseGenerator):
    """Generator for synthetic product data."""
    
    # Category definitions with subcategories
    CATEGORIES = {
        "Beverages": ["Soft Drinks", "Juices", "Water", "Energy Drinks", "Tea & Coffee"],
        "Dairy": ["Milk", "Cheese", "Yogurt", "Butter", "Cream"],
        "Bakery": ["Bread", "Pastries", "Cakes", "Cookies", "Muffins"],
        "Meat & Seafood": ["Beef", "Chicken", "Pork", "Fish", "Shellfish"],
        "Produce": ["Fruits", "Vegetables", "Salads", "Herbs", "Organic"],
        "Frozen Foods": ["Ice Cream", "Frozen Meals", "Frozen Vegetables", "Pizza", "Desserts"],
        "Snacks": ["Chips", "Nuts", "Popcorn", "Crackers", "Candy"],
        "Canned Goods": ["Soups", "Vegetables", "Fruits", "Beans", "Sauces"],
        "Condiments": ["Ketchup", "Mustard", "Mayo", "Dressings", "Spices"],
        "Personal Care": ["Shampoo", "Soap", "Toothpaste", "Deodorant", "Skincare"],
        "Household": ["Cleaning", "Paper Products", "Laundry", "Storage", "Disposables"],
        "Pet Supplies": ["Dog Food", "Cat Food", "Pet Treats", "Pet Toys", "Pet Care"],
        "Baby Products": ["Diapers", "Baby Food", "Formula", "Baby Care", "Baby Toys"],
        "Health": ["Vitamins", "Pain Relief", "Cold & Flu", "First Aid", "Supplements"],
        "Electronics": ["Batteries", "Chargers", "Headphones", "Accessories", "Small Appliances"],
    }
    
    BRAND_PREFIXES = [
        "Fresh", "Natural", "Organic", "Premium", "Value", "Select",
        "Choice", "Pure", "Golden", "Royal", "Classic", "Supreme",
        "Essential", "Daily", "Family", "Smart", "Green", "Healthy"
    ]
    
    BRAND_SUFFIXES = [
        "Foods", "Products", "Brands", "Co", "Inc", "Goods",
        "Essentials", "Naturals", "Organics", "Farms", "Kitchen"
    ]
    
    @property
    def filename(self) -> str:
        return "products.csv"
    
    def generate(self) -> pd.DataFrame:
        """
        Generate products data.
        
        Returns:
            DataFrame with product information.
        """
        try:
            num_products = self.config.products.count
            self.logger.info(f"Generating {num_products} products")
            
            # Generate brands first
            brands = self._generate_brands()
            
            # Generate product data
            products = []
            product_id = 1
            
            # Distribute products across categories
            categories = list(self.CATEGORIES.keys())
            products_per_category = num_products // len(categories)
            extra_products = num_products % len(categories)
            
            for i, category in enumerate(categories):
                subcategories = self.CATEGORIES[category]
                n_products = products_per_category + (1 if i < extra_products else 0)
                
                for _ in range(n_products):
                    subcategory = self.random_choice(subcategories)
                    brand = self.random_choice(brands)
                    
                    # Generate price based on category
                    base_price, cost_ratio = self._get_price_params(category)
                    price = round(float(self.random_normal(base_price, base_price * 0.3)), 2)
                    price = max(0.99, price)  # Minimum price
                    cost = round(price * cost_ratio, 2)
                    
                    # Generate product name
                    name = self._generate_product_name(category, subcategory, brand)
                    
                    products.append({
                        "product_id": product_id,
                        "product_name": name,
                        "category": category,
                        "subcategory": subcategory,
                        "brand": brand,
                        "unit_price": price,
                        "unit_cost": cost,
                        "weight_kg": round(float(self.random_float(0.1, 5.0)), 2),
                        "is_perishable": category in ["Dairy", "Meat & Seafood", "Produce", "Bakery"],
                        "shelf_life_days": self._get_shelf_life(category),
                    })
                    
                    product_id += 1
            
            df = pd.DataFrame(products)
            
            self.logger.info(
                f"Generated {len(df)} products across "
                f"{len(categories)} categories and {len(brands)} brands"
            )
            return df
            
        except Exception as e:
            raise DataGenerationException(f"Failed to generate products: {e}", sys)
    
    def _generate_brands(self) -> List[str]:
        """Generate unique brand names."""
        num_brands = self.config.products.brands
        brands = []
        used = set()
        
        for i in range(num_brands):
            while True:
                prefix = self.random_choice(self.BRAND_PREFIXES)
                suffix = self.random_choice(self.BRAND_SUFFIXES)
                brand = f"{prefix} {suffix}"
                
                if brand not in used:
                    used.add(brand)
                    brands.append(brand)
                    break
        
        return brands
    
    def _generate_product_name(
        self, 
        category: str, 
        subcategory: str, 
        brand: str
    ) -> str:
        """Generate a realistic product name."""
        descriptors = ["Large", "Small", "Family Size", "Value Pack", "Premium", 
                       "Organic", "Regular", "Extra", "Deluxe", "Classic"]
        
        descriptor = self.random_choice(descriptors)
        return f"{brand} {descriptor} {subcategory}"
    
    def _get_price_params(self, category: str) -> Tuple[float, float]:
        """Get base price and cost ratio for a category."""
        price_params = {
            "Beverages": (3.50, 0.55),
            "Dairy": (4.00, 0.60),
            "Bakery": (3.00, 0.50),
            "Meat & Seafood": (12.00, 0.65),
            "Produce": (3.50, 0.55),
            "Frozen Foods": (6.00, 0.55),
            "Snacks": (4.00, 0.50),
            "Canned Goods": (2.50, 0.55),
            "Condiments": (3.50, 0.50),
            "Personal Care": (8.00, 0.45),
            "Household": (7.00, 0.50),
            "Pet Supplies": (15.00, 0.55),
            "Baby Products": (12.00, 0.50),
            "Health": (10.00, 0.45),
            "Electronics": (20.00, 0.60),
        }
        return price_params.get(category, (5.00, 0.55))
    
    def _get_shelf_life(self, category: str) -> int:
        """Get shelf life in days based on category."""
        shelf_life = {
            "Beverages": 180,
            "Dairy": 14,
            "Bakery": 5,
            "Meat & Seafood": 5,
            "Produce": 7,
            "Frozen Foods": 365,
            "Snacks": 180,
            "Canned Goods": 730,
            "Condiments": 365,
            "Personal Care": 730,
            "Household": 1095,
            "Pet Supplies": 365,
            "Baby Products": 365,
            "Health": 730,
            "Electronics": 1825,
        }
        base = shelf_life.get(category, 180)
        # Add some variation
        return int(base * float(self.random_float(0.8, 1.2)))


if __name__ == "__main__":
    generator = ProductsGenerator()
    generator.run()
