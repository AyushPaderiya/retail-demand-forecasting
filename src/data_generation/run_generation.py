"""
Data Generation Orchestrator

Main orchestrator that runs all data generators in the correct order.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data_generation.base_generator import BaseGenerator
from src.data_generation.config import DataGenerationConfig
from src.data_generation.stores_generator import StoresGenerator
from src.data_generation.products_generator import ProductsGenerator
from src.data_generation.calendar_generator import CalendarGenerator
from src.data_generation.promotions_generator import PromotionsGenerator
from src.data_generation.customers_generator import CustomersGenerator
from src.data_generation.weather_generator import WeatherGenerator
from src.data_generation.sales_generator import SalesGenerator
from src.exception import DataGenerationException
from src.logger import get_logger


class DataGenerationOrchestrator:
    """
    Orchestrates the generation of all synthetic datasets.
    
    Ensures generators run in the correct order with proper dependencies.
    """
    
    def __init__(self, config: DataGenerationConfig = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Data generation configuration. If None, loads from YAML.
        """
        self.config = config or DataGenerationConfig.from_yaml()
        self.logger = get_logger("DataGenerationOrchestrator")
        
        # Store generated DataFrames for dependencies
        self.generated_data: Dict[str, pd.DataFrame] = {}
        self.generated_files: Dict[str, str] = {}
        self.generation_times: Dict[str, float] = {}
    
    def run(
        self,
        generators: List[str] = None,
        skip_existing: bool = False
    ) -> Dict[str, str]:
        """
        Run all or selected data generators.
        
        Args:
            generators: List of generators to run. If None, runs all.
                Options: stores, products, calendar, promotions, 
                         customers, weather, sales
            skip_existing: If True, skip generation if file already exists.
            
        Returns:
            Dictionary mapping generator names to output file paths.
        """
        start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("Starting Data Generation Pipeline")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"Random seed: {self.config.seed}")
        self.logger.info("=" * 60)
        
        # Define generation order (dependencies matter!)
        all_generators = [
            "stores",
            "products", 
            "calendar",
            "customers",
            "promotions",
            "weather",
            "sales",  # Must be last - depends on all others
        ]
        
        # Filter if specific generators requested
        if generators:
            generators_to_run = [g for g in all_generators if g in generators]
        else:
            generators_to_run = all_generators
        
        try:
            for gen_name in generators_to_run:
                self._run_generator(gen_name, skip_existing)
            
            total_time = time.time() - start_time
            self._log_summary(total_time)
            
            return self.generated_files
            
        except DataGenerationException:
            raise
        except Exception as e:
            raise DataGenerationException(
                f"Data generation pipeline failed: {e}",
                sys
            )
    
    def _run_generator(self, name: str, skip_existing: bool = False) -> None:
        """
        Run a specific generator.
        
        Args:
            name: Generator name.
            skip_existing: Skip if output file exists.
        """
        self.logger.info("-" * 40)
        self.logger.info(f"Running {name.upper()} generator")
        
        gen_start = time.time()
        
        try:
            generator = self._create_generator(name)
            
            # Check if file exists
            if skip_existing and generator.output_path.exists():
                self.logger.info(f"Skipping {name} - file already exists")
                # Load existing file for dependencies
                self.generated_data[name] = pd.read_csv(generator.output_path)
                self.generated_files[name] = str(generator.output_path)
                return
            
            # Run generator
            df = generator.generate()
            
            # Save if not empty (sales generator saves internally)
            if not df.empty:
                generator.save(df)
                self.generated_data[name] = df
            elif generator.output_path.exists():
                # For chunked generators like sales, load summary
                self.generated_data[name] = pd.DataFrame()
            
            self.generated_files[name] = str(generator.output_path)
            
            gen_time = time.time() - gen_start
            self.generation_times[name] = gen_time
            
            self.logger.info(f"Completed {name} in {gen_time:.2f}s")
            
        except Exception as e:
            raise DataGenerationException(
                f"Failed to run {name} generator: {e}",
                sys
            )
    
    def _create_generator(self, name: str) -> BaseGenerator:
        """
        Create a generator instance with dependencies.
        
        Args:
            name: Generator name.
            
        Returns:
            Generator instance.
        """
        if name == "stores":
            return StoresGenerator(self.config)
        
        elif name == "products":
            return ProductsGenerator(self.config)
        
        elif name == "calendar":
            return CalendarGenerator(self.config)
        
        elif name == "customers":
            return CustomersGenerator(self.config)
        
        elif name == "promotions":
            return PromotionsGenerator(
                self.config,
                stores_df=self.generated_data.get("stores"),
                products_df=self.generated_data.get("products"),
            )
        
        elif name == "weather":
            return WeatherGenerator(
                self.config,
                stores_df=self.generated_data.get("stores"),
            )
        
        elif name == "sales":
            return SalesGenerator(
                self.config,
                stores_df=self.generated_data.get("stores"),
                products_df=self.generated_data.get("products"),
                calendar_df=self.generated_data.get("calendar"),
                promotions_df=self.generated_data.get("promotions"),
                customers_df=self.generated_data.get("customers"),
            )
        
        else:
            raise DataGenerationException(f"Unknown generator: {name}")
    
    def _log_summary(self, total_time: float) -> None:
        """Log generation summary."""
        self.logger.info("=" * 60)
        self.logger.info("DATA GENERATION COMPLETE")
        self.logger.info("=" * 60)
        
        for name, path in self.generated_files.items():
            file_path = Path(path)
            if file_path.exists():
                size = self._format_size(file_path.stat().st_size)
                gen_time = self.generation_times.get(name, 0)
                self.logger.info(f"  {name}: {size} ({gen_time:.2f}s)")
        
        self.logger.info("-" * 40)
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Output directory: {self.config.output_dir}")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size for display."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


def main():
    """Main entry point for data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic retail data"
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        choices=["stores", "products", "calendar", "promotions",
                 "customers", "weather", "sales"],
        help="Specific generators to run (default: all)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generation if file already exists"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = DataGenerationConfig.from_yaml(args.config)
    else:
        config = DataGenerationConfig.from_yaml()
    
    # Run orchestrator
    orchestrator = DataGenerationOrchestrator(config)
    files = orchestrator.run(
        generators=args.generators,
        skip_existing=args.skip_existing
    )
    
    print("\nGenerated files:")
    for name, path in files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
