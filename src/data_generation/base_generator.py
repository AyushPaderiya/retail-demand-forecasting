"""
Base Generator Class

Abstract base class for all data generators with common functionality.
"""
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator, List 

import numpy as np
import pandas as pd

from src.data_generation.config import DataGenerationConfig
from src.exception import DataGenerationException
from src.logger import get_logger


class BaseGenerator(ABC):
    """
    Abstract base class for synthetic data generators.
    
    Provides common functionality for:
    - Configuration management
    - Random seed handling
    - Chunked file writing
    - Logging and exception handling
    """
    
    def __init__(self, config: DataGenerationConfig = None):
        """
        Initialize the base generator.
        
        Args:
            config: Data generation configuration. If None, loads from YAML.
        """
        self.config = config or DataGenerationConfig.from_yaml()
        self.logger = get_logger(self.__class__.__name__)
        self.rng = np.random.default_rng(self.config.seed)
        
        # Ensure output directory exists
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    @abstractmethod
    def filename(self) -> str:
        """Return the output filename for this generator."""
        pass
    
    @property
    def output_path(self) -> Path:
        """Get the full output path for the generated file."""
        return self.output_dir / self.filename
    
    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """
        Generate the synthetic data.
        
        Returns:
            Generated DataFrame.
        """
        pass
    
    def save(self, df: pd.DataFrame) -> str:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save.
            
        Returns:
            Path to saved file.
        """
        try:
            output_path = self.output_path
            df.to_csv(output_path, index=False)
            
            self.logger.info(
                f"Saved {len(df):,} rows to {output_path} "
                f"({self._get_file_size(output_path)})"
            )
            
            return str(output_path)
            
        except Exception as e:
            raise DataGenerationException(
                f"Failed to save {self.filename}: {e}", 
                sys
            )
    
    def save_chunked(
        self, 
        data_generator: Generator[pd.DataFrame, None, None],
        total_rows: int = None
    ) -> str:
        """
        Save data in chunks for memory efficiency.
        
        Args:
            data_generator: Generator yielding DataFrames.
            total_rows: Expected total rows (for logging).
            
        Returns:
            Path to saved file.
        """
        try:
            output_path = self.output_path
            first_chunk = True
            rows_written = 0
            chunk_count = 0
            
            for chunk_df in data_generator:
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                
                chunk_df.to_csv(
                    output_path, 
                    mode=mode, 
                    header=header, 
                    index=False
                )
                
                rows_written += len(chunk_df)
                chunk_count += 1
                first_chunk = False
                
                if chunk_count % 10 == 0:
                    progress = ""
                    if total_rows:
                        pct = (rows_written / total_rows) * 100
                        progress = f" ({pct:.1f}%)"
                    self.logger.info(
                        f"Written {rows_written:,} rows{progress}"
                    )
            
            self.logger.info(
                f"Completed: {rows_written:,} rows in {chunk_count} chunks "
                f"to {output_path} ({self._get_file_size(output_path)})"
            )
            
            return str(output_path)
            
        except Exception as e:
            raise DataGenerationException(
                f"Failed to save chunked data for {self.filename}: {e}",
                sys
            )
    
    def run(self) -> str:
        """
        Execute the generation and saving process.
        
        Returns:
            Path to the generated file.
        """
        try:
            self.logger.info(f"Starting generation for {self.filename}")
            df = self.generate()
            output_path = self.save(df)
            self.logger.info(f"Completed generation for {self.filename}")
            return output_path
            
        except DataGenerationException:
            raise
        except Exception as e:
            raise DataGenerationException(
                f"Unexpected error in {self.__class__.__name__}: {e}",
                sys
            )
    
    def _get_file_size(self, file_path: Path) -> str:
        """Get human-readable file size."""
        size_bytes = file_path.stat().st_size
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def random_choice(
        self, 
        options: List[Any], 
        size: int = None,
        p: List[float] = None
    ) -> Any:
        """
        Make a random choice from options.
        
        Args:
            options: List of options to choose from.
            size: Number of choices to make.
            p: Probability weights for each option.
            
        Returns:
            Selected option(s).
        """
        return self.rng.choice(options, size=size, p=p)
    
    def random_int(
        self, 
        low: int, 
        high: int, 
        size: int = None
    ) -> np.ndarray:
        """Generate random integers in range [low, high]."""
        return self.rng.integers(low, high + 1, size=size)
    
    def random_float(
        self, 
        low: float, 
        high: float, 
        size: int = None
    ) -> np.ndarray:
        """Generate random floats in range [low, high)."""
        return self.rng.uniform(low, high, size=size)
    
    def random_normal(
        self, 
        mean: float, 
        std: float, 
        size: int = None
    ) -> np.ndarray:
        """Generate random values from normal distribution."""
        return self.rng.normal(mean, std, size=size)
