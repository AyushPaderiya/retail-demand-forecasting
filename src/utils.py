"""
Utility Functions

Common utility functions used across the application.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import yaml

from src.exception import ConfigurationException, CustomException
from src.logger import logger


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config file. Defaults to project root config.yaml.
        
    Returns:
        Dictionary containing configuration.
        
    Raises:
        ConfigurationException: If config file cannot be loaded.
    """
    try:
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationException(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ConfigurationException(f"Error parsing YAML config: {e}", sys)
    except Exception as e:
        raise ConfigurationException(f"Error loading config: {e}", sys)


def save_object(file_path: str, obj: Any) -> None:
    """
    Save a Python object to disk using joblib.
    
    Args:
        file_path: Path where the object will be saved.
        obj: Object to save.
        
    Raises:
        CustomException: If object cannot be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        joblib.dump(obj, file_path)
        logger.info(f"Object saved to {file_path}")
        
    except Exception as e:
        raise CustomException(f"Error saving object to {file_path}: {e}", sys)


def load_object(file_path: str) -> Any:
    """
    Load a Python object from disk using joblib.
    
    Args:
        file_path: Path to the saved object.
        
    Returns:
        The loaded object.
        
    Raises:
        CustomException: If object cannot be loaded.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        obj = joblib.load(file_path)
        logger.info(f"Object loaded from {file_path}")
        return obj
        
    except Exception as e:
        raise CustomException(f"Error loading object from {file_path}: {e}", sys)


def create_directories(directories: List[str]) -> None:
    """
    Create multiple directories if they don't exist.
    
    Args:
        directories: List of directory paths to create.
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {directory}")


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    df_name: str = "DataFrame"
) -> bool:
    """
    Validate that a DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
        df_name: Name of the DataFrame for error messages.
        
    Returns:
        True if validation passes.
        
    Raises:
        CustomException: If validation fails.
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise CustomException(
            f"{df_name} is missing required columns: {missing_columns}",
            sys
        )
    
    logger.debug(f"{df_name} validation passed with {len(df)} rows")
    return True


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root.
    """
    return Path(__file__).parent.parent


def ensure_artifacts_directories() -> Dict[str, str]:
    """
    Ensure all artifact directories exist.
    
    Returns:
        Dictionary with artifact directory paths.
    """
    config = load_config()
    artifacts_config = config.get("artifacts", {})
    
    directories = {
        "base": artifacts_config.get("base_dir", "artifacts"),
        "data": artifacts_config.get("data_dir", "artifacts/data"),
        "models": artifacts_config.get("models_dir", "artifacts/models"),
        "logs": artifacts_config.get("logs_dir", "artifacts/logs"),
    }
    
    create_directories(list(directories.values()))
    return directories


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        
    Returns:
        Dictionary of metric names and values.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Handle edge cases
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # R2
    r2 = r2_score(y_true, y_pred)
    
    # Forecast Bias
    forecast_bias = np.mean(y_pred - y_true)
    
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 4) if not np.isnan(mape) else None,
        "r2": round(r2, 4),
        "forecast_bias": round(forecast_bias, 4)
    }


def format_number(number: float, decimals: int = 2) -> str:
    """
    Format a number with thousands separator.
    
    Args:
        number: Number to format.
        decimals: Number of decimal places.
        
    Returns:
        Formatted string.
    """
    return f"{number:,.{decimals}f}"


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get memory usage of a DataFrame in human-readable format.
    
    Args:
        df: DataFrame to analyze.
        
    Returns:
        Memory usage string (e.g., "125.5 MB").
    """
    bytes_used = df.memory_usage(deep=True).sum()
    
    if bytes_used < 1024:
        return f"{bytes_used:.1f} B"
    elif bytes_used < 1024 ** 2:
        return f"{bytes_used / 1024:.1f} KB"
    elif bytes_used < 1024 ** 3:
        return f"{bytes_used / (1024 ** 2):.1f} MB"
    else:
        return f"{bytes_used / (1024 ** 3):.1f} GB"


if __name__ == "__main__":
    # Test utilities
    config = load_config()
    print(f"Project name: {config['project']['name']}")
    
    dirs = ensure_artifacts_directories()
    print(f"Artifact directories: {dirs}")
