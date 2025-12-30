"""
Centralized Logging Module

Provides structured logging for the entire application with file and console output.
"""

import logging
from datetime import datetime
from pathlib import Path

import yaml


def load_logging_config() -> dict:
    """Load logging configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config.get("logging", {})
    return {}


def get_log_file_path() -> str:
    """Generate log file path with timestamp."""
    config = load_logging_config()
    log_dir = config.get("log_dir", "artifacts/logs")
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with date
    log_filename = f"application_{datetime.now().strftime('%Y_%m_%d')}.log"
    return str(log_path / log_filename)


def setup_logger(name: str = None) -> logging.Logger:
    """
    Set up and return a configured logger.
    
    Args:
        name: Logger name. If None, returns the root logger.
        
    Returns:
        Configured logger instance.
    """
    config = load_logging_config()
    
    # Get configuration values
    log_level = getattr(logging, config.get("level", "INFO").upper())
    log_format = config.get(
        "format", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        try:
            log_file = get_log_file_path()
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create file handler: {e}")
    
    return logger


# Create default application logger
logger = setup_logger("retail_forecast")


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with the application's configuration.
    
    Args:
        name: Name for the logger (typically __name__).
        
    Returns:
        Configured logger instance.
    """
    return setup_logger(name)


# Convenience logging functions
def log_info(message: str, logger_name: str = None):
    """Log an info message."""
    log = get_logger(logger_name) if logger_name else logger
    log.info(message)


def log_error(message: str, logger_name: str = None, exc_info: bool = False):
    """Log an error message."""
    log = get_logger(logger_name) if logger_name else logger
    log.error(message, exc_info=exc_info)


def log_warning(message: str, logger_name: str = None):
    """Log a warning message."""
    log = get_logger(logger_name) if logger_name else logger
    log.warning(message)


def log_debug(message: str, logger_name: str = None):
    """Log a debug message."""
    log = get_logger(logger_name) if logger_name else logger
    log.debug(message)


def log_critical(message: str, logger_name: str = None):
    """Log a critical message."""
    log = get_logger(logger_name) if logger_name else logger
    log.critical(message)


if __name__ == "__main__":
    # Test logging
    logger.info("Logger initialized successfully")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
