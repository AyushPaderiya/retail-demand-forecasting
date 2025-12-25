"""
Custom Exception Classes

Provides structured exception handling for the entire application.
Inspired by: https://github.com/krishnaik06/mlproject
"""

import sys
from typing import Optional

from src.logger import logger


def get_error_details(error: Exception, error_detail: sys) -> str:
    """
    Extract detailed error information including file name and line number.
    
    Args:
        error: The exception that was raised.
        error_detail: sys module for accessing exception info.
        
    Returns:
        Formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()
    
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return (
            f"Error occurred in script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"error message: [{str(error)}]"
        )
    return f"Error message: [{str(error)}]"


class CustomException(Exception):
    """
    Base custom exception class for the application.
    
    Provides detailed error messages including file and line number information.
    """
    
    def __init__(self, error_message: str, error_detail: sys = None):
        """
        Initialize the custom exception.
        
        Args:
            error_message: The error message.
            error_detail: sys module for accessing exception info.
        """
        super().__init__(error_message)
        
        if error_detail:
            self.error_message = get_error_details(
                Exception(error_message), 
                error_detail
            )
        else:
            self.error_message = error_message
            
        # Log the error
        logger.error(self.error_message)
    
    def __str__(self) -> str:
        return self.error_message


class DataGenerationException(CustomException):
    """Exception raised during data generation process."""
    
    def __init__(self, message: str, error_detail: sys = None):
        super().__init__(f"Data Generation Error: {message}", error_detail)


class DatabaseException(CustomException):
    """Exception raised during database operations."""
    
    def __init__(self, message: str, error_detail: sys = None):
        super().__init__(f"Database Error: {message}", error_detail)


class DataIngestionException(CustomException):
    """Exception raised during data ingestion."""
    
    def __init__(self, message: str, error_detail: sys = None):
        super().__init__(f"Data Ingestion Error: {message}", error_detail)


class DataTransformationException(CustomException):
    """Exception raised during data transformation."""
    
    def __init__(self, message: str, error_detail: sys = None):
        super().__init__(f"Data Transformation Error: {message}", error_detail)


class FeatureEngineeringException(CustomException):
    """Exception raised during feature engineering."""
    
    def __init__(self, message: str, error_detail: sys = None):
        super().__init__(f"Feature Engineering Error: {message}", error_detail)


class ModelTrainingException(CustomException):
    """Exception raised during model training."""
    
    def __init__(self, message: str, error_detail: sys = None):
        super().__init__(f"Model Training Error: {message}", error_detail)


class ModelEvaluationException(CustomException):
    """Exception raised during model evaluation."""
    
    def __init__(self, message: str, error_detail: sys = None):
        super().__init__(f"Model Evaluation Error: {message}", error_detail)


class PredictionException(CustomException):
    """Exception raised during prediction."""
    
    def __init__(self, message: str, error_detail: sys = None):
        super().__init__(f"Prediction Error: {message}", error_detail)


class ConfigurationException(CustomException):
    """Exception raised for configuration errors."""
    
    def __init__(self, message: str, error_detail: sys = None):
        super().__init__(f"Configuration Error: {message}", error_detail)


class ValidationException(CustomException):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, error_detail: sys = None):
        super().__init__(f"Validation Error: {message}", error_detail)


class APIException(CustomException):
    """Exception raised for API errors."""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 500, 
        error_detail: sys = None
    ):
        self.status_code = status_code
        super().__init__(f"API Error ({status_code}): {message}", error_detail)


if __name__ == "__main__":
    # Test exceptions
    try:
        raise DataGenerationException("Test error", sys)
    except CustomException as e:
        print(f"Caught exception: {e}")
