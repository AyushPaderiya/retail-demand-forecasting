"""
Database Connection Module

Provides SQLAlchemy-based database connectivity with connection pooling.
"""

import os
import sys
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src.exception import DatabaseException
from src.logger import get_logger
from src.utils import load_config

logger = get_logger(__name__)


class DatabaseConnection:
    """
    Manages database connections using SQLAlchemy.
    
    Supports both URL-based and individual parameter-based connection.
    """
    
    _instance: Optional["DatabaseConnection"] = None
    _engine: Optional[Engine] = None
    _session_factory: Optional[sessionmaker] = None
    
    def __new__(cls):
        """Singleton pattern to ensure single database connection pool."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize database connection if not already done."""
        if self._engine is None:
            self._initialize_connection()
    
    def _initialize_connection(self):
        """Set up database connection from config or environment."""
        try:
            # Try environment variable first
            database_url = os.getenv("DATABASE_URL")
            
            if not database_url:
                # Build from config
                config = load_config()
                db_config = config.get("database", {})
                
                host = os.getenv("DATABASE_HOST", db_config.get("host", "localhost"))
                port = os.getenv("DATABASE_PORT", db_config.get("port", 5432))
                name = os.getenv("DATABASE_NAME", db_config.get("name", "retail_forecast"))
                user = os.getenv("DATABASE_USER", db_config.get("user", "postgres"))
                password = os.getenv("DATABASE_PASSWORD", db_config.get("password", "postgres"))
                
                database_url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
            
            # Get pool settings
            config = load_config()
            db_config = config.get("database", {})
            pool_size = db_config.get("pool_size", 5)
            max_overflow = db_config.get("max_overflow", 10)
            
            # Create engine with connection pooling
            self._engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,  # Verify connections before use
                echo=False,  # Set to True for SQL debugging
            )
            
            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
            )
            
            logger.info(f"Database connection initialized (pool_size={pool_size})")
            
        except Exception as e:
            raise DatabaseException(f"Failed to initialize database connection: {e}", sys)
    
    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        if self._engine is None:
            self._initialize_connection()
        return self._engine
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with db.get_session() as session:
                result = session.execute(query)
        
        Yields:
            SQLAlchemy Session.
        """
        if self._session_factory is None:
            self._initialize_connection()
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error, rolled back: {e}")
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: dict = None) -> list:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query string.
            params: Optional query parameters.
            
        Returns:
            List of result rows.
        """
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                return [dict(row._mapping) for row in result]
        except Exception as e:
            raise DatabaseException(f"Query execution failed: {e}", sys)
    
    def execute_script(self, script_path: str) -> None:
        """
        Execute a SQL script file.
        
        Args:
            script_path: Path to SQL script file.
        """
        try:
            with open(script_path, "r") as f:
                script = f.read()
            
            with self.get_session() as session:
                # Split by semicolons and execute each statement
                statements = [s.strip() for s in script.split(";") if s.strip()]
                for statement in statements:
                    session.execute(text(statement))
            
            logger.info(f"Executed SQL script: {script_path}")
            
        except Exception as e:
            raise DatabaseException(f"Failed to execute script {script_path}: {e}", sys)
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection is successful.
        """
        try:
            with self.get_session() as session:
                result = session.execute(text("SELECT 1"))
                _ = result.fetchone()
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def close(self):
        """Close the database connection pool."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connection closed")


# Singleton instance
db = DatabaseConnection()


def get_db() -> DatabaseConnection:
    """Get the database connection instance."""
    return db


if __name__ == "__main__":
    # Test connection
    connection = DatabaseConnection()
    if connection.test_connection():
        print("Database connection successful!")
    else:
        print("Database connection failed!")
