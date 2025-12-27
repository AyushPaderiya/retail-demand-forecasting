"""
Database Connection Module

Provides PostgreSQL database connection handling with connection pooling.
"""

import sys
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from src.exception import DatabaseException
from src.logger import get_logger
from src.utils import load_config

logger = get_logger(__name__)


class DatabaseConnection:
    """
    Manages PostgreSQL database connections with pooling.
    
    Features:
    - Connection pooling for performance
    - Automatic reconnection on failure
    - Context manager support
    """
    
    def __init__(self):
        """Initialize database connection pool."""
        self.logger = get_logger(self.__class__.__name__)
        self.connection_pool: Optional[pool.SimpleConnectionPool] = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Create connection pool from config."""
        try:
            config = load_config()
            db_config = config.get("database", {})
            
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=db_config.get("pool_size", 5),
                host=db_config.get("host", "localhost"),
                port=db_config.get("port", 5432),
                database=db_config.get("name", "retail_forecast"),
                user=db_config.get("user", "postgres"),
                password=db_config.get("password", "postgres"),
            )
            
            self.logger.info(f"Database connection pool created: {db_config.get('name')}")
            
        except Exception as e:
            raise DatabaseException(f"Failed to create connection pool: {e}", sys)
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool (context manager).
        
        Usage:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ...")
        """
        if self.connection_pool is None:
            self._initialize_pool()
        
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def execute_query(self, query: str, params: tuple = None, fetch_all: bool = True):
        """
        Execute a query and return results.
        
        Args:
            query: SQL query string.
            params: Query parameters (for parameterized queries).
            fetch_all: If True, return all rows. If False, return one row.
            
        Returns:
            Query results as list of dicts (or single dict if fetch_all=False).
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            
            if query.strip().upper().startswith("SELECT"):
                return cursor.fetchall() if fetch_all else cursor.fetchone()
            else:
                return cursor.rowcount
    
    def close(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("Database connection pool closed")


# Singleton instance
_db_instance: Optional[DatabaseConnection] = None


def get_db_connection() -> DatabaseConnection:
    """Get or create database connection singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
    return _db_instance


if __name__ == "__main__":
    # Test connection
    db = get_db_connection()
    
    # Test query
    result = db.execute_query("SELECT version()")
    print(f"PostgreSQL version: {result}")
    
    db.close()
