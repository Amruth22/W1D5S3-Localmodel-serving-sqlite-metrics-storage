import os
import logging
import sqlite3
from sqlalchemy import inspect, text

from database import engine, full_path
from models import Base, RequestLog, ResponseLog

logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database and create tables if they don't exist."""
    try:
        # Create tables using SQLAlchemy
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        logger.info(f"Existing tables: {existing_tables}")
        
        if not existing_tables or 'request_logs' not in existing_tables or 'response_logs' not in existing_tables:
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully.")
        else:
            logger.info("Database tables already exist.")
            
            # Verify columns in request_logs table
            columns = inspector.get_columns('request_logs')
            column_names = [column['name'] for column in columns]
            logger.info(f"Request_logs columns: {column_names}")
            
            # Verify columns in response_logs table
            columns = inspector.get_columns('response_logs')
            column_names = [column['name'] for column in columns]
            logger.info(f"Response_logs columns: {column_names}")
            
            # If model_used column is missing from response_logs, add it
            if 'model_used' not in column_names:
                try:
                    # Need to use raw SQL for this since SQLAlchemy doesn't easily add columns
                    with sqlite3.connect(full_path) as conn:
                        conn.execute("ALTER TABLE response_logs ADD COLUMN model_used VARCHAR;")
                    logger.info("Added missing model_used column to response_logs table.")
                except Exception as e:
                    logger.error(f"Failed to add model_used column: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    init_db()