import os
import logging
import sqlite3
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

# Extract database path from URL for direct checks
db_path = Config.DATABASE_URL.replace('sqlite:///', '')
full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_path)
logger.info(f"SQLite database path: {full_path}")

# Check database directly first
try:
    # Try to connect directly with sqlite3 to verify the database file
    if os.path.exists(full_path):
        conn = sqlite3.connect(full_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info(f"Found tables in SQLite database: {tables}")
        conn.close()
    else:
        logger.warning(f"SQLite database file not found at {full_path}, will be created")
except Exception as e:
    logger.error(f"Error checking SQLite database: {str(e)}")

# Create database engine with more verbose error handling
try:
    # Adding echo=True for SQL statement logging during development
    engine = create_engine(
        Config.DATABASE_URL, 
        connect_args={"check_same_thread": False},
        echo=True  # Log all SQL statements
    )
    
    # Add event listener for connection issues
    @event.listens_for(engine, "connect")
    def on_connect(dbapi_con, con_record):
        logger.info("Successfully connected to database")
        
    @event.listens_for(engine, "checkout")
    def on_checkout(dbapi_con, con_record, con_proxy):
        try:
            # Test that the connection is active
            dbapi_con.execute("SELECT 1")
            logger.debug("Connection is valid")
        except Exception as e:
            logger.error(f"Connection is invalid: {str(e)}")
            raise
            
except Exception as e:
    logger.critical(f"Failed to create database engine: {str(e)}")
    raise

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Database dependency with better error handling
def get_db():
    db = SessionLocal()
    try:
        # Test the connection with properly declared text SQL
        db.execute(text("SELECT 1")).first()
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        db.close()