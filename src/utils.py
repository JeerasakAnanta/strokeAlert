import os
import logging
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase

# Load environment variables from .env file
load_dotenv(".env")

# Constants for MySQL Database
mysql_user = os.getenv("MYSQL_USER")
mysql_password = os.getenv("MYSQL_PASSWORD")
mysql_host = os.getenv("MYSQL_HOST")
mysql_port = os.getenv("MYSQL_PORT")
mysql_db = os.getenv("MYSQL_DATABASE")


def check_db_connection() -> bool:
    """
    Check connection to the MySQL database using LangChain SQLDatabase.
    Returns True if successful, False otherwise.
    """
    db_uri = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"
    logging.info(" -------- start checking database connection... --------")
    logging.info("checking database connection... 🔎")

    try:
        db = SQLDatabase.from_uri(db_uri)
        db.run("SELECT 1")
        logging.info("[OK] Database connection successful ✅")
        return True
    except Exception as e:
        logging.error(f"[ERROR] Failed to connect to the database: {str(e)}")
        raise Exception("")
        logging.info("Database not connected ❌")
        return False
