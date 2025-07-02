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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def check_db_connection() -> bool:
    """
    Check connection to the MySQL database using LangChain SQLDatabase.
    Returns True if successful, False otherwise.
    """
    db_uri = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"

    try:
        db = SQLDatabase.from_uri(db_uri)
        db.run("SELECT 1")
        logging.info("[OK] Database connection successful ✅")
        return True
    except Exception as e:
        logging.error(f"[ERROR] Failed to connect to the database: {str(e)}")
        return False


if __name__ == "__main__":

    # Check and raise error if connection fails
    logging.info(" -------- start checking database connection... --------")
    logging.info("checking database connection... 🔎")
    if not check_db_connection():
        raise Exception("Database not connected ❌")
    logging.info(" -------------------------------------------------------")
