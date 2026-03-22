from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")

def validate_db_env():
    missing = [
        k for k, v in {
            "DB_SERVER": DB_SERVER,
            "DB_NAME": DB_NAME,
            "DB_USER": DB_USER,
            "DB_PASSWORD": DB_PASSWORD,
        }.items() if not v
    ]
    if missing:
        raise RuntimeError(f"Missing DB env vars: {missing}")