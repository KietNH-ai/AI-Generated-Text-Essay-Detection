from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "raw_data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.csv"