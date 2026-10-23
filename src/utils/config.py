from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = ROOT_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"

MODEL_DIR = OUTPUT_DIR / "models"

RANDOM_SEED = 42

USABLE_DATASET_COLUMNS = {'estimated_owners': str, 'negative': int, 'positive': int, 'release_date': str, 'supported_languages': str, 'genres': str, 'categories': str, 'tags': str, 'price': float}

GAME_PRICE_BIN = [0.0, 1.0, 10.0, 30.0, 49.99]
GAME_PRICE_LABELS = ['Free To Play', 'Micro Indie', 'Standard Tier', 'Upper Tier']