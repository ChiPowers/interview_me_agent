import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SETTINGS_FILE = Path(__file__).resolve()
APP_DIR       = SETTINGS_FILE.parents[1]
PROJECT_ROOT  = APP_DIR.parent

DATA_DIR      = PROJECT_ROOT / "data"
INDEX_DIR     = DATA_DIR / "index"
FAISS_PATH    = INDEX_DIR / "faiss"

# Allow override via env; otherwise auto-detect common PDF locations.
_default_raw_candidates = [
    DATA_DIR / "raw",          # <project>/data/raw
    APP_DIR / "data" / "raw",  # <project>/app/data/raw
]
_detected_raw = next((p for p in _default_raw_candidates if p.exists()), _default_raw_candidates[0])
DATA_RAW_DIR = os.getenv("DATA_RAW_DIR", str(_detected_raw))
EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-3-large")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")
