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

# Allow override via env; default to <project>/data/raw
DATA_RAW_DIR = os.path.join(APP_DIR, "data", "raw")
# DATA_RAW_DIR = os.path.abspath(DATA_RAW_DIR)
EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-3-large")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")