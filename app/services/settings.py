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
INDEX_MANIFEST_PATH = INDEX_DIR / "manifest.json"
PROFILE_SNAPSHOT_PATH = APP_DIR / "data" / "sources" / "linkedin_profile.json"

# Allow override via env; otherwise auto-detect common raw-data locations.
_default_raw_candidates = [
    DATA_DIR / "raw",          # <project>/data/raw
    APP_DIR / "data" / "raw",  # <project>/app/data/raw
]
_detected_raw = next((p for p in _default_raw_candidates if p.exists()), _default_raw_candidates[0])
DATA_RAW_DIR = os.getenv("DATA_RAW_DIR", str(_detected_raw))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# Central model policy. Terra remains the safe default until the benchmark proves
# that the more efficient Luna candidate meets every promotion gate.
GENERATION_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")
GENERATION_CANDIDATE_MODEL = os.getenv(
    "OPENAI_CANDIDATE_MODEL",
    "gpt-5.6-luna",
)
GENERATION_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-5.6-terra")
QUERY_REWRITE_MODEL = os.getenv("OPENAI_REWRITE_MODEL", GENERATION_MODEL)
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", GENERATION_FALLBACK_MODEL)
OPENAI_MODEL = GENERATION_MODEL  # Backward-compatible alias.

LOCAL_CANDIDATE_K = int(os.getenv("LOCAL_CANDIDATE_K", "8"))
LOCAL_CONTEXT_K = int(os.getenv("LOCAL_CONTEXT_K", "4"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
WEB_FALLBACK_ENABLED = os.getenv("WEB_FALLBACK_ENABLED", "1").lower() in {
    "1", "true", "yes", "on"
}

LINKEDIN_PROFILE_URL = os.getenv(
    "LINKEDIN_PROFILE_URL",
    "https://www.linkedin.com/in/chivon-powers-phd-a6730610",
)
PORTFOLIO_URL = os.getenv("PORTFOLIO_URL", "https://www.chivonpowers.com/")
