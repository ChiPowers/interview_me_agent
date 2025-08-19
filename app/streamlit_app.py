# app/streamlit_app.py
import os
import time
import base64
from io import BytesIO
from urllib.parse import quote

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from services.ingest_index import ensure_index  # auto-build/load on boot
from agent.lc_controller import LCController

import logging, json
logging.basicConfig(level=os.getenv("APP_LOG_LEVEL", "INFO"))
logger = logging.getLogger("interview_agent")


# ---------- env & page ----------
load_dotenv()
st.set_page_config(
    page_title="Interview Chivon Powers",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- assets (robust for local & Streamlit Cloud) ----------
from pathlib import Path

def _first_existing_dir(candidates):
    for p in candidates:
        if p and p.exists() and p.is_dir():
            return p
    return None

APP_FILE = Path(__file__).resolve()
APP_DIR = APP_FILE.parent                        # repo/app
REPO_ROOT = APP_DIR.parent                       # repo/
CWD = Path.cwd()

# Allow override via env: STATIC_DIR=/mount/static
env_static = Path(os.getenv("STATIC_DIR", "")).resolve() if os.getenv("STATIC_DIR") else None

STATIC_DIR = _first_existing_dir([
    env_static,
    REPO_ROOT / "static",     # repo/static
    APP_DIR / "static",       # repo/app/static
    CWD / "static",           # cwd/static
])

if not STATIC_DIR:
    # Log once; UI will show fallback text
    logger.warning("No static/ directory found (checked env, repo/static, app/static, cwd/static).")

def _read_image_b64(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            import base64
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning("Image missing/unreadable: %s (%s)", path, e)
        return ""

def _path_if_exists(dirpath: Path, name: str) -> Path | None:
    if not dirpath:
        return None
    p = dirpath / name
    return p if p.exists() else None

# Try to find each image under the chosen STATIC_DIR
LOGO_FILE     = _path_if_exists(STATIC_DIR, "logotat.png")
HEADSHOT_FILE = _path_if_exists(STATIC_DIR, "headshot.png")
CHATSHOT_FILE = _path_if_exists(STATIC_DIR, "cp_face.png")

logo_b64     = _read_image_b64(LOGO_FILE)     if LOGO_FILE     else ""
headshot_b64 = _read_image_b64(HEADSHOT_FILE) if HEADSHOT_FILE else ""
chatshot_b64 = _read_image_b64(CHATSHOT_FILE) if CHATSHOT_FILE else ""

# Debug logging so you can see what Cloud/local picked
logger.info("STATIC_DIR: %s", str(STATIC_DIR) if STATIC_DIR else "None")
logger.info("Assets found -> logo:%s headshot:%s chatshot:%s",
            bool(logo_b64), bool(headshot_b64), bool(chatshot_b64))

# ---------- styles ----------
st.markdown(
    """
    <style>
    .stTextInput>div>div>input {
        font-size: 18px;
        border-radius: 8px;
        padding: 8px;
        border: 2px solid #4B0082;
    }
    .stButton>button {
        background-color: #4B0082;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 25px;
    }
    .stButton>button:hover {
        background-color: #6A5ACD;
        cursor: pointer;
    }
    .header-container {
        text-align: center;
        color: #4B0082;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 18px;
        color: #555555;
        margin-top: 5px;
    }
    .answer-container {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-top: 1rem;
    }
    .answer-headshot {
        width: 80px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .answer-text {
        flex-grow: 1;
        font-size: 18px;
        color: #333333;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- adaptive text color (uses Streamlit theme variables) ---
st.markdown("""
<style>
/* Let Streamlit's theme decide the correct foreground color */
.answer-text {
    color: var(--text-color) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- sidebar ----------
with st.sidebar:
    if headshot_b64:
        st.image(f"data:image/png;base64,{headshot_b64}", width=160, caption="Chivon Powers, PhD")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/chivon-powers-phd-a6730610/) · [GitHub](https://github.com/Chipowers/)")
    else:
        st.info("Headshot image not found in /static (see logs).")

# ---------- ensure index ready once ----------
@st.cache_resource(show_spinner="Preparing search index…")
def _ensure_index_ready():
    ensure_index()
    return True
_ensure_index_ready()

# ---------- controller & state ----------
if "controller" not in st.session_state:
    st.session_state.controller = LCController()

# ---------- header ----------
st.markdown(
    f"""
    <div class="header-container">
        {f'<img src="data:image/png;base64,{logo_b64}" width="150">' if logo_b64 else ""}
        <h1 style="margin-bottom: 0;">Interview Chivon Powers</h1>
        <p class="subtitle">
            This bot responds as me, using my resume and other documents to answer your interview questions.<br>
            It’s a practical demonstration of my AI product development skills and a fun way to learn about my work experience.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- footnotes renderer ----------
def render_footnotes_md(footnotes) -> str:
    """
    Accepts:
      - dict: {1:{title, url|path|href}, ...}
      - list: [{title, url|path|href}, ...]
      - str:  returned as-is
    Returns Markdown lines: [1] Title (link)
    """
    if not footnotes:
        return ""
    if isinstance(footnotes, str):
        return footnotes

    def _href(meta: dict) -> str:
        h = meta.get("url") or meta.get("path") or meta.get("href") or ""
        if not h:
            return ""
        # URL-encode local paths (spaces, etc.) so Markdown doesn’t break
        if h.startswith("local://"):
            return "local://" + quote(h[len("local://"):])
        return h

    lines = []
    if isinstance(footnotes, dict):
        keys = sorted(footnotes, key=lambda x: int(x) if str(x).isdigit() else str(x))
        for k in keys:
            meta = footnotes.get(k) or {}
            title = meta.get("title", f"Source {k}")
            h = _href(meta)
            lines.append(f"[{k}] [{title}]({h})" if h else f"[{k}] {title}")
    elif isinstance(footnotes, list):
        for i, meta in enumerate(footnotes, start=1):
            meta = meta or {}
            title = meta.get("title", f"Source {i}")
            h = _href(meta)
            lines.append(f"[{i}] [{title}]({h})" if h else f"[{i}] {title}")
    else:
        return str(footnotes)

    return "  \n".join(lines)

# ---------- form ----------
with st.form(key="qa_form", clear_on_submit=True):
    question = st.text_input(
        label="Enter your interview question:",
        placeholder="e.g., Tell me about a time you solved a tough problem",
        key="user_question",
    )
    submitted = st.form_submit_button("Ask the Question")

st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

# ---------- handle submit ----------
if submitted:
    if not question or not question.strip():
        st.warning("Please enter a question.")
    else:
        placeholder = st.empty()

        # run agent
        try:
            result = st.session_state.controller.respond(question.strip())
        except Exception as e:
            st.error(f"Error processing your question: {e}")
            raise

        answer = (result.get("answer") or "").strip()
        footnotes = result.get("footnotes") or {}
        trace = result.get("trace") or {}

        # log to console
        logger.info("Q: %s", question)
        logger.info("A: %s", answer)
        logger.debug("TRACE:\n%s", json.dumps(trace, indent=2, ensure_ascii=False))
        logger.info("Footnotes payload type: %s", type(footnotes).__name__)
        logger.info("Footnotes payload value: %r", footnotes)
        # keep refs for evaluation
        logger.info("FOOTNOTES (eval only): %s", json.dumps(footnotes, ensure_ascii=False))
        
        import re
        visible_answer = re.sub(r"\s*\[\d+\]", "", answer).strip()

        # --- chunked streaming by ~5 words (fast + smooth) ---
        words = visible_answer.split()
        chunk_size = 5
        accum_words = []
        for i in range(0, len(words), chunk_size):
            accum_words.extend(words[i:i+chunk_size])
            html = f"""
                <div class="answer-container">
                    {f'<img src="data:image/png;base64,{chatshot_b64}" class="answer-headshot">' if chatshot_b64 else ""}
                    <div class="answer-text">{' '.join(accum_words)}</div>
                </div>
            """
            placeholder.markdown(html, unsafe_allow_html=True)
            time.sleep(0.02)
