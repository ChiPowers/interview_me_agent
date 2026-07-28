# services/vectorstore.py
import logging
import os
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from .settings import EMBED_MODEL, INDEX_DIR, FAISS_PATH

logger = logging.getLogger("vectorstore")
_last_load_error: str | None = None


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL)


def persist_faiss(vs: FAISS) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(FAISS_PATH)
    invalidate_faiss_cache()


@lru_cache(maxsize=1)
def _load_faiss_cached() -> FAISS:
    return FAISS.load_local(
        FAISS_PATH,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def invalidate_faiss_cache() -> None:
    _load_faiss_cached.cache_clear()


def get_last_load_error() -> str | None:
    return _last_load_error


def load_faiss_or_none(*, raise_on_error: bool = False) -> FAISS | None:
    global _last_load_error
    if not os.path.isdir(FAISS_PATH):
        _last_load_error = f"FAISS directory not found: {FAISS_PATH}"
        return None
    if not os.getenv("OPENAI_API_KEY"):
        _last_load_error = "embedding_provider_not_configured"
        if raise_on_error:
            raise RuntimeError(_last_load_error)
        return None
    try:
        store = _load_faiss_cached()
        _last_load_error = None
        return store
    except Exception as exc:
        _last_load_error = f"{type(exc).__name__}: {exc}"
        logger.error("Could not load FAISS index: %s", _last_load_error)
        if raise_on_error:
            raise
        return None
