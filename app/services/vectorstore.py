# services/vectorstore.py
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from .settings import EMBED_MODEL, INDEX_DIR, FAISS_PATH

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL)

def persist_faiss(vs: FAISS) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(FAISS_PATH)

def load_faiss_or_none() -> FAISS | None:
    if not os.path.isdir(INDEX_DIR):
        return None
    try:
        return FAISS.load_local(FAISS_PATH, get_embeddings(), allow_dangerous_deserialization=True)
    except Exception:
        return None
