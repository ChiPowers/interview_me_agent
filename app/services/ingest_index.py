# app/services/ingest_index.py
"""
Build/ensure a FAISS index from PDFs.

Usage:
  # from project root
  python -m app.services.ingest_index
  # or as a direct script
  python app/services/ingest_index.py

Options:
  --dir /path/to/pdfs     (override PDF directory; default comes from settings or data/raw)
  --rebuild               (force rebuild even if an index exists)
  --quiet                 (reduce logging noise)
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
import argparse
from pathlib import Path
from .settings import DATA_RAW_DIR  # or .settings if you use relative

# --------- PATH PATCH (so direct script & module both work) ----------
THIS_FILE   = Path(__file__).resolve()                 # .../app/services/ingest_index.py
APP_DIR     = THIS_FILE.parents[1]                     # .../app
PROJECT_ROOT= APP_DIR.parent                           # .../
for p in (str(APP_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
# --------------------------------------------------------------------

# Try imports in a few layouts to be resilient
try:
    from app.services.vectorstore import get_embeddings, persist_faiss, load_faiss_or_none
    from app.services.settings import DATA_RAW_DIR
except ModuleNotFoundError:
    try:
        from services.vectorstore import get_embeddings, persist_faiss, load_faiss_or_none
        from services.settings import DATA_RAW_DIR
    except ModuleNotFoundError:
        # final fallback to relative imports
        from .vectorstore import get_embeddings, persist_faiss, load_faiss_or_none
        from .settings import DATA_RAW_DIR

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


logger = logging.getLogger("ingest")


def _load_pdfs(pdf_dir: Path) -> list:
    pdf_dir = Path(pdf_dir).expanduser().resolve()
    logger.info("PDF directory: %s", pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"PDF dir not found: {pdf_dir}\n"
            "Create it and add your PDFs (resume, CV, personal statement, projects, papers)."
        )
    pdfs = sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {pdf_dir}")
    docs = []
    for p in pdfs:
        loader = PyPDFLoader(str(p))
        docs.extend(loader.load())
    return docs


def _split_and_label(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    # Add friendly labels (used in citations)
    for c in chunks:
        src = Path(c.metadata.get("source", "local.pdf")).name
        page = int(c.metadata.get("page", 0)) + 1
        c.metadata["label"] = f"local • {src} p.{page}"
    return chunks


def build_index(pdf_dir: Path) -> FAISS:
    """Build FAISS index from PDFs and persist it."""
    docs = _load_pdfs(pdf_dir)
    chunks = _split_and_label(docs)
    vs = FAISS.from_documents(chunks, get_embeddings())
    persist_faiss(vs)  # persist to configured INDEX_DIR/FAISS_PATH
    logger.info("✅ FAISS index built and saved.")
    return vs


def ensure_index(pdf_dir: Path | str | None = None) -> None:
    """Load FAISS if present; otherwise build from PDFs once."""
    pdf_dir = Path(pdf_dir) if pdf_dir else Path(DATA_RAW_DIR)
    if load_faiss_or_none() is None:
        logger.info("No FAISS index found, building new one...")
        build_index(pdf_dir)
    else:
        logger.info("✅ FAISS index loaded.")


def main():
    parser = argparse.ArgumentParser(description="Build/ensure FAISS index from PDFs.")
    parser.add_argument("--dir", dest="pdf_dir", default=None, help="Directory containing PDFs")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if index exists")
    parser.add_argument("--quiet", action="store_true",
                    help="Reduce logging (INFO→WARNING)")

    # argparse doesn't have true_false; emulate:
    args = parser.parse_args()
    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)


    # Resolve PDF directory
    default_dir = Path(os.getenv("DATA_RAW_DIR", str(DATA_RAW_DIR)))
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else default_dir

    try:
        if args.rebuild:
            logger.info("Forcing rebuild...")
            build_index(pdf_dir)
        else:
            ensure_index(pdf_dir)
    except Exception as e:
        logger.exception("❌ Ingestion failed: %s", e)
        sys.exit(1)

    print("Done ensuring FAISS index.")


if __name__ == "__main__":
    main()
