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
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

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
    from app.services.vectorstore import (
        get_embeddings,
        persist_faiss,
        load_faiss_or_none,
        get_last_load_error,
    )
    from app.services.settings import (
        DATA_RAW_DIR,
        EMBED_MODEL,
        FAISS_PATH,
        INDEX_MANIFEST_PATH,
        PROFILE_SNAPSHOT_PATH,
    )
    from app.services.profile_snapshot import (
        load_profile_snapshot,
        refresh_profile_snapshot,
        snapshot_as_text,
    )
except ModuleNotFoundError:
    try:
        from services.vectorstore import (
            get_embeddings,
            persist_faiss,
            load_faiss_or_none,
            get_last_load_error,
        )
        from services.settings import (
            DATA_RAW_DIR,
            EMBED_MODEL,
            FAISS_PATH,
            INDEX_MANIFEST_PATH,
            PROFILE_SNAPSHOT_PATH,
        )
        from services.profile_snapshot import (
            load_profile_snapshot,
            refresh_profile_snapshot,
            snapshot_as_text,
        )
    except ModuleNotFoundError:
        from .vectorstore import (
            get_embeddings,
            persist_faiss,
            load_faiss_or_none,
            get_last_load_error,
        )
        from .settings import (
            DATA_RAW_DIR,
            EMBED_MODEL,
            FAISS_PATH,
            INDEX_MANIFEST_PATH,
            PROFILE_SNAPSHOT_PATH,
        )
        from .profile_snapshot import (
            load_profile_snapshot,
            refresh_profile_snapshot,
            snapshot_as_text,
        )

logger = logging.getLogger("ingest")
CHUNKING_VERSION = "structured-v1"

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", re.I)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
)
_STREET_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.' -]+\s+"
    r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|court|ct)\b"
    r"(?:[^\n|]*)",
    re.I,
)
_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")

_EMPLOYERS = (
    "Lime",
    "Intellica",
    "Butter Payments",
    "Butter",
    "Rocket Money",
    "Microsoft",
    "MileIQ",
    "Eaze",
    "Acxiom",
    "UC Davis",
    "Northwestern University",
)


def sanitize_professional_text(text: str) -> str:
    """Remove contact and street-level location details before embedding."""
    clean = _EMAIL_RE.sub("[email removed]", text or "")
    clean = _PHONE_RE.sub("[phone removed]", clean)
    clean = _STREET_RE.sub("[address removed]", clean)
    clean = _ZIP_RE.sub("[postal code removed]", clean)
    return clean


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(pdf_dir: Path) -> list[Path]:
    pdfs = sorted(p for p in Path(pdf_dir).glob("*.pdf") if p.is_file())
    profile = Path(PROFILE_SNAPSHOT_PATH)
    return pdfs + ([profile] if profile.exists() else [])


def _stable_source_id(path: Path, pdf_dir: Path) -> str:
    """Identify index inputs without encoding checkout-specific absolute paths."""
    resolved = path.resolve()
    profile = Path(PROFILE_SNAPSHOT_PATH).resolve()
    if resolved == profile:
        return f"profile/{path.name}"
    try:
        relative = resolved.relative_to(Path(pdf_dir).resolve())
    except ValueError:
        relative = Path(path.name)
    return f"pdf/{relative.as_posix()}"


def expected_manifest(pdf_dir: Path) -> dict[str, Any]:
    sources = {
        _stable_source_id(path, pdf_dir): _sha256(path)
        for path in source_files(pdf_dir)
    }
    return {
        "manifest_version": 2,
        "chunking_version": CHUNKING_VERSION,
        "embedding_model": EMBED_MODEL,
        "sources": sources,
    }


def load_manifest(path: Path = INDEX_MANIFEST_PATH) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def manifest_is_current(
    pdf_dir: Path,
    path: Path = INDEX_MANIFEST_PATH,
) -> bool:
    actual = load_manifest(path)
    expected = expected_manifest(pdf_dir)
    return all(actual.get(key) == value for key, value in expected.items())


def write_manifest(
    pdf_dir: Path,
    chunk_count: int,
    path: Path = INDEX_MANIFEST_PATH,
) -> dict[str, Any]:
    manifest = expected_manifest(pdf_dir)
    manifest.update(
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": chunk_count,
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


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
        loaded = loader.load()
        for doc in loaded:
            doc.page_content = sanitize_professional_text(doc.page_content)
            doc.metadata["source_type"] = "local_pdf"
        docs.extend(loaded)
    return docs


def _load_profile_document() -> list[Document]:
    snapshot = load_profile_snapshot()
    text = snapshot_as_text(snapshot)
    if not text:
        return []
    return [
        Document(
            page_content=sanitize_professional_text(text),
            metadata={
                "source": str(PROFILE_SNAPSHOT_PATH),
                "source_type": "public_profile",
                "url": snapshot.get("canonical_url"),
                "verified_at": snapshot.get("verified_at"),
                "section": "current_role",
                "employer": "Lime",
            },
        )
    ]


def _infer_chunk_metadata(text: str) -> dict[str, str]:
    low = (text or "").lower()
    employer = next((name for name in _EMPLOYERS if name.lower() in low), "")
    if any(term in low for term in ("experience", "data scientist", "engineer", "manager")):
        section = "experience"
    elif any(term in low for term in ("project", "pipeline", "model", "rag")):
        section = "projects"
    elif any(term in low for term in ("education", "ph.d", "phd", "publication")):
        section = "education_research"
    elif any(term in low for term in ("patent", "invent")):
        section = "patents"
    else:
        section = "professional_background"

    topics = []
    topic_terms = {
        "ai": ("llm", "langgraph", "agentic", "rag", "artificial intelligence"),
        "machine_learning": ("machine learning", "xgboost", "classifier", "model"),
        "experimentation": ("experiment", "a/b", "statistical"),
        "product": ("product", "roadmap", "customer", "prd"),
        "research": ("eeg", "neuroscience", "research", "publication"),
    }
    for topic, terms in topic_terms.items():
        if any(term in low for term in terms):
            topics.append(topic)
    return {
        "employer": employer,
        "section": section,
        "topics": ",".join(topics),
    }


def _split_and_label(docs: list) -> list:
    """
    Split around document structure before character fallbacks. Smaller, labeled
    chunks keep individual employers/projects from bleeding into one another.
    """
    profile_docs = [
        doc for doc in docs if doc.metadata.get("source_type") == "public_profile"
    ]
    pdf_docs = [
        doc for doc in docs if doc.metadata.get("source_type") != "public_profile"
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        separators=["\nEXPERIENCE\n", "\nSELECT PROJECTS\n", "\nEDUCATION\n", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(pdf_docs) + profile_docs
    for c in chunks:
        src = Path(c.metadata.get("source", "local.pdf")).name
        if c.metadata.get("source_type") == "public_profile":
            c.metadata["label"] = "public profile • LinkedIn • current employer"
        else:
            page = int(c.metadata.get("page", 0)) + 1
            c.metadata["page_number"] = page
            c.metadata["label"] = f"local • {src} p.{page}"
        inferred = _infer_chunk_metadata(c.page_content)
        for key, value in inferred.items():
            if value and not c.metadata.get(key):
                c.metadata[key] = value
        c.metadata["chunking_version"] = CHUNKING_VERSION
    return chunks


def build_index(pdf_dir: Path) -> FAISS:
    """Build FAISS index from PDFs and persist it."""
    docs = _load_pdfs(pdf_dir) + _load_profile_document()
    chunks = _split_and_label(docs)
    vs = FAISS.from_documents(chunks, get_embeddings())
    persist_faiss(vs)  # persist to configured INDEX_DIR/FAISS_PATH
    write_manifest(Path(pdf_dir), len(chunks))
    logger.info("✅ FAISS index built and saved.")
    return vs


def ensure_index(pdf_dir: Path | str | None = None) -> dict[str, Any]:
    """Load a current FAISS index or rebuild when sources/configuration changed."""
    pdf_dir = Path(pdf_dir) if pdf_dir else Path(DATA_RAW_DIR)
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set; skipping FAISS build.")
        return index_status(pdf_dir)
    store = load_faiss_or_none()
    if store is None or not manifest_is_current(pdf_dir):
        logger.info("Index missing or stale; rebuilding...")
        build_index(pdf_dir)
    else:
        logger.info("✅ FAISS index loaded.")
    return index_status(pdf_dir)


def index_status(pdf_dir: Path | str | None = None) -> dict[str, Any]:
    pdf_dir = Path(pdf_dir) if pdf_dir else Path(DATA_RAW_DIR)
    manifest = load_manifest()
    index_exists = Path(FAISS_PATH).is_dir()
    current = bool(manifest) and manifest_is_current(pdf_dir)
    provider_configured = bool(os.getenv("OPENAI_API_KEY"))
    load_error = get_last_load_error()
    if not provider_configured and not load_error:
        load_error = "embedding_provider_not_configured"
    if index_exists and current and provider_configured and not load_error:
        state = "ok"
    elif index_exists:
        state = "degraded"
    else:
        state = "not_ready"
    return {
        "status": state,
        "index_exists": index_exists,
        "index_current": current,
        "built_at": manifest.get("built_at"),
        "chunk_count": manifest.get("chunk_count"),
        "embedding_model": manifest.get("embedding_model") or EMBED_MODEL,
        "load_error": load_error,
    }


def main():
    parser = argparse.ArgumentParser(description="Build/ensure FAISS index from PDFs.")
    parser.add_argument("--dir", dest="pdf_dir", default=None, help="Directory containing PDFs")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if index exists")
    parser.add_argument(
        "--refresh-profile",
        action="store_true",
        help="Refresh the public LinkedIn snapshot before indexing",
    )
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
        if args.refresh_profile:
            refresh_profile_snapshot()
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
