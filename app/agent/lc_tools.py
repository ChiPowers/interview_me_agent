# agent/lc_tools.py
from __future__ import annotations
import json
import os
from pydantic import BaseModel, Field
from langchain_core.tools import tool

try:
    from app.services.web_search import search_web
    from app.services.web_fetch import fetch_and_clean
    from app.services.vectorstore import load_faiss_or_none
except ModuleNotFoundError:
    from services.web_search import search_web
    from services.web_fetch import fetch_and_clean
    from services.vectorstore import load_faiss_or_none

from .lg_utils import multiquery_local_search


# -----------------------------
# Local Retrieval Tool (FAISS)
# -----------------------------
class RetrieveInput(BaseModel):
    query: str = Field(..., description="Natural-language query to search across local PDFs")
    k: int = Field(6, description="Number of top snippets to return")


@tool("retrieve_local", args_schema=RetrieveInput)
def retrieve_local_tool(query: str, k: int = 6) -> str:
    """
    Search locally indexed PDF chunks (FAISS) and return labeled snippets.

    Returns a plain-text block composed of the top-k results, each including:
      - a label like: "local • <file> p.<n>"
      - the snippet text

    Use this tool first. If it returns "[retrieve_local] No results" or clearly irrelevant content,
    consider a web search as fallback.
    """
    vs = load_faiss_or_none()
    if vs is None:
        return "[retrieve_local] No index loaded. Click (Re)Build Index in the app."

    # Fan-out rewrites can improve recall but adds an extra LLM call.
    rewrites = int(os.getenv("LOCAL_RETRIEVAL_REWRITES", "0"))
    mq = multiquery_local_search(
        query,
        rewrites=max(0, rewrites),
        k_per_query=max(2, k // 2),
        top_k=k,
    )

    context = mq.get("context", "[retrieve_local] No results")
    rewrites = mq.get("rewrites") or []
    if rewrites:
        header = "Rewrites: " + " | ".join(rewrites[:5])
        return header + "\n\n" + context
    return context


# -----------------------------
# Web Search (Tavily) Tool
# -----------------------------
@tool("tavily_search")
def tavily_search_tool(query: str) -> str:
    """Search the web via Tavily and return structured JSON."""
    return json.dumps(search_web(query), ensure_ascii=False)


TAVILY = tavily_search_tool


# -----------------------------
# Fetch URL Tool
# -----------------------------
class FetchInput(BaseModel):
    url: str = Field(..., description="HTTP/HTTPS URL to fetch and clean into plain text")


@tool("fetch_url", args_schema=FetchInput)
def fetch_url_tool(url: str) -> str:
    """
    Fetch a web page and return a cleaned plain-text version.

    Uses trafilatura to strip boilerplate. Returns "[fetch_url] empty" if the
    page cannot be fetched or contains no extractable text.
    """
    return fetch_and_clean(url) or "[fetch_url] empty"
