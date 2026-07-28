# agent/lc_tools.py
from __future__ import annotations
import json
import os
from typing import Any
from pydantic import BaseModel, Field
from langchain.tools import tool
from tavily import TavilyClient

try:
    from app.services.web_fetch import fetch_and_clean
    from app.services.vectorstore import load_faiss_or_none
except ModuleNotFoundError:
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
# This tool emits JSON-like search results (title, url, content) by default.
# Requires TAVILY_API_KEY in the environment.
def search_web(query: str, max_results: int = 3) -> dict[str, Any]:
    """Return a stable, structured Tavily result contract."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "query": query,
            "results": [],
            "answer": None,
            "error": "missing_tavily_api_key",
        }
    try:
        client = TavilyClient(api_key=api_key)
        resp = client.search(
            query=query,
            max_results=max(1, min(max_results, 5)),
            include_answer=True,
            include_raw_content=False,
        )
        results = []
        for item in (resp or {}).get("results") or []:
            results.append(
                {
                    "title": str(item.get("title") or "Web result"),
                    "url": str(item.get("url") or ""),
                    "content": str(item.get("content") or ""),
                    "score": item.get("score"),
                    "published_date": item.get("published_date"),
                }
            )
        return {
            "query": query,
            "results": results,
            "answer": (resp or {}).get("answer"),
            "error": None,
        }
    except Exception as exc:
        return {
            "query": query,
            "results": [],
            "answer": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


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
