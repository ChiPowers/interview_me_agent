# agent/lc_tools.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from app.services.vectorstore import load_faiss_or_none
from app.services.web_fetch import fetch_and_clean


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
    vs: Optional[FAISS] = load_faiss_or_none()
    if vs is None:
        return "[retrieve_local] No index loaded. Click (Re)Build Index in the app."
    try:
        docs = vs.similarity_search(query, k=k)
    except Exception as e:
        return f"[retrieve_local] error: {e}"
    blocks: List[str] = []
    for d in docs:
        label = d.metadata.get("label") or d.metadata.get("source", "local.pdf")
        blocks.append(f"{label}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks) if blocks else "[retrieve_local] No results"


# -----------------------------
# Web Search (Tavily) Tool
# -----------------------------
# This tool emits JSON-like search results (title, url, content) by default.
# Requires TAVILY_API_KEY in the environment.
TAVILY = TavilySearch(
    max_results=3,
    include_answer=True, 
    include_raw_content=False
)


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
