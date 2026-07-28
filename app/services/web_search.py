from __future__ import annotations

import os
from typing import Any

from tavily import TavilyClient


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
        response = client.search(
            query=query,
            max_results=max(1, min(max_results, 5)),
            include_answer=True,
            include_raw_content=False,
        )
        results = [
            {
                "title": str(item.get("title") or "Web result"),
                "url": str(item.get("url") or ""),
                "content": str(item.get("content") or ""),
                "score": item.get("score"),
                "published_date": item.get("published_date"),
            }
            for item in (response or {}).get("results") or []
        ]
        return {
            "query": query,
            "results": results,
            "answer": (response or {}).get("answer"),
            "error": None,
        }
    except Exception as exc:
        return {
            "query": query,
            "results": [],
            "answer": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
