"""
LangGraph node implementations shared by both the upcoming LGController and legacy code.

These nodes are deliberately thin orchestrators that mutate AgentState; the heavy lifting
(local retrieval, footnote extraction, composition rules) lives in lg_utils so the
LangChain-based controller can reuse the same primitives during the migration.
"""
from __future__ import annotations

from typing import Dict, Any
import time

from .lg_state import AgentState, ToolEvent
from .lg_utils import (
    local_context_from_faiss,
    analyze_local_context,
    should_use_web_judge,
    compose_answer_with_policy,
    footnotes_from_events,
)
from .lc_tools import retrieve_local_tool, TAVILY, fetch_url_tool


def prepare_local_context(state: AgentState) -> AgentState:
    """Populate the FAISS-derived local context."""
    question = state.get("question") or state.get("input") or ""
    ctx = local_context_from_faiss(question, k=6)
    out = dict(state)
    out["local_context"] = ctx
    out["input"] = question  # keep alias in sync
    out["question"] = question
    return out


def decide_retrieval_strategy(state: AgentState) -> AgentState:
    """
    Decide whether to stay local-only or open the web.
    Heuristic first; falls back to a tiny judge when confidence is low.
    """
    question = state.get("question") or state.get("input") or ""
    local_ctx = state.get("local_context", "") or ""
    verdict = analyze_local_context(question, local_ctx)

    needs_web = verdict["tentative_use_web"]
    judge_reason = None

    if verdict["confidence"] == "low":
        judge = should_use_web_judge(question, local_ctx)
        needs_web = judge["use_web"]
        judge_reason = judge["reason"]

    out = dict(state)
    out["needs_web"] = needs_web
    out["routing"] = {
        **verdict,
        "final_use_web": needs_web,
        "judge_reason": judge_reason,
    }
    return out


def route_after_decision(state: AgentState) -> str:
    """Route to local retrieval or directly to answer composition."""
    if state.get("needs_web"):
        return "needs_web"
    if state.get("local_context") and not state["local_context"].startswith("[local] No index"):
        return "local_only"
    return "skip_retrieval"


def retrieve_local_pass(state: AgentState) -> AgentState:
    """Call the FAISS-backed retrieval tool and record its output."""
    question = state.get("question") or state.get("input") or ""
    k = 6
    try:
        result = retrieve_local_tool.invoke({"query": question, "k": k})
    except Exception as exc:  # pragma: no cover - defensive guard
        result = f"[retrieve_local] error: {exc}"
        error = str(exc)
    else:
        error = None

    event: ToolEvent = {
        "tool": "retrieve_local",
        "input": {"query": question, "k": k},
        "observation": result,
        "error": error,
    }
    out = dict(state)
    events = list(out.get("tool_events", []))
    events.append(event)
    out["tool_events"] = events
    out["local_retrieval"] = result
    return out


def route_after_local_pass(state: AgentState) -> str:
    """Decide if we should branch into web search after the local sweep."""
    return "web" if state.get("needs_web") else "no_web"


def maybe_web_search_pass(state: AgentState) -> AgentState:
    """
    Fire Tavily and optionally fetch the top URL for richer summaries.
    This is intentionally lightweight; richer branching will follow in Phase 3.
    """
    question = state.get("question") or state.get("input") or ""
    tavily_input = {"query": question, "max_results": 3}
    try:
        tavily_result = TAVILY.invoke(tavily_input)
    except Exception as exc:  # pragma: no cover - defensive guard
        tavily_result = {"error": str(exc)}

    events = list(state.get("tool_events", []))
    events.append({
        "tool": "tavily_search",
        "input": tavily_input,
        "observation": tavily_result,
        "error": tavily_result["error"] if isinstance(tavily_result, dict) and "error" in tavily_result else None,
    })

    # Optional: fetch first URL if present
    fetch_event: ToolEvent | None = None
    if isinstance(tavily_result, dict):
        hits = tavily_result.get("results") or []
        if hits:
            url = hits[0].get("url")
            if url:
                try:
                    content = fetch_url_tool.invoke({"url": url})
                except Exception as exc:  # pragma: no cover
                    content = f"[fetch_url] error: {exc}"
                    error = str(exc)
                else:
                    error = None
                fetch_event = {
                    "tool": "fetch_url",
                    "input": {"url": url},
                    "observation": content,
                    "error": error,
                }
    if fetch_event:
        events.append(fetch_event)

    out = dict(state)
    out["tool_events"] = events
    out["web_results"] = tavily_result
    if fetch_event:
        out["web_page"] = fetch_event["observation"]
    return out


def compose_answer_pass(state: AgentState) -> AgentState:
    """Compose the final short answer and attach footnotes."""
    question = state.get("question") or state.get("input") or ""
    answer = compose_answer_with_policy(
        question=question,
        local_context=state.get("local_context", ""),
        local_chunks=state.get("local_retrieval", ""),
        web_results=state.get("web_results"),
        web_page=state.get("web_page", ""),
    )
    events = state.get("tool_events", [])
    footnotes = footnotes_from_events(events)
    out = dict(state)
    out["answer"] = answer
    out["output"] = answer
    out["footnotes"] = footnotes
    return out


def finalize_pass(state: AgentState) -> AgentState:
    """Attach timestamps/metadata to the trace; answer is already in the state."""
    events = state.get("tool_events", [])
    trace: Dict[str, Any] = {
        "plan": "LangGraph RAG (local-first with web fallback)",
        "routing": state.get("routing"),
        "tool_events": events,
        "local_context_preview": (state.get("local_context") or "")[:800],
        "timestamp": time.time(),
    }

    out = dict(state)
    out["trace"] = trace
    out.setdefault("answer", state.get("output", ""))
    out.setdefault("footnotes", footnotes_from_events(events))
    return out
