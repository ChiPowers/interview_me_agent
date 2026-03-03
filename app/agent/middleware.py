"""
LangChain v1 middleware for the Interview Agent.

Replaces the hand-rolled LangGraph nodes (lg_nodes.py) with composable
middleware hooks that plug into ``create_agent``.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ToolCallRequest,
    wrap_model_call,
    after_model,
)

from langchain_core.messages import ToolMessage

from .lg_state import InterviewState
from .lg_utils import (
    local_context_from_faiss,
    analyze_local_context,
    should_use_web_judge,
    guard_answer_with_evidence,
    footnotes_from_events,
)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")
ENABLE_WEB_JUDGE = os.getenv("ENABLE_WEB_JUDGE", "0").lower() in ("1", "true", "yes", "on")
ENABLE_LLM_HALLUCINATION_GUARD = os.getenv("ENABLE_HALLUCINATION_GUARD", "1").lower() in ("1", "true", "yes", "on")
MIN_CONTEXT_CHARS_FOR_GROUNDING = int(os.getenv("MIN_CONTEXT_CHARS_FOR_GROUNDING", "300"))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]{4,}", (text or "").lower()))


def _fast_grounding_risk_flags(answer: str, local_ctx: str, tool_event_count: int) -> List[str]:
    flags: List[str] = []
    a = (answer or "").strip()
    c = (local_ctx or "").strip()

    if not a:
        return ["empty_answer"]
    if c.startswith("[local] No index loaded.") or c.startswith("[local] error:"):
        flags.append("no_local_evidence")
    if len(c) < MIN_CONTEXT_CHARS_FOR_GROUNDING:
        flags.append("thin_local_context")
    if tool_event_count == 0:
        flags.append("no_tool_evidence")
    if not re.search(r"\[\d+\]", a):
        flags.append("missing_citations")

    answer_tokens = _tokenize(a)
    ctx_tokens = _tokenize(c)
    if answer_tokens and ctx_tokens:
        overlap = len(answer_tokens & ctx_tokens) / max(1, len(answer_tokens))
        if overlap < 0.12:
            flags.append("low_context_overlap")

    return flags


# ---------------------------------------------------------------------------
# 1. Local-context pre-fetch & routing middleware (class-based)
# ---------------------------------------------------------------------------

class RAGRoutingMiddleware(AgentMiddleware):
    """
    Before every model call:
      - Pre-fetch FAISS context for the latest user question.
      - Run the routing heuristic (+ optional LLM judge).
      - Conditionally hide/expose the web-search tool.

    This replaces the old ``prepare_local_context``, ``decide_retrieval_strategy``,
    and the conditional edges in ``lg_graph.py``.
    """

    state_schema = InterviewState

    def wrap_model_call(self, request: ModelRequest, handler):
        messages = request.state.get("messages", [])
        if not messages:
            return handler(request)

        # Extract latest user question
        question = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                question = msg.get("content", "")
                break
        if not question:
            return handler(request)

        # Fetch local context from FAISS
        local_ctx = local_context_from_faiss(question, k=6)

        # Run routing heuristic
        verdict = analyze_local_context(question, local_ctx)
        needs_web = verdict["tentative_use_web"]

        if verdict["confidence"] == "low" and ENABLE_WEB_JUDGE:
            judge = should_use_web_judge(question, local_ctx)
            needs_web = judge["use_web"]

        # Store routing info and context in state for downstream use
        state_updates = {
            "local_context": local_ctx,
            "needs_web": needs_web,
            "routing": {**verdict, "final_use_web": needs_web},
        }
        request.state.update(state_updates)

        # Filter tools: hide tavily if web not needed
        if not needs_web:
            filtered_tools = [
                t for t in request.tools
                if getattr(t, "name", "") != "tavily_search"
            ]
            request = request.override(tools=filtered_tools)

        return handler(request)


# ---------------------------------------------------------------------------
# 2. Tool error handling middleware (decorator-based)
# ---------------------------------------------------------------------------

@wrap_model_call
def tool_error_middleware(request: ModelRequest, handler):
    """Pass-through; errors are handled per-tool via wrap_tool_call below."""
    return handler(request)


# ---------------------------------------------------------------------------
# 3. Hallucination guard middleware (after model responds)
# ---------------------------------------------------------------------------

class HallucinationGuardMiddleware(AgentMiddleware):
    """
    After the model produces a final text response (no tool calls),
    run the evidence-based hallucination guard and replace the answer
    if unsupported claims are detected.

    Replaces the old ``guard_answer_with_evidence`` call inside
    ``compose_answer_pass``.
    """

    state_schema = InterviewState

    def after_model(self, state, runtime):
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        # Only guard final text answers (not tool calls)
        has_tool_calls = (
            hasattr(last_msg, "tool_calls") and last_msg.tool_calls
        )
        if has_tool_calls:
            return None

        answer = getattr(last_msg, "content", "") or ""
        if not answer.strip():
            return None

        local_ctx = state.get("local_context", "")
        if not local_ctx:
            return None

        # Find the question
        question = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content
            elif isinstance(msg, dict) and msg.get("role") == "user":
                question = msg.get("content", "")

        tool_event_count = 0
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_event_count += 1

        risk_flags = _fast_grounding_risk_flags(answer, local_ctx, tool_event_count)
        if "no_local_evidence" in risk_flags:
            last_msg.content = (
                "I don't have enough grounded local evidence to answer this accurately yet. "
                "Please re-run indexing or share the relevant document section."
            )
            return None

        if risk_flags and ENABLE_LLM_HALLUCINATION_GUARD:
            guarded = guard_answer_with_evidence(question, answer, local_ctx)
            if guarded != answer:
                last_msg.content = guarded

        return None


# ---------------------------------------------------------------------------
# 4. Trace & footnotes middleware (after model responds)
# ---------------------------------------------------------------------------

class TraceMiddleware(AgentMiddleware):
    """
    After a final answer, build the trace and footnotes payload and
    store them in state for the controller to extract.
    """

    state_schema = InterviewState

    def after_model(self, state, runtime):
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        has_tool_calls = (
            hasattr(last_msg, "tool_calls") and last_msg.tool_calls
        )
        if has_tool_calls:
            return None

        # Build tool_events from message history
        tool_events: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_events.append({
                    "tool": getattr(msg, "name", "tool"),
                    "input": {},
                    "observation": msg.content or "",
                })

        footnotes = footnotes_from_events(tool_events)

        trace = {
            "plan": "create_agent RAG (local-first with web fallback)",
            "routing": state.get("routing"),
            "tool_events": tool_events,
            "local_context_preview": (state.get("local_context") or "")[:800],
            "timestamp": time.time(),
            "controller": "create_agent_v1",
        }

        state["tool_events"] = tool_events
        state["footnotes"] = footnotes
        state["trace"] = trace

        return None


# ---------------------------------------------------------------------------
# Middleware list for create_agent (order matters)
# ---------------------------------------------------------------------------

def get_middleware() -> list:
    """Return the ordered list of middleware for the interview agent."""
    return [
        RAGRoutingMiddleware(),
        HallucinationGuardMiddleware(),
        TraceMiddleware(),
    ]
