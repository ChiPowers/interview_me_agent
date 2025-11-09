# app/agent/lg_state.py
from __future__ import annotations
from typing import TypedDict, List, Any, Dict, Optional


class ToolEvent(TypedDict, total=False):
    """Normalized record of every tool call, regardless of source (FAISS, Tavily, fetch_url)."""
    tool: str
    input: Any
    observation: Any
    error: Optional[str]


class AgentState(TypedDict, total=False):
    """
    Shared LangGraph state container.

    The LangChain-based controller will also reuse this structure so we can swap controllers
    without changing the Streamlit surface area.
    """
    # request metadata
    thread_id: Optional[str]
    turn_index: int

    # user prompt & context
    question: str
    input: str  # alias for question to keep compatibility with legacy helpers
    local_context: str

    # routing + analysis
    needs_web: bool
    routing: Dict[str, Any]

    # conversation scaffolding (LangChain/ChatML messages)
    chat_history: List[Any]
    scratchpad: List[Any]

    # tool traces
    tool_events: List[ToolEvent]

    # outputs
    answer: str
    output: str  # alias for answer
    footnotes: Dict[int, Dict[str, str]]
    trace: Dict[str, Any]
    error: Optional[str]
