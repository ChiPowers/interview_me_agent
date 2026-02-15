# app/agent/lg_state.py
"""
State schemas for the Interview Agent.

LangChain v1 requires TypedDict for custom state. The base ``AgentState``
from ``langchain.agents`` provides the ``messages`` list; we extend it
with interview-specific fields.
"""
from __future__ import annotations
from typing import TypedDict, List, Any, Dict, Optional

from langchain.agents import AgentState


class ToolEvent(TypedDict, total=False):
    """Normalized record of every tool call, regardless of source."""
    tool: str
    input: Any
    observation: Any
    error: Optional[str]


class InterviewState(AgentState):
    """
    Extended state for the interview agent.

    Inherits ``messages: list`` from AgentState.
    Custom fields are populated by middleware and read by the controller.
    """
    # RAG context
    local_context: str
    needs_web: bool
    routing: Dict[str, Any]

    # tool traces
    tool_events: List[ToolEvent]

    # outputs (populated by middleware)
    footnotes: Dict[int, Dict[str, str]]
    trace: Dict[str, Any]
