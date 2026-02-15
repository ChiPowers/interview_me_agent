"""
Interview Agent graph builder — LangChain v1 ``create_agent`` with middleware.

Replaces the hand-rolled StateGraph with a single ``create_agent()`` call.
All RAG routing, hallucination guarding, and trace collection are handled
by middleware hooks defined in ``middleware.py``.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Optional

from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from .lc_prompts import SYSTEM
from .lc_tools import retrieve_local_tool, TAVILY, fetch_url_tool
from .lg_state import InterviewState
from .middleware import get_middleware

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")

# Answer-policy additions appended to the system prompt
POLICY = (
    "\nPolicy:\n"
    "1) ALWAYS call the retrieve_local tool FIRST before answering any question. "
    "You MUST search local documents before responding — never answer from your own knowledge alone.\n"
    "2) If retrieve_local returns insufficient context, then use tavily_search, "
    "then optionally fetch_url for more detail.\n"
    "3) Keep answers ≤ 3 sentences (≤ 90 words), first person, professional only.\n"
    "4) Add footnote markers [1], [2]. Cite local labels like "
    "'local • <file> p.<n>' and real URLs for web.\n"
)


def build_graph(checkpoint_path: Optional[str] = None):
    """
    Return a compiled agent graph using ``create_agent``.

    The agent uses the ReAct loop (model → tool → observation → repeat)
    with middleware controlling RAG routing, hallucination guard, and
    trace/footnote collection.
    """
    tools = [retrieve_local_tool, TAVILY, fetch_url_tool]

    checkpointer = None
    if checkpoint_path:
        conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    agent = create_agent(
        model=f"openai:{DEFAULT_MODEL}",
        tools=tools,
        system_prompt=SYSTEM + POLICY,
        state_schema=InterviewState,
        middleware=get_middleware(),
        checkpointer=checkpointer,
    )

    return agent
