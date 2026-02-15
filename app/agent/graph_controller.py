"""
Graph controller entrypoint for LangGraph deployment (langgraph.json).

Delegates to the ``create_agent``-based ``build_graph`` from ``lg_graph.py``.
The ``GraphController`` class provides backward compatibility with the
Streamlit UI and the legacy ``respond()`` API.
"""
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional

from .lg_graph import build_graph
from .lg_utils import footnotes_from_events

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")


class GraphController:
    """Drop-in replacement for LCController using create_agent under the hood."""

    def __init__(self):
        self.agent = build_graph()
        self.init_error: Optional[str] = None

    def respond(self, question: str) -> Dict[str, Any]:
        try:
            input_state = {
                "messages": [{"role": "user", "content": question}],
            }
            result = self.agent.invoke(input_state)

            # Extract answer from last AI message
            messages = result.get("messages", [])
            answer = ""
            for msg in reversed(messages):
                content = getattr(msg, "content", None) or (
                    msg.get("content") if isinstance(msg, dict) else ""
                )
                msg_type = getattr(msg, "type", None) or (
                    msg.get("role") if isinstance(msg, dict) else ""
                )
                if msg_type in ("ai", "assistant") and content:
                    answer = content.strip()
                    break

            footnotes = result.get("footnotes", {})
            if not footnotes:
                tool_events = []
                for msg in messages:
                    if getattr(msg, "type", None) == "tool":
                        tool_events.append({
                            "tool": getattr(msg, "name", "tool"),
                            "input": {},
                            "observation": getattr(msg, "content", "") or "",
                        })
                footnotes = footnotes_from_events(tool_events)

            trace = result.get("trace", {})
            if not trace:
                trace = {
                    "plan": "create_agent RAG (local-first with web fallback)",
                    "needs_web": bool(result.get("needs_web")),
                    "local_context_preview": (result.get("local_context") or "")[:800],
                }

            return {"answer": answer, "footnotes": footnotes, "trace": trace}

        except Exception:
            tb = traceback.format_exc()
            return {
                "answer": "Initialization or runtime error.",
                "footnotes": {},
                "trace": {"init_trace": tb},
            }
