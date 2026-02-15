# app/agent/lc_controller.py
"""
Legacy LCController — updated to use LangChain v1 ``create_agent``.

Provides the same ``respond(question) → {answer, footnotes, trace}`` API.
Set AGENT_BACKEND=langchain to use this controller instead of the default
LGController (which adds checkpointing and middleware).
"""
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

from .lc_prompts import SYSTEM
from .lc_tools import retrieve_local_tool, TAVILY, fetch_url_tool
from .lg_utils import (
    footnotes_from_events,
)
from .eval_utils import (
    POST_FEEDBACK_ENABLED,
    POST_FEEDBACK_SAMPLE_RATE,
    maybe_post_feedback_async,
)

# --- Config ---
DEFAULT_LLM = os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")
COMPOSER_LLM = os.getenv("OPENAI_COMPOSER_MODEL", DEFAULT_LLM)

# Optional: quick sanity log
if POST_FEEDBACK_ENABLED:
    print(f"[eval] POST_FEEDBACK enabled (sample={POST_FEEDBACK_SAMPLE_RATE})")
else:
    print("[eval] POST_FEEDBACK disabled")


# ---------- Prompt ----------
POLICY = (
    "\nPolicy:\n"
    "1) Use the Local context below first. If insufficient, then tools in order: "
    "retrieve_local → tavily_search → fetch_url.\n"
    "2) Keep answers ≤ 3 sentences (≤ 90 words), first person, professional only.\n"
    "3) Add footnote markers [1], [2]. Cite local labels like "
    "'local • <file> p.<n>' and real URLs for web.\n"
)


# ---------- Run ID capture ----------
class _RunIdCatcher(BaseCallbackHandler):
    """Capture the top-level run_id for the agent execution."""
    def __init__(self):
        self.top_run_id = None

    def on_chain_start(self, serialized, inputs, run_id, **kwargs):
        if self.top_run_id is None:
            self.top_run_id = run_id

    def on_llm_start(self, serialized, prompts, run_id, **kwargs):
        if self.top_run_id is None:
            self.top_run_id = run_id


# ---------- Controller ----------
class LCController:
    def __init__(self):
        self.agent = None
        self.init_error: Optional[str] = None
        self._ensure_agent()

    def _ensure_agent(self):
        if self.agent is not None:
            return
        try:
            tools = [retrieve_local_tool, TAVILY, fetch_url_tool]
            self.agent = create_agent(
                model=f"openai:{DEFAULT_LLM}",
                tools=tools,
                system_prompt=SYSTEM + POLICY,
            )
            self.init_error = None
        except Exception:
            self.agent = None
            self.init_error = traceback.format_exc()

    def respond(self, question: str) -> Dict[str, Any]:
        if self.agent is None:
            self._ensure_agent()
        if self.agent is None:
            return {
                "answer": "Initialization error.",
                "footnotes": {},
                "trace": {"init_trace": self.init_error or "No traceback."},
            }

        catcher = _RunIdCatcher()
        input_state = {
            "messages": [{"role": "user", "content": question}],
        }

        try:
            result = self.agent.invoke(
                input_state,
                config={"callbacks": [catcher]},
            )
        except Exception as exc:
            return {
                "answer": f"Error: {exc}",
                "footnotes": {},
                "trace": {"error": str(exc)},
            }

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

        # Build footnotes from tool messages
        tool_events = []
        for msg in messages:
            if getattr(msg, "type", None) == "tool":
                tool_events.append({
                    "tool": getattr(msg, "name", "tool"),
                    "input": {},
                    "observation": getattr(msg, "content", "") or "",
                })
        footnotes = footnotes_from_events(tool_events)

        run_id_str = str(catcher.top_run_id) if catcher.top_run_id else None
        trace = {
            "plan": "create_agent (legacy LCController path)",
            "tool_events": tool_events,
            "run_id": run_id_str,
            "controller": "lc_create_agent_v1",
        }

        maybe_post_feedback_async(
            run_id_str, question, answer, "",
            footnotes, reference=None, latency_ms=0.0,
        )

        return {"answer": answer, "footnotes": footnotes, "trace": trace}
