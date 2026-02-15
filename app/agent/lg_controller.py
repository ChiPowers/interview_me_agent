"""
LGController — LangChain v1 ``create_agent`` wrapper.

Provides the same ``respond(question) → {answer, footnotes, trace}`` API
expected by the Streamlit UI and evaluation pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .lg_graph import build_graph
from .lg_utils import footnotes_from_events
from .eval_utils import maybe_post_feedback_async, POST_FEEDBACK_ENABLED

logger = logging.getLogger("interview_agent.lg_controller")


def _default_checkpoint_path() -> str:
    base = Path(os.getenv("LANGGRAPH_CHECKPOINT_DIR", Path("data") / "checkpoints"))
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "lg_controller.sqlite")


class LGController:
    """LangGraph-based controller using create_agent with middleware."""

    def __init__(self, checkpoint_path: Optional[str] = None, thread_id: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or _default_checkpoint_path()
        self.agent = build_graph(checkpoint_path=self.checkpoint_path)
        self.thread_id = thread_id or os.getenv("LANGGRAPH_THREAD_ID") or str(uuid.uuid4())
        self.turn_index = 0
        self._last_trace = None
        logger.info("LGController initialized (thread_id=%s, checkpoint=%s)", self.thread_id, self.checkpoint_path)

    def respond(self, question: str) -> Dict[str, Any]:
        self.turn_index += 1
        logger.info("[LG] Q%d: %s", self.turn_index, question)

        # create_agent uses messages-based state
        input_state = {
            "messages": [{"role": "user", "content": question}],
        }
        config = {"configurable": {"thread_id": self.thread_id}}

        start = time.time()
        try:
            result = self.agent.invoke(input_state, config=config)
        except Exception as exc:
            logger.exception("[LG] Agent invocation failed: %s", exc)
            return {
                "answer": f"Error: {exc}",
                "footnotes": {},
                "trace": {"error": str(exc)},
            }
        latency_ms = (time.time() - start) * 1000.0

        # Extract answer from last AI message
        messages = result.get("messages", [])
        answer = ""
        for msg in reversed(messages):
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
            msg_type = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else "")
            if msg_type in ("ai", "assistant") and content:
                answer = content.strip()
                break

        # Extract footnotes and trace from state (populated by middleware)
        footnotes = result.get("footnotes") or {}
        if not footnotes:
            # Fallback: build from tool messages
            tool_events = result.get("tool_events", [])
            if not tool_events:
                tool_events = []
                for msg in messages:
                    msg_type = getattr(msg, "type", None)
                    if msg_type == "tool":
                        tool_events.append({
                            "tool": getattr(msg, "name", "tool"),
                            "input": {},
                            "observation": getattr(msg, "content", "") or "",
                        })
            footnotes = footnotes_from_events(tool_events)

        trace = (result.get("trace") or {}).copy()
        trace.setdefault("plan", "create_agent RAG (local-first with web fallback)")
        trace.setdefault("routing", result.get("routing"))
        trace.setdefault("local_context_preview", (result.get("local_context") or "")[:800])
        trace["run_id"] = trace.get("run_id") or self.thread_id
        trace["controller"] = "create_agent_v1"
        trace["latency_ms"] = latency_ms
        self._last_trace = trace

        logger.info(
            "[LG] Answered in %.1f ms (needs_web=%s, tools=%d)",
            latency_ms,
            bool(result.get("needs_web")),
            len(trace.get("tool_events", [])),
        )
        logger.debug("[LG] Footnotes: %s", json.dumps(footnotes, ensure_ascii=False))

        if POST_FEEDBACK_ENABLED:
            ctx = trace.get("local_context_preview", "")
            maybe_post_feedback_async(
                trace.get("run_id"), question, answer, ctx,
                footnotes, reference=None, latency_ms=latency_ms,
            )

        return {
            "answer": answer,
            "footnotes": footnotes,
            "trace": trace,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick smoke test for the LangGraph controller.")
    parser.add_argument("question", help="Interview question to pose to the agent")
    args = parser.parse_args()

    controller = LGController()
    result = controller.respond(args.question)
    print(json.dumps(result, indent=2, ensure_ascii=False))
