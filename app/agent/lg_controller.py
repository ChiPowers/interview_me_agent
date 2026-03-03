from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional

try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
except Exception:  # pragma: no cover
    def traceable(*args, **kwargs):
        def _deco(fn):
            return fn
        return _deco

    def get_current_run_tree():
        return None

from .lg_state import AgentState
try:
    from .lg_nodes import (
        prepare_local_context,
        decide_retrieval_strategy,
        route_after_decision,
        retrieve_local_pass,
        route_after_local_pass,
        maybe_web_search_pass,
    )
except Exception:
    from ._lg_nodes_deprecated import (
        prepare_local_context,
        decide_retrieval_strategy,
        route_after_decision,
        retrieve_local_pass,
        route_after_local_pass,
        maybe_web_search_pass,
    )
from .lg_utils import compose_answer_with_policy, compose_answer_with_policy_stream, footnotes_from_events
from .middleware import evaluate_answer_post
from .eval_utils import maybe_post_feedback_async, POST_FEEDBACK_ENABLED

logger = logging.getLogger("interview_agent.lg_controller")


class LGController:
    """
    Deterministic local-first RAG controller with optional web fallback.

    Built on the LangGraph node functions directly to keep execution stable while
    supporting true token streaming in the UI.
    """

    def __init__(self, thread_id: Optional[str] = None):
        self.thread_id = thread_id or str(uuid.uuid4())
        self.turn_index = 0
        self._last_trace: Optional[Dict[str, Any]] = None
        logger.info("LGController initialized (thread_id=%s)", self.thread_id)

    def _run_retrieval_path(self, question: str) -> AgentState:
        state: AgentState = {
            "thread_id": self.thread_id,
            "turn_index": self.turn_index,
            "question": question,
            "input": question,
            "tool_events": [],
        }
        state = prepare_local_context(state)
        state = decide_retrieval_strategy(state)

        route = route_after_decision(state)
        if route in ("local_only", "needs_web"):
            state = retrieve_local_pass(state)
            if route_after_local_pass(state) == "web":
                state = maybe_web_search_pass(state)
        return state

    @traceable(name="LGController.respond", run_type="chain")
    def respond(self, question: str, on_token: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        self.turn_index += 1
        start = time.time()
        question = (question or "").strip()
        logger.info("[LG] Q%d: %s", self.turn_index, question)

        try:
            state = self._run_retrieval_path(question)
            local_context = state.get("local_context", "")
            local_retrieval = state.get("local_retrieval", "")
            web_results = state.get("web_results")
            web_page = state.get("web_page", "")

            if on_token is None:
                answer = compose_answer_with_policy(
                    question=question,
                    local_context=local_context,
                    local_chunks=local_retrieval,
                    web_results=web_results,
                    web_page=web_page,
                )
            else:
                parts = []
                for token in compose_answer_with_policy_stream(
                    question=question,
                    local_context=local_context,
                    local_chunks=local_retrieval,
                    web_results=web_results,
                    web_page=web_page,
                ):
                    parts.append(token)
                    on_token(token)
                answer = "".join(parts).strip()

            tool_events = state.get("tool_events", [])
            footnotes = footnotes_from_events(tool_events)
            answer, eval_meta = evaluate_answer_post(
                question=question,
                answer=answer,
                context=local_context,
                footnote_count=len(footnotes or {}),
            )
            latency_ms = (time.time() - start) * 1000.0

            run_tree = get_current_run_tree()
            run_id = str(run_tree.id) if run_tree is not None else None
            trace = {
                "plan": "deterministic RAG (local-first with conditional web fallback)",
                "routing": state.get("routing"),
                "tool_events": tool_events,
                "local_context_preview": (local_context or "")[:800],
                "eval": eval_meta,
                "run_id": run_id,
                "controller": "lg_deterministic_v2",
                "latency_ms": latency_ms,
            }
            self._last_trace = trace

            if POST_FEEDBACK_ENABLED:
                maybe_post_feedback_async(
                    run_id,
                    question,
                    answer,
                    local_context,
                    footnotes,
                    reference=None,
                    latency_ms=latency_ms,
                )

            return {"answer": answer, "footnotes": footnotes, "trace": trace}
        except Exception as exc:
            logger.exception("[LG] Pipeline failed: %s", exc)
            return {"answer": f"Error: {exc}", "footnotes": {}, "trace": {"error": str(exc)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick smoke test for LGController.")
    parser.add_argument("question", help="Interview question to pose to the agent")
    args = parser.parse_args()

    controller = LGController()
    result = controller.respond(args.question)
    print(json.dumps(result, indent=2, ensure_ascii=False))
