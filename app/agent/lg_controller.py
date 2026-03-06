from __future__ import annotations

import argparse
import json
import logging
import os
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

from .lg_utils import (
    local_context_from_faiss,
    analyze_local_context,
    should_use_web_judge,
    multiquery_local_search,
    compose_answer_with_policy,
    compose_answer_with_policy_stream,
    footnotes_from_events,
)
from .lc_tools import TAVILY, fetch_url_tool
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

    @staticmethod
    def _tool_invoke(tool_obj, payload):
        try:
            return tool_obj.invoke(payload)
        except Exception:
            if isinstance(payload, dict) and "query" in payload:
                return tool_obj.invoke(payload["query"])
            raise

    def _run_retrieval_path(self, question: str) -> Dict[str, Any]:
        local_ctx = local_context_from_faiss(question, k=15)
        routing = analyze_local_context(question, local_ctx)
        needs_web = bool(routing.get("tentative_use_web"))
        if routing.get("confidence") == "low":
            try:
                judge = should_use_web_judge(question, local_ctx)
                needs_web = bool(judge.get("use_web"))
                routing["judge_reason"] = judge.get("reason")
            except Exception:
                pass
        routing["final_use_web"] = needs_web

        rewrites = int(os.environ.get("LOCAL_RETRIEVAL_REWRITES", "2"))
        mq = multiquery_local_search(question, rewrites=max(0, rewrites), k_per_query=4, top_k=10)
        tool_events = list(mq.get("events", []))
        if mq.get("rewrites"):
            tool_events.append({
                "tool": "rewrite_queries",
                "input": {"question": question, "count": len(mq["rewrites"])},
                "observation": mq["rewrites"],
            })

        web_results = None
        web_page = ""
        if needs_web:
            try:
                web_results = self._tool_invoke(TAVILY, {"query": question, "max_results": 3})
            except Exception as exc:
                web_results = {"error": str(exc)}
            tool_events.append({
                "tool": "tavily_search",
                "input": {"query": question, "max_results": 3},
                "observation": web_results,
                "error": web_results.get("error") if isinstance(web_results, dict) else None,
            })
            if isinstance(web_results, dict):
                hits = web_results.get("results") or []
                if hits:
                    url = hits[0].get("url")
                    if url:
                        try:
                            web_page = fetch_url_tool.invoke({"url": url})
                        except Exception as exc:
                            web_page = f"[fetch_url] error: {exc}"
                        tool_events.append({
                            "tool": "fetch_url",
                            "input": {"url": url},
                            "observation": web_page,
                        })

        return {
            "local_context": local_ctx,
            "local_retrieval": mq.get("context", ""),
            "routing": routing,
            "tool_events": tool_events,
            "web_results": web_results,
            "web_page": web_page,
        }

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
