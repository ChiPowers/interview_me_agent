"""LangGraph deployment adapter over the canonical deterministic RAG controller."""
from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from .lg_controller import LGController
from .lg_state import InterviewState


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def build_graph(checkpoint_path: Optional[str] = None):
    """
    Compile a one-node graph that calls the exact controller used by FastAPI,
    Streamlit, CLI smoke tests, and evaluations.

    ``checkpoint_path`` is retained for compatibility. The canonical v3 pipeline
    remains stateless until conversational memory has its own evaluated contract.
    """
    controller = LGController()

    def answer_node(state: InterviewState) -> dict[str, Any]:
        question = str(state.get("question") or state.get("input") or "").strip()
        if not question:
            for message in reversed(state.get("messages") or []):
                content = _message_content(message)
                role = (
                    message.get("role")
                    if isinstance(message, dict)
                    else getattr(message, "type", "")
                )
                if role in ("user", "human") and content:
                    question = content.strip()
                    break
        result = controller.respond(question)
        return {
            "question": question,
            "input": question,
            "answer": result["answer"],
            "output": result["answer"],
            "sources": result.get("sources") or [],
            "footnotes": result.get("footnotes") or {},
            "source_freshness": result.get("source_freshness") or {},
            "validation": result.get("validation") or {},
            "trace": result.get("trace") or {},
            "messages": [{"role": "assistant", "content": result["answer"]}],
        }

    graph = StateGraph(InterviewState)
    graph.add_node("respond", answer_node)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)
    return graph.compile()
