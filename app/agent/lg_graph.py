"""
LangGraph assembly helpers.

Phase 2 focuses on defining the state schema and creating a modular node layout so we can
drop in the modern LangGraph controller without touching the Streamlit surface area.
"""
from __future__ import annotations

from typing import Optional
import sqlite3

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .lg_state import AgentState
from . import lg_nodes


def build_graph(checkpoint_path: Optional[str] = None):
    """
    Return a compiled LangGraph pipeline.

    Nodes are currently lightweight wrappers around the existing LangChain logic.
    Later phases will attach richer tool-calling and evaluation hooks.
    """
    graph = StateGraph(AgentState)

    graph.add_node("prepare_context", lg_nodes.prepare_local_context)
    graph.add_node("decide_retrieval", lg_nodes.decide_retrieval_strategy)
    graph.add_node("retrieve_local", lg_nodes.retrieve_local_pass)
    graph.add_node("maybe_web", lg_nodes.maybe_web_search_pass)
    graph.add_node("compose_answer", lg_nodes.compose_answer_pass)
    graph.add_node("finalize", lg_nodes.finalize_pass)

    graph.set_entry_point("prepare_context")
    graph.add_edge("prepare_context", "decide_retrieval")

    graph.add_conditional_edges(
        "decide_retrieval",
        lg_nodes.route_after_decision,
        {
            "local_only": "retrieve_local",
            "needs_web": "retrieve_local",
            "skip_retrieval": "compose_answer",
        },
    )

    # After local retrieval, optionally branch into web search (when flagged)
    graph.add_conditional_edges(
        "retrieve_local",
        lg_nodes.route_after_local_pass,
        {
            "web": "maybe_web",
            "no_web": "compose_answer",
        },
    )

    graph.add_edge("maybe_web", "compose_answer")
    graph.add_edge("compose_answer", "finalize")
    graph.add_edge("finalize", END)

    checkpointer = None
    if checkpoint_path:
        conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)
