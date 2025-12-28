# app/agent/lc_controller.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import os
import traceback

from langchain_openai import ChatOpenAI
from importlib import import_module


def _import_attr(attr: str, modules: list[str]):
    """Try several module paths until the attribute is found (LangChain reorganizes often)."""
    for mod in modules:
        try:
            module = import_module(mod)
        except ImportError:
            continue
        obj = getattr(module, attr, None)
        if obj is not None:
            return obj
    raise ImportError(f"Cannot import {attr} from candidates: {modules}")


def _import_first_available(attrs: list[str], modules: list[str]):
    last_exc = None
    for attr in attrs:
        try:
            return _import_attr(attr, modules)
        except ImportError as e:
            last_exc = e
    raise last_exc or ImportError(f"Cannot import any of {attrs}")


AgentExecutor = _import_first_available(
    ["AgentExecutor", "RunnableAgentExecutor"],
    [
        "langchain.agents",
        "langchain.agents.agent",
        "langchain.agents.agent_executor",
        "langchain.agents.base",
        "langchain.agents.executor",
        "langchain.agents.runner",
    ],
)
create_tool_calling_agent = _import_first_available(
    ["create_tool_calling_agent", "create_openai_tools_agent"],
    [
        "langchain.agents",
        "langchain.agents.tool_calling",
        "langchain.agents.tool_calling.agent",
        "langchain.agents.tool_calling.core",
        "langchain.agents.tool_calling.base",
        "langchain.agents.openai_tools",
    ],
)
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler

from .lc_prompts import SYSTEM
from .lc_tools import retrieve_local_tool, TAVILY, fetch_url_tool
from .lg_utils import (
    multiquery_local_search,
    compose_from_observations,
    build_footnotes,
)
from .eval_utils import (
    POST_FEEDBACK_ENABLED,
    POST_FEEDBACK_SAMPLE_RATE,
    maybe_post_feedback_async,
)

# --- Config ---
MAX_ITER = 6
DEFAULT_LLM = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
COMPOSER_LLM = os.getenv("OPENAI_COMPOSER_MODEL", DEFAULT_LLM)

# Optional: quick sanity log
if POST_FEEDBACK_ENABLED:
    print(f"[eval] POST_FEEDBACK enabled (sample={POST_FEEDBACK_SAMPLE_RATE})")
else:
    print("[eval] POST_FEEDBACK disabled")


# ---------- Prompt ----------
POLICY = (
    "Policy:\n"
    "1) Use the Local context below first. If insufficient, then tools in order: retrieve_local → "
    "tavily_search_results_json → fetch_url.\n"
    "2) Keep answers ≤ 3 sentences (≤ 90 words), first person, professional only.\n"
    "3) Add footnote markers [1], [2]. Cite local labels like 'local • <file> p.<n>' and real URLs for web.\n"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM + "\n" + POLICY),
        MessagesPlaceholder("chat_history"),
        ("system", "Local context (may be empty):\n{local_context}"),
        ("human", "Question: {input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
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


# ---------- Agent factory ----------
def make_executor() -> AgentExecutor:
    llm = ChatOpenAI(model=DEFAULT_LLM, temperature=0.2)
    tools = [retrieve_local_tool, TAVILY, fetch_url_tool]
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=PROMPT)

    memory = ConversationBufferWindowMemory(
        k=8,
        memory_key="chat_history",
        input_key="input",   # Important for tool-calling agent + memory
        output_key="output",
        return_messages=True,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=MAX_ITER,
        memory=memory,
        return_intermediate_steps=True,
    )


# ---------- Controller ----------
class LCController:
    def __init__(self):
        self.exec: Optional[AgentExecutor] = None
        self.init_error: Optional[str] = None
        self._ensure_agent()

    def _ensure_agent(self):
        if self.exec is not None:
            return
        try:
            self.exec = make_executor()
            self.init_error = None
        except Exception:
            self.exec = None
            self.init_error = traceback.format_exc()

    def respond(self, question: str) -> Dict[str, Any]:
        if self.exec is None:
            self._ensure_agent()
        if self.exec is None:
            return {
                "answer": "Initialization error.",
                "footnotes": {},
                "trace": {"init_trace": self.init_error or "No traceback."},
            }

        # Pre-inject local context using multi-query retrieval for better recall
        mq = multiquery_local_search(question, rewrites=3, k_per_query=3, top_k=6)
        local_ctx = mq.get("context", "[local] No results")

        # Capture run_id from this invocation
        catcher = _RunIdCatcher()
        out = self.exec.invoke(
            {"input": question, "local_context": local_ctx},
            config={"callbacks": [catcher]},
        )

        text = (out.get("output") or "").strip()
        steps = out.get("intermediate_steps", [])
        pre_steps = []
        for ev in mq.get("events", []):
            pre_steps.append((ev.get("tool", "tool"), ev.get("observation", "")))

        # Fallback compose if agent halted
        if not text or "Agent stopped" in text:
            text = compose_from_observations(question, steps)

        footnotes_payload = build_footnotes(pre_steps + steps)
        run_id_str = str(catcher.top_run_id) if catcher.top_run_id else None

        trace = {
            "plan": "Tool-calling agent (local-first with pre-injected context; web fallback)",
            "steps": [
                {
                    "tool": getattr(a, "tool", ""),
                    "input": getattr(a, "tool_input", ""),
                    "observation": (obs[:240] + "…") if isinstance(obs, str) and len(obs) > 240 else obs,
                }
                for a, obs in steps
            ],
            "local_context_preview": local_ctx[:800] if isinstance(local_ctx, str) else str(local_ctx)[:800],
            "run_id": run_id_str,
        }
        if mq.get("rewrites"):
            trace["local_rewrites"] = mq["rewrites"]
        
        # compute latency if you have it; or pass 0.0
        latency_ms = 0.0
        if "latency_ms" in out:
            latency_ms = out["latency_ms"]  # if you add it
        # Post evals without blocking UI
        _run_id = trace.get("run_id")
        _local_ctx = trace.get("local_context_preview", "")
        maybe_post_feedback_async(_run_id, question, text, _local_ctx, footnotes_payload, reference=None, latency_ms=latency_ms)

        return {"answer": text, "footnotes": footnotes_payload, "trace": trace}
