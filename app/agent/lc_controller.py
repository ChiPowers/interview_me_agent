# app/agent/lc_controller.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import os
import re
import traceback

from langchain_openai import ChatOpenAI
try:  # langchain>=0.2 removed AgentExecutor from the package root
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:  # langchain>=0.3
    from langchain.agents import create_tool_calling_agent
    from langchain.agents.agent import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler

from .lc_prompts import SYSTEM
from .lc_tools import retrieve_local_tool, TAVILY, fetch_url_tool
from app.services.vectorstore import load_faiss_or_none  # prefetch local_context

import threading, random
from langsmith import Client
from app.eval.evaluators import EvalInput

# --- Config ---
MAX_ITER = 6
DEFAULT_LLM = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
COMPOSER_LLM = os.getenv("OPENAI_COMPOSER_MODEL", DEFAULT_LLM)

# LangSmith feedback toggles (consistent + permissive parsing)
_POST_FEEDBACK_RAW = os.getenv("POST_FEEDBACK", "0")
POST_FEEDBACK_ENABLED = _POST_FEEDBACK_RAW.lower() in ("1", "true", "yes", "on")
POST_FEEDBACK_SAMPLE_RATE = float(os.getenv("POST_FEEDBACK_SAMPLE_RATE", "0.25"))

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


# ---------- Helpers ----------
def _finalize_from_steps(question: str, steps: List[Tuple[Any, Any]]) -> str:
    """If the agent halts early, compose a final answer from the recent observations."""
    local_bits, urls, texts = [], [], []
    url_re = re.compile(r"https?://\S+")
    for (_act, obs) in steps[-3:]:
        if not isinstance(obs, str):
            continue
        texts.append(obs[:1200])
        for line in obs.splitlines()[:8]:
            if "local • " in line and len(local_bits) < 2:
                local_bits.append(line.strip())
        if len(urls) < 3:
            urls.extend(url_re.findall(obs))
        if len(local_bits) >= 2 and len(urls) >= 2:
            break
    urls = list(dict.fromkeys(urls))[:3]

    llm = ChatOpenAI(model=COMPOSER_LLM, temperature=0.2)
    compose = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Chivon. Write a final interview answer:\n"
                "- ≤ 3 sentences, ≤ 90 words.\n- Professional scope only.\n"
                "- Use footnote markers [1], [2] with provided local labels/URLs if applicable.",
            ),
            ("human", "Question: {q}\n\nObserved context:\n{ctx}\n\nLocal labels:\n{labels}\n\nURLs:\n{urls}"),
        ]
    )
    msgs = compose.format_messages(
        q=question,
        ctx="\n\n---\n\n".join(texts) if texts else "- none -",
        labels="\n".join(local_bits) if local_bits else "- none -",
        urls="\n".join(urls) if urls else "- none -",
    )
    return llm.invoke(msgs).content.strip()


def _local_context_from_faiss(question: str, k: int = 6) -> str:
    vs = load_faiss_or_none()
    if vs is None:
        return "[local] No index loaded."
    try:
        docs = vs.similarity_search(question, k=k)
    except Exception as e:
        return f"[local] error: {e}"
    blocks = []
    for d in docs:
        label = d.metadata.get("label") or d.metadata.get("source", "local.pdf")
        text = (d.page_content or "").strip().replace("\n\n", "\n")
        if len(text) > 1000:
            text = text[:1000] + "…"
        blocks.append(f"{label}\n{text}")
    return "\n\n---\n\n".join(blocks) if blocks else "[local] No results"


def _build_footnotes(steps: List[Tuple[Any, Any]]) -> Dict[int, Dict[str, str]]:
    """Collect web URLs and local labels into a numbered footnotes dict."""
    url_pattern = re.compile(r"https?://\S+")
    local_re = re.compile(r"local\s•\s(?P<file>.+?)\s+p\.(?P<page>\d+)", re.IGNORECASE)

    web_urls: List[str] = []
    local_labels: List[Dict[str, str]] = []

    for (_a, obs) in steps:
        if not isinstance(obs, str):
            continue
        web_urls.extend(url_pattern.findall(obs))
        for line in obs.splitlines()[:12]:
            m = local_re.search(line.strip())
            if m:
                label = line.strip()
                local_labels.append(
                    {"label": label, "file": m.group("file"), "page": m.group("page")}
                )

    def _dedup(seq, key=lambda x: x):
        seen = set()
        out_list = []
        for x in seq:
            k = key(x)
            if k in seen:
                continue
            seen.add(k)
            out_list.append(x)
        return out_list

    web_urls = _dedup(web_urls)[:3]
    local_labels = _dedup(local_labels, key=lambda d: d["label"])[:3]

    footnotes: Dict[int, Dict[str, str]] = {}
    idx = 1
    for url in web_urls:
        footnotes[idx] = {"title": "web", "url": url}
        idx += 1
    for d in local_labels:
        footnotes[idx] = {
            "title": f"local — {d['file']} p.{d['page']}",
            "path": f"local://{d['file']}#page={d['page']}",
        }
        idx += 1
    return footnotes


def _post_feedback_worker(run_id: str, ei: EvalInput, latency_ms: float):
    """Background poster: normalize evaluator outputs and post per-metric feedback."""
    try:
        import json
        from ..eval.evaluators import EvalInput, combined_eval_json  # your aggregator
        raw = combined_eval_json(ei, latency_ms=latency_ms)

        # ---- DEBUG (optional) ----
        if os.getenv("EVAL_DEBUG", "0") == "1":
            print("[eval] RAW from combined_eval_json:", type(raw), raw)

        # -- normalize helper --
        def normalize(raw_obj):
            """Return list[{'key': str, 'score': float|None, 'comment': str|None}]"""
            out = []

            def _add(key, score=None, comment=None, reason=None, value=None):
                out.append({
                    "key": str(key),
                    "score": (float(score) if isinstance(score, (int, float)) else None),
                    "comment": (
                        str(comment) if comment is not None else
                        str(reason) if reason is not None else
                        str(value) if value is not None else None
                    ),
                })

            # If it’s a JSON string, try to load it
            if isinstance(raw_obj, str):
                try:
                    raw_parsed = json.loads(raw_obj)
                except Exception:
                    # Not JSON; treat whole string as a single categorical metric
                    _add("eval", None, value=raw_obj)
                    return out
                raw_obj = raw_parsed

            # Dict case
            if isinstance(raw_obj, dict):
                if "key" in raw_obj:
                    _add(raw_obj.get("key"), raw_obj.get("score"),
                         raw_obj.get("comment"), raw_obj.get("reason"), raw_obj.get("value"))
                else:
                    # dict of metrics: {"relevance": {"score":..., "comment":...}, ...}
                    for k, v in raw_obj.items():
                        if isinstance(v, dict):
                            _add(k, v.get("score"), v.get("comment"), v.get("reason"), v.get("value"))
                        else:
                            _add(k, None, value=v)
                return out

            # List/Tuple case
            if isinstance(raw_obj, (list, tuple)):
                for item in raw_obj:
                    if isinstance(item, dict):
                        if "key" in item:
                            _add(item.get("key"), item.get("score"), item.get("comment"),
                                 item.get("reason"), item.get("value"))
                        else:
                            # maybe {"name": "...", "score": ..., "reason": "..."}
                            name = item.get("name") or item.get("metric") or "metric"
                            _add(name, item.get("score"), item.get("comment"),
                                 item.get("reason"), item.get("value"))
                    else:
                        # EvaluationResult-like object
                        key = getattr(item, "key", "metric")
                        score = getattr(item, "score", None)
                        comment = getattr(item, "comment", None)
                        reason = getattr(item, "reason", None)
                        value = getattr(item, "value", None)
                        _add(key, score, comment, reason, value)
                return out

            # Fallback: unknown type
            _add("eval", None, value=str(raw_obj))
            return out

        normalized = normalize(raw)

        if os.getenv("EVAL_DEBUG", "0") == "1":
            print("[eval] NORMALIZED metrics:", normalized)

        from langsmith import Client
        client = Client(
            api_key=os.getenv("LANGSMITH_EVAL_API_KEY"),
            api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        )

        posted = 0
        for m in normalized:
            try:
                client.create_feedback(
                    run_id,
                    key=m["key"],
                    score=m.get("score"),
                    comment=m.get("comment"),
                )
                posted += 1
            except Exception as e:
                print(f"[eval] failed to post metric {m.get('key')}: {e}")

        print(f"[eval] posted {posted}/{len(normalized)} feedback items to run {run_id}")

    except Exception as e:
        print(f"[eval] feedback post failed: {e}")


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


def _maybe_post_feedback_async(
    run_id: str,
    question: str,
    answer: str,
    ctx: str,
    footnotes: dict,
    reference: str | None,
    latency_ms: float,
):
    # must have: feedback enabled, a valid run_id, and tracing ON
    if not POST_FEEDBACK_ENABLED or not run_id:
        # Helpful breadcrumb:
        if POST_FEEDBACK_ENABLED and not run_id:
            print("[eval] skipping feedback: no run_id (is LANGCHAIN_TRACING_V2=true?)")
        return

    # sampling
    import random
    if random.random() > POST_FEEDBACK_SAMPLE_RATE:
        return

    ei = EvalInput(
        question=question,
        answer=answer,
        context=ctx or "",
        footnotes=footnotes or {},
        reference=reference,
    )
    t = threading.Thread(target=_post_feedback_worker, args=(run_id, ei, latency_ms), daemon=True)
    t.start()

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

        # Pre-inject local context
        local_ctx = _local_context_from_faiss(question, k=6)

        # Capture run_id from this invocation
        catcher = _RunIdCatcher()
        out = self.exec.invoke(
            {"input": question, "local_context": local_ctx},
            config={"callbacks": [catcher]},
        )

        text = (out.get("output") or "").strip()
        steps = out.get("intermediate_steps", [])

        # Fallback compose if agent halted
        if not text or "Agent stopped" in text:
            text = _finalize_from_steps(question, steps)

        footnotes_payload = _build_footnotes(steps)
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
        
        # compute latency if you have it; or pass 0.0
        latency_ms = 0.0
        if "latency_ms" in out:
            latency_ms = out["latency_ms"]  # if you add it
        # Post evals without blocking UI
        _run_id = trace.get("run_id")
        _local_ctx = trace.get("local_context_preview", "")
        _maybe_post_feedback_async(_run_id, question, text, _local_ctx, footnotes_payload, reference=None, latency_ms=latency_ms)

        return {"answer": text, "footnotes": footnotes_payload, "trace": trace}
