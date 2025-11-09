from __future__ import annotations
from typing import TypedDict, Dict, Any, Optional, List, Tuple
import os, re, json, threading, random, traceback

# LangGraph
from langgraph.graph import StateGraph, START, END

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory

# Tools / local vectorstore
from .lc_prompts import SYSTEM
from .lc_tools import retrieve_local_tool, TAVILY, fetch_url_tool
from app.services.vectorstore import load_faiss_or_none

# LangSmith (optional feedback “online evals”)
from langsmith import Client
from eval.evaluators import EvalInput, combined_eval_json  # your fast single-call evaluator

# ------------------------ Config ------------------------
DEFAULT_LLM = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
COMPOSER_LLM = os.getenv("OPENAI_COMPOSER_MODEL", DEFAULT_LLM)
MAX_ITER = 6

POST_FEEDBACK = os.getenv("POST_FEEDBACK", "0") in ("1", "true", "True")
POST_FEEDBACK_SAMPLE_RATE = float(os.getenv("POST_FEEDBACK_SAMPLE_RATE", "0.25"))

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
        ("system", "Local context (may be empty):\n{local_context}"),
        ("human", "Question: {input}"),
    ]
)

# ------------------------ State ------------------------
class GraphState(TypedDict, total=False):
    input: str
    chat_history: list  # (we keep memory outside, but include key for future)
    local_context: str
    local_hits_meta: list  # [{"label": "...", "score": 0.12}, ...] if available
    need_web: bool
    need_web_reason: str
    web_context: str
    draft: str
    answer: str
    footnotes: Dict[int, Dict[str, str]]
    trace_steps: List[Dict[str, Any]]
    run_id: Optional[str]

# ------------------------ Utilities ------------------------
class _RunIdCatcher(BaseCallbackHandler):
    """Capture the top-level run_id for Studio/LangSmith."""
    def __init__(self):
        self.top_run_id = None

    def on_chain_start(self, serialized, inputs, run_id, **kwargs):
        if self.top_run_id is None:
            self.top_run_id = run_id

    def on_llm_start(self, serialized, prompts, run_id, **kwargs):
        if self.top_run_id is None:
            self.top_run_id = run_id

def _local_context_from_faiss(question: str, k: int = 6) -> Tuple[str, List[Dict[str, Any]]]:
    """Return concatenated context + light metadata (label + optional score)."""
    vs = load_faiss_or_none()
    if vs is None:
        return "[local] No index loaded.", []
    try:
        # Try with scores; fallback to no-score
        try:
            docs_scores = vs.similarity_search_with_score(question, k=k)
            docs = [d for d, _ in docs_scores]
            scores = [s for _, s in docs_scores]
        except Exception:
            docs = vs.similarity_search(question, k=k)
            scores = [None] * len(docs)
    except Exception as e:
        return f"[local] error: {e}", []

    parts, meta = [], []
    for d, s in zip(docs, scores):
        label = d.metadata.get("label") or d.metadata.get("source", "local.pdf")
        text = (d.page_content or "").strip().replace("\n\n", "\n")
        if len(text) > 1000:
            text = text[:1000] + "…"
        parts.append(f"{label}\n{text}")
        meta.append({"label": label, "score": s})
    return ("\n\n---\n\n".join(parts) if parts else "[local] No results"), meta

def _assess_local_need_web(question: str, local_ctx: str, meta: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Heuristic + LLM rubric:
      - Heuristic: if no results or total text < 400 chars -> need web
      - Otherwise ask a small model to judge: "Is local context sufficient to answer?"
    """
    if not local_ctx or "No results" in local_ctx:
        return True, "No local hits."
    if len(local_ctx) < 400:
        return True, "Local context too short."

    judge = ChatOpenAI(model=os.getenv("OPENAI_RUBRIC_MODEL", "gpt-4o-mini"), temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a strict routing evaluator. Decide if LOCAL CONTEXT is sufficient to answer the QUESTION accurately WITHOUT web.\n"
                       "Reply with 'YES' or 'NO' on the first line, then a one-line reason."),
            ("user", "QUESTION:\n{q}\n\nLOCAL CONTEXT (snippets):\n{ctx}")
        ]
    )
    out = judge.invoke(prompt.format_messages(q=question, ctx=local_ctx))
    text = (out.content or "").strip()
    need_web = not text.splitlines()[0].strip().upper().startswith("YES")
    return need_web, text

def _web_gather(question: str, max_results: int = 3) -> Tuple[str, List[str]]:
    """Use Tavily search tool and fetch_url to build web_context and collect URLs."""
    web_bits, urls = [], []
    try:
        results = TAVILY.invoke({"query": question, "max_results": max_results})
        # results is typically a list of dicts with "url" & "content" keys
        for r in results[:max_results]:
            url = r.get("url")
            if url:
                urls.append(url)
                try:
                    page = fetch_url_tool.invoke({"url": url})
                    # Keep a short excerpt to avoid prompt bloat
                    text = str(page)[:1200]
                    web_bits.append(f"{url}\n{text}")
                except Exception:
                    pass
    except Exception:
        pass
    return ("\n\n---\n\n".join(web_bits) if web_bits else ""), urls

def _build_footnotes(local_ctx: str, web_urls: List[str]) -> Dict[int, Dict[str, str]]:
    """Turn observed labels + urls into indexed footnotes (kept internal)."""
    footnotes: Dict[int, Dict[str, str]] = {}
    idx = 1
    for url in web_urls[:3]:
        footnotes[idx] = {"title": "web", "url": url}
        idx += 1
    # parse local labels
    local_re = re.compile(r"local\s•\s(?P<file>.+?)\s+p\.(?P<page>\d+)", re.IGNORECASE)
    for line in local_ctx.splitlines():
        m = local_re.search(line.strip())
        if m:
            footnotes[idx] = {
                "title": f"local — {m.group('file')} p.{m.group('page')}",
                "path": f"local://{m.group('file')}#page={m.group('page')}",
            }
            idx += 1
            if idx > 6:  # cap locals
                break
    return footnotes

def _compose_answer(question: str, local_ctx: str, web_ctx: str) -> str:
    llm = ChatOpenAI(model=COMPOSER_LLM, temperature=0.2)
    messages = PROMPT.format_messages(input=question, local_context=(local_ctx + ("\n\n" + web_ctx if web_ctx else "")))
    return llm.invoke(messages).content.strip()

def _post_feedback_async(run_id: Optional[str], ei: EvalInput, latency_ms: float):
    """Fire-and-forget LangSmith feedback posting with sampling."""
    if not (POST_FEEDBACK and run_id):
        return
    if random.random() > POST_FEEDBACK_SAMPLE_RATE:
        return

    def _worker():
        try:
            metrics = combined_eval_json(ei, latency_ms=latency_ms)  # {"relevance": {...}, ...}
            client = Client(
                api_key=os.getenv("LANGSMITH_API_KEY"),
                api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
            )
            for k, v in metrics.items():
                client.create_feedback(run_id, key=k, score=v.get("score"), comment=v.get("comment"))
        except Exception as e:
            print(f"[eval] feedback post failed: {e}")

    threading.Thread(target=_worker, daemon=True).start()

# ------------------------ Nodes ------------------------
def n_retrieve_local(state: GraphState) -> GraphState:
    q = state["input"]
    local_ctx, meta = _local_context_from_faiss(q, k=6)
    return {**state, "local_context": local_ctx, "local_hits_meta": meta}

def n_assess(state: GraphState) -> GraphState:
    q = state["input"]
    local_ctx = state.get("local_context", "")
    need_web, reason = _assess_local_need_web(q, local_ctx, state.get("local_hits_meta", []))
    return {**state, "need_web": need_web, "need_web_reason": reason}

def n_web_search(state: GraphState) -> GraphState:
    q = state["input"]
    web_ctx, urls = _web_gather(q, max_results=3)
    fn = _build_footnotes(state.get("local_context", ""), urls)
    return {**state, "web_context": web_ctx, "footnotes": fn}

def n_compose(state: GraphState) -> GraphState:
    q = state["input"]
    answer = _compose_answer(q, state.get("local_context", ""), state.get("web_context", ""))
    # If footnotes weren’t created yet (when skipping web), still parse locals for eval
    if "footnotes" not in state:
        fn = _build_footnotes(state.get("local_context", ""), [])
    else:
        fn = state["footnotes"]
    return {**state, "answer": answer, "footnotes": fn}

def _route_from_assess(state: GraphState) -> str:
    return "web" if state.get("need_web") else "compose"

# ------------------------ Graph builder ------------------------
def build_graph():
    g = StateGraph(GraphState)

    # Nodes
    g.add_node("retrieve_local", n_retrieve_local)
    g.add_node("assess", n_assess)
    g.add_node("web_search", n_web_search)
    g.add_node("compose", n_compose)

    # Edges
    g.add_edge(START, "retrieve_local")
    g.add_edge("retrieve_local", "assess")
    g.add_conditional_edges("assess", _route_from_assess, {"web": "web_search", "compose": "compose"})
    g.add_edge("web_search", "compose")
    g.add_edge("compose", END)

    return g.compile()

# ------------------------ Controller (Streamlit-compatible) ------------------------
class GraphController:
    """Drop-in replacement for LCController using LangGraph under the hood."""
    def __init__(self):
        self.graph = build_graph()
        self.memory = ConversationBufferWindowMemory(
            k=8, memory_key="chat_history", input_key="input", output_key="answer", return_messages=True
        )
        self.init_error: Optional[str] = None

    def respond(self, question: str) -> Dict[str, Any]:
        try:
            # Capture run_id to show in UI and to post feedback
            catcher = _RunIdCatcher()
            # Inject minimal “chat_history” if you want memory later
            state_in: GraphState = {"input": question, "chat_history": []}
            out: GraphState = self.graph.invoke(state_in, config={"callbacks": [catcher]})
            answer = out.get("answer", "").strip()
            footnotes = out.get("footnotes", {})  # keep for later evals (UI can ignore)
            run_id = str(catcher.top_run_id) if catcher.top_run_id else None

            # Build a small trace preview for your UI
            trace = {
                "plan": "LangGraph: retrieve_local → assess → (web?) → compose",
                "need_web": bool(out.get("need_web")),
                "need_web_reason": out.get("need_web_reason", ""),
                "local_context_preview": (out.get("local_context") or "")[:800],
                "run_id": run_id,
            }

            # Optional: post online eval feedback asynchronously
            ei = EvalInput(
                question=question,
                answer=answer,
                context=trace["local_context_preview"],
                footnotes=footnotes or {},
                reference=None,
            )
            _post_feedback_async(run_id, ei, latency_ms=0.0)

            return {"answer": answer, "footnotes": footnotes, "trace": trace}

        except Exception:
            tb = traceback.format_exc()
            return {"answer": "Initialization or runtime error.", "footnotes": {}, "trace": {"init_trace": tb}}
