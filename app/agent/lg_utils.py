"""Shared helpers for both LangChain and LangGraph controllers."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.services.vectorstore import load_faiss_or_none
from .lc_prompts import SYSTEM


def local_context_from_faiss(question: str, k: int = 6) -> str:
    vs = load_faiss_or_none()
    if vs is None:
        return "[local] No index loaded."
    try:
        docs = vs.similarity_search(question, k=k)
    except Exception as e:  # pragma: no cover - defensive guard
        return f"[local] error: {e}"
    blocks = []
    for d in docs:
        label = d.metadata.get("label") or d.metadata.get("source", "local.pdf")
        text = (d.page_content or "").strip().replace("\n\n", "\n")
        if len(text) > 1000:
            text = text[:1000] + "…"
        blocks.append(f"{label}\n{text}")
    return "\n\n---\n\n".join(blocks) if blocks else "[local] No results"


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


def build_footnotes(steps: List[Tuple[Any, Any]]) -> Dict[int, Dict[str, str]]:
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


def footnotes_from_events(events: List[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    """Convert normalized tool events into the same numbered footnote payload."""
    steps = []
    for ev in events:
        observation = ev.get("observation", "")
        if isinstance(observation, (dict, list)):
            observation = json.dumps(observation, ensure_ascii=False)
        steps.append((ev.get("tool", "tool"), observation))
    return build_footnotes(steps)


def compose_from_observations(question: str, steps: List[Tuple[Any, Any]]) -> str:
    """Fallback composer for when the agent halts mid-run."""
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

    llm = ChatOpenAI(model=os.getenv("OPENAI_COMPOSER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")), temperature=0.2)
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


def compose_answer_with_policy(
    question: str,
    local_context: str,
    local_chunks: str,
    web_results: Any,
    web_page: str,
) -> str:
    """Primary composer used by the LangGraph path."""
    context_sections = []
    if local_context:
        context_sections.append(f"LOCAL_CONTEXT:\n{local_context}")
    if local_chunks and "[retrieve_local]" not in local_chunks:
        context_sections.append(f"LOCAL_RETRIEVAL:\n{local_chunks}")
    if web_results:
        snippet = json.dumps(web_results, ensure_ascii=False) if isinstance(web_results, (dict, list)) else str(web_results)
        context_sections.append(f"WEB_SEARCH:\n{snippet[:1800]}")
    if web_page:
        context_sections.append(f"WEB_PAGE:\n{web_page[:1800]}")

    combined_context = "\n\n---\n\n".join(context_sections) if context_sections else "No supporting context."

    policy = (
        "Answer strictly in first person, professional tone. "
        "Limit to ≤3 sentences and ≤90 words. "
        "Reference the provided context and mark citations with [1], [2]."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM + "\n" + policy),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ])
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
    return llm.invoke(prompt.format_messages(question=question, context=combined_context)).content.strip()


def analyze_local_context(question: str, local_ctx: str) -> Dict[str, Any]:
    """Deterministic heuristic used before invoking any LLM routing logic."""
    FRESHNESS_WORDS = [
        "latest","today","this year","this month","recent","currently","current","now",
        "2024","2025","news","announce","announced","update","updated","who is the ceo",
        "stock price","earnings","release date","deadline",
    ]
    OPEN_WEB_HINTS = ["paper","dataset","arxiv","pubmed","docs","documentation","blog","github","website","link","url"]

    reasons: List[str] = []
    q_low = (question or "").lower().strip()
    ctx = local_ctx or ""

    hard = False
    if ctx.startswith("[local] No index loaded."):
        reasons.append("no_index"); hard = True
    if ctx.startswith("[local] error:"):
        reasons.append("index_error"); hard = True

    chunks = [c for c in ctx.split("\n\n---\n\n") if c.strip()]
    total_chars = len("".join(chunks))
    if total_chars < 400:
        reasons.append("local_context_too_small")
    if len(chunks) < 2:
        reasons.append("too_few_chunks")

    if any(w in q_low for w in FRESHNESS_WORDS):
        reasons.append("freshness_cue")
    if any(w in q_low for w in OPEN_WEB_HINTS):
        reasons.append("open_web_intent")

    use_web = len(reasons) > 0
    confident = hard or (
        (("local_context_too_small" in reasons or "too_few_chunks" in reasons)
         and ("freshness_cue" in reasons or "open_web_intent" in reasons))
    )
    confidence = "high" if confident else "low"

    return {
        "tentative_use_web": use_web,
        "reasons": reasons,
        "chunk_count": len(chunks),
        "chars": total_chars,
        "confidence": confidence,
    }


def should_use_web_judge(question: str, local_ctx: str) -> Dict[str, Any]:
    """Tiny LLM judge to double-check the routing heuristic."""
    judge = ChatOpenAI(
        model=os.getenv("OPENAI_ROUTE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        temperature=0,
        max_tokens=32,
        timeout=8,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Decide if web search is needed. Respond exactly as:\n"
         "USE_WEB: yes|no | REASON: <short>"),
        ("human",
         "QUESTION:\n{q}\n\nLOCAL CONTEXT (may be empty):\n{ctx}\n\n"
         "Use web only when the local knowledge is clearly insufficient or outdated.")
    ])
    msg = prompt.format_messages(q=question, ctx=(local_ctx or "- none -")[:1200])
    out = judge.invoke(msg).content.strip()
    m = re.search(r"USE_WEB\s*:\s*(yes|no)", out, re.I)
    use_web = bool(m and m.group(1).lower() == "yes")
    return {"use_web": use_web, "reason": out}
