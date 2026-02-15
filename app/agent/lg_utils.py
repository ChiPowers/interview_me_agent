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


def rewrite_queries(question: str, n: int = 3) -> List[str]:
    """
    Generate short alternative queries to fan-out retrieval.
    Returns up to n unique rewrites (excluding the original question).
    """
    if n <= 0:
        return []
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07"),
        temperature=0.3,
        max_tokens=120,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the question into concise alternative search queries. "
         "Return one per line, no numbering."),
        ("human", "{q}"),
    ])
    msg = prompt.format_messages(q=question)
    raw = llm.invoke(msg).content
    candidates = [line.strip(" -•\t") for line in raw.splitlines()]
    uniq = []
    for c in candidates:
        if not c or c.lower() == question.lower().strip():
            continue
        if c not in uniq:
            uniq.append(c)
        if len(uniq) >= n:
            break
    return uniq


def multiquery_local_search(
    question: str,
    rewrites: int = 3,
    k_per_query: int = 3,
    top_k: int = 6,
) -> Dict[str, Any]:
    """
    Fan-out FAISS retrieval across multiple rewrites, dedupe, and return a merged context.
    """
    vs = load_faiss_or_none()
    if vs is None:
        return {
            "context": "[local] No index loaded.",
            "rewrites": [],
            "hits": [],
            "events": [],
        }

    queries = [question] + rewrite_queries(question, n=rewrites)
    hits: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    seen = set()

    for q in queries:
        try:
            docs = vs.similarity_search(q, k=k_per_query)
        except Exception as exc:  # pragma: no cover - defensive guard
            events.append({
                "tool": "faiss_search",
                "input": {"query": q, "k": k_per_query},
                "observation": f"[retrieve_local] error: {exc}",
                "error": str(exc),
            })
            continue

        obs_lines = []
        for d in docs:
            label = d.metadata.get("label") or d.metadata.get("source", "local.pdf")
            text = (d.page_content or "").strip().replace("\n", " ")
            snippet = text[:320] + ("…" if len(text) > 320 else "")
            obs_lines.append(f"{label} — {snippet}")

            key = (label, snippet[:160])
            if key in seen:
                continue
            seen.add(key)
            hits.append({
                "query": q,
                "label": label,
                "content": (d.page_content or "").strip(),
            })

        events.append({
            "tool": "faiss_search",
            "input": {"query": q, "k": k_per_query},
            "observation": "\n".join(obs_lines) if obs_lines else "[local] No results",
        })

    # Preserve original-query priority, then rewrites; hits already ordered that way.
    selected = hits[:top_k]
    blocks = []
    for h in selected:
        text = h["content"].replace("\n\n", "\n").strip()
        if len(text) > 1000:
            text = text[:1000] + "…"
        blocks.append(f"{h['label']}\n{text}")

    context = "\n\n---\n\n".join(blocks) if blocks else "[local] No results (multi-query)"

    return {
        "context": context,
        "rewrites": queries[1:],  # exclude original
        "hits": selected,
        "events": events,
    }


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

    llm = ChatOpenAI(model=os.getenv("OPENAI_COMPOSER_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")), temperature=0.2)
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
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07"), temperature=0.2)
    raw_answer = llm.invoke(prompt.format_messages(question=question, context=combined_context)).content.strip()
    return guard_answer_with_evidence(question, raw_answer, combined_context)


def guard_answer_with_evidence(question: str, answer: str, context: str) -> str:
    """
    Quick hallucination guard: ensure the answer is supported by the supplied context.
    If not, ask the model to revise or trim claims.
    """
    if not answer:
        return answer

    checker = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07"),
        temperature=0,
        max_tokens=160,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You verify if the answer is fully supported by the context. "
         "Reply as:\nSUPPORTED: yes|no\nANSWER: <supported or trimmed answer with ≤90 words>"),
        ("human",
         "QUESTION:\n{q}\n\nANSWER:\n{a}\n\nCONTEXT:\n{ctx}\n\n"
         "If unsupported, trim or restate using only what is grounded in context."),
    ])
    msg = prompt.format_messages(q=question, a=answer, ctx=context[:3200])
    out = checker.invoke(msg).content.strip()

    supported = "SUPPORTED: yes" in out.lower()
    revised_match = None
    for line in out.splitlines():
        if line.upper().startswith("ANSWER:"):
            revised_match = line.split(":", 1)[1].strip()
            break
    if supported and not revised_match:
        return answer
    return revised_match or answer


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
        model=os.getenv("OPENAI_ROUTE_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")),
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
