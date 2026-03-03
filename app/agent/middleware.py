"""
Middleware-style evaluation helpers for the LangGraph pipeline.

These checks run around retrieval/answer composition to keep answers grounded
without adding heavy latency on every turn.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

try:
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*args, **kwargs):
        def _deco(fn):
            return fn
        return _deco

from .lg_utils import guard_answer_with_evidence

ENABLE_LLM_GUARD = os.getenv("ENABLE_HALLUCINATION_GUARD", "1").lower() in ("1", "true", "yes", "on")
MIN_CONTEXT_CHARS = int(os.getenv("MIN_CONTEXT_CHARS_FOR_GROUNDING", "280"))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]{4,}", (text or "").lower()))


@traceable(name="middleware.pre_eval", run_type="chain")
def evaluate_routing_pre(question: str, local_context: str, routing: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fast pre-answer eval. Returns adjusted routing metadata (no model call).
    """
    out = dict(routing or {})
    flags: List[str] = list(out.get("reasons") or [])
    q_low = (question or "").lower()
    ctx = local_context or ""

    if ctx.startswith("[local] No index loaded.") or ctx.startswith("[local] error:"):
        flags.append("grounding_unavailable")
    if len(ctx) < MIN_CONTEXT_CHARS:
        flags.append("thin_context")
    if any(x in q_low for x in ("latest", "current", "today", "recent", "announced", "news")):
        flags.append("freshness_cue")

    out["middleware_flags"] = sorted(set(flags))
    out["needs_web"] = bool(out.get("final_use_web") or out.get("tentative_use_web"))
    return out


def _risk_flags(answer: str, context: str, footnote_count: int) -> List[str]:
    flags: List[str] = []
    if not answer.strip():
        flags.append("empty_answer")
        return flags
    if len(context or "") < MIN_CONTEXT_CHARS:
        flags.append("thin_context")
    if footnote_count == 0:
        flags.append("missing_footnotes")
    if not re.search(r"\[\d+\]", answer):
        flags.append("missing_citation_markers")
    at = _tokenize(answer)
    ct = _tokenize(context)
    if at and ct:
        overlap = len(at & ct) / max(1, len(at))
        if overlap < 0.12:
            flags.append("low_context_overlap")
    return flags


@traceable(name="middleware.post_eval", run_type="chain")
def evaluate_answer_post(
    question: str,
    answer: str,
    context: str,
    footnote_count: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Post-answer eval guard.
    - Always runs cheap checks.
    - Runs LLM grounding guard only when risk flags exist.
    """
    flags = _risk_flags(answer, context, footnote_count)
    meta = {"risk_flags": flags, "llm_guard_used": False}
    if not flags:
        return answer, meta

    if context.startswith("[local] No index loaded.") or context.startswith("[local] error:"):
        safe = (
            "I don't have enough grounded local context loaded to answer confidently right now. "
            "Please ensure the local document index is available, then ask again."
        )
        return safe, meta

    if ENABLE_LLM_GUARD:
        meta["llm_guard_used"] = True
        guarded = guard_answer_with_evidence(question, answer, context)
        return guarded, meta
    return answer, meta

