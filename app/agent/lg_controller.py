from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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

from app.services.ingest_index import load_manifest
from app.services.profile_snapshot import load_profile_snapshot, snapshot_as_text
from app.services.settings import (
    GENERATION_MODEL,
    LINKEDIN_PROFILE_URL,
    MAX_CONTEXT_TOKENS,
    WEB_FALLBACK_ENABLED,
)
from app.services.web_search import search_web

from .eval_utils import POST_FEEDBACK_ENABLED, maybe_post_feedback_async
from .lc_prompts import REFUSAL, SYSTEM
from .lg_utils import rewrite_queries
from .rag_types import Evidence, RetrievalResult
from .retrieval import (
    evidence_context,
    footnotes_from_evidence,
    question_requires_freshness,
    retrieve_hybrid,
    sources_from_evidence,
)

logger = logging.getLogger("interview_agent.lg_controller")

_PII_OUTPUT_RE = re.compile(
    r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})|"
    r"(?:(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4})|"
    r"(?:\b\d{1,6}\s+[A-Za-z0-9.' -]+\s+"
    r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|court|ct)\b)|"
    r"(?:\b\d{5}(?:-\d{4})?\b)",
    re.I,
)
_PRIVATE_QUESTION_PATTERNS = (
    r"\b(?:home|street|mailing|residential)\s+address\b",
    r"\bwhere\s+do\s+you\s+live\b",
    r"\b(?:your\s+)?(?:phone|telephone|mobile|cell)\s+number\b",
    r"\b(?:your\s+)?personal\s+email(?:\s+address)?\b",
    r"\b(?:your\s+)?email\s+address\b",
    r"\b(?:your|current|expected|desired|base|annual)\s+"
    r"(?:salary|income|compensation)\b",
    r"\b(?:salary|income|compensation)\s+"
    r"(?:expectations?|history|range|amount)\b",
    r"\b(?:your|my)\s+family\b(?!\s+of\b)",
    r"\b(?:family|personal)\s+(?:background|life|members?|details?)\b",
    r"\b(?:your\s+)?(?:spouse|husband|wife|children|kids)\b",
)
_GENERAL_WEB_TERMS = (
    "company",
    "industry",
    "market",
    "mission",
    "mobility",
    "news",
    "announcement",
    "announcements",
    "public announcement",
    "press release",
    "press releases",
    "recent developments",
    "updates",
    "what does lime",
)
_UNSUPPORTED_LIME_DETAIL_PATTERNS = (
    r"\bwhy\b.*\b(?:join|joined|choose|chose)\b.*\blime\b",
    r"\b(?:reason|motivation|decision)\b.*\b(?:join|joining)\b.*\blime\b",
    r"\b(?:manager|reports?)\b.*\blime\b",
    r"\blime\b.*\b(?:manager|reports?)\b",
    r"\b(?:confidential|internal|nonpublic|non-public)\b.*\blime\b",
    r"\blime\b.*\b(?:confidential|internal|nonpublic|non-public)\b",
    r"\b(?:fine[- ]?tun(?:e|ed|ing)|training data)\b.*\blime\b",
    r"\blime\b.*\b(?:fine[- ]?tun(?:e|ed|ing)|training data)\b",
)


def _normalized_url(url: str) -> str:
    parsed = urlparse(url or "")
    return f"{parsed.netloc.lower()}{parsed.path.rstrip('/').lower()}"


def _allows_general_web(question: str) -> bool:
    low = (question or "").lower()
    return "lime" in low and any(term in low for term in _GENERAL_WEB_TERMS)


def _asks_private_question(question: str) -> bool:
    """Match requests for personal data without blocking professional homonyms."""
    low = (question or "").lower()
    return any(re.search(pattern, low) for pattern in _PRIVATE_QUESTION_PATTERNS)


def _asks_for_unsupported_lime_detail(question: str) -> bool:
    low = (question or "").lower()
    return "lime" in low and any(
        re.search(pattern, low) for pattern in _UNSUPPORTED_LIME_DETAIL_PATTERNS
    )


def _asks_current_employer(question: str) -> bool:
    low = (question or "").lower()
    return any(
        phrase in low
        for phrase in (
            "current employer",
            "current company",
            "where do you work",
            "who do you work for",
            "what company are you at",
        )
    )


def _profile_snapshot_is_relevant(question: str) -> bool:
    return "lime" in (question or "").lower() or _asks_current_employer(question)


def _web_evidence(
    question: str,
    payload: dict[str, Any],
    *,
    start_index: int,
) -> list[Evidence]:
    """Convert structured web results into evidence under the disclosure policy."""
    canonical = _normalized_url(LINKEDIN_PROFILE_URL)
    allow_general = _allows_general_web(question)
    evidence: list[Evidence] = []
    for item in payload.get("results") or []:
        url = str(item.get("url") or "")
        if not url:
            continue
        is_canonical = _normalized_url(url) == canonical
        if not is_canonical and not allow_general:
            continue
        content = (item.get("content") or "").strip()
        if not content:
            continue
        source_type = "public_profile" if is_canonical else "web"
        evidence.append(
            Evidence(
                id=f"E{start_index + len(evidence)}",
                content=content[:1800],
                label=(
                    "public profile • LinkedIn"
                    if is_canonical
                    else str(item.get("title") or urlparse(url).netloc)
                ),
                source_type=source_type,
                score=float(item.get("score") or 0.0),
                url=url,
                metadata={
                    "published_date": item.get("published_date"),
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )
        if len(evidence) >= 2:
            break
    return evidence


def _renumber_evidence(items: list[Evidence]) -> list[Evidence]:
    for index, item in enumerate(items, start=1):
        item.id = f"E{index}"
    return items


def _profile_snapshot_evidence() -> Optional[Evidence]:
    snapshot = load_profile_snapshot()
    content = snapshot_as_text(snapshot)
    url = snapshot.get("canonical_url")
    if not content or not url:
        return None
    return Evidence(
        id="E1",
        content=content,
        label="public profile • LinkedIn • current employer",
        source_type="public_profile",
        score=1.0,
        url=url,
        metadata={"verified_at": snapshot.get("verified_at")},
    )


def _cap_context(context: str) -> str:
    # Conservative character approximation keeps the evidence section well below
    # the configured token ceiling without adding another tokenizer dependency.
    return context[: MAX_CONTEXT_TOKENS * 4]


def _source_freshness(evidence: list[Evidence]) -> dict[str, Any]:
    manifest = load_manifest()
    verified = [
        item.metadata.get("verified_at")
        for item in evidence
        if item.metadata.get("verified_at")
    ]
    retrieved = [
        item.metadata.get("retrieved_at")
        for item in evidence
        if item.metadata.get("retrieved_at")
    ]
    return {
        "index_built_at": manifest.get("built_at"),
        "profile_verified_at": max(verified) if verified else None,
        "web_retrieved_at": max(retrieved) if retrieved else None,
    }


def validate_answer(answer: str, source_count: int) -> dict[str, Any]:
    words = re.findall(r"\b[\w'-]+\b", answer or "")
    sentences = [
        item for item in re.split(r"(?<=[.!?])\s+", (answer or "").strip()) if item
    ]
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "within_target_length": len(words) <= 120,
        "pii_detected": bool(_PII_OUTPUT_RE.search(answer or "")),
        "source_count": source_count,
        "rewrote_streamed_answer": False,
    }


class LGController:
    """Canonical deterministic RAG controller used by every application surface."""

    def __init__(
        self,
        thread_id: Optional[str] = None,
        *,
        model_name: Optional[str] = None,
        include_context_in_trace: bool = False,
    ):
        self.thread_id = thread_id or str(uuid.uuid4())
        self.turn_index = 0
        self.model_name = model_name or GENERATION_MODEL
        self.include_context_in_trace = include_context_in_trace
        self._last_trace: Optional[Dict[str, Any]] = None
        logger.info(
            "LGController initialized (thread_id=%s, model=%s)",
            self.thread_id,
            self.model_name,
        )

    def _retrieve(self, question: str) -> tuple[RetrievalResult, dict[str, Any] | None]:
        result = retrieve_hybrid(question)
        web_payload = None
        if _profile_snapshot_is_relevant(question) and not any(
            item.source_type == "public_profile" for item in result.evidence
        ):
            profile = _profile_snapshot_evidence()
            if profile is not None:
                result.evidence = _renumber_evidence(
                    [profile] + result.evidence[:3]
                )
                if result.confidence == "low":
                    result.confidence = "medium"
                result.reasons.append("approved_profile_snapshot_fallback")

        if (
            result.confidence == "low"
            and not question_requires_freshness(question)
            and int(os.getenv("LOCAL_RETRIEVAL_REWRITES", "1")) > 0
            and os.getenv("OPENAI_API_KEY")
        ):
            try:
                rewrites = rewrite_queries(question, n=1)
                if rewrites:
                    rewritten = retrieve_hybrid(question, alternate_queries=rewrites)
                    rewritten.query_rewritten = True
                    if rewritten.confidence != "low" or len(rewritten.evidence) > len(result.evidence):
                        result = rewritten
            except Exception as exc:
                result.reasons.append(f"rewrite_failed:{type(exc).__name__}")

        needs_web = WEB_FALLBACK_ENABLED and (
            result.confidence == "low"
            or (
                question_requires_freshness(question)
                and not _asks_current_employer(question)
            )
        )
        if needs_web:
            web_payload = search_web(question, max_results=3)
            web_items = _web_evidence(
                question,
                web_payload,
                start_index=len(result.evidence) + 1,
            )
            if web_items:
                local_limit = max(0, 4 - len(web_items))
                result.evidence = _renumber_evidence(
                    result.evidence[:local_limit] + web_items
                )
                result.web_fallback_used = True
            elif web_payload.get("error"):
                result.reasons.append(f"web_unavailable:{web_payload['error']}")
            else:
                result.reasons.append("web_returned_no_allowed_evidence")
        return result, web_payload

    def _compose(
        self,
        question: str,
        evidence: list[Evidence],
        on_token: Optional[Callable[[str], None]],
    ) -> tuple[str, Optional[float]]:
        context = _cap_context(evidence_context(evidence))
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM),
                (
                    "human",
                    "Question:\n{question}\n\n"
                    "Approved evidence:\n{context}\n\n"
                    "Answer using only this evidence. The UI will render sources separately.",
                ),
            ]
        )
        llm = ChatOpenAI(
            model=self.model_name,
            timeout=30,
            max_retries=1,
            max_tokens=240,
        )
        messages = prompt.format_messages(question=question, context=context)
        first_token_ms: Optional[float] = None
        stream_start = time.perf_counter()
        if on_token is None:
            return (llm.invoke(messages).content or "").strip(), first_token_ms

        parts = []
        for chunk in llm.stream(messages):
            content = getattr(chunk, "content", "")
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text") or "")
                    for item in content
                    if isinstance(item, dict)
                )
            token = str(content or "")
            if not token:
                continue
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - stream_start) * 1000.0
            parts.append(token)
            on_token(token)
        return "".join(parts).strip(), first_token_ms

    @traceable(name="LGController.respond", run_type="chain")
    def respond(
        self,
        question: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        self.turn_index += 1
        started = time.perf_counter()
        question = (question or "").strip()
        logger.info("[RAG] Q%d: %s", self.turn_index, question)

        if _asks_private_question(question):
            if on_token is not None:
                on_token(REFUSAL)
            validation = validate_answer(REFUSAL, 0)
            return {
                "answer": REFUSAL,
                "sources": [],
                "footnotes": {},
                "source_freshness": _source_freshness([]),
                "validation": validation,
                "trace": {"controller": "lg_canonical_rag_v3", "refusal": True},
            }

        try:
            retrieval, web_payload = self._retrieve(question)
            evidence = retrieval.evidence
            if _asks_current_employer(question):
                evidence = [
                    item for item in evidence if item.source_type == "public_profile"
                ][:1]
                answer = "I’m currently at Lime."
                first_token_ms = None
                if on_token is not None:
                    on_token(answer)
            elif _asks_for_unsupported_lime_detail(question):
                evidence = []
                answer = (
                    "My approved resume and public profile do not provide that "
                    "Lime detail, so I won’t guess."
                )
                first_token_ms = None
                if on_token is not None:
                    on_token(answer)
            elif not evidence:
                answer = (
                    "I don’t have enough approved public evidence to answer that "
                    "confidently, so I’d rather leave the detail unstated than guess."
                )
                first_token_ms = None
                if on_token is not None:
                    on_token(answer)
            else:
                answer, first_token_ms = self._compose(question, evidence, on_token)

            sources = sources_from_evidence(evidence)
            footnotes = footnotes_from_evidence(evidence)
            validation = validate_answer(answer, len(sources))
            latency_ms = (time.perf_counter() - started) * 1000.0
            context = evidence_context(evidence)

            run_tree = get_current_run_tree()
            run_id = str(run_tree.id) if run_tree is not None else None
            trace = {
                "plan": "single-pass hybrid RAG with evidence-gated web fallback",
                "retrieval": retrieval.to_dict(include_content=False),
                "web_error": web_payload.get("error") if web_payload else None,
                "local_context_preview": (
                    context if self.include_context_in_trace else context[:800]
                ),
                "run_id": run_id,
                "controller": "lg_canonical_rag_v3",
                "model": self.model_name,
                "latency_ms": latency_ms,
                "first_token_ms": first_token_ms,
                "validation": validation,
            }
            self._last_trace = trace

            if POST_FEEDBACK_ENABLED:
                maybe_post_feedback_async(
                    run_id,
                    question,
                    answer,
                    context,
                    footnotes,
                    reference=None,
                    latency_ms=latency_ms,
                )

            return {
                "answer": answer,
                "sources": sources,
                "footnotes": footnotes,
                "source_freshness": _source_freshness(evidence),
                "validation": validation,
                "trace": trace,
            }
        except Exception as exc:
            logger.exception("[RAG] Pipeline failed: %s", exc)
            return {
                "answer": "I hit a retrieval error before I could form a grounded answer.",
                "sources": [],
                "footnotes": {},
                "source_freshness": _source_freshness([]),
                "validation": validate_answer("", 0),
                "trace": {
                    "controller": "lg_canonical_rag_v3",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test the canonical RAG controller.")
    parser.add_argument("question", help="Professional question to pose to the agent")
    parser.add_argument("--model", default=None, help="Optional generation model override")
    args = parser.parse_args()

    controller = LGController(model_name=args.model)
    result = controller.respond(args.question)
    print(json.dumps(result, indent=2, ensure_ascii=False))
