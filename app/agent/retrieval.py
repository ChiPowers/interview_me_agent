"""Single-pass hybrid retrieval with deterministic evidence contracts."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.services.settings import LOCAL_CANDIDATE_K, LOCAL_CONTEXT_K
from app.services.vectorstore import load_faiss_or_none

from .rag_types import Evidence, RetrievalResult

_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9+#.-]{1,}")
_STOPWORDS = {
    "about", "after", "also", "been", "from", "have", "into", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "through",
    "what", "when", "where", "which", "with", "would", "your", "you",
}
_FRESHNESS_TERMS = {
    "current", "currently", "latest", "new", "news", "now", "recent", "today",
    "this month", "this year", "updated",
}


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if token.lower() not in _STOPWORDS
    ]


def question_requires_freshness(question: str) -> bool:
    low = (question or "").lower()
    return any(term in low for term in _FRESHNESS_TERMS)


def _doc_key(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    stable = "|".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("page", "")),
            str(metadata.get("section", "")),
            (getattr(doc, "page_content", "") or "")[:240],
        ]
    )
    return hashlib.sha1(stable.encode("utf-8")).hexdigest()


def bm25_rank(
    query: str,
    docs: Sequence[Any],
    *,
    limit: int = LOCAL_CANDIDATE_K,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[Any, float]]:
    """Small in-process BM25 implementation; the corpus is only a few dozen chunks."""
    query_terms = tokenize(query)
    if not query_terms or not docs:
        return []

    tokenized = [tokenize(getattr(doc, "page_content", "")) for doc in docs]
    avg_len = sum(len(tokens) for tokens in tokenized) / max(1, len(tokenized))
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))

    scored: list[tuple[Any, float]] = []
    n_docs = len(docs)
    for doc, tokens in zip(docs, tokenized):
        counts = Counter(tokens)
        length = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * length / max(avg_len, 1))
            score += idf * ((tf * (k1 + 1)) / denom)
        if score > 0:
            scored.append((doc, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def _all_documents(store: Any) -> list[Any]:
    docstore = getattr(store, "docstore", None)
    mapping = getattr(docstore, "_dict", {}) if docstore is not None else {}
    return list(mapping.values())


def _semantic_rank(store: Any, query: str, limit: int) -> list[tuple[Any, float]]:
    try:
        return list(store.similarity_search_with_score(query, k=limit))
    except Exception:
        return []


def _fuse_rankings(
    rankings: Iterable[tuple[str, Sequence[tuple[Any, float]]]],
    *,
    rrf_k: int = 60,
) -> tuple[dict[str, Any], dict[str, float], dict[str, set[str]]]:
    documents: dict[str, Any] = {}
    fused: defaultdict[str, float] = defaultdict(float)
    rankers: defaultdict[str, set[str]] = defaultdict(set)
    for ranker, items in rankings:
        for rank, (doc, _raw_score) in enumerate(items, start=1):
            key = _doc_key(doc)
            documents[key] = doc
            fused[key] += 1.0 / (rrf_k + rank)
            rankers[key].add(ranker)
    return documents, dict(fused), dict(rankers)


def _select_diverse(
    ordered_keys: Sequence[str],
    documents: dict[str, Any],
    *,
    limit: int,
) -> list[str]:
    selected: list[str] = []
    group_counts: Counter[tuple[str, str, str]] = Counter()
    for key in ordered_keys:
        metadata = getattr(documents[key], "metadata", {}) or {}
        group = (
            Path(str(metadata.get("source", "unknown"))).name,
            str(metadata.get("employer", "")),
            str(metadata.get("section", "")),
        )
        if group_counts[group] >= 2:
            continue
        selected.append(key)
        group_counts[group] += 1
        if len(selected) >= limit:
            break
    return selected


def _named_employers(
    question: str,
    documents: dict[str, Any],
) -> set[str]:
    """Resolve employer names from indexed metadata, not a hard-coded company list."""
    low = (question or "").lower()
    employers = {
        str((getattr(doc, "metadata", {}) or {}).get("employer", "")).strip()
        for doc in documents.values()
    }
    return {
        employer
        for employer in employers
        if employer and employer.lower() in low
    }


def _confidence(
    question: str,
    evidence_keys: Sequence[str],
    documents: dict[str, Any],
    rankers: dict[str, set[str]],
) -> tuple[str, list[str]]:
    if not evidence_keys:
        return "low", ["no_local_evidence"]
    top_key = evidence_keys[0]
    top_rankers = rankers.get(top_key, set())
    query_terms = set(tokenize(question))
    top_terms = set(tokenize(getattr(documents[top_key], "page_content", "")))
    overlap = len(query_terms & top_terms) / max(1, len(query_terms))
    reasons = [f"top_query_overlap={overlap:.2f}"]
    if len(top_rankers) >= 2 and overlap >= 0.2:
        return "high", reasons + ["semantic_lexical_agreement"]
    if len(top_rankers) >= 2 or overlap >= 0.15:
        return "medium", reasons
    return "low", reasons + ["weak_ranker_agreement"]


def retrieve_hybrid(
    question: str,
    *,
    alternate_queries: Sequence[str] | None = None,
    store: Any | None = None,
    candidate_k: int = LOCAL_CANDIDATE_K,
    context_k: int = LOCAL_CONTEXT_K,
) -> RetrievalResult:
    store = store or load_faiss_or_none()
    if store is None:
        return RetrievalResult(
            evidence=[],
            confidence="low",
            reasons=["index_unavailable"],
        )

    docs = _all_documents(store)
    queries = [question] + [q for q in (alternate_queries or []) if q.strip()]
    rankings: list[tuple[str, Sequence[tuple[Any, float]]]] = []
    for query_idx, query in enumerate(queries):
        rankings.append(
            (f"semantic:{query_idx}", _semantic_rank(store, query, candidate_k))
        )
        rankings.append(
            (f"lexical:{query_idx}", bm25_rank(query, docs, limit=candidate_k))
        )

    documents, fused, rankers = _fuse_rankings(rankings)
    named_employers = _named_employers(question, documents)
    ordered = sorted(
        fused,
        key=lambda key: (
            str(
                (getattr(documents[key], "metadata", {}) or {}).get(
                    "employer", ""
                )
            )
            in named_employers,
            fused[key],
        ),
        reverse=True,
    )
    selected_keys = _select_diverse(
        ordered,
        documents,
        limit=min(context_k, 4),
    )
    confidence, reasons = _confidence(question, selected_keys, documents, rankers)
    if named_employers:
        matching_employers = {
            str(
                (getattr(documents[key], "metadata", {}) or {}).get(
                    "employer", ""
                )
            )
            for key in selected_keys
        } & named_employers
        if matching_employers and confidence == "low":
            confidence = "medium"
            reasons.append("explicit_employer_match")
        reasons.append(
            "employer_metadata_boost=" + ",".join(sorted(named_employers))
        )

    evidence: list[Evidence] = []
    for index, key in enumerate(selected_keys, start=1):
        doc = documents[key]
        metadata = dict(getattr(doc, "metadata", {}) or {})
        source_type = metadata.get("source_type", "local_pdf")
        source = str(metadata.get("source", "local.pdf"))
        evidence.append(
            Evidence(
                id=f"E{index}",
                content=(getattr(doc, "page_content", "") or "").strip(),
                label=metadata.get("label") or Path(source).name,
                source_type=source_type,
                score=round(fused[key], 6),
                url=metadata.get("url"),
                path=None if source_type == "public_profile" else Path(source).name,
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key
                    in {
                        "page_number",
                        "section",
                        "employer",
                        "topics",
                        "verified_at",
                    }
                    and value not in (None, "")
                },
            )
        )
    return RetrievalResult(
        evidence=evidence,
        confidence=confidence,
        reasons=reasons,
        query_rewritten=bool(alternate_queries),
    )


def evidence_context(evidence: Sequence[Evidence]) -> str:
    blocks = []
    for item in evidence:
        blocks.append(f"{item.id} | {item.label}\n{item.content}")
    return "\n\n---\n\n".join(blocks)


def sources_from_evidence(evidence: Sequence[Evidence]) -> list[dict[str, Any]]:
    return [item.to_dict(include_content=False) for item in evidence]


def footnotes_from_evidence(
    evidence: Sequence[Evidence],
) -> dict[int, dict[str, str]]:
    footnotes: dict[int, dict[str, str]] = {}
    for index, item in enumerate(evidence, start=1):
        payload = {"title": item.label}
        if item.url:
            payload["url"] = item.url
        elif item.path:
            page = item.metadata.get("page_number")
            suffix = f"#page={page}" if page else ""
            payload["path"] = f"local://{Path(item.path).name}{suffix}"
        footnotes[index] = payload
    return footnotes
