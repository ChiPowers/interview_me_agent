"""Typed contracts shared by retrieval, composition, API, and evaluations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    id: str
    content: str
    label: str
    source_type: str
    score: float
    url: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_content: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_content:
            data.pop("content", None)
        return data


@dataclass
class RetrievalResult:
    evidence: List[Evidence]
    confidence: str
    reasons: List[str] = field(default_factory=list)
    web_fallback_used: bool = False
    query_rewritten: bool = False

    def to_dict(self, *, include_content: bool = False) -> Dict[str, Any]:
        return {
            "evidence": [
                item.to_dict(include_content=include_content) for item in self.evidence
            ],
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "web_fallback_used": self.web_fallback_used,
            "query_rewritten": self.query_rewritten,
        }
