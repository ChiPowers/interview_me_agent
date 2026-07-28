# app/eval/evaluators.py
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.services.settings import JUDGE_MODEL


def combined_eval_json(inp: EvalInput, latency_ms: float | None = None) -> dict:
    """
    Run the default suite and return a DICT of metrics:
    {
      "relevance": {"score": 1.0, "comment": "..."},
      "faithfulness": {"score": 0.8, "comment": "..."},
      ...
    }
    NOTE: returns a Python dict (NOT a JSON string).
    """
    metrics = default_eval_suite(inp, latency_ms=latency_ms)
    out = {}
    for m in metrics:
        # accept both our LLM-based & rule-based format
        key = m.get("name") or m.get("key") or "metric"
        out[key] = {
            "score": m.get("score"),
            "comment": m.get("reason") or m.get("comment"),
        }
    return out


@dataclass
class EvalInput:
    question: str
    answer: str
    context: str           # any retrieved local snippets you want to pass (can be empty)
    footnotes: dict        # your structured footnotes dict {idx: {title, url|path}}
    reference: Optional[str] = None   # gold answer (optional)
    sources: List[Dict[str, Any]] | None = None
    retrieved_evidence_ids: List[str] | None = None
    abstained: bool = False


def _make_judge(model: Optional[str] = None) -> ChatOpenAI:
    """
    Central LLM factory for judge models so we can enforce sane defaults
    and avoid long hangs during eval runs.
    """
    return ChatOpenAI(
        model=model or JUDGE_MODEL,
        timeout=20,      # seconds; prevent stalls
        max_retries=1,   # strict during evals
    )


def _llm_judge(llm: ChatOpenAI, system: str, user: str) -> Dict[str, Any]:
    """Call an LLM judge and parse out a score in [0,1] + rationale."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "{x}")
    ])
    out = llm.invoke(prompt.format_messages(x=user))
    text = (out.content or "").strip()

    # Expected pattern: SCORE: <0-1>\nREASON: ...
    score = None
    m = re.search(r"SCORE\s*:\s*([01](?:\.\d+)?)", text, re.I)
    if m:
        try:
            score = float(m.group(1))
            score = max(0.0, min(1.0, score))
        except Exception:
            score = None

    # fallback: try to find a percentage
    if score is None:
        m = re.search(r"(\d{1,3})\s*%", text)
        if m:
            try:
                score = max(0.0, min(1.0, float(m.group(1)) / 100.0))
            except Exception:
                score = None

    return {
        "score": score if score is not None else 0.0,
        "reason": text
    }


# ---------------------------- LLM-based evaluators ----------------------------

def eval_relevance(inp: EvalInput, model: str = None) -> Dict[str, Any]:
    """Is the answer relevant to the question and context?"""
    llm = _make_judge(model)
    system = (
        "You are an evaluation model. Assess if the ANSWER directly and correctly addresses the QUESTION, "
        "appropriately using CONTEXT when helpful. Output:\n"
        "SCORE: 0.0-1.0\nREASON: short justification."
    )
    user = f"QUESTION:\n{inp.question}\n\nANSWER:\n{inp.answer}\n\nCONTEXT:\n{inp.context}"
    return {"name": "relevance", **_llm_judge(llm, system, user)}


def eval_conciseness(inp: EvalInput, model: str = None) -> Dict[str, Any]:
    """Is the answer direct and normally within the 60–120 word product target?"""
    llm = _make_judge(model)
    system = (
        "Evaluate the conciseness of the ANSWER. Penalize verbosity or repetition. "
        "Prefer 60–120 words for substantive questions and allow shorter single-fact answers.\n"
        "Output:\nSCORE: 0.0-1.0\nREASON: short justification."
    )
    user = f"QUESTION:\n{inp.question}\n\nANSWER:\n{inp.answer}"
    return {"name": "conciseness", **_llm_judge(llm, system, user)}


def eval_completeness(inp: EvalInput, model: str = None) -> Dict[str, Any]:
    """Does the answer cover the question's scope (including follow-ups implied)?"""
    llm = _make_judge(model)
    system = (
        "Judge whether the ANSWER sufficiently addresses all parts of the QUESTION. "
        "If a reference (gold) is provided, use it to guide expectations.\n"
        "Output:\nSCORE: 0.0-1.0\nREASON: short justification."
    )
    ref = f"\n\nREFERENCE:\n{inp.reference}" if inp.reference else ""
    user = f"QUESTION:\n{inp.question}\n\nANSWER:\n{inp.answer}{ref}"
    return {"name": "completeness", **_llm_judge(llm, system, user)}


def eval_faithfulness(inp: EvalInput, model: str = None) -> Dict[str, Any]:
    """Faithfulness (grounding): are factual claims supported by CONTEXT or FOOTNOTES?"""
    if inp.abstained:
        return {
            "name": "faithfulness",
            "score": 1.0,
            "reason": "Deterministic privacy abstention; no personal factual claim was made.",
        }
    llm = _make_judge(model)
    fnotes = json.dumps(inp.footnotes, ensure_ascii=False) if inp.footnotes else "{}"
    system = (
        "Judge if factual claims in the ANSWER are supported by the provided CONTEXT and/or FOOTNOTES. "
        "Penalize unsupported claims or contradictions.\n"
        "Output:\nSCORE: 0.0-1.0\nREASON: short justification."
    )
    user = f"ANSWER:\n{inp.answer}\n\nCONTEXT:\n{inp.context}\n\nFOOTNOTES:\n{fnotes}"
    return {"name": "faithfulness", **_llm_judge(llm, system, user)}


def eval_style_tone(inp: EvalInput, model: str = None) -> Dict[str, Any]:
    """Does the answer sound like Chivon: first-person, professional, interview-appropriate?"""
    llm = _make_judge(model)
    system = (
        "Evaluate style and tone: first-person, warm expert, practical, confident without overclaiming, "
        "evaluation-first, lightly conversational, and professional. "
        "Penalize personal-life disclosures or unprofessional language.\n"
        "Output:\nSCORE: 0.0-1.0\nREASON: short justification."
    )
    user = f"ANSWER:\n{inp.answer}"
    return {"name": "style_tone", **_llm_judge(llm, system, user)}


def eval_instruction_following(inp: EvalInput, model: str = None) -> Dict[str, Any]:
    """Did the answer follow the canonical product constraints?"""
    llm = _make_judge(model)
    system = (
        "Evaluate if the ANSWER follows instructions: 2–4 sentences, normally 60–120 words, "
        "professional scope, direct opening, and no unsupported claims. A simple fact may be shorter. "
        "Output:\nSCORE: 0.0-1.0\nREASON: short justification."
    )
    user = f"ANSWER:\n{inp.answer}"
    return {"name": "instruction_following", **_llm_judge(llm, system, user)}


# ---------------------------- Rule-based helpers ----------------------------

def eval_length_rule(inp: EvalInput, max_words: int = 120) -> Dict[str, Any]:
    words = re.findall(r"\b\w+\b", inp.answer)
    score = 1.0 if len(words) <= max_words else max(0.0, 1.0 - (len(words) - max_words) / max(max_words, 1))
    return {"name": "length_rule", "score": round(score, 3), "reason": f"{len(words)} words (max {max_words})."}

def eval_source_rule(inp: EvalInput) -> Dict[str, Any]:
    """Every displayed source must correspond to preassigned retrieved evidence."""
    sources = inp.sources or []
    displayed_ids = {
        str(source.get("id"))
        for source in sources
        if isinstance(source, dict) and source.get("id")
    }
    if not displayed_ids and inp.footnotes:
        displayed_ids = {f"E{key}" for key in inp.footnotes}
    retrieved_ids = (
        displayed_ids
        if inp.retrieved_evidence_ids is None
        else {str(item) for item in inp.retrieved_evidence_ids}
    )
    missing = displayed_ids - retrieved_ids

    if not displayed_ids:
        reason = "No sources displayed; citation precision is vacuously satisfied."
    elif missing:
        reason = (
            "Displayed source IDs were not present in retrieved evidence: "
            + ", ".join(sorted(missing))
        )
    else:
        reason = f"All {len(displayed_ids)} displayed sources match retrieved evidence."
    return {
        "name": "deterministic_sources_rule",
        "score": 0.0 if missing else 1.0,
        "reason": reason,
    }

def eval_latency_ms(latency_ms: Optional[float]) -> Dict[str, Any]:
    """Pass observed latency (ms) to score perf targets."""
    if latency_ms is None:
        return {"name": "latency", "score": None, "reason": "No latency provided."}
    target = 5000  # 5s non-web target you specified
    score = 1.0 if latency_ms <= target else max(0.0, 1.0 - (latency_ms - target) / target)
    return {"name": "latency", "score": round(score, 3), "reason": f"{int(latency_ms)} ms (target ≤ {target} ms)."}


# ---------------------------- Bundles ----------------------------

def default_eval_suite(inp: EvalInput, latency_ms: Optional[float] = None) -> List[Dict[str, Any]]:
    """Run a reasonable default suite."""
    out = [
        eval_relevance(inp),
        eval_faithfulness(inp),
        eval_completeness(inp),
        eval_conciseness(inp),
        eval_style_tone(inp),
        eval_instruction_following(inp),
        eval_length_rule(inp),
        eval_source_rule(inp),
    ]
    if latency_ms is not None:
        out.append(eval_latency_ms(latency_ms))
    return out
