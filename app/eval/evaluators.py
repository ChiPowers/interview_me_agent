# app/eval/evaluators.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os, re, math, json
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


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


def _make_judge(model: Optional[str] = None) -> ChatOpenAI:
    """
    Central LLM factory for judge models so we can enforce sane defaults
    and avoid long hangs during eval runs.
    """
    return ChatOpenAI(
        model=model or os.getenv("OPENAI_JUDGE_MODEL", "gpt-5-nano-2025-08-07"),
        temperature=0,
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
    """Is the answer concise (≤ ~90 words) without missing key info?"""
    llm = _make_judge(model)
    system = (
        "Evaluate the conciseness of the ANSWER. Penalize verbosity or repetition. "
        "Prefer ≤ ~90 words if possible.\nOutput:\nSCORE: 0.0-1.0\nREASON: short justification."
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
        "Evaluate style and tone: first-person voice, professional, clear, interview-appropriate. "
        "Penalize personal-life disclosures or unprofessional language.\n"
        "Output:\nSCORE: 0.0-1.0\nREASON: short justification."
    )
    user = f"ANSWER:\n{inp.answer}"
    return {"name": "style_tone", **_llm_judge(llm, system, user)}


def eval_instruction_following(inp: EvalInput, model: str = None) -> Dict[str, Any]:
    """Did the answer follow constraints (≤3 sentences, professional-only, short)?"""
    llm = _make_judge(model)
    system = (
        "Evaluate if the ANSWER follows instructions: ≤3 sentences, ≤~90 words, professional scope (no personal life), "
        "and concise. Output:\nSCORE: 0.0-1.0\nREASON: short justification."
    )
    user = f"ANSWER:\n{inp.answer}"
    return {"name": "instruction_following", **_llm_judge(llm, system, user)}


# ---------------------------- Rule-based helpers ----------------------------

def eval_length_rule(inp: EvalInput, max_words: int = 90) -> Dict[str, Any]:
    words = re.findall(r"\b\w+\b", inp.answer)
    score = 1.0 if len(words) <= max_words else max(0.0, 1.0 - (len(words) - max_words) / max(max_words, 1))
    return {"name": "length_rule", "score": round(score, 3), "reason": f"{len(words)} words (max {max_words})."}

def eval_marker_rule(inp: EvalInput) -> Dict[str, Any]:
    """Optional: award small credit if answer includes footnote markers that match provided footnotes."""
    markers = re.findall(r"\[(\d+)\]", inp.answer)
    have = 0
    for m in markers:
        try:
            if int(m) in {int(k) for k in inp.footnotes.keys()}:
                have += 1
        except Exception:
            pass
    score = min(1.0, have / max(1, len(markers))) if markers else 0.0
    return {"name": "citation_markers_rule", "score": round(score, 3), "reason": f"{have}/{len(markers)} markers matched."}

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
        eval_marker_rule(inp),
    ]
    if latency_ms is not None:
        out.append(eval_latency_ms(latency_ms))
    return out
