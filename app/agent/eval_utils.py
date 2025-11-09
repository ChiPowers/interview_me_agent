from __future__ import annotations

import os
import threading
import random
from typing import Dict, Any

from langsmith import Client

from app.eval.evaluators import EvalInput, combined_eval_json

# LangSmith feedback toggles (consistent + permissive parsing)
_POST_FEEDBACK_RAW = os.getenv("POST_FEEDBACK", "0")
POST_FEEDBACK_ENABLED = _POST_FEEDBACK_RAW.lower() in ("1", "true", "yes", "on")
POST_FEEDBACK_SAMPLE_RATE = float(os.getenv("POST_FEEDBACK_SAMPLE_RATE", "0.25"))


def _post_feedback_worker(run_id: str, ei: EvalInput, latency_ms: float):
    """Background worker that normalizes evaluator outputs and posts metrics to LangSmith."""
    try:
        raw = combined_eval_json(ei, latency_ms=latency_ms)

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

            if isinstance(raw_obj, str):
                try:
                    import json
                    raw_parsed = json.loads(raw_obj)
                except Exception:
                    _add("eval", None, value=raw_obj)
                    return out
                raw_obj = raw_parsed

            if isinstance(raw_obj, dict):
                if "key" in raw_obj:
                    _add(raw_obj.get("key"), raw_obj.get("score"),
                         raw_obj.get("comment"), raw_obj.get("reason"), raw_obj.get("value"))
                else:
                    for k, v in raw_obj.items():
                        if isinstance(v, dict):
                            _add(k, v.get("score"), v.get("comment"), v.get("reason"), v.get("value"))
                        else:
                            _add(k, None, value=v)
                return out

            if isinstance(raw_obj, (list, tuple)):
                for item in raw_obj:
                    if isinstance(item, dict):
                        if "key" in item:
                            _add(item.get("key"), item.get("score"), item.get("comment"),
                                 item.get("reason"), item.get("value"))
                        else:
                            name = item.get("name") or item.get("metric") or "metric"
                            _add(name, item.get("score"), item.get("comment"),
                                 item.get("reason"), item.get("value"))
                    else:
                        key = getattr(item, "key", "metric")
                        score = getattr(item, "score", None)
                        comment = getattr(item, "comment", None)
                        reason = getattr(item, "reason", None)
                        value = getattr(item, "value", None)
                        _add(key, score, comment, reason, value)
                return out

            _add("eval", None, value=str(raw_obj))
            return out

        normalized = normalize(raw)

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
            except Exception as e:  # pragma: no cover - non-critical logging
                print(f"[eval] failed to post metric {m.get('key')}: {e}")

        print(f"[eval] posted {posted}/{len(normalized)} feedback items to run {run_id}")

    except Exception as e:  # pragma: no cover - background logging
        print(f"[eval] feedback post failed: {e}")


def maybe_post_feedback_async(
    run_id: str,
    question: str,
    answer: str,
    ctx: str,
    footnotes: Dict[int, Dict[str, str]],
    reference: str | None,
    latency_ms: float,
):
    """Sampled async poster shared by both controllers."""
    if not POST_FEEDBACK_ENABLED or not run_id:
        if POST_FEEDBACK_ENABLED and not run_id:
            print("[eval] skipping feedback: no run_id (is LANGCHAIN_TRACING_V2=true?)")
        return

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
