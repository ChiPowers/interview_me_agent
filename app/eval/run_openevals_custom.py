# app/eval/run_openevals_custom.py
from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from ..agent.lc_controller import LCController
from .evaluators import EvalInput, default_eval_suite

# ---- Config (env overrides) ----
PROJECT = os.getenv("LANGCHAIN_PROJECT", "interview-agent-bot")
DATASET_NAME = os.getenv("LS_DATASET_NAME", "Agent QAS")  # exact dataset name in LangSmith
EXPERIMENT_PREFIX = os.getenv("LS_EXPERIMENT_PREFIX", "interview-agent")
EVAL_API_KEY = os.getenv("LANGSMITH_EVAL_API_KEY") or os.getenv("LANGSMITH_API_KEY")
ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Ensure the experiment shows under your project
os.environ["LANGCHAIN_PROJECT"] = PROJECT
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")  # optional
os.environ["LANGSMITH_API_KEY"] = EVAL_API_KEY or ""

client = Client(api_key=EVAL_API_KEY, api_url=ENDPOINT)

# ----- Your application target (what you want to evaluate) -----
_controller = None
def _controller_singleton():
    global _controller
    if _controller is None:
        _controller = LCController()
    return _controller

def target(inputs: dict) -> dict:
    """
    Must return a dict. We'll return answer + extras so evaluators can use them.
    inputs => {"question": "..."} (from your dataset)
    """
    q = inputs["question"]
    out = _controller_singleton().respond(q)
    # Keep answer minimal, but include context/footnotes for evaluators
    return {
        "answer": out.get("answer", ""),
        "context": (out.get("trace") or {}).get("local_context_preview", ""),
        "footnotes": out.get("footnotes") or {},
        "trace": out.get("trace") or {},
    }

# ----- Adapter: use your evaluators.py suite inside LangSmith evaluate() -----
# at top
from langsmith.evaluation.evaluator import EvaluationResult

# run_openevals_custom.py (or wherever your adapter lives)

def _coerce_score(v):
    try:
        return None if v is None else float(v)
    except Exception:
        return None

def chivon_eval_adapter(inputs: dict, outputs: dict, reference_outputs: dict, *, run=None, example=None):
    """
    Wrap app.eval.evaluators.default_eval_suite -> list of dicts that LangSmith accepts.
    """
    from app.eval.evaluators import EvalInput, default_eval_suite

    q = inputs.get("question") or inputs.get("input") or ""
    a = outputs.get("answer") or outputs.get("output") or ""
    ref = (reference_outputs or {}).get("answer") if reference_outputs else None

    # Optional: pull context/footnotes from run.outputs if your app recorded them
    ctx = ""
    footnotes = {}
    try:
        extra = (run and getattr(run, "outputs", None)) or {}
        ctx = (extra.get("trace") or {}).get("local_context_preview", "") or ""
        footnotes = extra.get("footnotes") or {}
    except Exception:
        pass

    ei = EvalInput(question=q, answer=a, context=ctx, footnotes=footnotes, reference=ref)
    metrics = default_eval_suite(ei, latency_ms=None)

    results = []
    for m in metrics:
        key = m.get("name") or "metric"
        score = _coerce_score(m.get("score"))
        comment = m.get("reason") or m.get("comment") or ""

        # If no numeric score, provide a categorical value instead
        if score is None:
            results.append({"key": key, "value": "n/a", "comment": comment})
        else:
            results.append({"key": key, "score": score, "comment": comment})

    return results

if __name__ == "__main__":
    print(f"Project: {PROJECT}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Experiment prefix: {EXPERIMENT_PREFIX}")

    results = client.evaluate(
        target,
        data=DATASET_NAME,                 # must match dataset name in LangSmith
        evaluators=[chivon_eval_adapter],  # <- your evaluators wrapped here
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=int(os.getenv("LS_MAX_CONCURRENCY", "4")),
    )

    # Best-effort print an Experiment URL if available
    try:
        print("Experiment URL:", results.get("url"))
    except Exception:
        pass
