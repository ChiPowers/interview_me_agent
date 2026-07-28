# app/eval/run_openevals_custom.py
from __future__ import annotations
import os

from dotenv import load_dotenv
from langsmith import Client
from app.eval.evaluators import EvalInput, default_eval_suite
from ..agent.lg_controller import LGController

load_dotenv()
# ---- Config (env overrides) ----
PROJECT = os.getenv("LANGCHAIN_PROJECT", "interview-agent-bot")
DATASET_NAME = os.getenv("LS_DATASET_NAME", "Agent QAS")  # exact dataset name in LangSmith
EXPERIMENT_PREFIX = os.getenv("LS_EXPERIMENT_PREFIX", "interview-agent")
EVAL_API_KEY = os.getenv("LANGSMITH_EVAL_API_KEY") or os.getenv("LANGSMITH_API_KEY")
ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Ensure the experiment shows under your project
os.environ["LANGCHAIN_PROJECT"] = PROJECT
if EVAL_API_KEY:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGSMITH_API_KEY"] = EVAL_API_KEY

client = Client(api_key=EVAL_API_KEY, api_url=ENDPOINT) if EVAL_API_KEY else None

# ----- Your application target (what you want to evaluate) -----
_controller = None
def _controller_singleton():
    global _controller
    if _controller is None:
        _controller = LGController(include_context_in_trace=True)
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
        "sources": out.get("sources") or [],
        "footnotes": out.get("footnotes") or {},
        "trace": out.get("trace") or {},
    }

def _coerce_score(v):
    try:
        return None if v is None else float(v)
    except Exception:
        return None


def eval_input_from_run(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
    *,
    run=None,
) -> EvalInput:
    """Reconstruct the full source contract from a LangSmith target result."""
    q = inputs.get("question") or inputs.get("input") or ""
    a = outputs.get("answer") or outputs.get("output") or ""
    ref = (
        (reference_outputs or {}).get("answer")
        or (reference_outputs or {}).get("reference")
    )
    extra = dict(outputs or {})
    try:
        run_outputs = (run and getattr(run, "outputs", None)) or {}
        extra.update(run_outputs)
    except Exception:
        pass

    trace = extra.get("trace") or {}
    retrieval = trace.get("retrieval") or {}
    return EvalInput(
        question=q,
        answer=a,
        context=trace.get("local_context_preview", "") or "",
        footnotes=extra.get("footnotes") or {},
        reference=ref,
        sources=extra.get("sources") or [],
        retrieved_evidence_ids=[
            item["id"]
            for item in retrieval.get("evidence") or []
            if isinstance(item, dict) and item.get("id")
        ],
        abstained=bool(trace.get("refusal")),
    )


def chivon_eval_adapter(inputs: dict, outputs: dict, reference_outputs: dict, *, run=None, example=None):
    """
    Wrap app.eval.evaluators.default_eval_suite -> list of dicts that LangSmith accepts.
    """
    ei = eval_input_from_run(
        inputs,
        outputs,
        reference_outputs,
        run=run,
    )
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
    if client is None:
        raise SystemExit("Set LANGSMITH_EVAL_API_KEY or LANGSMITH_API_KEY first.")
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
