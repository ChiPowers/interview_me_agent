# app/eval/run_eval.py
from __future__ import annotations
import csv
import json
import logging
import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from langsmith import Client

from ..agent.lg_controller import LGController
from .evaluators import EvalInput, default_eval_suite

load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH_EVAL_API_KEY", "")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
PROJECT = os.getenv("LANGCHAIN_PROJECT", "evaluators")

langsmith_client = (
    Client(api_key=LANGSMITH_API_KEY, api_url=LANGCHAIN_ENDPOINT)
    if LANGSMITH_API_KEY
    else None
)

# ------------------------------- Config --------------------------------
REPO = Path(__file__).resolve().parents[2]
DATASET_YAML = Path(os.getenv("EVAL_DATASET", REPO / "app"/ "eval" / "qas.yaml"))
OUT_DIR = REPO / "eval_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POST_FEEDBACK = os.getenv("POST_FEEDBACK", "0") in ("1", "true", "yes")

logging.basicConfig(level=os.getenv("APP_LOG_LEVEL", "INFO"))
log = logging.getLogger("eval")
logging.getLogger("langsmith.client").setLevel(logging.ERROR)

# ---- Quiet noisy warnings (LangChain/Tavily deprecations, etc.) ----
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"langchain.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"langchain.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"openai.*")

# ----------------------------- Utilities -------------------------------
def load_golden(path: Path) -> List[Dict[str, Any]]:
    """
    YAML format:
    - q: "Question..."
      a: "Gold answer..."   # optional
    """
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    out = []
    for row in data:
        if not isinstance(row, dict):
            continue
        q = (row.get("q") or row.get("question") or "").strip()
        a = (row.get("a") or row.get("answer") or "").strip() or None
        if q:
            out.append({"q": q, "a": a})
    return out

def safe_default_eval_suite(ei: EvalInput, latency_ms: Optional[float]) -> List[Dict[str, Any]]:
    """Run default evals with hard error boundaries so a single bad call doesn't stop the run."""
    try:
        return default_eval_suite(ei, latency_ms=latency_ms)
    except Exception as e:
        log.error("Evaluator crash: %s", e)
        log.debug("Evaluator traceback:\n%s", traceback.format_exc())
        # Return minimal signal so the row is not lost
        return [{"name": "eval_error", "score": 0.0, "reason": f"{type(e).__name__}: {e}"}]


def eval_input_from_output(
    question: str,
    reference: Optional[str],
    output: Dict[str, Any],
) -> EvalInput:
    """Preserve the controller's retrieval/source contract for source alignment."""
    trace = output.get("trace") or {}
    retrieval = trace.get("retrieval") or {}
    return EvalInput(
        question=question,
        answer=(output.get("answer") or "").strip(),
        context=trace.get("local_context_preview") or "",
        footnotes=(
            output.get("footnotes")
            if isinstance(output.get("footnotes"), dict)
            else {}
        ),
        reference=reference,
        sources=output.get("sources") or [],
        retrieved_evidence_ids=[
            item["id"]
            for item in retrieval.get("evidence") or []
            if isinstance(item, dict) and item.get("id")
        ],
        abstained=bool(trace.get("refusal")),
    )


def make_langsmith_client() -> Optional[Client]:
    if not (POST_FEEDBACK and LANGSMITH_API_KEY):
        return None
    try:
        # NOTE: langsmith Client expects api_url in most recent versions
        return Client(api_key=LANGSMITH_API_KEY, api_url=LANGCHAIN_ENDPOINT)
    except TypeError:
        # Older versions use base_url; fallback
        return Client(api_key=LANGSMITH_API_KEY, base_url=LANGCHAIN_ENDPOINT)
    except Exception as e:
        log.warning("LangSmith client init failed: %s", e)
        return None

def post_feedback_batch(client: Client, rows: List[Dict[str, Any]]) -> None:
    """Attach feedback to runs if run_id present in each row."""
    if not client:
        return
    count = 0
    for r in rows:
        run_id = r.get("run_id")
        if not run_id:
            continue
        metrics = r.get("metrics") or []
        for m in metrics:
            name = m.get("name")
            score = m.get("score")
            if name is None or score is None:
                continue
            try:
                client.create_feedback(
                    run_id=run_id,
                    key=name,
                    score=float(score) if isinstance(score, (int, float)) else None,
                    comment=m.get("reason") or "",
                )
                count += 1
            except Exception as e:
                log.debug("Feedback post failed for %s: %s", name, e)
    log.info("Posted %d feedback items to LangSmith.", count)

# ------------------------------- Main ----------------------------------
def main():
    log.info("Dataset: %s", DATASET_YAML)
    rows = load_golden(DATASET_YAML)
    log.info("Loaded %d examples.", len(rows))

    # Controller (agent) – ensure it won’t trace unless you explicitly enabled it
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    controller = LGController(include_context_in_trace=True)

    # Progress UI
    results = []
    start = time.time()
    from tqdm import tqdm  # optional but nice; pip install tqdm
    iterator = tqdm(enumerate(rows, start=1), total=len(rows), ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

    for i, row in iterator:
        q = row["q"]
        ref = row.get("a")
        t0 = time.time()
        try:
            out = controller.respond(q)
        except Exception as e:
            log.error("Agent error on #%d: %s", i, e)
            out = {"answer": "", "footnotes": {}, "trace": {}, "run_id": None}

        latency_ms = (time.time() - t0) * 1000.0
        answer = (out.get("answer") or "").strip()
        footnotes = out.get("footnotes") or {}
        trace = out.get("trace") or {}
        run_id = trace.get("run_id")

        ei = eval_input_from_output(q, ref, out)
        metrics = safe_default_eval_suite(ei, latency_ms=latency_ms)

        if not run_id and langsmith_client:
            try:
                created = langsmith_client.create_run(
                    name="batch-eval-4o-mini",
                    run_type="chain",
                    project_name=PROJECT,
                    inputs={"question": q},
                    outputs={"answer": answer},
                    id=str(uuid4()),  # optional: let server assign if omitted
                    tags=["offline-eval"],
                    metadata={"eval_batch": True},
                )
                run_id = created.id
                print("Created LS run:", run_id)
            except Exception as e:
                print("WARN: could not create run in LangSmith:", e)
                run_id = None

        # Optionally attach all metric feedback
        if POST_FEEDBACK and run_id and langsmith_client:
            for m in metrics:
                try:
                    langsmith_client.create_feedback(
                        run_id=run_id,
                        key=m["name"],
                        score=m.get("score") if isinstance(m.get("score"), (int, float)) else None,
                        comment=(m.get("reason") or "")[:500],
                    )
                except Exception as e:
                    print(f"WARN: feedback post failed ({m['name']}):", e)

        row_out = {
            "id": i,
            "question": q,
            "answer": answer,
            "reference": ref,
            "latency_ms": round(latency_ms, 1),
            "metrics": metrics,
            "footnotes": footnotes,
            "retrieval": trace.get("retrieval", {}),
            "run_id": run_id,
        }
        results.append(row_out)

        log.info("[%d] %.1f ms | %s", i, row_out["latency_ms"], q[:70])

    # Outputs
    json_path = OUT_DIR / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("Wrote %s", json_path)

    metric_names = sorted({m["name"] for r in results for m in r["metrics"]})
    csv_path = OUT_DIR / "eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "latency_ms"] + metric_names)
        for r in results:
            rowm = {m["name"]: m["score"] for m in r["metrics"]}
            w.writerow([r["id"], r["question"], r["latency_ms"]] + [rowm.get(k) for k in metric_names])
    log.info("Wrote %s", csv_path)

    # Optional feedback to LangSmith (quiet if no key or disabled)
    feedback_client = make_langsmith_client()
    if feedback_client and POST_FEEDBACK:
        log.info("Feedback posting to LangSmith is ENABLED.")
        post_feedback_batch(feedback_client, results)
    else:
        log.info("Feedback posting DISABLED (set POST_FEEDBACK=1 and a valid LANGSMITH_API_KEY to enable).")

    log.info("Done in %.1fs", time.time() - start)

if __name__ == "__main__":
    main()
