# app/eval/run_eval.py
from __future__ import annotations
import os, time, json, csv, warnings, traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
from uuid import uuid4

from langsmith import Client

LS_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LS_API_KEY = os.getenv("LANGSMITH_API_KEY")
PROJECT = os.getenv("LANGCHAIN_PROJECT", "interview-agent-bot")

client = Client(api_key=LS_API_KEY, api_url=LS_ENDPOINT)

# ---- Minimal, controlled logging before imports that emit warnings ----
import logging
logging.basicConfig(level=os.getenv("APP_LOG_LEVEL", "INFO"))
log = logging.getLogger("eval")
logging.getLogger("langsmith.client").setLevel(logging.ERROR)

# ---- Quiet noisy warnings (LangChain/Tavily deprecations, etc.) ----
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"langchain.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"langchain.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"openai.*")

from ..agent.lc_controller import LCController
from .evaluators import EvalInput, default_eval_suite
from dotenv import load_dotenv

# Optional LangSmith client (only used for feedback posting)
try:
    from langsmith import Client
    LS_AVAILABLE = True
except Exception:
    LS_AVAILABLE = False

load_dotenv()

# ------------------------------- Config --------------------------------
REPO = Path(__file__).resolve().parents[2]
DATASET_YAML = Path(os.getenv("EVAL_DATASET", REPO / "app"/ "eval" / "qas.yaml"))
OUT_DIR = REPO / "eval_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POST_FEEDBACK = os.getenv("POST_FEEDBACK", "0") in ("1", "true", "yes")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

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

def make_langsmith_client() -> Optional[Client]:
    if not (LS_AVAILABLE and POST_FEEDBACK and LANGSMITH_API_KEY):
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
    controller = LCController()

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
        run_id = out.get("run_id")  # make sure LCController.respond() returns this

        local_ctx = trace.get("local_context_preview") or ""
        ei = EvalInput(
            question=q,
            answer=answer,
            context=local_ctx,
            footnotes=footnotes if isinstance(footnotes, dict) else {},
            reference=ref,
        )
        metrics = safe_default_eval_suite(ei, latency_ms=latency_ms)

        run_id = out.get("run_id")  # from LCController.respond if tracer worked
        if not run_id:
            try:
                created = client.create_run(
                    name="batch-eval",
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
        if POST_FEEDBACK and run_id:
            for m in metrics:
                try:
                    client.create_feedback(
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
            "trace_excerpt": trace.get("steps", [])[:3],
            "run_id": run_id,
        }
        results.append(row_out)

        # Compact console line
        try:
            score_summary = {m["name"]: m["score"] for m in metrics}
        except Exception:
            score_summary = {}
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
    client = make_langsmith_client()
    if client and POST_FEEDBACK:
        log.info("Feedback posting to LangSmith is ENABLED.")
        post_feedback_batch(client, results)
    else:
        log.info("Feedback posting DISABLED (set POST_FEEDBACK=1 and a valid LANGSMITH_API_KEY to enable).")

    log.info("Done in %.1fs", time.time() - start)

if __name__ == "__main__":
    main()
