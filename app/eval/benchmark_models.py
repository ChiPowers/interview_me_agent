"""Benchmark Luna vs Terra on the production controller and apply promotion gates."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from app.agent.lg_controller import LGController
from app.eval.evaluators import EvalInput, default_eval_suite
from app.services.settings import (
    GENERATION_CANDIDATE_MODEL,
    GENERATION_FALLBACK_MODEL,
)

CRITICAL_METRICS = ("relevance", "faithfulness", "completeness", "style_tone")
LUNA = GENERATION_CANDIDATE_MODEL
TERRA = GENERATION_FALLBACK_MODEL


def _load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return rows[:limit] if limit else rows


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def run_model(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    controller = LGController(model_name=model, include_context_in_trace=True)
    metric_values: dict[str, list[float]] = {}
    latencies = []
    cases = []
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        result = controller.respond(row["q"])
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        trace = result.get("trace") or {}
        retrieval = trace.get("retrieval") or {}
        eval_input = EvalInput(
            question=row["q"],
            answer=result.get("answer") or "",
            context=trace.get("local_context_preview", ""),
            footnotes=result.get("footnotes") or {},
            reference=row.get("a"),
            sources=result.get("sources") or [],
            retrieved_evidence_ids=[
                item["id"]
                for item in retrieval.get("evidence") or []
                if isinstance(item, dict) and item.get("id")
            ],
            abstained=bool(trace.get("refusal")),
        )
        metrics = default_eval_suite(eval_input, latency_ms=latency_ms)
        for metric in metrics:
            score = metric.get("score")
            if isinstance(score, (int, float)):
                metric_values.setdefault(metric["name"], []).append(float(score))
        cases.append(
            {
                "question": row["q"],
                "answer": result.get("answer"),
                "latency_ms": round(latency_ms, 1),
                "metrics": metrics,
            }
        )
        if index == 1 or index == len(rows) or index % 10 == 0:
            print(f"[{model}] {index}/{len(rows)} cases", file=sys.stderr)
    return {
        "model": model,
        "averages": {
            name: round(statistics.fmean(values), 4)
            for name, values in metric_values.items()
            if values
        },
        "p95_latency_ms": round(_percentile(latencies, 0.95), 1),
        "cases": cases,
    }


def choose_model(luna: dict[str, Any], terra: dict[str, Any]) -> dict[str, Any]:
    misses = []
    for metric in CRITICAL_METRICS:
        luna_score = luna["averages"].get(metric, 0.0)
        terra_score = terra["averages"].get(metric, 0.0)
        if luna_score < 0.90:
            misses.append(f"{metric} below 0.90")
        if terra_score - luna_score > 0.02:
            misses.append(f"{metric} trails Terra by more than 0.02")
    if luna["p95_latency_ms"] > 5000:
        misses.append("local p95 latency exceeds 5000 ms")
    return {
        "recommended_model": TERRA if misses else LUNA,
        "luna_passed": not misses,
        "reasons": misses or ["Luna met every quality and latency promotion gate."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark production RAG model candidates.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("qas.yaml"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = _load_rows(args.dataset, args.limit)
    luna = run_model(LUNA, rows)
    terra = run_model(TERRA, rows)
    report = {
        "dataset_size": len(rows),
        "decision": choose_model(luna, terra),
        "models": [luna, terra],
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        summary = {
            "output": str(args.output),
            "dataset_size": report["dataset_size"],
            "decision": report["decision"],
            "models": [
                {
                    "model": item["model"],
                    "averages": item["averages"],
                    "p95_latency_ms": item["p95_latency_ms"],
                }
                for item in report["models"]
            ],
        }
        print(json.dumps(summary, indent=2))
    else:
        print(text)


if __name__ == "__main__":
    main()
