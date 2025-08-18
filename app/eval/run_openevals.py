# app/eval/run_openevals.py
from __future__ import annotations
import os
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

# --- LangSmith + OpenEvals imports
from langsmith import Client, wrappers
from openevals.llm import create_llm_as_judge
try:
    from openevals.prompts import (
        CORRECTNESS_PROMPT,
        RELEVANCE_PROMPT,
        CONCISENESS_PROMPT,
        FAITHFULNESS_PROMPT,
        INSTRUCTION_FOLLOWING_PROMPT,
    )
except Exception:
    # Fallback prompts if your openevals version names differ
    CORRECTNESS_PROMPT = "Judge if OUTPUT answers INPUT exactly as the REFERENCE does.\nReturn:\nSCORE: 0.0-1.0\nREASON: ..."
    RELEVANCE_PROMPT = "Judge how well OUTPUT addresses INPUT (on-topic, specific).\nReturn:\nSCORE: 0.0-1.0\nREASON: ..."
    CONCISENESS_PROMPT = "Judge if OUTPUT is concise (≤~90 words), non-redundant.\nReturn:\nSCORE: 0.0-1.0\nREASON: ..."
    FAITHFULNESS_PROMPT = "Judge if OUTPUT's claims are supported by CONTEXT/FOOTNOTES.\nReturn:\nSCORE: 0.0-1.0\nREASON: ..."
    INSTRUCTION_FOLLOWING_PROMPT = (
        "Judge whether OUTPUT follows constraints: ≤3 sentences, ≤~90 words, "
        "professional scope only. Return:\nSCORE: 0.0-1.0\nREASON: ..."
    )

# --- Optional: wrap OpenAI for tracing direct OpenAI calls in this script
from openai import OpenAI
openai_client = wrappers.wrap_openai(OpenAI())

# --- Your agent (evaluated end-to-end)
from ..agent.lc_controller import LCController

# Config
DATASET_NAME = os.getenv("LS_DATASET_NAME", "Interview_Agent_QAS")
EXPERIMENT_PREFIX = os.getenv("LS_EXPERIMENT_PREFIX", "interview-agent")
PROJECT = os.getenv("LANGCHAIN_PROJECT", "interview-agent-bot")
# Judge model (OpenEvals supports "openai:MODEL" notation)
JUDGE_MODEL = os.getenv("LS_JUDGE_MODEL", "openai:gpt-4.1-nano")

# LangSmith client
client = Client(
    api_key=os.getenv("LANGSMITH_EVAL_API_KEY"),
    api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
)

# Single controller instance for all rows
_controller = None
def _controller_singleton():
    global _controller
    if _controller is None:
        _controller = LCController()
    return _controller

# ---- Target app logic to evaluate (inputs come from your dataset rows)
def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Given {'question': ...} return {'answer': ...} using your real agent."""
    question = (inputs.get("question") or "").strip()
    ctrl = _controller_singleton()
    out = ctrl.respond(question)
    # Return only fields you want evaluated
    return {
        "answer": (out.get("answer") or "").strip(),
        # Optional pass-through for custom evaluators later:
        "context": out.get("trace", {}).get("local_context_preview", ""),
        "footnotes": out.get("footnotes") or {},
    }

# ---- Helper: build OpenEvals-style evaluator from a prompt
def make_llm_judge(prompt: str, feedback_key: str):
    judge = create_llm_as_judge(
        prompt=prompt,
        model=JUDGE_MODEL,
        feedback_key=feedback_key,  # metric name in LangSmith
    )
    def _fn(inputs: dict, outputs: dict, reference_outputs: dict):
        # OpenEvals expects these 3; not all prompts need the reference
        return judge(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
        )
    return _fn

# ---- Evaluators (you can add/remove freely)
evaluators = [
    make_llm_judge(CORRECTNESS_PROMPT, "correctness"),
    make_llm_judge(RELEVANCE_PROMPT, "relevance"),
    make_llm_judge(CONCISENESS_PROMPT, "conciseness"),
    make_llm_judge(FAITHFULNESS_PROMPT, "faithfulness"),
    make_llm_judge(INSTRUCTION_FOLLOWING_PROMPT, "instruction_following"),
]

if __name__ == "__main__":
    print(f"Project: {PROJECT}")
    print(f'Dataset: {DATASET_NAME}')
    print(f"Experiment prefix: {EXPERIMENT_PREFIX}")

    results = client.evaluate(
        target,
        data=DATASET_NAME,               # <- exact dataset name in LangSmith (no quotes)
        evaluators=evaluators,
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=int(os.getenv("LS_MAX_CONCURRENCY", "2")),
        # NOTE: no project_name=... here (older client doesn’t support it)
    )

    # Try printing the experiment URL if available
    try:
        print("Experiment URL:", results.get("url"))
    except Exception:
        pass
