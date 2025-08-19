# app/eval/push_dataset.py
from __future__ import annotations
import os, yaml
from pathlib import Path
from typing import Dict, Any, List
from langsmith import Client

DATASET_NAME = os.getenv("LS_DATASET", "Agent QAS")
DATASET_DESC = "Golden questions for Chivon interview agent (short, 1P voice)."

def load_qas(path: Path) -> List[Dict[str, Any]]:
    """
    Accepts either:
      questions:
        - q: "..."
          a: "..."
        - ...
    or a top-level list:
      - q: "..."
        a: "..."
    Returns a list of {"q": str, "a": Optional[str]}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return []

    # Normalize to a list of rows
    if isinstance(data, dict) and "questions" in data:
        rows = data["questions"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(
            f"Unrecognized YAML structure in {path}. "
            "Expected either a top-level 'questions:' key or a list of items."
        )

    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows, start=1):
        if not isinstance(r, dict):
            # Skip and warn if someone put a stray string or malformed item
            print(f"[load_qas] Skipping non-dict item at index {i}: {r!r}")
            continue
        q = r.get("q") or r.get("question")
        a = r.get("a") or r.get("answer")
        if not q or not isinstance(q, str):
            print(f"[load_qas] Skipping row {i}: missing 'q' field. Row: {r!r}")
            continue
        out.append({"q": q.strip(), "a": (a.strip() if isinstance(a, str) and a else None)})
    return out

def main():
    repo = Path(__file__).resolve().parents[2]  # repo/
    qas_path = Path(os.getenv("QAS_PATH", repo / "app"/ "eval" / "qas.yaml"))
    print(f"Reading: {qas_path}")

    examples = load_qas(qas_path)
    print(f"Loaded {len(examples)} examples")

    client = Client()

    # Create or fetch dataset
    try:
        ds = client.create_dataset(dataset_name=DATASET_NAME, description=DATASET_DESC)
    except Exception:
        ds = client.read_dataset(dataset_name=DATASET_NAME)
    # print(f"Dataset: {DATASET_NAME} ({ds['id']})")

    # # Index existing examples by question text to avoid dupes
    # existing = {
    #     (e["inputs"] or {}).get("question"): e
    #     for e in client.list_examples(dataset_id=ds["id"])
    # }

    created = 0
    for ex in examples:
        q, a = ex["q"], ex.get("a")
        # if q in existing:
        #     continue
        client.create_example(
            inputs={"question": q},
            outputs={"reference": a} if a else None,
            dataset_id="aab76a15-278e-4cce-a24d-f801679dd715",
        )
        created += 1

    print(f"Created {created} new examples (skipped {len(examples) - created} existing).")
    print("Done.")

if __name__ == "__main__":
    main()
