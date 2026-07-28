"""Refresh the small, public-facts-only snapshot of Chivon's LinkedIn profile."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .settings import LINKEDIN_PROFILE_URL, PROFILE_SNAPSHOT_PATH
from .web_fetch import fetch_and_clean

logger = logging.getLogger("profile_snapshot")


def load_profile_snapshot(path: Path = PROFILE_SNAPSHOT_PATH) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _public_facts_from_text(text: str) -> list[dict[str, str]]:
    """Return only facts explicitly visible in the fetched public page."""
    low = (text or "").lower()
    if "chivon powers" not in low or "lime" not in low:
        return []
    return [{"field": "current_employer", "value": "Lime"}]


def refresh_profile_snapshot(
    path: Path = PROFILE_SNAPSHOT_PATH,
    url: str = LINKEDIN_PROFILE_URL,
    fetcher=fetch_and_clean,
) -> Dict[str, Any]:
    """
    Refresh public facts. A failed/blocked LinkedIn fetch never destroys the last
    known-good snapshot.
    """
    path = Path(path)
    previous = load_profile_snapshot(path)
    text = fetcher(url)
    facts = _public_facts_from_text(text)
    if not facts:
        logger.warning("LinkedIn refresh unavailable; retaining last successful snapshot.")
        return previous

    snapshot = {
        "source_type": "public_profile",
        "canonical_url": url,
        "profile_name": "Chivon Powers, PhD",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "facts": facts,
        "disclosure_policy": "public_facts_only",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return snapshot


def snapshot_as_text(snapshot: Dict[str, Any]) -> str:
    facts = snapshot.get("facts") or []
    fact_lines = []
    for fact in facts:
        if fact.get("field") == "current_employer" and fact.get("value"):
            fact_lines.append(
                f"Current professional chapter: Chivon Powers is currently at {fact['value']}."
            )
    return "\n".join(fact_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the canonical LinkedIn snapshot.")
    parser.add_argument("--refresh", action="store_true", help="Fetch and replace public facts")
    args = parser.parse_args()
    snapshot = (
        refresh_profile_snapshot() if args.refresh else load_profile_snapshot()
    )
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
