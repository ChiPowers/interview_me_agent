"""Backward-compatible facade over the canonical deterministic RAG controller."""
from __future__ import annotations

from typing import Any, Dict

from .lg_controller import LGController


class GraphController:
    def __init__(self):
        self.controller = LGController()

    def respond(self, question: str) -> Dict[str, Any]:
        return self.controller.respond(question)
