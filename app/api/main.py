from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Any, Dict, Generator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.services.ingest_index import ensure_index
from app.agent.lg_controller import LGController

load_dotenv()

logger = logging.getLogger("interview_agent.api")
logging.basicConfig(level="INFO")

app = FastAPI(title="Interview Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_controller = LGController()
_index_ready = False


def _ensure_index_once() -> None:
    global _index_ready
    if _index_ready:
        return
    ensure_index()
    _index_ready = True


class ChatRequest(BaseModel):
    question: str


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (
        "<html><body style='font-family:sans-serif;max-width:760px;margin:2rem auto;'>"
        "<h2>Interview Agent API</h2>"
        "<p>Use <code>POST /chat</code> for JSON or <code>GET /chat/stream?question=...</code> for SSE.</p>"
        "</body></html>"
    )


@app.post("/chat")
def chat(req: ChatRequest) -> JSONResponse:
    _ensure_index_once()
    result = _controller.respond(req.question.strip())
    return JSONResponse(result)


def _sse_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_chat(question: str) -> Generator[str, None, None]:
    _ensure_index_once()
    q: "queue.Queue[tuple[str, Any]]" = queue.Queue()

    def on_token(tok: str) -> None:
        q.put(("token", tok))

    def run() -> None:
        try:
            out = _controller.respond(question.strip(), on_token=on_token)
            q.put(("final", out))
        except Exception as exc:  # pragma: no cover
            q.put(("error", {"message": str(exc)}))
        finally:
            q.put(("done", None))

    threading.Thread(target=run, daemon=True).start()

    while True:
        kind, payload = q.get()
        if kind == "done":
            yield _sse_event("done", {"ok": True})
            break
        yield _sse_event(kind, payload)


@app.get("/chat/stream")
def chat_stream(question: str = Query(..., min_length=1)) -> StreamingResponse:
    return StreamingResponse(_stream_chat(question), media_type="text/event-stream")

