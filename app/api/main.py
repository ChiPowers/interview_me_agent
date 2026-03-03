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
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Interview Agent</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#f7f8fb; margin:0; }
      .wrap { max-width: 860px; margin: 24px auto; padding: 0 16px; }
      .card { background: white; border:1px solid #e6e8ef; border-radius:14px; padding:16px; box-shadow:0 2px 10px rgba(0,0,0,.04); }
      h1 { font-size: 24px; margin: 0 0 8px; }
      .sub { color:#59627a; margin-bottom:14px; }
      .row { display:flex; gap:10px; }
      input { flex:1; padding:12px; border-radius:10px; border:1px solid #c9cfe0; font-size:15px; }
      button { padding:12px 16px; border-radius:10px; border:none; background:#1f5cff; color:white; font-weight:600; cursor:pointer; }
      button:disabled { opacity:.5; cursor:not-allowed; }
      #answer { white-space: pre-wrap; line-height: 1.45; margin-top:12px; min-height: 40px; }
      .meta { color:#6b7280; font-size:13px; margin-top:8px; }
      .error { color:#b42318; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <h1>Interview Agent</h1>
        <div class="sub">Ask about Chivon's background. Responses stream live.</div>
        <div class="row">
          <input id="q" placeholder="How does your background fit a principal applied scientist role?" />
          <button id="ask">Ask</button>
        </div>
        <div id="answer"></div>
        <div id="meta" class="meta"></div>
      </div>
    </div>
    <script>
      const input = document.getElementById("q");
      const btn = document.getElementById("ask");
      const answer = document.getElementById("answer");
      const meta = document.getElementById("meta");
      let es = null;

      function closeStream() {
        if (es) { es.close(); es = null; }
      }

      function ask() {
        const q = input.value.trim();
        if (!q) return;
        closeStream();
        answer.textContent = "";
        meta.textContent = "Streaming response...";
        meta.classList.remove("error");
        btn.disabled = true;

        es = new EventSource(`/chat/stream?question=${encodeURIComponent(q)}`);
        es.addEventListener("token", (evt) => {
          answer.textContent += JSON.parse(evt.data);
        });
        es.addEventListener("final", (evt) => {
          const payload = JSON.parse(evt.data);
          const trace = payload.trace || {};
          const latency = trace.latency_ms ? `${Math.round(trace.latency_ms)} ms` : "n/a";
          const tools = (trace.tool_events || []).length;
          meta.textContent = `Done • latency: ${latency} • tools: ${tools}`;
        });
        es.addEventListener("error", (evt) => {
          meta.textContent = "Stream error. Try again.";
          meta.classList.add("error");
          btn.disabled = false;
          closeStream();
        });
        es.addEventListener("done", () => {
          btn.disabled = false;
          closeStream();
        });
      }

      btn.addEventListener("click", ask);
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
    </script>
  </body>
</html>
"""


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
