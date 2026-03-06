from __future__ import annotations

import base64
import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.services.ingest_index import ensure_index
from app.agent.lg_controller import LGController

load_dotenv()

# Normalize LangSmith/LangChain tracing env vars so both old and new SDK versions activate.
import os as _os
if _os.getenv("LANGSMITH_API_KEY"):
    _os.environ.setdefault("LANGSMITH_TRACING", "true")
    _os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    _os.environ.setdefault("LANGCHAIN_PROJECT", "interview-me-agent")
if _os.getenv("LANGCHAIN_TRACING_V2") and not _os.getenv("LANGSMITH_TRACING"):
    _os.environ["LANGSMITH_TRACING"] = _os.getenv("LANGCHAIN_TRACING_V2", "")
if _os.getenv("LANGSMITH_TRACING") and not _os.getenv("LANGCHAIN_TRACING_V2"):
    _os.environ["LANGCHAIN_TRACING_V2"] = _os.getenv("LANGSMITH_TRACING", "")

# Embed logo as base64 so it works without static file serving
def _load_logo_b64() -> str:
    candidates = [
        Path(__file__).parent.parent / "static" / "logotat.png",
        Path(__file__).parent.parent.parent / "static" / "logotat.png",
    ]
    for p in candidates:
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    return ""

_LOGO_B64 = _load_logo_b64()

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
    logo_tag = (
        f'<img src="data:image/png;base64,{_LOGO_B64}" alt="logo" style="width:150px;margin-bottom:16px;" />'
        if _LOGO_B64 else ""
    )
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Interview Chivon Powers</title>
    <style>
      *, *::before, *::after {{ box-sizing: border-box; }}
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #ffffff;
        margin: 0;
        color: #333;
      }}
      .wrap {{
        max-width: 700px;
        margin: 48px auto;
        padding: 0 24px;
        text-align: center;
      }}
      h1 {{
        font-size: 2.4rem;
        font-weight: 800;
        color: #4B0082;
        margin: 0 0 12px;
      }}
      .sub {{
        color: #555;
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 32px;
      }}
      .card {{
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 24px 28px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        text-align: left;
      }}
      .card label {{
        display: block;
        font-size: 14px;
        color: #333;
        margin-bottom: 8px;
      }}
      input {{
        width: 100%;
        padding: 10px 12px;
        border-radius: 6px;
        border: 2px solid #4B0082;
        font-size: 15px;
        outline: none;
        color: #333;
      }}
      input:focus {{
        box-shadow: 0 0 0 3px rgba(75,0,130,0.12);
      }}
      button {{
        margin-top: 12px;
        padding: 8px 20px;
        border-radius: 6px;
        border: 1px solid #ccc;
        background: #fff;
        color: #333;
        font-size: 14px;
        cursor: pointer;
      }}
      button:hover {{ background: #f5f5f5; }}
      button:disabled {{ opacity: .5; cursor: not-allowed; }}
      #answer {{
        white-space: pre-wrap;
        line-height: 1.6;
        margin-top: 20px;
        font-size: 16px;
        min-height: 0;
      }}
      .meta {{ color: #9ca3af; font-size: 12px; margin-top: 8px; }}
      .error {{ color: #b42318; }}
    </style>
  </head>
  <body>
    <div class="wrap">
      {logo_tag}
      <h1>Interview Chivon Powers</h1>
      <p class="sub">
        This bot responds as me, using my resume and other documents to answer your interview questions.<br/>
        It's a practical demonstration of my AI product development skills and a fun way to learn about my work experience.
      </p>
      <div class="card">
        <label for="q">Enter your interview question:</label>
        <input id="q" placeholder="How has your experience prepared you for a role in AI?" />
        <button id="ask">Ask the Question</button>
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

      function closeStream() {{
        if (es) {{ es.close(); es = null; }}
      }}

      function ask() {{
        const q = input.value.trim();
        if (!q) return;
        closeStream();
        answer.textContent = "";
        meta.textContent = "Streaming response...";
        meta.classList.remove("error");
        btn.disabled = true;

        es = new EventSource(`/chat/stream?question=${{encodeURIComponent(q)}}`);
        es.addEventListener("token", (evt) => {{
          answer.textContent += JSON.parse(evt.data);
        }});
        es.addEventListener("final", (evt) => {{
          const payload = JSON.parse(evt.data);
          const trace = payload.trace || {{}};
          const latency = trace.latency_ms ? `${{Math.round(trace.latency_ms)}} ms` : "n/a";
          const tools = (trace.tool_events || []).length;
          meta.textContent = `Done • latency: ${{latency}} • tools: ${{tools}}`;
        }});
        es.addEventListener("error", (evt) => {{
          meta.textContent = "Stream error. Try again.";
          meta.classList.add("error");
          btn.disabled = false;
          closeStream();
        }});
        es.addEventListener("done", () => {{
          btn.disabled = false;
          closeStream();
        }});
      }}

      btn.addEventListener("click", ask);
      input.addEventListener("keydown", (e) => {{ if (e.key === "Enter") ask(); }});
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
