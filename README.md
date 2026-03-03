# 🧠 Interview Chivon Powers – Resume Bot

A conversational AI agent app that lets people ask questions about my work experience. It uses Retrieval-Augmented Generation (RAG) and Web search Tools with OpenAI and FAISS to provide accurate, context-rich answers from my resume and CV and other sources.

## 🔍 Features

- 🔎 Retrieves relevant resume sections using vector search
- 🤖 Uses OpenAI (e.g. GPT-4 or GPT-3.5) to generate intelligent answers
- 💬 Built with Streamlit for a friendly web interface
- 📄 Embeds my resume and CV (PDFs) into FAISS
- ⚡ Streams responses for fast and interactive UX
  


## Quickstart
1. `pip install -r requirements.txt`
2. Copy `.env.example` → `.env` and fill keys (OpenAI, Tavily, LangSmith optional).
3. Put your PDFs into `data/raw/`.
4. `uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload`
5. Call `POST /chat` or `GET /chat/stream?question=...`.
6. Use `python -m app.services.ingest_index --rebuild` when documents change.

## FastAPI (Alternative)
- Run API server: `uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload`
- Health: `GET /healthz`
- JSON chat: `POST /chat` with body `{"question":"..."}`
- Streaming chat (SSE): `GET /chat/stream?question=...`

## Streamlit (Legacy UI)
- Streamlit remains available for local testing: `streamlit run app/streamlit_app.py`

## Notes
- Local-first retrieval; Tavily fallback only if needed.
- Answers: ≤ 3 sentences / ≤ 90 words; footnote markers [1], [2].
- Local citations appear as `local • <file> p.<n>`; web citations show real URLs.
- Session memory: last 8 turns.
- LangSmith tracing enabled with `LANGSMITH_API_KEY`.

## Controllers & Testing
- The default backend is the LangGraph controller (local-first with web fallback). Set `AGENT_BACKEND=langchain` to fall back to the legacy LangChain agent if needed.
- Rebuild the FAISS index before first run via the UI or `python -m app.services.ingest_index`.
- Smoke-test the LangGraph pipeline directly: `python -m app.agent.lg_controller "Tell me about your work on NLP"` (requires `.env` with OpenAI/Tavily keys).
