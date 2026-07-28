# 🧠 Interview Chivon Powers – Resume Bot

A conversational AI portfolio that answers professional questions in Chivon's
voice. It uses a deterministic, evidence-first RAG pipeline with hybrid retrieval,
approved public sources, and an automatic web fallback.

## 🔍 Features

- 🔎 Combines FAISS semantic search with BM25 lexical search
- 🧾 Returns deterministic source records derived from retrieved evidence
- 🤖 Uses a centrally configured OpenAI model (`gpt-5.6-terra` until Luna passes the benchmark gates)
- 💬 Built with Streamlit for a friendly web interface
- 📄 Indexes structured, PII-sanitized resume/CV/project chunks
- 🛴 Uses the approved resume for documented Lime role and impact details
- ⚡ Streams responses for fast and interactive UX
  


## Quickstart
1. `pip install -r requirements.txt`
2. Create `.env` and fill keys (OpenAI required; Tavily and LangSmith optional).
3. Put PDFs into `app/data/raw/` or set `DATA_RAW_DIR`.
4. `uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload`
5. Call `POST /chat` or `GET /chat/stream?question=...`.
6. Build explicitly with `python -m app.services.ingest_index --rebuild`.

The service hashes every source plus the embedding/chunking configuration. On
startup it rebuilds automatically when that manifest is stale.

## Public profile refresh

The canonical public-profile source is LinkedIn. Its committed snapshot contains
public facts only and currently approves one claim: Lime is Chivon's current
employer. The indexed resume is a separate, user-approved source for documented
Lime title, dates, projects, and outcomes.

```bash
python -m app.services.profile_snapshot --refresh
python -m app.services.ingest_index --rebuild
```

Or combine both steps:

```bash
python -m app.services.ingest_index --refresh-profile --rebuild
```

If LinkedIn blocks the refresh, the last successful snapshot is retained.

## FastAPI (Alternative)
- Run API server: `uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload`
- Health: `GET /healthz` (`ok`, `degraded`, or `not_ready`, with index freshness)
- JSON chat: `POST /chat` with body `{"question":"..."}`
- Streaming chat (SSE): `GET /chat/stream?question=...`
- Stable response keys: `answer`, `sources`, `source_freshness`, `validation`,
  `footnotes` (compatibility), and `trace`

## Streamlit (Legacy UI)
- Streamlit remains available for local testing: `streamlit run app/streamlit_app.py`

## Notes
- One hybrid retrieval pass for normal local questions; one low-confidence query
  rewrite at most.
- Tavily is used only for weak evidence or explicitly fresh questions.
- Chivon-specific web claims are restricted to the canonical LinkedIn profile;
  user-approved indexed documents may support additional professional details.
- Answers target 2–4 sentences and normally 60–120 words in a warm-expert voice.
- The model does not invent citation numbers; the UI renders the retrieved
  `sources` list directly.
- LangSmith tracing enabled with `LANGSMITH_API_KEY`.

## Controllers & Testing
- `LGController` is the canonical production pipeline for FastAPI, Streamlit,
  LangGraph deployment, CLI smoke tests, and evaluation.
- Smoke-test it with `python -m app.agent.lg_controller "Tell me about your work on NLP"`.
- Run deterministic tests with `python -m unittest discover -s tests -v`.
- Run the 60+ case evaluation suite with `python -m app.eval.run_eval`.
- Compare production candidates with:

  ```bash
  python -m app.eval.benchmark_models --output eval_outputs/model_benchmark.json
  ```

  Luna is promoted by setting `OPENAI_MODEL=gpt-5.6-luna` only when it scores at
  least 0.90 on critical metrics, stays within 0.02 of Terra, and keeps local p95
  latency at or below five seconds.
