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
4. `streamlit run app/streamlit_app.py`
5. Use **(Re)Build Index** to (re)create FAISS.

## Notes
- Local-first retrieval; Tavily fallback only if needed.
- Answers: ≤ 3 sentences / ≤ 90 words; footnote markers [1], [2].
- Local citations appear as `local • <file> p.<n>`; web citations show real URLs.
- Session memory: last 8 turns.
- LangSmith tracing enabled with `LANGSMITH_API_KEY`.