# agent/lc_controller.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import os, re, traceback

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.lc_prompts import SYSTEM
from agent.lc_tools import retrieve_local_tool, TAVILY, fetch_url_tool
from services.vectorstore import load_faiss_or_none  # used for prefetch local_context

MAX_ITER = 6

POLICY = (
    "Policy:\n"
    "1) Use the Local context below first. If insufficient, then tools in order: retrieve_local → "
    "tavily_search_results_json → fetch_url.\n"
    "2) Keep answers ≤ 3 sentences (≤ 90 words), first person, professional only.\n"
    "3) Add footnote markers [1], [2]. Cite local labels like 'local • <file> p.<n>' and real URLs for web.\n"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\n" + POLICY),
    MessagesPlaceholder("chat_history"),
    ("system", "Local context (may be empty):\n{local_context}"),
    ("human", "Question: {input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


def _finalize_from_steps(question: str, steps: List[Tuple[Any, Any]]) -> str:
    local_bits, urls, texts = [], [], []
    url_re = re.compile(r"https?://\S+")
    for (_act, obs) in steps[-3:]:
        if not isinstance(obs, str):
            continue
        texts.append(obs[:1200])
        for line in obs.splitlines()[:8]:
            if "local • " in line and len(local_bits) < 2:
                local_bits.append(line.strip())
        if len(urls) < 3:
            urls.extend(url_re.findall(obs))
        if len(local_bits) >= 2 and len(urls) >= 2:
            break
    urls = list(dict.fromkeys(urls))[:3]

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"), temperature=0.2)
    compose = ChatPromptTemplate.from_messages([
        ("system",
         "You are Chivon. Write a final interview answer:\n"
         "- ≤ 3 sentences, ≤ 90 words.\n- Professional scope only.\n"
         "- Use footnote markers [1], [2] with provided local labels/URLs if applicable."
        ),
        ("human", "Question: {q}\n\nObserved context:\n{ctx}\n\nLocal labels:\n{labels}\n\nURLs:\n{urls}")
    ])
    msgs = compose.format_messages(
        q=question,
        ctx="\n\n---\n\n".join(texts) if texts else "- none -",
        labels="\n".join(local_bits) if local_bits else "- none -",
        urls="\n".join(urls) if urls else "- none -",
    )
    return llm.invoke(msgs).content.strip()


def _local_context_from_faiss(question: str, k: int = 6) -> str:
    vs = load_faiss_or_none()
    if vs is None:
        return "[local] No index loaded."
    try:
        docs = vs.similarity_search(question, k=k)
    except Exception as e:
        return f"[local] error: {e}"
    blocks = []
    for d in docs:
        label = d.metadata.get("label") or d.metadata.get("source", "local.pdf")
        text = (d.page_content or "").strip().replace("\n\n", "\n")
        if len(text) > 1000:
            text = text[:1000] + "…"
        blocks.append(f"{label}\n{text}")
    return "\n\n---\n\n".join(blocks) if blocks else "[local] No results"


def make_executor() -> AgentExecutor:
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"), temperature=0.2)
    tools = [retrieve_local_tool, TAVILY, fetch_url_tool]
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=PROMPT)

    # IMPORTANT: tell memory which key is the user input/output
    memory = ConversationBufferWindowMemory(
        k=8,
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        return_messages=True,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=MAX_ITER,
        memory=memory,
        return_intermediate_steps=True,
    )


class LCController:
    def __init__(self):
        self.exec: Optional[AgentExecutor] = None
        self.init_error: Optional[str] = None
        self._ensure_agent()

    def _ensure_agent(self):
        if self.exec is not None:
            return
        try:
            self.exec = make_executor()
            self.init_error = None
        except Exception:
            self.exec = None
            self.init_error = traceback.format_exc()

    def respond(self, question: str) -> Dict[str, Any]:
        if self.exec is None:
            self._ensure_agent()
        if self.exec is None:
            return {
                "answer": "Initialization error.",
                "footnotes": {},
                "trace": {"init_trace": self.init_error or "No traceback."}
            }

        # Prefetch local snippets every turn and inject
        local_ctx = _local_context_from_faiss(question, k=6)
        out = self.exec.invoke({"input": question, "local_context": local_ctx})

        text = (out.get("output") or "").strip()
        steps = out.get("intermediate_steps", [])

        if not text or "Agent stopped" in text:
            text = _finalize_from_steps(question, steps)

        # -------------------- citations (with fallback to local_ctx) --------------------
        url_pattern = re.compile(r"https?://\S+")
        local_re = re.compile(r"local\s•\s(?P<file>.+?)\s+p\.(?P<page>\d+)", re.IGNORECASE)

        web_urls: List[str] = []
        local_labels: List[Dict[str, str]] = []

        def harvest(text_block: str):
            if not isinstance(text_block, str) or not text_block:
                return
            # URLs
            web_urls.extend(url_pattern.findall(text_block))
            # Local labels (look at first several lines for labels)
            for line in text_block.splitlines()[:20]:
                m = local_re.search(line.strip())
                if m:
                    local_labels.append({
                        "label": line.strip(),
                        "file": m.group("file"),
                        "page": m.group("page"),
                    })

        # 1) harvest from tool observations if any
        for (_a, obs) in steps:
            harvest(obs)

        # 2) fallback: if nothing found, harvest from the pre-injected local context
        if not web_urls and not local_labels:
            harvest(local_ctx)

        # de-dup & cap
        def _dedup(seq, key=lambda x: x):
            seen = set(); out_list = []
            for x in seq:
                k = key(x)
                if k in seen:
                    continue
                seen.add(k); out_list.append(x)
            return out_list

        web_urls = _dedup(web_urls)[:3]
        local_labels = _dedup(local_labels, key=lambda d: d["label"])[:3]

        # structured footnotes dict
        footnotes_map: Dict[int, Dict[str, str]] = {}
        idx = 1
        for u in web_urls:
            domain = re.sub(r"^https?://(www\.)?", "", u).split("/")[0]
            footnotes_map[idx] = {"title": f"web — {domain}", "url": u}
            idx += 1
        for d in local_labels:
            title = f"local — {d['file']} p.{d['page']}"
            footnotes_map[idx] = {"title": title, "path": f"local://{d['file']}#page={d['page']}"}
            idx += 1
        # -------------------------------------------------------------------------------

        trace = {
            "plan": "Tool-calling agent (local-first with pre-injected context; web fallback)",
            "steps": [
                {
                    "tool": getattr(a, "tool", ""),
                    "input": getattr(a, "tool_input", ""),
                    "observation": (obs[:240] + "…") if isinstance(obs, str) and len(obs) > 240 else obs,
                }
                for a, obs in steps
            ],
            "local_context_preview": local_ctx[:600] if isinstance(local_ctx, str) else str(local_ctx)[:600],
        }
        return {"answer": text, "footnotes": footnotes_map, "trace": trace}
