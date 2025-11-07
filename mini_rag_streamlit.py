#!/usr/bin/env python3
# Minimal RAG Chat — streaming + sturdy (+ answer-mode toggle) + ChatGPT-like memory
# + Copy button for completed answers + Input lock while generating

import os
import io
import glob
import json
import pickle
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

try:
    from openai import OpenAI, AzureOpenAI
except Exception:
    from openai import OpenAI
    AzureOpenAI = None

load_dotenv()

DATA_DIR = "docs"
INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "gpt-4o-mini"

HIST_DIR = Path("chat_history")
HIST_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class Chunk:
    text: str
    source: str

def normalize_messages(raw) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for item in raw:
        try:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if isinstance(role, str) and isinstance(content, str):
                    out.append((role, content))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                role, content = item[0], item[1]
                if isinstance(role, str) and isinstance(content, str):
                    out.append((role, content))
        except Exception:
            continue
    return out

def derive_title_from_messages(messages: List[Tuple[str, str]]) -> str:
    for role, msg in normalize_messages(messages):
        if role == "user" and msg.strip():
            first = msg.strip().split("\n")[0]
            return (first[:30] + "…") if len(first) > 30 else first
    return "New chat"

def default_session_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def session_path(session_id: str) -> Path:
    return HIST_DIR / f"{session_id}.json"

def load_session(session_id: str):
    p = session_path(session_id)
    if not p.exists():
        return "New chat", []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "New chat", []
    if isinstance(data, dict) and "messages" in data:
        title = data.get("title") or "New chat"
        messages = normalize_messages(data["messages"])
        return title, messages
    if isinstance(data, list):
        messages = normalize_messages(data)
        title = derive_title_from_messages(messages)
        return title, messages
    return "New chat", []

def save_session(session_id: str, title: str, messages) -> None:
    msgs = normalize_messages(messages)
    if not msgs:
        return
    payload = {"title": title or "New chat", "messages": msgs}
    with open(session_path(session_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def list_sessions_with_titles() -> List[Tuple[str, str]]:
    items = []
    files = sorted(HIST_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files:
        sid = p.stem
        title, _ = load_session(sid)
        items.append((sid, title))
    return items

def rename_session(session_id: str, new_title: str) -> None:
    _, messages = load_session(session_id)
    new_title = (new_title or "").strip()
    if not new_title:
        return
    save_session(session_id, new_title, messages)

def delete_session(session_id: str) -> None:
    p = session_path(session_id)
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner=False)
def blank_index(dim: int):
    return faiss.IndexFlatIP(dim)

def chunk_text(text: str, size: int = 350, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]

def embed_chunks(chunks: List[Chunk]):
    model = get_embedder()
    vecs = model.encode([c.text for c in chunks], normalize_embeddings=True)
    idx = blank_index(vecs.shape[1])
    idx.add(vecs)
    return idx, chunks

def save_index(idx, meta):
    faiss.write_index(idx, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

def load_index():
    idx = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return idx, meta

def extract_pdf_text_bytes(data: bytes) -> str:
    try:
        r = PdfReader(io.BytesIO(data))
        return "\n".join([(p.extract_text() or "") for p in r.pages]).strip()
    except Exception:
        return ""

def extract_pdf_text_path(path: str) -> str:
    try:
        r = PdfReader(path)
        return "\n".join([(p.extract_text() or "") for p in r.pages]).strip()
    except Exception:
        return ""

def read_txt_md(folder: str):
    paths = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
    paths += glob.glob(os.path.join(folder, "**", "*.md"), recursive=True)
    out = []
    for p in sorted(set(paths)):
        try:
            with open(p, "r", encoding="utf8", errors="ignore") as f:
                text = f.read().strip()
                if text:
                    out.append((p, text))
        except Exception:
            pass
    return out

def read_pdfs(folder: str):
    paths = glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True)
    out = []
    for p in sorted(set(paths)):
        txt = extract_pdf_text_path(p)
        if txt:
            out.append((p, txt))
    return out

def read_documents(folder: str):
    return read_txt_md(folder) + read_pdfs(folder)

def build_index(docs):
    chunks = []
    for path, text in docs:
        for ch in chunk_text(text):
            chunks.append(Chunk(ch, path))
    if not chunks:
        raise RuntimeError("No chunks produced. Add documents first.")
    idx, meta = embed_chunks(chunks)
    save_index(idx, meta)

def retrieve(question: str, k: int):
    idx, meta = load_index()
    qvec = get_embedder().encode([question], normalize_embeddings=True)
    _, I = idx.search(qvec, k)
    return [meta[i] for i in I[0] if i != -1]

def ensure_llm_client():
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        if AzureOpenAI is None:
            raise RuntimeError("Upgrade 'openai' for Azure support.")
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        ), True
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or AZURE credentials.")
    return OpenAI(api_key=api_key), False

# ---------------- ChatGPT-like answer streaming with memory ----------------
def build_chat_messages(
    history: List[Tuple[str, str]],
    current_user_text: str,
    context_text: str,
    mode: str = "strict",
    history_limit: int = 16,
):
    if mode == "strict":
        sys = (
            "You are a helpful RAG assistant.\n"
            "Use ONLY the supplied Document Context for factual content.\n"
            "You MAY rephrase, simplify, or expand previous assistant replies upon request, "
            "but do not introduce facts that are not supported by the Document Context.\n"
            "If the requested information is not present in the context, say you don't know.\n"
            "Cite filenames in brackets (e.g., [file.pdf]) only if helpful."
        )
    else:
        sys = (
            "You are a helpful assistant.\n"
            "Prefer the supplied Document Context when relevant, but you MAY use general knowledge.\n"
            "If the user's request is a rephrasing or formatting change, use conversation history to comply."
        )

    messages = [{"role": "system", "content": sys}]
    doc_header = "Document Context:\n\n" + (context_text if context_text.strip() else "(none)")
    messages.append({"role": "system", "content": doc_header})

    hist = normalize_messages(history)[-history_limit:]
    for role, content in hist:
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": current_user_text})
    return messages

def llm_answer_stream_with_memory(
    history: List[Tuple[str, str]],
    user_text: str,
    context_chunks: List[Chunk],
    mode: str = "strict",
):
    client, is_azure = ensure_llm_client()
    context_text = "\n\n".join(f"[{os.path.basename(c.source)}]\n{c.text}" for c in context_chunks)
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT") if is_azure else os.getenv("OPENAI_CHAT_MODEL", DEFAULT_MODEL)

    messages = build_chat_messages(history, user_text, context_text, mode=mode)

    stream = client.chat.completions.create(
        model=model,
        stream=True,
        messages=messages,
        temperature=0.2,
    )

    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content
        except Exception:
            continue

# ---------------- UI helpers ----------------

def _truncate_title(title: str, max_chars: int = 28) -> str:
    title = (title or "").strip()
    return title if len(title) <= max_chars else (title[:max_chars - 1] + "…")

def copy_button(text: str, key: str):
    """Polished copy button with icon + tooltip."""
    js_text = json.dumps(text)
    components.html(
        f"""
<style>
  .copy-btn {{
    display:inline-flex;align-items:center;gap:8px;
    padding:6px 10px;border:1px solid #E5E7EB;border-radius:10px;
    background:#FFFFFF;cursor:pointer;font-size:13px;line-height:1;
    transition:all .15s ease; box-shadow:0 1px 2px rgba(0,0,0,.04);
  }}
  .copy-btn:hover {{ background:#F9FAFB; border-color:#D1D5DB; }}
  .copy-btn:active {{ transform:translateY(1px); }}
  .copy-btn svg {{ width:16px;height:16px;opacity:.9; }}
  .copy-tooltip {{
    position:relative; display:inline-block;
  }}
  .copy-tooltip .tip {{
    visibility:hidden; opacity:0; position:absolute; bottom:125%; left:50%; transform:translateX(-50%);
    background:#111827;color:#fff;padding:4px 8px;border-radius:6px;font-size:12px;white-space:nowrap;
    transition:opacity .15s ease; pointer-events:none;
  }}
  .copy-tooltip:hover .tip {{ visibility:visible; opacity:1; }}
</style>
<div class="copy-tooltip">
  <button id="{key}" class="copy-btn" aria-label="Copy answer">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
    </svg>
    <span>Copy</span>
  </button>
  <span class="tip">Copy to clipboard</span>
</div>
<script>
  const btn = document.getElementById("{key}");
  if (btn) {{
    btn.onclick = async () => {{
      try {{
        await navigator.clipboard.writeText({js_text});
        const label = btn.querySelector("span");
        const old = label.innerText;
        label.innerText = "Copied!";
        setTimeout(() => label.innerText = old, 1200);
      }} catch (e) {{
        const label = btn.querySelector("span");
        label.innerText = "Unable to copy";
        setTimeout(() => label.innerText = "Copy", 1500);
      }}
    }};
  }}
</script>
        """,
        height=48,
    )

# ----------------------------- PAGE -----------------------------

st.set_page_config(page_title="Mini RAG — Minimal Chat", page_icon="💬", layout="wide")

# CSS tweaks
st.markdown("""
<style>
section[data-testid="stSidebar"] .stButton > button {
  height: 36px;
}
section[data-testid="stSidebar"] .stButton > button p {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
section[data-testid="stSidebar"] [data-testid="stPopover"] > button {
  height: 36px;
  padding-left: 6px !important;
  padding-right: 6px !important;
  min-width: 36px;
}
/* Dim input while generating */
div[data-baseweb="input"] textarea[disabled],
div[data-testid="stChatInput"] textarea[disabled] {
  opacity: .6;
  cursor: not-allowed;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 Mini RAG — Minimal Chat")
st.caption("Chat with your files — upload, search, and get fast contextual answers.")

# ---------------- Session state ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = default_session_id()
if "title" not in st.session_state:
    st.session_state.title = "New chat"
if "messages" not in st.session_state:
    st.session_state.messages: List[Tuple[str, str]] = []
if "k" not in st.session_state:
    st.session_state.k = 4
if "mode" not in st.session_state:
    st.session_state.mode = "strict"
if "generating" not in st.session_state:
    st.session_state.generating = False  # lock input while generating
if "pending_q" not in st.session_state:
    st.session_state.pending_q = None     # <-- for two-step handoff

# --- Front-end guard: block Enter when generating ---
components.html(f"""
<script>
  (function() {{
    const generating = {str(st.session_state.get("generating", False)).lower()};
    if (!generating) return;
    const handler = (e) => {{
      if (e.key === 'Enter' && !e.shiftKey) {{
        e.preventDefault();
        e.stopImmediatePropagation();
        return false;
      }}
    }};
    window.addEventListener('keydown', handler, true);
  }})();
</script>
""", height=0)

with st.sidebar:
    st.subheader("💬 Chats")
    if st.button("＋ New chat", use_container_width=True, disabled=st.session_state.generating):
        st.session_state.session_id = default_session_id()
        st.session_state.title = "New chat"
        st.session_state.messages = []
        st.rerun()

    for sid, title in list_sessions_with_titles():
        display_title = _truncate_title(title)
        cols = st.columns([0.82, 0.18])

        if cols[0].button(display_title, key=f"open_{sid}", use_container_width=True, help=title, disabled=st.session_state.generating):
            t, msgs = load_session(sid)
            st.session_state.session_id = sid
            st.session_state.title = t
            st.session_state.messages = msgs
            st.rerun()

        with cols[1]:
            try:
                pop = st.popover("")
            except Exception:
                pop = st.expander("")
            with pop:
                new_name = st.text_input("Rename", value=title, key=f"name_{sid}")
                if st.button("Rename", key=f"rename_{sid}", use_container_width=True, disabled=st.session_state.generating):
                    rename_session(sid, new_name)
                    st.rerun()
                if st.button("Delete", key=f"delete_{sid}", use_container_width=True, disabled=st.session_state.generating):
                    deleting_current = (sid == st.session_state.session_id)
                    delete_session(sid)
                    if deleting_current:
                        st.session_state.session_id = default_session_id()
                        st.session_state.title = "New chat"
                        st.session_state.messages = []
                    st.rerun()

    st.divider()

    st.subheader("🧠 Answer Mode")
    st.session_state.mode = st.radio(
        "How should answers be generated?",
        options=["strict", "gk"],
        format_func=lambda x: "📚 Strict — docs only" if x == "strict" else "🌐 Allow general knowledge",
        index=0,
        disabled=st.session_state.generating,
    )

    st.subheader("📄 Documents & Index")
    st.session_state.k = st.number_input(
        "Top-K Retrieval", min_value=1, max_value=12, value=st.session_state.k, step=1,
        disabled=st.session_state.generating
    )
    use_docs = st.toggle("Use docs/ folder", True, disabled=st.session_state.generating)
    uploaded = st.file_uploader(
        "Upload .txt/.md/.pdf", type=["txt", "md", "pdf"], accept_multiple_files=True,
        disabled=st.session_state.generating
    )

    if st.button("📦 Build / Refresh Index", type="primary", use_container_width=True, disabled=st.session_state.generating):
        try:
            docs = []
            if use_docs:
                os.makedirs(DATA_DIR, exist_ok=True)
                docs += read_documents(DATA_DIR)
            for f in uploaded or []:
                data = f.read()
                os.makedirs(DATA_DIR, exist_ok=True)
                dst = Path(DATA_DIR) / f.name
                with open(dst, "wb") as out:
                    out.write(data)
                text = (
                    data.decode("utf8", errors="ignore")
                    if f.name.lower().endswith((".txt", ".md"))
                    else extract_pdf_text_bytes(data)
                )
                if text:
                    docs.append((str(dst), text))
            build_index(docs)
            st.success("Index built ✅")
        except Exception as e:
            st.error(f"Index build failed: {e}")

# Render history (with copy buttons for assistant messages)
for i, (role, msg) in enumerate(normalize_messages(st.session_state.messages)):
    with st.chat_message(role):
        st.markdown(msg)
        if role == "assistant" and msg.strip():
            copy_button(msg, key=f"copy_btn_{i}")

# ---------------- Chat input (two-step handoff) ----------------
# 1) If not generating: capture input, set lock + pending, and rerun immediately.
if not st.session_state.generating:
    q = st.chat_input("Ask something about your documents…")
    if q:
        st.session_state.pending_q = q
        st.session_state.generating = True
        st.rerun()
else:
    # show disabled input while generating (prevents accidental Enter)
    st.chat_input("Generating… please wait", disabled=True)

# 2) If we have a pending query and we're in generating mode, process it now.
if st.session_state.generating and st.session_state.pending_q:
    q = st.session_state.pending_q

    # log user message
    st.session_state.messages.append(("user", q))
    if not session_path(st.session_state.session_id).exists():
        st.session_state.title = derive_title_from_messages(st.session_state.messages)
    save_session(st.session_state.session_id, st.session_state.title, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(q)

    # Retrieval
    try:
        ctx = retrieve(q, st.session_state.k)
    except Exception as e:
        ctx = []
        with st.chat_message("assistant"):
            st.error(f"Retrieval error: {e}")
        # unlock & clear pending, then persist
        st.session_state.pending_q = None
        st.session_state.generating = False
        save_session(st.session_state.session_id, st.session_state.title, st.session_state.messages)
        st.stop()

    # Streaming answer using conversation history + context
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""
        try:
            for token in llm_answer_stream_with_memory(
                history=st.session_state.messages[:-1],   # all messages before this assistant turn
                user_text=q,
                context_chunks=ctx,
                mode=st.session_state.mode,
            ):
                streamed_text += token
                placeholder.markdown(streamed_text)
        except Exception as e:
            streamed_text += f"\n\n[Streaming error: {e}]"
            placeholder.markdown(streamed_text)

    # Save, unlock, clear pending, and rerun so Copy button renders on the final message
    st.session_state.messages.append(("assistant", streamed_text))
    save_session(st.session_state.session_id, st.session_state.title, st.session_state.messages)
    st.session_state.pending_q = None
    st.session_state.generating = False
    st.rerun()
