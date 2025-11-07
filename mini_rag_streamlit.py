#!/usr/bin/env python3
# Minimal RAG Chat — streaming + sturdy
# - Ingest .txt/.md/.pdf → chunk → embed (MiniLM) → FAISS → retrieve → LLM (OpenAI/Azure)
# - Titles stored in same JSON; Rename/Delete via label-less popover (no icon text)
# - No empty chat saves; First load = new chat
# - Robust legacy history handling
# - Streaming assistant responses (ChatGPT-like)

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
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

# ---- LLM clients ----
try:
    from openai import OpenAI, AzureOpenAI
except Exception:
    from openai import OpenAI
    AzureOpenAI = None

load_dotenv()

# ---- Constants ----
DATA_DIR = "docs"
INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "gpt-4o-mini"

HIST_DIR = Path("chat_history")
HIST_DIR.mkdir(parents=True, exist_ok=True)

# ---- Data model ----
@dataclass
class Chunk:
    text: str
    source: str

# ---- Utils: normalize message shapes ----
def normalize_messages(raw) -> List[Tuple[str, str]]:
    """
    Convert various message shapes to list[(role, content)].
    Accepts dicts with {role,content} or 2-item lists/tuples.
    Skips malformed entries safely.
    """
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

# ---- Session storage (title + messages inside ONE JSON) ----
def default_session_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def session_path(session_id: str) -> Path:
    return HIST_DIR / f"{session_id}.json"

def load_session(session_id: str):
    """
    Returns (title, messages). Backward compatible with legacy list-only files.
    """
    p = session_path(session_id)
    if not p.exists():
        return "New chat", []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "New chat", []
    # Modern format: {"title": str, "messages": [...]}
    if isinstance(data, dict) and "messages" in data:
        title = data.get("title") or "New chat"
        messages = normalize_messages(data["messages"])
        return title, messages
    # Legacy format: just a list of messages
    if isinstance(data, list):
        messages = normalize_messages(data)
        title = derive_title_from_messages(messages)
        return title, messages
    return "New chat", []

def save_session(session_id: str, title: str, messages) -> None:
    """
    Saves {"title": title, "messages": normalized_messages} to one JSON file.
    Won't save empty-chats (no messages).
    """
    msgs = normalize_messages(messages)
    if not msgs:
        return
    payload = {"title": title or "New chat", "messages": msgs}
    with open(session_path(session_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def list_sessions_with_titles() -> List[Tuple[str, str]]:
    """
    Returns list of (session_id, title), newest first by mtime.
    """
    items = []
    files = sorted(HIST_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files:
        sid = p.stem
        title, _ = load_session(sid)
        items.append((sid, title))
    return items

def rename_session(session_id: str, new_title: str) -> None:
    """
    Update the title inside the same JSON file.
    """
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

# ---- Embeddings / FAISS ----
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

# ---- Ingestion (.txt/.md/.pdf) ----
def read_txt_md(folder: str) -> List[Tuple[str, str]]:
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

def read_pdfs(folder: str) -> List[Tuple[str, str]]:
    paths = glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True)
    out = []
    for p in sorted(set(paths)):
        txt = extract_pdf_text_path(p)
        if txt:
            out.append((p, txt))
    return out

def read_documents(folder: str) -> List[Tuple[str, str]]:
    return read_txt_md(folder) + read_pdfs(folder)

def build_index(docs: List[Tuple[str, str]]):
    chunks: List[Chunk] = []
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

# ---- LLM ----
def ensure_llm_client():
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        if AzureOpenAI is None:
            raise RuntimeError("Upgrade 'openai' for Azure support: pip install --upgrade openai")
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        ), True
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or AZURE_* environment variables.")
    return OpenAI(api_key=api_key), False

def llm_answer_stream(question: str, context_chunks: List[Chunk]):
    """
    Stream tokens as they arrive (ChatGPT-style).
    Yields small string pieces progressively.
    """
    client, is_azure = ensure_llm_client()
    context_text = "\n\n".join(f"[{os.path.basename(c.source)}]\n{c.text}" for c in context_chunks)
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT") if is_azure else os.getenv("OPENAI_CHAT_MODEL", DEFAULT_MODEL)

    stream = client.chat.completions.create(
        model=model,
        stream=True,
        messages=[
            {"role": "system", "content": "Answer only using the provided context. If not present, say you don't know."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_text}"},
        ],
        temperature=0.2,
    )

    # OpenAI ChatCompletionChunk has choices[0].delta.content pieces
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content
        except Exception:
            # Skip malformed chunks silently
            continue

# ==== STREAMLIT UI ====
st.set_page_config(page_title="Mini RAG — Minimal Chat", page_icon="💬", layout="wide")
st.title("💬 Mini RAG — Minimal Chat (Streaming)")
st.caption("Titles stored per JSON • Label-less menu • No empty chat saves • PDF/TXT/MD • FAISS + MiniLM • OpenAI/Azure • Streaming")

# First load → new chat in memory (unsaved until first user message)
if "session_id" not in st.session_state:
    st.session_state.session_id = default_session_id()
if "title" not in st.session_state:
    st.session_state.title = "New chat"
if "messages" not in st.session_state:
    st.session_state.messages: List[Tuple[str, str]] = []
if "k" not in st.session_state:
    st.session_state.k = 4

# ----- Sidebar: Chats & Docs -----
with st.sidebar:
    st.subheader("💬 Chats")

    # New chat (unsaved until first message)
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.session_id = default_session_id()
        st.session_state.title = "New chat"
        st.session_state.messages = []
        st.rerun()

    # Chat list: main button + label-less popover (Rename/Delete)
    for sid, title in list_sessions_with_titles():
        cols = st.columns([7, 2], vertical_alignment="center")

        if cols[0].button(title, key=f"open_{sid}", use_container_width=True):
            t, msgs = load_session(sid)
            st.session_state.session_id = sid
            st.session_state.title = t
            st.session_state.messages = msgs
            st.rerun()

        with cols[1]:
            # NOTE: per your request, no visible label/icon on the trigger
            try:
                pop = st.popover("")
            except Exception:
                pop = st.expander("")
            with pop:
                new_name = st.text_input("Rename", value=title, key=f"name_{sid}")
                if st.button("Rename", key=f"rename_{sid}", use_container_width=True):
                    rename_session(sid, new_name)
                    st.rerun()
                if st.button("Delete", key=f"delete_{sid}", use_container_width=True):
                    deleting_current = (sid == st.session_state.session_id)
                    delete_session(sid)
                    if deleting_current:
                        # return to fresh, unsaved chat
                        st.session_state.session_id = default_session_id()
                        st.session_state.title = "New chat"
                        st.session_state.messages = []
                    st.rerun()

    st.divider()
    st.subheader("📄 Documents & Index")
    st.session_state.k = st.number_input("Top-K Retrieval", min_value=1, max_value=12, value=st.session_state.k, step=1)
    use_docs = st.toggle("Use docs/ folder", True)
    uploaded = st.file_uploader("Upload .txt/.md/.pdf", type=["txt", "md", "pdf"], accept_multiple_files=True)

    if st.button("📦 Build / Refresh Index", type="primary", use_container_width=True):
        try:
            docs: List[Tuple[str, str]] = []
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

# ----- Main chat area -----
# Render messages
for role, msg in normalize_messages(st.session_state.messages):
    with st.chat_message(role):
        st.markdown(msg)

# Chat input
q = st.chat_input("Ask something about your documents…")
if q:
    # Append user message in memory
    st.session_state.messages.append(("user", q))

    # If this is the first message, derive a title and persist the session
    if not session_path(st.session_state.session_id).exists():
        st.session_state.title = derive_title_from_messages(st.session_state.messages)
    # Persist (title + messages) after user turn
    save_session(st.session_state.session_id, st.session_state.title, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(q)

    # Retrieve context
    try:
        ctx = retrieve(q, st.session_state.k)
    except Exception as e:
        ctx = []
        with st.chat_message("assistant"):
            st.error(f"Retrieval error: {e}")

    # Stream assistant answer
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""
        if ctx:
            try:
                for token in llm_answer_stream(q, ctx):
                    streamed_text += token
                    placeholder.markdown(streamed_text)
            except Exception as e:
                streamed_text += f"\n\n[Streaming error: {e}]"
                placeholder.markdown(streamed_text)
        else:
            streamed_text = "No relevant information found. Try building the index first."
            placeholder.markdown(streamed_text)

    # Save final assistant message
    st.session_state.messages.append(("assistant", streamed_text))
    save_session(st.session_state.session_id, st.session_state.title, st.session_state.messages)

    # Rerun to refresh sidebar titles/order if this was the first user turn
    st.rerun()
