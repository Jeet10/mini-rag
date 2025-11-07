#!/usr/bin/env python3
# Mini RAG Chat — ChatGPT-like sessions (auto-load on click, auto-save, auto-scroll)
# RAG: FAISS + MiniLM (local embeddings) + OpenAI/Azure; supports .txt/.md/.pdf

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

# LLM clients
try:
    from openai import OpenAI, AzureOpenAI
except Exception:
    from openai import OpenAI
    AzureOpenAI = None

load_dotenv()

# ----------- Constants -----------
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


# ----------- Embeddings / FAISS -----------
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


# ----------- Ingestion (.txt/.md/.pdf) -----------
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


# ----------- LLM -----------
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

def llm_answer(question: str, context_chunks: List[Chunk]) -> str:
    client, is_azure = ensure_llm_client()
    context_text = "\n\n".join(f"[{os.path.basename(c.source)}]\n{c.text}" for c in context_chunks)
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT") if is_azure else os.getenv("OPENAI_CHAT_MODEL", DEFAULT_MODEL)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer only using the provided context. If not present, say you don't know."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_text}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ----------- Chat History (auto-save, auto-load, titles) -----------
def default_session_id() -> str:
    # Stable ID with timestamp; title is derived from first user message
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def session_file(session_id: str) -> Path:
    return HIST_DIR / f"{session_id}.json"

def save_chat(session_id: str, messages: List[Tuple[str, str]]) -> None:
    with open(session_file(session_id), "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def load_chat(session_id: str) -> List[Tuple[str, str]]:
    p = session_file(session_id)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def list_sessions() -> List[str]:
    # Return IDs sorted by mtime desc (newest first)
    files = sorted(HIST_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in files]

def session_title(session_id: str, messages: List[Tuple[str, str]]) -> str:
    # Title = first user message (trimmed) or timestamp
    for role, msg in messages:
        if role == "user" and msg.strip():
            title = msg.strip().split("\n")[0]
            return (title[:30] + "…") if len(title) > 30 else title
    # Fallback to session id formatted
    return session_id.replace("_", " ")

def get_all_session_titles() -> List[Tuple[str, str]]:
    # [(session_id, title)]
    ids = list_sessions()
    out = []
    for sid in ids:
        msgs = load_chat(sid)
        out.append((sid, session_title(sid, msgs)))
    return out


# ----------- Streamlit UI -----------
st.set_page_config(page_title="Mini RAG — Chat (ChatGPT-like)", page_icon="💬", layout="wide")
st.title("💬 Mini RAG — Chat with Your Documents")
st.caption("ChatGPT-like sidebar • Auto-load on click • Auto-save continuously • Auto-scroll • PDF/TXT/MD")

# State init
if "session_id" not in st.session_state:
    st.session_state.session_id = default_session_id()
if "messages" not in st.session_state:
    st.session_state.messages: List[Tuple[str, str]] = load_chat(st.session_state.session_id)
if "k" not in st.session_state:
    st.session_state.k = 4

# ----- Sidebar: Chat list + New Chat + Documents -----
with st.sidebar:
    st.subheader("💬 Chats")

    # New chat button (like ChatGPT)
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.session_id = default_session_id()
        st.session_state.messages = []
        save_chat(st.session_state.session_id, st.session_state.messages)
        st.rerun()

    # Clickable chat list (radio). Selecting a chat auto-loads it (no button).
    chat_items = get_all_session_titles()
    if not chat_items:
        st.info("No chats yet. Click **＋ New chat** to start.")
        selected_label = None
    else:
        labels = [f"{title}" for _, title in chat_items]
        ids = [sid for sid, _ in chat_items]
        # Preselect current session if present
        try:
            default_idx = ids.index(st.session_state.session_id)
        except ValueError:
            default_idx = 0
        selected_label = st.radio(
            "History",
            options=labels,
            index=default_idx,
            label_visibility="collapsed",
        )
        # Detect selection change and auto-load
        if selected_label is not None:
            sel_idx = labels.index(selected_label)
            sel_id = ids[sel_idx]
            if sel_id != st.session_state.session_id:
                st.session_state.session_id = sel_id
                st.session_state.messages = load_chat(sel_id)
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

# ----- Render chat history -----
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

# Small HTML snippet to auto-scroll to the bottom after render
components.html(
    """
    <script>
      const out = window.parent.document.querySelector('section.main');
      if (out) out.scrollTo(0, out.scrollHeight);
    </script>
    """,
    height=0,
)

# ----- Chat input -----
q = st.chat_input("Ask something about your documents…")
if q:
    # If it's a brand-new chat, session_id already set; title will be first user msg
    st.session_state.messages.append(("user", q))
    save_chat(st.session_state.session_id, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(q)

    # Retrieve & answer
    try:
        ctx = retrieve(q, st.session_state.k)
        ans = llm_answer(q, ctx) if ctx else "No relevant information found. Try building the index first."
    except Exception as e:
        ans = f"Error: {e}"

    st.session_state.messages.append(("assistant", ans))
    save_chat(st.session_state.session_id, st.session_state.messages)

    with st.chat_message("assistant"):
        st.markdown(ans)

    # Re-render to update sidebar titles if this was the first user turn (title derives from it)
    st.rerun()
