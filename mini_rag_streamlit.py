#!/usr/bin/env python3
# Streamlit UI for the minimal RAG (FAISS + MiniLM + OpenAI/Azure OpenAI)
# --------------------------------------------------------------
# Features
# - Upload or load .txt/.md/.pdf files
# - Build/refresh FAISS index
# - Ask questions; see retrieved chunks and source attributions
# - Works with OpenAI (OPENAI_API_KEY) or Azure OpenAI (AZURE_* envs)
# --------------------------------------------------------------

import os
import io
import glob
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader  # PDF extraction

# LLM clients
try:
    from openai import OpenAI, AzureOpenAI
except Exception:  # fallback for older openai package
    from openai import OpenAI  # type: ignore
    AzureOpenAI = None  # type: ignore

# ---------------- Consts ----------------
DATA_DIR = "docs"
INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"  # for OpenAI; Azure uses deployment name

# ---------------- Data model ----------------
@dataclass
class Chunk:
    text: str
    source: str

# ---------------- Helpers ----------------
@st.cache_resource(show_spinner=False)
def get_embedder(name: str = EMBED_MODEL_NAME):
    return SentenceTransformer(name)


def _read_txt_md(folder: str) -> List[Tuple[str, str]]:
    paths = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
    paths += glob.glob(os.path.join(folder, "**", "*.md"), recursive=True)
    items: List[Tuple[str, str]] = []
    for p in sorted(set(paths)):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
                if txt.strip():
                    items.append((p, txt))
                else:
                    st.warning(f"Empty text in {p}, skipping.")
        except Exception as e:
            st.warning(f"Skip {p}: {e}")
    return items


def extract_text_from_pdf_path(path: str) -> str:
    try:
        reader = PdfReader(path)
        chunks = []
        for pg in reader.pages:
            t = pg.extract_text() or ""
            chunks.append(t)
        return "\n".join(chunks)
    except Exception as e:
        st.warning(f"Failed to read PDF {os.path.basename(path)}: {e}")
        return ""


def extract_text_from_pdf_bytes(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        chunks = []
        for pg in reader.pages:
            t = pg.extract_text() or ""
            chunks.append(t)
        return "\n".join(chunks)
    except Exception as e:
        st.warning(f"Failed to read uploaded PDF: {e}")
        return ""


def _read_pdfs(folder: str) -> List[Tuple[str, str]]:
    pdfs = glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True)
    items: List[Tuple[str, str]] = []
    for p in sorted(set(pdfs)):
        txt = extract_text_from_pdf_path(p)
        if txt.strip():
            items.append((p, txt))
        else:
            st.warning(f"No extractable text in {p}, skipping.")
    return items


def read_documents(folder: str) -> List[Tuple[str, str]]:
    """Return (source_path, text) for txt, md, and pdf files under folder."""
    items = []
    items.extend(_read_txt_md(folder))
    items.extend(_read_pdfs(folder))
    return items


def simple_chunk(text: str, max_words: int = 350, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    step = max(1, max_words - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i:i + max_words]))
        i += step
    return chunks


@st.cache_resource(show_spinner=False)
def _blank_index(dim: int):
    return faiss.IndexFlatIP(dim)


def embed_chunks(chunks: List[Chunk]):
    model = get_embedder()
    vectors = model.encode([c.text for c in chunks], normalize_embeddings=True)
    index = _blank_index(vectors.shape[1])
    index.add(vectors)
    return index, chunks


def save_index(index, metas: List[Chunk]):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)


def load_index():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        raise FileNotFoundError("No index. Build it first.")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metas = pickle.load(f)
    return index, metas


def ensure_llm_client():
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_endpoint and azure_key:
        if AzureOpenAI is None:
            raise RuntimeError("Upgrade 'openai' package for Azure support: pip install --upgrade openai")
        return AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        ), True
    # OpenAI default
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or AZURE_* environment variables.")
    return OpenAI(api_key=api_key), False


def llm_answer(question: str, contexts: List[Chunk]) -> str:
    client, is_azure = ensure_llm_client()
    context_text = "\n\n".join([f"[Source: {os.path.basename(c.source)}]\n{c.text}" for c in contexts])

    sys_prompt = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "If the answer isn't in the context, say you don't know."
    )

    model = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT") if is_azure
        else os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_text}\n\nCite sources by filename in brackets like [file.txt]."},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def build_index(docs_items: List[Tuple[str, str]]):
    chunks: List[Chunk] = []
    for path, text in docs_items:
        for ch in simple_chunk(text):
            chunks.append(Chunk(text=ch, source=path))
    if not chunks:
        raise RuntimeError("No chunks produced. Add documents first.")
    index, metas = embed_chunks(chunks)
    save_index(index, metas)
    return len(metas), len(docs_items)


def retrieve(question: str, k: int = 4) -> List[Chunk]:
    index, metas = load_index()
    model = get_embedder()
    qv = model.encode([question], normalize_embeddings=True)
    D, I = index.search(qv, k)
    hits = []
    for idx in I[0]:
        if idx == -1:
            continue
        hits.append(metas[idx])
    return hits

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Mini RAG", page_icon="📚", layout="wide")
st.title("📚 Mini RAG — Streamlit UI (PDF-ready)")
st.caption("Local embeddings (MiniLM) + FAISS + OpenAI/Azure OpenAI + PDF ingestion")

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.number_input("Top-K chunks", min_value=1, max_value=20, value=4, step=1)
    st.markdown("**Provider:** auto-detected via env vars")
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        st.info("Azure OpenAI detected. Using deployment from AZURE_OPENAI_DEPLOYMENT.")
    elif os.getenv("OPENAI_API_KEY"):
        st.info(f"OpenAI detected. Model: {os.getenv('OPENAI_CHAT_MODEL', DEFAULT_CHAT_MODEL)}")
    else:
        st.warning("No API credentials detected. Set OPENAI_API_KEY or AZURE_* envs.")

    st.divider()
    st.markdown("**Docs source**")
    use_docs_folder = st.toggle("Use existing `docs/` folder", value=True)

    uploaded = st.file_uploader(
        "Or upload .txt/.md/.pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    if st.button("📦 Build / Refresh Index", type="primary"):
        try:
            items: List[Tuple[str, str]] = []
            if use_docs_folder:
                os.makedirs(DATA_DIR, exist_ok=True)
                items.extend(read_documents(DATA_DIR))
            # Include uploaded files and persist to docs/
            for f in uploaded or []:
                name = f.name
                data = f.read()
                os.makedirs(DATA_DIR, exist_ok=True)
                save_path = os.path.join(DATA_DIR, name)
                with open(save_path, "wb") as out:
                    out.write(data)
                if name.lower().endswith((".txt", ".md")):
                    text = data.decode("utf-8", errors="ignore")
                elif name.lower().endswith(".pdf"):
                    text = extract_text_from_pdf_bytes(data)
                else:
                    text = ""
                if text.strip():
                    items.append((save_path, text))
                else:
                    st.warning(f"No extractable text in {name}, skipping.")

            n_chunks, n_files = build_index(items)
            st.success(f"Indexed {n_chunks} chunks from {n_files} files.")
        except Exception as e:
            st.error(f"Index build failed: {e}")

st.subheader("🔎 Ask a question")
question = st.text_input("Your question", placeholder="e.g., What is the leave policy?")

colA, colB = st.columns([1, 2])
with colA:
    ask_btn = st.button("Ask", type="primary")
with colB:
    show_ctx = st.checkbox("Show retrieved context", value=True)

if ask_btn and question.strip():
    try:
        with st.spinner("Retrieving..."):
            ctx = retrieve(question, k=top_k)
        if not ctx:
            st.warning("No context retrieved. Did you build the index?")
        else:
            if show_ctx:
                with st.expander("Retrieved context chunks", expanded=False):
                    for i, c in enumerate(ctx, 1):
                        st.markdown(f"**[{i}] {os.path.basename(c.source)}**")
                        st.code(c.text[:2000])

            with st.spinner("Calling LLM..."):
                answer = llm_answer(question, ctx)
            st.markdown("### ✅ Answer")
            st.write(answer)
    except Exception as e:
        st.error(f"Query failed: {e}")

st.divider()
st.caption("Tip: Add .txt/.md/.pdf files to the `docs/` folder, click Build, then ask questions.")