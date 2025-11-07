#!/usr/bin/env python3
# Chat UI for Mini RAG (FAISS + MiniLM + OpenAI/Azure + PDF Support)

import os
import io
import glob
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

# LLM Clients
try:
    from openai import OpenAI, AzureOpenAI
except:
    from openai import OpenAI
    AzureOpenAI = None

load_dotenv()

DATA_DIR = "docs"
INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class Chunk:
    text: str
    source: str


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


def read_txt_md(folder):
    paths = glob.glob(folder + "/**/*.txt", recursive=True)
    paths += glob.glob(folder + "/**/*.md", recursive=True)
    out = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf8", errors="ignore") as f:
                text = f.read().strip()
                if text:
                    out.append((p, text))
        except:
            pass
    return out


def extract_pdf_text_bytes(data):
    try:
        r = PdfReader(io.BytesIO(data))
        return "\n".join([(p.extract_text() or "") for p in r.pages]).strip()
    except:
        return ""


def extract_pdf_text_path(path):
    try:
        r = PdfReader(path)
        return "\n".join([(p.extract_text() or "") for p in r.pages]).strip()
    except:
        return ""


def read_pdfs(folder):
    paths = glob.glob(folder + "/**/*.pdf", recursive=True)
    out = []
    for p in paths:
        txt = extract_pdf_text_path(p)
        if txt:
            out.append((p, txt))
    return out


def read_documents(folder):
    return read_txt_md(folder) + read_pdfs(folder)


def chunk_text(text, size=350, overlap=50):
    words = text.split()
    chunks = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + size]))
    return chunks


@st.cache_resource
def blank_index(dim):
    return faiss.IndexFlatIP(dim)


def embed_chunks(chunks):
    model = get_embedder()
    vecs = model.encode([c.text for c in chunks], normalize_embeddings=True)
    idx = blank_index(vecs.shape[1])
    idx.add(vecs)
    return idx, chunks


def save_index(idx, meta):
    faiss.write_index(idx, INDEX_PATH)
    pickle.dump(meta, open(META_PATH, "wb"))


def load_index():
    idx = faiss.read_index(INDEX_PATH)
    meta = pickle.load(open(META_PATH, "rb"))
    return idx, meta


def ensure_llm_client():
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        ), True
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY")), False


def llm_answer(question, context_chunks):
    client, is_azure = ensure_llm_client()

    context_text = "\n\n".join(
        f"[{os.path.basename(c.source)}]\n{c.text}" for c in context_chunks
    )

    model = os.getenv("AZURE_OPENAI_DEPLOYMENT") if is_azure else os.getenv("OPENAI_CHAT_MODEL", DEFAULT_MODEL)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer only using provided context. If not present, say you don't know."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_text}"},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()


def build_index(docs):
    chunks = []
    for path, text in docs:
        for ch in chunk_text(text):
            chunks.append(Chunk(ch, path))
    idx, meta = embed_chunks(chunks)
    save_index(idx, meta)


def retrieve(question, k):
    idx, meta = load_index()
    qvec = get_embedder().encode([question], normalize_embeddings=True)
    _, I = idx.search(qvec, k)
    return [meta[i] for i in I[0] if i != -1]


# ------------------- UI -------------------
st.set_page_config(page_title="Mini RAG Chat", page_icon="💬", layout="wide")
st.title("💬 Mini RAG — Chat with Your Documents")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.sidebar:
    st.header("📄 Load Documents")
    k = st.number_input("Top-K Retrieval", 1, 12, 4)

    use_docs = st.toggle("Use docs/ folder", True)
    uploaded = st.file_uploader("Upload .txt/.md/.pdf", type=["txt", "md", "pdf"], accept_multiple_files=True)

    if st.button("📦 Build / Refresh Index"):
        docs = []
        if use_docs:
            docs += read_documents(DATA_DIR)
        for f in uploaded or []:
            data = f.read()
            path = os.path.join(DATA_DIR, f.name)
            os.makedirs(DATA_DIR, exist_ok=True)
            open(path, "wb").write(data)
            txt = data.decode("utf8", errors="ignore") if f.name.endswith((".txt", ".md")) else extract_pdf_text_bytes(data)
            if txt:
                docs.append((path, txt))
        build_index(docs)
        st.success("Index built ✅")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat = []


# Display chat history
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

# Chat input
q = st.chat_input("Ask something about your documents…")
if q:
    st.session_state.chat.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    ctx = retrieve(q, k)
    ans = llm_answer(q, ctx) if ctx else "No relevant information found. Try indexing documents first."

    st.session_state.chat.append(("assistant", ans))
    with st.chat_message("assistant"):
        st.markdown(ans)
