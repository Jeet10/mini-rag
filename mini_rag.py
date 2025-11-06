#!/usr/bin/env python3
# Minimal RAG: index .txt/.md files in docs/, retrieve with FAISS, answer with OpenAI/Azure OpenAI

import argparse
import glob
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# OpenAI clients
try:
    from openai import OpenAI, AzureOpenAI
except Exception:
    # Older openai package fallback
    from openai import OpenAI  # type: ignore
    AzureOpenAI = None  # type: ignore

load_dotenv()

DATA_DIR = "docs"
INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast


@dataclass
class Chunk:
    text: str
    source: str


# ---------- IO ----------
def read_text_files(folder: str) -> List[Tuple[str, str]]:
    paths = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
    paths += glob.glob(os.path.join(folder, "**", "*.md"), recursive=True)
    items = []
    for p in sorted(set(paths)):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                items.append((p, f.read()))
        except Exception as e:
            print(f"Skip {p}: {e}")
    return items


# ---------- Chunking ----------
def simple_chunk(text: str, max_tokens: int = 350, overlap: int = 50) -> List[str]:
    """Very simple word-based chunker (token-agnostic)."""
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    step = max(1, max_tokens - overlap)
    while i < len(words):
        chunk_words = words[i : i + max_tokens]
        chunks.append(" ".join(chunk_words))
        i += step
    return chunks


# ---------- Embeddings + FAISS ----------
def embed_chunks(chunks: List[Chunk]) -> Tuple[faiss.IndexFlatIP, List[Chunk]]:
    model = SentenceTransformer(EMBED_MODEL_NAME)
    vecs = model.encode([c.text for c in chunks], normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, chunks


def save_index(index: faiss.IndexFlatIP, metas: List[Chunk]) -> None:
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)


def load_index() -> Tuple[faiss.IndexFlatIP, List[Chunk]]:
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        raise FileNotFoundError(
            "Index not found. Build it first with: python mini_rag.py index"
        )
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metas = pickle.load(f)
    return index, metas


# ---------- LLM ----------
def ensure_llm_client():
    """
    Returns an OpenAI or Azure OpenAI client based on env vars.
    Azure usage requires:
      AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_endpoint and azure_key:
        if AzureOpenAI is None:
            raise RuntimeError(
                "Your 'openai' package is too old for AzureOpenAI. Upgrade: pip install --upgrade openai"
            )
        return AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )
    # Default: OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY (or Azure env vars) before running."
        )
    return OpenAI(api_key=api_key)


def llm_answer(question: str, contexts: List[Chunk]) -> str:
    client = ensure_llm_client()
    sys_prompt = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "If the answer isn't in the context, say you don't know."
    )
    context_text = "\n\n".join(
        [f"[Source: {os.path.basename(c.source)}]\n{c.text}" for c in contexts]
    )

    # Model name:
    # - For OpenAI, set OPENAI_CHAT_MODEL (defaults to gpt-4o-mini)
    # - For Azure, set AZURE_OPENAI_DEPLOYMENT (your chat deployment name)
    model = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT",
        os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nContext:\n{context_text}\n\n"
                    "Cite sources by filename in brackets like [notes.txt]."
                ),
            },
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ---------- Retrieval ----------
def retrieve(question: str, k: int = 4) -> List[Chunk]:
    index, metas = load_index()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    qv = model.encode([question], normalize_embeddings=True)
    D, I = index.search(qv, k)
    hits = []
    for idx in I[0]:
        if idx == -1:
            continue
        hits.append(metas[idx])
    return hits


# ---------- Commands ----------
def build() -> None:
    docs = read_text_files(DATA_DIR)
    if not docs:
        print(f"No .txt/.md files found in {DATA_DIR}/")
        return
    chunks: List[Chunk] = []
    for path, text in docs:
        for ch in simple_chunk(text):
            chunks.append(Chunk(text=ch, source=path))
    index, metas = embed_chunks(chunks)
    save_index(index, metas)
    print(f"Indexed {len(metas)} chunks from {len(docs)} files.")


def ask(q: str) -> None:
    ctx = retrieve(q, k=int(os.getenv("TOP_K", "4")))
    if not ctx:
        print("No context retrieved. Did you build the index?")
        return
    answer = llm_answer(q, ctx)
    print("\n=== ANSWER ===\n")
    print(answer)


def main():
    ap = argparse.ArgumentParser(description="Mini RAG (FAISS + MiniLM + OpenAI/Azure)")
    ap.add_argument("command", choices=["index", "ask"])
    ap.add_argument("query", nargs="?", help="Question to ask (for 'ask')")
    args = ap.parse_args()

    if args.command == "index":
        os.makedirs(DATA_DIR, exist_ok=True)
        build()
    elif args.command == "ask":
        if not args.query:
            raise SystemExit('Provide a question, e.g. python mini_rag.py ask "What is X?"')
        ask(args.query)


if __name__ == "__main__":
    main()
