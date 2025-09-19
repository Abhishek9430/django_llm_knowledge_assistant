# knowledge/utils/rag.py
import os
import numpy as np
import faiss
from typing import List, Tuple
from django.conf import settings
from ..models import Chunk
import openai
import requests

openai.api_key = os.getenv("OPENAI_API_KEY")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss.index")
TOP_K = int(os.getenv("TOP_K", 5))


def load_embeddings_and_meta():
    """
    Load all chunk embeddings from DB and return (emb_matrix, metas)
    metas is list of dicts: {"id": chunk_id, "text": chunk_text, "doc": doc.filename}
    """
    chunks = Chunk.objects.all()
    metas = []
    embs = []
    for c in chunks:
        if c.embedding is None:
            continue
        arr = np.frombuffer(c.embedding, dtype=np.float32)
        embs.append(arr)
        metas.append({"id": c.id, "text": c.chunk_text, "doc": c.document.filename})
    if len(embs) == 0:
        return None, []
    emb_matrix = np.vstack(embs).astype(np.float32)
    return emb_matrix, metas


def build_faiss_index(emb_matrix: np.ndarray) -> faiss.IndexFlatIP:
    dim = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    # normalize vectors for cosine (if using inner product)
    faiss.normalize_L2(emb_matrix)
    index.add(emb_matrix)
    return index


def get_top_k_chunks(query: str, k: int = TOP_K, use_openai=True):
    # embed query
    if use_openai and openai.api_key:
        emb = openai.Embedding.create(input=[query], model="text-embedding-3-small")[
            "data"
        ][0]["embedding"]
        qvec = np.array(emb, dtype=np.float32)
    else:
        # fallback: load sentence-transformers dynamically
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        )
        qvec = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)

    # load index and metas
    emb_matrix, metas = load_embeddings_and_meta()
    if emb_matrix is None:
        return []

    # build index each time (ok for small datasets). For prod, persist index.
    faiss.normalize_L2(emb_matrix)
    index = build_faiss_index(emb_matrix)
    qvec = qvec.reshape(1, -1)
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metas):
            continue
        m = metas[idx].copy()
        m["score"] = float(score)
        results.append(m)
    return results


# --- RAG prompt assembly + LLM call ---------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the question. "
    "When you use facts from the context, mention which document the facts came from. "
    "If the answer is not contained in the context, say you don't know and suggest next steps."
)


def generate_answer(question: str, top_chunks):
    DEEPSEEK_API_KEY = os.getenv("OPENAI_API_KEY")  # key is of deepseek
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(top_chunks)])
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer clearly, and reference sources."

    payload = {
        "model": "deepseek-chat",  # or deepseek-reasoner
        "messages": [
            {"role": "system", "content": "You are a helpful knowledge assistant."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
