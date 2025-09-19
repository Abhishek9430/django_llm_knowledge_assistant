# knowledge/utils/ingest.py
import os
import io
import numpy as np
from typing import List, Tuple
from django.conf import settings
from pdfplumber import open as pdf_open
from sentence_transformers import SentenceTransformer
import openai
from ..models import Document, Chunk

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
MAX_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", 500))

openai.api_key = OPENAI_API_KEY

# --- chunking ---------------------------------------------------------------
def simple_chunk_text(text: str, max_chars: int = 1500) -> List[Tuple[str,int,int]]:
    """
    Very simple character-based chunker returning (chunk, start, end)
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        chunks.append((chunk.strip(), start, end))
        start = end
    return chunks

# --- extractors -------------------------------------------------------------
def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    text_parts = []
    with pdf_open(file_stream) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def extract_text_from_bytes(filename: str, file_bytes: bytes) -> str:
    # basic dispatch by extension
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(io.BytesIO(file_bytes))
    else:
        # assume plain text or markdown
        return file_bytes.decode(errors="ignore")

# --- embedding helpers -----------------------------------------------------
def embed_with_openai(texts: List[str], model: str = "text-embedding-3-small") -> List[np.ndarray]:
    # call OpenAI embeddings in batches
    out = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        resp = openai.Embedding.create(input=batch, model=model)
        for item in resp["data"]:
            out.append(np.array(item["embedding"], dtype=np.float32))
    return out

def embed_with_sentence_transformer(texts: List[str], model_name=SENTENCE_TRANSFORMER_MODEL):
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [emb.astype(np.float32) for emb in embs]

# --- main ingestion --------------------------------------------------------
def ingest_document(file_name: str, file_bytes: bytes, source: str = None, use_openai=False):
    text = extract_text_from_bytes(file_name, file_bytes)
    doc = Document.objects.create(filename=file_name, source=source or "")
    # chunk
    chunks_meta = simple_chunk_text(text, max_chars=1500)
    chunk_texts = [c[0] for c in chunks_meta]

    # embed
    if use_openai and OPENAI_API_KEY:
        embeddings = embed_with_openai(chunk_texts)
    else:
        embeddings = embed_with_sentence_transformer(chunk_texts)

    # save chunks
    for (chunk_text, start, end), emb in zip(chunks_meta, embeddings):
        Chunk.objects.create(
            document=doc,
            chunk_text=chunk_text,
            start_offset=start,
            end_offset=end,
            embedding=emb.tobytes()
        )
    return doc
