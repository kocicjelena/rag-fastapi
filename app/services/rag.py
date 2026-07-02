"""
RAG service: text chunking, embedding generation, and answer synthesis.
"""
import re
import uuid
from typing import Generator

from openai import OpenAI

from app.core.config import settings
from app.models import DocumentChunk

_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


# ──────────────────────────── Chunking ────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, respecting sentence/paragraph boundaries."""
    # Normalise whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at a paragraph or sentence boundary
        if end < text_len:
            for boundary in ["\n\n", "\n", ". ", "! ", "? ", " "]:
                idx = text.rfind(boundary, start, end)
                if idx > start:
                    end = idx + len(boundary)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap
        if start >= text_len:
            break

    return chunks


# ──────────────────────────── Embeddings ────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of strings. Returns list of float vectors."""
    client = get_openai_client()
    response = client.embeddings.create(
        input=texts,
        model=settings.EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIMENSIONS,
    )
    return [item.embedding for item in response.data]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


# ──────────────────────────── Document ingestion ────────────────────────────

def ingest_text(
    text: str,
    document_id: uuid.UUID,
) -> list[DocumentChunk]:
    """Chunk raw text and generate embeddings; returns list of DocumentChunk objects."""
    raw_chunks = chunk_text(text)
    if not raw_chunks:
        return []

    embeddings = embed_texts(raw_chunks)

    chunks: list[DocumentChunk] = []
    for idx, (content, embedding) in enumerate(zip(raw_chunks, embeddings)):
        chunks.append(
            DocumentChunk(
                document_id=document_id,
                content=content,
                chunk_index=idx,
                embedding=embedding,
            )
        )
    return chunks


# ──────────────────────────── Answer synthesis ────────────────────────────

def build_context(chunk_contents: list[str]) -> str:
    return "\n\n---\n\n".join(chunk_contents)


def generate_answer(question: str, context: str) -> str:
    """Call the LLM with a RAG prompt."""
    client = get_openai_client()
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "context provided below. If the answer is not present in the context, say "
        "'I don't have enough information to answer that question.' "
        "Be concise and accurate.\n\n"
        f"CONTEXT:\n{context}"
    )
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""


def generate_answer_stream(question: str, context: str) -> Generator[str, None, None]:
    """Streaming version of generate_answer."""
    client = get_openai_client()
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided context. If the answer is not in the context, say "
        "'I don't have enough information to answer that question.'\n\n"
        f"CONTEXT:\n{context}"
    )
    with client.chat.completions.stream(
        model=settings.LLM_MODEL,
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    ) as stream:
        for text in stream.text_stream:
            yield text
