"""In-memory RAG store — chunk scraped pages, embed, cosine-similarity search."""

import math
import logging

from tools.embeddings import embed

logger = logging.getLogger("shopping_advisor.tools")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class RagStore:
    """Chunk and embed pages once; retrieve top-k chunks by cosine similarity."""

    def __init__(self, chunk_size: int = 700, overlap: int = 100):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._chunks: list[str] = []
        self._vectors: list[list[float]] = []

    def add(self, pages: list[str]) -> None:
        """Chunk and embed *pages*. Must be called before any query()."""
        chunks: list[str] = []
        for page in pages:
            chunks.extend(self._chunk(page))
        if not chunks:
            return

        print(f"\n[RAG] Embedding {len(chunks)} chunks from {len(pages)} pages...")
        vectors = embed(chunks)
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)
        print(f"[RAG] Store ready — {len(self._chunks)} chunks indexed.")

    def query(self, text: str, top_k: int = 6) -> list[str]:
        """Return top-*k* most relevant chunks for *text*."""
        if not self._vectors:
            return []
        q_vec = embed([text])[0]
        scored = sorted(
            range(len(self._vectors)),
            key=lambda i: _cosine(q_vec, self._vectors[i]),
            reverse=True,
        )
        return [self._chunks[i] for i in scored[:top_k]]

    def _chunk(self, text: str) -> list[str]:
        step = max(1, self._chunk_size - self._overlap)
        chunks = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self._chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks
