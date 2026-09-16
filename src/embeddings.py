"""Sentence-transformers embeddings (CPU-friendly)."""

from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns list of float vectors."""
    if not texts:
        return []
    model = _model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    )
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    v = embed_one("Quantum error correction for surface codes")
    print(f"dim={len(v)} time={time.perf_counter() - t0:.2f}s")
