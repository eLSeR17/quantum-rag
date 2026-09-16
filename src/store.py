"""ChromaDB persistent vector store."""

from __future__ import annotations

from . import config


def _client():
    import chromadb

    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def _collection(client):
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def add_chunks(chunks: list[dict], vectors: list[list[float]]) -> int:
    """Add chunk dicts (with .text/.metadata) + precomputed vectors."""
    client = _client()
    coll = _collection(client)
    existing = set(coll.get(include=[])["ids"])
    ids, docs, metas, vecs = [], [], [], []
    for chunk, vec in zip(chunks, vectors):
        cid = chunk["id"]
        if cid in existing:
            continue
        ids.append(cid)
        docs.append(chunk["text"])
        metas.append(chunk.get("metadata") or {})
        vecs.append(vec)
    if ids:
        coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)
    return len(ids)


def count() -> int:
    client = _client()
    coll = _collection(client)
    return coll.count()


def query(query_vector: list[float], top_k: int = None) -> list[dict]:
    """Vector-only query; returns top-k chunks with metadata + distances."""
    top_k = top_k or config.TOP_K
    client = _client()
    coll = _collection(client)
    res = coll.query(query_embeddings=[query_vector], n_results=top_k, include=["documents", "metadatas", "distances"])
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": doc, "metadata": meta, "distance": dist, "score": 1.0 - dist})
    return out


def query_all(field: str = "doc_id") -> list[str]:
    """All distinct values of a metadata field (for analytics)."""
    client = _client()
    coll = _collection(client)
    res = coll.get(include=["metadatas"])
    return sorted({m.get(field) for m in res["metadatas"] if m.get(field)})


def metadata_all() -> list[dict]:
    client = _client()
    coll = _collection(client)
    return coll.get(include=["metadatas"])["metadatas"]


if __name__ == "__main__":
    print(f"chunks in store: {count()}")
    print(f"docs: {query_all()}")
