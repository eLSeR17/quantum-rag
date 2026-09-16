"""Hybrid retriever: vector similarity + BM25 + metadata filtering.

Pure-Python BM25 implementation (no external index required) keeps the
dependency tree small and the pipeline CPU-only.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from . import config, store

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 over a static document set (built from store metadata)."""

    def __init__(self, docs: list[str]):
        self.doc_terms: list[Counter] = []
        self.df: Counter = Counter()
        self.N = len(docs)
        self.avgdl = 0.0
        total_len = 0
        for d in docs:
            terms = Counter(_tokenize(d))
            self.doc_terms.append(terms)
            for t in terms:
                self.df[t] += 1
            total_len += sum(terms.values())
        self.avgdl = total_len / self.N if self.N else 1.0
        self.k1 = 1.5
        self.b = 0.75

    def score(self, query: str, doc_idx: int) -> float:
        qterms = Counter(_tokenize(query))
        d = self.doc_terms[doc_idx]
        dl = sum(d.values())
        s = 0.0
        for t, qf in qterms.items():
            tf = d.get(t, 0)
            if not tf:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * (num / den) * (qf ** 0.5)
        return s


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    mx = max(scores)
    mn = min(scores)
    rng = mx - mn
    if rng == 0:
        return [1.0] * len(scores)
    return [(s - mn) / rng for s in scores]


def hybrid_search(query: str, top_k: int | None = None, doc_filter: str | None = None) -> list[dict]:
    """Vector + BM25 hybrid with optional doc_id metadata filter."""
    top_k = top_k or config.TOP_K
    alpha = config.ALPHA_HYBRID

    # 1) Vector hits
    qvec = _embed_query(query)
    vec_results = store.query(qvec, top_k=max(top_k * 4, 20))

    # 2) BM25 over the same candidate pool (cheap, local)
    pool_texts = [r["text"] for r in vec_results]
    bm25 = BM25(pool_texts)
    bm25_scores = [bm25.score(query, i) for i in range(len(pool_texts))]
    bm25_norm = _normalize(bm25_scores)

    # 3) Metadata filter (doc-level)
    if doc_filter:
        pool = [r for r in vec_results if r["metadata"].get("doc_id") == doc_filter]
        if not pool:
            pool = vec_results
    else:
        pool = vec_results

    # 4) Fuse + rank
    fused = []
    for idx, r in enumerate(pool):
        vscore = r.get("score", 0.0)
        bscore = bm25_norm[idx] if idx < len(bm25_norm) else 0.0
        fused.append({**r, "score": alpha * vscore + (1 - alpha) * bscore})
    fused.sort(key=lambda r: r["score"], reverse=True)
    return fused[:top_k]


def _embed_query(query: str):
    from .embeddings import embed_one

    return embed_one(query)


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "surface codes error correction overhead"
    results = hybrid_search(q, top_k=5)
    for r in results:
        print(f"  {r['score']:.3f}  {r['text'][:90]}...")
