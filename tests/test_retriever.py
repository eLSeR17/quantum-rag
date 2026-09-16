"""Tests for BM25 retrieval -- isolated, no chromadb/sentence-transformers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_bm25_ranking():
    from src.retriever import BM25
    docs = [
        "surface code threshold error rate fault tolerance",
        "quantum machine learning variational circuit",
        "decoherence NISQ noise channel qubit",
    ]
    bm25 = BM25(docs)
    scores = [bm25.score("surface code threshold", i) for i in range(3)]
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_bm25_empty():
    from src.retriever import BM25
    bm25 = BM25([])
    assert bm25.N == 0
    assert bm25.avgdl == 1.0


def test_bm25_single_doc():
    from src.retriever import BM25
    bm25 = BM25(["hello world foo bar"])
    s = bm25.score("hello world", 0)
    assert s > 0.0


def test_bm25_no_match():
    from src.retriever import BM25
    bm25 = BM25(["alpha beta gamma"])
    s = bm25.score("quantum error correction", 0)
    assert s == 0.0


def test_normalize():
    from src.retriever import _normalize
    assert _normalize([]) == []
    assert _normalize([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]
    normed = _normalize([1.0, 3.0, 5.0])
    assert normed[0] == 0.0
    assert normed[2] == 1.0
    assert 0.0 < normed[1] < 1.0
