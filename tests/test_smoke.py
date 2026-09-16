"""Smoke tests for the QuantumRAG pipeline modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_chunker_sections():
    from src.chunker import chunk_document

    doc = (
        "Abstract\nWe study surface codes.\n\n"
        "Introduction\nQuantum computers are noisy.\n\n"
        "Results\nWe reach 99.9% fidelity.\n"
    )
    chunks = chunk_document(doc)
    assert len(chunks) >= 3
    assert all(c.text for c in chunks)


def test_bm25_rank():
    from src.retriever import BM25

    # Doc 0 shares all query terms; doc 1 one term; doc 2 none → desc order
    docs = [
        "Surface code error correction threshold.",
        "Quantum machine learning uses variational circuits.",
        "Graphene is a two-dimensional material.",
    ]
    bm25 = BM25(docs)
    scores = [bm25.score("surface code error correction threshold", i) for i in range(3)]
    assert scores[0] > 0.0
    assert scores[0] > scores[1]
    assert scores[1] >= scores[2]


def test_grounding_verdict():
    from src.grounding import grounding_verdict

    results = [
        {"text": "Surface codes achieve high error correction thresholds", "score": 0.8},
        {"text": "The magic angle in graphene is 1.1 degrees", "score": 0.3},
    ]
    v = grounding_verdict("surface codes error correction", results)
    assert v["verdict"].startswith("GROUNDING")
    assert v["supporting_chunks"] >= 1
    assert v["score"] >= 0.5

    # 0 supporting chunks → never claim grounding (honest failure)
    bad = [
        {"text": "Graphene has a magic angle at 1.1 degrees", "score": 0.5},
        {"text": "Neural networks classify images with high accuracy", "score": 0.4},
    ]
    v2 = grounding_verdict("surface codes error correction", bad)
    assert v2["verdict"].startswith("NO GROUNDING")
    assert v2["supporting_chunks"] == 0
