"""Deterministic tests for the eval pipeline (CI-safe, no chromadb/torch)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_recall_at_k():
    from src.evals import recall_at_k
    assert recall_at_k(["a", "b", "c"], "b", k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], "d", k=3) == 0.0
    assert recall_at_k(["a", "b", "c"], "a", k=1) == 1.0


def test_mrr():
    from src.evals import mrr
    assert mrr(["a", "b", "c"], "a") == 1.0
    assert mrr(["a", "b", "c"], "b") == 0.5
    assert mrr(["a", "b", "c"], "c") == 1.0 / 3
    assert mrr(["a", "b", "c"], "d") == 0.0


def test_heuristic_judge():
    from src.evals import HeuristicJudge
    judge = HeuristicJudge()
    result = judge.score(
        question="What is the surface code threshold?",
        retrieved_texts=[
            "The surface code achieves a threshold error rate of approximately 1%.",
            "Surface codes are topological codes for quantum error correction.",
        ],
        ground_truth="The threshold defines the maximum error rate below which fault-tolerant computation is possible.",
        expected_keywords=["threshold", "surface code", "error rate"],
    )
    assert 0.0 <= result["faithfulness"] <= 1.0
    assert 0.0 <= result["relevance"] <= 1.0
    assert result["faithfulness"] > 0.3  # should find overlap with ground truth


def test_load_golden_set():
    from src.evals import load_golden_set
    cases = load_golden_set()
    assert len(cases) == 12
    assert all(c.id.startswith("qr-") for c in cases)
    assert all(c.relevant_doc_id.startswith("arxiv-") for c in cases)
    assert all(c.expected_keywords for c in cases)
