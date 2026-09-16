"""Eval pipeline for QuantumRAG: HeuristicJudge + OllamaJudge + metrics.

Deterministic by default (no GPU, no network). Ollama judge is optional and
activates only when Ollama is reachable at http://localhost:11434.

Scoring (both judges): faithfulness 0-5 + relevance 0-5, normalized to [0,1].
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from . import config

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    topic: str
    expected_keywords: list[str]
    relevant_doc_id: str
    relevant_doc_hint: str
    ground_truth: str


@dataclass
class RetrievalResult:
    """Result for one eval case after retrieval + judgment."""
    case: EvalCase
    retrieved_doc_ids: list[str]
    recall_at_k: float
    mrr: float
    faithfulness: float   # 0-1
    relevance: float      # 0-1
    judge: str            # "heuristic" | "ollama"
    reasoning: str = ""


def load_golden_set(path: str | Path | None = None) -> list[EvalCase]:
    path = Path(path or (config.PROJECT_ROOT / "data/evals/golden_set.json"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase(**c) for c in raw]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_id: str, k: int) -> float:
    """1 if relevant_doc_id appears in top-k results, else 0."""
    return 1.0 if relevant_doc_id in retrieved_doc_ids[:k] else 0.0


def mrr(retrieved_doc_ids: list[str], relevant_doc_id: str) -> float:
    """Mean reciprocal rank: 1/rank of the first hit, 0 if not found."""
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id == relevant_doc_id:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------

class Judge(Protocol):
    def score(self, question: str, retrieved_texts: list[str],
              ground_truth: str, expected_keywords: list[str]) -> dict:
        ...


@dataclass
class JudgeScore:
    faithfulness: float
    relevance: float
    reasoning: str = ""


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOP = frozenset(
    ["the", "and", "are", "is", "was", "were", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "can", "could", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by", "from", "this", "that", "these", "those", "it", "its"]
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP}


class HeuristicJudge:
    """Deterministic judge: lexical overlap + keyword coverage. No LLM needed."""

    def score(self, question: str, retrieved_texts: list[str],
              ground_truth: str, expected_keywords: list[str]) -> dict:
        combined = " ".join(retrieved_texts)
        r_tokens = _tokens(combined)
        gt_tokens = _tokens(ground_truth)
        q_tokens = _tokens(question)

        # Faithfulness: how much of the ground truth is in the retrieved evidence
        if gt_tokens:
            faithfulness = len(r_tokens & gt_tokens) / len(gt_tokens)
        else:
            faithfulness = 0.0

        # Relevance: how much of the query terms appear in the retrieved evidence
        if q_tokens:
            relevance = len(r_tokens & q_tokens) / len(q_tokens)
        else:
            relevance = 0.0

        # Keyword bonus: expected keywords that appear in retrieved text
        kw_hits = sum(1 for kw in expected_keywords if kw.lower() in combined.lower())
        kw_bonus = kw_hits / len(expected_keywords) if expected_keywords else 0.0

        faithfulness = min(faithfulness + kw_bonus * 0.2, 1.0)
        relevance = min(relevance + kw_bonus * 0.1, 1.0)

        return {
            "faithfulness": round(faithfulness, 3),
            "relevance": round(relevance, 3),
            "reasoning": f"faithfulness={faithfulness:.3f} relevance={relevance:.3f} kw_hits={kw_hits}/{len(expected_keywords)}",
        }


class OllamaJudge:
    """LLM-as-judge via Ollama (optional, requires http://localhost:11434)."""

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def score(self, question: str, retrieved_texts: list[str],
              ground_truth: str, expected_keywords: list[str]) -> dict:
        combined = "\n".join(f"[{i+1}] {t[:300]}" for i, t in enumerate(retrieved_texts[:5]))
        prompt = (
            f"Rate the retrieved evidence (below) for answering the question.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved evidence:\n{combined}\n\n"
            f'Respond with ONLY a JSON object: {{"faithfulness": <0-5>, "relevance": <0-5>, "reasoning": "<brief>"}}'
        )
        try:
            data = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                               "format": "json", "options": {"temperature": 0.1, "num_predict": 200}}).encode()
            req = urllib.request.Request(f"{self.base_url}/api/generate", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                out = json.loads(resp.read())
            result = json.loads(out.get("response", "{}"))
            return {
                "faithfulness": min(max(result.get("faithfulness", 0) / 5.0, 0.0), 1.0),
                "relevance": min(max(result.get("relevance", 0) / 5.0, 0.0), 1.0),
                "reasoning": result.get("reasoning", "ollama judge"),
            }
        except Exception:  # noqa: BLE001
            # Fall back to heuristic if Ollama unreachable
            return HeuristicJudge().score(question, retrieved_texts, ground_truth, expected_keywords)


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

def run_eval(golden_path: str | None = None, top_k: int = 8, use_ollama: bool = False) -> dict:
    """Run the full eval loop and return a structured report."""
    cases = load_golden_set(golden_path)
    judge = OllamaJudge() if use_ollama else HeuristicJudge()
    judge_name = "ollama" if use_ollama else "heuristic"

    # Lazy imports — chromadb/sentence-transformers not needed at import time
    from .retriever import hybrid_search

    results = []
    for case in cases:
        hits = hybrid_search(case.question, top_k=top_k)
        # Dedupe preserving ranking order (set() would scramble it)
        seen: set[str] = set()
        doc_ids: list[str] = []
        for h in hits:
            d = h["metadata"].get("doc_id", "")
            if d and d not in seen:
                seen.add(d)
                doc_ids.append(d)
        texts = [h["text"] for h in hits]

        r_at_k = recall_at_k(doc_ids, case.relevant_doc_id, top_k)
        mm = mrr(doc_ids, case.relevant_doc_id)
        j = judge.score(case.question, texts, case.ground_truth, case.expected_keywords)

        results.append(RetrievalResult(
            case=case,
            retrieved_doc_ids=doc_ids[:top_k],
            recall_at_k=r_at_k,
            mrr=mm,
            faithfulness=j["faithfulness"],
            relevance=j["relevance"],
            judge=judge_name,
            reasoning=j["reasoning"],
        ))

    n = len(results)
    avg_recall = sum(r.recall_at_k for r in results) / n if n else 0
    avg_mrr = sum(r.mrr for r in results) / n if n else 0
    avg_faithfulness = sum(r.faithfulness for r in results) / n if n else 0
    avg_relevance = sum(r.relevance for r in results) / n if n else 0

    return {
        "n_cases": n,
        "judge": judge_name,
        "top_k": top_k,
        "recall_at_k": round(avg_recall, 3),
        "mrr": round(avg_mrr, 3),
        "faithfulness": round(avg_faithfulness, 3),
        "relevance": round(avg_relevance, 3),
        "results": [asdict(r) for r in results],
    }
