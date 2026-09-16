"""Lexical grounding: verify that retrieved evidence supports the query.

No LLM needed — pure lexical overlap + metadata validation. This is what
makes the system honest: we only claim grounding when the evidence actually
matches the query terms.
"""

from __future__ import annotations

import re
from . import config

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOP = frozenset(
    "the and are is was were been being have has had do does did will would "
    "shall should may might can could a an in on at to for of with by from "
    "this that these those it its".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP}


def grounding_verdict(query: str, results: list[dict]) -> dict:
    """Score + verdict for a set of retrieved chunks against the query."""
    qtokens = _tokens(query)
    if not qtokens:
        return {"score": 0.0, "verdict": "N/A", "supporting_chunks": 0, "evidence": []}

    supporting = 0
    evidence = []
    for r in results:
        ctokens = _tokens(r["text"])
        shared = qtokens & ctokens
        ratio = len(shared) / len(qtokens) if qtokens else 0.0
        if len(shared) >= config.GROUNDING_MIN_SHARED:
            supporting += 1
            evidence.append(
                {
                    "text": r["text"][:300],
                    "shared_tokens": sorted(shared),
                    "ratio": round(ratio, 2),
                    "score": round(r.get("score", 0.0), 3),
                }
            )

    total_ratio = supporting / len(results) if results else 0.0
    score = round(total_ratio * 2 + (len(results) > 0) * 0.3, 3)
    score = min(score, 1.0)

    if score >= config.GROUNDING_MIN_SCORE:
        verdict = "GROUNDING: Answer is supported by retrieved evidence"
    else:
        verdict = "NO GROUNDING: Evidence insufficient or irrelevant"

    return {
        "score": score,
        "verdict": verdict,
        "supporting_chunks": supporting,
        "total_chunks": len(results),
        "evidence": evidence[:5],  # top 5 supporting
    }


if __name__ == "__main__":
    fake_results = [
        {"text": "Surface codes achieve high error correction thresholds", "score": 0.8},
        {"text": "The magic angle in graphene is 1.1 degrees", "score": 0.3},
    ]
    print(grounding_verdict("surface codes error correction", fake_results))
