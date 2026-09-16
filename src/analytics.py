"""Corpus analytics: trends over the quantum literature corpus.

All statistics are computed deterministically from the vector-store metadata
(no LLM). Dates, authors, categories and keywords come straight from arXiv.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime

from . import config
from . import store

_KEYWORD_RE = re.compile(r"[a-z][a-z-]{3,}")


def compute_analytics() -> dict:
    """Compute corpus-level analytics from store metadata."""
    metas = store.metadata_all()
    if not metas:
        return {"empty": True}

    years = Counter()
    cats = Counter()
    authors = Counter()
    keywords = Counter()
    n_chunks = len(metas)
    n_docs = len({m.get("doc_id") for m in metas if m.get("doc_id")})

    for m in metas:
        pub = (m.get("published") or "")[:4]
        if pub:
            years[pub] += 1
        for c in (m.get("categories") or []):
            cats[c] += 1
        for a in (m.get("authors") or []):
            authors[a] += 1
        kw = _KEYWORD_RE.findall((m.get("abstract") or "").lower())
        keywords.update(kw)

    top_keywords = [{"keyword": k, "count": n} for k, n in keywords.most_common(20)]
    return {
        "empty": False,
        "n_chunks": n_chunks,
        "n_docs": n_docs,
        "years": [{"year": y, "count": n} for y, n in sorted(years.items())],
        "categories": [{"category": c, "count": n} for c, n in cats.most_common(15)],
        "top_authors": [{"author": a, "count": n} for a, n in authors.most_common(15)],
        "top_keywords": top_keywords,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_analytics() -> None:
    config.ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.ANALYTICS_DIR / "analytics.json"
    out.write_text(json.dumps(compute_analytics(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[analytics] wrote {out}")


if __name__ == "__main__":
    save_analytics()
