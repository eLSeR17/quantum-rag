"""QuantumRAG — Streamlit demo UI.

Multi-tab demo over a RAG corpus of quantum computing research:

* **Ask** -- grounded Q&A over the quantum papers corpus (evidence-first).
* **Analytics** -- trends over the corpus (years, categories, authors, keywords).
* **About** -- architecture, corpus, limitations, honest deployment notes.

Run:  streamlit run demo_ui/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Make the package importable when run from the repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src import analytics as qanalytics
    from src import grounding, retriever, store
    IMPORT_OK = True
except Exception as exc:  # noqa: BLE001  # pragma: no cover — deploy edge cases
    IMPORT_OK = False
    IMPORT_ERR = repr(exc)


@st.cache_data(show_spinner=False)
def _load_analytics() -> dict | None:
    path = ROOT / "data" / "analytics" / "analytics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Ask tab
# ---------------------------------------------------------------------------

def _ask_tab() -> None:
    st.header("Ask the quantum literature corpus")
    st.caption(
        "Hybrid retrieval over arXiv papers (quantum error correction, quantum "
        "machine learning). Every answer is grounded: retrieved chunks shown "
        "with a lexical grounding verdict — no LLM lies."
    )
    if not IMPORT_OK:
        st.error(f"Package import failed: {IMPORT_ERR}")
        return

    query = st.text_input(
        "Question",
        placeholder="e.g. What is the threshold of the surface code?",
    )
    top_k = st.slider("Retrieved chunks", 3, 12, 6)
    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Retrieving and grounding..."):
            results = retriever.hybrid_search(query, top_k=top_k)
            verdict = grounding.grounding_verdict(query, results)
        st.markdown(f"### Verdict\n**{verdict['verdict']}** (score {verdict['score']:.2f})")
        st.markdown(
            f"`{verdict['supporting_chunks']}/{verdict['total_chunks']} chunks support the query`"
        )
        for i, r in enumerate(results, 1):
            meta = r["metadata"] or {}
            doc = (meta.get("doc_id") or "").replace("arxiv-", "arXiv ")
            section = meta.get("section", "body")
            with st.expander(f"#{i} — {doc} · section '{section}' · score {r['score']:.3f}"):
                st.markdown(r["text"])
                if meta.get("published"):
                    st.caption(
                        f"{meta.get('published')} · {' '.join(meta.get('authors', [])[:3])}"
                    )
    elif store and not query.strip():
        st.info("Type a question above. Try: 'surface code threshold' or 'quantum advantage'.")


# ---------------------------------------------------------------------------
# Analytics tab
# ---------------------------------------------------------------------------

def _analytics_tab() -> None:
    st.header("Corpus analytics")
    data = _load_analytics() or (qanalytics.compute_analytics() if IMPORT_OK else None)
    if not data or data.get("empty"):
        st.info("No analytics yet — run `python scripts/build_store.py` first.")
        return
    st.metric("Documents", data["n_docs"])
    st.metric("Chunks", data["n_chunks"])
    st.subheader("Papers per year")
    st.bar_chart({y["year"]: y["count"] for y in data.get("years", [])})
    st.subheader("Top categories")
    st.bar_chart({c["category"]: c["count"] for c in data.get("categories", [])})
    st.subheader("Top authors")
    st.bar_chart({a["author"]: a["count"] for a in data.get("top_authors", [])})
    st.subheader("Hot keywords")
    st.bar_chart({k["keyword"]: k["count"] for k in data.get("top_keywords", [])})


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------

_ABOUT = """## QuantumRAG

RAG over quantum computing research papers. Built to answer questions about
quantum error correction and quantum machine learning with **verifiable
citations** — showing evidence, not inventing answers.

### Architecture

- **Corpus**: arXiv abstracts + metadata (quant-ph, cs.ET, cs.LG)
- **Embeddings**: all-MiniLM-L6-v2 (sentence-transformers) on CPU
- **Vector store**: ChromaDB (PersistentClient, cosine)
- **Retrieval**: hybrid vector + BM25 + metadata filters
- **Grounding**: lexical overlap verdict — no LLM needed for the default mode

### Honest limitations

- Corpus is abstract-level (no full PDFs) for the MVP.
- Grounding is lexical, not semantic: term overlap is checked, meaning is not.
- LLM generation (Ollama, optional) is CPU-bound: ~15-30s per answer.

### How to reproduce locally

```
python scripts/download_corpus.py     # fetch arXiv papers
python scripts/build_store.py         # embed + store in ChromaDB
streamlit run demo_ui/app.py          # run this UI
```
"""


def _about_tab() -> None:
    st.markdown(_ABOUT)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="QuantumRAG", page_icon="⚛️", layout="wide")
    st.title("⚛️ QuantumRAG — Quantum Computing Research Assistant")
    tab_ask, tab_analytics, tab_about = st.tabs(["Ask", "Analytics", "About"])
    with tab_ask:
        _ask_tab()
    with tab_analytics:
        _analytics_tab()
    with tab_about:
        _about_tab()


if __name__ == "__main__":
    main()
