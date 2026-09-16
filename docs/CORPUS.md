# Corpus — QuantumRAG

## What is in the corpus

**240+ peer-reviewed and preprint papers from arXiv** covering three
intersecting research fronts:

| Theme | Topics | Typical queries |
|-------|--------|-----------------|
| Quantum Error Correction (QEC) | surface codes, stabilizer codes, topological codes, thresholds, fault tolerance | "surface code threshold", "stabilizer syndrome measurement" |
| Quantum Machine Learning (QML) | variational circuits, quantum neural networks, kernels, quantum advantage claims | "variational quantum eigensolver", "quantum machine learning circuit" |
| Quantum Computing fundamentals | QFT, QKD, entanglement, decoherence, NISQ limitations | "quantum Fourier transform period finding", "decoherence NISQ" |

## Sourcing

- **API**: arXiv public API (`http://export.arxiv.org/api/query`), documented
  and free; queries are rate-limited to 3s between requests.
- **Categories**: `quant-ph` (primary), `cs.ET`, `cs.AI`, `stat.ML` (adjacent).
- **Recency**: 2014–2026 window; 58 papers already published in 2026.
- **No PDFs re-hosted**: only metadata + abstracts (public domain under arXiv
  license) are stored locally in `data/raw/corpus.json` (~2.5 MB).

## Processing pipeline

```
download_corpus.py ───► corpus.json (240 papers)
       │
       ▼
chunker.py ───────────► 613 overlapping chunks (~700 chars, 80 overlap)
       │
       ▼
embeddings.py ────────► all-MiniLM-L6-v2 (384-dim, CPU, ~1s/paper)
       │
       ▼
store.py ─────────────► data/chroma/ ChromaDB PersistentClient
```

## Corpus statistics (computed from the store)

- **n_docs**: 240
- **n_chunks**: 613
- **years**: 2005 → 2026, majority 2020+
- **categories**: `quant-ph` dominant, plus `cs.ET`, `cs.AI`, etc.

> Run `python scripts/build_store.py` to rebuild the store from scratch, or
> `python scripts/download_corpus.py` to refresh the raw corpus first.

## Grounding limits (honesty)

The corpus is **abstract-only**. For questions whose answer requires
methodological details living deeper than the abstract (e.g., exact gate
schedules of a decoder), retrieval may return partial evidence. The UI shows
an explicit "insufficient sources" state in those cases rather than guessing.
