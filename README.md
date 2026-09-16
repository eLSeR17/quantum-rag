# QuantumRAG — Quantum Computing Research Assistant

Retrieval-Augmented Generation over quantum error correction and quantum machine
learning papers. Answers questions about the latest quantum computing research
with **verifiable citations** — no hallucinations, no guessing.

## What it does

- **Corpus**: 200+ papers from arXiv (quant-ph, cs.LG, cs.ET) + Qiskit/PennyLane docs
- **Retrieval**: hybrid vector search + BM25 + metadata filtering (author, date, arXiv category)
- **Grounding**: every answer is cross-checked against the retrieved evidence
- **Analytics**: trends dashboard (papers per year, keyword growth, top labs, technique hotness)
- **Mode**: evidence-first by default (shows chunks + grounding verdict); LLM generation optional

## Quick start

```bash
# Install
pip install -e .

# Download corpus (~200 papers from arXiv)
python scripts/download_corpus.py

# Build vector store
python scripts/build_store.py

# Run the demo
streamlit run demo_ui/app.py
```

## Architecture

```
arXiv API  ──→  fetcher  ──→  PDFs/text
                                  │
                              chunker
                                  │
                           sentence-transformers
                                  │
                            ChromaDB store
                                  │
                        retriever (hybrid)
                                  │
                      grounding (lexical + metadata)
                                  │
                         Streamlit demo UI
```

## GPU needed?

**No.** Embeddings run on CPU (all-MiniLM-L6-v2, ~1s per paper).
Vector search is CPU-only (ChromaDB). LLM generation is optional and runs
on Ollama CPU (~15-30s/query). The default mode is **evidence-first**:
retrieved chunks + grounding verdict, no generation needed.

## Project structure

```
quantum-rag/
├── src/
│   ├── fetcher.py      # arXiv API + PDF download
│   ├── chunker.py      # section-aware chunking
│   ├── embeddings.py   # sentence-transformers wrapper
│   ├── store.py        # ChromaDB persistent store
│   ├── retriever.py    # hybrid search (vector + BM25 + metadata)
│   ├── grounding.py    # lexical grounding verdict
│   ├── analytics.py    # corpus statistics + trends
│   └── config.py       # paths + constants
├── demo_ui/
│   └── app.py          # Streamlit multi-tab demo
├── scripts/
│   ├── download_corpus.py   # fetch papers from arXiv
│   ├── build_store.py       # embed + store in ChromaDB
│   └── run_eval.py          # golden dataset evaluation
├── data/
│   ├── raw/            # downloaded PDFs/text
│   ├── chroma/         # vector store (git-ignored)
│   └── analytics/      # generated statistics
├── docs/
│   ├── LIVE_DEMO.md    # archived demo run
│   └── CORPUS.md       # corpus description + sources
├── tests/
└── pyproject.toml
```

## Scope

This project demonstrates:
- **Data engineering**: API ingestion, PDF parsing, intelligent chunking
- **Embeddings**: sentence-transformers on CPU (no GPU needed)
- **Vector search**: ChromaDB with hybrid retrieval
- **Grounding**: anti-hallucination verification
- **Analytics**: trend analysis over scientific literature
- **Deploy**: Streamlit Community Cloud (free)

## License

MIT
