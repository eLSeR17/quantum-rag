# QuantumRAG — Quantum Computing Research Assistant

[![CI](https://github.com/eLSeR17/quantum-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/eLSeR17/quantum-rag/actions/workflows/ci.yml)

Retrieval-Augmented Generation over quantum error correction and quantum machine
learning papers. Answers questions about the latest quantum computing research
with **verifiable citations** — no hallucinations, no guessing.

## The problem

Quantum computing research moves too fast for manual reading: keeping up with
QEC and quantum ML means working through hundreds of arXiv papers. Generic
LLMs are worse than useless here — they confidently cite papers that do not
exist and confabulate results. In a field where precision is the whole point,
an ungrounded answer is a bug.

## The solution

QuantumRAG answers questions **from the corpus itself**: 240 arXiv papers,
indexed with hybrid retrieval (vector + BM25 + metadata filters) and grounded
with a deterministic lexical verdict. Retrieved chunks and the grounding
verdict are part of the answer — no retrieval, no answer. The system refuses
to guess.

- **Evidence-first by default**: the demo shows the retrieved chunks and the
  grounding verdict; LLM generation is an option, never the default
- **Verifiable quality**: 12 golden cases — recall@8 **1.000** means every
  golden question retrieves its ground-truth paper in the top-8
- **Runs on CPU**: no GPU, no paid API, deployed free on Streamlit Community
  Cloud (public, no login)

## Live demo

Try the deployed app: **[quantum-rag.streamlit.app](https://quantum-rag.streamlit.app)**
— hybrid retrieval over the 240-paper corpus with grounded, cited answers and a
trends analytics tab. Runs entirely on CPU, deployed on Streamlit Community Cloud
(public, no login).

## What it does

- **Corpus**: 240 papers from arXiv (quant-ph, cs.ET) — open-access abstracts only
- **Retrieval**: hybrid vector search + BM25 + metadata filtering (author, date, arXiv category)
- **Grounding**: every answer is cross-checked against the retrieved evidence
- **Analytics**: trends dashboard (papers per year, keyword growth, top labs, technique hotness)
- **Evals**: golden dataset (12 quantum topics) + heuristic/LLM judges, recall@k & MRR
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

# Run the eval suite (12 golden cases)
python scripts/run_eval.py
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
│   ├── raw/            # arXiv metadata (abstracts)
│   ├── chroma/         # vector store (committed for Cloud deploy)
│   └── evals/          # golden dataset (12 cases)
├── docs/
│   ├── LIVE_DEMO.md    # archived demo run
│   └── CORPUS.md       # corpus description + sources
├── tests/
└── pyproject.toml
```

## Evaluation

Automatic evals run over 12 golden questions covering the main corpus
fronts (surface codes, stabilizer codes, QML, VQE, QKD, QFT, entanglement,
decoherence, topological codes). Two judges:

- **HeuristicJudge** (default, CI-safe): lexical overlap + keyword coverage
- **OllamaJudge** (opt-in): LLM-as-judge against local Ollama, falls back to heuristic

| Metric | Current |
|--------|---------|
| Recall@8 | 1.000 |
| MRR | 0.674 |
| Faithfulness | 0.623 |
| Relevance | 0.805 |

Full per-case breakdown in [`docs/LIVE_DEMO.md`](docs/LIVE_DEMO.md). A 12/12
recall@8 means every golden question retrieves its ground-truth paper in the
top-8 results; MRR 0.674 reflects strong first-hit ranking.

## Security

See [`SECURITY.md`](SECURITY.md): zero API keys in the repo, no runtime network calls
(except optional local Ollama), ChromaDB in-process only, verified supply chain,
`pip-audit` + gitleaks in CI.

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
