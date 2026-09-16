# Technical Architecture — QuantumRAG

## System overview

```
┌──────────────────────────────────────────────────────────────┐
│                     Streamlit UI (demo_ui/app.py)             │
│  ┌──────────────┐ ┌───────────────┐ ┌─────────────────────┐  │
│  │ Ask tab      │ │ Analytics tab │ │ About tab           │  │
│  │ (evidence-   │ │ (charts/      │ │ (model, corpus,     │  │
│  │  first RAG)  │ │  tables)      │ │  sources, evals)    │  │
│  └──────┬───────┘ └───────────────┘ └─────────────────────┘  │
└─────────┼────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────────────────┐
│    retriever.py     │     │        embeddings.py         │
│  hybrid search:     │◄───►│  all-MiniLM-L6-v2 (CPU, 384d)│
│  α*vector + β*BM25  │     │   embed_one / embed_batch    │
└─────────┬───────────┘     └──────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│     store.py        │
│   ChromaDB          │
│  (in-process, no    │
│   HTTP server)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────────────────┐
│    grounding.py     │◄───►│  Ollama (optional, --llm     │
│  evidence check     │     │   localhost:11434)           │
│  (>=1 chunk, ≥50%   │     │  qwen2.5:7b via generate    │
│   token overlap)    │     └──────────────────────────────┘
└─────────────────────┘
```

## Module map

| Module | Responsibility | Network? |
|--------|---------------|----------|
| `src/fetcher.py` | arXiv API query + Atom XML parsing | Yes (build time) |
| `src/chunker.py` | Text → overlapping chunks with metadata | No |
| `src/embeddings.py` | SentenceTransformer load + batch encode | No (model cached) |
| `src/store.py` | ChromaDB create/upsert/query | No (in-process) |
| `src/retriever.py` | Hybrid vector + BM25, doc filter | No |
| `src/grounding.py` | Evidence grounding verdict | No |
| `src/analytics.py` | Corpus stats from store metadata | No |
| `src/evals.py` | Eval pipeline (judges + metrics) | No (opt-in Ollama) |
| `src/config.py` | Central paths/constants | — |

## Key design decisions

### 1. CPU-only embeddings → demo anywhere
`all-MiniLM-L6-v2` (22M params, 384 dims) removes the GPU requirement.
Embedding 240 papers takes ~4-5 min on a laptop CPU; queries are ~10-50 ms.

### 2. Hybrid retrieval: semantic + lexical
- **Vector**: cosine similarity on 384-dim embeddings (semantic match)
- **BM25**: pure-Python Okapi BM25 over the candidate pool (keyword match)
- **Fusion**: `score = 0.7*vs + 0.3*bm25` (`ALPHA_HYBRID`), normalized.
  Completes the pool from the vector recall but re-ranks with lexical signal —
  handles exact terms like "surface code threshold" that pure embeddings miss.

### 3. Evidence-first default (no LLM generation)
- Grounding check requires ≥1 retrieved chunk with ≥50% token overlap.
- If grounded → show sources; if not → explicit "insufficient sources".
- LLM answers are opt-in (`--llm`) and always shown **with** the sources.
- This materially reduces hallucination (a core RAG-in-production lesson).

### 4. Deterministic, CI-safe evals
- `HeuristicJudge`: lexical overlap + keyword coverage, 0-5 → normalized [0,1].
- `OllamaJudge`: LLM-as-judge against local Ollama, opt-in, falls back to
  heuristic on connection failure.
- Metrics: `Recall@k`, `MRR`, `faithfulness`, `relevance`.
- Tests never touch chromadb/torch: pure functions only (BM25, judges,
  XML parsing) → CI runs on bare `pytest` without heavy deps.

## Scaling characteristics

| Knob | Current | Scaling path |
|------|---------|--------------|
| Corpus | 240 papers / 613 chunks | parallel fetch, chunk store in Parquet |
| Embeddings | batch encode, CPU | GPU batch (same code path) |
| Store | ChromaDB in-process | persistent server (opt-in) |
| Query | <100 ms hybrid search | index BM25 over full corpus, ANN fallback |
| Evals | 12 golden cases | extend `data/evals/golden_set.json` |

## Deployment

- Daily dev: `streamlit run demo_ui/app.py` (local, no .env needed)
- Public demo: Streamlit Community Cloud, store committed in `data/chroma/`
- CI: GitHub Actions — test matrix (3.11/3.12), ruff, pip-audit, gitleaks
