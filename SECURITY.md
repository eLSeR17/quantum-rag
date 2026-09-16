# Security Policy — QuantumRAG

## Scope

This document covers the security posture of **QuantumRAG**, a CPU-only RAG
pipeline over 240+ open-access arXiv papers on quantum error correction and
quantum machine learning.

The project is **intentionally dependency-light and fully local**: no external
API keys, no network calls at runtime (except the optional local Ollama
instance), and no user data collection.

## Reporting a vulnerability

If you find a security issue, **do not open a public issue**. Contact the
maintainer directly (email in the README) with:

1. A description of the vulnerability
2. Steps to reproduce
3. Impact assessment

We will acknowledge receipt within 48h and provide a fix timeline within 7 days.

## Security model

| Layer | Guarantee |
|-------|-----------|
| **Runtime network** | No outbound calls from `src/`. The optional LLM step targets `localhost:11434` (Ollama) only. |
| **Embedding model** | `all-MiniLM-L6-v2` (sentence-transformers, official repo, 22M params). Runs locally on CPU; the model file is cached under the melon-transformers cache in the venv. |
| **Vector store** | ChromaDB **in-process** `PersistentClient` only. No HTTP server, no auth surface, no multi-tenancy exposed. |
| **Data provenance** | All corpus data comes from the public arXiv API. No PII. Abstract-only (no PDF re-uploads). |
| **Secrets** | Zero API keys/tokens in the repo. `.gitignore` excludes `.env`, caches and builds. CI runs gitleaks to enforce. |

## Known software advisories (tracked)

`pip-audit` runs in CI and is the source of truth for the pinned dependency
set (`requirements.txt`). Current policy for acknowledged advisories:

| Advisory | Package | Status |
|----------|---------|--------|
| PYSEC-2026-3813 / 3814 / 3815 | chromadb 0.6.x | **not_used** — affect the bundled HTTP server path (Python+Pydantic details under Chroma's own server), which this repo never starts. In-process `PersistentClient` only; no `chromadb.config.Settings(anonymized_telemetry...)` server. Tracked; will bump when upstream releases a wheel. |
| Any new advisory | — | The pre-push hook + CI fail the build; fix = bump pin or document `not_used` with evidence here. |

## Verified supply chain

- `sentence-transformers` / `transformers` / `torch`: official Hugging Face / PyTorch
  PyPI packages, pinned in `pyproject.toml`, installed via the project venv.
- `chromadb`: official PyPI package, pinned `>=0.4.22,<1.0`.
- All downloads happen through PyPI only (no raw URLs, no `git+https` installs).

## Data handling

- `data/chroma/` is committed to the repo **intentionally** so the Streamlit
  Community Cloud demo works without a build step, and contains only public
  paper metadata + float embeddings — no sensitive content.
- `data/raw/corpus.json` is public arXiv metadata; regenerate with
  `scripts/download_corpus.py` if needed.

## Local development notes

- Model cache lives in the venv (`.venv/`), which is gitignored.
- The eval pipeline never makes network calls (HeuristicJudge is lexical and
  deterministic). The Ollama judge is opt-in (`--ollama`) and fallible — it
  degrades to the heuristic judge on connection errors.
