"""Build the ChromaDB vector store from the downloaded corpus.

Usage:
    python scripts/build_store.py [--corpus data/raw/corpus.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import chunker, config, embeddings, store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB store")
    parser.add_argument("--corpus", type=str, default=str(config.RAW_DIR / "corpus.json"))
    args = parser.parse_args()

    raw = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    print(f"[build] {len(raw)} papers loaded from {args.corpus}")

    # Build corpus entries (abstract text = body for MVP)
    entries = []
    for p in raw:
        text = (
            f"Title: {p['title']}\n"
            f"Published: {p['published']}\n"
            f"Authors: {', '.join(p['authors'])}\n"
            f"Categories: {', '.join(p['categories'])}\n"
            f"Abstract: {p['abstract']}"
        )
        entries.append({"id": f"arxiv-{p['arxiv_id']}", "text": text, "metadata": p})

    chunks = chunker.chunk_corpus(entries)
    print(f"[build] {len(chunks)} chunks from {len(entries)} docs")

    texts = [c["text"] for c in chunks]
    print(f"[build] embedding {len(texts)} chunks (CPU, ~1-2 min per 100)...")
    vectors = embeddings.embed(texts)

    added = store.add_chunks(chunks, vectors)
    print(f"[build] added {added} new chunks; store now has {store.count()}")


if __name__ == "__main__":
    main()
