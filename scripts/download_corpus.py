"""Download the quantum corpus from arXiv and persist raw JSON.

Usage:
    python scripts/download_corpus.py [--max 80] [--out data/raw/corpus.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, fetcher


def main() -> None:
    parser = argparse.ArgumentParser(description="Download quantum arXiv corpus")
    parser.add_argument("--max", type=int, default=config.ARXIV_MAX_RESULTS,
                        help="results per arXiv query (default: %(default)s)")
    parser.add_argument("--out", type=str, default=str(config.RAW_DIR / "corpus.json"))
    args = parser.parse_args()

    print(f"[download] fetching up to {args.max} results per query from arXiv API...")
    papers = fetcher.fetch_papers(max_results=args.max)
    if not papers:
        print("[download] ERROR: no papers fetched — check network / arXiv API")
        sys.exit(1)

    out = Path(args.out)
    fetcher.export_papers_json(papers, out)
    print("[download] done. Source of truth for the corpus.")


if __name__ == "__main__":
    main()
