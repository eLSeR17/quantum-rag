"""arXiv paper fetcher: query the arXiv API and download abstracts/PDFs.

Uses the arXiv API (export.arxiv.org/api/query) which is free and has no
auth required. Respects arXiv rate limits (3s between requests).
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from . import config

ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class Paper:
    """Metadata for a single arXiv paper."""

    arxiv_id: str
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    published: str = ""          # ISO date
    categories: list[str] = field(default_factory=list)
    pdf_url: str = ""


def _fetch(query: str, max_results: int, start: int = 0) -> str:
    """Query the arXiv API and return raw Atom XML."""
    params = urllib.parse.urlencode(
        {"search_query": query, "start": start, "max_results": max_results}
    )
    url = f"{config.ARXIV_API_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "QuantumRAG/0.1 (research portfolio)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def parse_feed(xml_text: str) -> list[Paper]:
    """Parse arXiv Atom XML into Paper objects."""
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall("a:entry", ARXIV_NS):
        aid = entry.find("a:id", ARXIV_NS).text.strip().split("/abs/")[-1]
        title = " ".join(entry.find("a:title", ARXIV_NS).text.split())
        abstract = " ".join(entry.find("a:summary", ARXIV_NS).text.split())
        authors = [a.find("a:name", ARXIV_NS).text for a in entry.findall("a:author", ARXIV_NS)]
        published = entry.find("a:published", ARXIV_NS).text[:10]
        categories = [c.get("term") for c in entry.findall("a:category", ARXIV_NS)]
        pdf_url = f"https://arxiv.org/pdf/{aid}"
        papers.append(
            Paper(aid, title, abstract, authors, published, categories, pdf_url)
        )
    return papers


def fetch_papers(max_results: int | None = None) -> list[Paper]:
    """Fetch papers for all configured arXiv queries."""
    max_results = max_results or config.ARXIV_MAX_RESULTS
    seen: dict[str, Paper] = {}
    for label, query in config.ARXIV_QUERIES.items():
        try:
            xml_text = _fetch(query, max_results)
            papers = parse_feed(xml_text)
            print(f"[fetcher] {label}: {len(papers)} papers")
            for p in papers:
                if p.arxiv_id not in seen:
                    seen[p.arxiv_id] = p
            time.sleep(3)  # arXiv rate limit
        except Exception as exc:  # noqa: BLE001
            print(f"[fetcher] ERROR fetching {label}: {exc}")
    print(f"[fetcher] total unique papers: {len(seen)}")
    return list(seen.values())


def export_papers_json(papers: list[Paper], out_path: Path) -> None:
    """Persist paper metadata as JSON (lightweight, no heavy deps)."""
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "arxiv_id": p.arxiv_id,
            "title": p.title,
            "abstract": p.abstract,
            "authors": p.authors,
            "published": p.published,
            "categories": p.categories,
            "pdf_url": p.pdf_url,
        }
        for p in papers
    ]
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[fetcher] wrote {len(data)} papers → {out_path}")


def corpus_text(papers: list[Paper]) -> list[dict]:
    """Build raw corpus entries: id + full text (abstract + metadata).

    PDF extraction from arXiv can be done later (requires pdftotext or similar);
    for the MVP the abstract + metadata already give a working RAG corpus.
    """
    entries = []
    for p in papers:
        text = (
            f"Title: {p.title}\n"
            f"Published: {p.published}\n"
            f"Authors: {', '.join(p.authors)}\n"
            f"Categories: {', '.join(p.categories)}\n"
            f"Abstract: {p.abstract}"
        )
        entries.append({"id": f"arxiv-{p.arxiv_id}", "text": text, "metadata": p.__dict__})
    return entries
