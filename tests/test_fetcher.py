"""Tests for arXiv fetcher -- XML parsing, zero network calls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "arxiv_feed.xml"


def test_parse_feed_count():
    from src.fetcher import parse_feed
    xml = FIXTURE.read_text(encoding="utf-8")
    papers = parse_feed(xml)
    assert len(papers) == 2


def test_parse_feed_fields():
    from src.fetcher import parse_feed
    xml = FIXTURE.read_text(encoding="utf-8")
    papers = parse_feed(xml)
    p1 = papers[0]
    assert p1.arxiv_id == "2510.24181v1"
    assert "Surface Code" in p1.title
    assert "threshold" in p1.abstract.lower()
    assert p1.authors == ["Alice Quantum", "Bob Threshold"]
    assert p1.published == "2025-10-28"
    assert "quant-ph" in p1.categories
    assert p1.pdf_url == "https://arxiv.org/pdf/2510.24181v1"


def test_parse_feed_empty():
    from src.fetcher import parse_feed
    empty = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert parse_feed(empty) == []


def test_parse_feed_one_entry():
    from src.fetcher import parse_feed
    xml = FIXTURE.read_text(encoding="utf-8")
    papers = parse_feed(xml)
    p2 = papers[1]
    assert p2.arxiv_id == "2410.19921v2"
    assert len(p2.authors) == 1
    assert "decoherence" in p2.abstract.lower()
