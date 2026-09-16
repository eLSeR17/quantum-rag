"""Tests for analytics module -- deterministic, loads pre-built JSON."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ANALYTICS_PATH = Path(__file__).resolve().parent.parent / "data" / "analytics" / "analytics.json"


def test_analytics_json_exists():
    assert ANALYTICS_PATH.exists(), "data/analytics/analytics.json not found; run the analytics builder first"


def test_analytics_structure():
    data = json.loads(ANALYTICS_PATH.read_text(encoding="utf-8"))
    assert "n_docs" in data
    assert "n_chunks" in data
    assert "years" in data
    assert isinstance(data["n_docs"], int)
    assert isinstance(data["n_chunks"], int)
    assert data["n_docs"] > 0
    assert data["n_chunks"] > 0


def test_analytics_years():
    data = json.loads(ANALYTICS_PATH.read_text(encoding="utf-8"))
    years = data["years"]
    assert isinstance(years, list)
    assert all(isinstance(entry, dict) and "year" in entry and "count" in entry for entry in years)
    assert sum(e["count"] for e in years) >= data["n_docs"]


def test_analytics_categories():
    data = json.loads(ANALYTICS_PATH.read_text(encoding="utf-8"))
    cats = data["categories"]
    assert isinstance(cats, list)
    for entry in cats:
        if isinstance(entry, dict):
            assert "category" in entry and "count" in entry
            assert entry["count"] >= 0
