"""Section-aware chunking for academic papers.

Papers have a clear structure (Abstract, Introduction, Methods, Results...).
Chunking by section preserves semantic boundaries; a sliding window handles
long sections without losing context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config


@dataclass
class Chunk:
    text: str
    section: str = ""
    char_start: int = 0
    char_end: int = 0


_SECTION_PAT = re.compile(
    r"^\s*(?:\d+\.?\s*)?"
    r"(Abstract|Introduction|Background|Related\s+Work|Methods?|Methodology|"
    r"Approach|Proposed\s+Method|Experiments?|Evaluation|Results?|Discussion|"
    r"Conclusion(?:s)?|Acknowledgments?|References)\s*[:.]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a doc into (section_name, section_text) chunks at section headers."""
    # Find section header positions
    matches = list(_SECTION_PAT.finditer(text))
    if not matches:
        return [("body", text.strip())]
    sections: list[tuple[str, str]] = []
    for idx, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((name.lower(), body))
    return sections


def _window(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Sliding-window split of a long block."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    if len(text) <= size:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        parts.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return parts


def chunk_document(text: str, doc_id: str = "") -> list[Chunk]:
    """Full pipeline: split sections → window long sections → Chunk objects."""
    chunks: list[Chunk] = []
    for section, body in split_sections(text):
        for piece in _window(body):
            chunks.append(Chunk(text=piece, section=section))
    # Assign char offsets for traceability
    cursor = 0
    for c in chunks:
        c.char_start = cursor
        c.char_end = cursor + len(c.text)
        cursor = c.char_end
    return chunks


def chunk_corpus(entries: list[dict]) -> list[dict]:
    """Chunk many corpus entries → list of chunk dicts for the store."""
    out = []
    for entry in entries:
        doc_id = entry["id"]
        chunks = chunk_document(entry["text"], doc_id)
        for i, c in enumerate(chunks):
            out.append(
                {
                    "id": f"{doc_id}#{i}",
                    "doc_id": doc_id,
                    "text": c.text,
                    "section": c.section,
                    "metadata": {
                        **entry.get("metadata", {}),
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "section": c.section,
                    },
                }
            )
    return out


if __name__ == "__main__":
    sample = (
        "Abstract\nWe study surface codes for quantum error correction.\n\n"
        "Introduction\nQuantum computers are noisy. Error correction is essential.\n\n"
        "Results\nWe achieve 99.9% fidelity.\n"
    )
    for c in chunk_document(sample):
        print(f"[{c.section}] {c.text[:60]}...")
