"""Central configuration: paths, model names, and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"
ANALYTICS_DIR = DATA_DIR / "analytics"
DOCS_DIR = PROJECT_ROOT / "docs"

# --- Embeddings -----------------------------------------------------------
# all-MiniLM-L6-v2: 384 dims, ~1s/paper on CPU, no GPU needed.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- arXiv API ------------------------------------------------------------
ARXIV_API_URL = "http://export.arxiv.org/api/query"
# Quantum categories: quantum physics + adjacent ML/CS fields
ARXIV_QUERIES = {
    "quantum_error_correction": "cat:quant-ph AND (abs:\"error correction\" OR abs:\"surface code\")",
    "quantum_machine_learning": "cat:quant-ph AND (abs:\"quantum machine learning\" OR abs:\"quantum neural network\")",
    "quantum_algorithms": "cat:cs.ET AND (abs:\"quantum algorithm\" OR abs:\"quantum computing\")",
}
ARXIV_MAX_RESULTS = 80  # per query — adjust for bigger corpus

# --- Chunking -------------------------------------------------------------
CHUNK_SIZE = 700       # characters per chunk
CHUNK_OVERLAP = 80     # overlap between chunks
SECTIONS = [
    "Abstract", "Introduction", "Background", "Methods", "Methodology",
    "Results", "Discussion", "Conclusion", "Related Work", "Experiments",
]

# --- Retrieval ------------------------------------------------------------
TOP_K = 8
# BM25 weight in hybrid score: score = alpha*vs + (1-alpha)*bm25
ALPHA_HYBRID = 0.7

# --- ChromaDB -------------------------------------------------------------
CHROMA_COLLECTION = "quantum_papers"
CHROMA_PERSIST_DIR = str(CHROMA_DIR)

# --- Grounding ------------------------------------------------------------
# Minimum shared tokens for a retrieved chunk to count as "grounded"
GROUNDING_MIN_SHARED = 3
GROUNDING_MIN_SCORE = 0.25
