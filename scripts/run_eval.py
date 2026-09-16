#!/usr/bin/env python3
"""Run QuantumRAG evals from the CLI.

Usage:
    python scripts/run_eval.py                     # heuristic judge (CI-safe)
    python scripts/run_eval.py --ollama            # LLM judge via Ollama
    python scripts/run_eval.py --json              # JSON output for CI
    python scripts/run_eval.py --golden data/evals/golden_set.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evals import run_eval as _run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantumRAG eval runner")
    parser.add_argument("--golden", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ollama", action="store_true", help="Use Ollama LLM judge")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    report = _run_eval(args.golden, top_k=args.top_k, use_ollama=args.ollama)

    if args.json:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return

    # Human-readable report
    print("\n=== QuantumRAG Eval Report ===")
    print(f"Cases: {report['n_cases']}  |  Judge: {report['judge']}  |  top_k: {report['top_k']}")
    print(f"Recall@k:    {report['recall_at_k']:.3f}")
    print(f"MRR:         {report['mrr']:.3f}")
    print(f"Faithfulness:{report['faithfulness']:.3f}")
    print(f"Relevance:   {report['relevance']:.3f}")
    print()

    for r in report["results"]:
        case = r["case"]
        hit = "✓" if r["recall_at_k"] > 0 else "✗"
        print(f"  {hit} {case['id']:8s}  recall={r['recall_at_k']:.0f}  mrr={r['mrr']:.3f}  f={r['faithfulness']:.3f}  r={r['relevance']:.3f}  {case['topic']}")

    # Write report to docs/LIVE_DEMO.md
    out_dir = ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / "LIVE_DEMO.md"
    md = (
        f"# Live Demo Report — QuantumRAG Evals\n\n"
        f"> Generated automatically by ``scripts/run_eval.py``.\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Cases | {report['n_cases']} |\n"
        f"| Judge | {report['judge']} |\n"
        f"| top-k | {report['top_k']} |\n"
        f"| Recall@k | {report['recall_at_k']:.3f} |\n"
        f"| MRR | {report['mrr']:.3f} |\n"
        f"| Faithfulness | {report['faithfulness']:.3f} |\n"
        f"| Relevance | {report['relevance']:.3f} |\n\n"
        f"## Per-case results\n\n"
        f"| ID | Topic | Recall | MRR | Faith | Rel | Hit |\n"
        f"|----|-------|--------|-----|-------|-----|-----|\n"
    )
    for r in report["results"]:
        c = r["case"]
        hit = "✓" if r["recall_at_k"] > 0 else "✗"
        md += f"| {c['id']} | {c['topic']} | {r['recall_at_k']:.0f} | {r['mrr']:.3f} | {r['faithfulness']:.3f} | {r['relevance']:.3f} | {hit} |\n"

    md_path.write_text(md, encoding="utf-8")
    print(f"\nReport saved to {md_path}")


if __name__ == "__main__":
    main()
