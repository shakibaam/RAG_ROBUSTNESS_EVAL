#!/usr/bin/env python3
"""
Minimal Ragnarok RAG example.

Usage (from Code/ragnarok):
  python scripts/run_my_rag.py --requests /path/to/requests.jsonl --run-id demo

Set OPENAI_API_KEY (or OPEN_AI_API_KEY) / ANTHROPIC_API_KEY in the repo `.env`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/run_my_rag.py` from Code/ragnarok
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import read_requests_from_file  # noqa: E402
from generate.api_keys import get_openai_api_key  # noqa: E402
from generate.generator import RAG  # noqa: E402
from generate.gpt import SafeOpenai  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True, help="Input JSONL requests")
    parser.add_argument("--run-id", default="ragnarok_run")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--dataset-name", default="TREC")
    args = parser.parse_args()

    if not args.requests.exists():
        raise SystemExit(f"Missing requests file: {args.requests}")
    if not get_openai_api_key():
        raise SystemExit("Set OPENAI_API_KEY (or OPEN_AI_API_KEY) in .env")

    requests = read_requests_from_file(str(args.requests))
    agent = SafeOpenai(model=args.model, keys=get_openai_api_key())
    rag = RAG(agent=agent, run_id=args.run_id)
    results = rag.answer_batch(requests=requests, logging=True)
    rag.write_answer_results(
        retrieval_method_name=args.run_id,
        results=results,
        dataset_name=args.dataset_name,
    )
    print(f"Wrote results for run_id={args.run_id} n={len(results)}")


if __name__ == "__main__":
    main()
