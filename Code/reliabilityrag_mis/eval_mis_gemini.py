#!/usr/bin/env python3
"""
Gemini stance evaluation for ReliabilityRAG MIS vs Baseline.

Reads released/generated MIS instance JSONs and Extreme_Pool baseline CSV,
writes mis_vs_baseline_comparison.csv under --out-dir (no RobustRAG dependency).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Code" / "evaluation"))
from stance_utils import (  # noqa: E402
    format_stance_prompt,
    load_trec_prompts,
    normalize_stance,
)

GEMINI_MODEL = "gemini-2.5-flash"


def call_gemini(client, prompt: str):
    from google.genai import types

    t0 = time.time()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    text = getattr(resp, "text", None)
    if text is None:
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            text = None
    return normalize_stance(text), {
        "model": GEMINI_MODEL,
        "elapsed_sec": time.time() - t0,
        "raw_text": text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=[2020, 2021], required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument(
        "--mis-instances",
        type=Path,
        required=True,
        help="Directory of MIS instance JSON files",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        required=True,
        help="Extreme_Pool baseline results CSV with llm_response + predicted_stance_gemini",
    )
    parser.add_argument(
        "--prompts-csv",
        type=Path,
        default=None,
        help="Default: Data/prompts_trec/prompts_{year}_v2.csv",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY / GEMINI_API_KEY not set")

    prompts_csv = args.prompts_csv or (
        REPO_ROOT / "Data" / "prompts_trec" / f"prompts_{args.year}_v2.csv"
    )
    prompts = load_trec_prompts(prompts_csv)
    baseline = pd.read_csv(args.baseline_csv)
    base_by_tid = {str(r["topic_id"]): r for _, r in baseline.iterrows()}

    from google import genai

    client = genai.Client(api_key=api_key)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "stance_cache"
    cache_dir.mkdir(exist_ok=True)

    rows = []
    paths = sorted(args.mis_instances.glob("*.json"))
    for path in paths:
        rec = json.loads(path.read_text())
        rq = rec.get("request_qid") or path.stem
        qid = str(rec.get("trec_qid") or "")
        base = base_by_tid.get(rq, {})
        gt = prompts.get(qid, {}).get("gt_stance") or rec.get("baseline_gt_stance")
        desc = prompts.get(qid, {}).get("description", "")
        query = rec.get("query") or desc

        base_resp = str(base.get("llm_response") or "")
        base_pred = normalize_stance(base.get("predicted_stance_gemini"))
        if base_pred is None and base_resp:
            cache_b = cache_dir / f"baseline__{qid}.json"
            if cache_b.exists():
                base_pred = json.loads(cache_b.read_text()).get("normalized")
            else:
                prompt = format_stance_prompt(query, desc, base_resp)
                base_pred, meta = call_gemini(client, prompt)
                cache_b.write_text(json.dumps({"normalized": base_pred, **meta}, indent=2))

        mis_resp = str(rec.get("final_response") or "")
        cache_m = cache_dir / f"mis__{qid}.json"
        if cache_m.exists():
            mis_pred = json.loads(cache_m.read_text()).get("normalized")
        else:
            prompt = format_stance_prompt(query, desc, mis_resp)
            mis_pred, meta = call_gemini(client, prompt)
            cache_m.write_text(json.dumps({"normalized": mis_pred, **meta}, indent=2))

        rows.append(
            {
                "qid": qid,
                "request_qid": rq,
                "gt_stance": gt,
                "baseline_predicted_stance": base_pred,
                "baseline_aligned": bool(base_pred == gt) if base_pred and gt else False,
                "mis_predicted_stance": mis_pred,
                "mis_aligned": bool(mis_pred == gt) if mis_pred and gt else False,
                "mis_selected_indices": rec.get("mis_selected_indices"),
                "mis_size": len(rec.get("mis_selected_indices") or []),
                "mis_selected_doc_types": rec.get("mis_selected_doc_types"),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = args.out_dir / "mis_vs_baseline_comparison.csv"
    df.to_csv(out_csv, index=False)
    summary = {
        "year": args.year,
        "condition": args.condition,
        "n": len(df),
        "baseline_aligned": int(df["baseline_aligned"].sum()),
        "baseline_rate": float(df["baseline_aligned"].mean()) if len(df) else None,
        "mis_aligned": int(df["mis_aligned"].sum()),
        "mis_rate": float(df["mis_aligned"].mean()) if len(df) else None,
        "judge": GEMINI_MODEL,
    }
    (args.out_dir / "gemini_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
