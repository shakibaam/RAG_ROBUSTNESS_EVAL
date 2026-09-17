#!/usr/bin/env python3
"""
Run ReliabilityRAG MIS on Extreme_Pool inputs (paper mitigation pipeline).

Expects Extreme_Pool layout under --data-root:
  neutral_requests/<bias>/<prefix>_requests.jsonl
  neutral_batch_requests/<bias>/<prefix>_requests_batch.jsonl
  neutral_results/<bias>/csv_files/<prefix>_ragnarok_results_format.csv

API keys via environment / .env (OPENAI_API_KEY or OPENROUTER_API_KEY).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

import openai
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mis_core import (
    NLI_MODEL_ID,
    build_conflict_graph,
    build_final_messages,
    doc_type,
    load_isolate_template,
    select_mis,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

YEAR_DEFAULTS = {
    2020: {"expected_n": 16, "default_model": "openai/gpt-5"},
    2021: {"expected_n": 23, "default_model": "gpt-5-2025-08-07"},
}

CONDITIONS = {
    "extreme_harmful_biased_liar": {
        "bias": "harmful_biased",
        "file_stem": "liar_attack",
        "label": "Harmful-biased Liar pool 8:2",
        "expect_helpful": 2,
        "expect_adversarial": 8,
    },
    "extreme_harmful_biased_fsap_interq": {
        "bias": "harmful_biased",
        "file_stem": "fsap_interq_03_examples",
        "label": "Harmful-biased FSAP-InterQ pool 8:2",
        "expect_helpful": 2,
        "expect_adversarial": 8,
    },
    "extreme_harmful_biased_fsap_intraq": {
        "bias": "harmful_biased",
        "file_stem": "fsap_intraq",
        "label": "Harmful-biased FSAP-IntraQ pool 8:2",
        "expect_helpful": 2,
        "expect_adversarial": 8,
    },
    "extreme_helpful_biased_liar": {
        "bias": "helpful_biased",
        "file_stem": "liar_attack",
        "label": "Helpful-biased Liar pool 8:2",
        "expect_helpful": 8,
        "expect_adversarial": 2,
    },
    "extreme_helpful_biased_fsap_interq": {
        "bias": "helpful_biased",
        "file_stem": "fsap_interq_03_examples",
        "label": "Helpful-biased FSAP-InterQ pool 8:2",
        "expect_helpful": 8,
        "expect_adversarial": 2,
    },
    "extreme_helpful_biased_fsap_intraq": {
        "bias": "helpful_biased",
        "file_stem": "fsap_intraq",
        "label": "Helpful-biased FSAP-IntraQ pool 8:2",
        "expect_helpful": 8,
        "expect_adversarial": 2,
    },
}

MAX_COMPLETION_TOKENS = 2000
REASONING_EFFORT = "minimal"
OPENAI_TIMEOUT_SEC = 180.0


def parse_refs(val):
    if isinstance(val, list):
        return [str(x) for x in val]
    if val is None:
        return []
    s = str(val).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [s]


def extract_batch_docs(user_content: str) -> list[str]:
    m = re.search(r"Documents:\s*(.*)\n\nQuery:", user_content, flags=re.S)
    if not m:
        raise SystemExit("Could not parse Documents block from baseline batch prompt")
    blob = m.group(1).strip()
    parts = re.split(r"\n(?=\[\d+\] )", blob)
    docs = []
    for i, part in enumerate(parts, 1):
        mm = re.match(r"\[(\d+)\] (.*)$", part, flags=re.S)
        if not mm:
            raise SystemExit(f"Bad doc block {i}: {part[:80]!r}")
        docs.append(mm.group(2))
    return docs


def make_client():
    load_dotenv(REPO_ROOT / ".env")
    openrouter = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if openrouter:
        return openai.OpenAI(
            api_key=openrouter,
            base_url=base_url or "https://openrouter.ai/api/v1",
            timeout=OPENAI_TIMEOUT_SEC,
        ), True
    if not openai_key:
        raise SystemExit("Set OPENAI_API_KEY or OPENROUTER_API_KEY")
    kwargs = {"api_key": openai_key, "timeout": OPENAI_TIMEOUT_SEC}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs), False


def call_llm(client, model: str, messages: list[dict]) -> dict:
    t0 = time.time()
    err = None
    resp = None
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        }
        # reasoning_effort only for OpenAI GPT-5 family when supported
        if "gpt-5" in model or model.startswith("openai/gpt-5"):
            kwargs["reasoning_effort"] = REASONING_EFFORT
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    elapsed = time.time() - t0
    if err is not None:
        return {
            "success": False,
            "error": err,
            "output_text": None,
            "elapsed_sec": elapsed,
            "model": model,
        }
    text = resp.choices[0].message.content or ""
    return {
        "success": True,
        "error": None,
        "output_text": text,
        "elapsed_sec": elapsed,
        "model": getattr(resp, "model", model),
        "finish_reason": resp.choices[0].finish_reason,
        "response_id": getattr(resp, "id", None),
    }


def instance_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rec = json.loads(path.read_text())
    except Exception:
        return False
    resp = rec.get("final_response")
    return bool(resp is not None and str(resp).strip()) and "mis_selected_indices" in rec


def condition_paths(data_root: Path, out_root: Path, cond_key: str):
    meta = CONDITIONS[cond_key]
    bias = meta["bias"]
    stem = meta["file_stem"]
    prefix = f"neutral_{bias}_{stem}"
    run_dir = out_root / f"full_run_{cond_key}"
    return {
        "meta": meta,
        "out_root": run_dir,
        "inst_dir": run_dir / "instances",
        "request_jsonl": data_root
        / f"neutral_requests/{bias}/{prefix}_requests.jsonl",
        "batch_jsonl": data_root
        / f"neutral_batch_requests/{bias}/{prefix}_requests_batch.jsonl",
        "baseline_csv": data_root
        / f"neutral_results/{bias}/csv_files/{prefix}_ragnarok_results_format.csv",
    }


def load_pools(paths: dict, expected_n: int):
    import pandas as pd

    meta = paths["meta"]
    requests = {}
    with paths["request_jsonl"].open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            requests[obj["query"]["qid"]] = obj

    batches = {}
    with paths["batch_jsonl"].open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            batches[obj["custom_id"]] = obj

    baseline = pd.read_csv(paths["baseline_csv"])
    base_by_tid = {str(r["topic_id"]): r for _, r in baseline.iterrows()}

    all_qids = list(requests.keys())
    if len(all_qids) != expected_n:
        raise SystemExit(f"Expected {expected_n} requests, got {len(all_qids)}")

    pools = []
    for rq in sorted(all_qids, key=lambda x: int(re.search(r"_(\d+)$", x).group(1))):
        req = requests[rq]
        doc_ids = [c["docid"] for c in req["candidates"]]
        batch = batches[rq]
        user = batch["body"]["messages"][1]["content"]
        batch_docs = extract_batch_docs(user)
        query = str(base_by_tid[rq]["description"]).strip()
        m_q = re.search(r"\n\nQuery: (.*?)\n\nInstruction:", user, flags=re.S)
        if m_q:
            query = m_q.group(1).strip()
        pools.append(
            {
                "request_qid": rq,
                "trec_qid": str(base_by_tid[rq]["qid"]),
                "query": query,
                "doc_ids": doc_ids,
                "baseline_prompt_docs": batch_docs,
                "baseline_llm_response": base_by_tid[rq].get("llm_response"),
                "baseline_predicted_stance_gemini": base_by_tid[rq].get(
                    "predicted_stance_gemini"
                ),
                "baseline_gt_stance": base_by_tid[rq].get("gt_stance"),
                "expect_helpful": meta["expect_helpful"],
                "expect_adversarial": meta["expect_adversarial"],
            }
        )
    return pools


def run_mis_for_query(client, model, nli_tok, nli_model, device, pool, cond_key, year):
    isolate_tmpl = load_isolate_template()
    query = pool["query"]
    docs = pool["baseline_prompt_docs"]
    doc_ids = pool["doc_ids"]
    k = len(docs)

    isolates = []
    for i, (did, text) in enumerate(zip(doc_ids, docs)):
        prompt = isolate_tmpl.format(query_str=query, context_str=text)
        api = call_llm(client, model, [{"role": "user", "content": prompt}])
        if not api.get("success"):
            raise RuntimeError(f"Isolate API failed: {api.get('error')}")
        ans = api.get("output_text") or ""
        isolates.append(
            {
                "doc_index": i,
                "doc_id": did,
                "doc_type": doc_type(did),
                "isolated_response": ans,
                "api": {kk: vv for kk, vv in api.items() if kk != "output_text"},
                "contains_idk": ("I don't know" in ans),
            }
        )
        time.sleep(0.1)

    answers = [x["isolated_response"] for x in isolates]
    graph, nli_pairs = build_conflict_graph(
        query, answers, nli_tok, nli_model, device
    )
    best_set, z = select_mis(graph, answers)
    selected_docs = [docs[i] for i in best_set]
    selected_ids = [doc_ids[i] for i in best_set]
    selected_types = [doc_type(d) for d in selected_ids]

    final_messages = build_final_messages(query, selected_docs)
    final_api = call_llm(client, model, final_messages)
    if not final_api.get("success"):
        raise RuntimeError(f"Final API failed: {final_api.get('error')}")
    final_text = final_api.get("output_text") or ""
    edges = sorted([(i, j) for i in range(k) for j in graph[i] if i < j])

    return {
        "defense": "ReliabilityRAG MIS",
        "year": year,
        "condition": cond_key,
        "request_qid": pool["request_qid"],
        "trec_qid": pool["trec_qid"],
        "query": query,
        "document_order": doc_ids,
        "document_types": [doc_type(d) for d in doc_ids],
        "nli": {"model": NLI_MODEL_ID, "pairs": nli_pairs},
        "isolates": isolates,
        "mis_vertex_set_z": z,
        "graph_edges": edges,
        "mis_selected_indices": best_set,
        "mis_selected_doc_ids": selected_ids,
        "mis_selected_doc_types": selected_types,
        "final_prompt_messages": final_messages,
        "final_response": final_text,
        "final_api": {kk: vv for kk, vv in final_api.items() if kk != "output_text"},
        "baseline_gt_stance": pool["baseline_gt_stance"],
    }


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=[2020, 2021], required=True)
    parser.add_argument(
        "--condition",
        required=True,
        choices=list(CONDITIONS.keys()) + ["all_harmful", "all"],
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Extreme_Pool root containing neutral_requests/batch/results",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory (default: Experiment_Results/TREC{year}/ReliabilityRAG_MIS/runs)",
    )
    parser.add_argument("--model", default=None, help="Override LLM model id")
    args = parser.parse_args()

    year_cfg = YEAR_DEFAULTS[args.year]
    expected_n = year_cfg["expected_n"]
    model = args.model or year_cfg["default_model"]
    out_root = args.out_root or (
        REPO_ROOT
        / "Experiment_Results"
        / f"TREC{args.year}"
        / "ReliabilityRAG_MIS"
        / "runs"
    )

    if args.condition == "all":
        conds = list(CONDITIONS.keys())
    elif args.condition == "all_harmful":
        conds = [k for k in CONDITIONS if "harmful" in k]
    else:
        conds = [args.condition]

    client, _ = make_client()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading NLI {NLI_MODEL_ID} on {device}", flush=True)
    nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID).to(
        device
    )
    nli_model.eval()

    for cond_key in conds:
        paths = condition_paths(args.data_root, out_root, cond_key)
        for req in ("request_jsonl", "batch_jsonl", "baseline_csv"):
            if not paths[req].exists():
                raise SystemExit(f"Missing {req}: {paths[req]}")
        paths["inst_dir"].mkdir(parents=True, exist_ok=True)
        pools = load_pools(paths, expected_n)
        n_new = n_skip = 0
        for i, p in enumerate(pools, 1):
            dest = paths["inst_dir"] / f"{p['request_qid']}.json"
            if instance_complete(dest):
                n_skip += 1
                print(f"[{i}/{len(pools)}] skip {p['request_qid']}", flush=True)
                continue
            print(f"[{i}/{len(pools)}] MIS {p['request_qid']}", flush=True)
            rec = run_mis_for_query(
                client, model, nli_tok, nli_model, device, p, cond_key, args.year
            )
            dest.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
            n_new += 1
        complete = sum(
            1 for p in paths["inst_dir"].glob("*.json") if instance_complete(p)
        )
        summary = {
            "condition": cond_key,
            "year": args.year,
            "expected_n": expected_n,
            "complete": complete,
            "newly_generated": n_new,
            "skipped": n_skip,
        }
        (paths["out_root"] / "generation_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        print(summary, flush=True)
        if complete != expected_n:
            raise SystemExit(f"{cond_key} incomplete: {complete}/{expected_n}")


if __name__ == "__main__":
    main()
