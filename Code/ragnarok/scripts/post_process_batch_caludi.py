

import os
import re
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# --- NLTK sentence tokenizer setup -----------------------------------------
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

punkt_param = PunktParameters()
punkt_tokenizer = PunktSentenceTokenizer(punkt_param)

def sent_tokenize(text: str) -> List[str]:
    return punkt_tokenizer.tokenize(text or "")

# --- Ragnarok datatypes -----------------------------------------------------
from data import Query, CitedSentence, Result, DataWriter




def extract_cited_sentences(text: str) -> List[CitedSentence]:
    """
    Split text into sentences, extract inline numeric citations [1], [2], ...
    Return list[CitedSentence(text, citations=[0-based indices])]
    """
    sentences = sent_tokenize(text)
    out: List[CitedSentence] = []
    for sent in sentences:
        matches = re.findall(r"\[(\d+)\]", sent)
        citation_indices = sorted({int(m) - 1 for m in matches if m.isdigit()})
        clean_sent = re.sub(r"\s*\[\d+\]", "", sent).strip()
        out.append(CitedSentence(text=clean_sent, citations=citation_indices))
    return out


def build_results_from_batch(
    input_path: str,
    output_path: str,
    request_path: str,
    run_id: str = "openai_batch",
) -> List[Result]:
    """
    Builds Ragnarok-style results from OpenAI Batch outputs.
    Expects:
      - input_path  : OpenAI batch requests .jsonl (lines contain {"custom_id", ...})
      - output_path : OpenAI batch results  .jsonl (lines have 'response.body.choices[0].message.content')
      - request_path: Ragnarok requests     .jsonl (lines have query + candidates(docid))
    Assumes custom_id == qid.
    """

    # 1) qid -> {text, references}
    request_info: Dict[str, Dict[str, Any]] = {}
    with open(request_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item["query"]["qid"]
            request_info[qid] = {
                "text": item["query"]["text"],
                "references": [c["docid"] for c in item.get("candidates", [])],
            }

    # Optional: read input file for debugging structure (not used further)
    _ = {}
    if os.path.exists(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            _ = {json.loads(line)["custom_id"]: json.loads(line)
                 for line in f if line.strip()}

    # 2) parse outputs
    results: List[Result] = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id")
            if not custom_id:
                continue
            # OpenAI batch success shape
            try:
                response_content = row["response"]["body"]["choices"][0]["message"]["content"]
            except Exception:
                continue

            qid = custom_id
            info = request_info.get(qid)
            if not info:
                print(f"⚠️ Skipping unmatched qid: {qid}")
                continue

            query = Query(text=info["text"], qid=qid)
            cited_sentences = extract_cited_sentences(response_content)
            results.append(Result(query=query, references=info["references"], answer=cited_sentences))

    return results


def postprocess_nonrag_responses(
    csv_path: str,
    jsonl_path: str,
    output_csv_path: str,
) -> None:
    """
    OpenAI non-RAG results → flat CSV with:
      qid, query, description, tone, llm_response, gt_stance

    Assumes custom_id pattern: {tone}_qid_{N}
    Expects OpenAI shape row['response']['body']['choices'][0]['message']['content']
    """
    # CSV keyed by qid
    with open(csv_path, newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    by_qid = {row["qid"]: row for row in csv_rows}

    out_rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("custom_id")
            if not cid or "response" not in row:
                continue

            m = re.match(r"(\w+)_qid_(\d+)", cid)
            if not m:
                continue
            tone = m.group(1)
            qid = f"qid_{m.group(2)}"
            prompt_col = f"{tone}_prompt"

            try:
                llm_response = row["response"]["body"]["choices"][0]["message"]["content"]
            except Exception:
                continue

            csv_row = by_qid.get(qid)
            if not csv_row:
                continue

            out_rows.append({
                "qid": qid,
                "query": csv_row["title"],
                "description": csv_row.get(prompt_col, ""),
                "tone": tone,
                "llm_response": llm_response,
                "gt_stance": csv_row["answer"],
            })

    if out_rows:
        os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
        pd.DataFrame(out_rows, columns=["qid", "query", "description", "tone", "llm_response", "gt_stance"])\
          .to_csv(output_csv_path, index=False, encoding="utf-8")


def find_lines_missing_choices(jsonl_path: str) -> List[tuple]:
    """
    Return [(line_no, line_str)] where 'choices' missing under 'response.body' (OpenAI format).
    """
    missing = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                missing.append((idx, s))
                continue
            if (
                "response" in row and
                "body" in row["response"] and
                (
                    "choices" not in row["response"]["body"] or
                    row["response"]["body"]["choices"] is None
                )
            ):
                missing.append((idx, s))
    return missing




def _claude_concat_text(message_obj: Optional[Dict[str, Any]]) -> str:
    """
    Concatenate all "text" blocks from Anthropic message.content.
    message_obj shape: {"content": [{"type":"text","text":"..."}, ...], ...}
    """
    if not message_obj:
        return ""
    parts: List[str] = []
    for blk in message_obj.get("content", []):
        if isinstance(blk, dict) and blk.get("type") == "text":
            parts.append(blk.get("text", ""))
        elif isinstance(blk, str):
            parts.append(blk)
    return "\n".join(parts).strip()




def build_results_from_claude_batch(
    results_jsonl_path: str,
    requests_jsonl_path: str,
    run_id: str = "anthropic_batch",
    id_mapping_path: str | None = None,   # <-- NEW: optional short->original mapping
) -> List[Result]:
    """
    Build Ragnarok-style results from Anthropic (Claude) batch outputs.

    results_jsonl_path lines look like:
      {"custom_id": "...",
       "result": {"type": "succeeded",
                  "message": {"content":[{"type":"text","text":"..."}], ...}}}

    If you shortened IDs for Claude, provide id_mapping_path (JSON with short->original).
    """
    # 0) load optional mapping (short -> original)
    short2orig: Dict[str, str] = {}
    if id_mapping_path and os.path.exists(id_mapping_path):
        with open(id_mapping_path, "r", encoding="utf-8") as mf:
            short2orig = json.load(mf)

    # 1) qid -> {text, references}
    request_info: Dict[str, Dict[str, Any]] = {}
    with open(requests_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            item = json.loads(s)
            qid = item["query"]["qid"]
            request_info[qid] = {
                "text": item["query"]["text"],
                "references": [c["docid"] for c in item.get("candidates", [])],
            }

    # 2) parse results
    results: List[Result] = []
    unmatched_count = 0

    with open(results_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            cid = row.get("custom_id")
            res = row.get("result") or {}

            # only keep successful replies
            if not cid or res.get("type") != "succeeded":
                continue

            # map back to original qid if a mapping was used
            qid = short2orig.get(cid, cid)

            info = request_info.get(qid)
            if not info:
                unmatched_count += 1
                # keep going; just skip if we can't map
                # print(f"⚠️ Skipping unmatched qid/custom_id: {cid}")
                continue

            msg = res.get("message") or {}
            response_text = _claude_concat_text(msg)

            query = Query(text=info["text"], qid=qid)
            cited_sentences = extract_cited_sentences(response_text)
            results.append(Result(query=query, references=info["references"], answer=cited_sentences))

    if unmatched_count:
        print(f"⚠️ Skipped {unmatched_count} results due to missing mapping or unknown qids.")

    return results


def postprocess_nonrag_responses_claude(
    csv_path: str,
    jsonl_path: str,        # Anthropic results .jsonl
    output_csv_path: str,
) -> None:
    """
    Anthropic non-RAG results → flat CSV with:
      qid, query, description, tone, llm_response, gt_stance

    Assumes custom_id pattern: {tone}_qid_{N}
    Expects Anthropic shape row['result']['type']=='succeeded' and row['result']['message']
    """
    # CSV keyed by qid
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_qid = {row["qid"]: row for row in rows}

    out_rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("custom_id")
            res = row.get("result") or {}
            if not cid or res.get("type") != "succeeded":
                continue

            m = re.match(r"(\w+)_qid_(\d+)", cid)
            if not m:
                continue
            tone = m.group(1)
            qid = f"qid_{m.group(2)}"
            prompt_col = f"{tone}_prompt"

            llm_response = _claude_concat_text(res.get("message"))
            # print(llm_response)

            csv_row = by_qid.get(qid)
            # print(csv_row)
            if not csv_row:
                continue

            out_rows.append({
                "qid": qid,
                "query": csv_row["title"],
                "description": csv_row.get(prompt_col, ""),
                "tone": tone,
                "llm_response": llm_response,
                "gt_stance": csv_row["answer"],
            })
            # print(out_rows)

    if out_rows:
        os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
        pd.DataFrame(out_rows, columns=["qid", "query", "description", "tone", "llm_response", "gt_stance"])\
          .to_csv(output_csv_path, index=False, encoding="utf-8")


def find_claude_non_succeeded(jsonl_path: str) -> List[tuple]:
    """
    Return list of (line_no, custom_id, result_type) where result.type != 'succeeded'
    or message/content missing (Anthropic format).
    """
    bad = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                bad.append((i, None, "invalid_json"))
                continue
            cid = row.get("custom_id")
            res = row.get("result") or {}
            rtype = res.get("type")
            if rtype != "succeeded":
                bad.append((i, cid, rtype or "missing_type"))
                continue
            msg = res.get("message") or {}
            if not msg.get("content"):
                bad.append((i, cid, "no_content"))
    return bad





