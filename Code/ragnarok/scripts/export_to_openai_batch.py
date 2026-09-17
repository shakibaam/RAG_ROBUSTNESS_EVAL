import os
import json
from tqdm import tqdm
from generate.gpt import SafeOpenai
from generate.llm import PromptMode
from data import read_requests_from_file
from typing import Tuple, List, Dict, Any
import hashlib
import re


def update_custom_id_and_qid(custom_id):
    """
    Modifies the 'custom_id' field and updates 'request.query.qid' accordingly.
    
    Example:
    From: consistent_harmful_qid_102_doc_en.noclean.c4-train.00061-of-07168.101911.md
    To:   consistent_qid_102
    And updates: request.query.qid = 102
    """
    
    # Match the pattern: (consistent|inconsistent|neutral)_.*?_qid_(\d+)
    match = re.search(r'(consistent|inconsistent|neutral)_.*?_qid_(\d+)', custom_id)
    if match:
        consistency = match.group(1)
        qid = match.group(2)
        new_custom_id = f"{consistency}_qid_{qid}"
 
    
    return new_custom_id



def update_jsonl_custom_ids(input_path, output_path):
    """
    Reads a JSONL file, updates each record's custom_id and request.query.qid,
    and writes the updated records to a new file.
    
    - custom_id pattern: (consistent|inconsistent|neutral)_.*?_qid_(\d+)
    - Example:
        From: consistent_harmful_qid_102_doc_en.noclean.c4-train.00061-of-07168.101911.md
        To:   consistent_qid_102
        And request.query.qid = 102
    """
    def update_custom_id_and_qid(custom_id):
        match = re.search(r'(consistent|inconsistent|neutral)_.*?_qid_(\d+)', custom_id)
        if match:
            consistency = match.group(1)
            qid = match.group(2)
            new_custom_id = f"{consistency}_qid_{qid}"
            return new_custom_id, int(qid)
        return custom_id, None

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            record = json.loads(line)
            
            old_custom_id = record.get("custom_id", "")
            new_custom_id, qid = update_custom_id_and_qid(old_custom_id)
            
            record["custom_id"] = new_custom_id
            if qid is not None:
                record.setdefault("request", {}).setdefault("query", {})["qid"] = qid
            
            outfile.write(json.dumps(record) + "\n")

    print(f"Updated file saved to {output_path}")

def export_batch(dir_go, output_dir):
    """
    Processes all .jsonl files in dir_go, creates batch .jsonl files in output_dir with '_batch' appended to the filename.
    """
    MODEL = "gpt-4.1-2025-04-14"
    CONTEXT_SIZE = 131072
    MAX_TOKENS = 1000
    TOP_K = 5
    TEMPERATURE = 0.1
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(dir_go):
        if not fname.endswith('.jsonl'):
            continue
        input_path = os.path.join(dir_go, fname)
        base, ext = os.path.splitext(fname)
        output_fname = f"{base}_batch{ext}"
        output_path = os.path.join(output_dir, output_fname)

        # Step 1: Load requests
        requests = read_requests_from_file(input_path)

        # Step 2: Initialize SafeOpenai
        agent = SafeOpenai(
            model=MODEL,
            context_size=CONTEXT_SIZE,
            prompt_mode=PromptMode.CHATQA,
            max_output_tokens=MAX_TOKENS,
            num_few_shot_examples=0,
            keys=OPENAI_KEY,
        )

        # Step 3: Build OpenAI Batch JSONL
        batch_requests = []
        for i, request in tqdm(enumerate(requests), total=len(requests), desc=fname):
            try:
                prompt, _ = agent.create_prompt(request, topk=TOP_K)
                batch_entry = {
                    "custom_id": request.query.qid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": prompt,
                        "max_tokens": MAX_TOKENS,
                        "temperature": TEMPERATURE,
                    }
                }
                batch_requests.append(batch_entry)
            except Exception as e:
                print(f"⚠️ Error in request {i} in {fname}: {e}")

        # Step 4: Write to .jsonl file
        with open(output_path, "w") as f:
            for entry in batch_requests:
                f.write(json.dumps(entry) + "\n")

        print(f"\n✅ Exported {len(batch_requests)} batch requests to {output_path}")
    print("All files processed.")


def export_single_file_to_openai_batch(input_path, output_path):
    """
    Processes a single .jsonl file, creates a batch .jsonl file at the given output_path.
    """
    MODEL = "gpt-4.1-2025-04-14"
    CONTEXT_SIZE = 131072
    MAX_TOKENS = 1000
    TOP_K = 10
    TEMPERATURE = 0.1
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fname = os.path.basename(input_path)

    # Step 1: Load requests
    requests = read_requests_from_file(input_path)

    # Step 2: Initialize SafeOpenai
    agent = SafeOpenai(
        model=MODEL,
        context_size=CONTEXT_SIZE,
        prompt_mode=PromptMode.CHATQA,
        max_output_tokens=MAX_TOKENS,
        # max_tokens=MAX_TOKENS,
        num_few_shot_examples=0,
        keys=OPENAI_KEY
    )

    # Step 3: Build OpenAI Batch JSONL
    batch_requests = []
    for i, request in tqdm(enumerate(requests), total=len(requests), desc=fname):
        try:
            prompt, _ = agent.create_prompt(request, topk=TOP_K)
            batch_entry = {
                "custom_id": request.query.qid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": prompt,
                    # "max_completion_tokens": MAX_TOKENS,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                }
            }
            batch_requests.append(batch_entry)
        except Exception as e:
            print(f"⚠️ Error in request {i} in {fname}: {e}")

    # Step 4: Write to .jsonl file
    with open(output_path, "w") as f:
        for entry in batch_requests:
            f.write(json.dumps(entry) + "\n")

    print(f"\n✅ Exported {len(batch_requests)} batch requests to {output_path}")



def _to_anthropic_messages_and_system(openai_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str | None]:
    """
    Convert OpenAI Chat Completion messages to Anthropic format.
    - Collect all 'system' messages into a single top-level system string.
    - Keep 'user' and 'assistant' messages.
    - If content is a list of blocks, join text parts; otherwise, stringify.
    - Silently drops roles Anthropic doesn't accept in simple batches (e.g., 'tool').
    """
    def _normalize_content(c):
        if isinstance(c, list):
            texts = []
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text" and "text" in b:
                    texts.append(b["text"])
                elif isinstance(b, str):
                    texts.append(b)
                else:
                    # fallback (rare)
                    texts.append(json.dumps(b, ensure_ascii=False))
            return "\n".join(texts)
        return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)

    system_chunks = []
    conv: List[Dict[str, str]] = []
    for m in openai_messages or []:
        role = m.get("role")
        content = _normalize_content(m.get("content", ""))

        if role == "system":
            system_chunks.append(content)
        elif role in ("user", "assistant"):
            conv.append({"role": role, "content": content})
        else:
            # skip roles like 'tool' for simple batch usage
            continue

    system_str = "\n\n".join(system_chunks) if system_chunks else None
    return conv, system_str



def shorten_id_with_mapping(original_id: str, mapping: dict) -> str:
    """
    Return a Claude-safe <=64 char ID, store mapping from short→original.
    Allowed chars: [a-zA-Z0-9_-]
    """
    # Replace disallowed chars with underscore
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', original_id)

    if len(safe_id) <= 64:
        mapping[safe_id] = original_id
        return safe_id

    # If still too long, truncate and append hash to guarantee uniqueness
    hash_part = hashlib.md5(original_id.encode()).hexdigest()[:8]
    short_id = f"{safe_id[:50]}_{hash_part}"
    mapping[short_id] = original_id
    return short_id

def convert_openai_batch_jsonl_to_anthropic(
    input_path: str,
    output_path: str,
    mapping_path: str,   # new: where to save mapping
    anthropic_model: str = "claude-3-7-sonnet-20250219",
    default_max_tokens: int = 1000,
    default_temperature: float = 0.1,
    output_format: str = "json",
):
    """
    Convert OpenAI batch JSONL → Anthropic batch JSON(L) with <=64-char IDs.
    Saves a mapping file to restore original IDs later.
    """
    requests_out = []
    mapping = {}

    with open(input_path, "r", encoding="utf-8") as fin:
        for ln, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"Skipping line {ln}: JSON decode error: {e}")
                continue

            if item.get("url") not in ("/v1/chat/completions", "/v1/responses", None):
                print(f"Warning line {ln}: unexpected url {item.get('url')}")

            orig_id = item.get("custom_id") or f"line-{ln}"
            short_id = shorten_id_with_mapping(orig_id, mapping)

            body = item.get("body") or {}
            model = anthropic_model
            openai_messages = body.get("messages", [])
            temperature = body.get("temperature", default_temperature)
            max_tokens = body.get("max_tokens", default_max_tokens)

            messages, system_str = _to_anthropic_messages_and_system(openai_messages)

            params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system_str:
                params["system"] = system_str

            requests_out.append({"custom_id": short_id, "params": params})

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)

    if output_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as fout:
            for r in requests_out:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump({"requests": requests_out}, fout, ensure_ascii=False)

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"✅ Converted {len(requests_out)} requests → {output_path}")
    print(f"✅ Saved ID mapping → {mapping_path}")


def restore_custom_ids(claude_results_jsonl, mapping_path, restored_jsonl):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    with open(claude_results_jsonl, "r", encoding="utf-8") as fin, \
         open(restored_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("custom_id")
            if cid in mapping:
                row["custom_id"] = mapping[cid]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def remove_duplicate_custom_ids(input_path, output_path):
    seen_ids = set()
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                obj = json.loads(line)
                cid = obj.get("custom_id")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                continue  # skip malformed lines



def remove_request_query_qid(input_path, output_path):
    """
    Reads a JSONL file and removes the 'request': {'query': {'qid': ...}} structure from each line.
    Writes the cleaned JSONL to output_path.
    """
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            record = json.loads(line)
            
            if "request" in record and isinstance(record["request"], dict):
                if "query" in record["request"] and isinstance(record["request"]["query"], dict):
                    record["request"]["query"].pop("qid", None)
                    # If query becomes empty, remove it
                    if not record["request"]["query"]:
                        record["request"].pop("query", None)
                # If request becomes empty, remove it
                if not record["request"]:
                    record.pop("request", None)
            
            outfile.write(json.dumps(record) + "\n")
    
    return output_path


