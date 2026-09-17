import json
import re
from typing import List
from pathlib import Path
from data import Query, CitedSentence, Result, DataWriter
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

import pandas as pd

# Ensure sentence tokenizer is available
nltk.download("punkt")

# Setup custom tokenizer
punkt_param = PunktParameters()
punkt_tokenizer = PunktSentenceTokenizer(punkt_param)

def sent_tokenize(text: str) -> List[str]:
    return punkt_tokenizer.tokenize(text)

def extract_cited_sentences(text: str) -> List[CitedSentence]:
    """
    Splits text into sentences, extracts inline citations, and removes them from the sentence text.
    Returns a list of CitedSentence objects.
    """
    sentences = sent_tokenize(text)
    cited_sentences = []

    for sent in sentences:
        matches = re.findall(r"\[(\d+)\]", sent)
        citation_indices = sorted(set(int(m) - 1 for m in matches if m.isdigit()))  # convert to 0-based
        clean_sent = re.sub(r"\s*\[\d+\]", "", sent).strip()
        cited_sentences.append(CitedSentence(text=clean_sent, citations=citation_indices))

    return cited_sentences

def build_results_from_batch(
    input_path: str,
    output_path: str,
    request_path: str,
    run_id: str = "openai_batch"
) -> List[Result]:
    """
    Builds a list of Ragnarok-style Result objects from OpenAI batch input/output.
    Maps custom_id to qid, retrieves references from candidates.
    """
    # Load references and queries from request file
    request_info = {}
    with open(request_path, "r") as f:
        for line in f:
            item = json.loads(line)
            qid = item["query"]["qid"]
            query_text = item["query"]["text"]
            candidates = item.get("candidates", [])
            references = [c["docid"] for c in candidates]
            request_info[qid] = {
                "text": query_text,
                "references": references
            }

    # Load input file just for structure (can skip if not needed)
    with open(input_path, "r") as f:
        input_data = {
            json.loads(line)["custom_id"]: json.loads(line)
            for line in f if line.strip()
        }

    results = []

    # Load output JSONL: response data with citations
    with open(output_path, "r") as f:
        for line in f:
            row = json.loads(line)
            print(row)
            custom_id = row.get("custom_id")
            if not custom_id or "response" not in row:
                continue

            response_content = row["response"]["body"]["choices"][0]["message"]["content"]
            qid = custom_id  # custom_id is assumed to match qid exactly

            if qid not in request_info:
                print(f"⚠️ Skipping unmatched qid: {qid}")
                continue

            query_text = request_info[qid]["text"]
            references = request_info[qid]["references"]

            query = Query(text=query_text, qid=qid)
            cited_sentences = extract_cited_sentences(response_content)

            result = Result(query=query, references=references, answer=cited_sentences)
            results.append(result)

    return results


def postprocess_nonrag_responses(
    csv_path,
    jsonl_path,
    output_csv_path
):
    # 1. Load CSV into a dict keyed by qid
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
        csv_by_qid = {row['qid']: row for row in csv_rows}

    # 2. Parse JSONL and collect results
    results = []
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            custom_id = row.get('custom_id')
            if not custom_id or 'response' not in row:
                continue
            # Extract prompt_type and qid
            m = re.match(r'(\w+)_qid_(\d+)', custom_id)
            if not m:
                continue
            prompt_type = m.group(1)
            qid = f"qid_{m.group(2)}"
            prompt_col = f'{prompt_type}_prompt'

            # Get LLM response
            llm_response = row['response']['body']['choices'][0]['message']['content']

            # Find CSV row
            csv_row = csv_by_qid.get(qid)
            if not csv_row:
                continue

            # Compose output row with only the relevant prompt column
            out_row = {
                'qid': qid,
                'query': csv_row['title'],
                'description': csv_row.get(prompt_col, ''),
                'tone': prompt_type,
                'llm_response': llm_response,
                'gt_stance': csv_row['answer'],
            }
            results.append((out_row, prompt_col))

    # 3. Write output CSV using pandas
    if results:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        # Group by prompt_col to ensure each group has the right columns
        dfs = []
        for out_row, prompt_col in results:
            columns = ['qid', 'query', 'description', 'tone', 'llm_response', 'gt_stance']
            df = pd.DataFrame([out_row], columns=columns)
            dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(output_csv_path, index=False, encoding='utf-8')

def find_lines_missing_choices(jsonl_path):
    """
    Returns a list of (line_number, line_content) for lines in the JSONL file where
    'choices' is missing under 'response.body'.
    """
    missing = []
    with open(jsonl_path, encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except Exception:
                missing.append((idx, line.strip()))
                continue
            if (
                'response' in row and
                'body' in row['response'] and
                (
                    'choices' not in row['response']['body'] or
                    row['response']['body']['choices'] is None
                )
            ):
                missing.append((idx, line.strip()))
    return missing

def batch_process_paired_experiments(root_dir: str):
    """
    Processes all paired experiments in the given root_dir using the naming convention provided.
    """
    from pathlib import Path
    import os

    batch_input_dir = os.path.join(root_dir, "neutral_batch_requests")
    request_dir = os.path.join(root_dir, "neutral_requests")
    output_dir = os.path.join(root_dir, "neutral_results", "batch_results")
    save_dir = os.path.join(root_dir, "neutral_results", "ragnarok_format")
    os.makedirs(save_dir, exist_ok=True)

    # Find all batch input files
    batch_input_files = list(Path(batch_input_dir).glob("*_requests_batch.jsonl"))
    print(f"Found {len(batch_input_files)} batch input files in {batch_input_dir}")

    for batch_input_path in batch_input_files:
        basename = batch_input_path.name.replace("_requests_batch.jsonl", "")
        request_path = Path(request_dir) / f"{basename}_requests.jsonl"
        batch_output_path = Path(output_dir) / f"{basename}_requests_batch_output.jsonl"
        output_file = Path(save_dir) / f"{basename}_ragnarok_results_format.jsonl"
        run_id = "openai_batch_run"

        # Check if all files exist
        if not request_path.exists():
            print(f"❌ Request file missing: {request_path}")
            continue
        if not batch_output_path.exists():
            print(f"❌ Batch output file missing: {batch_output_path}")
            continue

        print(f"\nProcessing: {basename}")
        print(f"  Input:   {batch_input_path}")
        print(f"  Output:  {batch_output_path}")
        print(f"  Request: {request_path}")
        print(f"  Save to: {output_file}")

        results = build_results_from_batch(
            str(batch_input_path),
            str(batch_output_path),
            str(request_path),
            run_id
        )
        writer = DataWriter(results)
        writer.write_in_jsonl_format(str(output_file), run_id=run_id)
        print(f"✅ Saved {len(results)} results to {output_file}")



from pathlib import Path

from pathlib import Path

def process_batch_dirs(
    batch_inputs_dir: str,
    batch_results_dir: str,
    requests_dir: str,
    results_dir: str,
    run_id: str = "openai_batch_run",
):


    batch_inputs_dir  = Path(batch_inputs_dir).resolve()
    batch_results_dir = Path(batch_results_dir).resolve()
    requests_dir      = Path(requests_dir).resolve()
    results_dir       = Path(results_dir).resolve()

    results_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(list(batch_results_dir.rglob("*_requests_batch_output.jsonl"))) \
                 + sorted(list(batch_results_dir.rglob("*_requests_batch_results.jsonl")))

    if not result_files:
        print(f"⚠️ No batch results found in: {batch_results_dir}")
        return

    total_ok = total_skip = 0

    for batch_result in result_files:
        # Keep subdir path relative to results_dir
        rel_subdir = batch_result.parent.relative_to(batch_results_dir)

        name = batch_result.name
        if name.endswith("_requests_batch_output.jsonl"):
            base = name[: -len("_requests_batch_output.jsonl")]
        elif name.endswith("_requests_batch_results.jsonl"):
            base = name[: -len("_requests_batch_results.jsonl")]
        else:
            print(f"❌ Skip unknown pattern: {name}")
            total_skip += 1
            continue

        batch_input  = batch_inputs_dir / rel_subdir / f"{base}_requests_batch.jsonl"
        request_file = requests_dir     / rel_subdir / f"{base}_requests.jsonl"
        out_file     = results_dir      / rel_subdir / f"{base}_ragnarok_format_results.jsonl"

        out_file.parent.mkdir(parents=True, exist_ok=True)

        missing = []
        if not batch_input.exists():
            missing.append(f"batch_input missing: {batch_input}")
        if not request_file.exists():
            missing.append(f"request_file missing: {request_file}")
        if missing:
            print(f"❌ {base}: " + " | ".join(missing))
            total_skip += 1
            continue

        try:
            results = build_results_from_batch(
                str(batch_input),
                str(batch_result),
                str(request_file),
                run_id,
            )
            writer = DataWriter(results)
            writer.write_in_jsonl_format(str(out_file), run_id=run_id)
            print(f"✅ {base}: saved {len(results)} results → {out_file}")
            total_ok += 1
        except Exception as e:
            print(f"❌ {base}: error: {e}")
            total_skip += 1

    print(f"\nDone. ✅ {total_ok} processed, ❌ {total_skip} skipped.")




 
    