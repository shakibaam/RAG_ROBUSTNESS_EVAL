import csv
import json
import argparse
from data import Query, Candidate, Request, DataWriter
import os
import random
import re
import pandas as pd


def sanitize_filename(s):
    # Replace all non-alphanumeric characters with underscores
    return re.sub(r'[^A-Za-z0-9_\-]', '_', s)

def load_queries(csv_path: str, query_col: str = "query", qid_col: str = None):
    queries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            query_text = row[query_col]
            if qid_col and qid_col in row:
                qid = row[qid_col]
            else:
                qid = f"q{idx+1}"
            queries.append(Query(text=query_text, qid=qid))
    print(f"Loaded {len(queries)} queries from {csv_path}")
    return queries

def load_documents(jsonl_path: str):
    documents = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            documents.append(doc)
    print(f"Loaded {len(documents)} documents from {jsonl_path}")
    return documents
def build_requests(queries, documents):
    # First, index documents by qid (prefix of docid)
    doc_lookup = {}
    for doc in documents:
        docid = doc["id"]
        # Example: "q1_support" → "q1"
        qid = docid.split("_")[0]
        doc_lookup[qid] = doc
    
    # Now, build requests
    requests = []
    for query in queries:
        qid = str(query.qid)  # ensure it's str to match
        if qid in doc_lookup:
            doc = doc_lookup[qid]
            candidate = Candidate(
                docid=doc["id"],
                score=1.0,  # can adjust
                doc={"contents": doc.get("contents", "")}
            )
            request = Request(query=query, candidates=[candidate])
            requests.append(request)
        else:
            print(f"Warning: No document found for query {qid}")
    
    print(f"Prepared {len(requests)} Request objects")
    return requests

def main(args):
    queries = load_queries(args.queries, args.query_col, args.qid_col)
    documents = load_documents(args.docs)
    requests = build_requests(queries, documents)

    # Save as JSONL
    with open(args.output, "w") as f:
        for request in requests:
            request_dict = {
                "query": request.query.__dict__,
                "candidates": [cand.__dict__ for cand in request.candidates],
                "ranking_exec_summary": request.ranking_exec_summary,
            }
            json.dump(request_dict, f)
            f.write("\n")


def read_document(filepath):
    with open(filepath, "r") as f:
        return f.read().strip()



def create_rag_requests_from_csv(csv_path, trec_data_dir, output_dir):
    # Open CSV
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)

       
        harmful_files = {
            tone: open(f"{output_dir}/{tone}_harmful_requests.jsonl", "w")
            for tone in ["neutral", "consistent", "inconsistent"]
        }

        for row in reader:
            qid = row["qid"].strip()
            print(row['consistent_prompt'])

     
            harmful_dir = os.path.join(trec_data_dir, qid, "harmful_documents")

            # helpful_files_list = [f for f in os.listdir(helpful_dir) if f.endswith(".md")]
            harmful_files_list = [f for f in os.listdir(harmful_dir) if f.endswith(".md")]

            for tone in ["neutral", "consistent", "inconsistent"]:
                prompt_text = row[f"{tone}_prompt"].strip()

               

                for doc in harmful_files_list:
                    doc_path = os.path.join(harmful_dir, doc)
                    content = read_document(doc_path)
                    request = {
                        "query": {
                            "text": prompt_text,
                            "qid": f"{tone}_harmful_{qid}_{doc}"
                        },
                        "candidates": [
                            {
                                "docid": doc,
                                "score": 1.0,
                                "doc": {"contents": content}
                            }
                        ]
                    }
                    harmful_files[tone].write(json.dumps(request) + "\n")

        for f in list(harmful_files.values()):
            f.close()

        print(f"✅ Created helpful & harmful request files for all tones and all documents.")



def create_adversarial_rag_requests_from_csv(csv_path, trec_data_dir, output_dir):
    models = ["GPT-4o", "DeepSeek-R1-claude3.7"]
    tones = ["neutral", "consistent", "inconsistent"]


    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)


        adv_files = {
            model: {
                tone: open(f"{output_dir}/{tone}_{model.replace('-', '_')}_adversarial_requests.jsonl", "w")
                for tone in tones
            }
            for model in models
        }

        for row in reader:
            qid = row["qid"].strip()

      
            adv_root = os.path.join(trec_data_dir, qid, "adversarial_documents")

        
            for doc_folder in os.listdir(adv_root):
                doc_folder_path = os.path.join(adv_root, doc_folder)
                if not os.path.isdir(doc_folder_path):
                    continue

                for model in models:
                    model_folder_path = os.path.join(doc_folder_path, model)
                    if not os.path.exists(model_folder_path):
                        continue

                    for file_name in os.listdir(model_folder_path):
                        if not file_name.endswith(".md"):
                            continue

                        file_path = os.path.join(model_folder_path, file_name)

                   
                        file_content = read_document(file_path)

                        docid = f"{doc_folder}_{model}_{file_name}"

                        for tone in tones:
                            prompt_text = row[f"{tone}_prompt"].strip()

                            
                            custom_id = f"{tone}_adversarial_{qid}_{doc_folder}_{model}_{file_name.replace('.md', '')}"

                            adversarial_request = {
                                "query": {
                                    "text": prompt_text,
                                    "qid": custom_id
                                },
                                "candidates": [
                                    {
                                        "docid": docid,
                                        "score": 1.0,
                                        "doc": {"contents": file_content}
                                    }
                                ]
                            }

                            adv_files[model][tone].write(json.dumps(adversarial_request) + "\n")

 
        for model_tone_dict in adv_files.values():
            for f in model_tone_dict.values():
                f.close()

        print("✅ Created adversarial request files for GPT-4o and DeepSeek across all tones.")


def update_qid_in_jsonl(input_path, output_path, new_prefix):
    """
    For each line, extract the qid_XXX part from the original qid, set the new qid to newprefix_qid_XXX,
    remove duplicates, and save the modified file.

    Args:
        input_path (str): Path to the input .jsonl file.
        output_path (str): Path to save the updated .jsonl file.
        new_prefix (str): The new prefix to use for all entries.
    """
    seen_qids = set()
    qid_pattern = re.compile(r"(qid_\d+)")
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            old_qid = data["query"]["qid"]
            match = qid_pattern.search(old_qid)
            if match:
                qid_part = match.group(1)
                new_qid = f"{new_prefix}_{qid_part}"
                data["query"]["qid"] = new_qid
                # Only write if this new qid hasn't been seen yet
                if new_qid not in seen_qids:
                    seen_qids.add(new_qid)
                    outfile.write(json.dumps(data) + "\n")
    print(f"Set all qids to '{new_prefix}_qid_XXX', removed duplicates, and saved to: {output_path}")





def create_rag_requests_from_pairs_json(
    pairs_json_path, trec_data_dir, output_dir, prompt_text="Paired RAG request"
):
    """
    Creates RAG requests from document pairs in a JSON file.
    Each request contains two documents as candidates.
    """
    import os
    import json


    with open(pairs_json_path, "r") as f:
        pairs_data = json.load(f)


    pair_types = set()
    for q_pairs in pairs_data.values():
        pair_types.update(q_pairs.keys())
    out_files = {
        pair_type: open(os.path.join(output_dir, f"{pair_type.replace(' ', '_')}_requests.jsonl"), "w")
        for pair_type in pair_types
    }

    def read_doc(qid, doc_path):
      
        if "/" in doc_path:
 
            adv_path = os.path.join(
                trec_data_dir, qid, "adversarial_documents", *doc_path.split("/")
            )
            if os.path.exists(adv_path):
                with open(adv_path, "r") as f:
                    return f.read().strip()
        else:
        
            helpful_path = os.path.join(trec_data_dir, qid, "helpful_documents", doc_path)
            if os.path.exists(helpful_path):
                with open(helpful_path, "r") as f:
                    return f.read().strip()
     
            harmful_path = os.path.join(trec_data_dir, qid, "harmful_documents", doc_path)
            if os.path.exists(harmful_path):
                with open(harmful_path, "r") as f:
                    return f.read().strip()
      
        print(f"Warning: Could not find document for qid={qid}, doc_path={doc_path}")
        return ""

    for qid, q_pairs in pairs_data.items():
        for pair_type, pairs in q_pairs.items():
            for idx, (doc1, doc2) in enumerate(pairs):
                content1 = read_doc(qid, doc1)
                content2 = read_doc(qid, doc2)
                request = {
                    "query": {
                        "text": prompt_text,
                        "qid": f"{pair_type}_{qid}_{idx}"
                    },
                    "candidates": [
                        {
                            "docid": doc1,
                            "score": 1.0,
                            "doc": {"contents": content1}
                        },
                        {
                            "docid": doc2,
                            "score": 1.0,
                            "doc": {"contents": content2}
                        }
                    ]
                }
                out_files[pair_type].write(json.dumps(request) + "\n")

    for f in out_files.values():
        f.close()
    print("✅ Created paired RAG request files for all pair types.")


def create_rag_requests_from_pairs_json_and_csv(
    csv_path, pairs_json_path, trec_data_dir, output_dir
):
    """
    For each qid in the CSV, uses the prompts in the row and the document pairs in the JSON.
    Creates RAG requests for each pair and each prompt type.

    Args:
        csv_path (str): Path to the CSV file with qid and prompt columns.
        pairs_json_path (str): Path to the document_pairs.json file.
        trec_data_dir (str): The FULL root directory containing all qid_* folders (e.g., .../TREC 2021).
        output_dir (str): Directory to write output JSONL files.
    """
    import os
    import csv
    import json

    # Load pairs
    with open(pairs_json_path, "r") as f:
        pairs_data = json.load(f)

    # Read CSV and build a mapping: qid -> {prompt_type: prompt_text}
    qid_to_prompts = {}
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            qid = row["qid"].strip()
            # Adjust these keys if your CSV uses different column names
            qid_to_prompts[qid] = {
                "neutral": row.get("neutral_prompt", "").strip(),
                "consistent": row.get("consistent_prompt", "").strip(),
                "inconsistent": row.get("inconsistent_prompt", "").strip(),
            }

    # Prepare output files for each pair type and prompt type
    pair_types = set()
    for q_pairs in pairs_data.values():
        pair_types.update(q_pairs.keys())
    prompt_types = ["neutral", "consistent", "inconsistent"]
    out_files = {}

    for pair_type in pair_types:
        for prompt_type in prompt_types:
            safe_pair_type = sanitize_filename(pair_type)
            safe_prompt_type = sanitize_filename(prompt_type)
            output_path = os.path.join(output_dir, f"{safe_prompt_type}_{safe_pair_type}_requests.jsonl")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out_files[(pair_type, prompt_type)] = open(output_path, "w")

    def read_doc(trec_data_dir, qid, doc_path):
        full_path = os.path.join(trec_data_dir, qid, doc_path)
        print(full_path)
        # if os.path.exists(full_path):
        #     print("exists")
        with open(full_path, "r") as f:
            return f.read().strip()
        # print(f"Warning: Could not find document for qid={qid}, doc_path={doc_path}")
        # return ""

    for qid, q_pairs in pairs_data.items():
        if qid not in qid_to_prompts:
            print(f"Warning: qid {qid} not found in CSV, skipping.")
            continue
        for pair_type, pairs in q_pairs.items():
            for idx, (doc1, doc2) in enumerate(pairs):
                content1 = read_doc(trec_data_dir, qid, doc1)
                content2 = read_doc(trec_data_dir, qid, doc2)
                for prompt_type in prompt_types:
                    prompt_text = qid_to_prompts[qid][prompt_type]
                    request = {
                        "query": {
                            "text": prompt_text,
                            "qid": f"{prompt_type}_{pair_type}_{qid}_{idx}"
                        },
                        "candidates": [
                            {
                                "docid": doc1,
                                "score": 1.0,
                                "doc": {"contents": content1}
                            },
                            {
                                "docid": doc2,
                                "score": 1.0,
                                "doc": {"contents": content2}
                            }
                        ]
                    }
                    out_files[(pair_type, prompt_type)].write(json.dumps(request) + "\n")

    for f in out_files.values():
        f.close()
    print("✅ Created paired RAG request files for all pair types and prompt types, using prompts from CSV.")


def print_jsonl_line_counts(directory):
    """
    Lists all .jsonl files in the given directory and prints the number of lines in each file.
    Also prints the total number of lines across all files.
    """
    import os
    total_lines = 0
    for fname in os.listdir(directory):
        if fname.endswith('.jsonl'):
            fpath = os.path.join(directory, fname)
            with open(fpath, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"{fname}: {line_count} lines")
            total_lines += line_count
    print(f"Total lines in all .jsonl files: {total_lines}")




def create_requests_from_reranker_csv(csv_path, output_dir, prompt_text="RAG request with top segments"):
    """
    Creates RAG requests from reranker CSV file.
    For each QID, creates two types of requests:
    1. Highest score segment from each helpful doc + highest score segment from each harmful doc
    2. Highest score segment from each helpful doc + highest score segment from each attack type
    
    Args:
        csv_path (str): Path to the reranker CSV file
        output_dir (str): Directory to save output JSONL files
        prompt_text (str): Text to use as the query prompt
    """
    import pandas as pd
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} segments from {csv_path}")
    
    # Group by QID
    qid_groups = df.groupby('QID')
    
    # Prepare output files
    helpful_harmful_file = open(os.path.join(output_dir, "helpful_harmful_requests.jsonl"), "w")
    helpful_attack_file = open(os.path.join(output_dir, "helpful_attack_requests.jsonl"), "w")
    
    for qid, group in qid_groups:
        print(f"Processing QID: {qid}")
        
        # Get highest score segment for each document (group by DocName and get max score)
        doc_best_segments = group.loc[group.groupby('DocName')['Score'].idxmax()]
        
        # Separate by DocType
        helpful_segments = doc_best_segments[doc_best_segments['DocType'] == 'helpful']
        harmful_segments = doc_best_segments[doc_best_segments['DocType'] == 'harmful']
        
        # Create Request Type 1: Helpful + Harmful
        if len(helpful_segments) > 0 and len(harmful_segments) > 0:
            candidates = []
            
            # Add helpful segments
            for _, segment in helpful_segments.iterrows():
                candidates.append({
                    "docid": f"helpful_{segment['DocName']}_{segment['SegmentIndex']}",
                    "score": segment['Score'],
                    "doc": {"contents": segment['SegmentText']}
                })
            
            # Add harmful segments
            for _, segment in harmful_segments.iterrows():
                candidates.append({
                    "docid": f"harmful_{segment['DocName']}_{segment['SegmentIndex']}",
                    "score": segment['Score'],
                    "doc": {"contents": segment['SegmentText']}
                })
            
            request = {
                "query": {
                    "text": prompt_text,
                    "qid": f"helpful_harmful_{qid}"
                },
                "candidates": candidates
            }
            helpful_harmful_file.write(json.dumps(request) + "\n")
        
        # Create Request Type 2: Helpful + Attack (assuming attack types are in DocName)
        if len(helpful_segments) > 0:
            # Group harmful segments by attack type (extract from DocName)
            attack_groups = harmful_segments.groupby('DocName')
            
            for attack_doc, attack_segments in attack_groups:
                # Get the highest scoring segment for this attack type
                best_attack_segment = attack_segments.loc[attack_segments['Score'].idxmax()]
                
                candidates = []
                
                # Add helpful segments
                for _, segment in helpful_segments.iterrows():
                    candidates.append({
                        "docid": f"helpful_{segment['DocName']}_{segment['SegmentIndex']}",
                        "score": segment['Score'],
                        "doc": {"contents": segment['SegmentText']}
                    })
                
                # Add the best attack segment
                candidates.append({
                    "docid": f"attack_{best_attack_segment['DocName']}_{best_attack_segment['SegmentIndex']}",
                    "score": best_attack_segment['Score'],
                    "doc": {"contents": best_attack_segment['SegmentText']}
                })
                
                request = {
                    "query": {
                        "text": prompt_text,
                        "qid": f"helpful_attack_{qid}_{attack_doc.replace('.md', '')}"
                    },
                    "candidates": candidates
                }
                helpful_attack_file.write(json.dumps(request) + "\n")
    
    helpful_harmful_file.close()
    helpful_attack_file.close()
    print(f"✅ Created request files in {output_dir}")
    print(f"  - helpful_harmful_requests.jsonl: Combined helpful + harmful segments")
    print(f"  - helpful_attack_requests.jsonl: Combined helpful + attack segments")



def create_requests_from_reranker_csv_with_prompts(reranker_csv_path, prompts_csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    reranker_df = pd.read_csv(reranker_csv_path)
    prompts_df = pd.read_csv(prompts_csv_path)
    print(f"Loaded {len(reranker_df)} segments from reranker CSV")
    print(f"Loaded {len(prompts_df)} prompts from prompts CSV")
    adv_df = reranker_df[reranker_df['DocType'] == 'adversarial_GPT-4o']

    qid_to_attacks = adv_df.groupby('QID')['AttackType'].unique()

    for qid, attack_types in qid_to_attacks.items():
        print(f"QID {qid}: {sorted(attack_types)}")

    # Build mapping: QID -> {neutral/consistent/inconsistent: prompt}
    qid_to_prompts = {
        row['qid']: {
            'neutral': row['neutral_prompt'],
            'consistent': row['consistent_prompt'],
            'inconsistent': row['inconsistent_prompt']
        } for _, row in prompts_df.iterrows()
    }

    prompt_types = ['neutral', 'consistent', 'inconsistent']
    attack_types = reranker_df[reranker_df['DocType'] == 'adversarial_GPT-4o']['AttackType'].dropna().unique()
    all_attack_types = reranker_df[reranker_df['DocType'] == 'adversarial_GPT-4o']['AttackType'].dropna().unique()


    # Prepare output files
    output_files = {pt: {} for pt in prompt_types}
    for pt in prompt_types:
        for atype in attack_types:
            fname = f"{pt}_helpful_{atype}_requests.jsonl"
            output_files[pt][atype] = open(os.path.join(output_dir, fname), "w")
        output_files[pt]['helpful_harmful'] = open(os.path.join(output_dir, f"{pt}_helpful_harmful_requests.jsonl"), "w")

    # Iterate QIDs
    for qid, group in reranker_df.groupby('QID'):
        # Support string keys like "qid_102"
        qid_key = f'qid_{qid}' if f'qid_{qid}' in qid_to_prompts else qid
        if qid_key not in qid_to_prompts:
            print(f"[SKIP] QID {qid} (key: {qid_key}): No prompts found.")
            continue
        prompts = qid_to_prompts[qid_key]

        # Select best segment per DocName (highest score)
        group['AttackType'] = group['AttackType'].fillna('None')

        # Group by DocName, DocType, and AttackType — take idxmax for Score in each group
        idx = group.groupby(['DocName', 'DocType', 'AttackType'])['Score'].idxmax()
        doc_best_segments = group.loc[idx]

        # Now split as usual, but ignore AttackType for helpful/harmful
        helpful_segments = doc_best_segments[(doc_best_segments['DocType'] == 'helpful')]
        harmful_segments = doc_best_segments[(doc_best_segments['DocType'] == 'harmful')]

        # For adversarial, AttackType is meaningful
        adv_gpt4o_segments = doc_best_segments[(doc_best_segments['DocType'] == 'adversarial_GPT-4o')]
        # or whatever processing you need

        # --- helpful + harmful for each prompt type ---
        if not helpful_segments.empty and not harmful_segments.empty:
            for pt in prompt_types:
                candidates = [
                    {
                        "docid": f"helpful_{s['DocName']}_{s['SegmentIndex']}",
                        "score": s['Score'],
                        "doc": {"contents": s['SegmentText']}
                    } for _, s in helpful_segments.iterrows()
                ] + [
                    {
                        "docid": f"harmful_{s['DocName']}_{s['SegmentIndex']}",
                        "score": s['Score'],
                        "doc": {"contents": s['SegmentText']}
                    } for _, s in harmful_segments.iterrows()
                ]
                request = {
                    "query": {
                        "text": prompts[pt],
                        "qid": f"{pt}_helpful_harmful_{qid}"
                    },
                    "candidates": candidates
                }
                output_files[pt]['helpful_harmful'].write(json.dumps(request) + "\n")

        # --- helpful + adversarial (per attack type) ---
        if not helpful_segments.empty and not adv_gpt4o_segments.empty:
            for atype in attack_types:
                attack_type_segments = adv_gpt4o_segments[adv_gpt4o_segments['AttackType'] == atype]
                if attack_type_segments.empty:
                    # print(f"[EMPTY] QID {qid} missing attack type: {atype}")
                    continue
                for pt in prompt_types:
                    candidates = [
                        {
                            "docid": f"helpful_{s['DocName']}_{s['SegmentIndex']}",
                            "score": s['Score'],
                            "doc": {"contents": s['SegmentText']}
                        } for _, s in helpful_segments.iterrows()
                    ]
                    # Add best segment per DocName for this attack type
                    for docname, doc_segs in attack_type_segments.groupby('DocName'):
                        best_adv_segment = doc_segs.loc[doc_segs['Score'].idxmax()]
                        candidates.append({
                            "docid": f"{best_adv_segment['DocName']}_{best_adv_segment['AttackType']}",
                            "score": best_adv_segment['Score'],
                            "doc": {"contents": best_adv_segment['SegmentText']}
                        })
                    request = {
                        "query": {
                            "text": prompts[pt],
                            "qid": f"{pt}_helpful_{atype}_{qid}"
                        },
                        "candidates": candidates
                    }
                    output_files[pt][atype].write(json.dumps(request) + "\n")

    # Close all files
    for d in output_files.values():
        for f in d.values():
            f.close()
    print(f"✅ Created request files in {output_dir}")



    # Print summary: for each attack type, how many unique QIDs have at least one adversarial_GPT-4o segment
    print("\nSummary: Unique QIDs per attack type (adversarial_GPT-4o)")
    adv_gpt4o = reranker_df[reranker_df['DocType'] == 'adversarial_GPT-4o']
    if 'AttackType' in adv_gpt4o.columns:
        for atype in adv_gpt4o['AttackType'].unique():
            qids = adv_gpt4o[adv_gpt4o['AttackType'] == atype]['QID'].unique()
            print(f"Attack type: {atype:30} | QIDs: {len(qids)} -> {sorted(qids)}")
    else:
        print("No AttackType column found in reranker CSV.")

    # Print summary: unique QIDs with at least one helpful segment
    print("\nSummary: Unique QIDs with at least one helpful segment")
    helpful = reranker_df[reranker_df['DocType'] == 'helpful']
    helpful_qids = sorted(helpful['QID'].unique())
    print(f"Helpful QIDs: {len(helpful_qids)} -> {helpful_qids}")


def create_biased_pool_requests_from_reranker_csv(reranker_csv_path, prompts_csv_path, output_dir):
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    reranker_df = pd.read_csv(reranker_csv_path)
    prompts_df = pd.read_csv(prompts_csv_path)
    print(f"Loaded {len(reranker_df)} segments from reranker CSV")
    print(f"Loaded {len(prompts_df)} prompts from prompts CSV")
    
    # Build mapping: QID -> {neutral/consistent/inconsistent: prompt}
    qid_to_prompts = {
        row['qid']: {
            'neutral': row['neutral_prompt'],
            'consistent': row['consistent_prompt'],
            'inconsistent': row['inconsistent_prompt']
        } for _, row in prompts_df.iterrows()
    }
    
    # Debug: Print available QIDs in prompts
    print(f"Available QIDs in prompts: {sorted([qid for qid in qid_to_prompts.keys()])}")
    print(f"Missing QIDs 120, 140, 144, 146 in prompts: {[qid for qid in [120, 140, 144, 146] if qid not in qid_to_prompts and f'qid_{qid}' not in qid_to_prompts]}")
    
    prompt_types = ['neutral', 'consistent', 'inconsistent']
    
    # Get all adversarial attack types (only from GPT-4o)
    adv_df = reranker_df[reranker_df['DocType'] == 'adversarial_GPT-4o']
    attack_types = adv_df['AttackType'].dropna().unique()
    print(f"Found attack types (GPT-4o only): {sorted(attack_types)}")
    
    # Debug: Check what other adversarial document types exist
    all_adv_types = reranker_df[reranker_df['DocType'].str.contains('adversarial', na=False)]['DocType'].unique()
    print(f"All adversarial document types in data: {sorted(all_adv_types)}")
    
    # Prepare output files for each prompt type and attack type
    output_files = {}
    for pt in prompt_types:
        output_files[pt] = {}
        # Files for helpful vs original harmful (no adversarial)
        output_files[pt]['helpful_vs_harmful'] = open(os.path.join(output_dir, f"{pt}_helpful_vs_harmful_requests.jsonl"), "w")
        output_files[pt]['harmful_vs_helpful'] = open(os.path.join(output_dir, f"{pt}_harmful_vs_helpful_requests.jsonl"), "w")
        
        # Separate files for each adversarial attack type
        for atype in attack_types:
            # Harmful-biased pool files (8 harmful + 2 helpful + adversarial)
            harmful_biased_fname = f"{pt}_harmful_biased_{atype}_requests.jsonl"
            output_files[pt][f"harmful_biased_{atype}"] = open(os.path.join(output_dir, harmful_biased_fname), "w")
            
            # Helpful-biased pool files (8 helpful + 2 harmful + adversarial)
            helpful_biased_fname = f"{pt}_helpful_biased_{atype}_requests.jsonl"
            output_files[pt][f"helpful_biased_{atype}"] = open(os.path.join(output_dir, helpful_biased_fname), "w")
    
    # Iterate through each QID
    for qid, group in reranker_df.groupby('QID'):
        # Debug: Print all QIDs being processed
        if qid in [120, 140, 144, 146]:
            print(f"[DEBUG] Processing QID {qid} with {len(group)} segments")
        
        # Support string keys like "qid_102"
        qid_key = f'qid_{qid}' if f'qid_{qid}' in qid_to_prompts else qid
        if qid_key not in qid_to_prompts:
            print(f"[SKIP] QID {qid} (key: {qid_key}): No prompts found.")
            continue
        
        prompts = qid_to_prompts[qid_key]
        
        # Select best segment per DocName (highest score)
        group['AttackType'] = group['AttackType'].fillna('None')
        
        # Group by DocName, DocType, and AttackType — take idxmax for Score in each group
        idx = group.groupby(['DocName', 'DocType', 'AttackType'])['Score'].idxmax()
        doc_best_segments = group.loc[idx]
        
        # Split by document type
        helpful_segments = doc_best_segments[doc_best_segments['DocType'] == 'helpful']
        harmful_segments = doc_best_segments[doc_best_segments['DocType'] == 'harmful']
        # Only consider GPT-4o adversarial documents, skip DeepSeek-R1-claude3.7
        adv_segments = doc_best_segments[doc_best_segments['DocType'] == 'adversarial_GPT-4o']
        
        # For biased pools: harmful can be either original harmful OR adversarial (not combined)
        # We'll handle this in the pool creation logic
        
        # First, create helpful vs original harmful pools (no adversarial)
        if len(helpful_segments) >= 2:
            if len(harmful_segments) >= 1:
                if len(helpful_segments) >= 8:
                    if len(harmful_segments) >= 8:
                        helpful_majority_count = 8
                        helpful_minority_count = 2
                        harmful_majority_count = 8
                        harmful_minority_count = 2
                        scaling = "8/2"
                    else:
                        helpful_majority_count = 8
                        helpful_minority_count = 2
                        harmful_majority_count = min(4, len(harmful_segments))
                        harmful_minority_count = min(1, len(harmful_segments))
                        scaling = "8/2"
                else:
                    if len(harmful_segments) >= 8:
                        helpful_majority_count = min(4, len(helpful_segments))
                        helpful_minority_count = min(1, len(helpful_segments))
                        harmful_majority_count = 8
                        harmful_minority_count = 2
                        scaling = "4/1"
                    else:
                        helpful_majority_count = min(4, len(helpful_segments))
                        helpful_minority_count = min(1, len(helpful_segments))
                        harmful_majority_count = min(4, len(harmful_segments))
                        harmful_minority_count = min(1, len(harmful_segments))
                        scaling = "4/1"
            else:
                print(f'QID:{qid}*********************************************')
                print(f"[SKIP] QID {qid}: Not enough harmful segments (need >=1, have {len(harmful_segments)})")
                continue
        else:
            print(f'QID:{qid}*********************************************')
            print(f"[SKIP] QID {qid}: Not enough helpful segments (need >=2, have {len(helpful_segments)})")
            continue
            
        if scaling != "8/2":
            print(f"[INFO] QID {qid} helpful_vs_harmful: Using {helpful_majority_count}/{harmful_minority_count} scaling (helpful: {len(helpful_segments)}, original_harmful: {len(harmful_segments)})")
        
        if  helpful_minority_count  >  harmful_majority_count :
             print(f"[SKIP] QID {qid}: helpful_minority_count  >  harmful_majority_count ")
             continue

        if  harmful_minority_count  >  helpful_majority_count :
             print(f"[SKIP] QID {qid}: harmful_minority_count  >  helpful_majority_count ")
             continue

        
        # Create helpful vs original harmful pools for each prompt type
        for pt in prompt_types:
            prompt_text = prompts[pt]
            
            # Helpful-biased pool: 8 helpful + 2 original harmful
            helpful_biased_candidates = []
            helpful_selected = helpful_segments.sample(n=helpful_majority_count, random_state=42)
            for _, segment in helpful_selected.iterrows():
                helpful_biased_candidates.append({
                    "docid": f"helpful_{segment['DocName']}_{segment['SegmentIndex']}",
                    "score": segment['Score'],
                    "doc": {"contents": segment['SegmentText']}
                })
            
            harmful_selected = harmful_segments.sample(n=harmful_minority_count, random_state=42)
            for _, segment in harmful_selected.iterrows():
                helpful_biased_candidates.append({
                    "docid": f"harmful_{segment['DocName']}_{segment['SegmentIndex']}",
                    "score": segment['Score'],
                    "doc": {"contents": segment['SegmentText']}
                })
            
            request = {
                "query": {
                    "text": prompt_text,
                    "qid": f"{pt}_helpful_vs_harmful_{qid}"
                },
                "candidates": helpful_biased_candidates
            }
            output_files[pt]['helpful_vs_harmful'].write(json.dumps(request) + "\n")
            
            # Harmful-biased pool: 8 harmful + 2 helpful
            harmful_biased_candidates = []
            harmful_selected = harmful_segments.sample(n=harmful_majority_count, random_state=42)
            for _, segment in harmful_selected.iterrows():
                harmful_biased_candidates.append({
                    "docid": f"harmful_{segment['DocName']}_{segment['SegmentIndex']}",
                    "score": segment['Score'],
                    "doc": {"contents": segment['SegmentText']}
                })
            
            helpful_selected = helpful_segments.sample(n=helpful_minority_count, random_state=42)
            for _, segment in helpful_selected.iterrows():
                harmful_biased_candidates.append({
                    "docid": f"helpful_{segment['DocName']}_{segment['SegmentIndex']}",
                    "score": segment['Score'],
                    "doc": {"contents": segment['SegmentText']}
                })
            
            request = {
                "query": {
                    "text": prompt_text,
                    "qid": f"{pt}_harmful_vs_helpful_{qid}"
                },
                "candidates": harmful_biased_candidates
            }
            output_files[pt]['harmful_vs_helpful'].write(json.dumps(request) + "\n")
        
        # Process each attack type
        for atype in attack_types:
            attack_type_segments = adv_segments[adv_segments['AttackType'] == atype]
            
            if attack_type_segments.empty:
                print(f"[EMPTY] QID {qid} missing attack type: {atype}")
                continue
            
            # Debug: Print segment counts for problematic QIDs
            if qid in [120, 140, 144, 146]:
                print(f"[DEBUG] QID {qid} attack {atype}: helpful={len(helpful_segments)}, harmful={len(harmful_segments)}, adv={len(attack_type_segments)}")
            
            # For each attack type: use ONLY that specific adversarial attack as "harmful"
            harmful_for_this_attack = attack_type_segments
            harmful_type = f"adversarial_{atype}"
            
            # Determine the scaling based on available segments
            if len(helpful_segments) >= 2 and len(harmful_for_this_attack) >= 2:
                if len(helpful_segments) >= 8 and len(harmful_for_this_attack) >= 8:
                    # Use 8/2 scaling
                    helpful_majority_count = 8
                    helpful_minority_count = 2
                    harmful_majority_count = 8
                    harmful_minority_count = 2
                    scaling = "8/2"
                else:
                    # Use 4/1 scaling
                    helpful_majority_count = min(4, len(helpful_segments))
                    helpful_minority_count = min(1, len(helpful_segments))
                    harmful_majority_count = min(4, len(harmful_for_this_attack))
                    harmful_minority_count = min(1, len(harmful_for_this_attack))
                    scaling = "4/1"
            else:
                print(f"[SKIP] QID {qid} attack {atype}: Not enough segments (helpful: {len(helpful_segments)}, harmful_for_attack: {len(harmful_for_this_attack)})")
                continue
            

            
            # Log only when not using 8/2 scaling
            if scaling != "8/2":
                print(f"[INFO] QID {qid} attack {atype}: Using {helpful_majority_count}/{harmful_minority_count} scaling (helpful: {len(helpful_segments)}, {harmful_type}: {len(harmful_for_this_attack)})")
            
            # Create biased pools for each prompt type
            for pt in prompt_types:
                prompt_text = prompts[pt]
                
                # 1. HARMFUL-BIASED POOL: harmful_majority_count harmful FIRST, then helpful_minority_count helpful
                harmful_biased_candidates = []
                
                # Add harmful segments FIRST (majority in harmful-biased pool) - RANDOM SELECTION
                harmful_selected = harmful_for_this_attack.sample(n=harmful_majority_count, random_state=42)
                for _, segment in harmful_selected.iterrows():
                    if segment['DocType'] == 'harmful':
                        docid = f"harmful_{segment['DocName']}_{segment['SegmentIndex']}"
                    else:  # adversarial
                        docid = f"adversarial_{segment['DocName']}_{segment['AttackType']}"
                    harmful_biased_candidates.append({
                        "docid": docid,
                        "score": segment['Score'],
                        "doc": {"contents": segment['SegmentText']}
                    })
                
                # Add helpful segments SECOND (minority in harmful-biased pool) - RANDOM SELECTION
                helpful_selected = helpful_segments.sample(n=helpful_minority_count, random_state=42)
                for _, segment in helpful_selected.iterrows():
                    harmful_biased_candidates.append({
                        "docid": f"helpful_{segment['DocName']}_{segment['SegmentIndex']}",
                        "score": segment['Score'],
                        "doc": {"contents": segment['SegmentText']}
                    })
                

                
                # Create harmful-biased request
                harmful_biased_request = {
                    "query": {
                        "text": prompt_text,
                        "qid": f"{pt}_harmful_biased_{atype}_{qid}"
                    },
                    "candidates": harmful_biased_candidates
                }
                output_files[pt][f"harmful_biased_{atype}"].write(json.dumps(harmful_biased_request) + "\n")
                
                # 2. HELPFUL-BIASED POOL: helpful_majority_count helpful FIRST, then harmful_minority_count harmful
                helpful_biased_candidates = []
                
                # Add helpful segments FIRST (majority in helpful-biased pool) - RANDOM SELECTION
                helpful_selected = helpful_segments.sample(n=helpful_majority_count, random_state=42)
                for _, segment in helpful_selected.iterrows():
                    helpful_biased_candidates.append({
                        "docid": f"helpful_{segment['DocName']}_{segment['SegmentIndex']}",
                        "score": segment['Score'],
                        "doc": {"contents": segment['SegmentText']}
                    })
                
                # Add harmful segments SECOND (minority in helpful-biased pool) - RANDOM SELECTION
                # Use the same harmful type as the biased harmful pool (original harmful OR adversarial)
                harmful_selected = harmful_for_this_attack.sample(n=harmful_minority_count, random_state=42)
                for _, segment in harmful_selected.iterrows():
                    if segment['DocType'] == 'harmful':
                        docid = f"harmful_{segment['DocName']}_{segment['SegmentIndex']}"
                    else:  # adversarial
                        docid = f"adversarial_{segment['DocName']}_{segment['AttackType']}"
                    helpful_biased_candidates.append({
                        "docid": docid,
                        "score": segment['Score'],
                        "doc": {"contents": segment['SegmentText']}
                    })
                
                # NO adversarial documents in helpful-biased pools (only helpful + harmful)
                
                # Create helpful-biased request
                helpful_biased_request = {
                    "query": {
                        "text": prompt_text,
                        "qid": f"{pt}_helpful_biased_{atype}_{qid}"
                    },
                    "candidates": helpful_biased_candidates
                }
                output_files[pt][f"helpful_biased_{atype}"].write(json.dumps(helpful_biased_request) + "\n")
    
    # Close all files
    for pt_dict in output_files.values():
        for f in pt_dict.values():
            f.close()
    
    print(f"✅ Created biased pool request files in {output_dir}")
    
    # Print summary statistics
    print("\nSummary: Biased pool creation statistics")
    for pt in prompt_types:
        for atype in attack_types:
            harmful_biased_fname = f"{pt}_harmful_biased_{atype}_requests.jsonl"
            helpful_biased_fname = f"{pt}_helpful_biased_{atype}_requests.jsonl"
            
            harmful_biased_path = os.path.join(output_dir, harmful_biased_fname)
            helpful_biased_path = os.path.join(output_dir, helpful_biased_fname)
            
            harmful_count = 0
            helpful_count = 0
            
            if os.path.exists(harmful_biased_path):
                with open(harmful_biased_path, 'r') as f:
                    harmful_count = sum(1 for _ in f)
            
            if os.path.exists(helpful_biased_path):
                with open(helpful_biased_path, 'r') as f:
                    helpful_count = sum(1 for _ in f)
            
            print(f"  {pt} + {atype}: {harmful_count} harmful-biased, {helpful_count} helpful-biased pools")


def create_biased_pool_requests_from_reranker_csv_separated(
    reranker_csv_path,
    prompts_csv_path,
    output_dir
):
   
    os.makedirs(output_dir, exist_ok=True)

    reranker_df = pd.read_csv(reranker_csv_path)
    prompts_df = pd.read_csv(prompts_csv_path)
    print(f"Loaded {len(reranker_df)} segments from reranker CSV")
    print(f"Loaded {len(prompts_df)} prompts from prompts CSV")

    # QID → prompts mapping
    qid_to_prompts = {
        row['qid']: {
            'neutral': row['neutral_prompt'],
            'consistent': row['consistent_prompt'],
            'inconsistent': row['inconsistent_prompt'],
        }
        for _, row in prompts_df.iterrows()
    }

    prompt_types = ['neutral', 'consistent', 'inconsistent']

    # Take all AttackType values present in CSV (already filtered upstream)
    attack_types = reranker_df['AttackType'].dropna().unique()
    print(f"Found attack types in CSV: {sorted(attack_types)}")

    # Prepare output files
    output_files = {}
    for pt in prompt_types:
        output_files[pt] = {}
        output_files[pt]['helpful_vs_harmful'] = open(
            os.path.join(output_dir, f"{pt}_helpful_vs_harmful_requests.jsonl"), "w"
        )
        output_files[pt]['harmful_vs_helpful'] = open(
            os.path.join(output_dir, f"{pt}_harmful_vs_helpful_requests.jsonl"), "w"
        )
        for atype in attack_types:
            output_files[pt][f"harmful_biased_{atype}"] = open(
                os.path.join(output_dir, f"{pt}_harmful_biased_{atype}_requests.jsonl"), "w"
            )
            output_files[pt][f"helpful_biased_{atype}"] = open(
                os.path.join(output_dir, f"{pt}_helpful_biased_{atype}_requests.jsonl"), "w"
            )

    def require_8_2(n_major, n_minor):
        """Only allow pool if we can make 8/2."""
        return (8, 2) if (n_major >= 8 and n_minor >= 2) else (0, 0)

    # Iterate per QID
    for qid, group in reranker_df.groupby('QID'):
        # allow prompts keyed by 'qid_{qid}' or raw qid
        qid_key = f'qid_{qid}' if f'qid_{qid}' in qid_to_prompts else qid
        if qid_key not in qid_to_prompts:
            print(f"**[SKIP PROMPTS]** QID {qid}: No prompts found (tried keys: {qid} / qid_{qid})")
            continue

        prompts = qid_to_prompts[qid_key]

        # Per-DocType pools (no per-doc dedup)
        helpful_segments = group[group['DocType'] == 'helpful'].copy()
        harmful_segments = group[group['DocType'] == 'harmful'].copy()

        # Helpful-biased (original): 8 helpful + 2 harmful
        hb_major, hb_minor = require_8_2(len(helpful_segments), len(harmful_segments))
        # Harmful-biased (original): 8 harmful + 2 helpful
        hm_major, hm_minor = require_8_2(len(harmful_segments), len(helpful_segments))

        # Sort deterministically
        helpful_sorted = helpful_segments.sort_values(
            by=["Score", "DocName", "SegmentIndex"] if "DocName" in helpful_segments.columns and "SegmentIndex" in helpful_segments.columns else ["Score"],
            ascending=False
        )
        harmful_sorted = harmful_segments.sort_values(
            by=["Score", "DocName", "SegmentIndex"] if "DocName" in harmful_segments.columns and "SegmentIndex" in harmful_segments.columns else ["Score"],
            ascending=False
        )

        for pt in prompt_types:
            prompt_text = prompts[pt]

            # Helpful-biased (original)
            if hb_major and hb_minor:
                helpful_sel = helpful_sorted.head(hb_major)
                harmful_sel = harmful_sorted.head(hb_minor)

                hb_candidates = []
                for _, s in helpful_sel.iterrows():
                    hb_candidates.append({
                        "docid": f"helpful_{s['DocName']}_{s['SegmentIndex']}",
                        "score": float(s['Score']),
                        "doc": {"contents": s['SegmentText']},
                    })
                for _, s in harmful_sel.iterrows():
                    hb_candidates.append({
                        "docid": f"harmful_{s['DocName']}_{s['SegmentIndex']}",
                        "score": float(s['Score']),
                        "doc": {"contents": s['SegmentText']},
                    })
                output_files[pt]['helpful_vs_harmful'].write(json.dumps({
                    "query": {"text": prompt_text, "qid": f"{pt}_helpful_vs_harmful_{qid}"},
                    "candidates": hb_candidates
                }) + "\n")
            else:
                print(f"**[A-SKIP]** QID {qid} helpful_vs_harmful lacks 8/2 "
                      f"(helpful={len(helpful_segments)}, harmful={len(harmful_segments)})")

            # Harmful-biased (original)
            if hm_major and hm_minor:
                harmful_sel = harmful_sorted.head(hm_major)
                helpful_sel = helpful_sorted.head(hm_minor)

                hm_candidates = []
                for _, s in harmful_sel.iterrows():
                    hm_candidates.append({
                        "docid": f"harmful_{s['DocName']}_{s['SegmentIndex']}",
                        "score": float(s['Score']),
                        "doc": {"contents": s['SegmentText']},
                    })
                for _, s in helpful_sel.iterrows():
                    hm_candidates.append({
                        "docid": f"helpful_{s['DocName']}_{s['SegmentIndex']}",
                        "score": float(s['Score']),
                        "doc": {"contents": s['SegmentText']},
                    })
                output_files[pt]['harmful_vs_helpful'].write(json.dumps({
                    "query": {"text": prompt_text, "qid": f"{pt}_harmful_vs_helpful_{qid}"},
                    "candidates": hm_candidates
                }) + "\n")
            else:
                print(f"**[A-SKIP]** QID {qid} harmful_vs_helpful lacks 8/2 "
                      f"(harmful={len(harmful_segments)}, helpful={len(helpful_segments)})")

        # Per-attack pools
        for atype in attack_types:
            attack_df = group[group['AttackType'] == atype]
            if attack_df.empty:
                print(f"**[B-SKIP]** QID {qid} attack {atype}: no segments")
                continue

            adv_sorted = attack_df.sort_values(
                by=["Score", "DocName", "SegmentIndex"] if "DocName" in attack_df.columns and "SegmentIndex" in attack_df.columns else ["Score"],
                ascending=False
            )

            # Harmful-biased (adv majority): 8 adv + 2 helpful
            b_hm_major, b_hm_minor = require_8_2(len(adv_sorted), len(helpful_sorted))
            # Helpful-biased (helpful majority): 8 helpful + 2 adv
            b_hb_major, b_hb_minor = require_8_2(len(helpful_sorted), len(adv_sorted))

            for pt in prompt_types:
                prompt_text = prompts[pt]

                # Harmful-biased
                if b_hm_major and b_hm_minor:
                    adv_sel = adv_sorted.head(b_hm_major)
                    helpful_sel = helpful_sorted.head(b_hm_minor)

                    hm_candidates = []
                    for _, s in adv_sel.iterrows():
                        hm_candidates.append({
                            "docid": f"adversarial_{s['DocName']}_{s['AttackType']}",
                            "score": float(s['Score']),
                            "doc": {"contents": s['SegmentText']},
                        })
                    for _, s in helpful_sel.iterrows():
                        hm_candidates.append({
                            "docid": f"helpful_{s['DocName']}_{s['SegmentIndex']}",
                            "score": float(s['Score']),
                            "doc": {"contents": s['SegmentText']},
                        })
                    output_files[pt][f"harmful_biased_{atype}"].write(json.dumps({
                        "query": {"text": prompt_text, "qid": f"{pt}_harmful_biased_{atype}_{qid}"},
                        "candidates": hm_candidates
                    }) + "\n")
                else:
                    print(f"**[B-SKIP]** QID {qid} attack {atype} harmful-biased lacks 8/2 "
                          f"(adv={len(adv_sorted)}, helpful={len(helpful_sorted)})")

                # Helpful-biased
                if b_hb_major and b_hb_minor:
                    helpful_sel = helpful_sorted.head(b_hb_major)
                    adv_sel = adv_sorted.head(b_hb_minor)

                    hb_candidates = []
                    for _, s in helpful_sel.iterrows():
                        hb_candidates.append({
                            "docid": f"helpful_{s['DocName']}_{s['SegmentIndex']}",
                            "score": float(s['Score']),
                            "doc": {"contents": s['SegmentText']},
                        })
                    for _, s in adv_sel.iterrows():
                        hb_candidates.append({
                            "docid": f"adversarial_{s['DocName']}_{s['AttackType']}",
                            "score": float(s['Score']),
                            "doc": {"contents": s['SegmentText']},
                        })
                    output_files[pt][f"helpful_biased_{atype}"].write(json.dumps({
                        "query": {"text": prompt_text, "qid": f"{pt}_helpful_biased_{atype}_{qid}"},
                        "candidates": hb_candidates
                    }) + "\n")
                else:
                    print(f"**[B-SKIP]** QID {qid} attack {atype} helpful-biased lacks 8/2 "
                          f"(helpful={len(helpful_sorted)}, adv={len(adv_sorted)})")

    # Close files
    for pt_dict in output_files.values():
        for f in pt_dict.values():
            f.close()

    print(f"✅ Created biased pool request files in {output_dir}")


def _slugify(x) -> str:
    # keep letters, numbers, dot, underscore, hyphen; replace others with "_"
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(x))

def create_ragnarok_requests_topk(
    reranker_csv_path: str,
    prompts_csv_path: str,
    output_dir: str,
    *,
    top_k: int = 10,
    file_prefix: str = ""  # optional: e.g., "trec2021_monoT5"
):
    """
    Build Ragnarok request JSONL files (neutral / consistent / inconsistent) from a
    GLOBAL reranker CSV, using the top_k segments per QID.
    The candidate docid includes DocType + AttackType + DocName + SegmentIndex.

    Expected CSV columns:
      QID, Score, SegmentText, DocName, SegmentIndex
    Optional (used if present):
      DocType, AttackType

    Prompts CSV must have columns:
      qid, neutral_prompt, consistent_prompt, inconsistent_prompt
      ('qid' can be '102' or 'qid_102')
    """

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(reranker_csv_path)
    prompts_df = pd.read_csv(prompts_csv_path)

    # Build prompt lookup that accepts both "102" and "qid_102"
    qid_to_prompts = {}
    for _, row in prompts_df.iterrows():
        raw_qid = str(row["qid"]).strip()
        norm_ids = {raw_qid}
        if raw_qid.startswith("qid_"):
            norm_ids.add(raw_qid.split("qid_", 1)[1])
        else:
            norm_ids.add(f"qid_{raw_qid}")
        for q in norm_ids:
            qid_to_prompts[q] = {
                "neutral": row["neutral_prompt"],
                "consistent": row["consistent_prompt"],
                "inconsistent": row["inconsistent_prompt"],
            }

    # Required columns
    required_cols = {"QID", "Score", "SegmentText", "DocName", "SegmentIndex"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"reranker CSV missing required columns: {missing}")

    # Normalize optional columns
    if "DocType" not in df.columns:
        df["DocType"] = "doc"
    if "AttackType" not in df.columns:
        df["AttackType"] = "None"
    df["AttackType"] = df["AttackType"].fillna("None")

    # Sort and take top_k per QID
    df_sorted = df.sort_values(["QID", "Score"], ascending=[True, False])
    topk = df_sorted.groupby("QID", as_index=False).head(top_k).copy()

    # Prepare 3 output files
    prefix = f"{file_prefix}_" if file_prefix else ""
    outs = {
        "neutral": open(os.path.join(output_dir, f"{prefix}neutral_requests.jsonl"), "w"),
        "consistent": open(os.path.join(output_dir, f"{prefix}consistent_requests.jsonl"), "w"),
        "inconsistent": open(os.path.join(output_dir, f"{prefix}inconsistent_requests.jsonl"), "w"),
    }

    num_written = {k: 0 for k in outs}
    for qid, group in topk.groupby("QID"):
        qid_str = str(qid)

        # Find prompts (support both '102' and 'qid_102')
        prompt_key = qid_str if qid_str in qid_to_prompts else f"qid_{qid_str}"
        if prompt_key not in qid_to_prompts:
            alt_key = qid_str.split("qid_", 1)[1] if qid_str.startswith("qid_") else None
            if alt_key and alt_key in qid_to_prompts:
                prompt_key = alt_key
            else:
                continue  # skip if no prompts
        prompts = qid_to_prompts[prompt_key]

        # Build candidates from all top_k segments for this QID
        candidates = []
        for _, row in group.iterrows():
            dtype = _slugify(row["DocType"])
            atype = _slugify(row["AttackType"])
            dname = _slugify(row["DocName"])
            try:
                sidx = int(row["SegmentIndex"])
            except Exception:
                sidx = _slugify(row["SegmentIndex"])
            docid = f"{dtype}_{atype}_{dname}_{sidx}"

            candidates.append({
                "docid": docid,
                "score": float(row["Score"]),
                "doc": {"contents": row["SegmentText"]},
            })

        # Emit 3 requests (one per prompt type)
        for pt in ("neutral", "consistent", "inconsistent"):
            req = {
                "query": {"text": prompts[pt], "qid": f"{pt}_{qid_str}"},
                "candidates": candidates,
            }
            outs[pt].write(json.dumps(req) + "\n")
            num_written[pt] += 1

    for f in outs.values():
        f.close()

    print(
        f"✅ Wrote requests: neutral={num_written['neutral']}, "
        f"consistent={num_written['consistent']}, "
        f"inconsistent={num_written['inconsistent']} → {output_dir}"
    )


    

    
