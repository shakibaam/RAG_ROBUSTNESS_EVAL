import json
import time
import os
from openai import OpenAI
from typing import Optional
from datetime import datetime
from google import genai
from google.genai import types
import requests
import anthropic



client = OpenAI()
METADATA_FILE = "batch_metadata.json"


def upload_batch_file(file_path: str) -> str:
    print(f"📤 Uploading batch file: {file_path}")
    batch_file = client.files.create(file=open(file_path, "rb"), purpose="batch")
    print(f"✅ Uploaded File ID: {batch_file.id}")
    return batch_file.id


def submit_gemini_batch_with_metadata(jsonl_path, metadata_path,
                                      model="gemini-2.0-flash",
                                      display_name="stance-eval-batch"):
    """
    Upload a JSONL file, submit a Gemini Batch job, and log metadata to a JSONL file.

    Args:
        jsonl_path (str): Path to the JSONL request file.
        metadata_path (str): Path to the JSONL file where metadata will be stored.
        model (str): Gemini model name, e.g., 'gemini-2.0-flash'.
        display_name (str): Display name for the batch job.
    """
    client = genai.Client() 


    uploaded = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl")
    )

  
    job = client.batches.create(
        model=model,
        src=uploaded.name, 
        config={"display_name": display_name},
    )


    metadata_entry = {
        "input_file": jsonl_path,
        "uploaded_file_id": uploaded.name,  # from Files API
        "batch_job_name": job.name,         # from Batches API
        "model": model,
        "display_name": display_name,
        "status": job.state.name if hasattr(job, "state") else None
    }


    with open(metadata_path, "a", encoding="utf-8") as meta_file:
        meta_file.write(json.dumps(metadata_entry, ensure_ascii=False) + "\n")

    print(f"✅ Batch submitted: {job.name}")
    print(f"📄 Metadata saved to: {metadata_path}")

    return job.name


def get_gemini_batch_status(job_name, api_key=None):
    """
    Check the status of a Gemini batch job and return its state + output file IDs.

    Args:
        job_name (str): e.g., "batches/123456789"
        api_key (str|None): API key or None to use GOOGLE_API_KEY env var.

    Returns:
        (state, output_ids) where output_ids is a list of file IDs or None
    """
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    job = client.batches.get(name=job_name)

    state = getattr(getattr(job, "state", None), "name", None)
    output_ids = []

    
    if getattr(job, "dest", None) and getattr(job.dest, "file_name", None):
        output_ids.append(job.dest.file_name)


    try:
        if getattr(job.output, "file", None):
            output_ids.append(job.output.file.name)
    except AttributeError:
        pass

    try:
        if getattr(job.output, "files", None):
            for f in job.output.files:
                output_ids.append(f.name)
    except AttributeError:
        pass

    if not output_ids:
        output_ids = None

    print(f"{state} {output_ids}")
    return state, output_ids


def download_from_metadata_gemini(metadata_path, api_key=None, job_index=-1, save_path=None):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_entries = [json.loads(line) for line in f]

    if not metadata_entries:
        print("❌ No metadata entries found.")
        return None

    entry = metadata_entries[job_index]
    job_name = entry.get("batch_job_name")
    if not job_name:
        print("❌ No batch_job_name found in metadata entry.")
        return None

    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    job = client.batches.get(name=job_name)

    state = getattr(getattr(job, "state", None), "name", None)
    if state != "JOB_STATE_SUCCEEDED":
        print(f"⏳ Job not finished. Current state: {state}")
        return None

    file_id = None
    if getattr(job, "dest", None) and getattr(job.dest, "file_name", None):
        file_id = job.dest.file_name

    if not file_id:
        print("⚠️ No output file found yet.")
        return None

    if not save_path:
        save_path = f"{job_name.replace('/', '_')}_results.jsonl"

    # ✅ New way: download returns bytes
    content = client.files.download(file=file_id)
    with open(save_path, "wb") as f:
        f.write(content)

    print(f"✅ Downloaded {file_id} → {save_path}")
    return save_path

def submit_claude_batch(requests_json_path: str, metadata_path: str):
    client = anthropic.Anthropic()

    requests_list = []
    with open(requests_json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            requests_list.append(json.loads(line))

    batch = client.messages.batches.create(requests=requests_list)

    metadata = {
        "batch_id": batch.id,
        "submit_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Submitted batch {batch.id}")
    print(f"📄 Metadata saved to {metadata_path}")
    return batch.id


def get_claude_batch_status(metadata_path: str):
    """
    Reads batch_id from metadata file and retrieves current status.
    """
    client = anthropic.Anthropic()

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    batch_id = metadata["batch_id"]
    status = client.messages.batches.retrieve(batch_id)
    print(status)
    print(f"📌 Batch {batch_id} status: {status.processing_status}")
    return status

def _asdict(x):
    return x.model_dump() if hasattr(x, "model_dump") else (
        x.dict() if hasattr(x, "dict") else x
    )


def _load_batch_id(metadata_path: str) -> str:
    # Try single-JSON first
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if "batch_id" in obj:
                return obj["batch_id"]
    except json.JSONDecodeError:
        pass


    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                o = json.loads(s)
            except Exception:
                continue
            if isinstance(o, dict) and "batch_id" in o:
                return o["batch_id"]

    raise ValueError(f"Could not find 'batch_id' in {metadata_path}")

def download_results(metadata_path: str = "batch_metadata.json",
                     out_jsonl: str = "claude_results.jsonl"):
    client = anthropic.Anthropic()
    batch_id = _load_batch_id(metadata_path)

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for entry in client.messages.batches.results(batch_id):
            out.write(json.dumps(_asdict(entry), ensure_ascii=False) + "\n")
    print("Saved:", out_jsonl)
# === Step 2: Submit batch job ===
def submit_batch_job(file_id: str, description: str = "batch eval job") -> str:
    print("🚀 Submitting batch job...")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    print(f"✅ Batch Job ID: {batch.id}")
    return batch.id


def save_metadata(file_id: str, batch_id: str, description: str, output_path: str):
    metadata = {
        "file_id": file_id,
        "batch_id": batch_id,
        "description": description,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📁 Saved batch metadata to {output_path}")


def load_metadata(metadata_file: str = METADATA_FILE):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata["batch_id"], metadata["file_id"], metadata.get("description", "")


def get_output_file_id(batch_id: str) -> str:
    batch = client.batches.retrieve(batch_id)
    print(f"📊 Batch status: {batch.status}")
    if batch.status == "completed":
        return batch.output_file_id
    elif batch.status in ["failed", "cancelled", "expired"]:
        raise RuntimeError(f"Batch failed (status: {batch.status})")
    else:
        raise RuntimeError(f"Batch not yet complete (status: {batch.status})")

def download_output_file(file_id: str, save_path: str):
    print(f"⬇️ Downloading output file: {file_id}")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_response = client.files.content(file_id)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(file_response.text)
    print(f"✅ Results saved to {save_path}")


def run_openai_batch_pipeline(input_jsonl: str, metadata_path, description: str = "OpenAI batch run"):
    file_id = upload_batch_file(input_jsonl)
    batch_id = submit_batch_job(file_id, description)
    save_metadata(file_id, batch_id, description,output_path=metadata_path)


def fetch_and_download_batch_output(save_path: str, metadata_file: str ):
    batch_id, _, description = load_metadata(metadata_file)
    print(f"🔍 Checking batch: {batch_id} | Description: {description}")
    output_file_id = get_output_file_id(batch_id)
    download_output_file(output_file_id, save_path)

def run_batch_pipeline_for_directory(input_dir: str, metadata_dir: str, description: str = "OpenAI batch run"):
    """
    For each .jsonl file in input_dir, submit a batch job and save metadata in metadata_dir with '_metadata' appended to the filename.
    """
    os.makedirs(metadata_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.endswith('.jsonl'):
            continue
        input_path = os.path.join(input_dir, fname)
        base, ext = os.path.splitext(fname)
        metadata_fname = f"{base}_metadata{ext}"
        metadata_path = os.path.join(metadata_dir, metadata_fname)
        print(f"Processing {input_path} -> {metadata_path}")
        run_openai_batch_pipeline(input_jsonl=input_path, metadata_path=metadata_path, description=description)


def run_batch_pipeline_for_directory_gemini(input_dir: str, metadata_dir: str):
    """
    For each .jsonl file in input_dir, submit a batch job and save metadata in metadata_dir with '_metadata' appended to the filename.
    """
    os.makedirs(metadata_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.endswith('.jsonl'):
            continue
        input_path = os.path.join(input_dir, fname)
        base, ext = os.path.splitext(fname)
        metadata_fname = f"{base}_metadata{ext}"
        metadata_path = os.path.join(metadata_dir, metadata_fname)
        print(f"Processing {input_path} -> {metadata_path}")
        submit_gemini_batch_with_metadata(
        jsonl_path=input_path, 
        metadata_path=metadata_path,
        model="gemini-2.0-flash",
        display_name="stance-eval-batch")


def check_batch_statuses_in_directory(metadata_dir: str):
    """
    Goes through all metadata files in metadata_dir and reports the status of each batch.
    """
    print(f"🔍 Checking batch statuses in: {metadata_dir}")
    print("-" * 80)
    
    completed_batches = []
    pending_batches = []
    failed_batches = []
    
    for fname in os.listdir(metadata_dir):
        if not fname.endswith('.jsonl'):
            continue
            
        metadata_path = os.path.join(metadata_dir, fname)
        try:
            batch_id, file_id, description = load_metadata(metadata_path)
            batch = client.batches.retrieve(batch_id)
            
            status_info = {
                'filename': fname,
                'batch_id': batch_id,
                'status': batch.status,
                'description': description
            }
            
            if batch.status == "completed":
                completed_batches.append(status_info)
            elif batch.status in ["failed", "cancelled", "expired"]:
                failed_batches.append(status_info)
            else:
                pending_batches.append(status_info)
                
        except Exception as e:
            print(f"❌ Error reading {fname}: {e}")
    
    # Report results
    print(f"✅ COMPLETED ({len(completed_batches)}):")
    for batch in completed_batches:
        print(f"  - {batch['filename']} | {batch['batch_id']} | {batch['description']}")
    
    print(f"\n⏳ PENDING ({len(pending_batches)}):")
    for batch in pending_batches:
        print(f"  - {batch['filename']} | {batch['batch_id']} | Status: {batch['status']} | {batch['description']}")
    
    print(f"\n❌ FAILED ({len(failed_batches)}):")
    for batch in failed_batches:
        print(f"  - {batch['filename']} | {batch['batch_id']} | Status: {batch['status']} | {batch['description']}")
    
    print(f"\n📊 SUMMARY: {len(completed_batches)} completed, {len(pending_batches)} pending, {len(failed_batches)} failed")
    return completed_batches, pending_batches, failed_batches

def download_all_completed_batches(metadata_dir: str, output_dir: str):
    """
    Goes through all metadata files in metadata_dir, downloads completed batch outputs, and saves them in output_dir.
    """
    print(f"⬇️ Downloading completed batches from: {metadata_dir}")
    print(f"💾 Saving to: {output_dir}")
    print("-" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    downloaded_count = 0
    skipped_count = 0
    
    for fname in os.listdir(metadata_dir):
        if not fname.endswith('.jsonl'):
            continue
            
        metadata_path = os.path.join(metadata_dir, fname)
        try:
            batch_id, file_id, description = load_metadata(metadata_path)
            batch = client.batches.retrieve(batch_id)
            
            if batch.status == "completed":
                # Create output filename based on metadata filename
                base, ext = os.path.splitext(fname)
                # Remove '_metadata' suffix if present
                if base.endswith('_metadata'):
                    base = base[:-9]  # Remove '_metadata'
                output_fname = f"{base}_output{ext}"
                output_path = os.path.join(output_dir, output_fname)
                
                print(f"📥 Downloading: {fname} -> {output_fname}")
                download_output_file(batch.output_file_id, output_path)
                downloaded_count += 1
                
            elif batch.status in ["failed", "cancelled", "expired"]:
                print(f"❌ Skipping failed batch: {fname} (status: {batch.status})")
                skipped_count += 1
            else:
                print(f"⏳ Skipping pending batch: {fname} (status: {batch.status})")
                skipped_count += 1
                
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
            skipped_count += 1
    
    print(f"\n📊 SUMMARY: {downloaded_count} downloaded, {skipped_count} skipped")
    return downloaded_count, skipped_count


def load_metadata_gemini(metadata_path):
    """
    Loads Gemini batch metadata from a JSONL file.
    Expected format: {"job_name": "...", "description": "..."}
    """
    with open(metadata_path, "r") as f:
        line = f.readline().strip()
        data = json.loads(line)
        job_name = data.get("job_name")
        description = data.get("description", "")
    return job_name, description


def download_all_completed_batches_gemini(metadata_dir: str, output_dir: str):
    """
    Scans `metadata_dir` for Gemini batch metadata .jsonl files, then uses
    `download_from_metadata_gemini` to download the final results for each,
    saving them into `output_dir`.

    Naming: turns `*_metadata.jsonl` -> `*_results.jsonl` (or appends `_results.jsonl`).
    """
    print(f"⬇️ Downloading completed Gemini batches from: {metadata_dir}")
    print(f"💾 Saving to: {output_dir}")
    print("-" * 80)

    os.makedirs(output_dir, exist_ok=True)
    downloaded_count = 0
    skipped_count = 0

    for fname in sorted(os.listdir(metadata_dir)):
        if not fname.endswith(".jsonl"):
            continue

        metadata_path = os.path.join(metadata_dir, fname)

        # Derive output filename
        base, ext = os.path.splitext(fname)
        if base.endswith("_metadata"):
            base = base[:-9]  # strip the '_metadata' suffix
        output_fname = f"{base}_results{ext}"
        output_path = os.path.join(output_dir, output_fname)

        try:
            print(f"📥 Downloading: {fname} -> {output_fname}")
            # Use your existing utility; job_index=-1 means "latest" job in the file
            download_from_metadata_gemini(
                metadata_path=metadata_path,
                job_index=-1,
                save_path=output_path
            )
            downloaded_count += 1
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
            skipped_count += 1

    print(f"\n📊 SUMMARY: {downloaded_count} downloaded, {skipped_count} skipped")
    return downloaded_count, skipped_count

