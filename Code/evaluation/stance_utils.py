"""Shared stance-evaluation helpers for paper reproduction."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STANCE_PROMPT_PATH = REPO_ROOT / "Prompts" / "Stance_Classification_Prompt.txt"


def normalize_stance(text) -> str | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    t = str(text).strip().lower()
    if t in ("", "nan", "none"):
        return None
    m = re.search(r"\b(helpful|unhelpful)\b", t)
    if m:
        return m.group(1)
    tok = t.split()[0] if t else None
    return tok if tok in ("helpful", "unhelpful") else None


def load_stance_prompt(path: Path | None = None) -> str:
    p = path or DEFAULT_STANCE_PROMPT_PATH
    return p.read_text(encoding="utf-8")


def format_stance_prompt(
    query: str,
    description: str,
    response: str,
    template: str | None = None,
) -> str:
    tmpl = template if template is not None else load_stance_prompt()
    return tmpl.format(query=query, description=description, response=response)


def load_trec_prompts(csv_path: Path) -> dict:
    """Map qid -> {description, gt_stance} from prompts_*_v2.csv."""
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in df.iterrows():
        ans = str(r["answer"]).strip().lower()
        gt = {"yes": "helpful", "no": "unhelpful"}.get(ans, ans)
        out[str(r["qid"])] = {
            "description": str(r.get("description", "")),
            "gt_stance": gt,
            "title": str(r.get("title", "")),
        }
    return out


def alignment_rate(series) -> float:
    s = series.dropna().astype(bool)
    return float(s.mean()) if len(s) else float("nan")
