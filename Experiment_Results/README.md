# Experiment Results

Organized by TREC year (`TREC2020/`, `TREC2021/`) and setting.

## Naming

| Folder prefix | Meaning |
|---------------|---------|
| `Single_Document_<Model>/` | One document as context |
| `Passage_Based_<Model>/` | Two passages (helpful ↔ adversarial) |
| `Biased_Pool_<Model>/` | Biased retrieval pools (harmful-biased / helpful-biased) |
| `ReliabilityRAG_MIS/` | Manuscript mitigation (Baseline vs MIS summaries) |

### Models for `Passage_Based_*` and `Biased_Pool_*`

- `GPT_4.1`
- `GPT_5`
- `Phi_4`
- `Llama-3.3-70B-Instruct`
- `Qwen3-30B-A3B-Instruct-2507`

### Judges (subfolders)

- `gemini2.0flash/` / `gemini2.5flash/` — Gemini stance labels
- `gpt4omini/` — GPT-4o-mini stance labels

(Availability of a given judge depends on when that model was evaluated.)

## Mitigation (manuscript)

`TREC2020|TREC2021/ReliabilityRAG_MIS/<condition>/gemini_eval/mis_vs_baseline_comparison.csv`

See also `../Tables/harmful_baseline_vs_mis.csv`.
