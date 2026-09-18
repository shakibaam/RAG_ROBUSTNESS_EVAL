# Experiment Results

Organized by TREC year (`TREC2020/`, `TREC2021/`) and setting.

## Naming

| Folder prefix | Meaning |
|---------------|---------|
| `Single_Document_<Model>/` | One document as context |
| `Natural_Pool_<Model>/` | Realistic / passage-based pooling |
| `Passage_Based_<Model>/` | Paired documents (helpful ↔ adversarial) |
| `Biased_Pool_<Model>/` | Biased retrieval pools (harmful-biased / helpful-biased) |
| `ReliabilityRAG_MIS/` | Mitigation summaries (Baseline vs MIS) |
| `Check_Covid/` | Check-COVID single-document + paired-document results |

See [`Check_Covid/README.md`](Check_Covid/README.md) for the Check-COVID layout (from [A-Structured-Evaluation-of-LLM-Verification-Robustness](https://github.com/shakibaam/A-Structured-Evaluation-of-LLM-Verification-Robustness)).

### Models for `Single_Document_*`

- `GPT_4.1`
- `GPT_5`
- `Phi_4`
- `Llama-3-8B-Instruct`
- `DeepSeek-R1-Distill-Qwen-32B`
- `Claudi_3.5_Haiku`

### Models for `Natural_Pool_*`, `Passage_Based_*`, and `Biased_Pool_*`

- `GPT_4.1`
- `GPT_5`
- `Phi_4`
- `Llama-3.3-70B-Instruct`
- `Qwen3-30B-A3B-Instruct-2507`

### Models for Check-COVID

- `deepseek`, `GPT4.1`, `GPT5`, `llama3.1`, `phi4`

### Judges (subfolders)

- `gemini2.0flash/` — Gemini 2.0 Flash
- `gemini2.5flash/` — Gemini 2.5 Flash
- `gpt4omini/` — GPT-4o-mini

(Availability of a given judge depends on when that model was evaluated.)

## Mitigation

`TREC2020|TREC2021/ReliabilityRAG_MIS/<condition>/gemini_eval/mis_vs_baseline_comparison.csv`
