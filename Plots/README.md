# Plots

Released figures only (no plot-generation code in this repo).

## Judges

- `gemini2.5flash/` — Gemini 2.5 Flash stance labels
- `gemini2.0flash/` — Gemini 2.0 Flash stance labels (GPT-4.1 pools/paired + Single-Document)
- `gpt4omini/` — GPT-4o-mini stance labels

## Settings

| Folder | Contents |
|--------|----------|
| `Pools/` | Merged Passage-Based Pooling + Biased helpful/harmful by attack |
| `Paired/` | Paired documents: Helpful First vs Helpful Second by attack |
| `Single-Document/` | Single-document stacked plots |

### Models (`Pools/` + `Paired/`)

`GPT_4.1`, `GPT_5`, `Phi_4`, `Llama-3.3-70B-Instruct`, `Qwen3-30B-A3B-Instruct-2507`

Filename patterns:

- `Pools/bias_pools_<Model>_2020_2021.png`
- `Paired/paired_<Model>_2020_2021.png`
