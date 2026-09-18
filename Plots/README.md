# Plots

## Judges

- `gemini/` — stance labels from Gemini (2.0 or 2.5 Flash, depending on the model run)
- `gpt4omini/` — stance labels from GPT-4o-mini

## Settings

| Folder | Contents |
|--------|----------|
| `Biased_Pool/` | Helpful-biased vs harmful-biased alignment by attack (TREC 2020 & 2021) |
| `Passage_Based/` | Helpful-first vs helpful-second passage pairs by attack |
| `Single-Document/` | Legacy single-document stacked plots (existing) |

### Models (Biased_Pool + Passage_Based)

`GPT_4.1`, `GPT_5`, `Phi_4`, `Llama-3.3-70B-Instruct`, `Qwen3-30B-A3B-Instruct-2507`

Filename pattern: `{biased_pool|passage_based}_<Model>_2020_2021.png`

Regenerate from curated CSVs:

```bash
python Code/evaluation/plot_biased_and_passage.py
```

## Manuscript set

`manuscript/` holds additional paper-facing figure copies (pooling / paired / single-doc stacks).
