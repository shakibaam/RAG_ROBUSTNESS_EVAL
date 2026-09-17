# ReliabilityRAG MIS (paper mitigation pipeline)

Exact Maximum Independent Set (MIS) defense adapted for TREC health misinformation Extreme pools.

## Algorithm

1. **Isolate** — answer the query from each retrieved document alone (`Prompts/mis/Isolate_Prompt.txt`).
2. **NLI graph** — DeBERTa-v3 MNLI pairwise contradiction edges between isolate answers (β=0.5, err=0).
3. **MIS** — exact maximum independent set over non-IDK vertices (lexicographic tie-break).
4. **Final answer** — generate on MIS-selected documents with IEEE citation prompt.

## Reproduce tables without regeneration

Use released CSVs under:

- `Experiment_Results/TREC2020/ReliabilityRAG_MIS/`
- `Experiment_Results/TREC2021/ReliabilityRAG_MIS/`
- `Tables/harmful_baseline_vs_mis.csv`

## Optional: regenerate

```bash
# from repo root
export OPENAI_API_KEY=...          # or OPENROUTER_API_KEY
export GOOGLE_API_KEY=...

python Code/reliabilityrag_mis/run_mis.py \
  --year 2021 \
  --condition all_harmful \
  --data-root /path/to/Extreme_Pool_gpt5_gemini2.5flash

python Code/reliabilityrag_mis/eval_mis_gemini.py \
  --year 2021 \
  --condition extreme_harmful_biased_liar \
  --mis-instances Experiment_Results/TREC2021/ReliabilityRAG_MIS/runs/full_run_extreme_harmful_biased_liar/instances \
  --baseline-csv /path/to/neutral_results/.../csv \
  --out-dir Experiment_Results/TREC2021/ReliabilityRAG_MIS/extreme_harmful_biased_liar
```

RobustRAG is **not** part of this pipeline.
