# Ragnarok (generation backend)

Adapted Ragnarok-style retrieve-and-generate stack used to produce the
single-document / paired / pooling responses released under `Experiment_Results/`.

## Layout

| Path | Role |
|------|------|
| `generate/` | LLM wrappers (OpenAI, Claude, open-source), RAG generator, prompt templates |
| `data.py` | Request/result I/O |
| `retrieve_and_generate.py` | High-level orchestration |
| `scripts/` | Batch export/import and a minimal `run_my_rag.py` entrypoint |

Prompt text used in RAG settings is also mirrored under [`../../Prompts/`](../../Prompts/).

## Quick example

```bash
cd Code/ragnarok
# from repo root: cp .env.example .env  and set OPENAI_API_KEY
python scripts/run_my_rag.py --requests /path/to/requests.jsonl --run-id demo
```

Released stance-labeled CSVs (no need to regenerate for table reproduction):

- `Experiment_Results/TREC2020|TREC2021/Single_Document_*`
- `Experiment_Results/TREC2020|TREC2021/Paired_Document_*`
- `Experiment_Results/TREC2020|TREC2021/Biased_Controlled_Pooling_*`
- `Experiment_Results/TREC2020|TREC2021/Realistic_Pooling_*`

For the manuscript **mitigation** defense, see [`../reliabilityrag_mis/`](../reliabilityrag_mis/).
