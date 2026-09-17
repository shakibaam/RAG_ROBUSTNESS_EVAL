# Prompts

Templates used for generation and evaluation in this project.

| File | Role |
|------|------|
| `Ragnarok_RAG_Settings_Prompt.txt` | Multi-document RAG answer (system + IEEE citations) |
| `Ragnarok_Non_RAG_Settings_Prompt.txt` | Non-RAG / constrained citation instruction block |
| `Stance_Classification_Prompt.txt` | Gemini / GPT stance judge (`helpful` / `unhelpful`) |
| `Stance_Classification_Prompt_.txt` | Same as above (legacy filename kept for compatibility) |
| `mis/Isolate_Prompt.txt` | ReliabilityRAG per-document isolate answer |
| `mis/MIS_Final_Answer_Prompt.txt` | Final answer over MIS-selected documents |

TREC query metadata (qid, description, ground-truth answer) lives in [`../Data/prompts_trec/`](../Data/prompts_trec/).
