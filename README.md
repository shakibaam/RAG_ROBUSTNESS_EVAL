# Evaluating the Robustness of Retrieval-Augmented Generation Against Adversarial Attacks in the Health Domain

Curated experimental code, prompts, results, and plots for evaluating how retrieval-augmented generation (RAG) systems behave under adversarial and misleading health information on **TREC Misinformation 2020** and **2021**.

This repository is intended for **paper reproduction**: released CSVs/summaries, manuscript figures, prompt templates, evaluation utilities, and the **ReliabilityRAG (MIS)** mitigation pipeline used in the final manuscript.

## What is included

| Path | Contents |
|------|----------|
| [`Prompts/`](Prompts/) | RAG / non-RAG generation templates, stance judge prompt, MIS isolate & final prompts |
| [`Code/`](Code/) | Evaluation helpers and ReliabilityRAG MIS runners (relative paths, env-based API keys) |
| [`Experiment_Results/`](Experiment_Results/) | Stance-labeled CSVs for single-doc, paired, pooling, extreme, and MIS summaries |
| [`Tables/`](Tables/) | Compact manuscript-facing tables (e.g., harmful Baseline vs MIS) |
| [`Plots/`](Plots/) | Figures used in analysis / manuscript |
| [`Data/prompts_trec/`](Data/prompts_trec/) | TREC query/prompt metadata CSVs |

## What is not in the main reproduction pipeline

- **RobustRAG** KeywordAgg experiments are **not** required to reproduce the manuscript mitigation results. The reported defense is **ReliabilityRAG MIS**.
- Raw per-query MIS instance JSON dumps, API keys, and large private corpora are excluded (see `.gitignore`).

## Experimental settings (overview)

1. **Single-document** — one retrieved/adversarial/helpful document as context.
2. **Paired-document** — helpful + adversarial pairs.
3. **Pooling** — biased-controlled vs realistic / extreme pools (e.g., 8:2 harmful-biased).
4. **Mitigation** — **ReliabilityRAG MIS** (isolate → DeBERTa NLI conflict graph → exact maximum independent set → final answer on selected docs).

Each generation setting is typically evaluated under **consistent / inconsistent / neutral** query tones. Stance labels use Gemini and/or GPT-4o-mini judges (`helpful` / `unhelpful`).

## Repository layout

```text
RAG_ROBUSTNESS_EVAL/
  README.md
  requirements.txt
  .gitignore
  Prompts/
  Code/
    evaluation/           # stance helpers, aggregation, paired stats utilities
    single_document/
    paired_document/
    pooling/
    reliabilityrag_mis/   # MIS run + Gemini eval for TREC 2020/2021
  Experiment_Results/
    TREC2020/ … TREC2021/
  Tables/
  Plots/
  Data/prompts_trec/
```

### Experiment results layout

Under `Experiment_Results/TREC2020|TREC2021/`:

- `Single_Document_<Model>/` with `gemini2.0flash/` and `gpt4omini/`
- `Paired_Document_GPT_4.1/`
- `Biased_Controlled_Pooling_GPT_4.1/`, `Realistic_Pooling_GPT_4.1/`
- `Extreme_Pooling_GPT_5/` — summary CSVs for extreme 8:2 pools
- `ReliabilityRAG_MIS/` — comparison CSVs and summaries (Baseline vs MIS)

### CSV columns (generation + stance)

Typical columns include: `qid`, `tone`, `setting`, `query`, `description`, `llm_response`, `gt_stance`, `references`, `topic_id`, `doc_name`, `attack_type`, `predicted_stance_<evaluator>`.

## Quick start

```bash
git clone https://github.com/shakibaam/RAG_ROBUSTNESS_EVAL.git
cd RAG_ROBUSTNESS_EVAL
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
cp .env.example .env   # set GOOGLE_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY as needed
```

### Reproduce manuscript tables from released artifacts

Harmful-biased Baseline vs ReliabilityRAG MIS rates are in:

- [`Tables/harmful_baseline_vs_mis.csv`](Tables/harmful_baseline_vs_mis.csv)
- [`Experiment_Results/TREC2020/ReliabilityRAG_MIS/`](Experiment_Results/TREC2020/ReliabilityRAG_MIS/)
- [`Experiment_Results/TREC2021/ReliabilityRAG_MIS/`](Experiment_Results/TREC2021/ReliabilityRAG_MIS/)

### Run ReliabilityRAG MIS (optional regeneration)

See [`Code/reliabilityrag_mis/README.md`](Code/reliabilityrag_mis/README.md). You need Extreme_Pool inputs and API access; released comparison CSVs are sufficient for table reproduction without re-running generation.

## Citation

If you use this repository, please cite:

```
[Citation information to be added upon publication]
```

## Acknowledgements

ReliabilityRAG builds on ideas from the ReliabilityRAG / RobustRAG literature. This health-domain evaluation adapts MIS for TREC misinformation pools; see `Code/reliabilityrag_mis/` for the paper pipeline used here.
