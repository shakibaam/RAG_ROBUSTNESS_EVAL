# Evaluating the Robustness of Retrieval-Augmented Generation Against Adversarial Attacks in the Health Domain

Curated experimental code, prompts, results, tables, and figures for evaluating how retrieval-augmented generation (RAG) systems behave under adversarial and misleading health information on **TREC Misinformation 2020/2021**, plus released **Check-COVID** verification results.

This repository is intended for **paper reproduction**: released CSVs/summaries, paper figures, prompt templates, evaluation utilities, and the **ReliabilityRAG (MIS)** mitigation pipeline.

## What is included

| Path | Contents |
|------|----------|
| [`Prompts/`](Prompts/) | RAG / non-RAG generation templates, stance judge prompt, MIS isolate & final prompts |
| [`Code/`](Code/) | Ragnarok generation backend, evaluation helpers, and ReliabilityRAG MIS runners |
| [`Experiment_Results/`](Experiment_Results/) | Stance-labeled CSVs for TREC + Check-COVID settings |
| [`Tables/`](Tables/) | Compact paper tables (e.g., harmful Baseline vs MIS) |
| [`Plots/`](Plots/) | Released figures (Pools / Paired / Single-Document), organized by judge |
| [`Data/prompts_trec/`](Data/prompts_trec/) | TREC query/prompt metadata CSVs |

## What is not in the main reproduction pipeline

- **RobustRAG** KeywordAgg experiments are **not** required to reproduce the mitigation results. The reported defense is **ReliabilityRAG MIS**.
- Plot-generation scripts are **not** included; figures under [`Plots/`](Plots/) are released as images.
- Raw per-query MIS instance JSON dumps, API keys, and large private corpora are excluded (see `.gitignore`).

## Experimental settings (overview)

Generation uses the **Ragnarok**-style RAG stack in [`Code/ragnarok/`](Code/ragnarok/). Released settings include:

1. **Single-document** (`Single_Document_<Model>/`) — one retrieved/adversarial/helpful document as context.
2. **Paired documents** (`Passage_Based_<Model>/`) — helpful + adversarial passages (helpful-first vs helpful-second).
3. **Natural / passage-based pooling** (`Natural_Pool_<Model>/`) — realistic top-k retrieval pools.
4. **Biased pool** (`Biased_Pool_<Model>/`) — retrieval pools skewed toward harmful or helpful evidence (e.g., 8:2).
5. **Mitigation** — **ReliabilityRAG MIS** (isolate → DeBERTa NLI conflict graph → exact maximum independent set → final answer on selected docs).

**Additional benchmark (Check-COVID):** released verification results under [`Experiment_Results/Check_Covid/`](Experiment_Results/Check_Covid/) (single-document + paired-document). Full code for that benchmark lives in [A-Structured-Evaluation-of-LLM-Verification-Robustness](https://github.com/shakibaam/A-Structured-Evaluation-of-LLM-Verification-Robustness).

Each TREC generation setting is typically evaluated under **consistent / inconsistent / neutral** query tones. Stance labels use Gemini and/or GPT-4o-mini judges (`helpful` / `unhelpful`).

## Repository layout

```text
RAG_ROBUSTNESS_EVAL/
  README.md
  requirements.txt
  .gitignore
  Prompts/
  Code/
    ragnarok/             # Ragnarok-style RAG generation backend
    evaluation/           # stance helpers, optional paired stats
    reliabilityrag_mis/   # MIS run + Gemini eval for TREC 2020/2021
  Experiment_Results/
    TREC2020/ … TREC2021/
    Check_Covid/          # Check-COVID single + paired results
  Tables/
  Plots/
    gemini2.0flash/
    gemini2.5flash/
    gpt4omini/
  Data/prompts_trec/
```

### Experiment results layout

Under `Experiment_Results/TREC2020|TREC2021/` (see also [`Experiment_Results/README.md`](Experiment_Results/README.md)):

| Folder prefix | Meaning |
|---------------|---------|
| `Single_Document_<Model>/` | One-document settings |
| `Passage_Based_<Model>/` | Paired helpful ↔ adversarial passages |
| `Natural_Pool_<Model>/` | Realistic / passage-based pooling |
| `Biased_Pool_<Model>/` | Harmful-biased / helpful-biased pools |
| `ReliabilityRAG_MIS/` | Baseline vs MIS comparison CSVs |
| `Check_Covid/` | Check-COVID single-document + paired-document results |

**Models** for TREC `Passage_Based_*`, `Natural_Pool_*`, and `Biased_Pool_*`:

- `GPT_4.1`, `GPT_5`, `Phi_4`, `Llama-3.3-70B-Instruct`, `Qwen3-30B-A3B-Instruct-2507`

**Models** for Check-COVID (`Experiment_Results/Check_Covid/`):

- `deepseek`, `GPT4.1`, `GPT5`, `llama3.1`, `phi4`

**Judges** (TREC subfolders under each model setting; availability varies by model):

- `gemini2.0flash/` — Gemini 2.0 Flash (e.g., GPT-4.1 pools/paired; Single-Document)
- `gemini2.5flash/` — Gemini 2.5 Flash (other models’ pools/paired)
- `gpt4omini/` — GPT-4o-mini

### Plots layout

Under `Plots/` (see also [`Plots/README.md`](Plots/README.md)), organized by **judge**, then setting:

| Folder | Contents |
|--------|----------|
| `Pools/` | Merged natural-pool + biased-pool figures (`bias_pools_<Model>_2020_2021.png`) |
| `Paired/` | Paired-document figures (`paired_<Model>_2020_2021.png`) |
| `Single-Document/` | Single-document stacked figures |

Judge folders: `gemini2.0flash/`, `gemini2.5flash/`, `gpt4omini/`.

### CSV columns (generation + stance)

Typical columns include: `qid`, `tone`, `setting`, `query`, `description`, `llm_response`, `gt_stance`, `references`, `topic_id`, `doc_name`, `attack_type`, `pair_setting`, `predicted_stance` / `predicted_stance_gemini`.

## Quick start

```bash
git clone <REPO_URL> RAG_ROBUSTNESS_EVAL
cd RAG_ROBUSTNESS_EVAL
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
cp .env.example .env   # set GOOGLE_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY as needed
```

### Reproduce tables from released artifacts

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

The **mitigation** strategy is based on **ReliabilityRAG**:

> ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search

```bibtex
@article{shen2026reliabilityrag,
  title={Reliabilityrag: Effective and provably robust defense for rag-based web-search},
  author={Shen, Zeyu and Imana, Basileal and Wu, Tong and Xiang, Chong and Mittal, Prateek and Korolova, Aleksandra},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={45662--45702},
  year={2026}
}
```

Generation builds on **Ragnarök**:

> Ragnarök: A reusable RAG framework and baselines for TREC 2024 retrieval-augmented generation track

```bibtex
@inproceedings{Pradeep:2025:RRF,
  title={Ragnar{\"o}k: A reusable RAG framework and baselines for TREC 2024 retrieval-augmented generation track},
  author={Pradeep, Ronak and Thakur, Nandan and Sharifymoghaddam, Sahel and Zhang, Eric and Nguyen, Ryan and Campos, Daniel and Craswell, Nick and Lin, Jimmy},
  booktitle={European Conference on Information Retrieval},
  pages={132--148},
  year={2025},
  organization={Springer}
}
```
