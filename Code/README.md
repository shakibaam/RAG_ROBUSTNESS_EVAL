# Code

Paper reproduction code for the health-domain RAG robustness evaluation.

| Subfolder | Purpose |
|-----------|---------|
| [`ragnarok/`](ragnarok/) | Generation backend (Ragnarok-style RAG) used for released experiment CSVs |
| [`evaluation/`](evaluation/) | Stance helpers and optional paired stats |
| [`reliabilityrag_mis/`](reliabilityrag_mis/) | **Manuscript mitigation**: ReliabilityRAG MIS run + Gemini eval |

Released results under `Experiment_Results/` are sufficient to reproduce manuscript tables without re-running LLMs.
