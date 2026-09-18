# Check-COVID results

Released Check-COVID experiment outputs (single-document and paired-document conditions).

**Source:** [A-Structured-Evaluation-of-LLM-Verification-Robustness](https://github.com/shakibaam/A-Structured-Evaluation-of-LLM-Verification-Robustness)  
**Paper:** *When Evidence Disagrees: A Structured Evaluation of LLM Verification Robustness*

This folder mirrors `check_covid/data/` from that repository (results only; no generation/bootstrap scripts).

## Layout

```text
Check_Covid/
  supported_claims_filtered.csv
  refuted_claims_filtered.csv
  generate_adversarial/
    support_to_refute/
    refute_to_support/
  single_document/
    <model>/          # deepseek | GPT4.1 | GPT5 | llama3.1 | phi4
      RAG_support/
      RAG_refute/
      RAG_support_adversarial/
      RAG_refute_adversarial/
      non_RAG_support/
      non_RAG_refute/
  pair_document/
    <model>_pair/     # DeepSeek_pair | GPT4.1_pair | GPT5_pair | llama3.1_pair | phi4_pair
      helpful_first_original_support/
      helpful_first_original_refute/
      helpful_second_original_support/
      helpful_second_original_refute/
```

Each condition leaf folder contains a `results.csv` (and often bootstrap summary text).

## Models

`deepseek`, `GPT4.1`, `GPT5`, `llama3.1`, `phi4`
