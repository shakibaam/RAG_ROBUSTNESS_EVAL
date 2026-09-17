"""Core ReliabilityRAG MIS algorithm (exact MIS + DeBERTa NLI conflict graph)."""

from __future__ import annotations

import random
from itertools import chain, combinations
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_MIS = REPO_ROOT / "Prompts" / "mis"

NLI_MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
CONTRADICTION_THRESHOLD = 0.5
ERR = 0.0


def load_isolate_template() -> str:
    return (PROMPTS_MIS / "Isolate_Prompt.txt").read_text(encoding="utf-8")


def doc_type(doc_id: str) -> str:
    if str(doc_id).startswith("adversarial_doc"):
        return "adversarial"
    if str(doc_id).startswith("helpful_doc"):
        return "helpful"
    return "unknown"


def is_independent(subset, graph) -> bool:
    for v in subset:
        for u in subset:
            if u != v and u in graph[v]:
                return False
    return True


def max_independent_set(graph, vertices):
    best_size = 0
    best_sets = []
    vertices_list = list(vertices)
    for subset in chain.from_iterable(
        combinations(vertices_list, r) for r in range(len(vertices_list) + 1)
    ):
        subset = set(subset)
        if is_independent(subset, graph):
            subset_size = len(subset)
            if subset_size > best_size:
                best_size = subset_size
                best_sets = [tuple(sorted(subset))]
            elif subset_size == best_size:
                best_sets.append(tuple(sorted(subset)))
    if best_sets:
        return min(best_sets)
    return tuple()


def build_conflict_graph(
    query: str,
    answers: list[str],
    nli_tok,
    nli_model,
    device,
    contradiction_threshold: float = CONTRADICTION_THRESHOLD,
    err: float = ERR,
    seed: int = 0,
):
    k = len(answers)
    graph = {i: set() for i in range(k)}
    nli_pairs = []
    premises, hypotheses, pair_indices = [], [], []
    for i in range(k):
        for j in range(i + 1, k):
            premises.append(f"The answer to the question: {query}\nis {answers[i]}.")
            hypotheses.append(f"The answer to the question: {query}\nis {answers[j]}.")
            pair_indices.append((i, j))

    inputs = nli_tok(
        premises, hypotheses, return_tensors="pt", truncation=True, padding=True
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)

    random.seed(seed)
    for idx, (i, j) in enumerate(pair_indices):
        p_ent = float(probs[idx][0].item())
        p_neu = float(probs[idx][1].item())
        p_con = float(probs[idx][2].item())
        x = random.random()
        edge = False
        if ("I don't know" not in answers[i]) and ("I don't know" not in answers[j]):
            if p_con >= contradiction_threshold:
                if x >= err:
                    graph[i].add(j)
                    graph[j].add(i)
                    edge = True
            else:
                if x <= err:
                    graph[i].add(j)
                    graph[j].add(i)
                    edge = True
        nli_pairs.append(
            {
                "i": i,
                "j": j,
                "p_entailment": p_ent,
                "p_neutral": p_neu,
                "p_contradiction": p_con,
                "random_x": x,
                "edge": edge,
            }
        )
    return graph, nli_pairs


def select_mis(graph, answers: list[str]):
    k = len(answers)
    z = {i for i in range(k) if "I don't know" not in answers[i]}
    best = max_independent_set(graph, z)
    if not best:
        best_set = list(z) if z else list(range(k))
    else:
        best_set = list(best)
    best_set.sort()
    return best_set, sorted(z)


def build_final_messages(query: str, selected_docs: list[str]) -> list[dict]:
    system = (
        "This is a chat between a user and an artificial intelligence assistant. "
        "The assistant gives helpful and detailed answers to the user's question based on the context references. "
        "The assistant should also indicate when the answer cannot be found in the context references."
    )
    instruction = (
        "Please give a full and complete answer for the question. "
        "Cite each context document inline that supports your answer within brackets [] using the IEEE format. "
        "Ensure each sentence is properly cited."
    )
    numbered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(selected_docs, 1))
    user = (
        f"Instruction: {instruction}\n\n"
        f"Documents: {numbered}\n\n"
        f"Query: {query}\n\n"
        f"Instruction: {instruction}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
