from typing import Any, Dict, List, Union

from ragnarok_inspired.src.data import Query, Request

# from ragnarok.evaluation.nugget_eval import EvalFunction
from ragnarok_inspired.src.generate.api_keys import get_azure_openai_args, get_openai_api_key
from ragnarok_inspired.src.generate.cohere import Cohere
from ragnarok_inspired.src.generate.generator import RAG
from ragnarok_inspired.src.generate.gpt import SafeOpenai
from ragnarok_inspired.src.generate.llm import PromptMode
from ragnarok_inspired.src.generate.os_llm import OSLLM



def run_rag_on_requests(
    requests: List[Request],
    generator_path: str,
    run_id: str = "my-experiment",
    context_size: int = 8192,
    max_output_tokens: int = 1500,
    topk: int = 20,
    shuffle_candidates: bool = False,
    logging: bool = True,
    save_results: bool = True,
):
    # Build agent
    if "gpt" in generator_path:
        print(f"Using OpenAI model: {generator_path}")
        openai_keys = get_openai_api_key()
        agent = SafeOpenai(
            model=generator_path,
            context_size=context_size,
            prompt_mode=PromptMode.CHATQA,
            max_output_tokens=max_output_tokens,
            num_few_shot_examples=0,
            keys=openai_keys,
        )
    elif "command-r" in generator_path:
        print(f"Using Cohere model: {generator_path}")
        agent = Cohere(
            model=generator_path,
            context_size=context_size,
            prompt_mode=PromptMode.COHERE,
            max_output_tokens=max_output_tokens,
            num_few_shot_examples=0,
        )
    elif "llama" in generator_path.lower() or "mistral" in generator_path.lower():
        print(f"Using OSLLM model: {generator_path}")
        agent = OSLLM(
            model=generator_path,
            context_size=context_size,
            prompt_mode=PromptMode.CHATQA,
            max_output_tokens=max_output_tokens,
            num_few_shot_examples=0,
            device="cuda",
            num_gpus=1,
        )
    else:
        raise ValueError(f"Unsupported model: {generator_path}")

    # Run RAG
    rag = RAG(agent=agent, run_id=run_id)
    print("Running RAG...")
    rag_results = rag.answer_batch(
        requests,
        topk=topk,
        shuffle_candidates=shuffle_candidates,
        logging=logging,
    )

    # Save results
    if save_results:
        rag.write_answer_results(
            retrieval_method_name="manual_retrieval",
            results=rag_results,
            shuffle_candidates=shuffle_candidates,
            top_k_candidates=topk,
            dataset_name=run_id,
        )
        print("Results saved.")

    return rag_results