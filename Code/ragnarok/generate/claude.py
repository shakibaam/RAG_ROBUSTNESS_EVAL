import time
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import anthropic
import tiktoken

from data import RAGExecInfo, Request
from generate.llm import LLM, PromptMode
from generate.post_processor import ClaudePostProcessor
from generate.templates.ragnarok_templates import RagnarokTemplates


class SafeClaude(LLM):
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode = PromptMode.CHATQA,
        max_output_tokens: int = 1500,
        num_few_shot_examples: int = 0,
        keys=None,
        key_start_id=None,
    ) -> None:
        """
        Creates instance of the SafeClaude class, a specialized version of RankLLM designed for safely handling Anthropic API calls with
        support for key cycling.

        Parameters:
        - model (str): The model identifier for the LLM (e.g., 'claude-3-opus-20240229', 'claude-3-sonnet-20240229').
        - context_size (int): The maximum number of tokens that the model can handle in a single request.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to CHATQA.
        - max_output_tokens (int, optional): Maximum number of tokens that can be generated in a single response. Defaults to 1500.
        - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt. Defaults to 0.
        - keys (Union[List[str], str], optional): A list of Anthropic API keys or a single Anthropic API key.
        - key_start_id (int, optional): The starting index for the Anthropic API key cycle.

        Raises:
        - ValueError: If an unsupported prompt mode is provided or if no Anthropic API keys are supplied.
        """
        super().__init__(
            model, context_size, prompt_mode, max_output_tokens, num_few_shot_examples
        )
        self._max_output_tokens = max_output_tokens
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise ValueError("Please provide Anthropic API Keys.")
        if prompt_mode not in [
            PromptMode.CHATQA,
            PromptMode.RAGNAROK_V2,
            PromptMode.RAGNAROK_V3,
            PromptMode.RAGNAROK_V4,
            PromptMode.RAGNAROK_V4_BIOGEN,
            PromptMode.RAGNAROK_V5_BIOGEN,
            PromptMode.RAGNAROK_V5_BIOGEN_NO_CITE,
            PromptMode.RAGNAROK_V4_NO_CITE,
        ]:
            raise ValueError(
                f"unsupported prompt mode for Claude models: {prompt_mode}, expected one of {PromptMode.CHATQA}, {PromptMode.RAGNAROK_V2}, {PromptMode.RAGNAROK_V3}, {PromptMode.RAGNAROK_V4}, {PromptMode.RAGNAROK_V4_NO_CITE}."
            )

        self._keys = keys
        self._cur_key_id = key_start_id or 0
        self._cur_key_id = self._cur_key_id % len(self._keys)
        self._post_processor = ClaudePostProcessor()
        self._client = anthropic.Anthropic(api_key=self._keys[self._cur_key_id])

    def _call_completion(
        self,
        *args,
        return_text=False,
        reduce_length=False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        while True:
            try:
                completion = self._client.messages.create(
                    *args,
                    max_tokens=self._max_output_tokens,
                    **kwargs,
                    timeout=30
                )
                break
            except Exception as e:
                print(str(e))
                if "context_length_exceeded" in str(e):
                    print("reduce_length")
                    return "ERROR::reduce_length"
                if "content_filter" in str(e):
                    print("The response was filtered")
                    return "ERROR::The response was filtered"
                self._cur_key_id = (self._cur_key_id + 1) % len(self._keys)
                self._client = anthropic.Anthropic(api_key=self._keys[self._cur_key_id])
                time.sleep(0.1)
        if return_text:
            completion = completion.content[0].text
        return completion

    def run_llm(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        logging: bool = False,
    ) -> Tuple[str, RAGExecInfo]:
        if logging:
            print(f"Prompt: {prompt}")
        response = self._call_completion(
            system=prompt["system"],
            messages=prompt["messages"],
            temperature=0.1,
            return_text=True,
            model=self._model,
        )
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
        if logging:
            print(f"Response: {response}")
        answers, rag_exec_response = self._post_processor(response)
        if logging:
            print(f"Answers: {answers}")
        rag_exec_info = RAGExecInfo(
            prompt=prompt,
            response=rag_exec_response,
            input_token_count=self.get_num_tokens(prompt),
            output_token_count=sum([len(ans.text) for ans in answers]),
            candidates=[],
        )
        if logging:
            print(f"RAG Exec Info: {rag_exec_info}")
        return answers, rag_exec_info

    def create_prompt(
        self, request: Request, topk: int
    ) -> Tuple[List[Dict[str, str]], int]:
        query = request.query.text
        max_length = (self._context_size - 200) // topk
        while True:
            rank = 0
            context = []
            for cand in request.candidates[:topk]:
                rank += 1
                content = self.convert_doc_to_prompt_content(cand.doc, max_length)
                context.append(
                    f"[{rank}] {self._replace_number(content)}",
                )
            if self._prompt_mode in [
                PromptMode.CHATQA,
                PromptMode.RAGNAROK_V2,
                PromptMode.RAGNAROK_V3,
                PromptMode.RAGNAROK_V4,
                PromptMode.RAGNAROK_V4_BIOGEN,
                PromptMode.RAGNAROK_V5_BIOGEN,
                PromptMode.RAGNAROK_V5_BIOGEN_NO_CITE,
                PromptMode.RAGNAROK_V4_NO_CITE,
            ]:
                ragnarok_template = RagnarokTemplates(self._prompt_mode)
                messages = ragnarok_template(query, context, "claude")
            else:
                raise ValueError(
                    f"Unsupported prompt mode: {self._prompt_mode}, expected one of CHATQA or RAGNAROK_V..."
                )
            num_tokens = self.get_num_tokens(messages)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.max_tokens() + self.num_output_tokens())
                    // (topk * 4),
                )
        return messages, self.get_num_tokens(messages)

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]], Dict]) -> int:
        """Returns the number of tokens used by a list of messages in prompt."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except:
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0

        if isinstance(prompt, dict) and "messages" in prompt:
            # Claude v3 / v4 style
            for message in prompt["messages"]:
                for content_block in message["content"]:
                    if content_block["type"] == "text":
                        num_tokens += len(encoding.encode(content_block["text"]))
            # System message
            if "system" in prompt and prompt["system"]:
                num_tokens += len(encoding.encode(prompt["system"]))

        elif isinstance(prompt, list):
            # Fallback for GPT style (list of message dicts)
            for message in prompt:
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))

        else:
            # Fallback for plain string
            num_tokens += len(encoding.encode(prompt))

        return num_tokens

    def cost_per_1k_token(self, input_token: bool) -> float:
        # Brought in from https://www.anthropic.com/pricing on 2024-03-19
        cost_dict = {
            ("claude-3-opus", 200000): 15.0 if input_token else 75.0,
            ("claude-3-sonnet", 200000): 3.0 if input_token else 15.0,
            ("claude-3-haiku", 200000): 0.25 if input_token else 1.25,
        }
        model_key = "claude-3-opus" if "opus" in self._model else "claude-3-sonnet" if "sonnet" in self._model else "claude-3-haiku"
        return cost_dict[(model_key, self._context_size)] 