"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import abstractmethod
import os
import torch
from transformers import pipeline

from openai import OpenAI


class AbstractLLMJudge:
    @abstractmethod
    def pre_process_text(self, prompts):
        raise NotImplementedError

    @abstractmethod
    def eval_batch(self, prompts):
        raise NotImplementedError


class DeepSeekJudge(AbstractLLMJudge):
    def __init__(self, model_name="deepseek-ai/DeepSeek-V3"):
        self.model_name = model_name
        self.openai = OpenAI(
            api_key=os.environ["DEEPINFRA_TOKEN"],
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def pre_process_text(self, prompts):
        return [[{"role": "user", "content": prompt}] for prompt in prompts]

    def eval_batch(self, prompts):
        prompts = self.pre_process_text(prompts)
        correct = []
        for prompt in prompts:
            response = self.openai.chat.completions.create(
                model=self.model_name,
                messages=prompt,
            )
            correct.append(
                1 if "yes" in response.choices[0].message.content.lower().strip() else 0
            )
        return correct


class LlamaJudge(AbstractLLMJudge):
    def __init__(
        self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=32
    ):
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.model = pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="cuda:0",
        )

    def pre_process_text(self, prompts):
        return [[{"role": "user", "content": prompt}] for prompt in prompts]

    def eval_batch(self, prompts, return_output=False):
        output = self.model(
            prompts,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.model.tokenizer.eos_token_id,
            do_sample=False,
        )
        if return_output:
            return [x[0]["generated_text"][-1]["content"] for x in output]
        return [
            1 if "yes" in x[0]["generated_text"][-1]["content"] else 0 for x in output
        ]