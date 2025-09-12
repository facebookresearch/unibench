"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import itertools
from torch.distributions import Categorical
from torch import softmax
import torch
import random

import transformers

from transformers import AutoTokenizer

from .benchmark_handler import BenchmarkHandler
from unibench.benchmarks_zoo.wrappers.llm_judge_models import DeepSeekJudge, LlamaJudge
from unibench.common_utils.constants import MISC_DIR


class VLLMBenchmarkHandler(BenchmarkHandler):
    def __init__(
        self,
        benchmark_name,
        benchmark,
        task_name,
        class_names,
        num_classes=4,
        prompt="What type of object is in this photo? Choose one from {class_names}.",
        random_seed=1337,
    ):
        BenchmarkHandler.__init__(self, benchmark_name, benchmark, task_name)
        self.class_names = class_names
        self.prompt = prompt
        self.num_classes = num_classes
        self.random_seed = random_seed
        random.seed(self.random_seed)

    def get_prompts(self, targets_names):
        prompts = []
        prompt_classes = []
        for target in targets_names:
            if self.num_classes == -1:
                prompts.append(
                    self.prompt.format(class_names=", ".join(self.class_names))
                )
                prompt_classes.append("all")
            else:
                random_classes = [target]
                random_classes += random.sample(
                    [cls for cls in self.class_names if cls != target],
                    (
                        self.num_classes - 1
                        if len(self.class_names) > self.num_classes
                        else len(self.class_names) - 1
                    ),
                )
                random.shuffle(random_classes)
                prompts.append(
                    self.prompt.format(class_names=", ".join(random_classes))
                )
                prompt_classes.append(", ".join(random_classes))
        return prompts, prompt_classes


class TextClassificationBenchmarkHandler(VLLMBenchmarkHandler):
    def __init__(
        self,
        task_name="text_classification",
        **kwargs,
    ):
        VLLMBenchmarkHandler.__init__(self, task_name=task_name, **kwargs)

    def eval_batch(self, model, batch):
        split = ""
        if len(batch) == 4:
            images, targets, sample_id, split = batch
        elif len(batch) == 3:
            images, targets, sample_id = batch
        else:
            images, targets = batch

        if len(targets.shape) > 1:
            targets_names = [self.class_names[i.argmax() - 1].lower() for i in targets]
        else:
            targets_names = [self.class_names[i - 1].lower() for i in targets]
        prompts, prompt_classes = self.get_prompts(targets_names)
        text_outputs = model.get_text_from_image(images, prompts)

        correct = [
            1 if target.lower() in text_output.lower() else 0
            for text_output, target in zip(text_outputs, targets_names)
        ]

        res = {
            "image_class": targets,
            "target_class": targets_names,
            "prompt_classes": prompt_classes,
            "split": split,
            "benchmark_name": self.benchmark_name,
            "correctness": correct,
            "prompt": self.prompt,
            "num_classes": self.num_classes,
            "model_output": text_outputs,
        }

        if len(batch) > 2:
            res["image_name"] = sample_id

        res["task_name"] = self.task_name

        return res


class MultiChoiceClassificationBenchmarkHandler(VLLMBenchmarkHandler):
    def __init__(
        self,
        task_name="multi_choice_classification",
        prompt="Look carefully at the image. Only one of the following options correctly describes the objects and their relationships. Choose the most accurate option and respond with **only the letter**.\n{class_names}",
        **kwargs,
    ):
        VLLMBenchmarkHandler.__init__(
            self, task_name=task_name, prompt=prompt, **kwargs
        )

    def get_prompts(self, targets_names):
        prompts = []
        prompt_classes = []
        target_letters = []
        for target in targets_names:
            if self.num_classes == -1:
                class_list = self.class_names
            else:
                class_list = [target]
                class_list += random.sample(
                    [cls for cls in self.class_names if cls != target],
                    (
                        self.num_classes - 1
                        if len(self.class_names) > self.num_classes
                        else len(self.class_names) - 1
                    ),
                )
                random.shuffle(class_list)

            # Find the correct answer letter
            target_index = class_list.index(target)
            target_letter = chr(65 + target_index)  # A, B, C, etc.
            target_letters.append(target_letter)

            # Multiple choice formatting: A. class1 B. class2 ...
            choices = [f"({chr(65+i)}) {name}" for i, name in enumerate(class_list)]
            prompt_text = self.prompt.replace(
                "{class_names}", f"\n" + " ".join(choices)
            )
            prompts.append(prompt_text)
            prompt_classes.append(", ".join(class_list))
        return prompts, prompt_classes, target_letters

    def eval_batch(self, model, batch):
        split = ""
        if len(batch) == 4:
            images, targets, sample_id, split = batch
        elif len(batch) == 3:
            images, targets, sample_id = batch
        else:
            images, targets = batch

        if len(targets.shape) > 1:
            targets_names = [self.class_names[i.argmax()].lower() for i in targets]
        else:
            targets_names = [self.class_names[i].lower() for i in targets]
        prompts, prompt_classes, target_letters = self.get_prompts(targets_names)
        text_outputs = model.get_text_from_image(images, prompts)

        # Check if the model output contains the correct letter option
        correct = []
        for text_output, target_letter in zip(text_outputs, target_letters):
            # Look for the target letter in the output (e.g., "A", "(A)", "A.", etc.)
            output_upper = text_output
            is_correct = (
                target_letter in output_upper
                or f"({target_letter})" in output_upper
                or f"{target_letter}." in output_upper
                or f"{target_letter}:" in output_upper
            )
            correct.append(1 if is_correct else 0)

        res = {
            "image_class": targets,
            "target_class": targets_names,
            "target_letters": target_letters,
            "prompt_classes": prompt_classes,
            "split": split,
            "benchmark_name": self.benchmark_name,
            "correctness": correct,
            "prompt": prompts,
            "num_classes": self.num_classes,
            "model_output": text_outputs,
        }

        if len(batch) > 2:
            res["image_name"] = sample_id

        res["task_name"] = self.task_name

        return res


class MultiChoiceRelationBenchmarkHandler(VLLMBenchmarkHandler):
    def __init__(
        self,
        task_name="multi_choice_relation",
        prompt="Look carefully at the image. Only one of the following options correctly describes the objects and their relationships. Choose the most accurate option and respond with **only the letter**.\n{class_names}",
        num_classes=-1,
        **kwargs,
    ):
        VLLMBenchmarkHandler.__init__(
            self,
            task_name=task_name,
            prompt=prompt,
            num_classes=num_classes,
            class_names=None,
            **kwargs,
        )

    def get_prompts(self, captions):
        prompts = []
        for caption in captions:
            choices = [f"({chr(65+i)}) {c}" for i, c in enumerate(caption)]
            prompt_text = self.prompt.replace(
                "{class_names}", "\n".join(choices)
            )
            prompts.append(prompt_text)
        return prompts

    def eval_batch(self, model, batch):
        attribute = None
        if len(batch) == 4:
            images, captions, sample_id, attribute = batch
        else:
            images, captions, sample_id = batch

        captions = [list(items) for items in zip(*captions)]
        
        prompts = self.get_prompts(captions)
        
        if isinstance(images, list):
            c_i0 = model.get_text_from_image(images[0], prompts)
            correct_i0 = [1 if 'A' in text else 0 for text in c_i0]
            c_i1 = model.get_text_from_image(images[1], prompts)
            correct_i1 = [1 if 'B' in text else 0 for text in c_i1]

            correct = [int(a & b) for a, b in zip(correct_i0, correct_i1)]
            outputs = [f"{a};{b}" for a, b in zip(c_i0, c_i1)]
        else:    
            outputs = model.get_text_from_image(images, prompts)
            correct = [1 if 'A' in text else 0 for text in outputs]

        res = {
            "benchmark_name": self.benchmark_name,
            "correctness": correct,
            "prompt": prompts,
            "num_classes": self.num_classes,
            "model_output": outputs,
        }

        if len(batch) > 2:
            res["image_name"] = sample_id

        if attribute is not None:
            if isinstance(attribute[0], str) and "\n" in attribute[0]:
                attribute = [x.split("\n") for x in attribute]
            res["attribute"] = attribute

        res["task_name"] = self.task_name

        return res


class CLIPJudgeBenchmarkHandler(VLLMBenchmarkHandler):
    def __init__(
        self,
        templates,
        task_name="clip_judge_classification",
        clip_model="openai/clip-vit-large-patch14-336",
        topk=1,
        **kwargs,
    ):
        VLLMBenchmarkHandler.__init__(self, task_name=task_name, **kwargs)
        self.topk = topk
        self.templates = templates
        self.clip_model = clip_model

    def on_validation_start(self, model):
        self.embedding_model = transformers.CLIPTextModelWithProjection.from_pretrained(
            self.clip_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.clip_model)
        self.embedding_model.eval()
        zeroshot_weights = []
        for class_name in self.class_names:
            texts = [template.format(class_name) for template in self.templates]

            class_embedding = self.get_text_embeddings(texts)

            class_embedding = class_embedding.mean(dim=0)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)

            zeroshot_weights.append(class_embedding)
        self.zeroshot_weights = torch.stack(zeroshot_weights).T

    @torch.no_grad()
    def get_text_embeddings(self, captions):
        text_descriptor = self.embedding_tokenizer(
            captions, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"].cuda()
        text_descriptor_embeds = self.embedding_model(text_descriptor).text_embeds
        text_descriptor_embeds = text_descriptor_embeds / text_descriptor_embeds.norm(
            p=2, dim=-1, keepdim=True
        )
        return text_descriptor_embeds

    def get_zeroshot_predictions(self, model, images, target_names):
        prompts = self.get_prompts(target_names)
        return (
            (
                self.get_text_embeddings(
                    model.get_text_from_image(images, prompts)
                ).unsqueeze(1)
                @ self.zeroshot_weights
            )
            .squeeze()
            .float()
        )

    def eval_batch(self, model, batch):
        split = ""
        if len(batch) == 4:
            images, targets, sample_id, split = batch
        elif len(batch) == 3:
            images, targets, sample_id = batch
        else:
            images, targets = batch

        if len(targets.shape) > 1:
            targets_names = [self.class_names[i.argmax() - 1].lower() for i in targets]
        else:
            targets_names = [self.class_names[i - 1].lower() for i in targets]

        logits = self.get_zeroshot_predictions(model, images, targets_names)

        if len(targets.shape) > 1:
            pred = softmax(logits, dim=-1).topk(1)[1].squeeze()
            entropy = Categorical(probs=softmax(logits, dim=-1)).entropy()
            correct = targets[range(len(targets)), pred.squeeze()].clamp(0, 1)
            confidence = softmax(logits, dim=-1).topk(1)[0].squeeze()
            top5 = softmax(logits, dim=-1).topk(5)[1]
            correct_top5 = (
                torch.bitwise_and(
                    torch.nn.functional.one_hot(top5, len(self.class_names)).sum(1),
                    targets,
                )
                .sum(1)
                .int()
                .clamp(0, 1)
            )
            targets = targets.topk(1)[1].squeeze()
            top5 = top5.tolist()

        else:
            pred = softmax(logits, dim=-1)
            confidence = pred.max(1)[0].squeeze()
            entropy = Categorical(probs=pred).entropy()
            _, pred = pred.topk(self.topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred)).int().sum(0)

            if len(self.class_names) < 5:
                top5 = targets
                correct_top5 = [1] * len(targets)
            else:
                pred = softmax(logits, dim=-1)
                _, top5 = pred.topk(5, 1, True, True)
                correct_top5 = (
                    torch.bitwise_and(
                        torch.nn.functional.one_hot(top5, len(self.class_names)).sum(1),
                        torch.nn.functional.one_hot(targets, len(self.class_names)),
                    )
                    .sum(1)
                    .int()
                )
            pred = pred.topk(1, 1, True, True)[1].squeeze()

        res = {
            "entropy": entropy,
            "image_class": targets,
            "split": split,
            "benchmark_name": self.benchmark_name,
            "correctness": correct,
            "correctness_top5": correct_top5,
            "predictions": pred,
            "predictions_top5": top5,
            "confidence": confidence,
        }

        if len(batch) > 2:
            res["image_name"] = sample_id

        res["task_name"] = self.task_name

        return res


class LLMJudgeBenchmarkHandler(VLLMBenchmarkHandler):
    def __init__(
        self,
        task_name="llm_judge_classification",
        llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        llm_prompt="""
The target class is: {class_names}
The model's prediction is: {prediction}
Question: Is the prediction correct?
Answer with only 'yes' or 'no': 
""",
        **kwargs,
    ):
        VLLMBenchmarkHandler.__init__(self, task_name=task_name, **kwargs)
        self.llm_model = llm_model
        self.llm_prompt = llm_prompt

    def on_validation_start(self, model):
        if "deepseek" in self.llm_model:
            self.llm_model = DeepSeekJudge(
                model_name=self.llm_model,
            )
        elif "llama" in self.llm_model:
            self.llm_model = LlamaJudge(
                model_name=self.llm_model,
            )
        else:
            raise ValueError(
                f"LLM model {self.llm_model} not supported. Please use either DeepSeek or Llama models."
            )

    def eval_batch(self, model, batch):
        split = ""
        if len(batch) == 4:
            images, targets, sample_id, split = batch
        elif len(batch) == 3:
            images, targets, sample_id = batch
        else:
            images, targets = batch

        if len(targets.shape) > 1:
            targets_names = [self.class_names[i.argmax() - 1].lower() for i in targets]
        else:
            targets_names = [self.class_names[i - 1].lower() for i in targets]
        prompts = self.get_prompts(targets_names)
        text_outputs = model.get_text_from_image(images, prompts)

        llm_prompts = [
            self.llm_prompt.format(
                class_names=target.lower(), prediction=text_output.lower()
            )
            for target, text_output in zip(targets_names, text_outputs)
        ]
        non_mod_prompts = llm_prompts.copy()

        llm_prompts = self.llm_model.pre_process_text(llm_prompts)

        correct = self.llm_model.eval_batch(llm_prompts)

        res = {
            "image_class": targets,
            "split": split,
            "benchmark_name": self.benchmark_name,
            "correctness": correct,
            "prompt": prompts,
            "llm_prompt": non_mod_prompts,
            "llm_model_name": self.llm_model.model_name,
        }

        if len(batch) > 2:
            res["image_name"] = sample_id

        res["task_name"] = self.task_name

        return res


class CLIPJudgeRelationBenchmarkHandler(VLLMBenchmarkHandler):
    def __init__(
        self,
        task_name="relation_classification",
        clip_model="openai/clip-vit-large-patch14-336",
        max_new_tokens=64,
        **kwargs,
    ):
        VLLMBenchmarkHandler.__init__(
            self,
            task_name=task_name,
            prompt="Describe objects in the photo:",
            class_names=None,
            **kwargs,
        )
        self.max_new_tokens = max_new_tokens
        self.clip_model = clip_model

    def on_validation_start(self, model):
        self.embedding_model = transformers.CLIPTextModelWithProjection.from_pretrained(
            self.clip_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.clip_model)
        self.embedding_model.eval()
        model.max_new_tokens = self.max_new_tokens

    def get_prompts(self, num_images):
        prompts = []
        for _ in range(num_images):
            prompts.append(self.prompt)
        return prompts

    @torch.no_grad()
    def get_text_embeddings(self, captions):
        text_descriptor = self.embedding_tokenizer(
            captions, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"].cuda()
        text_descriptor_embeds = self.embedding_model(text_descriptor).text_embeds
        text_descriptor_embeds = text_descriptor_embeds / text_descriptor_embeds.norm(
            p=2, dim=-1, keepdim=True
        )
        return text_descriptor_embeds

    def get_image_embeddings(self, model, images):
        prompts = self.get_prompts(len(images))
        return (
            self.get_text_embeddings(
                model.get_text_from_image(images, prompts)
            ).unsqueeze(1)
        ).float()

    def get_similarity(self, model, images, captions):
        image_features = self.get_image_embeddings(model, images)
        num_captions = len(captions)
        batch_size = len(captions[0])

        caption_features = (
            self.get_text_embeddings(list(itertools.chain.from_iterable(captions)))
            .reshape(num_captions, batch_size, -1)
            .permute(1, 0, 2)
        )

        scores = torch.einsum("nkd,nld->nkl", image_features, caption_features)

        return scores

    def eval_batch(self, model, batch):
        attribute = None
        if len(batch) == 4:
            images, captions, sample_id, attribute = batch
        else:
            images, captions, sample_id = batch

        if isinstance(images, list):
            c_i0 = self.get_similarity(model, images[0], captions).squeeze()
            c_i1 = self.get_similarity(model, images[1], captions).squeeze()
            text_correct = torch.logical_and(
                c_i0[:, 0] > c_i0[:, 1], c_i1[:, 1] > c_i1[:, 0]
            ).int()
            image_correct = torch.logical_and(
                c_i0[:, 0] > c_i1[:, 0], c_i1[:, 1] > c_i0[:, 1]
            ).int()
            correct = torch.logical_and(text_correct, image_correct).int()

            res = {
                "image_name": sample_id,
                "benchmark_name": self.benchmark_name,
                "correctness": correct,
                "text_correctness": text_correct,
                "image_correctness": image_correct,
            }
        else:
            scores = self.get_similarity(model, images, captions)
            preds = torch.argmax(scores.squeeze(), axis=-1)
            correct = (preds == 0).int()

            res = {
                "image_name": sample_id,
                "benchmark_name": self.benchmark_name,
                "correctness": correct,
                "confidence": scores.squeeze(1).max(1)[0],
                "entropy": Categorical(
                    probs=softmax(scores.squeeze(1), dim=-1)
                ).entropy(),
            }

        if attribute is not None:
            if "\n" in attribute[0]:
                attribute = [x.split("\n") for x in attribute]
            res["attribute"] = attribute

        res["task_name"] = self.task_name

        return res


class InContextTextClassificationBenchmarkHandler(VLLMBenchmarkHandler):
    def __init__(
        self,
        task_name="in_context_text_classification",
        llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        description_prompt="Provide a brief description of what a {class_name} is:",
        prompt="What type of object is in this photo? Choose one from {class_name}, here is a description of each:\n{class_descriptions}",
        llm_max_new_tokens=32,
        **kwargs,
    ):
        VLLMBenchmarkHandler.__init__(
            self, task_name=task_name, prompt=prompt, **kwargs
        )
        self.description_prompt = description_prompt
        self.class_descriptions = {}
        self.llm_model = llm_model
        self.llm_max_new_tokens = llm_max_new_tokens
        self.cache_file = MISC_DIR / f"{self.benchmark_name}_class_descriptions.pt"

    def on_validation_start(self, model):
        if self.cache_file.exists():
            # Load class descriptions from cache
            try:
                self.class_descriptions = torch.load(self.cache_file)
                return None
            except:
                print(
                    f"Failed to load class descriptions from {self.cache_file}. Regenerating."
                )

        # Initialize LLM for generating class descriptions
        if "deepseek" in self.llm_model:
            self.llm = DeepSeekJudge(model_name=self.llm_model)
        elif "llama" in self.llm_model:
            self.llm = LlamaJudge(
                model_name=self.llm_model, max_new_tokens=self.llm_max_new_tokens
            )
        else:
            raise ValueError(
                f"LLM model {self.llm_model} not supported. Please use either DeepSeek or Llama models."
            )

        # Generate descriptions for each class using the LLM in batches
        batch_size = 8  # Process 8 class descriptions at a time
        for i in range(0, len(self.class_names), batch_size):
            batch_classes = self.class_names[i : i + batch_size]

            # Create prompts for the batch
            prompts = [
                self.description_prompt.format(class_name=class_name)
                for class_name in batch_classes
            ]
            processed_prompts = self.llm.pre_process_text(prompts)

            # Get descriptions for the batch
            descriptions = self.llm.eval_batch(processed_prompts, return_output=True)

            # Store the descriptions
            for class_name, description in zip(batch_classes, descriptions):
                if class_name.lower() in self.class_descriptions:
                    print(
                        f"Warning: Class {class_name} already has a description. Overwriting."
                    )
                self.class_descriptions[class_name.lower()] = (
                    f"{class_name}: {description}"
                )

        # Save the class descriptions to cache
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.class_descriptions, self.cache_file)
        del self.llm

    def get_prompts(self, targets_names):
        prompts = []
        for target in targets_names:
            if self.num_classes == -1:
                # Include all class descriptions
                class_descs = [
                    self.class_descriptions[c.lower()] for c in self.class_names
                ]
                prompts.append(
                    self.prompt.format(class_descriptions="\n".join(class_descs))
                )
            else:
                # Include target and random subset of classes
                random_classes = [target]
                random_classes += random.sample(
                    [cls for cls in self.class_names if cls.lower() != target],
                    (
                        self.num_classes - 1
                        if len(self.class_names) > self.num_classes
                        else len(self.class_names) - 1
                    ),
                )
                random.shuffle(random_classes)
                class_descs = [
                    self.class_descriptions[c.lower()] for c in random_classes
                ]
                prompts.append(
                    self.prompt.format(
                        class_names=", ".join(random_classes),
                        class_descriptions="\n".join(class_descs),
                    )
                )
        return prompts

    def eval_batch(self, model, batch):
        split = ""
        if len(batch) == 4:
            images, targets, sample_id, split = batch
        elif len(batch) == 3:
            images, targets, sample_id = batch
        else:
            images, targets = batch

        if len(targets.shape) > 1:
            targets_names = [self.class_names[i.argmax() - 1].lower() for i in targets]
        else:
            targets_names = [self.class_names[i - 1].lower() for i in targets]
        prompts = self.get_prompts(targets_names)
        text_outputs = model.get_text_from_image(images, prompts)

        correct = [
            1 if target.lower() in text_output.lower() else 0
            for text_output, target in zip(text_outputs, targets_names)
        ]

        res = {
            "image_class": targets,
            "split": split,
            "benchmark_name": self.benchmark_name,
            "correctness": correct,
            "prompt": prompts,
            "num_classes": self.num_classes,
            "class_descriptions": self.class_descriptions,
        }

        if len(batch) > 2:
            res["image_name"] = sample_id

        res["task_name"] = self.task_name

        return res
