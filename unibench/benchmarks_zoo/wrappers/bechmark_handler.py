"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import abstractmethod
import itertools
from torch.distributions import Categorical
from torch import softmax
import torch
import random

import transformers

from transformers import AutoTokenizer

class BenchmarkHandler:
    def __init__(self, benchmark_name, benchmark, task_name):
        self.benchmark_name = benchmark_name
        self.benchmark = benchmark
        self.task_name = task_name

    @abstractmethod
    def eval_batch(self, model, batch):
        raise NotImplementedError

    @abstractmethod
    def on_validation_start(self, model):
        pass


class ZeroShotBenchmarkHandler(BenchmarkHandler):
    def __init__(self, benchmark_name, benchmark, classes, templates, topx=1, task_name='zeroshot_classification'):
        BenchmarkHandler.__init__(self, benchmark_name, benchmark, task_name)
        assert classes is not None, "Classes must be provided for zero shot benchmarks"
        assert (
            templates is not None
        ), "Templates must be provided for zero shot benchmarks"
        self.classes = classes
        self.templates = templates
        self.topx = topx

    def on_validation_start(self, model):
        model.set_classes(self.classes)
        model.set_templates(self.templates)
        model.compute_zeroshot_weights()

    def get_zeroshot_predictions(self, model, images):
        logit_scale = (
            model.logit_scale.exp()
            if model.logit_scale is not None
            else torch.tensor(100.0)
        )

        return (
            (logit_scale * model.get_image_embeddings(images) @ model.zeroshot_weights)
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

        logits = self.get_zeroshot_predictions(model, images)

        if len(targets.shape) > 1:
            pred = softmax(logits, dim=-1).topk(1)[1].squeeze()
            entropy = Categorical(probs=softmax(logits, dim=-1)).entropy()
            correct = targets[range(len(targets)), pred.squeeze()].clamp(0, 1)
            confidence = softmax(logits, dim=-1).topk(1)[0].squeeze()
            top5 = softmax(logits, dim=-1).topk(5)[1]
            correct_top5 = (
                torch.bitwise_and(
                    torch.nn.functional.one_hot(top5, len(self.classes)).sum(1),
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
            _, pred = pred.topk(self.topx, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred)).int().sum(0)

            if len(self.classes) < 5:
                top5 = targets
                correct_top5 = [1] * len(targets)
            else:
                pred = softmax(logits, dim=-1)
                _, top5 = pred.topk(5, 1, True, True)
                correct_top5 = (
                    torch.bitwise_and(
                        torch.nn.functional.one_hot(top5, len(self.classes)).sum(1),
                        torch.nn.functional.one_hot(targets, len(self.classes)),
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
            
        res['task_name'] = self.task_name

        return res


class RelationBenchmarkHandler(BenchmarkHandler):
    def __init__(self, benchmark_name, benchmark, task_name='relation_classification'):
        BenchmarkHandler.__init__(self, benchmark_name, benchmark, task_name)

    def get_similarity(self, model, images, captions):
        image_features = model.get_image_embeddings(images)
        num_captions = len(captions)
        batch_size = len(captions[0])

        caption_features = (
            model.get_text_embeddings(list(itertools.chain.from_iterable(captions)))
            .reshape(num_captions, batch_size, -1)
            .permute(1, 0, 2)
        )

        scores = torch.einsum("nkd,nld->nkl", image_features, caption_features)

        if model.use_itm_head:
            scores = model.use_mlp_head(
                scores,
                model.model.visual_encoder(images.to(model.device)).unsqueeze(1),
                captions,
            )

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

        res['task_name'] = self.task_name

        return res

class TextClassificationBenchmarkHandler(BenchmarkHandler):
    def __init__(self, benchmark_name, benchmark, class_names, prompt='What type of object is in this photo? Choose one from {class_names}', num_classes=-1, task_name='text_classification'):
        BenchmarkHandler.__init__(self, benchmark_name, benchmark, task_name)
        self.class_names = class_names
        self.prompt = prompt
        self.num_classes = num_classes

    def eval_batch(self, model, batch):
        split = ""
        if len(batch) == 4:
            images, targets, sample_id, split = batch
        elif len(batch) == 3:
            images, targets, sample_id = batch
        else:
            images, targets = batch
            
        targets_names = [self.class_names[i - 1] for i in targets]
        prompts = []
        for target in targets_names:
            if self.num_classes == -1:
                prompts.append(self.prompt.format(class_names=", ".join(self.class_names)))
            else:
                random_classes = [target]
                random_classes += random.sample([cls for cls in self.class_names if cls != target], self.num_classes - 1)
                random.shuffle(random_classes)
                prompts.append(self.prompt.format(class_names=", ".join(random_classes)))

        text_outputs = model.get_text_from_image(images, prompts)
        
        correct = [1 if target in text_output else 0 for text_output, target in zip(text_outputs, targets_names)]

        res = {
            "image_class": targets,
            "split": split,
            "benchmark_name": self.benchmark_name,
            "correctness": correct,
        }

        if len(batch) > 2:
            res["image_name"] = sample_id
            
        res['task_name'] = self.task_name

        return res



class CLIPJudgeBenchmarkHandler(BenchmarkHandler):
    def __init__(self, benchmark_name, benchmark, classes, templates, prompt="Fill in the Blank: This is a photo of a {}.", task_name='clip_judge_classification', clip_model="openai/clip-vit-large-patch14"):
        BenchmarkHandler.__init__(self, benchmark_name, benchmark, task_name)
        self.prompt = prompt
        
        self.embedding_model = transformers.CLIPTextModelWithProjection.from_pretrained(
            clip_model, torch_dtype=torch.float16
        ).cuda()
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            clip_model
        )
        self.classes = classes
        self.templates = templates
        self.embedding_model.eval()

    def on_validation_start(self, model):
        zeroshot_weights = []
        for class_name in self.classes:
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
    
    def eval_batch(self, model, batch):
        split = ""
        if len(batch) == 4:
            images, targets, sample_id, split = batch
        elif len(batch) == 3:
            images, targets, sample_id = batch
        else:
            images, targets = batch
            
        targets_names = [self.class_names[i - 1] for i in targets]
        prompts = []
        for target in targets_names:
            if self.num_classes == -1:
                prompts.append(self.prompt.format(class_names=", ".join(self.class_names)))
            else:
                random_classes = [target]
                random_classes += random.sample([cls for cls in self.class_names if cls != target], self.num_classes - 1)
                random.shuffle(random_classes)
                prompts.append(self.prompt.format(class_names=", ".join(random_classes)))

        text_outputs = model.get_text_from_image(images, prompts)
        
        correct = [1 if target in text_output else 0 for text_output, target in zip(text_outputs, targets_names)]

        res = {
            "image_class": targets,
            "split": split,
            "benchmark_name": self.benchmark_name,
            "correctness": correct,
        }

        if len(batch) > 2:
            res["image_name"] = sample_id
            
        res['task_name'] = self.task_name

        return res
    
    
class LLMJudgeBenchmarkHandler(BenchmarkHandler):
    def __init__(self, benchmark_name, benchmark, classes, templates, prompt="Fill in the Blank: This is a photo of a {}.", task_name='llm_judge_classification', clip_model="openai/clip-vit-large-patch14"):
        BenchmarkHandler.__init__(self, benchmark_name, benchmark, task_name)
        self.prompt = prompt
        
        self.embedding_model = transformers.CLIPTextModelWithProjection.from_pretrained(
            clip_model, torch_dtype=torch.float16
        ).cuda()
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            clip_model
        )
        self.classes = classes
        self.templates = templates
        self.embedding_model.eval()

    def on_validation_start(self, model):
        zeroshot_weights = []
        for class_name in self.classes:
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
    
    def eval_batch(self, model, batch):
        split = ""
        if len(batch) == 4:
            images, targets, sample_id, split = batch
        elif len(batch) == 3:
            images, targets, sample_id = batch
        else:
            images, targets = batch
            
        targets_names = [self.class_names[i - 1] for i in targets]
        prompts = []
        for target in targets_names:
            if self.num_classes == -1:
                prompts.append(self.prompt.format(class_names=", ".join(self.class_names)))
            else:
                random_classes = [target]
                random_classes += random.sample([cls for cls in self.class_names if cls != target], self.num_classes - 1)
                random.shuffle(random_classes)
                prompts.append(self.prompt.format(class_names=", ".join(random_classes)))

        text_outputs = model.get_text_from_image(images, prompts)
        
        correct = [1 if target in text_output else 0 for text_output, target in zip(text_outputs, targets_names)]

        res = {
            "image_class": targets,
            "split": split,
            "benchmark_name": self.benchmark_name,
            "correctness": correct,
        }

        if len(batch) > 2:
            res["image_name"] = sample_id
            
        res['task_name'] = self.task_name

        return res