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

from .benchmark_handler import BenchmarkHandler


class ZeroShotBenchmarkHandler(BenchmarkHandler):
    def __init__(
        self,
        class_names,
        templates,
        task_name="zeroshot_classification",
        topk=1,
        num_classes=-1,
        **kwargs,
    ):
        BenchmarkHandler.__init__(self, task_name=task_name, **kwargs)
        assert (
            class_names is not None
        ), "Classes must be provided for zero shot benchmarks"
        assert (
            templates is not None
        ), "Templates must be provided for zero shot benchmarks"
        self.class_names = class_names
        self.templates = templates
        self.topk = topk
        self.num_classes = num_classes

    def on_validation_start(self, model):
        model.set_classes(self.class_names)
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
            if self.num_classes != -1 and self.num_classes < len(self.class_names):
                selected_targets = logits.gather(1, targets.unsqueeze(1))
                batch_size, num_classes = logits.shape
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[torch.arange(batch_size), targets] = False
                pred_without_targets = logits[mask].view(batch_size, num_classes - 1)
                pred_without_targets = pred_without_targets.gather(1, torch.randint(0, num_classes - 1, (batch_size, self.num_classes - 1), device=logits.device))
                logits_ = torch.cat([selected_targets, pred_without_targets], axis=1)
                pred = softmax(logits_, dim=-1)
                _, pred = pred.topk(self.topk, 1, True, True)
                pred = pred.squeeze()
                correct = (pred == 0).int()
            else:
                _, pred = pred.topk(self.topk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred)).int().sum(0)

            # if len(self.class_names) < 5:
            #     top5 = targets
            #     correct_top5 = [1] * len(targets)
            # else:
            #     pred = softmax(logits, dim=-1)
            #     _, top5 = pred.topk(5, 1, True, True)
            #     correct_top5 = (
            #         torch.bitwise_and(
            #             torch.nn.functional.one_hot(top5, len(self.class_names)).sum(1),
            #             torch.nn.functional.one_hot(targets, len(self.class_names)),
            #         )
            #         .sum(1)
            #         .int()
            #     )

        res = {
            "entropy": entropy,
            "image_class": targets,
            "split": split,
            "benchmark_name": self.benchmark_name,
            "correctness": correct,
            # "correctness_top5": correct_top5,
            "predictions": pred,
            # "predictions_top5": top5,
            "confidence": confidence,
        }

        if len(batch) > 2:
            res["image_name"] = sample_id

        res["task_name"] = self.task_name

        return res


class VQABenchmarkHandler(BenchmarkHandler):
    """
    Handler for VQA datasets with multiple-choice questions using CLIP models.
    For each answer option, the caption is formed as: "<question> <option>".
    The image is compared against all such captions and the highest similarity
    caption index is used as the prediction.
    """

    def __init__(self, task_name="vqa_multiple_choice", **kwargs):
        BenchmarkHandler.__init__(self, task_name=task_name, **kwargs)

    def get_similarity(self, model, images, captions):
        """
        captions: list of strings of length batch_size, each already formatted
                  as "question option_i".
        Returns similarity scores of shape (batch_size,).
        """
        image_features = model.get_image_embeddings(images)
        text_features = model.get_text_embeddings(captions)
        logit_scale = (
            model.logit_scale.exp()
            if model.logit_scale is not None
            else torch.tensor(100.0)
        )
        scores = logit_scale * (image_features * text_features).sum(dim=-1)
        return scores

    def eval_batch(self, model, batch):
        """
        Expected batch format:
          (images, questions, options, targets, sample_id)
          or
          (images, questions, options, targets)

        - images: tensor of shape (B, C, H, W)
        - questions: list of B question strings
        - options: list of B lists, each containing N answer-option strings
        - targets: tensor of shape (B,) with the correct option index (0-based)
        - sample_id: optional list of B sample identifiers
        """
        sample_id = None
        split = None
        if len(batch) == 6:
            images, questions, options, targets, sample_id, split = batch
        else:
            images, questions, options, targets = batch
            
        if isinstance(targets, dict):
            targets = targets['index']
            
        if isinstance(options, dict):
            options = options['list']

        batch_size = images.shape[0]
        num_options = len(options)

        # Build per-option scores: shape (B, num_options)
        option_scores = []
        for opt_idx in range(num_options):
            # Build captions: "question option" for every item in the batch
            captions = [
                f"{questions[i]} {options[opt_idx][i]}" for i in range(batch_size)
            ]
            scores = self.get_similarity(model, images, captions)
            # scores = torch.diagonal(scores)# (B,)
            option_scores.append(torch.diagonal(scores))

        logits = torch.stack(option_scores, dim=1).float()   # (B, num_options)

        probs = softmax(logits, dim=-1)
        confidence, pred = probs.topk(1, dim=1)
        pred = pred.squeeze(1).cpu()          # (B,)
        confidence = confidence.squeeze(1).cpu()  # (B,)
        entropy = Categorical(probs=probs).entropy()  # (B,)
        correct = pred.eq(targets).int()

        res = {
            "entropy": entropy,
            "image_class": targets,
            "benchmark_name": self.benchmark_name,
            "correctness": correct,
            "predictions": pred,
            "confidence": confidence,
            "split": split,
        }

        if sample_id is not None:
            res["image_name"] = sample_id

        res["task_name"] = self.task_name

        return res


class RelationBenchmarkHandler(BenchmarkHandler):
    def __init__(self, benchmark_name, benchmark, task_name="relation_classification"):
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

        res["task_name"] = self.task_name

        return res
    
    
