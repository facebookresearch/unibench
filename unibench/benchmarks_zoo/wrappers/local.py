"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class OpenAppsDataset(Dataset):
	"""
	Local dataset wrapper for OpenApps UI multiple-choice questions.

	Supported output formats:
	- vllm: (image, question, choices, correct_letter, sample_id, split)
	- clip: (image, question, choices, correct_index, sample_id)
	"""

	def __init__(
		self,
		questions_json="openapps/ui_questions.json",
		root=None,
		transform=None,
		max_num_samples=None,
		subset_kwargs=None,
		output_format="vllm",
		shuffle=False,
		random_seed=42,
		*args,
		**kwargs,
	):
		Dataset.__init__(self, *args, **kwargs)

		self.transform = transform
		self.output_format = output_format
		self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
		self.questions_json = self.root / questions_json

		if not self.questions_json.exists():
			raise FileNotFoundError(f"OpenApps questions file not found: {self.questions_json}")

		with open(self.questions_json, "r") as f:
			dataset = json.load(f)

		if subset_kwargs is not None:
			for cond in subset_kwargs:
				key, val = next(iter(cond.items()))
				dataset = [x for x in dataset if key in x and str(val).lower() in str(x[key]).lower()]

		if shuffle:
			random.Random(random_seed).shuffle(dataset)

		if max_num_samples is not None:
			dataset = dataset[:max_num_samples]

		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def _resolve_image_path(self, screenshot_path):
		screenshot_path = Path(screenshot_path)

		# Try path as-is from repo root first.
		candidate = self.root / screenshot_path
		if candidate.exists():
			return candidate

		# OpenApps data stores screenshots under openapps/generated_screenshots/...,
		# while json entries may point to tests/generated_screenshots/....
		parts = list(screenshot_path.parts)
		if len(parts) >= 2 and parts[0] == "tests" and parts[1] == "generated_screenshots":
			candidate = self.root / Path("openapps", *parts[1:])
			if candidate.exists():
				return candidate

		# Fallback to openapps/ prefix if relative path was provided differently.
		candidate = self.root / "openapps" / screenshot_path
		if candidate.exists():
			return candidate

		raise FileNotFoundError(f"OpenApps screenshot not found: {screenshot_path}")

	def _choices_to_list(self, choices_dict):
		if isinstance(choices_dict, dict):
			return [choices_dict[k] for k in sorted(choices_dict.keys())]
		return list(choices_dict)

	def __getitem__(self, index):
		item = self.dataset[index]

		image_path = self._resolve_image_path(item["screenshot_path"])
		image = Image.open(image_path).convert("RGB")
		if self.transform is not None:
			image = self.transform(image)

		question = item.get("question", "")
		choices_list = self._choices_to_list(item.get("choices", []))
		choices_dict = item.get("choices", {})
		if isinstance(choices_dict, dict):
			choices_letters = ", ".join([f"{k}) {choices_dict[k]}" for k in sorted(choices_dict.keys())])
		else:
			choices_letters = str(choices_dict)
		correct_letter = str(item.get("correct", "")).strip().upper()
		correct_index = max(0, ord(correct_letter) - ord("A")) if correct_letter else 0

		sample_id = f"openapps_{index}"
		split = "|".join(
			[
				str(item.get("app", "")),
				str(item.get("category", "")),
				str(item.get("difficulty", "")),
			]
		)

		return image, question, {'letters': choices_letters, 'list': choices_list}, {'letters': correct_letter, 'index': correct_index}, sample_id, split
