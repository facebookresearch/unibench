"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import ast
import json
import os
import random
import urllib.request
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

_OPENAPPS_QUESTIONS_URL = (
	"https://raw.githubusercontent.com/facebookresearch/OpenApps/main"
	"/tests/ui_questions/ui_questions.json"
)
_OPENAPPS_SCREENSHOTS_BASE_URL = (
	"https://raw.githubusercontent.com/facebookresearch/OpenApps/main"
)


class OpenAppsDataset(Dataset):
	"""
	Local dataset wrapper for OpenApps UI multiple-choice questions.

	Supported output formats:
	- vllm: (image, question, choices, correct_letter, sample_id, split)
	- clip: (image, question, choices, correct_index, sample_id)

	Set ``download=True`` (the default) to automatically fetch missing data
	from https://github.com/facebookresearch/OpenApps on first use.
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
		download=True,
		*args,
		**kwargs,
	):
		Dataset.__init__(self, *args, **kwargs)

		self.transform = transform
		self.output_format = output_format
		self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
		self.questions_json = self.root / questions_json

		if download:
			self._ensure_questions_json()

		if not self.questions_json.exists():
			raise FileNotFoundError(f"OpenApps questions file not found: {self.questions_json}")

		with open(self.questions_json, "r") as f:
			dataset = json.load(f)

		if download:
			self._ensure_screenshots(dataset)

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

	@staticmethod
	def _download_atomic(url, dest):
		"""Download *url* to *dest* atomically.

		Writes to a temporary file in the same directory and only renames it
		into place once the download completes, so an interrupted download
		never leaves a truncated/corrupt file that later runs would mistake
		for valid cached data.
		"""
		dest = Path(dest)
		dest.parent.mkdir(parents=True, exist_ok=True)
		tmp = dest.with_name(dest.name + ".part")
		try:
			urllib.request.urlretrieve(url, tmp)
			os.replace(tmp, dest)
		finally:
			if tmp.exists():
				tmp.unlink()

	def _ensure_questions_json(self):
		"""Download ui_questions.json from GitHub if it is missing."""
		if self.questions_json.exists():
			return
		print(f"Downloading OpenApps questions JSON to {self.questions_json} …")
		self._download_atomic(_OPENAPPS_QUESTIONS_URL, self.questions_json)

	def _ensure_screenshots(self, dataset):
		"""Download any screenshots that are referenced in *dataset* but missing locally."""
		unique_paths = sorted({item["screenshot_path"] for item in dataset})
		for rel_str in unique_paths:
			rel = Path(rel_str)
			# Screenshot paths in the JSON are like:
			#   tests/generated_screenshots/default/todo.png
			# We save them under openapps/generated_screenshots/...
			parts = rel.parts
			if len(parts) >= 2 and parts[0] == "tests" and parts[1] == "generated_screenshots":
				dest = self.root / "openapps" / Path(*parts[1:])
			else:
				dest = self.root / "openapps" / rel

			if dest.exists():
				continue

			url = f"{_OPENAPPS_SCREENSHOTS_BASE_URL}/{rel.as_posix()}"
			print(f"Downloading {dest.name} from {url} …")
			self._download_atomic(url, dest)

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


class MMMUProDataset(Dataset):
	"""
	HuggingFace dataset wrapper for MMMU-Pro (MMMU/MMMU_Pro).

	A massive multi-discipline multimodal understanding benchmark with
	multiple-choice questions spanning 30+ academic subjects.

	Supported output formats (same API as OpenAppsDataset):
	- vllm: (image, question, choices, correct_letter, sample_id, split)
	- clip: (image, question, choices, correct_index, sample_id)
	"""

	def __init__(
		self,
		dataset_url="MMMU/MMMU_Pro",
		root=None,
		transform=None,
		max_num_samples=None,
		subset_kwargs=None,
		output_format="vllm",
		shuffle=False,
		random_seed=42,
		config_name="standard (4 options)",  # "standard (4 options)", "standard (10 options)", or "vision"
		split="test",
		download_num_workers=4,
		*args,
		**kwargs,
	):
		Dataset.__init__(self, *args, **kwargs)

		self.transform = transform
		self.output_format = output_format

		dataset = load_dataset(
			dataset_url,
			config_name,
			split=split,
			trust_remote_code=True,
		)

		if subset_kwargs is not None:
			for cond in subset_kwargs:
				key, val = next(iter(cond.items()))
				dataset = dataset.filter(
					lambda x, k=key, v=val: k in x and str(v).lower() in str(x[k]).lower(),
					num_proc=download_num_workers,
				)

		if shuffle:
			dataset = dataset.shuffle(seed=random_seed)

		if max_num_samples is not None:
			dataset = dataset.select(range(min(max_num_samples, len(dataset))))

		self.dataset = dataset

		# Pre-compute the max number of options so all samples pad to the same
		# length, which is required by PyTorch's default collate_fn.
		self.max_options = max(
			len(self._parse_options(item.get("options", [])))
			for item in self.dataset
		)

	def __len__(self):
		return len(self.dataset)

	def _parse_options(self, options_raw):
		"""Parse options, which may be a stringified list or an actual list."""
		if isinstance(options_raw, list):
			return options_raw
		try:
			parsed = ast.literal_eval(options_raw)
			if isinstance(parsed, list):
				return parsed
		except (ValueError, SyntaxError):
			pass
		return [str(options_raw)]

	def _get_image(self, item):
		"""Return the first non-null image from image_1 .. image_7."""
		for key in [f"image_{i}" for i in range(1, 8)]:
			img = item.get(key)
			if img is not None:
				return img.convert("RGB")
		raise ValueError(f"No image found for MMMU-Pro sample id={item.get('id')}")

	def __getitem__(self, index):
		item = self.dataset[index]

		image = self._get_image(item)
		if self.transform is not None:
			image = self.transform(image)

		question = item.get("question", "")
		options_list = self._parse_options(item.get("options", []))

		# Pad to max_options so every sample in a batch has the same list length,
		# which is required for PyTorch's default collate_fn.
		options_list = options_list + [""] * (self.max_options - len(options_list))

		letters = [chr(ord("A") + i) for i in range(len(options_list))]
		choices_dict = dict(zip(letters, options_list))
		# Exclude padding entries from the formatted prompt string.
		choices_letters = ", ".join([f"{k}) {v}" for k, v in choices_dict.items() if v != ""])

		correct_letter = str(item.get("answer", "A")).strip().upper()
		correct_index = max(0, ord(correct_letter) - ord("A")) if correct_letter else 0

		sample_id = str(item.get("id", f"mmmupro_{index}"))
		split = "|".join(
			[
				str(item.get("subject", "")),
				str(item.get("subfield", "")),
			]
		)

		return image, question, {'letters': choices_letters, 'list': options_list}, {'letters': correct_letter, 'index': correct_index}, sample_id, split
