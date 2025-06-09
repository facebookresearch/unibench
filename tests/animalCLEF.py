from functools import partial
from typing import List

import fire
import torch
from unibench import Evaluator
from unibench.benchmarks_zoo.handlers import ZeroShotBenchmarkHandler

import requests
import json
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from unibench.benchmarks_zoo.handlers.vllm_handlers import (
    TextClassificationBenchmarkHandler,
)

# URL of the JSON file
url = "https://raw.githubusercontent.com/LAION-AI/CLIP_benchmark/refs/heads/main/clip_benchmark/datasets/en_zeroshot_classification_templates.json"

# Step 1: Download and load the JSON
response = requests.get(url)
response.raise_for_status()
data = response.json()

# Step 2: Extract the ImageNet-1K prompts
templates = [x.replace("{c}", "{}") for x in data["imagenet1k"]]

data_path = "/storage/home/hcoda1/6/haltahan6/p-rmurty7-0/haider/.cache/datasets"


class FungiTastic(torch.nn.Module):
    """
    Dataset class for the FewShot subset of the Danish Fungi dataset (size 300, closed-set).

    This dataset loader supports training, validation, and testing splits, and provides
    convenient access to images, class IDs, and file paths. It also supports optional
    image transformations.
    """

    SPLIT2STR = {"train": "Train", "val": "Val", "test": "Test"}

    def __init__(self, root: str, split: str = "val", transform=None):
        """
        Initializes the FungiTastic dataset.

        Args:
            root (str): The root directory of the dataset.
            split (str, optional): The dataset split to use. Must be one of {'train', 'val', 'test'}.
                Defaults to 'val'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.split = split
        self.transform = transform
        self.df = self._get_df(root, split)

        assert "image_path" in self.df
        if self.split != "test":
            assert "category_id" in self.df
            self.n_classes = len(self.df["category_id"].unique())
            self.category_id2label = {
                k: v[0]
                for k, v in self.df.groupby("category_id")["species"]
                .unique()
                .to_dict()
                .items()
            }
            self.class_names = [
                self.category_id2label[x]
                for x in sorted(list(self.category_id2label.keys()))
            ]
            self.label2category_id = {
                idx: i
                for i, idx in enumerate(sorted(list(self.category_id2label.keys())))
            }

    def add_embeddings(self, embeddings: pd.DataFrame):
        """
        Updates the dataset instance with new embeddings.

        Args:
            embeddings (pd.DataFrame): A DataFrame containing an 'embedding' column.
                                       It must align with `self.df` in terms of indexing.
        """
        assert isinstance(
            embeddings, pd.DataFrame
        ), "Embeddings must be a pandas DataFrame."
        assert (
            "embedding" in embeddings.columns
        ), "Embeddings DataFrame must have an 'embedding' column."
        assert len(embeddings) == len(self.df), "Embeddings must match dataset length."

        self.df = pd.merge(self.df, embeddings, on="filename", how="inner")

    def get_embeddings_for_class(self, id):
        # return the embeddings for class class_idx
        class_idxs = self.df[self.df["category_id"] == id].index
        return self.df.iloc[class_idxs]["embedding"]

    @staticmethod
    def _get_df(data_path: str, split: str) -> pd.DataFrame:
        """
        Loads the dataset metadata as a pandas DataFrame.

        Args:
            data_path (str): The root directory where the dataset is stored.
            split (str): The dataset split to load. Must be one of {'train', 'val', 'test'}.

        Returns:
            pd.DataFrame: A DataFrame containing metadata and file paths for the split.
        """
        df_path = os.path.join(
            data_path,
            "metadata",
            "FungiTastic-FewShot",
            f"FungiTastic-FewShot-{FungiTastic.SPLIT2STR[split]}.csv",
        )
        df = pd.read_csv(df_path)
        df["image_path"] = df.filename.apply(
            lambda x: os.path.join(data_path, "FungiTastic-FewShot", split, "300p", x)
        )
        return df

    def __getitem__(self, idx: int):
        """
        Retrieves a single data sample by index.

        Args:
            idx (int): Index of the sample to retrieve.
            ret_image (bool, optional): Whether to explicitly return the image. Defaults to False.

        Returns:
            tuple:
                - If embeddings exist: (image?, embedding, category_id, file_path)
                - If no embeddings: (image, category_id, file_path) (original version)
        """
        file_path = (
            self.df["image_path"]
            .iloc[idx]
            .replace("FungiTastic-FewShot", "images/FungiTastic-FewShot")
        )

        if self.split != "test":
            category_id = self.df["category_id"].iloc[idx]
        else:
            category_id = None

        image = Image.open(file_path)

        if self.transform:
            image = self.transform(image)

        return image, self.label2category_id[category_id], file_path

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.df)

    def get_class_id(self, idx: int) -> int:
        """
        Returns the class ID of a specific sample.
        """
        return self.df["category_id"].iloc[idx]

    def get_category_idxs(self, category_id: int) -> List[int]:
        """
        Retrieves all indexes for a given category ID.
        """
        return self.df[self.df.category_id == category_id].index.tolist()


def main(model_id: int = 4, num_workers: int = 8):
    # Create benchmark using TestDataset with the correct paths
    benchmark = FungiTastic(
        root=data_path,
        split="val",  # Use 'test' split for evaluation
        transform=None,  # Add your transforms here if needed
    )

    # Get class names from the benchmark dataset
    class_names = benchmark.class_names

    benchmark = partial(
        FungiTastic,
        root=data_path,
        split="val",  # Use
    )

    eval = Evaluator(
        # model_id=model_id,
        num_workers=num_workers,
        models=[
            "llama_4_maverick",
            "llama_4_maverick_instruct",
            # 'llama_4_scout',
            # 'llama_4_scout_instruct',
            "llama_3_2_90b_vision_instruct",
            "llama_3_2_11b_vision_instruct",
            "llava_1_5_7b",
            "llava_next_llama_8b",
            "chameleon_7b",
            "paligemma_3b_224",
            "paligemma_3b_mix_224",
        ],
    )

    eval.add_benchmark(
        benchmark_name="fungi_tastic_2025",
        benchmark=benchmark,
        handlers={
            "text_classification": partial(
                TextClassificationBenchmarkHandler,
                class_names=class_names,
            ),
        },
        meta_data={
            "benchmark_type": "object recognition",
        },
    )
    
    # eval.add_benchmark(
    #     benchmark_name="fungi_tastic_2025_num_classes_2",
    #     benchmark=benchmark,
    #     handlers={
    #         "text_classification": partial(
    #             TextClassificationBenchmarkHandler,
    #             class_names=class_names,
    #             num_classes=2,
    #         ),
    #     },
    #     meta_data={
    #         "benchmark_type": "object recognition",
    #     },
    # )
    # eval.add_benchmark(
    #     benchmark_name="fungi_tastic_2025_num_classes_4",
    #     benchmark=benchmark,
    #     handlers={
    #         "text_classification": partial(
    #             TextClassificationBenchmarkHandler,
    #             class_names=class_names,
    #             num_classes=4,
    #         ),
    #     },
    #     meta_data={
    #         "benchmark_type": "object recognition",
    #     },
    # )
    # eval.add_benchmark(
    #     benchmark_name="fungi_tastic_2025_num_classes_8",
    #     benchmark=benchmark,
    #     handlers={
    #         "text_classification": partial(
    #             TextClassificationBenchmarkHandler,
    #             class_names=class_names,
    #             num_classes=8,
    #         ),
    #     },
    #     meta_data={
    #         "benchmark_type": "object recognition",
    #     },
    # )
    # eval.add_benchmark(
    #     benchmark_name="fungi_tastic_2025_num_classes_16",
    #     benchmark=benchmark,
    #     handlers={
    #         "text_classification": partial(
    #             TextClassificationBenchmarkHandler,
    #             class_names=class_names,
    #             num_classes=16,
    #         ),
    #     },
    #     meta_data={
    #         "benchmark_type": "object recognition",
    #     },
    # )
    # eval.add_benchmark(
    #     benchmark_name="fungi_tastic_2025_num_classes_32",
    #     benchmark=benchmark,
    #     handlers={
    #         "text_classification": partial(
    #             TextClassificationBenchmarkHandler,
    #             class_names=class_names,
    #             num_classes=32,
    #         ),
    #     },
    #     meta_data={
    #         "benchmark_type": "object recognition",
    #     },
    # )
    # eval.add_benchmark(
    #     benchmark_name="fungi_tastic_2025_num_classes_64",
    #     benchmark=benchmark,
    #     handlers={
    #         "text_classification": partial(
    #             TextClassificationBenchmarkHandler,
    #             class_names=class_names,
    #             num_classes=64,
    #         ),
    #     },
    #     meta_data={
    #         "benchmark_type": "object recognition",
    #     },
    # )
    # eval.add_benchmark(
    #     benchmark_name="fungi_tastic_2025_num_classes_128",
    #     benchmark=benchmark,
    #     handlers={
    #         "text_classification": partial(
    #             TextClassificationBenchmarkHandler,
    #             class_names=class_names,
    #             num_classes=128,
    #         ),
    #     },
    #     meta_data={
    #         "benchmark_type": "object recognition",
    #     },
    # )
    # eval.add_benchmark(
    #     benchmark_name="fungi_tastic_2025_num_classes_256",
    #     benchmark=benchmark,
    #     handlers={
    #         "text_classification": partial(
    #             TextClassificationBenchmarkHandler,
    #             class_names=class_names,
    #             num_classes=256,
    #         ),
    #     },
    #     meta_data={
    #         "benchmark_type": "object recognition",
    #     },
    # )
    # eval.add_benchmark(
    #     benchmark_name="fungi_tastic_2025_num_classes_512",
    #     benchmark=benchmark,
    #     handlers={
    #         "text_classification": partial(
    #             TextClassificationBenchmarkHandler,
    #             class_names=class_names,
    #             num_classes=512,
    #         ),
    #     },
    #     meta_data={
    #         "benchmark_type": "object recognition",
    #     },
    # )
    eval.update_benchmark_list(
        [
            "fungi_tastic_2025",
            "imagenet1k",
            # "fungi_tastic_2025_num_classes_2", "fungi_tastic_2025_num_classes_4",
            # "fungi_tastic_2025_num_classes_8", "fungi_tastic_2025_num_classes_16",
            # "fungi_tastic_2025_num_classes_32", "fungi_tastic_2025_num_classes_64",
            # "fungi_tastic_2025_num_classes_128", "fungi_tastic_2025_num_classes_256", "fungi_tastic_2025_num_classes_512"
        ]
    )
    eval.evaluate(batch_per_gpu=2)


if __name__ == "__main__":
    fire.Fire(main)
