"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Union
import os

import fire
import torch
import pandas as pd
from rich.progress import Progress
from torch.utils.data import Dataset

from unibench.benchmarks_zoo.registry import load_benchmark, register_benchmark
from unibench.benchmarks_zoo.wrappers.bechmark_handler import BenchmarkHandler
import unibench.benchmarks_zoo.benchmarks as benchmarks_module
import unibench.models_zoo.models as models_module

from unibench.benchmarks_zoo import list_benchmarks
from unibench.models_zoo import list_models
from unibench.models_zoo.registry import load_model, register_model
from unibench.models_zoo.wrappers.base import AbstractModel
from unibench.output import OutputHandler
from unibench.common_utils import (
    seed_everything,
    get_model_mappings,
    get_benchmark_mappings,
    df_to_table,
    OUTPUT_DIR,
    DATA_DIR,
)

from rich.console import Console
from rich.table import Table


class Evaluator(object):
    """
    The Evaluator class is responsible for evaluating machine learning models on various benchmarks.
    It provides methods to update the list of models and benchmarks, download benchmarks, add new models and benchmarks,
    generate aggregate results, and evaluate models.

    Attributes:
        seed (int): Random seed for reproducibility.
        num_workers (int): Number of workers for data loading.
        models (Union[List[str], str]): List of models to evaluate or "all" to evaluate all available models.
        benchmarks (Union[List[str], str]): List of benchmarks to evaluate or "all" to evaluate all available benchmarks.
        model_id (Union[int, None]): Specific model ID to evaluate.
        benchmark_id (Union[int, None]): Specific benchmark ID to evaluate.
        output_dir (str): Directory to save evaluation results.
        benchmarks_dir (str): Directory containing benchmark data.
        download_aggregate_precomputed (bool): Whether to download aggregate precomputed results. Used for minor analysis and fast loading.
        download_all_precomputed (bool): Whether to download all precomputed results. Used for slow loading and comprehensive analysis.

    Methods:
        update_benchmark_list(benchmarks, benchmark_id=None):
            Updates the list of benchmarks to evaluate.

        update_model_list(models, model_id=None):
            Updates the list of models to evaluate.

        download_benchmarks():
            Downloads the specified benchmarks.

        list_models():
            Lists all available models.

        add_benchmark(benchmark, handler, meta_data={}):
            Adds a new benchmark to the list of benchmarks.

        generate_aggregate_results():
            Generates aggregate results from the evaluation.

        list_benchmarks():
            Lists all available benchmarks.

        add_model(model, meta_data={}):
            Adds a new model to the list of models.

        show_results():
            Displays the evaluation results.

        evaluate(save_freq=1000, face_blur=False, device="cuda" if torch.cuda.is_available() else "cpu", batch_per_gpu=32):
            Evaluates the models on the benchmarks and saves the results.
    """

    def __init__(
        self,
        seed: int = 1337,
        num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK") or 96),
        models: Union[List[str], str] = "all",
        benchmarks: Union[List[str], str] = "all",
        model_id: Union[int, None] = None,
        benchmark_id: Union[int, None] = None,
        output_dir: str = OUTPUT_DIR,
        benchmarks_dir: str = DATA_DIR,
        download_aggregate_precomputed: bool = True,
        download_all_precomputed: bool = False,
    ):
        self.seed = seed
        self.num_workers = num_workers
        self.benchmarks_dir = benchmarks_dir
        self.output_dir = output_dir

        self.update_model_list(models, model_id)
        self.update_benchmark_list(benchmarks, benchmark_id)

        seed_everything(self.seed)
        self.outputhandler = OutputHandler(
            output_dir=output_dir,
            download_aggregate_precomputed=download_aggregate_precomputed,
            download_all_precomputed=download_all_precomputed,
        )

    def update_benchmark_list(
        self, benchmarks: Union[List[str], str], benchmark_id: Union[int, None] = None
    ):
        if isinstance(benchmarks, str):
            self.benchmarks = list_benchmarks(benchmarks)
        elif isinstance(benchmarks, list):
            self.benchmarks = benchmarks

        if benchmark_id is not None:
            self.benchmarks = [self.benchmarks[int(benchmark_id)]]
            print("Evaluating only benchmark {}".format(self.benchmarks[0]))

        assert (
            isinstance(self.benchmarks, list) and len(self.benchmarks) > 0
        ), "Please provide benchmarks to evaluate!"
        print("There are {} benchmarks to evaluate".format(len(self.benchmarks)))

    def update_model_list(
        self, models: Union[List[str], str], model_id: Union[int, None] = None
    ):
        if isinstance(models, str):
            self.models = list_models(models)
        elif isinstance(models, list):
            self.models = models

        if model_id is not None:
            self.models = [self.models[int(model_id)]]
            print("Evaluating only model {}".format(self.models[0]))

        assert (
            isinstance(self.models, list) and len(self.models) > 0
        ), "Please provide models to evaluate!"

        print("There are {} models to evaluate".format(len(self.models)))

    def download_benchmarks(self):
        for benchmark in self.benchmarks:
            print(f"Loading {benchmark}")
            load_benchmark(benchmark, root=self.benchmarks_dir)
            print(f"Done Loading {benchmark}")

    def list_models(self):
        model_mappings = get_model_mappings(None)
        # print(pd.DataFrame(model_mappings).transpose().to_markdown())
        Console().print(
            df_to_table(
                pd.DataFrame(model_mappings).transpose(),
                Table(show_header=True, header_style="bold magenta"),
                index_name="model_name",
            )
        )

    def add_benchmark(
        self, benchmark: Dataset, handler: BenchmarkHandler, meta_data={}
    ):
        def temp_func(benchmark_name, transform=None, **kwargs):
            bm = benchmark(transform=transform, **kwargs)

            return handler(benchmark=bm)

        benchmark_name = handler.keywords["benchmark_name"]

        temp_func.__name__ = benchmark_name
        register_benchmark("new_benchmark", meta_data)(temp_func)

        setattr(benchmarks_module, benchmark_name, temp_func)
        self.benchmarks.append(benchmark_name)

    def generate_aggregate_results(self):
        self.outputhandler.load_all_csv(self.models, self.benchmarks)
        self.outputhandler.generate_aggregate_results()
        print("Aggregate results generated")

    def list_benchmarks(self):
        benchmark_mappings = get_benchmark_mappings(None)
        Console().print(
            df_to_table(
                pd.DataFrame(benchmark_mappings).transpose(),
                Table(show_header=True, header_style="bold magenta"),
                index_name="benchmark_name",
            )
        )

    def add_model(
        self,
        model: AbstractModel,
        meta_data: dict = {},
    ):
        def temp_func(model_name, **kwargs):
            return model(**kwargs)

        model_name = model.keywords["model_name"]

        temp_func.__name__ = model_name
        register_model("new_model", meta_data)(temp_func)

        setattr(models_module, model_name, temp_func)
        self.models.append(model_name)

    def show_results(self):
        Console().print(
            df_to_table(
                self.outputhandler.print_dataframe(
                    **{"benchmark_name": self.benchmarks, "model_name": self.models}
                ).round(4)
                * 100,
                Table(show_header=True, header_style="bold magenta"),
                index_name="model_name",
            )
        )

    def evaluate(
        self,
        save_freq: int = 1000,
        face_blur: bool = False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_per_gpu: int = 32,
    ):
        """
        Evaluate models on benchmarks and return and saving the results.

        Args:
            save_freq (int): The frequency at which to save results. Defaults to 1000.
            face_blur (bool): Whether to use face blurring during evaluation. Defaults to False.
            device (str): The device to use for evaluation. Defaults to "cuda" if available otherwise "cpu".
            batch_per_gpu (int): The batch size per GPU. Defaults to 32.

        Returns:
            query results: The results of the query for the specified benchmarks and models.
        """
        with Progress(transient=True) as progress:
            pg_models = progress.add_task(
                "[green]Processing...", total=len(self.models)
            )
            pg_benchmarks = progress.add_task(
                "[green]Processing...", total=len(self.benchmarks)
            )
            pg_benchmark = progress.add_task(
                "[green]Processing...", total=len(self.benchmarks), visible=False
            )
            for model_name in self.models:
                progress.update(
                    pg_models, description=f"[green]Processing {model_name}..."
                )

                model = None

                for benchmark_name in self.benchmarks:
                    progress.update(
                        pg_benchmarks,
                        description=f"[green]Processing {benchmark_name}...",
                    )

                    number_entries = self.outputhandler.check_if_computed(
                        model_name=model_name, benchmark_name=benchmark_name
                    )

                    if number_entries == True:
                        progress.update(pg_benchmarks, advance=1, refresh=True)
                        continue

                    if model is None:
                        model = load_model(
                            model_name=model_name,
                            batch_per_gpu=batch_per_gpu,
                            face_blur=face_blur,
                            device=device,
                        )

                    if model is None:
                        raise ValueError(
                            f"{model_name} does not exist in the currently supported models"
                        )

                    dh = load_benchmark(
                        benchmark_name,
                        transform=model.get_preprocess_transforms(),
                        root=self.benchmarks_dir,
                    )

                    ds = dh.benchmark

                    dh.on_validation_start(model)

                    dl = torch.utils.data.DataLoader(
                        ds,
                        batch_size=model.get_batch_size(),
                        shuffle=False,
                        num_workers=self.num_workers,
                        pin_memory=True,
                    )

                    if number_entries == len(ds):
                        progress.update(pg_benchmarks, advance=1, refresh=True)
                        continue
                    elif number_entries > len(ds) or (0 < number_entries < len(ds)):
                        print(f"Reseting results for {model_name}")
                        self.outputhandler.delete_rows(
                            model_name=model_name, benchmark_name=benchmark_name
                        )

                    progress.update(
                        pg_benchmark, total=len(dl), completed=0, visible=True
                    )
                    for idx, batch in enumerate(dl):
                        progress.update(
                            pg_benchmark,
                            description=f"[green]Processing Batch #{idx}...",
                            visible=True,
                        )

                        for i, sample in enumerate(batch):
                            if isinstance(sample, torch.Tensor) and device == "cuda":
                                batch[i] = batch[i].to(device)

                        with torch.no_grad(), torch.amp.autocast('cuda'):
                            values_to_save = dh.eval_batch(model, batch)

                        self.outputhandler.add_values(
                            model_name=model_name, **values_to_save
                        )

                        if idx % save_freq == 0 and idx > 0:
                            self.outputhandler.save_csv(model_name, benchmark_name)
                        progress.update(pg_benchmark, advance=1)
                    progress.update(pg_benchmark, visible=False)
                    self.outputhandler.save_csv(model_name, benchmark_name)
                    self.outputhandler.save_aggregate_results(model_name, benchmark_name)
                    progress.update(pg_benchmarks, advance=1)
                progress.update(pg_models, advance=1)

        Console().print(
            df_to_table(
                self.outputhandler.print_dataframe(
                    **{"benchmark_name": self.benchmarks, "model_name": self.models}
                ).round(4)
                * 100,
                Table(show_header=True, header_style="bold magenta"),
                index_name="model_name",
            )
        )
        Console().print(f"The results are saved in {self.output_dir}")


def run():
    fire.Fire(Evaluator)
