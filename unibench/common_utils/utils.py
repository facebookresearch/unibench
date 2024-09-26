"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
import random
from typing import Optional
import numpy as np
import shutil
from huggingface_hub import hf_hub_download, snapshot_download
import torch

from ..benchmarks_zoo.registry import get_benchmark_info
from ..benchmarks_zoo import benchmarks, list_benchmarks
from ..models_zoo.registry import get_model_info
from ..models_zoo import models, list_models

import pandas as pd
from rich.table import Table
from rich import box


def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Table,
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(pandas_dataframe.index.to_list()[index])] if show_index else []
        if isinstance(value_list[0], float):
            row += [str(round(float(x), 2)) for x in value_list]
        else:
            row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    rich_table.row_styles = ["none", "dim"]
    rich_table.box = box.SIMPLE_HEAD

    return rich_table


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_benchmark_mappings(axis, benchmarks=None):
    if benchmarks is None:
        benchmarks = list_benchmarks()
    benchmark_mappings = {}
    for benchmark in benchmarks:
        if axis is None:
            benchmark_mappings[benchmark] = get_benchmark_info(benchmark)
        else:
            benchmark_mappings[benchmark] = get_benchmark_info(benchmark)[axis]
    return benchmark_mappings


def get_model_mappings(axis, models=None):
    if models is None:
        models = list_models()
    model_mappings = {}
    for model in models:
        if axis is None:
            model_mappings[model] = get_model_info(model)
        else:
            model_mappings[model] = get_model_info(model)[axis]
    return model_mappings


def download_only_aggregate(output_dir):
    print(f"Downloading only aggregate results...{output_dir}")
    hf_hub_download(
        repo_id="haideraltahan/unibench",
        cache_dir=output_dir,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        repo_type="dataset",
        filename="aggregate.f",
    )

def download_all_results(output_dir):
    print(f"Downloading all results...{output_dir}")
    snapshot_download(
        repo_id="haideraltahan/unibench",
        cache_dir=output_dir,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        repo_type="dataset",
    )
