"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os.path
from pathlib import Path

import pandas as pd
import torch

from unibench.common_utils.utils import download_all_results, download_only_aggregate
from oslo_concurrency import lockutils

from .benchmarks_zoo.registry import get_benchmark_info, list_benchmarks

from .common_utils.constants import OUTPUT_DIR, LOCK_DIR


class OutputHandler(object):
    def __init__(
        self,
        load_all_csv=False,
        round_values=4,
        output_dir=OUTPUT_DIR,
        download_all_precomputed=False,
        download_aggregate_precomputed=False,
    ):
        self.output_dir = Path(output_dir)
        if download_all_precomputed:
            download_all_results(self.output_dir)
        elif download_aggregate_precomputed:
            download_only_aggregate(self.output_dir)
        self.round_values = round_values
        self.reset_local_csv()
        lockutils.set_defaults(lock_path=LOCK_DIR)
        self.load_aggregate_results()
        if load_all_csv:
            self.load_all_csv()

    def reset_local_csv(self):
        self._local_csv = pd.DataFrame()

    def check_if_computed(self, model_name, benchmark_name, **kwargs):
        self.load_aggregate_results()
        res = self.query(
            df=self._aggregate,
            **{"model_name": model_name, "benchmark_name": benchmark_name}
        )
        if len(res) >= 1:
            return True

        self.load_csv(model_name, benchmark_name)
        return len(self.query(**kwargs))

    def load_all_csvs(self, model_names):
        self._model_csv = pd.DataFrame()
        dfs = []
        for model in model_names:
            model_folder = self.output_dir.joinpath(model)
            for benchmark_file in os.listdir(model_folder):
                file = model_folder.joinpath(benchmark_file)
                if ".f" in file.suffix and file.exists():
                    try:
                        dfs.append(pd.read_feather(file))
                    except:
                        print("Error reading file: ", file)
                else:
                    print("File not found: ", file)

        self._model_csv = pd.concat(dfs).reset_index(drop=True).round(self.round_values)

    def load_all_csv(self, model_name, benchmark_name):
        self._model_csv = pd.DataFrame()
        dfs = []
        for model in model_name:
            model_folder = self.output_dir.joinpath(model)
            for benchmark in benchmark_name:
                file = model_folder.joinpath(benchmark + ".f")
                if file.exists():
                    try:
                        dfs.append(pd.read_feather(file))
                    except:
                        print("Error reading file: ", file)
                else:
                    print("File not found: ", file)

        self._model_csv = pd.concat(dfs).reset_index(drop=True).round(self.round_values)

    def load_csv(self, model_name, benchmark_name):
        file_name = str(
            self.output_dir.joinpath(model_name).joinpath(benchmark_name + ".f")
        )

        # Load the csv if it exists
        if os.path.exists(file_name):
            self._model_csv = pd.read_feather(file_name)
        else:
            self._model_csv = pd.DataFrame()

    def load_model_csvs(self, model_name, use_cols=None):
        model_folder = self.output_dir.joinpath(model_name)

        self._model_csv = pd.DataFrame()
        dfs = []
        for file in os.listdir(model_folder):
            if file.endswith(".f"):
                dfs.append(
                    pd.read_feather(model_folder.joinpath(file), columns=use_cols)
                )

        self._model_csv = pd.concat(dfs).reset_index(drop=True).round(self.round_values)

    def get_csv(self):
        return pd.concat([self._local_csv, self._model_csv])

    def add_values(self, **kwargs):
        for k in kwargs.keys():
            if isinstance(kwargs[k], torch.Tensor):
                kwargs[k] = kwargs[k].cpu().squeeze().tolist()
        self._local_csv = pd.concat([self._local_csv, pd.DataFrame(kwargs)])

    def query(self, df=None, **kwargs):
        if df is None:
            df = self._model_csv
        if len(kwargs) == 0:
            return df

        mask = pd.Series([True] * len(df))

        for k, v in kwargs.items():
            if isinstance(v, list):
                mask &= df[k].isin(v)
            else:
                mask &= (df[k] == v)

        return df[mask]

    def delete_rows(self, model_name, benchmark_name, **kwargs):
        # file_name = str(OUTPUT_DIR.joinpath(model_name + ".f"))
        self.output_dir.joinpath(model_name).mkdir(parents=True, exist_ok=True)
        file_name = str(
            self.output_dir.joinpath(model_name).joinpath(benchmark_name + ".f")
        )

        # Load the csv if it exists
        if os.path.exists(file_name):
            self._model_csv = pd.read_feather(file_name)
        else:
            pass

        self._model_csv.drop(self.query(**kwargs).index, inplace=True)
        self._model_csv = self._model_csv.reset_index(drop=True)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._model_csv.to_feather(file_name)

    def _get_benchmark_mappings(self, axis):
        benchmark_mappings = {}
        for benchmark in list_benchmarks():
            if axis is None:
                benchmark_mappings[benchmark] = get_benchmark_info(benchmark)
            else:
                benchmark_mappings[benchmark] = get_benchmark_info(benchmark)[axis]
        return benchmark_mappings

    @lockutils.synchronized(name="aggregate", external=True, fair=True)
    def load_aggregate_results(self):
        file = self.output_dir.joinpath("aggregate.f")
        if file.exists():
            self._aggregate = pd.read_feather(file)

    @lockutils.synchronized(name="aggregate", external=True, fair=True)
    def save_aggregate_results(self, model_name, benchmark_name):
        file_dir = self.output_dir.joinpath("aggregate.f")
        if file_dir.exists():
            self._aggregate = pd.read_feather(file_dir)

        df = self.query(
            self._model_csv,
            **{"model_name": [model_name], "benchmark_name": [benchmark_name]}
        )

        df = (
            df.groupby(["model_name", "benchmark_name"])["correctness"]
            .mean()
            .reset_index()
        )

        df = (
            pd.concat([self._aggregate, df])
            .drop_duplicates(subset=["model_name", "benchmark_name"], keep="last")
            .reset_index(drop=True)
        )

        df.to_feather(file_dir)

    def print_dataframe(self, **kwargs):
        self.load_aggregate_results()
        df = self.query(df=self._aggregate, **kwargs)
        benchmark_mappings = self._get_benchmark_mappings("benchmark_type")
        df["benchmark_type"] = df["benchmark_name"].map(benchmark_mappings)
        df = (
            df.groupby(["model_name", "benchmark_name", "benchmark_type"])[
                "correctness"
            ]
            .mean()
            .reset_index()
        )

        df = (
            df.groupby(["model_name", "benchmark_type"])["correctness"]
            .mean()
            .reset_index()
        )
        return df.pivot(
            index="model_name", columns="benchmark_type", values="correctness"
        )

    def save_csv(self, model_name, benchmark_name):
        self.output_dir.joinpath(model_name).mkdir(parents=True, exist_ok=True)
        file_name = str(
            self.output_dir.joinpath(model_name).joinpath(benchmark_name + ".f")
        )

        # Load the csv if it exists
        if os.path.exists(file_name):
            self._model_csv = pd.read_feather(file_name)
        else:
            self._model_csv = pd.DataFrame()

        # Add the local csv to the model csv
        self._model_csv = (
            pd.concat(
                [self._model_csv, self._local_csv.reset_index(drop=True)],
                axis=0,
                ignore_index=True,
            )
            .round(self.round_values)
            .reset_index(drop=True)
        )

        # Save the model csv
        self._model_csv.to_feather(file_name)
        self.reset_local_csv()