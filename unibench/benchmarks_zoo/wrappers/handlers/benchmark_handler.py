"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import abstractmethod


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
