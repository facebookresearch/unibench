"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict

_benchmark_registry = defaultdict(list)  # mapping of benchmark names to entrypoint fns
_benchmark_info_registry = defaultdict(defaultdict)


def register_benchmark(names, info=None):
    def inner_decorator(fn):
        # add entries to registry dict/sets
        benchmark_name = fn.__name__
        if isinstance(names, str):
            _benchmark_registry[names].append(benchmark_name)
        else:
            for name in names:
                _benchmark_registry[name].append(benchmark_name)

        if info is not None:
            _benchmark_info_registry[benchmark_name] = info

        # register_benchmark_handler(benchmark_name, fn)

        return fn

    return inner_decorator


# def register_benchmark_handler(benchmark_name, benchmark_handler):
#     _benchmark_handlers[benchmark_name] = benchmark_handler


# def get_benchmark_handler(benchmark_name):
#     return _benchmark_handlers[benchmark_name]

def load_benchmark(benchmark_name, **kwargs):
    import unibench.benchmarks_zoo.benchmarks as benchmarks
    supported_benchmarks = list_benchmarks("all")
    module_name = supported_benchmarks.index(benchmark_name)
    if module_name is None:
        raise NameError(
            f"Benchmark {benchmark_name} is not supported, "
            f"please select from {list(supported_benchmarks.keys())}"
        )

    return eval(f"benchmarks.{benchmark_name}")(benchmark_name, **kwargs)


def get_benchmark_info(benchmark_name):
    return _benchmark_info_registry[benchmark_name]


def get_benchmark_types():
    return list(_benchmark_registry.keys())


def list_benchmarks(framework="all"):
    """Return list of available benchmark names, sorted alphabetically"""
    r = []
    if framework == "all":
        for _, v in _benchmark_registry.items():
            r.extend(v)
    else:
        r = _benchmark_registry[framework]
    if isinstance(r, str):
        return [r]
    return sorted(list(r))
