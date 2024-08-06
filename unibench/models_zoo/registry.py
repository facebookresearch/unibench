"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from collections import defaultdict

_model_registry = defaultdict(list)  # mapping of model names to entrypoint fns
_model_info_registry = defaultdict(defaultdict)


def register_model(frameworks, info):
    def inner_decorator(fn):
        # add entries to registry dict/sets
        model_name = fn.__name__
        if isinstance(frameworks, str):
            _model_registry[frameworks].append(model_name)
        else:
            for framework in frameworks:
                _model_registry[framework].append(model_name)

        if info is not None:
            _model_info_registry[model_name] = info

        return fn

    return inner_decorator


def get_model_info(model_name):
    return _model_info_registry[model_name]


def load_model(model_name, **kwargs):
    import unibench.models_zoo.models as models
    if model_name in list_models("all"):
        model = eval(f"models.{model_name}")(model_name, **kwargs)
    else:
        return None
    return model


def list_models(framework="all"):
    """Return list of available model names, sorted alphabetically"""
    r = []
    if framework == "all":
        for _, v in _model_registry.items():
            r.extend(v)
    else:
        r = _model_registry[framework]
    return sorted(list(r))
