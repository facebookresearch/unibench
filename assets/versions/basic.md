# UniBench - Minimal Version Guide

UniBench is a unified benchmarking framework that allows you to download, analyze, and visualize machine learning benchmark results across different models and datasets.

## Installation

Install UniBench directly from the GitHub repository using pip:

```bash
pip install unibench
```

## Core Functionality

UniBench provides three main capabilities:

- **Download existing results**: Access pre-computed benchmark results from various models and datasets
- **Visualize downloaded results**: Create charts and graphs to analyze performance metrics  
- **Get DataFrames of results**: Load benchmark data into pandas DataFrames for custom analysis

## Basic Commands

### List Available Models
View all supported models in the UniBench database:

```bash
unibench list_models
```

### List Available Benchmarks
View all supported benchmarks and datasets:

```bash
unibench list_benchmarks
```

## Loading Results as DataFrames

### Aggregate Results
Load summarized results across all models and benchmarks. Run this in a Jupyter notebook or Python file:

```python
from unibench.output import OutputHandler

# Initialize handler and download aggregate results
outputhandler = OutputHandler(download_aggregate_precomputed=True)
results = outputhandler.get_aggregate_results()
print(results)
```

### Detailed Results
Load complete results for specific models and benchmarks with full granularity:

```python
from unibench.output import OutputHandler

# Initialize handler and download all precomputed results
outputhandler = OutputHandler(download_all_precomputed=True)

# Load specific model and benchmark combinations
outputhandler.load_all_csv(
    model_name=['siglip2_so400_16_512'],
    benchmark_name=['imagenet1k'],
)

# Query and retrieve the filtered results
results = outputhandler.query()
print(results)
```

## Dependencies

UniBench automatically installs the following libraries and their purposes:

| Library Name | Purpose |
|--------------|---------|
| `fire` | Handles command line interface and function calls |
| `pandas` | Loads and manipulates benchmark results data |
| `rich` | Provides enhanced progress bars and terminal formatting |
| `huggingface_hub` | Downloads pre-computed results from Hugging Face |
| `oslo.concurrency` | Manages parallel processing for loading multiple results |
| `pyarrow` | Required for reading feather file format (results storage format) |
