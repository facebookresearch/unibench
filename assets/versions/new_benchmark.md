# Evaluating New Benchmark Versions with UniBench

This guide explains how to add and evaluate new benchmarks using the UniBench framework. You can integrate custom datasets and evaluate all available models against your new benchmark.

## Installation

Install UniBench with the new benchmark evaluation capabilities:

```bash
pip install unibench[new_benchmark]
```

## Core Functionality

The new benchmark evaluation feature allows you to:

- **Add custom benchmarks**: Integrate your own datasets into the UniBench evaluation pipeline
- **Evaluate all models**: Test all available UniBench models against your new benchmark
- **Flexible benchmark types**: Support for classification, object detection, and other vision tasks

## Benchmark Integration Example

### Adding and Evaluating a New Benchmark

The following example demonstrates how to add FashionMNIST as a new benchmark and evaluate models against it. Run this code in a Jupyter notebook or Python file:

```python
from functools import partial
from unibench import Evaluator
from unibench.benchmarks_zoo.wrappers import ZeroShotBenchmarkHandler
from torchvision.datasets import FashionMNIST

# Define class names for FashionMNIST dataset
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Define templates for zero-shot classification
templates = ["an image of {}"]

# Create benchmark dataset loader
benchmark = partial(
    FashionMNIST, train=False, download=True
)

# Create benchmark handler with metadata
handler = partial(
    ZeroShotBenchmarkHandler,
    benchmark_name="fashion_mnist_new",
    classes=class_names,
    templates=templates,
)

# Initialize the evaluator
eval = Evaluator()

# Add your benchmark to the evaluation pipeline
eval.add_benchmark(
    benchmark,
    handler,
    meta_data={
        "benchmark_type": "object recognition",
    },
)

# Specify which benchmarks to run
eval.update_benchmark_list(["fashion_mnist_new"])

# Run the evaluation across all available models
eval.evaluate()
```

### Benchmark Integration Process

1. **Dataset Definition**: Define your dataset using standard PyTorch/torchvision datasets or custom loaders
2. **Class Mapping**: Specify class names and templates for zero-shot evaluation
3. **Handler Creation**: Create a benchmark handler that interfaces with UniBench's evaluation system
4. **Metadata Addition**: Add relevant metadata about your benchmark type and characteristics
5. **Evaluation Execution**: Run comprehensive evaluation across all models in UniBench

## Benchmark Components

### Required Elements

- **Dataset Loader**: PyTorch-compatible dataset class
- **Class Names**: List of human-readable class labels
- **Templates**: Text templates for zero-shot classification (e.g., "a photo of {}")
- **Handler**: UniBench wrapper that manages evaluation logic
- **Metadata**: Information about benchmark type and characteristics

## Dependencies

UniBench with new benchmark evaluation installs the following libraries:

| Library Name | Purpose |
|--------------|---------|
| `fire` | Handles command line interface and function calls |
| `pandas` | Loads and manipulates benchmark results data |
| `rich` | Provides enhanced progress bars and terminal formatting |
| `huggingface_hub` | Downloads pre-computed results and model access |
| `oslo.concurrency` | Manages parallel processing during evaluation |
| `pyarrow` | Required for reading/writing feather file format |
| `torch`, `torchvision` | PyTorch framework and computer vision utilities |
| `open_clip_torch` | OpenCLIP implementation for CLIP models |
| `openai-clip` | Original OpenAI CLIP implementation |
| `timm` | Dependency for various vision models in open_clip_torch |
| `transformers` | HuggingFace transformers for loading models and tokenizers |
| `GitPython` | Git operations for loading certain models (e.g., BLIP) |
| `fairscale` | Distributed training utilities required by some models |

## Supported Benchmark Types

- **Object Recognition**: Image classification tasks
- **Object Detection**: Bounding box detection tasks
- **Semantic Segmentation**: Pixel-level classification
- **Zero-Shot Classification**: Template-based classification without training
- **Custom Tasks**: Flexible framework for domain-specific evaluations

## Next Steps

After adding your benchmark, you can:
- Compare performance across different model architectures
- Analyze model strengths and weaknesses on your specific dataset
- Contribute your benchmark to the UniBench community
- Generate comprehensive evaluation reports and visualizations
