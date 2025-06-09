# Evaluating New Model Versions with UniBench

This guide explains how to evaluate new models using the UniBench framework. You can test any HuggingFace model or custom model against all available benchmarks in the UniBench suite.

## Installation

Install UniBench with the new model evaluation capabilities:

```bash
pip install unibench[new_model]
```

## Core Functionality

The new model evaluation feature allows you to:

- **Evaluate existing HuggingFace models**: Test any pre-trained model from the HuggingFace model hub
- **Run comprehensive benchmarks**: Evaluate your model against all available UniBench datasets
- **Custom model integration**: Add your own models using the UniBench wrapper system

## Model Evaluation Example

### Loading and Evaluating a New Model

The following example demonstrates how to evaluate the ViTamin-L model on UniBench benchmarks. Run this code in a Jupyter notebook or Python file:

```python
from functools import partial
from unibench import Evaluator
from unibench.models_zoo.wrappers.clip import ClipModel
import open_clip

# Load the pre-trained model and tokenizer
model, _, _ = open_clip.create_model_and_transforms(
    "ViTamin-L", pretrained="datacomp1b"
)

tokenizer = open_clip.get_tokenizer("ViTamin-L")

# Wrap the model for UniBench compatibility
model = partial(
    ClipModel,
    model=model,
    model_name="vitamin_l_comp1b",
    tokenizer=tokenizer,
    input_resolution=model.visual.image_size[0],
    logit_scale=model.logit_scale,
)

# Initialize the evaluator
eval = Evaluator()

# Add your model to the evaluation pipeline
eval.add_model(model=model)

# Specify which benchmarks to run (example: MNIST)
eval.update_benchmark_list(["mnist"])

# Specify which models to evaluate
eval.update_model_list(["vitamin_l_comp1b"])

# Run the evaluation
eval.evaluate()
```

### Evaluation Process

1. **Model Loading**: Load your model using the appropriate framework (OpenCLIP, HuggingFace, etc.)
2. **Model Wrapping**: Wrap your model using UniBench's model wrappers for compatibility
3. **Benchmark Selection**: Choose which benchmarks to evaluate against
4. **Evaluation Execution**: Run the comprehensive evaluation pipeline

## Dependencies

UniBench with new model evaluation installs the following additional libraries:

| Library Name | Purpose |
|--------------|---------|
| `fire` | Handles command line interface and function calls |
| `pandas` | Loads and manipulates benchmark results data |
| `rich` | Provides enhanced progress bars and terminal formatting |
| `huggingface_hub` | Downloads pre-computed results and model access |
| `oslo.concurrency` | Manages parallel processing during evaluation |
| `pyarrow` | Required for reading/writing feather file format |
| `torch`, `torchvision` | PyTorch framework for deep learning models |
| `datasets` | HuggingFace datasets library for benchmark data loading |

## Next Steps

After evaluation, you can:
- Compare your model's results with existing benchmarks
- Visualize performance across different datasets
- Export results for further analysis
- Contribute your model's results back to the UniBench database
