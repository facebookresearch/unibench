
|      ![header](./assets/header.png "header")       |
| :------------------------------------------------: |
| *[[Arxiv link](https://arxiv.org/abs/2408.04810)]* |

<p align="center">
  <a href="#getting-started">Getting Started</a> •
  <a href="#usage">Usage</a> •
  <a href="#sparkles-supported-models-and-benchmarks">Benchmarks & Models</a> •
  <a href="#credit_card-citation">Credit & Citation</a>
</p>

# Vision-Language Model Evaluation Repository

This repository is designed to simplify the evaluation process of vision-language models. It provides a comprehensive set of tools and scripts for evaluating VLM models and benchmarks. We offer 60+ VLMs, inclusive of recent large-scale models like EVACLIP, with scales reaching up to 4.3B parameters and 12.8B training samples. Additionally, we provide implementations for 40+ evaluation benchmarks.

## News and Updates

For the latest news and updates, see the snippet below.

#### April 15, 2025 - v0.4.0
- Removed FaceNet from required libraries.
- Added SigLIP2 models
- Added bivlc benchmark
- Created benchmark_builder for future benchmark implementations
- Added News & Updates section in README
- Fixed Sun397 benchmark
  
For full details, refer to the [UPDATES.md](./assets/UPDATES.md) file.

## Coming Soon

- [ ] L-VLM (e.g. PaliGemma, LlavaNext)

## Getting Started

Install the package: 
```
pip install unibench -U
```

<details > 
<summary><b>[option 2]</b> Install Dependencies
</summary>

1. Install the necessary dependencies by:
    - Option 1, creating a new conda env: `conda env create -f environment.yml`
    - Option 2, updating your conda env with required libraries: `conda env update --file environment.yml --prune`
2. Activate the environment: `conda activate unibench`
3. Install Spacy english language model: `python -m spacy download en_core_web_sm`
4. Install the package: `pip install git+https://github.com/facebookresearch/unibench`
  
</details> 

## Usage

### Print out Results from Evaluated Models

The following command will print the results of the evaluations on all benchmarks and models:

```console
unibench show_results
```

### Run Evaluation using Command Line

The following command will run the evaluation on all benchmarks and models:

```console
unibench evaluate
```

### Run Evaluation using Custom Script

The following command will run the evaluation on all benchmarks and models:

```python
import unibench as vlm

evaluator = vlm.Evaluator()
evaluator.evaluate()
```

### Arguments for Evaluation

`evaluate` function takes the following arguments:

```console
Args:
    save_freq (int): The frequency at which to save results. Defaults to 1000.
    face_blur (bool): Whether to use face blurring during evaluation. Defaults to False.
    device (str): The device to use for evaluation. Defaults to "cuda" if available otherwise "cpu".
    batch_per_gpu (int): Evaluation batch size per GPU. Defaults to 32.
```


The `Evaluator` class takes the following arguments:

```console
Args:
    seed (int): Random seed for reproducibility.
    num_workers (int): Number of workers for data loading.
    models (Union[List[str], str]): List of models to evaluate or "all" to evaluate all available models.
    benchmarks (Union[List[str], str]): List of benchmarks to evaluate or "all" to evaluate all available benchmarks.
    model_id (Union[int, None]): Specific model ID to evaluate.
    benchmark_id (Union[int, None]): Specific benchmark ID to evaluate.
    output_dir (str): Directory to save evaluation results.
    benchmarks_dir (str): Directory containing benchmark data.
    download_aggregate_precomputed (bool): Whether to download aggregate precomputed results.
    download_all_precomputed (bool): Whether to download all precomputed results.
```

### Example

The following command will run the evaluation for openclip_vitB32 trained on metaclip400m and CLIP ResNet50 on vg_relation,clevr_distance,fer2013,pcam,imageneta benchmarks:

```console
unibench evaluate --models=[openclip_vitB32_metaclip_400m,clip_resnet50] --benchmarks=[vg_relation,clevr_distance,fer2013,pcam,imageneta]
```

In addition to saving the results in `~/.cache/unibench`, the output would be a summary of the evaluation results:

```console
  model_name                      non-natural images   reasoning   relation   robustness  
 ──────────────────────────────────────────────────────────────────────────────────────── 
  clip_resnet50                   63.95                 14.89       54.13      23.27       
  openclip_vitB32_metaclip_400m   63.87                 19.46       51.54      28.71   
```

## Supported Models and benchmarks
Full list of models and benchmarks are available in the [models_zoo](unibench/models_zoo/README.md) and [benchmarks_zoo](unibench/benchmarks_zoo/README.md). You are also able to run the following commands:

```console
unibench list_models
# or
unibench list_benchmarks
```

### Sample Models


|                    | Dataset Size (Million) | Number of Parameters (Million) | Learning Objective | Architecture | Model Name    |
| :----------------- | ---------------------: | -----------------------------: | :----------------- | :----------- | :------------ |
| blip_vitB16_14m    |                     14 |                             86 | BLIP               | vit          | BLIP ViT B 16 |
| blip_vitL16_129m   |                    129 |                            307 | BLIP               | vit          | BLIP ViT L 16 |
| blip_vitB16_129m   |                    129 |                             86 | BLIP               | vit          | BLIP ViT B 16 |
| blip_vitB16_coco   |                    129 |                             86 | BLIP               | vit          | BLIP ViT B 16 |
| blip_vitB16_flickr |                    129 |                             86 | BLIP               | vit          | BLIP ViT B 16 |


### Sample benchmarks
|                | benchmark | benchmark_type |
| :------------- | :-------- | :------------- |
| clevr_distance | zero-shot | vtab           |
| fgvc_aircraft  | zero-shot | transfer       |
| objectnet      | zero-shot | robustness     |
| winoground     | relation  | relation       |
| imagenetc      | zero-shot | corruption     |

### benchmarks Overview

| benchmark type | number of benchmarks |
| :------------- | :------------------: |
| ImageNet       |          1           |
| vtab           |          18          |
| transfer       |          7           |
| robustness     |          6           |
| relation       |          6           |
| corruption     |          1           |

<!-- ## :sparkles: Features/Objectives

- Ease-of-use VLM evaluation repo. 
- Evaluate existing and future VLMs on benchmarks without extensive code
- Evaluate on existing and future benchmarks without extensive code


## :pencil2: Repository Structure


The repository is organized into the following directories:

- `common_utils`: Scripts for common utilities used throughout the repository.
- `benchmarks_zoo`: Scripts for loading various benchmarks.
- `models_zoo`: Scripts for loading various models.
- `slurm_scripts`: Scripts for running the evaluation in parallel on a SLURM cluster.

- `main.py`: Script for running the evaluation.
- `plotter.py`: Script for plotting the results.
- `output.py`: Script for saving the results. -->



### How results are saved

For each model, the results are saved in the output directory defined in constants: `~./.cache/unibench/outputs`.


### Add new Benchmark

To add new benchmark, you can simply inherit from the `torch.utils.data.Dataset` class and implement the `__getitem__`, and `__len__` methods. For example, here is how to add ImageNetA as a new benchmark:
    
```python
from functools import partial
from unibench import Evaluator
from unibench.benchmarks_zoo import ZeroShotBenchmarkHandler
from torchvision.datasets import FashionMNIST

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

templates = ["an image of {}"]

benchmark = partial(
    FashionMNIST, root="/fsx-robust/haideraltahan", train=False, download=True
)
handler = partial(
    ZeroShotBenchmarkHandler,
    benchmark_name="fashion_mnist_new",
    classes=class_names,
    templates=templates,
)


eval = Evaluator()

eval.add_benchmark(
    benchmark,
    handler,
    meta_data={
        "benchmark_type": "object recognition",
    },
)
eval.update_benchmark_list(["fashion_mnist_new"])
eval.update_model_list(["blip_vitB16_129m"])
eval.evaluate()
```

### Add new Model

The most important compontent of adding a new model is creating or using pre-existing `AbstractModel` and implementing `compute_zeroshot_weights`, `get_image_embeddings`, and `get_text_embeddings`, similar to how `ClipModel` works:

```python
class ClipModel(AbstractModel):
    def __init__(
        self,
        model,
        model_name,
        **kwargs,
    ):
        super(ClipModel, self).__init__(model, model_name, **kwargs)

    def compute_zeroshot_weights(self):
        zeroshot_weights = []
        for class_name in self.classes:
            texts = [template.format(class_name) for template in self.templates]

            class_embedding = self.get_text_embeddings(texts)

            class_embedding = class_embedding.mean(dim=0)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)

            zeroshot_weights.append(class_embedding)
        self.zeroshot_weights = torch.stack(zeroshot_weights).T

    @torch.no_grad()
    def get_image_embeddings(self, images):
        image_features = self.model.encode_image(images.to(self.device))
        image_features /= image_features.norm(dim=1, keepdim=True)
        return image_features.unsqueeze(1)

    @torch.no_grad()
    def get_text_embeddings(self, captions):
        if (
            "truncate" in inspect.getfullargspec(self.tokenizer.__call__)[0]
            or "truncate" in inspect.getfullargspec(self.tokenizer)[0]
        ):
            caption_tokens = self.tokenizer(
                captions, context_length=self.context_length, truncate=True
            ).to(self.device)
        else:
            caption_tokens = self.tokenizer(
                captions, context_length=self.context_length
            ).to(self.device)

        caption_embeddings = self.model.encode_text(caption_tokens)
        caption_embeddings /= caption_embeddings.norm(dim=-1, keepdim=True)

        return caption_embeddings

```

Using the following class, we can then add models to the list of models. Here we have an example of adding and evaluating `ViTamin-L`.

```python
from functools import partial
from io import open_code
from unibench import Evaluator
from unibench.models_zoo.wrappers.clip import ClipModel
import open_clip

model, _, _ = open_clip.create_model_and_transforms(
    "ViTamin-L", pretrained="datacomp1b"
)

tokenizer = open_clip.get_tokenizer("ViTamin-L")

model = partial(
    ClipModel,
    model=model,
    model_name="vitamin_l_comp1b",
    tokenizer=tokenizer,
    input_resolution=model.visual.image_size[0],
    logit_scale=model.logit_scale,
)


eval = Evaluator(benchmarks_dir="/fsx-checkpoints/haideraltahan/.cache/unibench/data")

eval.add_model(model=model)
eval.update_benchmark_list(["imagenet1k"])
eval.update_model_list(["vitamin_l_comp1b"])
eval.evaluate()

```


## Contributing

[Contributions](CONTRIBUTING.md) (e.g. adding new benchmarks/models), issues, and feature requests are welcome! For any changes, please open an issue first to discuss what you would like to change or improve.


## License

The majority of UniBench is licensed under [CC-BY-NC](LICENSE), however portions of the project are available under separate license terms: 

| License            |                                                 Libraries                                                 |
| :----------------- | :-------------------------------------------------------------------------------------------------------: |
| MIT license        |                           zipp, tabulate, rich, openai-clip, latextable, gdown                            |
| Apache 2.0 license | transformers, timm, opencv-python, open-clip-torch, ftfy, fire, debtcollector, datasets, oslo.concurrency |
| BSD license        |     torchvision, torch, seaborn, scipy, scikit-learn, fairscale, cycler, contourpy, click, GitPython      |

## Citation

If you use this repository in your research, please cite it as follows:

```bibtex
@inproceedings{altahan2024unibenchvisualreasoningrequires,
      title={UniBench: Visual Reasoning Requires Rethinking Vision-Language Beyond Scaling}, 
      author={Haider Al-Tahan and Quentin Garrido and Randall Balestriero and Diane Bouchacourt and Caner Hazirbas and Mark Ibrahim},
      year={2024},
      eprint={2408.04810},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.04810}, 
}
```

## Recognition

Library structure was inspired by [Robert Geirhos](https://github.com/rgeirhos)'s work https://github.com/bethgelab/model-vs-human
