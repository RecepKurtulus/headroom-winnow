# Installation

Winnow is a Python package that registers itself with Headroom through the
`headroom.compressor` entry point. It needs Python 3.10+ and Headroom 0.39.

```bash
pip install "headroom-winnow[model]"
```

The `model` extra pulls in `torch` and `transformers`. Without it the package
still installs and registers, but every block passes straight through to
Headroom's own compressors: Winnow fails open whenever it cannot load a model.

On first use Winnow downloads its default model,
[`rbk4209/winnow-pooled-32m`](https://huggingface.co/rbk4209/winnow-pooled-32m)
(128 MB), from the Hugging Face Hub at a pinned commit. Nothing else to set up.

!!! tip "Use a GPU"
    Winnow runs anywhere, but a CUDA GPU is what makes it fast enough to score
    every tool output. If you have one, install the CUDA build of PyTorch first
    ([pytorch.org/get-started](https://pytorch.org/get-started/locally/)).

## Check the install

```python
from headroom.transforms.compressor_registry import CompressorRegistry

registry = CompressorRegistry()
print(registry.discover())  # ['winnow']
```

## Offline or air-gapped machines

Download the model once and point Winnow at the folder:

```bash
hf download rbk4209/winnow-pooled-32m --local-dir ./winnow-pooled-32m
export HEADROOM_WINNOW_MODEL=./winnow-pooled-32m
```

Next: [turn it on](quickstart.md).
