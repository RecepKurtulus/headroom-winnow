# Installation

Winnow is a Python package that registers itself with Headroom through the
`headroom.compressor` entry point. It needs Python 3.10+ and Headroom 0.39.

```bash
pip install "headroom-winnow[model] @ git+https://github.com/RecepKurtulus/headroom-winnow"
```

The `model` extra pulls in `torch` and `transformers`. Without it the package
still installs and registers, but every block passes straight through to
Headroom's own compressors: Winnow fails open whenever it cannot load a model.

!!! tip "Use a GPU"
    Winnow runs anywhere, but a CUDA GPU is what makes it fast enough to score
    every tool output. If you have one, install the CUDA build of PyTorch first
    ([pytorch.org/get-started](https://pytorch.org/get-started/locally/)).

## Get the recommended model

The fastest and most accurate option is the 32M **pooled** line classifier
trained for this project. It ships as a release asset:

1. Download `squeez_pooled_ettin32m.zip` from the
   [latest release](https://github.com/RecepKurtulus/headroom-winnow/releases/latest).
2. Unzip it anywhere.
3. Point Winnow at it:

    ```bash
    export HEADROOM_WINNOW_BACKEND=pooled
    export HEADROOM_WINNOW_MODEL=/path/to/squeez_pooled_ettin32m
    ```

If you skip this step, Winnow uses the 150M **highlighter**
(`KRLabsOrg/verbatim-rag-modern-bert-v2`), downloaded from the Hugging Face Hub
on first use. It is slower, so by default it only scores short outputs. See
[Models](../guide/models.md) for the trade-offs.

## Check the install

```python
from headroom.transforms.compressor_registry import CompressorRegistry

registry = CompressorRegistry()
print(registry.discover())  # ['winnow']
```

Next: [turn it on](quickstart.md).
