# Training a model

The recommended pooled model was trained with a single notebook on a free Kaggle
GPU: [`training/kaggle_train_pooled.ipynb`](https://github.com/RecepKurtulus/headroom-winnow/blob/main/training/kaggle_train_pooled.ipynb).

## What it does

1. Clones [Squeez](https://github.com/KRLabsOrg/squeez) at a pinned commit and
   uses its own training code, so the recipe is reproducible.
2. Downloads [`KRLabsOrg/tool-output-extraction-swebench`](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench):
   10,508 training, 240 dev and 618 test examples across 27 tool types.
3. Fine-tunes `jhu-clsp/ettin-encoder-32m` as a pooled line classifier:
   4,096-token windows, effective batch 96, learning rate 5e-5, 3 epochs, fp16.
4. Evaluates on the held-out test split and zips the model.

It takes about six hours on one T4. Validation loss fell from 0.63 to 0.31.

## Running it

1. Create a Kaggle notebook from the file (or push it with the Kaggle CLI using
   `training/kernel-metadata.json`).
2. Settings → Accelerator → **GPU T4**, and Settings → **Internet on**.
3. *Run All*, then download `squeez_pooled_ettin32m.zip` from the Output tab.

!!! warning "Two fixes the notebook applies"
    - Training is pinned to **one GPU**. Kaggle's T4 ×2 makes the Hugging Face
      Trainer wrap the model in `DataParallel`, which hides an attribute Squeez's
      loss function reads.
    - Squeez's checkpoint hook passes a `metrics` argument that
      `transformers` 5.2 no longer accepts. The notebook patches the call so
      it forwards whatever the Trainer passes.

## Trying other encoders

Change `--base-model` in the training cell. Larger encoders
(`answerdotai/ModernBERT-base`) trade speed for capacity; smaller ones
(`jhu-clsp/ettin-encoder-17m`) go the other way. Measure any new model with:

```bash
HEADROOM_WINNOW_MAX_TOKENS=1000000000 python benchmarks/compare.py --device cuda \
    --backend pooled --model-path path/to/model --methods squeez-raw,headroom-winnow
```
