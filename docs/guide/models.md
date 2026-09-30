# Models

Winnow can run two kinds of models. Both come from the
[Squeez](https://github.com/KRLabsOrg/squeez) project's approach and are trained
on the same data: tool outputs from SWE-bench agent runs, labelled with the
lines the agent actually needed.

| | `pooled` (recommended) | `highlighter` |
|---|---|---|
| Model | 32M line classifier, fine-tuned from [`jhu-clsp/ettin-encoder-32m`](https://huggingface.co/jhu-clsp/ettin-encoder-32m) | [`KRLabsOrg/verbatim-rag-modern-bert-v2`](https://huggingface.co/KRLabsOrg/verbatim-rag-modern-bert-v2), 150M span model |
| Output | a probability per line | character spans |
| Where to get it | [GitHub release](https://github.com/RecepKurtulus/headroom-winnow/releases/latest) (121 MB) | downloaded from the Hub on first use |
| Median latency¹ | **0.29 s** | 1.5 s |
| p95 latency¹ | **0.92 s** | 9.0 s |
| Recall as shipped¹ | 0.847 | 0.836 |
| Token reduction¹ | **41.4%** | 39.7% |

¹ Full Squeez test split, consumer GPU, no token budget.

## Pooled

The pooled model reads the task and the tool output in one pass, with a
separator token between lines, and mean-pools each line's tokens into a
relevance score. It was trained for this project with
[`training/kaggle_train_pooled.ipynb`](training.md) and matches the
highlighter's recall at a fraction of its cost, which lets Winnow score
practically every tool output on a GPU instead of only small ones.

```bash
export HEADROOM_WINNOW_BACKEND=pooled
export HEADROOM_WINNOW_MODEL=/path/to/squeez_pooled_ettin32m
```

## Highlighter

The default when nothing is configured, because it needs no manual download.
Loading is pinned to a reviewed commit (weights and tokenizer) because the model
runs its own code via `trust_remote_code`. Its latency grows quickly with input
length, so its default budget is 2,048 tokens on a GPU.

## Bring your own

Any object with a `find_spans(query, content) -> list[tuple[int, int]]` method
works as a backend:

```python
from headroom_winnow import WinnowCompressor


class KeywordBackend:
    def find_spans(self, query: str, content: str) -> list[tuple[int, int]]:
        spans, start = [], content.find("ERROR")
        while start != -1:
            spans.append((start, start + 5))
            start = content.find("ERROR", start + 1)
        return spans


winnow = WinnowCompressor(KeywordBackend())
```

Give it a `max_input_tokens` attribute to set a token budget.
