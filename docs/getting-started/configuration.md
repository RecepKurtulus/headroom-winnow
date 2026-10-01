# Configuration

Winnow is configured through environment variables, so it works the same under
the Headroom proxy and in your own code. Every variable is optional.

| Variable | Default | Meaning |
|---|---|---|
| `HEADROOM_WINNOW_BACKEND` | `pooled` | `pooled` or `highlighter` |
| `HEADROOM_WINNOW_MODEL` | the backend's published model | Hub id or local path |
| `HEADROOM_WINNOW_REVISION` | pinned commit of the default model | Model revision to load |
| `HEADROOM_WINNOW_DEVICE` | `auto` | `auto` (CUDA when available), `cuda` or `cpu` |
| `HEADROOM_WINNOW_DTYPE` | `float32` | `float16` is faster only on GPUs with tensor cores |
| `HEADROOM_WINNOW_MAX_TOKENS` | per backend and device | Largest output the model scores; larger ones go to Headroom |

## Token budgets

Each backend knows how much it can score within a reasonable latency on each
kind of device. Outputs above the budget skip the model entirely and go to
Headroom's own path, so Winnow never stalls the agent.

| Backend | GPU budget | CPU budget |
|---|---|---|
| `pooled` | 32,768 tokens | 2,048 tokens |
| `highlighter` | 2,048 tokens | 512 tokens |

Override with `HEADROOM_WINNOW_MAX_TOKENS` if your hardware is faster or slower.

## Pruning settings

The pruning rules can be tuned in code through `WinnowSettings`:

```python
from headroom_winnow import WinnowCompressor, WinnowSettings

winnow = WinnowCompressor(
    settings=WinnowSettings(
        min_lines=40,  # shorter outputs pass through
        context_lines=2,  # neighbours kept around every kept line
        edge_lines=2,  # first/last lines always kept
        min_savings=0.2,  # pass through if pruning saves less than this
        max_tokens=None,  # None = use the backend's budget
    )
)
```

The defaults are recall-first: dropping a line the agent needed costs a
retrieval round trip or a re-run, which is far more expensive than a few spare
tokens.
