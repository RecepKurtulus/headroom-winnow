# headroom-squeez

Task-conditioned line pruning for [Headroom](https://github.com/headroomlabs-ai/headroom),
powered by the [Squeez](https://github.com/KRLabsOrg/squeez) span model.

Headroom already compresses agent tool output. This plugin replaces one piece
of it: deciding which lines of a log, test run, grep result or diff the agent
needs next. The decision comes from a model trained on real SWE-bench agent
traces instead of lexical similarity. Dropped lines are never lost; each run
becomes a `<<ccr:HASH N_lines_offloaded>>` marker the agent can retrieve.

> Status: pre-alpha. The v1 go/no-go gate is a benchmark against Headroom's
> built-in `relevance_split` (see `benchmarks/`).

## Install

```bash
pip install "headroom-squeez[model]"
```

Installing changes nothing. Opt in by name:

```python
from headroom.transforms.content_router import ContentRouterConfig

ContentRouterConfig(active_external_compressors=["squeez"])
```

or `--compressor squeez` on the Headroom proxy.

Runs locally on CPU. No API key, GPU or cloud account.

## What it keeps

- Every line the model marks as relevant to the task.
- Always: error, failure, traceback, panic, exception, exit code and fatal
  lines, whole Python tracebacks and pytest `E` lines.
- Two neighbours on each side of every kept line.
- The first and last two lines.

It passes through unchanged when there is no task query, the output is under
40 lines, the output is larger than the model can score in time on this machine
(`HEADROOM_SQUEEZ_MAX_TOKENS`), the model found nothing, or the saving is under
20%.

## Speed

The model has to see the whole output at once: cutting it into smaller windows
made it 20x faster but halved recall. So instead of shrinking the window, the
plugin only scores outputs it can finish quickly and leaves larger ones to
Headroom. Measured forward time (float32):

| Tokens | GTX 1650 Ti | 4-core laptop CPU |
|---|---|---|
| 512 | 70 ms | 660 ms |
| 2048 | 490 ms | 3.7 s |
| 8192 | 4.2 s | 31 s |

A GPU is strongly recommended; on CPU the plugin only handles short outputs. It never raises.
If the model can't load, it logs one warning and Headroom's own path takes over.

## Configuration

| Variable | Default |
|---|---|
| `HEADROOM_SQUEEZ_MODEL` | `KRLabsOrg/verbatim-rag-modern-bert-v2` |
| `HEADROOM_SQUEEZ_REVISION` | pinned commit of the default model |
| `HEADROOM_SQUEEZ_DEVICE` | `auto` (CUDA if available, else CPU) |
| `HEADROOM_SQUEEZ_DTYPE` | `float32` |
| `HEADROOM_SQUEEZ_MAX_TOKENS` | 2048 on GPU, 512 on CPU |

## Known limitations

Headroom 0.39.1 ignores `compressed=False`: when this plugin declines a block
(no query, too short, model unavailable, ...) the router still adopts the
unchanged block and skips its own compressors for it. The one-line fix is in
`upstream/router-respect-passthrough.patch`; until it lands, the
`test_router_falls_back_when_backend_unavailable` test is a strict `xfail`.

Headroom runs its lossless fold (and, for logs and search results, its
`relevance_split`) *before* the external-compressor hook, and returns early
when either succeeds. Such blocks never reach this plugin today. See
`tests/test_router.py::test_known_gap_lossless_fold_preempts_external`.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy headroom_squeez && pytest
pytest -m slow   # loads the real model
```

## License

Apache-2.0. See `NOTICE` for attributions.
