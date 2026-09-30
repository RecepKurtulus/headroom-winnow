# Benchmark results

Full `test` split of [`KRLabsOrg/tool-output-extraction-swebench`](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench)
(618 tool outputs from SWE-bench agent runs, 27 tool types), measured with
`benchmarks/compare.py` on a consumer GPU in float32.

Metrics follow Squeez's own evaluation: predicted and gold lines are compared as
sets of stripped lines. **Recall** is the share of gold lines kept, **token
reduction** is measured with Headroom's own token estimator (markers included),
and **mandatory lost** counts error / failure / traceback lines that were dropped.

## Head to head at equal compression

Each method ranks lines with its own score and keeps the top share, so recall
compares ranking quality directly. This is the project's go/no-go gate.

| Lines dropped | relevance_split (BM25) | relevance_split (hybrid) | Squeez 150M highlighter | **Squeez 32M pooled** |
|---|---|---|---|---|
| 50% | 0.711 | 0.729 | 0.874 | **0.886** |
| 70% | 0.597 | 0.629 | **0.847** | 0.842 |
| 90% | 0.444 | 0.459 | **0.731** | 0.681 |

## As shipped

The compressor with its safety rules (failure lines, whole tracebacks, ±2
context, first/last lines) and gates.

| Configuration | Recall | Token reduction | Mandatory lost | p50 | p95 |
|---|---|---|---|---|---|
| relevance_split, BM25 (tail dropped)¹ | 0.725 | 58.6% | 7,953 | 1 ms | 4 ms |
| relevance_split, hybrid (tail dropped)¹ | 0.753 | 57.6% | 7,771 | 2.2 s | 8.6 s |
| Winnow, 150M highlighter, unbounded | 0.836 | 39.7% | **0** | 1.5 s | 9.0 s |
| Winnow, 150M highlighter, 2,048-token budget | 0.905 | 9.0% | **0** | 1 ms² | 0.8 s |
| **Winnow, 32M pooled** | **0.847** | **41.4%** | **0** | **0.29 s** | **0.92 s** |

¹ Inside Headroom the low-relevance tail is Kompressed rather than dropped, so
these token reductions are an upper bound.
² Most outputs exceed the budget and pass straight through to Headroom.

## Takeaways

- **Squeez ranks lines far better than lexical or embedding similarity**: 15-27
  recall points over Headroom's built-in `relevance_split` at the same
  compression.
- **The safety rules work**: zero failure or traceback lines lost, against
  ~7,800 for `relevance_split`.
- **The 32M pooled model makes it practical**: ~5x faster than the 150M
  highlighter at the median and ~10x at p95, with the same recall and slightly
  better compression, so it can score every output instead of only small ones.

## Reproduce

```bash
# Headroom baselines + 150M highlighter
python benchmarks/compare.py --device cuda

# 32M pooled model (trained with training/kaggle_train_pooled.ipynb)
HEADROOM_WINNOW_MAX_TOKENS=1000000000 python benchmarks/compare.py --device cuda \
    --backend pooled --model-path path/to/squeez_pooled_ettin32m \
    --methods squeez-raw,headroom-winnow
```
