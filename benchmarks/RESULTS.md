# Benchmark results

Full `test` split of `KRLabsOrg/tool-output-extraction-swebench` (618 examples),
`benchmarks/compare.py --device cuda`, GTX 1650 Ti, float32, 2026-09-28.
Metrics are Squeez's own (sets of stripped gold lines). `headroom-squeez` is the
shipped configuration (2048-token budget on GPU); `-unbounded` scores every
output regardless of size. `@N` rows keep the top (100-N)% of lines by each
method's own score, so recall compares ranking quality at equal compression.

| method | recall | precision | F1 | line compr. | token red. | mandatory lost | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| headroom-squeez | 0.905 | 0.105 | 0.162 | 0.110 | 0.090 | 0 | 1 | 767 |
| headroom-squeez-unbounded | 0.836 | 0.174 | 0.248 | 0.432 | 0.397 | 0 | 1547 | 9016 |
| rs-bm25 | 0.725 | 0.148 | 0.208 | 0.619 | 0.586 | 7953 | 1 | 4 |
| rs-bm25@50 | 0.711 | 0.108 | 0.161 | 0.507 | 0.425 | 3679 |  |  |
| rs-bm25@70 | 0.597 | 0.134 | 0.182 | 0.705 | 0.623 | 7019 |  |  |
| rs-bm25@90 | 0.444 | 0.225 | 0.246 | 0.899 | 0.855 | 10937 |  |  |
| rs-hybrid | 0.753 | 0.155 | 0.217 | 0.611 | 0.576 | 7771 | 2247 | 8572 |
| rs-hybrid@50 | 0.729 | 0.107 | 0.161 | 0.507 | 0.357 | 3130 |  |  |
| rs-hybrid@70 | 0.629 | 0.145 | 0.196 | 0.705 | 0.594 | 6798 |  |  |
| rs-hybrid@90 | 0.459 | 0.242 | 0.260 | 0.899 | 0.854 | 10907 |  |  |
| squeez-raw | 0.750 | 0.604 | 0.629 | 0.915 | 0.904 | 12491 | 1545 | 9016 |
| squeez@50 | 0.874 | 0.138 | 0.206 | 0.507 | 0.461 | 5479 |  |  |
| squeez@70 | 0.847 | 0.209 | 0.281 | 0.705 | 0.674 | 8283 |  |  |
| squeez@90 | 0.731 | 0.404 | 0.437 | 0.899 | 0.883 | 11570 |  |  |

Gate 3 (fixed budget, recall): 
- 50%: squeez 0.874 vs rs-hybrid 0.729 -> PASS
- 70%: squeez 0.847 vs rs-hybrid 0.629 -> PASS
- 90%: squeez 0.731 vs rs-hybrid 0.459 -> PASS

## Reading it

- **Gate 3 passes.** At equal compression Squeez keeps 15-27 points more of the
  gold lines than Headroom's best built-in (`relevance_split`, hybrid).
- **Safety holds.** `headroom-squeez` drops zero failure/traceback lines;
  `relevance_split` drops ~7,800 across the split.
- **The budget is the bottleneck.** Most outputs here exceed 2048 tokens, so the
  shipped plugin only prunes a few of them (9% token reduction overall). Unbounded
  it reaches 40% at 0.84 recall, but at 1.5 s p50 / 9 s p95 on this GPU. A smaller
  model (see `training/`) is the way to lift the budget.
- `rs-*` natural rows drop their tail outright here; inside Headroom that tail
  is Kompressed instead, so their token reduction is an upper bound.
