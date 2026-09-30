# Limitations

## Headroom's lossless fold runs first

Headroom applies its byte-exact lossless fold (and, for logs and search
results, `relevance_split`) *before* the external compressor hook, and returns
early when either succeeds. Blocks that fold, such as logs with repeated lines,
never reach Winnow. This is pinned by
`tests/test_router.py::test_known_gap_lossless_fold_preempts_external`, which
will start failing if Headroom changes the order.

## Passthroughs in Headroom 0.39.1

When an external compressor declines a block (`compressed=False`), Headroom
0.39.1 still adopts the unchanged block as the result and skips its built-in
compressors for it. A one-line fix is proposed upstream
([`upstream/router-respect-passthrough.patch`](https://github.com/RecepKurtulus/headroom-winnow/blob/main/upstream/router-respect-passthrough.patch));
until it lands, `test_router_falls_back_when_backend_unavailable` is a strict
`xfail` that will flip automatically.

## CPU performance

On CPU the pooled model's budget is 2,048 tokens and the highlighter's is 512.
Larger outputs go to Headroom's own compressors. A GPU is recommended.

## Scope

Winnow handles logs, search results, diffs and plain text. JSON, source code,
HTML, CSV and config files are left to Headroom's structure-aware compressors,
which already handle them well.
