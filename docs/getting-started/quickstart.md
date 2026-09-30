# Quick start

Installing Winnow changes nothing on its own. Headroom only routes traffic
through external compressors you select by name.

=== "Headroom proxy"

    ```bash
    export HEADROOM_WINNOW_BACKEND=pooled
    export HEADROOM_WINNOW_MODEL=/path/to/squeez_pooled_ettin32m

    headroom proxy --compressor winnow
    ```

    Point your agent at the proxy as usual (for example `headroom wrap claude`).
    Every tool result that Headroom classifies as a log, search result, diff or
    plain text now goes through Winnow first.

=== "Python"

    ```python
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    router = ContentRouter(ContentRouterConfig(active_external_compressors=["winnow"]))
    result = router.compress(tool_output, context="why does test_login_redirect fail")

    print(result.strategy_chain)  # ['external:winnow']
    print(result.compressed)  # pruned output with <<ccr:...>> markers
    ```

=== "Standalone"

    Winnow's compressor follows Headroom's pure-data contract, so you can call
    it directly, for example in your own pipeline or a notebook:

    ```python
    from headroom.transforms.compressor_registry import CompressInput
    from headroom_winnow import PooledBackend, WinnowCompressor

    winnow = WinnowCompressor(PooledBackend("/path/to/squeez_pooled_ettin32m"))
    out = winnow.compress(
        CompressInput(
            content=open("build.log").read(),
            content_type="text/x-log",
            query="find the missing GetObject method error",
        )
    )

    print(out.compressed, out.tokens_before, "->", out.tokens_after)
    print(out.content)  # kept lines + markers
    print(out.recoverable)  # {hash: original text of each dropped run}
    ```

## What you will see

- **Pruned blocks** contain the relevant lines verbatim plus one
  `<<ccr:HASH N_lines_offloaded>>` marker per dropped run. The agent can expand
  any marker with Headroom's retrieval tool.
- **Passthroughs**: short outputs, outputs without a task query, or outputs
  where pruning would save under 20% are returned unchanged
  (`compressed=False`) so Headroom's own compressors handle them.

!!! note
    With Headroom 0.39.1, a passthrough from any external compressor is not yet
    handed back to the built-in compressors. See
    [Limitations](../guide/limitations.md).
