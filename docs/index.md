---
hide:
  - navigation
---

<p align="center"><img src="assets/logo.png" width="112" alt="Winnow logo"></p>

# Winnow

**Separate the grain from the chaff in your coding agent's tool output.**
A task-aware pruner for [Headroom](https://github.com/headroomlabs-ai/headroom), powered by [Squeez](https://github.com/KRLabsOrg/squeez) models.

Coding agents spend most of their context window on tool output: test runs,
builds, grep hits, logs. Usually only a handful of those lines matter for the
next step. Winnow plugs a small model trained on real agent traces
into Headroom's compression pipeline. It keeps the lines the task needs, never
drops an error or a traceback, and turns everything else into retrievable
markers.

<div class="grid cards" markdown>

-   :material-content-cut:{ .lg .middle } **41% fewer tokens**

    ---

    across 618 real tool outputs, with 0.85 of the relevant lines kept

-   :material-shield-check:{ .lg .middle } **Zero lost errors**

    ---

    failure lines and whole tracebacks always survive, by rule

-   :material-lightning-bolt:{ .lg .middle } **0.29 s median**

    ---

    on a consumer GPU with the 32M pooled model

-   :material-undo-variant:{ .lg .middle } **Nothing is lost**

    ---

    every dropped run is one `<<ccr:…>>` marker the agent can expand

</div>

## See it work

The agent asks *"Find the build output block that reports the missing
`GetObject` method in the `UserHandler` implementation of
`storage.StorageService`"* and runs `go build ./...`. The output is 119 lines
and 1,838 tokens. After Winnow, it is 32 lines and 479 tokens
(**−74%**):

```text
$ go build ./...
# .../handlers/user_handler.go:23:12: cannot use h (type *UserHandler) as type storage.StorageService:
	*UserHandler does not implement storage.StorageService (missing GetObject method)
# .../handlers/order_handler.go:45:9: cannot use orderSvc (type *order.Service) ...
	*order.Service does not implement storage.StorageService (missing DeleteObject method)
# .../auth/auth.go:12:5: import cycle not allowed
	imports github.com/example/project/internal/permissions
<<ccr:e2e73bc3f2f290a77efc7426 52_lines_offloaded>>
--- FAIL: TestUserHandler_Get (0.00s)
    user_handler_test.go:45:
        Error:        Not equal:
                        expected: &storage.Object{ID:"obj-123", Size:1024}
                        actual  : <nil>
FAIL
exit status 1
FAIL    github.com/example/project/internal/handlers   0.012s

<<ccr:0bac82ae58f9e3ba08a67445 15_lines_offloaded>>
--- FAIL: TestConcurrentAccess (0.01s)
    lru_test.go:92:
        panic: runtime error: invalid memory address or nil pointer dereference
FAIL
exit status 1
...
```

The block the agent asked for is kept verbatim, every failure line survives,
and the 90 lines of goroutine dumps and unrelated packages became three markers.
(Real model output on an example from the Squeez test set; long paths shortened.)

## Better than similarity search

At equal compression, Squeez keeps far more of the lines that matter than
Headroom's built-in `relevance_split` (BM25 or embeddings):

| Lines dropped | `relevance_split` (hybrid) | **Winnow** |
|---|---|---|
| 50% | 0.729 | **0.886** |
| 70% | 0.629 | **0.842** |
| 90% | 0.459 | **0.681** |

[Full benchmark →](guide/benchmarks.md){ .md-button .md-button--primary }
[Get started →](getting-started/installation.md){ .md-button }

## Built on

- [Headroom](https://github.com/headroomlabs-ai/headroom): the context
  optimization proxy, its CCR retrieval store and the external compressor
  contract this plugin implements.
- [Squeez](https://github.com/KRLabsOrg/squeez) by KRLabs: task-conditioned
  pruning of tool output, the training code, the highlighter model and the
  dataset.
