# Development

```bash
git clone https://github.com/RecepKurtulus/headroom-winnow
cd headroom-winnow
pip install -e ".[dev,model]"
```

## Checks

The repository follows Headroom's own tooling (ruff, mypy in strict-ish mode,
pytest), so code could move upstream without restyling.

```bash
ruff check . && ruff format --check .
mypy headroom_winnow tests benchmarks
pytest              # fast: fake model, no downloads
pytest -m slow      # downloads and runs the real highlighter
```

CI runs lint, type checks and the fast tests on Python 3.10 and 3.12.

## Tests

| File | Covers |
|---|---|
| `tests/test_selection.py` | line splitting, span → line mapping, safety rules, rendering |
| `tests/test_markers.py` | hash and marker format, parity with Headroom's marker regex |
| `tests/test_compressor.py` | the contract, gates, fail-open behaviour, token budgets |
| `tests/test_backends.py` | backend selection, devices, budgets, config via environment |
| `tests/test_router.py` | end to end through Headroom's `ContentRouter`, CCR retrieval |
| `tests/test_real_model.py` | the real highlighter (`-m slow`) |

## Documentation

```bash
pip install mkdocs-material "mkdocstrings[python]"
mkdocs serve
```

The site is built and deployed to GitHub Pages on every push to `main`.

## Project layout

```text
headroom_winnow/
  compressor.py    WinnowCompressor: Headroom's compressor contract, gates, fail-open
  selection.py     spans → lines, safety rules, rendering with markers
  markers.py       CCR marker format and deterministic hashes
  backends.py      pooled and highlighter backends: lazy, pinned, device-aware
benchmarks/        compare.py and RESULTS.md
training/          Kaggle notebook for the pooled model
upstream/          fix proposed to Headroom
docs/              this site
```
