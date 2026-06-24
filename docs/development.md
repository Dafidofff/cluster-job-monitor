# Development

## Running the tests

```bash
pip install -e ".[dev]"
pytest                       # run the suite
pytest --cov --cov-report=term-missing   # with coverage (~94%)
```

Tests live in `tests/` and mock SSH / `subprocess`, so they run **anywhere** —
no cluster access needed.

## Project layout

```
cluster-job-monitor/
  run.py                 # entry point (--once, --overview, --json, --demo, --config)
  core/collector.py      # UI-agnostic: SSH + squeue/sinfo -> Snapshot dataclasses
  mcp_server.py          # MCP wrapper exposing the capacity overview to agents
  tui/app.py             # Textual app (live loop, filters, keybindings)
  tui/render.py          # Rich renderables (shared by TUI, --once, --overview)
  tui/sample.py          # synthetic snapshot for --demo
  clusters.example.json  # config template (copy to clusters.json)
```

`core/collector.py` has **no third-party dependencies** and returns a
`Snapshot` whose `.to_dict()` is JSON-ready — that's the seam for a future
web/phone dashboard (push the dict to an authenticated endpoint and render it in
a browser), without changing the collector.

## Continuous integration

CI ([GitHub Actions](https://github.com/Dafidofff/cluster-job-monitor/actions/workflows/ci.yml))
runs the suite on Python 3.10–3.12 and reports coverage to
[Codecov](https://codecov.io/gh/Dafidofff/cluster-job-monitor). The coverage
upload runs only on push (not on fork PRs), keeping `id-token: write` away from
untrusted code.

## Building these docs

This site is built with [MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme.

```bash
pip install -r docs-requirements.txt
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # what CI runs; fails on broken links/nav
```

On every push to `main`, the
[`docs.yml`](https://github.com/Dafidofff/cluster-job-monitor/blob/main/.github/workflows/docs.yml)
workflow builds the site with `--strict` and deploys it to GitHub Pages.
