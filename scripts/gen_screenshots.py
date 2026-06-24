#!/usr/bin/env python3
"""Generate the colored README screenshots from the tool's own ``--demo`` data.

Renders the Rich TUI renderables to SVG headlessly (no real terminal, no real
clusters) so the images in the README are reproducible and theme-stable. Run
from the repo root::

    python scripts/gen_screenshots.py

This writes ``docs/img/dashboard.svg`` and ``docs/img/overview.svg``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the repo root importable when run as a script from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console  # noqa: E402
from rich.terminal_theme import MONOKAI  # noqa: E402

from cluster_job_monitor import build_overview  # noqa: E402
from cluster_job_monitor.tui.render import (  # noqa: E402
    Filters,
    render_body,
    render_header,
    render_overview,
)
from cluster_job_monitor.tui.sample import make_demo_snapshot  # noqa: E402

# Console width (cells). Wide enough that neither view wraps/looks cramped.
WIDTH = 100


def _out_dir() -> Path:
    d = REPO_ROOT / "docs" / "img"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    snap = make_demo_snapshot()
    out = _out_dir()

    # Live dashboard: header + per-cluster body.
    dash = Console(record=True, width=WIDTH)
    dash.print(render_header(snap, Filters(), refreshing=False))
    dash.print(render_body(snap, Filters(), set()))
    dash_path = out / "dashboard.svg"
    dash.save_svg(str(dash_path), title="cluster-jobs", theme=MONOKAI)

    # Capacity overview (--overview).
    over = Console(record=True, width=WIDTH)
    over.print(render_overview(build_overview(snap)))
    over_path = out / "overview.svg"
    over.save_svg(str(over_path), title="cluster-jobs --overview", theme=MONOKAI)

    for p in (dash_path, over_path):
        rel = os.path.relpath(p, REPO_ROOT)
        print(f"wrote {rel} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
