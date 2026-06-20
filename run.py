#!/usr/bin/env python3
"""Entry point for the cross-cluster SLURM job dashboard (read-only).

Examples:
    python run.py                    # live TUI using ./clusters.json
    python run.py --config c.json    # live TUI with a specific config
    python run.py --once             # print one snapshot and exit
    python run.py --demo             # TUI with synthetic data (no clusters)
    python run.py --once --demo      # print synthetic snapshot and exit
    python run.py --overview         # capacity table: free cpus/gpus + my jobs
    python run.py --overview --json  # same, as JSON for coding agents
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python run.py` from anywhere by making this dir importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.collector import build_overview, collect, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).parent / "clusters.json"),
                        help="path to clusters.json (default: ./clusters.json)")
    parser.add_argument("--once", action="store_true",
                        help="print a single snapshot and exit")
    parser.add_argument("--overview", action="store_true",
                        help="print per-cluster/partition capacity (free cpus/gpus "
                             "+ my queued/running jobs) and exit")
    parser.add_argument("--json", action="store_true",
                        help="with --overview/--once, emit JSON instead of a table")
    parser.add_argument("--demo", action="store_true",
                        help="use synthetic data instead of real clusters")
    args = parser.parse_args()

    if args.demo:
        from tui.sample import make_demo_snapshot
        config = {"refresh_seconds": 5, "hosts": []}
        collect_fn = make_demo_snapshot
    else:
        try:
            config = load_config(args.config)
        except FileNotFoundError:
            example = Path(__file__).parent / "clusters.example.json"
            print(f"config not found: {args.config}\n"
                  f"Copy {example.name} to clusters.json and fill in your hosts,\n"
                  f"or try a quick look with:  python run.py --demo",
                  file=sys.stderr)
            return 2
        except ValueError as exc:
            print(f"invalid config: {exc}", file=sys.stderr)
            return 2
        collect_fn = collect

    if args.overview:
        snapshot = collect_fn(config) if args.demo \
            else collect(config, with_partitions=True)
        overview = build_overview(snapshot)
        if args.json:
            print(json.dumps(overview, indent=2))
        else:
            from rich.console import Console
            from tui.render import render_overview
            Console().print(render_overview(overview))
        return 0

    if args.once:
        snapshot = collect_fn(config)
        if args.json:
            print(json.dumps(snapshot.to_dict(), indent=2))
            return 0
        from rich.console import Console
        from tui.render import Filters, render_body, render_header
        minimized = {h["name"] for h in config.get("hosts", [])
                     if h.get("minimized") and h.get("name")}
        console = Console()
        console.print(render_header(snapshot, Filters(), refreshing=False))
        console.print(render_body(snapshot, Filters(), minimized))
        return 0

    from tui.app import JobMonitorApp
    JobMonitorApp(config, collect_fn=collect_fn).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
