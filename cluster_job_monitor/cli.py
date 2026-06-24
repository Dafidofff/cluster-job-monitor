#!/usr/bin/env python3
"""Entry point for the cross-cluster SLURM job dashboard (read-only).

Examples:
    cluster-jobs                    # live TUI using ./clusters.json
    cluster-jobs --config c.json    # live TUI with a specific config
    cluster-jobs --once             # print one snapshot and exit
    cluster-jobs --demo             # TUI with synthetic data (no clusters)
    cluster-jobs --once --demo      # print synthetic snapshot and exit
    cluster-jobs --overview         # capacity table: free cpus/gpus + my jobs
    cluster-jobs --overview --json  # same, as JSON for coding agents

(``python run.py ...`` still works via the root shim.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cluster_job_monitor.collector import build_overview, collect, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path.cwd() / "clusters.json"),
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
        from cluster_job_monitor.tui.sample import make_demo_snapshot
        config = {"refresh_seconds": 5, "hosts": []}
        collect_fn = make_demo_snapshot
    else:
        try:
            config = load_config(args.config)
        except FileNotFoundError:
            print(f"config not found: {args.config}\n"
                  f"Copy clusters.example.json to clusters.json and fill in your hosts,\n"
                  f"or try a quick look with:  cluster-jobs --demo",
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
            from cluster_job_monitor.tui.render import render_overview
            Console().print(render_overview(overview))
        return 0

    if args.once:
        snapshot = collect_fn(config)
        if args.json:
            print(json.dumps(snapshot.to_dict(), indent=2))
            return 0
        from rich.console import Console
        from cluster_job_monitor.tui.render import Filters, render_body, render_header
        minimized = {h["name"] for h in config.get("hosts", [])
                     if h.get("minimized") and h.get("name")}
        console = Console()
        console.print(render_header(snapshot, Filters(), refreshing=False))
        console.print(render_body(snapshot, Filters(), minimized))
        return 0

    from cluster_job_monitor.tui.app import JobMonitorApp
    JobMonitorApp(config, collect_fn=collect_fn).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
