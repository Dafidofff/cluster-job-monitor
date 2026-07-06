#!/usr/bin/env python3
"""MCP server exposing the cluster capacity overview to coding agents.

A thin wrapper over ``cluster_job_monitor.collector`` — it adds no new ways to
touch the clusters, it just surfaces the same read-only ``squeue``/``sinfo``
data as MCP tools so an agent (Claude Code, Cursor, …) can call it natively.

Run it directly (stdio transport):

    python mcp_server.py        # via the root shim
    python -m cluster_job_monitor.mcp_server

Point it at a config with the ``CLUSTER_MONITOR_CONFIG`` env var, otherwise it
uses ``clusters.json`` in the current directory. Register with Claude Code via:

    claude mcp add cluster-monitor -- python /abs/path/to/mcp_server.py

Requires the MCP SDK (``pip install "cluster-job-monitor[mcp]"``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from cluster_job_monitor.collector import collect, collect_overview, load_config

# The SDK is optional: importing this module (e.g. from tests) must work even
# when ``mcp`` is absent. We only hard-require it when actually serving.
try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # pragma: no cover - exercised only without the SDK
    FastMCP = None


def _config_path() -> str:
    """Config file to read (env override, else clusters.json in the cwd)."""
    return os.environ.get("CLUSTER_MONITOR_CONFIG") or str(
        Path.cwd() / "clusters.json"
    )


def _load() -> dict:
    return load_config(_config_path())


def cluster_overview() -> dict:
    """Queued/running jobs, free CPUs/GPUs, and compute budget, per cluster.

    Call this before submitting work to see where there is capacity *and*
    whether you have budget left to run there. For each cluster it returns:
    your own running/pending job counts, the free and total CPUs/GPUs, a
    per-partition breakdown (free/alloc/total CPUs and GPUs, node states, and
    how many of your jobs sit in each partition), and — for hosts whose config
    sets an ``"accounting"`` source — a per-account ``budget`` list with
    granted/used/remaining and a ``unit`` label.

    Budget ``source`` is either ``"sacctmgr"`` (native SLURM accounting;
    well-supported; ``granted``/``remaining`` may be null when the site doesn't
    cap via GrpTRESMins) or ``"budget-overview"`` (a SURF-specific script whose
    output format is unverified — treat its numbers as best-effort). ``budget``
    is an empty list for hosts that don't configure accounting.

    Read-only: it runs only ``squeue --me``, ``sinfo``, and (when configured)
    ``sacctmgr``/``sreport`` or ``budget-overview`` on each host.
    """
    try:
        return collect_overview(_load())
    except FileNotFoundError:
        return {"error": f"config not found: {_config_path()}", "clusters": []}
    except ValueError as exc:
        return {"error": f"invalid config: {exc}", "clusters": []}


def my_jobs() -> dict:
    """Your current jobs across every configured cluster (no capacity data).

    Lighter than cluster_overview — skips the ``sinfo`` call. Returns one entry
    per cluster with its job list and running/pending/other counts. Use this
    when you only need to know what you have queued, not where free GPUs are.
    """
    try:
        return collect(_load()).to_dict()
    except FileNotFoundError:
        return {"error": f"config not found: {_config_path()}", "hosts": []}
    except ValueError as exc:
        return {"error": f"invalid config: {exc}", "hosts": []}


def build_server() -> "FastMCP":
    """Construct the FastMCP server with both tools registered."""
    if FastMCP is None:
        raise RuntimeError(
            "The MCP SDK is not installed. "
            'Install it with:  pip install "cluster-job-monitor[mcp]"'
        )
    server = FastMCP("cluster-monitor")
    server.tool()(cluster_overview)
    server.tool()(my_jobs)
    return server


def main() -> int:
    if FastMCP is None:
        sys.stderr.write(
            "The MCP SDK is not installed.\n"
            'Install it with:  pip install "cluster-job-monitor[mcp]"\n'
        )
        return 1
    build_server().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
