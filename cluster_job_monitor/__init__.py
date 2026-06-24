"""Read-only multi-cluster SLURM dashboard + agent capacity overview.

This package collects SLURM/GPU job and capacity data over SSH and exposes it
through a Textual terminal UI, a one-shot CLI, and an MCP server. The collector
is UI-agnostic and has no third-party dependencies, so its dataclasses (whose
``to_dict()`` is JSON-ready) can be reused anywhere.
"""

from .collector import (  # noqa: F401
    Job,
    Host,
    Partition,
    Snapshot,
    collect,
    collect_overview,
    build_overview,
    load_config,
)

__all__ = [
    "Job",
    "Host",
    "Partition",
    "Snapshot",
    "collect",
    "collect_overview",
    "build_overview",
    "load_config",
]

__version__ = "0.1.0"
