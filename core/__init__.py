"""Core, UI-agnostic SLURM job collection.

This package has no third-party dependencies so it can be reused by the
terminal UI today and by a web pusher later (see plan Phase 2).
"""

from .collector import Job, Host, Snapshot, collect, load_config  # noqa: F401
