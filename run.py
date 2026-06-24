#!/usr/bin/env python3
"""Back-compat shim: ``python run.py ...`` still works after packaging.

The real entry point now lives in ``cluster_job_monitor.cli`` (installed as the
``cluster-jobs`` console script). This file is kept so existing deployments that
invoke ``python run.py`` keep working unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when run straight from a checkout (no install).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_job_monitor.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
