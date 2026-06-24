#!/usr/bin/env python3
"""Back-compat shim: ``python mcp_server.py`` still works after packaging.

The real MCP server now lives in ``cluster_job_monitor.mcp_server``. This file
is kept so existing deployments (e.g. a ``claude mcp add ... python
/abs/path/mcp_server.py`` registration) keep working unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when run straight from a checkout (no install).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_job_monitor.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
