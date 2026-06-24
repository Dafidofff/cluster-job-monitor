# MCP

The [agent overview](agent-overview.md) is exposed over
[MCP](https://modelcontextprotocol.io) so an agent (Claude Code, Cursor, …) can
call it natively. The server is a **thin wrapper** over the same read-only
collector — it adds **no new ways to touch the clusters**, it just surfaces the
existing `squeue`/`sinfo` data as MCP tools.

## Install the SDK

The MCP SDK is optional — the core collector and TUI don't need it. Install it
only when you want to run the server:

```bash
pip install "cluster-job-monitor[mcp]"
```

## Register with Claude Code

Point `CLUSTER_MONITOR_CONFIG` at your config (otherwise the server uses
`clusters.json` in the current working directory):

```bash
claude mcp add cluster-monitor \
  -e CLUSTER_MONITOR_CONFIG=/abs/path/to/clusters.json \
  -- python -m cluster_job_monitor.mcp_server
```

The server speaks the stdio transport, so you can also run it directly to test:

```bash
python -m cluster_job_monitor.mcp_server
```

(From a source checkout, `python mcp_server.py` is an equivalent shim.)

## Tools

| tool               | returns                                                              |
|--------------------|---------------------------------------------------------------------|
| `cluster_overview` | the [overview JSON](agent-overview.md#the-json-shape) — free CPUs/GPUs + your jobs, per cluster & partition |
| `my_jobs`          | just your jobs per cluster (skips `sinfo`, lighter)                  |

- **`cluster_overview`** — queued/running jobs and free CPUs/GPUs, per cluster
  and per partition. Call it before submitting work to see where there is
  capacity. Runs only `squeue --me` and `sinfo` on each host.
- **`my_jobs`** — your current jobs across every configured cluster, with no
  capacity data. Lighter than `cluster_overview` (skips the `sinfo` call); use
  it when you only need to know what you have queued, not where free GPUs are.

Both tools return a config/error object (rather than raising) if the config file
is missing or invalid, so an agent gets a structured error instead of a crash.
