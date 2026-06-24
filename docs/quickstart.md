# Quick start

## Install

```bash
pip install cluster-job-monitor
```

!!! tip "SSL error on `pip install`?"
    If `pip install` fails with an SSL / "ssl module is not available" error,
    your default `python3` was built without OpenSSL. Use Homebrew's instead:
    `/opt/homebrew/bin/python3 -m venv .venv && source .venv/bin/activate`.

??? note "Run from source instead (for contributors)"
    ```bash
    git clone git@github.com:Dafidofff/cluster-job-monitor.git
    cd cluster-job-monitor
    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    ```
    From a clone, `python run.py …` is a thin shim equivalent to the
    `cluster-jobs …` command used below.

## Try it with no clusters (demo)

The fastest way to see the dashboard is the synthetic `--demo` data — no SSH,
no config, no cluster access needed.

```bash
cluster-jobs --demo          # live TUI with synthetic data
cluster-jobs --once --demo   # print one synthetic snapshot and exit
```

## Want the agent capacity overview?

The overview also works against the demo data, so you can see the JSON shape
without a cluster:

```bash
cluster-jobs --overview --demo          # human-readable capacity table
cluster-jobs --overview --demo --json   # machine-readable JSON (for agents)
```

See [Agent overview](agent-overview.md) for the full field-by-field breakdown.

## Optional: the MCP wrapper

To expose the overview to a coding agent over MCP, install the extra SDK:

```bash
pip install "cluster-job-monitor[mcp]"
```

The core collector and TUI do **not** depend on this — it's only needed to run
the MCP server. See the [MCP page](mcp.md) for how to register it with Claude
Code.

## Run it for real

Once the demo looks right, point the tool at your own clusters. That means
making each cluster a non-interactive SSH alias and writing a small
`clusters.json` in your working directory:

```bash
cluster-jobs                       # uses ./clusters.json
cluster-jobs --config ~/my-clusters.json
```

Full instructions — including the `clusters.json` schema — are on the
**[Configuration](configuration.md)** page.
