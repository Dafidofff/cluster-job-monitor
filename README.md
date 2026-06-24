# Cluster Jobs — terminal dashboard

[![CI](https://github.com/Dafidofff/cluster-job-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/Dafidofff/cluster-job-monitor/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Dafidofff/cluster-job-monitor/branch/main/graph/badge.svg)](https://codecov.io/gh/Dafidofff/cluster-job-monitor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A read-only terminal dashboard that shows your SLURM jobs across several
clusters **and** your desktop in one view. It SSHes into each host, runs
`squeue --me`, and renders a live, colour-coded overview.

**Read-only by design:** the only commands ever run are `squeue` and `sinfo`
(SLURM hosts) and `nvidia-smi` + `ps` (non-SLURM GPU hosts). There is no code
path that can cancel or submit jobs. It uses *your* existing SSH config and
keys — nothing new is exposed, no server, no stored secrets.

```
┌ CLUSTER JOBS    3 running   2 pending      updated 4s ago      filter: none ┐

╭ 1 ▾ ● Snellius   2 run  2 pend   54 cpu  12 gpu ───────────────────────────╮
│ ● 8123456  diffusion-pretrain   gpu_a100  2n 36c 8gpu  ███░░░░░░  23%  Running│
│ ● 8123460  sweep-seed-7         gpu_a100  1n 18c 4gpu  limit 1-00:00:00  Priority│
╰────────────────────────────────────────────────────────────────────────────╯
  2 ▸ ● Hipster   4 run  7 pend   192 cpu  14 gpu        (press 2 to expand)
```

## Quick look (no clusters needed)

> If `pip install` fails with an SSL / "ssl module is not available" error,
> your default `python3` was built without OpenSSL. Use Homebrew's instead:
> `/opt/homebrew/bin/python3 -m venv .venv`.

```bash
git clone git@github.com:Dafidofff/cluster-job-monitor.git && cd cluster-job-monitor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run.py --demo          # live TUI with synthetic data
python run.py --once --demo   # print one synthetic snapshot and exit
```

## Real setup

1. **Make sure each cluster is an SSH alias you can reach non-interactively.**
   In `~/.ssh/config`, e.g.:

   ```sshconfig
   Host mycluster
       HostName login.mycluster.example.edu
       User myuser
       # Reuse one connection so polling every 30s is fast and doesn't re-auth:
       ControlMaster auto
       ControlPath ~/.ssh/cm-%r@%h:%p
       ControlPersist 10m
   ```

   Test it: `ssh mycluster "squeue --me --noheader | head"` should return
   instantly with no password prompt.

2. **Create your config** from the template:

   ```bash
   cp clusters.example.json clusters.json
   ```

   Edit `clusters.json` — one entry per host. `ssh` is the `~/.ssh/config`
   alias; set `"local": true` for the machine you run the tool on (no SSH).
   Add `"minimized": true` to start a cluster collapsed (see below).
   `clusters.json` is git-ignored.

   **Non-SLURM GPU box?** Add `"scheduler": "gpu"` to that host. Instead of
   `squeue` it runs `nvidia-smi` + `ps` and shows one row per GPU process —
   the run name (from a `--name`/`--run-name`/`--experiment` arg, else the
   script name), GPU memory used, elapsed time, and the owner — plus a GPU
   utilisation/memory line. Works through any login shell (fish, csh, …).

3. **Run it:**

   ```bash
   python run.py                 # uses ./clusters.json
   python run.py --config ~/my-clusters.json
   ```

## Agent overview (capacity, per cluster & partition)

For coding agents (or scripts) that need to decide *where* to launch a job,
there's a one-shot overview that answers, in a single call:

- how many jobs you have **queued / running**, per cluster and per partition,
- how many **CPUs and GPUs are still free** — broken down **by GPU type**
  (a100/h100/…) and with the largest free block on a **single node**, and
- an **approximate queueing time**: SLURM's estimated start for your pending
  jobs, plus a per-partition pre-submission wait estimate.

```bash
python run.py --overview          # human-readable capacity table
python run.py --overview --json   # machine-readable JSON (for agents)
python run.py --overview --demo --json   # try it with synthetic data
```

This is the only place the tool runs `sinfo` and a cluster-wide `squeue` (still
read-only). All of it is folded into the **same SSH round-trip** as
`squeue --me`, so an overview is one connection per host. The JSON shape:

```json
{
  "generated_at": 1718900000.0,
  "clusters": [
    {
      "name": "Snellius", "ok": true, "error": null, "kind": "slurm",
      "my_jobs": { "running": 2, "pending": 1 },
      "my_pending_jobs": [
        { "jobid": "8123460", "name": "sweep-7", "partition": "gpu_a100",
          "est_start": "2026-06-30T03:00:00" }
      ],
      "free":     { "cpus": 224, "gpus": 14 },
      "capacity": { "cpus": 768, "gpus": 48 },
      "partitions": [
        {
          "name": "gpu_a100", "my_running": 1, "my_pending": 2,
          "cpus":  { "free": 0, "alloc": 512, "total": 512 },
          "gpus":  {
            "free": 0, "alloc": 32, "total": 32,
            "by_type": { "a100": { "free": 0, "alloc": 32, "total": 32 } },
            "max_free_per_node": 0
          },
          "nodes": { "idle": 0, "mixed": 0, "alloc": 8, "other": 0, "total": 8 },
          "queue": {
            "pending": 11, "running": 8,
            "soonest_free_sec": 9300,
            "wait_estimate": ">=2h35m (11 queued)"
          }
        }
      ]
    }
  ]
}
```

Notes:
- `cpus.free` is sinfo's *idle* CPU count (down/drained nodes already excluded);
  `gpus.free` (and `by_type`/`max_free_per_node`) is counted only on usable
  nodes (idle/mixed/allocated). `max_free_per_node` tells you whether a
  multi-GPU job fits on one node.
- A node shared between partitions counts toward each partition's tally but is
  counted **once** in the cluster-level `free`/`capacity` totals.
- `my_running`/`my_pending` and `my_pending_jobs[].est_start` come from
  `squeue --me` (`est_start` is SLURM's backfill estimate, `null` until it's
  computed). The per-partition `queue` block comes from a cluster-wide `squeue`
  (all users).
- `queue.wait_estimate` is a **hint, not a promise**: `immediate` when GPUs are
  free now, else `~<t>`/`>=<t>` derived from the soonest-finishing running job
  (`soonest_free_sec`) and the pending depth. It's optimistic — it doesn't model
  scheduler priority — so treat it as "ballpark".

### As an MCP tool

The same overview is exposed over MCP so an agent can call it natively, via a
thin wrapper that adds no new cluster access:

```bash
pip install -r requirements-mcp.txt        # installs the MCP SDK

# Register with Claude Code (point CLUSTER_MONITOR_CONFIG at your config):
claude mcp add cluster-monitor \
  -e CLUSTER_MONITOR_CONFIG=/abs/path/to/clusters.json \
  -- python /abs/path/to/mcp_server.py
```

It serves two tools:

| tool               | returns                                                         |
|--------------------|----------------------------------------------------------------|
| `cluster_overview` | the JSON above — free CPUs/GPUs + your jobs, per cluster & part |
| `my_jobs`          | just your jobs per cluster (skips `sinfo`, lighter)            |

## Keys

| key     | action                                          |
|---------|-------------------------------------------------|
| `r`     | refresh now                                     |
| `f`     | cycle state filter (all → running → …)          |
| `c`     | cycle cluster filter                            |
| `p`     | cycle partition filter                          |
| `/`     | search by job name (Enter applies)              |
| `1`–`9` | collapse / expand the cluster with that number  |
| `m`     | collapse / expand **all** clusters              |
| `esc`   | clear all filters                               |
| `q`     | quit                                            |

Each cluster shows a number (`1 ▾ Snellius`); press it to collapse that
cluster to a one-line summary (`▸`) and again to expand it. Start a cluster
collapsed by adding `"minimized": true` to its entry in `clusters.json`.

Auto-refresh interval is `refresh_seconds` in the config (default 30).

## Layout

```
cluster-jobs/
  run.py                 # entry point (--once, --overview, --json, --demo, --config)
  core/collector.py      # UI-agnostic: SSH + squeue/sinfo -> Snapshot dataclasses
  mcp_server.py          # MCP wrapper exposing the capacity overview to agents
  tui/app.py             # Textual app (live loop, filters, keybindings)
  tui/render.py          # Rich renderables (shared by TUI, --once, --overview)
  tui/sample.py          # synthetic snapshot for --demo
  clusters.example.json  # config template (copy to clusters.json)
```

`core/collector.py` has **no third-party dependencies** and returns a
`Snapshot` whose `.to_dict()` is JSON-ready — that's the seam for a future
web/phone dashboard (push the dict to an authenticated endpoint and render it
in a browser), without changing the collector.

## Development

```bash
pip install -r requirements-dev.txt
pytest                       # run the suite
pytest --cov --cov-report=term-missing   # with coverage (~94%)
```

Tests live in `tests/` and mock SSH/`subprocess`, so they run anywhere — no
cluster access needed. CI ([GitHub Actions](.github/workflows/ci.yml)) runs
them on Python 3.10–3.12 and reports coverage to Codecov.

## License

[MIT](LICENSE) © David Wessels. You're free to use, modify, and redistribute
it; just keep the copyright notice and license text in any copies.
