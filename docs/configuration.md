# Configuration

The tool reads a single JSON file — `clusters.json` by default, or whatever you
pass to `--config`. Start from the template:

```bash
cp clusters.example.json clusters.json
```

`clusters.json` is git-ignored, so your host list and aliases stay out of the
repo.

## SSH must work non-interactively first

Each cluster is referenced by its `~/.ssh/config` alias. Before configuring the
monitor, make sure that alias resolves and authenticates **without a prompt** —
the tool polls every 30 seconds, so an interactive password would be unusable.

```sshconfig
Host mycluster
    HostName login.mycluster.example.edu
    User myuser
    # Reuse one connection so polling every 30s is fast and doesn't re-auth:
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

!!! tip "Use ControlMaster"
    The `ControlMaster`/`ControlPath`/`ControlPersist` block above multiplexes a
    single SSH connection. Without it, every refresh re-authenticates, which is
    slow and can trip rate limits. With it, polling reuses one warm connection.

Test the alias:

```bash
ssh mycluster "squeue --me --noheader | head"
```

It should return instantly with no password prompt.

## The `clusters.json` schema

Top level:

| field             | type | default | description                                  |
|-------------------|------|---------|----------------------------------------------|
| `refresh_seconds` | int  | `30`    | TUI auto-refresh interval.                    |
| `hosts`           | list | —       | One entry per machine to monitor.             |

Each entry in `hosts`:

| field        | type        | default   | description                                                                 |
|--------------|-------------|-----------|-----------------------------------------------------------------------------|
| `name`       | string      | —         | Display name shown in the dashboard.                                        |
| `ssh`        | string/null | —         | The `~/.ssh/config` alias. Use `null` together with `"local": true`.        |
| `local`      | bool        | `false`   | `true` for the machine you run the tool on (commands run locally, no SSH).  |
| `scheduler`  | string      | `"slurm"` | `"slurm"` for SLURM hosts; `"gpu"` for a non-SLURM GPU box (see below).      |
| `partitions` | list        | `[]`      | Partition names to surface; informational for filtering/labels.             |
| `minimized`  | bool        | `false`   | Start this cluster collapsed to a one-line summary.                         |

Example:

```json
{
  "refresh_seconds": 30,
  "hosts": [
    {
      "name": "Snellius",
      "ssh": "snellius",
      "partitions": ["gpu", "gpu_a100"],
      "local": false
    },
    {
      "name": "Desktop",
      "ssh": null,
      "partitions": [],
      "local": true
    },
    {
      "name": "Workstation GPU",
      "ssh": "my-gpu-box",
      "scheduler": "gpu",
      "partitions": [],
      "local": false
    }
  ]
}
```

## Local host

Set `"local": true` (and `"ssh": null`) for the machine you run the tool on.
Commands run directly instead of over SSH, so you can include your own desktop
or login node alongside remote clusters.

## Minimized clusters

Add `"minimized": true` to start a cluster collapsed to a one-line summary
(`▸`). You can still toggle it live with the number keys or `m` — see the
[keybindings](#keybindings) below.

## Non-SLURM GPU hosts

Got a workstation or GPU box that isn't behind SLURM? Add
`"scheduler": "gpu"` to that host. Instead of `squeue` it runs `nvidia-smi` +
`ps` and shows **one row per GPU process** — the run name (from a
`--name`/`--run-name`/`--experiment` arg, else the script name), GPU memory
used, elapsed time, and the owner — plus a GPU utilisation/memory line. It works
through any login shell (fish, csh, …).

## Running

```bash
cluster-jobs                 # uses ./clusters.json
cluster-jobs --config ~/my-clusters.json
cluster-jobs --once          # print one snapshot and exit
```

## Keybindings

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

Each cluster shows a number (`1 ▾ Snellius`); press it to collapse that cluster
to a one-line summary (`▸`) and again to expand it.
