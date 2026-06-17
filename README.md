# Cluster Jobs — terminal dashboard

A read-only terminal dashboard that shows your SLURM jobs across several
clusters **and** your desktop in one view. It SSHes into each host, runs
`squeue --me`, and renders a live, colour-coded overview.

**Read-only by design:** the only command ever run on a cluster is `squeue`.
There is no code path that can cancel or submit jobs. It uses *your* existing
SSH config and keys — nothing new is exposed, no server, no stored secrets.

```
┌ CLUSTER JOBS    3 running   2 pending      updated 4s ago      filter: none ┐

╭ ● Snellius   2 run  2 pend   54 cpu  12 gpu ───────────────────────────────╮
│ ● 8123456  diffusion-pretrain   gpu_a100  2n 36c 8gpu  ███░░░░░░  23%  Running│
│ ● 8123460  sweep-seed-7         gpu_a100  1n 18c 4gpu  limit 1-00:00:00  Priority│
╰────────────────────────────────────────────────────────────────────────────╯
```

## Quick look (no clusters needed)

> If `pip install` fails with an SSL / "ssl module is not available" error,
> your default `python3` was built without OpenSSL. Use Homebrew's instead:
> `/opt/homebrew/bin/python3 -m venv .venv`.

```bash
git clone <your-repo-url> cluster-jobs && cd cluster-jobs
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run.py --demo          # live TUI with synthetic data
python run.py --once --demo   # print one synthetic snapshot and exit
```

## Real setup

1. **Make sure each cluster is an SSH alias you can reach non-interactively.**
   In `~/.ssh/config`, e.g.:

   ```sshconfig
   Host snellius
       HostName snellius.surf.nl
       User dwessels
       # Reuse one connection so polling every 30s is fast and doesn't re-auth:
       ControlMaster auto
       ControlPath ~/.ssh/cm-%r@%h:%p
       ControlPersist 10m
   ```

   Test it: `ssh snellius "squeue --me --noheader | head"` should return
   instantly with no password prompt.

2. **Create your config** from the template:

   ```bash
   cp clusters.example.json clusters.json
   ```

   Edit `clusters.json` — one entry per host. `ssh` is the `~/.ssh/config`
   alias; set `"local": true` for the machine you run the tool on (no SSH).
   `clusters.json` is git-ignored.

3. **Run it:**

   ```bash
   python run.py                 # uses ./clusters.json
   python run.py --config ~/my-clusters.json
   ```

## Keys

| key   | action                                   |
|-------|------------------------------------------|
| `r`   | refresh now                              |
| `f`   | cycle state filter (all → running → …)   |
| `c`   | cycle cluster filter                     |
| `p`   | cycle partition filter                   |
| `/`   | search by job name (Enter applies)       |
| `esc` | clear all filters                        |
| `q`   | quit                                     |

Auto-refresh interval is `refresh_seconds` in the config (default 30).

## Layout

```
cluster-jobs/
  run.py                 # entry point (--once, --demo, --config)
  core/collector.py      # UI-agnostic: SSH + squeue -> Snapshot dataclasses
  tui/app.py             # Textual app (live loop, filters, keybindings)
  tui/render.py          # Rich renderables (shared by TUI and --once)
  tui/sample.py          # synthetic snapshot for --demo
  clusters.example.json  # config template (copy to clusters.json)
```

`core/collector.py` has **no third-party dependencies** and returns a
`Snapshot` whose `.to_dict()` is JSON-ready — that's the seam for a future
web/phone dashboard (push the dict to an authenticated endpoint and render it
in a browser), without changing the collector.
