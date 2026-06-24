# Agent overview

For coding agents (or scripts) that need to decide *where* to launch a job,
there's a one-shot overview that answers, in a single call:

- how many jobs you have **queued / running**, per cluster and per partition,
- how many **CPUs and GPUs are still free** — broken down **by GPU type**
  (a100/h100/…) and with the largest free block on a **single node**, and
- an **approximate queueing time**: SLURM's estimated start for your pending
  jobs, plus a per-partition pre-submission wait estimate.

## Usage

```bash
cluster-jobs --overview          # human-readable capacity table
cluster-jobs --overview --json   # machine-readable JSON (for agents)
cluster-jobs --overview --demo --json   # try it with synthetic data
```

This is the only place the tool runs `sinfo` and a cluster-wide `squeue` (still
read-only). All of it is folded into the **same SSH round-trip** as
`squeue --me`, so an overview is one connection per host.

Prefer to call it natively from an agent? The same overview is exposed over
[MCP](mcp.md) as the `cluster_overview` tool.

## The JSON shape

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
          "is_default": false,
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

## Field notes and caveats

!!! warning "Read these before trusting the numbers"

- **`cpus.free` is sinfo's *idle* CPU count** (down/drained nodes already
  excluded). `gpus.free` (and `by_type`/`max_free_per_node`) is counted only on
  **usable** nodes (idle/mixed/allocated).
- **`max_free_per_node`** tells you whether a multi-GPU job fits on **one**
  node — a cluster can have many free GPUs total yet none of them on a single
  box.
- **Shared-node dedupe:** a node shared between partitions counts toward each
  partition's tally but is counted **once** in the cluster-level
  `free`/`capacity` totals.
- **`is_default`** flags the partition that untargeted jobs (`sbatch` without
  `-p`) land on.
- **`my_running` / `my_pending` and `my_pending_jobs[].est_start`** come from
  `squeue --me`. `est_start` is SLURM's backfill estimate and is `null` until
  the scheduler computes it. The per-partition `queue` block comes from a
  cluster-wide `squeue` (all users).
- **`queue.wait_estimate` is a hint, not a promise:** `immediate` when GPUs are
  free now, else `~<t>` / `>=<t>` derived from the soonest-finishing running job
  (`soonest_free_sec`) and the pending depth. It's optimistic — it doesn't model
  scheduler priority — so treat it as "ballpark".
