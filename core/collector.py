"""Collect job info from one or more hosts over SSH.

Read-only by design: the only remote commands ever run are ``squeue`` and
``sinfo`` (SLURM hosts) and ``nvidia-smi`` + ``ps`` (non-SLURM GPU hosts). There
is no code path that can cancel, submit, or otherwise mutate a host.

The module is deliberately UI-agnostic — it returns plain dataclasses whose
``to_dict()`` produces a JSON-ready structure. The terminal UI consumes the
dataclasses directly; a future web pusher can ship ``Snapshot.to_dict()``
verbatim.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from functools import partial
from pathlib import Path
from typing import Optional

# squeue output format. Pipe-delimited so it is trivial to parse and stable
# across SLURM versions. Field order MUST match _FIELDS below.
#   %i jobid   %j name        %T state     %P partition  %M elapsed
#   %l limit   %D nodes       %C cpus      %b tres/gpu   %R reason/nodelist
#   %V submit-time
SQUEUE_FORMAT = "%i|%j|%T|%P|%M|%l|%D|%C|%b|%R|%V"
_FIELDS = [
    "jobid", "name", "state", "partition", "elapsed",
    "time_limit", "nodes", "cpus", "tres", "reason", "submit_time",
]

# sinfo output for the agent capacity overview. One line per node *per
# partition* (-N), so it aggregates straight into per-partition tallies.
# Field values never contain whitespace, so the fixed widths split cleanly.
#   NodeHost   node name (used to dedupe shared nodes for cluster totals)
#   Partition  partition name        StateLong  node state (idle/mixed/...)
#   CPUsState  alloc/idle/other/total            Gres  configured GPUs
#   GresUsed   allocated GPUs
SINFO_FORMAT = "NodeHost:30,Partition:30,StateLong:20,CPUsState:25,Gres:50,GresUsed:50"
SINFO_SENTINEL = "@@SINFO"

# How states map to a coarse bucket used for summaries + colouring.
RUNNING_STATES = {"RUNNING"}
PENDING_STATES = {"PENDING"}

DEFAULT_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
# Per-host wall-clock cap so a hung connection can never stall the refresh.
HOST_TIMEOUT = 25


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    jobid: str
    name: str
    state: str
    partition: str
    elapsed: str
    time_limit: str
    nodes: int
    cpus: int
    gpus: int
    reason: str
    submit_time: str
    # Fraction of time limit consumed (0..1), or None if limit is unbounded.
    progress: Optional[float] = None
    # Set only for non-SLURM GPU hosts (process view): GPU memory used + owner.
    gpu_mem_mb: Optional[int] = None
    user: str = ""

    @property
    def bucket(self) -> str:
        if self.state in RUNNING_STATES:
            return "running"
        if self.state in PENDING_STATES:
            return "pending"
        return "other"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bucket"] = self.bucket
        return d


@dataclass
class Partition:
    """Capacity / availability for one SLURM partition on one cluster.

    ``cpus_free`` is the idle-CPU count straight from sinfo's CPUsState, so it
    already excludes down/drained nodes. ``gpus_free`` is counted only on
    usable nodes (idle/mixed/allocated). ``my_running``/``my_pending`` are this
    user's jobs in the partition (folded in from ``squeue --me``).
    """

    name: str
    cpus_free: int = 0
    cpus_alloc: int = 0
    cpus_total: int = 0
    gpus_free: int = 0
    gpus_alloc: int = 0
    gpus_total: int = 0
    nodes_idle: int = 0
    nodes_mixed: int = 0
    nodes_alloc: int = 0
    nodes_other: int = 0
    nodes_total: int = 0
    my_running: int = 0
    my_pending: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "my_running": self.my_running,
            "my_pending": self.my_pending,
            "cpus": {
                "free": self.cpus_free,
                "alloc": self.cpus_alloc,
                "total": self.cpus_total,
            },
            "gpus": {
                "free": self.gpus_free,
                "alloc": self.gpus_alloc,
                "total": self.gpus_total,
            },
            "nodes": {
                "idle": self.nodes_idle,
                "mixed": self.nodes_mixed,
                "alloc": self.nodes_alloc,
                "other": self.nodes_other,
                "total": self.nodes_total,
            },
        }


@dataclass
class Host:
    name: str
    ok: bool = True
    error: Optional[str] = None
    jobs: list[Job] = field(default_factory=list)
    kind: str = "slurm"  # "slurm" | "gpu"
    note: str = ""       # freeform status line (e.g. GPU util/memory)
    # Capacity overview (populated only when collected with_partitions). The
    # cluster-level totals dedupe nodes shared across partitions.
    partitions: list[Partition] = field(default_factory=list)
    cpus_free: int = 0
    cpus_total: int = 0
    gpus_free: int = 0
    gpus_total: int = 0

    @property
    def running(self) -> int:
        return sum(1 for j in self.jobs if j.bucket == "running")

    @property
    def pending(self) -> int:
        return sum(1 for j in self.jobs if j.bucket == "pending")

    @property
    def other(self) -> int:
        return sum(1 for j in self.jobs if j.bucket == "other")

    @property
    def cpus_in_use(self) -> int:
        return sum(j.cpus for j in self.jobs if j.bucket == "running")

    @property
    def gpus_in_use(self) -> int:
        return sum(j.gpus for j in self.jobs if j.bucket == "running")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "error": self.error,
            "kind": self.kind,
            "note": self.note,
            "summary": {
                "running": self.running,
                "pending": self.pending,
                "other": self.other,
                "cpus_in_use": self.cpus_in_use,
                "gpus_in_use": self.gpus_in_use,
            },
            "capacity": {
                "cpus_free": self.cpus_free,
                "cpus_total": self.cpus_total,
                "gpus_free": self.gpus_free,
                "gpus_total": self.gpus_total,
            },
            "partitions": [p.to_dict() for p in self.partitions],
            "jobs": [j.to_dict() for j in self.jobs],
        }


@dataclass
class Snapshot:
    generated_at: float
    hosts: list[Host] = field(default_factory=list)

    @property
    def total_running(self) -> int:
        return sum(h.running for h in self.hosts)

    @property
    def total_pending(self) -> int:
        return sum(h.pending for h in self.hosts)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "totals": {
                "running": self.total_running,
                "pending": self.total_pending,
            },
            "hosts": [h.to_dict() for h in self.hosts],
        }


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
_UNBOUNDED = {"", "UNLIMITED", "INVALID", "NOT_SET", "N/A"}
_GPU_RE = re.compile(r"gpu(?::[A-Za-z0-9_]+)*:(\d+)", re.IGNORECASE)


def parse_slurm_time(value: str) -> Optional[int]:
    """Parse a SLURM duration ([D-]HH:MM:SS / MM:SS / SS) into seconds."""
    value = (value or "").strip()
    if value in _UNBOUNDED:
        return None
    days = 0
    if "-" in value:
        d, value = value.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            return None
    try:
        parts = [int(p) for p in value.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, s = 0, 0, parts[0]
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + s


def parse_gpus(tres: str) -> int:
    """Best-effort GPU count from a TRES/GRES string like 'gres/gpu:a100:2'."""
    if not tres:
        return 0
    return sum(int(n) for n in _GPU_RE.findall(tres))


def _to_int(value: str) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def parse_job_line(line: str) -> Optional[Job]:
    """Parse one pipe-delimited squeue line into a Job (None if malformed)."""
    parts = line.rstrip("\n").split("|")
    if len(parts) < len(_FIELDS):
        return None
    raw = dict(zip(_FIELDS, parts))

    elapsed_s = parse_slurm_time(raw["elapsed"])
    limit_s = parse_slurm_time(raw["time_limit"])
    progress: Optional[float] = None
    if elapsed_s is not None and limit_s and limit_s > 0:
        progress = max(0.0, min(1.0, elapsed_s / limit_s))

    return Job(
        jobid=raw["jobid"].strip(),
        name=raw["name"].strip(),
        state=raw["state"].strip().upper(),
        partition=raw["partition"].strip(),
        elapsed=raw["elapsed"].strip(),
        time_limit=raw["time_limit"].strip(),
        nodes=_to_int(raw["nodes"].strip()),
        cpus=_to_int(raw["cpus"].strip()),
        gpus=parse_gpus(raw["tres"].strip()),
        reason=raw["reason"].strip(),
        submit_time=raw["submit_time"].strip(),
        progress=progress,
    )


def parse_squeue_output(text: str) -> list[Job]:
    jobs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        job = parse_job_line(line)
        if job is not None:
            jobs.append(job)
    return jobs


# Node states we treat as having usable (allocatable) GPUs. Down/drained nodes
# may report 0 GPUs used while their hardware is unavailable, so we exclude
# them from free-GPU counts.
_USABLE_NODE_STATES = {"idle", "mixed", "allocated"}


def _norm_node_state(raw: str) -> str:
    """Collapse a sinfo node state into idle/mixed/allocated/other.

    sinfo may append flag characters (``*~#$@+``) to the state; strip them.
    """
    s = (raw or "").strip().lower().rstrip("*~#$@+-.")
    if s.startswith("idle"):
        return "idle"
    if s.startswith("mix"):
        return "mixed"
    if s.startswith("alloc"):
        return "allocated"
    return "other"


def parse_sinfo_output(text: str) -> tuple[list[Partition], dict]:
    """Parse ``sinfo -N -O SINFO_FORMAT`` into per-partition capacity.

    Returns ``(partitions, host_totals)`` where ``host_totals`` holds
    cluster-wide free/total CPUs and GPUs with shared nodes counted once.
    Malformed lines are skipped, so partial output never raises.
    """
    parts: dict[str, Partition] = {}
    seen_nodes: dict[str, tuple[int, int, int, int]] = {}
    for line in text.splitlines():
        toks = line.split()
        if len(toks) < 6:
            continue
        node, pname, raw_state, cpustate, gres, gres_used = toks[:6]

        cstate = cpustate.split("/")
        if len(cstate) == 4:
            alloc_c, idle_c, _other_c, total_c = (_to_int(x) for x in cstate)
        else:
            alloc_c = idle_c = total_c = 0

        gpu_total = parse_gpus(gres)
        gpu_used = parse_gpus(gres_used)
        state = _norm_node_state(raw_state)
        gpu_free = max(0, gpu_total - gpu_used) if state in _USABLE_NODE_STATES else 0

        p = parts.get(pname)
        if p is None:
            p = parts[pname] = Partition(name=pname)
        p.cpus_free += idle_c
        p.cpus_alloc += alloc_c
        p.cpus_total += total_c
        p.gpus_free += gpu_free
        p.gpus_alloc += gpu_used
        p.gpus_total += gpu_total
        p.nodes_total += 1
        if state == "idle":
            p.nodes_idle += 1
        elif state == "mixed":
            p.nodes_mixed += 1
        elif state == "allocated":
            p.nodes_alloc += 1
        else:
            p.nodes_other += 1

        # First time we see a node, record its contribution to cluster totals
        # (a node in N partitions appears N times but must count once).
        if node not in seen_nodes:
            seen_nodes[node] = (idle_c, total_c, gpu_free, gpu_total)

    host_totals = {
        "cpus_free": sum(v[0] for v in seen_nodes.values()),
        "cpus_total": sum(v[1] for v in seen_nodes.values()),
        "gpus_free": sum(v[2] for v in seen_nodes.values()),
        "gpus_total": sum(v[3] for v in seen_nodes.values()),
    }
    return list(parts.values()), host_totals


def _fold_my_jobs(partitions: list[Partition], jobs: list[Job]) -> None:
    """Add each job's running/pending count to its partition(s) in place.

    A pending job may list several partitions (``gpu,gpu_a100``); count it in
    each. Partitions named by a job but absent from sinfo are appended with
    zero capacity so the user's queue is never dropped.
    """
    by_name = {p.name: p for p in partitions}
    for job in jobs:
        for pname in job.partition.split(","):
            pname = pname.strip()
            if not pname:
                continue
            p = by_name.get(pname)
            if p is None:
                p = by_name[pname] = Partition(name=pname)
                partitions.append(p)
            if job.bucket == "running":
                p.my_running += 1
            elif job.bucket == "pending":
                p.my_pending += 1


# --------------------------------------------------------------------------- #
# Non-SLURM GPU hosts (nvidia-smi + ps)
# --------------------------------------------------------------------------- #
# Single SSH round-trip: emit a GPU inventory section then one line per GPU
# process, joining each PID to its command/elapsed/user via `ps`. Read-only.
GPU_CMD = (
    'echo "@@GPUS"; '
    'nvidia-smi --query-gpu=uuid,index,name,utilization.gpu,memory.used,memory.total '
    '--format=csv,noheader,nounits 2>/dev/null; '
    'echo "@@PROCS"; '
    'nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory '
    '--format=csv,noheader,nounits 2>/dev/null | '
    'while IFS="," read -r pid uuid mem; do '
    'pid=$(echo "$pid" | tr -d " "); [ -z "$pid" ] && continue; '
    'uuid=$(echo "$uuid" | tr -d " "); mem=$(echo "$mem" | tr -d " "); '
    'et=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d " "); '
    'us=$(ps -o user= -p "$pid" 2>/dev/null | tr -d " "); '
    'ar=$(ps -o args= -p "$pid" 2>/dev/null | tr "|" "/" | tr "\\n" " "); '
    'echo "${pid}|${uuid}|${mem}|${et}|${us}|${ar}"; '
    'done'
)

# Flags ML scripts commonly use to name a run; value becomes the display name.
_RUN_NAME_FLAGS = {
    "--name", "--run-name", "--run_name", "--exp", "--exp-name", "--exp_name",
    "--experiment", "--experiment-name", "--experiment_name", "--job-name",
    "--job_name", "-n",
}
_INTERPRETERS = {"python", "python3", "python2", "sh", "bash", "torchrun"}


def gpu_run_label(args: str) -> str:
    """Derive a readable run name from a process command line."""
    toks = args.split()
    if not toks:
        return "(unknown)"
    # 1) explicit run-name flag (--name foo  or  --name=foo)
    for i, tok in enumerate(toks):
        if "=" in tok:
            key, val = tok.split("=", 1)
            if key in _RUN_NAME_FLAGS and val:
                return val
        elif tok in _RUN_NAME_FLAGS and i + 1 < len(toks):
            return toks[i + 1]
    # 2) first script argument (e.g. train.py)
    for tok in toks:
        if tok.endswith(".py"):
            return os.path.basename(tok)
    # 3) fall back to the executable basename, skipping the interpreter
    base = os.path.basename(toks[0])
    if base in _INTERPRETERS and len(toks) > 1:
        return os.path.basename(toks[1])
    return base


def parse_gpu_output(text: str) -> tuple[list[dict], list[Job]]:
    """Parse GPU_CMD output into (gpu inventory, aggregated process Jobs)."""
    gpus: list[dict] = []
    raw_procs: list[list[str]] = []
    section = None
    for line in text.splitlines():
        s = line.strip()
        if s == "@@GPUS":
            section = "gpus"
            continue
        if s == "@@PROCS":
            section = "procs"
            continue
        if not s:
            continue
        if section == "gpus":
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "uuid": parts[0], "index": parts[1], "name": parts[2],
                    "util": parts[3], "mem_used": parts[4], "mem_total": parts[5],
                })
        elif section == "procs":
            parts = s.split("|", 5)
            if len(parts) == 6:
                raw_procs.append(parts)

    uuid_to_index = {g["uuid"]: g["index"] for g in gpus}
    by_pid: dict[str, dict] = {}
    for pid, uuid, mem, etime, user, args in raw_procs:
        entry = by_pid.setdefault(
            pid, {"mem": 0, "gpus": set(), "etime": etime, "user": user, "args": args}
        )
        entry["mem"] += _to_int(mem)
        entry["gpus"].add(uuid_to_index.get(uuid, uuid))

    jobs = []
    for pid, e in by_pid.items():
        idxs = sorted(e["gpus"])
        partition = "GPU " + ",".join(idxs) if idxs else "GPU"
        jobs.append(Job(
            jobid=pid, name=gpu_run_label(e["args"]), state="RUNNING",
            partition=partition, elapsed=e["etime"], time_limit="",
            nodes=1, cpus=0, gpus=max(1, len(e["gpus"])), reason="",
            submit_time="", progress=None, gpu_mem_mb=e["mem"], user=e["user"],
        ))
    return gpus, jobs


def gpu_note(gpus: list[dict]) -> str:
    """One-line GPU status (shown even when the GPU is idle)."""
    if not gpus:
        return ""
    used = sum(_to_int(g["mem_used"]) for g in gpus)
    total = sum(_to_int(g["mem_total"]) for g in gpus)
    utils = [_to_int(g["util"]) for g in gpus]
    avg_util = sum(utils) // len(utils) if utils else 0
    used_gb, total_gb = used / 1024, total / 1024
    if len(gpus) == 1:
        name = gpus[0]["name"].replace("NVIDIA ", "").replace("GeForce ", "")
        return f"{name} · {avg_util}% util · {used_gb:.1f}/{total_gb:.0f} GB"
    return f"{len(gpus)}× GPU · {avg_util}% util · {used_gb:.1f}/{total_gb:.0f} GB"


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def _build_squeue_cmd(with_partitions: bool = False) -> str:
    squeue = f"squeue --me --noheader -o '{SQUEUE_FORMAT}'"
    if not with_partitions:
        return squeue
    # Same single SSH round-trip: emit jobs, a sentinel, then partition
    # capacity. ``exit $rc`` propagates squeue's status so a squeue failure is
    # still surfaced, while a missing/failing sinfo only loses capacity data.
    sinfo = f"sinfo -h -N -O '{SINFO_FORMAT}'"
    return (f"{squeue}; rc=$?; echo '{SINFO_SENTINEL}'; "
            f"{sinfo} 2>/dev/null || true; exit $rc")


def _collect_host(spec: dict, with_partitions: bool = False) -> Host:
    name = spec.get("name", spec.get("ssh") or "host")
    scheduler = spec.get("scheduler", "slurm").lower()
    host = Host(name=name, kind="gpu" if scheduler in ("gpu", "nvidia") else "slurm")
    remote_cmd = GPU_CMD if host.kind == "gpu" else _build_squeue_cmd(with_partitions)

    if spec.get("local"):
        argv = ["/bin/sh", "-c", remote_cmd]
    else:
        alias = spec.get("ssh")
        if not alias:
            host.ok = False
            host.error = "no 'ssh' alias and not marked 'local'"
            return host
        # Force POSIX sh: the remote login shell may be fish/csh, which can't
        # parse our command. shlex.quote keeps this safe across shells.
        argv = ["ssh", *DEFAULT_SSH_OPTS, alias, "/bin/sh -c " + shlex.quote(remote_cmd)]

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=HOST_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        host.ok = False
        host.error = f"timed out after {HOST_TIMEOUT}s"
        return host
    except FileNotFoundError as exc:
        host.ok = False
        host.error = f"command not found: {exc.filename}"
        return host
    except OSError as exc:
        host.ok = False
        host.error = str(exc)
        return host

    if proc.returncode != 0:
        host.ok = False
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        host.error = err[-1] if err else f"exit code {proc.returncode}"
        return host

    if host.kind == "gpu":
        gpus, jobs = parse_gpu_output(proc.stdout)
        if not gpus:
            host.ok = False
            host.error = "nvidia-smi returned no GPUs (driver/tool missing?)"
            return host
        host.note = gpu_note(gpus)
        host.jobs = jobs
        # A GPU is "free" if no compute process is bound to its index.
        busy = set()
        for job in jobs:
            for idx in job.partition.replace("GPU", "").split(","):
                idx = idx.strip()
                if idx:
                    busy.add(idx)
        host.gpus_total = len(gpus)
        host.gpus_free = sum(1 for g in gpus if g["index"] not in busy)
    else:
        out = proc.stdout
        if with_partitions and SINFO_SENTINEL in out:
            squeue_text, sinfo_text = out.split(SINFO_SENTINEL, 1)
        else:
            squeue_text, sinfo_text = out, ""
        host.jobs = parse_squeue_output(squeue_text)
        if with_partitions:
            partitions, totals = parse_sinfo_output(sinfo_text)
            _fold_my_jobs(partitions, host.jobs)
            host.partitions = partitions
            host.cpus_free = totals["cpus_free"]
            host.cpus_total = totals["cpus_total"]
            host.gpus_free = totals["gpus_free"]
            host.gpus_total = totals["gpus_total"]
    return host


def collect(config: dict, max_workers: Optional[int] = None,
            with_partitions: bool = False) -> Snapshot:
    """Poll every host in ``config`` concurrently and return a Snapshot.

    Set ``with_partitions`` to also query ``sinfo`` for per-partition capacity
    (used by the agent overview). The live TUI leaves it off to stay light.
    """
    hosts_cfg = config.get("hosts", [])
    workers = max_workers or max(1, len(hosts_cfg))
    results: list[Host] = []
    if hosts_cfg:
        worker = partial(_collect_host, with_partitions=with_partitions)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Preserve config order in the output.
            results = list(pool.map(worker, hosts_cfg))
    return Snapshot(generated_at=time.time(), hosts=results)


def build_overview(snapshot: Snapshot) -> dict:
    """Reshape a Snapshot into the agent-facing capacity overview.

    Answers "what's queued/running and what's free, per cluster and
    partition" in a compact JSON-ready dict.
    """
    clusters = []
    for h in snapshot.hosts:
        clusters.append({
            "name": h.name,
            "ok": h.ok,
            "error": h.error,
            "kind": h.kind,
            "my_jobs": {"running": h.running, "pending": h.pending},
            "free": {"cpus": h.cpus_free, "gpus": h.gpus_free},
            "capacity": {"cpus": h.cpus_total, "gpus": h.gpus_total},
            "partitions": [p.to_dict() for p in h.partitions],
        })
    return {"generated_at": snapshot.generated_at, "clusters": clusters}


def collect_overview(config: dict, max_workers: Optional[int] = None) -> dict:
    """Collect a capacity overview (jobs + free CPUs/GPUs) for every cluster."""
    snapshot = collect(config, max_workers=max_workers, with_partitions=True)
    return build_overview(snapshot)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if "hosts" not in data or not isinstance(data["hosts"], list):
        raise ValueError("config must contain a 'hosts' array")
    data.setdefault("refresh_seconds", 30)
    return data
