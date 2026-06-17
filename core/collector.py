"""Collect job info from one or more hosts over SSH.

Read-only by design: the only remote commands ever run are ``squeue`` (SLURM
hosts) and ``nvidia-smi`` + ``ps`` (non-SLURM GPU hosts). There is no code path
that can cancel, submit, or otherwise mutate a host.

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
class Host:
    name: str
    ok: bool = True
    error: Optional[str] = None
    jobs: list[Job] = field(default_factory=list)
    kind: str = "slurm"  # "slurm" | "gpu"
    note: str = ""       # freeform status line (e.g. GPU util/memory)

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
def _build_squeue_cmd() -> str:
    return f"squeue --me --noheader -o '{SQUEUE_FORMAT}'"


def _collect_host(spec: dict) -> Host:
    name = spec.get("name", spec.get("ssh") or "host")
    scheduler = spec.get("scheduler", "slurm").lower()
    host = Host(name=name, kind="gpu" if scheduler in ("gpu", "nvidia") else "slurm")
    remote_cmd = GPU_CMD if host.kind == "gpu" else _build_squeue_cmd()

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
    else:
        host.jobs = parse_squeue_output(proc.stdout)
    return host


def collect(config: dict, max_workers: Optional[int] = None) -> Snapshot:
    """Poll every host in ``config`` concurrently and return a Snapshot."""
    hosts_cfg = config.get("hosts", [])
    workers = max_workers or max(1, len(hosts_cfg))
    results: list[Host] = []
    if hosts_cfg:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Preserve config order in the output.
            results = list(pool.map(_collect_host, hosts_cfg))
    return Snapshot(generated_at=time.time(), hosts=results)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if "hosts" not in data or not isinstance(data["hosts"], list):
        raise ValueError("config must contain a 'hosts' array")
    data.setdefault("refresh_seconds", 30)
    return data
