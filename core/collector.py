"""Collect SLURM job info from one or more hosts over SSH.

Read-only by design: the only remote command ever run is ``squeue``. There is
no code path that can cancel, submit, or otherwise mutate a cluster.

The module is deliberately UI-agnostic — it returns plain dataclasses whose
``to_dict()`` produces a JSON-ready structure. The terminal UI consumes the
dataclasses directly; a future web pusher can ship ``Snapshot.to_dict()``
verbatim.
"""

from __future__ import annotations

import json
import re
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
# Collection
# --------------------------------------------------------------------------- #
def _build_squeue_cmd() -> str:
    return f"squeue --me --noheader -o '{SQUEUE_FORMAT}'"


def _collect_host(spec: dict) -> Host:
    name = spec.get("name", spec.get("ssh") or "host")
    host = Host(name=name)
    squeue = _build_squeue_cmd()

    if spec.get("local"):
        argv = ["/bin/sh", "-c", squeue]
    else:
        alias = spec.get("ssh")
        if not alias:
            host.ok = False
            host.error = "no 'ssh' alias and not marked 'local'"
            return host
        argv = ["ssh", *DEFAULT_SSH_OPTS, alias, squeue]

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
