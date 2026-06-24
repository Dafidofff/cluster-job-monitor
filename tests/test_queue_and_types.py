"""Tests for GPU-type breakdown, single-node pocket, and queue-time estimates."""

from __future__ import annotations

import types

import pytest

from cluster_job_monitor import collector
from cluster_job_monitor.collector import (
    Partition, _human_duration, build_overview, collect_overview,
    parse_gpus_by_type, parse_job_line, parse_queue_output, parse_sinfo_output,
)


# --------------------------------------------------------------------------- #
# GPU type parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tres,expected", [
    ("gpu:a100:4,gpu:h100:2", {"a100": 4, "h100": 2}),
    ("gres/gpu:a100:8", {"a100": 8}),
    ("gpu:8", {"gpu": 8}),                       # untyped
    ("gpu:1g.10gb:3", {"1g.10gb": 3}),           # MIG profile (dot in type)
    ("gpu:a100:2(IDX:0-1)", {"a100": 2}),        # GresUsed form
    ("cpu=8,mem=16G", {}),
    ("", {}),
])
def test_parse_gpus_by_type(tres, expected):
    assert parse_gpus_by_type(tres) == expected


def test_by_type_sums_to_total():
    s = "gpu:a100:4,gpu:h100:2,gpu:3"
    assert sum(parse_gpus_by_type(s).values()) == collector.parse_gpus(s)


@pytest.mark.parametrize("secs,expected", [
    (None, "?"), (0, "0m"), (90, "1m"), (3600, "1h"),
    (5400, "1h30m"), (86400, "1d"), (90000, "1d1h"),
])
def test_human_duration(secs, expected):
    assert _human_duration(secs) == expected


# --------------------------------------------------------------------------- #
# sinfo: by-type breakdown + single-node free pocket
# --------------------------------------------------------------------------- #
def test_sinfo_by_type_and_node_pocket():
    text = (
        "node01 mixed_part mixed 8/120/0/128 gpu:a100:8 gpu:a100:2\n"   # 6 a100 free
        "node02 mixed_part idle 0/128/0/128 gpu:h100:4 gpu:h100:0\n"    # 4 h100 free
        "node03 mixed_part drained 0/0/128/128 gpu:a100:8 gpu:a100:0\n"  # excluded
    )
    parts, _ = parse_sinfo_output(text)
    p = parts[0]
    assert p.gpus_by_type["a100"] == {"free": 6, "alloc": 2, "total": 16}  # node03 in total
    assert p.gpus_by_type["h100"] == {"free": 4, "alloc": 0, "total": 4}
    # Largest free block on a single usable node: node02 has 4 free h100.
    assert p.gpus_max_free_node == 6  # node01 has 6 free a100 (the biggest block)
    d = p.to_dict()
    assert d["gpus"]["by_type"]["h100"]["free"] == 4
    assert d["gpus"]["max_free_per_node"] == 6


# --------------------------------------------------------------------------- #
# cluster-wide queue parsing
# --------------------------------------------------------------------------- #
def test_parse_queue_output():
    text = (
        "gpu_a100|RUNNING|2:00:00\n"
        "gpu_a100|RUNNING|0:30:00\n"     # soonest free = 30m
        "gpu_a100|PENDING|1-00:00:00\n"  # pending: counts, time ignored
        "gpu_h100|PENDING|5:00\n"
        "junk line\n"
    )
    stats = parse_queue_output(text)
    assert stats["gpu_a100"] == {"pending": 1, "running": 2, "soonest_free_sec": 1800}
    assert stats["gpu_h100"] == {"pending": 1, "running": 0, "soonest_free_sec": None}


def test_parse_queue_output_multi_partition_pending():
    # A pending job may list several partitions; count it in each.
    stats = parse_queue_output("gpu,gpu_a100|PENDING|1:00:00\n")
    assert stats["gpu"]["pending"] == 1 and stats["gpu_a100"]["pending"] == 1


def test_default_partition_marker_stripped_and_matches():
    # sinfo marks the default partition with '*'; the name must be cleaned so
    # squeue-derived my-jobs and queue stats fold onto it.
    parts, _ = parse_sinfo_output("node01 capacity* idle 0/64/0/64 gpu:l4:8 gpu:l4:0\n")
    p = parts[0]
    assert p.name == "capacity" and p.is_default is True
    assert p.to_dict()["is_default"] is True
    from cluster_job_monitor.collector import _fold_queue, _fold_my_jobs, Job
    _fold_queue(parts, parse_queue_output("capacity|PENDING|1:00:00\n"))
    assert p.queue_pending == 1  # "capacity" (queue) matched "capacity" (cleaned)
    _fold_my_jobs(parts, [Job(jobid="1", name="n", state="PENDING",
                              partition="capacity", elapsed="0", time_limit="",
                              nodes=1, cpus=1, gpus=0, reason="", submit_time="")])
    assert p.my_pending == 1


# --------------------------------------------------------------------------- #
# wait estimate
# --------------------------------------------------------------------------- #
def test_wait_estimate_immediate_when_gpus_free():
    p = Partition(name="g", gpus_total=8, gpus_free=2)
    assert p.wait_estimate == "immediate"


def test_wait_estimate_cpu_partition_immediate():
    p = Partition(name="cpu", cpus_total=128, cpus_free=10, gpus_total=0)
    assert p.wait_estimate == "immediate"


def test_wait_estimate_uses_soonest_free_and_queue():
    full = Partition(name="g", gpus_total=8, gpus_free=0,
                     soonest_free_sec=5400, queue_pending=0)
    assert full.wait_estimate == "~1h30m"
    busy = Partition(name="g", gpus_total=8, gpus_free=0,
                     soonest_free_sec=5400, queue_pending=3)
    assert busy.wait_estimate == ">=1h30m (3 queued)"


def test_wait_estimate_unknown_without_data():
    p = Partition(name="g", gpus_total=8, gpus_free=0)
    assert p.wait_estimate == "unknown"


# --------------------------------------------------------------------------- #
# squeue %S (estimated start) parsing — old and new field counts
# --------------------------------------------------------------------------- #
def test_parse_job_line_with_start_time():
    line = "9|sweep|PENDING|gpu|0:00|2:00|1|4|gres/gpu:1|Priority|sub|2026-06-30T03:00:00"
    job = parse_job_line(line)
    assert job.start_time == "2026-06-30T03:00:00" and job.bucket == "pending"


def test_parse_job_line_backcompat_without_start_time():
    # 11-field line (pre-%S) still parses; start_time falls back to "".
    line = "1|a|RUNNING|p|1:00|2:00|1|1|gres/gpu:1|None|sub"
    job = parse_job_line(line)
    assert job is not None and job.start_time == ""


# --------------------------------------------------------------------------- #
# End-to-end through collect_overview (subprocess mocked)
# --------------------------------------------------------------------------- #
_SQUEUE = (
    "7|train|RUNNING|gpu_a100|1:00|2:00|1|8|gres/gpu:a100:2|node01|sub|N/A\n"
    "9|sweep|PENDING|gpu_h100|0:00|2:00|1|8|gres/gpu:h100:1|Priority|sub|2026-06-30T03:00:00\n"
)
_SINFO = (
    "node01 gpu_a100 mixed 8/120/0/128 gpu:a100:8 gpu:a100:2\n"
    "node02 gpu_a100 idle 0/128/0/128 gpu:a100:8 gpu:a100:0\n"
    "node03 gpu_h100 allocated 128/0/0/128 gpu:h100:4 gpu:h100:4\n"
)
_QUEUE = (
    "gpu_a100|RUNNING|3:00:00\n"
    "gpu_h100|RUNNING|1:30:00\n"
    "gpu_h100|PENDING|2:00:00\n"
    "gpu_h100|PENDING|2:00:00\n"
)
_COMBINED = _SQUEUE + "@@SINFO\n" + _SINFO + "@@QUEUE\n" + _QUEUE


def test_overview_end_to_end(monkeypatch):
    monkeypatch.setattr(collector.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(
                            returncode=0, stdout=_COMBINED, stderr=""))
    ov = collect_overview({"hosts": [{"name": "C", "ssh": "c"}]})
    c = ov["clusters"][0]

    # est-start surfaced for my pending job
    assert c["my_pending_jobs"] == [
        {"jobid": "9", "name": "sweep", "partition": "gpu_h100",
         "est_start": "2026-06-30T03:00:00"}
    ]

    parts = {p["name"]: p for p in c["partitions"]}
    a100, h100 = parts["gpu_a100"], parts["gpu_h100"]
    # a100: free now -> immediate; by-type + pocket present
    assert a100["gpus"]["free"] == 14 and a100["gpus"]["by_type"]["a100"]["free"] == 14
    assert a100["gpus"]["max_free_per_node"] == 8
    assert a100["queue"]["wait_estimate"] == "immediate"
    # h100: full -> estimate from soonest running (1h30m) with 2 queued ahead
    assert h100["gpus"]["free"] == 0
    assert h100["queue"]["pending"] == 2 and h100["queue"]["soonest_free_sec"] == 5400
    assert h100["queue"]["wait_estimate"] == ">=1h30m (2 queued)"


def test_overview_no_queue_section_is_safe(monkeypatch):
    # Older remote (no @@QUEUE) must still yield partitions, just no queue data.
    out = _SQUEUE + "@@SINFO\n" + _SINFO
    monkeypatch.setattr(collector.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(
                            returncode=0, stdout=out, stderr=""))
    ov = collect_overview({"hosts": [{"name": "C", "ssh": "c"}]})
    parts = {p["name"]: p for p in ov["clusters"][0]["partitions"]}
    assert parts["gpu_h100"]["queue"]["pending"] == 0
    # a100 still free -> immediate; h100 full + no queue data -> unknown
    assert parts["gpu_a100"]["queue"]["wait_estimate"] == "immediate"
    assert parts["gpu_h100"]["queue"]["wait_estimate"] == "unknown"
