"""Tests for the capacity overview: sinfo parsing, folding, build_overview."""

from __future__ import annotations

import json
import subprocess
import types

import pytest

from cluster_job_monitor import collector
from cluster_job_monitor.collector import (
    Host, Job, Partition, Snapshot, build_overview, collect, collect_overview,
    parse_sinfo_output,
)
from cluster_job_monitor.collector import _fold_my_jobs, _norm_node_state


# --------------------------------------------------------------------------- #
# Node-state normalisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("idle", "idle"),
    ("IDLE", "idle"),
    ("idle*", "idle"),         # unreachable flag stripped
    ("mixed", "mixed"),
    ("mix", "mixed"),
    ("allocated", "allocated"),
    ("alloc", "allocated"),
    ("allocated+", "allocated"),
    ("drained", "other"),
    ("down*", "other"),
    ("reserved", "other"),
    ("", "other"),
])
def test_norm_node_state(raw, expected):
    assert _norm_node_state(raw) == expected


# --------------------------------------------------------------------------- #
# sinfo parsing
# --------------------------------------------------------------------------- #
def test_parse_sinfo_basic_capacity():
    # cols: node partition state cpus(A/I/O/T) gres gresused
    text = (
        "node01 gpu_a100 mixed 32/96/0/128 gpu:a100:8 gpu:a100:2(IDX:0-1)\n"
        "node02 gpu_a100 idle 0/128/0/128 gpu:a100:8 gpu:a100:0\n"
    )
    parts, totals = parse_sinfo_output(text)
    assert len(parts) == 1
    p = parts[0]
    assert p.name == "gpu_a100"
    assert p.cpus_free == 96 + 128          # idle column summed
    assert p.cpus_alloc == 32
    assert p.cpus_total == 256
    assert p.gpus_total == 16
    assert p.gpus_alloc == 2
    assert p.gpus_free == (8 - 2) + (8 - 0)  # 6 + 8
    assert p.nodes_idle == 1 and p.nodes_mixed == 1 and p.nodes_total == 2
    assert totals == {"cpus_free": 224, "cpus_total": 256,
                      "gpus_free": 14, "gpus_total": 16}


def test_parse_sinfo_excludes_unusable_nodes_from_free_gpus():
    # A drained node may report 0 GPUs used, but its GPUs are not allocatable.
    text = (
        "node01 gpu idle 0/64/0/64 gpu:rtx:4 gpu:rtx:0\n"
        "node02 gpu drained 0/0/64/64 gpu:rtx:4 gpu:rtx:0\n"
    )
    parts, totals = parse_sinfo_output(text)
    p = parts[0]
    assert p.gpus_total == 8           # both nodes' hardware counted in total
    assert p.gpus_free == 4            # only the idle node contributes free GPUs
    assert p.nodes_other == 1          # drained -> "other"
    assert p.cpus_free == 64           # drained node has 0 idle CPUs
    assert totals["gpus_free"] == 4


def test_parse_sinfo_dedupes_shared_nodes_for_cluster_totals():
    # node01 belongs to two partitions -> two lines, but one physical machine.
    text = (
        "node01 short mixed 4/12/0/16 gpu:1 gpu:0\n"
        "node01 long mixed 4/12/0/16 gpu:1 gpu:0\n"
    )
    parts, totals = parse_sinfo_output(text)
    names = {p.name for p in parts}
    assert names == {"short", "long"}
    # Each partition sees the node, but the cluster total counts it once.
    assert totals == {"cpus_free": 12, "cpus_total": 16,
                      "gpus_free": 1, "gpus_total": 1}


def test_parse_sinfo_handles_null_gres_and_blank_lines():
    text = (
        "\n"
        "node01 cpu idle 0/64/0/64 (null) (null)\n"
        "garbage line skipped\n"
    )
    parts, totals = parse_sinfo_output(text)
    assert len(parts) == 1
    assert parts[0].gpus_total == 0 and parts[0].gpus_free == 0
    assert parts[0].cpus_free == 64
    assert totals["gpus_total"] == 0


def test_parse_sinfo_empty():
    parts, totals = parse_sinfo_output("")
    assert parts == []
    assert totals == {"cpus_free": 0, "cpus_total": 0, "gpus_free": 0, "gpus_total": 0}


# --------------------------------------------------------------------------- #
# Folding my jobs into partitions
# --------------------------------------------------------------------------- #
def _job(state, partition):
    return Job(jobid="1", name="n", state=state, partition=partition, elapsed="0",
               time_limit="", nodes=1, cpus=1, gpus=0, reason="", submit_time="t")


def test_fold_my_jobs_counts_running_and_pending():
    parts = [Partition(name="gpu_a100"), Partition(name="gpu")]
    jobs = [_job("RUNNING", "gpu_a100"), _job("PENDING", "gpu_a100"),
            _job("RUNNING", "gpu")]
    _fold_my_jobs(parts, jobs)
    by_name = {p.name: p for p in parts}
    assert by_name["gpu_a100"].my_running == 1 and by_name["gpu_a100"].my_pending == 1
    assert by_name["gpu"].my_running == 1


def test_fold_my_jobs_multi_partition_and_unknown():
    parts = [Partition(name="gpu_a100")]
    # Pending job lists two partitions; "gpu" isn't in sinfo and must be added.
    _fold_my_jobs(parts, [_job("PENDING", "gpu_a100,gpu")])
    by_name = {p.name: p for p in parts}
    assert by_name["gpu_a100"].my_pending == 1
    assert "gpu" in by_name and by_name["gpu"].my_pending == 1
    assert by_name["gpu"].cpus_total == 0  # capacity unknown for the added one


# --------------------------------------------------------------------------- #
# build_overview shape
# --------------------------------------------------------------------------- #
def test_partition_to_dict_shape():
    p = Partition(name="gpu", cpus_free=10, cpus_alloc=6, cpus_total=16,
                  gpus_free=2, gpus_alloc=1, gpus_total=4, nodes_idle=1,
                  nodes_total=1, my_running=1, my_pending=2)
    d = p.to_dict()
    assert d["name"] == "gpu" and d["my_running"] == 1 and d["my_pending"] == 2
    assert d["cpus"] == {"free": 10, "alloc": 6, "total": 16}
    assert d["gpus"]["free"] == 2 and d["gpus"]["alloc"] == 1 and d["gpus"]["total"] == 4
    assert d["gpus"]["by_type"] == {} and d["gpus"]["max_free_per_node"] == 0
    assert d["nodes"]["idle"] == 1 and d["nodes"]["total"] == 1
    assert d["queue"]["wait_estimate"] == "immediate"  # free GPUs > 0


def test_host_to_dict_has_capacity_and_partitions():
    h = Host(name="C", cpus_free=10, cpus_total=20, gpus_free=2, gpus_total=4,
             partitions=[Partition(name="gpu")])
    d = h.to_dict()
    assert d["capacity"] == {"cpus_free": 10, "cpus_total": 20,
                             "gpus_free": 2, "gpus_total": 4}
    assert d["partitions"][0]["name"] == "gpu"


def test_build_overview_shape_and_json():
    snap = Snapshot(generated_at=1.0, hosts=[
        Host(name="Snellius", cpus_free=96, cpus_total=512, gpus_free=6,
             gpus_total=32, jobs=[_job("RUNNING", "gpu")],
             partitions=[Partition(name="gpu", cpus_free=96, cpus_total=512,
                                   gpus_free=6, gpus_total=32, my_running=1)]),
        Host(name="LISA", ok=False, error="timed out"),
    ])
    ov = build_overview(snap)
    json.dumps(ov)  # must be serialisable
    assert ov["generated_at"] == 1.0
    c0 = ov["clusters"][0]
    assert c0["name"] == "Snellius"
    assert c0["my_jobs"] == {"running": 1, "pending": 0}
    assert c0["free"] == {"cpus": 96, "gpus": 6}
    assert c0["capacity"] == {"cpus": 512, "gpus": 32}
    assert c0["partitions"][0]["name"] == "gpu"
    # Unreachable host still appears, flagged.
    assert ov["clusters"][1]["ok"] is False


# --------------------------------------------------------------------------- #
# collect(with_partitions=...) and collect_overview() — subprocess mocked
# --------------------------------------------------------------------------- #
_SQUEUE_LINE = "1|train|RUNNING|gpu_a100|1:00|2:00|1|8|gres/gpu:2|node01|t\n"
_SINFO = (
    "node01 gpu_a100 mixed 8/120/0/128 gpu:a100:8 gpu:a100:2\n"
    "node02 gpu_a100 idle 0/128/0/128 gpu:a100:8 gpu:a100:0\n"
)
_COMBINED = _SQUEUE_LINE + "@@SINFO\n" + _SINFO


def _fake_run_combined(argv, **kwargs):
    return types.SimpleNamespace(returncode=0, stdout=_COMBINED, stderr="")


def test_collect_with_partitions_parses_and_folds(monkeypatch):
    monkeypatch.setattr(collector.subprocess, "run", _fake_run_combined)
    snap = collect({"hosts": [{"name": "Snellius", "ssh": "s"}]},
                   with_partitions=True)
    h = snap.hosts[0]
    assert h.jobs[0].jobid == "1"               # squeue half still parsed
    assert h.cpus_free == 248 and h.gpus_free == 6 + 8
    assert len(h.partitions) == 1
    p = h.partitions[0]
    assert p.name == "gpu_a100" and p.cpus_free == 248
    assert p.my_running == 1                     # folded from squeue --me


def test_collect_without_partitions_leaves_partitions_empty(monkeypatch):
    # Even if sinfo data is present in stdout, without the flag we don't split.
    monkeypatch.setattr(collector.subprocess, "run", _fake_run_combined)
    snap = collect({"hosts": [{"name": "S", "ssh": "s"}]})
    assert snap.hosts[0].partitions == []
    assert snap.hosts[0].cpus_free == 0


def test_collect_overview_end_to_end(monkeypatch):
    monkeypatch.setattr(collector.subprocess, "run", _fake_run_combined)
    ov = collect_overview({"hosts": [{"name": "Snellius", "ssh": "s"}]})
    c = ov["clusters"][0]
    assert c["name"] == "Snellius"
    assert c["free"]["gpus"] == 14
    assert c["partitions"][0]["my_running"] == 1


# --------------------------------------------------------------------------- #
# Command construction + robustness (real /bin/sh, no network)
# --------------------------------------------------------------------------- #
def test_build_squeue_cmd_with_and_without_partitions():
    plain = collector._build_squeue_cmd(False)
    assert "squeue" in plain and "sinfo" not in plain
    full = collector._build_squeue_cmd(True)
    assert "squeue" in full and "sinfo" in full
    assert collector.SINFO_SENTINEL in full and "exit $rc" in full


def test_combined_command_propagates_squeue_exit_status():
    # Run the generated command through a real POSIX shell with squeue/sinfo
    # stubbed: a squeue failure must surface even though sinfo succeeds.
    cmd = (
        "squeue() { return 7; }; sinfo() { echo SINFODATA; }; "
        + collector._build_squeue_cmd(True)
    )
    proc = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 7                 # squeue's status, not sinfo's
    assert "SINFODATA" in proc.stdout            # sinfo still ran
    assert collector.SINFO_SENTINEL in proc.stdout


def test_combined_command_tolerates_failing_sinfo():
    # squeue OK, sinfo missing/failing -> overall success, jobs preserved.
    cmd = (
        "squeue() { echo JOBLINE; return 0; }; sinfo() { return 127; }; "
        + collector._build_squeue_cmd(True)
    )
    proc = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "JOBLINE" in proc.stdout
