"""Tests for the UI-agnostic collector core."""

from __future__ import annotations

import json
import subprocess
import types

import pytest

from core import collector
from core.collector import (
    Host, Job, Snapshot, collect, gpu_note, gpu_run_label, load_config,
    parse_gpus, parse_gpu_output, parse_job_line, parse_slurm_time,
    parse_squeue_output,
)


# --------------------------------------------------------------------------- #
# Time / gpu / int parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [
    ("1-02:03:04", 1 * 86400 + 2 * 3600 + 3 * 60 + 4),
    ("18:00:00", 18 * 3600),
    ("12:34", 12 * 60 + 34),
    ("45", 45),
    ("UNLIMITED", None),
    ("", None),
    ("N/A", None),
    ("garbage", None),
    ("3-bad", None),
])
def test_parse_slurm_time(value, expected):
    assert parse_slurm_time(value) == expected


@pytest.mark.parametrize("tres,expected", [
    ("gres/gpu:a100:2", 2),
    ("gpu:4", 4),
    ("gres/gpu:2,gres/gpu:1", 3),
    ("cpu=8,mem=16G", 0),
    ("", 0),
])
def test_parse_gpus(tres, expected):
    assert parse_gpus(tres) == expected


def test_to_int_fallback():
    assert collector._to_int("7") == 7
    assert collector._to_int("x") == 0
    assert collector._to_int(None) == 0


# --------------------------------------------------------------------------- #
# squeue line parsing
# --------------------------------------------------------------------------- #
def test_parse_job_line_running_progress():
    line = "8123457|ablation|RUNNING|gpu|18:00:00|1-00:00:00|1|18|gres/gpu:4|None|2026-06-17T09:00:00"
    job = parse_job_line(line)
    assert job.jobid == "8123457"
    assert job.state == "RUNNING" and job.bucket == "running"
    assert job.nodes == 1 and job.cpus == 18 and job.gpus == 4
    assert job.progress == pytest.approx(0.75)


def test_parse_job_line_pending_no_progress():
    line = "9|sweep|pending|gpu|0:00|UNLIMITED|1|4|gres/gpu:1|Priority|t"
    job = parse_job_line(line)
    assert job.bucket == "pending"
    assert job.state == "PENDING"        # upper-cased
    assert job.progress is None          # unbounded limit


def test_parse_job_line_malformed_returns_none():
    assert parse_job_line("too|few|fields") is None


def test_parse_squeue_output_skips_blanks_and_garbage():
    text = (
        "1|a|RUNNING|p|1:00|2:00|1|1|gres/gpu:1|None|t\n"
        "\n"
        "this is a banner line with no pipes\n"
        "2|b|PENDING|p|0:00|2:00|1|1||Priority|t\n"
    )
    jobs = parse_squeue_output(text)
    assert [j.jobid for j in jobs] == ["1", "2"]


# --------------------------------------------------------------------------- #
# GPU (non-SLURM) parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cmd,expected", [
    ("/home/d/env/bin/python train.py --config c.yaml --name big-run-7", "big-run-7"),
    ("python3 train.py --run_name=ablation_lr3", "ablation_lr3"),
    ("/opt/conda/bin/python /work/scripts/finetune.py --lr 1e-4", "finetune.py"),
    ("python -m torch.distributed.run main.py", "main.py"),
    ("/usr/bin/python3.10 sample.py", "sample.py"),
    ("", "(unknown)"),
    ("--name", "--name"),  # flag with no following value -> falls through
])
def test_gpu_run_label(cmd, expected):
    assert gpu_run_label(cmd) == expected


def test_parse_gpu_output_aggregates_by_pid():
    sample = (
        "@@GPUS\n"
        "GPU-abc,0,NVIDIA GeForce RTX 3090,73,18234,24576\n"
        "GPU-def,1,NVIDIA GeForce RTX 3090,10,500,24576\n"
        "@@PROCS\n"
        "40123|GPU-abc|12000|2-03:14:05|david|python train.py --name diff-v2\n"
        "40123|GPU-def|3000|2-03:14:05|david|python train.py --name diff-v2\n"
        "40555|GPU-abc|6200|11:02|david|python eval.py\n"
    )
    gpus, jobs = parse_gpu_output(sample)
    assert len(gpus) == 2
    by_pid = {j.jobid: j for j in jobs}
    # pid 40123 spans both GPUs -> mem summed, 2 gpus, partition lists both
    assert by_pid["40123"].gpu_mem_mb == 15000
    assert by_pid["40123"].gpus == 2
    assert by_pid["40123"].partition == "GPU 0,1"
    assert by_pid["40123"].name == "diff-v2"
    assert by_pid["40123"].user == "david"
    assert by_pid["40555"].name == "eval.py" and by_pid["40555"].gpu_mem_mb == 6200


def test_parse_gpu_output_idle():
    gpus, jobs = parse_gpu_output("@@GPUS\nGPU-x,0,RTX,0,309,24576\n@@PROCS\n")
    assert len(gpus) == 1 and jobs == []


def test_gpu_note():
    assert gpu_note([]) == ""
    one = gpu_note([{"name": "NVIDIA GeForce RTX 3090", "util": "73",
                     "mem_used": "18234", "mem_total": "24576"}])
    assert "RTX 3090" in one and "73% util" in one
    multi = gpu_note([
        {"name": "A", "util": "10", "mem_used": "1024", "mem_total": "2048"},
        {"name": "B", "util": "30", "mem_used": "1024", "mem_total": "2048"},
    ])
    assert multi.startswith("2× GPU")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
def _job(state, cpus=4, gpus=1):
    return Job(jobid="1", name="n", state=state, partition="p", elapsed="1:00",
               time_limit="2:00", nodes=1, cpus=cpus, gpus=gpus, reason="",
               submit_time="t")


def test_host_summaries_and_to_dict():
    host = Host(name="H", jobs=[_job("RUNNING", cpus=8, gpus=2),
                                _job("PENDING"), _job("COMPLETING")])
    assert host.running == 1 and host.pending == 1 and host.other == 1
    assert host.cpus_in_use == 8 and host.gpus_in_use == 2  # running only
    d = host.to_dict()
    assert d["kind"] == "slurm" and d["summary"]["running"] == 1
    assert "note" in d and isinstance(d["jobs"], list)


def test_snapshot_totals_json_serializable():
    snap = Snapshot(generated_at=1.0, hosts=[
        Host(name="A", jobs=[_job("RUNNING")]),
        Host(name="B", jobs=[_job("PENDING"), _job("PENDING")]),
    ])
    assert snap.total_running == 1 and snap.total_pending == 2
    json.dumps(snap.to_dict())  # must not raise


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def test_load_config_ok(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"hosts": [{"name": "X", "ssh": "x"}]}))
    cfg = load_config(p)
    assert cfg["hosts"][0]["name"] == "X"
    assert cfg["refresh_seconds"] == 30  # default applied


def test_load_config_rejects_missing_hosts(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"refresh_seconds": 5}))
    with pytest.raises(ValueError):
        load_config(p)


# --------------------------------------------------------------------------- #
# collect() — subprocess mocked for determinism
# --------------------------------------------------------------------------- #
_SQUEUE = "1|train|RUNNING|gpu|1:00|2:00|1|8|gres/gpu:2|None|t\n"
_GPU = ("@@GPUS\nGPU-a,0,RTX,50,1000,24576\n@@PROCS\n"
        "9|GPU-a|2000|10:00|me|python x.py --name r1\n")


def _fake_run(returns):
    def run(argv, **kwargs):
        joined = " ".join(argv)
        out = _GPU if "nvidia-smi" in joined else _SQUEUE
        return types.SimpleNamespace(returncode=returns, stdout=out, stderr="boom")
    return run


def test_collect_parses_slurm_and_gpu(monkeypatch):
    monkeypatch.setattr(collector.subprocess, "run", _fake_run(0))
    cfg = {"hosts": [
        {"name": "Cluster", "ssh": "c"},
        {"name": "Box", "ssh": "b", "scheduler": "gpu"},
    ]}
    snap = collect(cfg)
    assert [h.name for h in snap.hosts] == ["Cluster", "Box"]  # order preserved
    assert snap.hosts[0].kind == "slurm" and snap.hosts[0].jobs[0].gpus == 2
    assert snap.hosts[1].kind == "gpu" and snap.hosts[1].jobs[0].name == "r1"
    assert snap.hosts[1].note.startswith("RTX")


def test_collect_nonzero_exit_marks_host_unreachable(monkeypatch):
    monkeypatch.setattr(collector.subprocess, "run", _fake_run(1))
    snap = collect({"hosts": [{"name": "C", "ssh": "c"}]})
    assert snap.hosts[0].ok is False and snap.hosts[0].error == "boom"


def test_collect_missing_alias():
    snap = collect({"hosts": [{"name": "C", "local": False}]})
    assert snap.hosts[0].ok is False and "no 'ssh' alias" in snap.hosts[0].error


def test_collect_timeout(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=25)
    monkeypatch.setattr(collector.subprocess, "run", boom)
    snap = collect({"hosts": [{"name": "C", "ssh": "c"}]})
    assert snap.hosts[0].ok is False and "timed out" in snap.hosts[0].error


def test_collect_empty_hosts():
    snap = collect({"hosts": []})
    assert snap.hosts == []
