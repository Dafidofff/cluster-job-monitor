"""Tests for the tres-per-job GPU fallback.

squeue's ``-o "%b"`` only carries TRES-per-node, so jobs that request GPUs with
``--gpus`` / ``--gpus-per-task`` (TRES-per-job) show 0 GPUs. A second ``--me``
pass over ``JobID,tres-per-job`` fills those in. These tests cover the parser,
the command construction, and the end-to-end fold in ``_collect_host``.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from cluster_job_monitor import collector
from cluster_job_monitor.collector import collect, parse_squeue_tres


# --------------------------------------------------------------------------- #
# parse_squeue_tres
# --------------------------------------------------------------------------- #
def test_parse_squeue_tres_basic():
    text = (
        "203321                     gres/gpu:2\n"
        "203316                     gres/gpu:1\n"
    )
    assert parse_squeue_tres(text) == {"203321": 2, "203316": 1}


def test_parse_squeue_tres_skips_na_and_cpu_only():
    text = (
        "203288                     N/A\n"          # --gres job: GPUs in %b, not here
        "203290                     cpu=4\n"        # cpu-only
        "\n"                                        # blank
        "shortline\n"                               # < 2 tokens
    )
    assert parse_squeue_tres(text) == {}


def test_parse_squeue_tres_typed_gpu():
    assert parse_squeue_tres("1  gres/gpu:a100:3\n") == {"1": 3}


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #
def test_build_squeue_cmd_always_includes_tres_pass():
    plain = collector._build_squeue_cmd(False)
    assert collector.TRES_SENTINEL in plain
    assert "tres-per-job" in plain
    assert "sinfo" not in plain                     # no capacity probe requested
    assert plain.rstrip().endswith("exit $rc")


def test_build_squeue_cmd_section_order():
    full = collector._build_squeue_cmd(True)
    assert (full.index(collector.TRES_SENTINEL)
            < full.index(collector.SINFO_SENTINEL)
            < full.index(collector.QUEUE_SENTINEL)
            < full.index("exit $rc"))


def test_tres_pass_failure_is_tolerated():
    # Primary squeue OK; the -O tres pass failing must not change the exit code.
    cmd = (
        "squeue() { case \"$*\" in *tres-per-job*) return 1;; *) echo JOBLINE; "
        "return 0;; esac; }; " + collector._build_squeue_cmd(False)
    )
    proc = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "JOBLINE" in proc.stdout
    assert collector.TRES_SENTINEL in proc.stdout


# --------------------------------------------------------------------------- #
# End-to-end fold in _collect_host (subprocess mocked)
# --------------------------------------------------------------------------- #
# 203321: --gpus job, empty %b (field 9) -> must be filled from the tres pass.
# 203288: --gres job, %b already has the GPU -> must be left untouched.
# 203290: cpu-only -> stays 0.
_PRIMARY = (
    "203321|train|RUNNING|gpu|1:00|2:00|1|12||None|t\n"
    "203288|eval|RUNNING|gpu|1:00|2:00|1|12|gres/gpu:1|None|t\n"
    "203290|prep|RUNNING|cpu|1:00|2:00|1|4||None|t\n"
)
_TRES = (
    "203321                     gres/gpu:2\n"
    "203288                     N/A\n"
    "203290                     N/A\n"
)
_COMBINED = _PRIMARY + f"{collector.TRES_SENTINEL}\n" + _TRES


def _fake_run(stdout: str):
    def run(argv, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    return run


def test_collect_fills_gpus_from_tres_pass(monkeypatch):
    monkeypatch.setattr(collector.subprocess, "run", _fake_run(_COMBINED))
    snap = collect({"hosts": [{"name": "IVI-H1", "ssh": "ivi-h1"}]})
    by_id = {j.jobid: j for j in snap.hosts[0].jobs}
    assert by_id["203321"].gpus == 2      # filled from tres-per-job
    assert by_id["203288"].gpus == 1      # %b value preserved, not overwritten
    assert by_id["203290"].gpus == 0      # cpu-only stays 0
    # Host GPU-in-use rollup now reflects the previously-missed job.
    assert snap.hosts[0].gpus_in_use == 3


def test_collect_overview_also_folds_tres(monkeypatch):
    combined = (
        _PRIMARY
        + f"{collector.TRES_SENTINEL}\n" + _TRES
        + f"{collector.SINFO_SENTINEL}\n"
        + "node01 gpu idle 0/64/0/64 gpu:2 gpu:0\n"
        + f"{collector.QUEUE_SENTINEL}\n"
        + "gpu|RUNNING|1:00:00\n"
    )
    monkeypatch.setattr(collector.subprocess, "run", _fake_run(combined))
    snap = collect({"hosts": [{"name": "IVI-H1", "ssh": "ivi-h1"}]},
                   with_partitions=True)
    h = snap.hosts[0]
    by_id = {j.jobid: j for j in h.jobs}
    assert by_id["203321"].gpus == 2                 # fallback still applied
    assert "gpu" in {p.name for p in h.partitions}   # sinfo still parsed
    assert h.cpus_free == 64                          # totals unaffected
