"""Tests for the Rich rendering layer (pure functions)."""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from cluster_job_monitor.collector import Host, Job, Snapshot
from cluster_job_monitor.tui import render
from cluster_job_monitor.tui.render import (
    Filters, all_hosts, all_partitions, render_body, render_header,
)


def _plain(renderable) -> str:
    console = Console(width=200)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def _job(name="job", state="RUNNING", partition="gpu", gpus=1, gpu_mem_mb=None,
         elapsed="1:00", progress=0.5):
    return Job(jobid="42", name=name, state=state, partition=partition,
               elapsed=elapsed, time_limit="2:00", nodes=1, cpus=8, gpus=gpus,
               reason="Priority", submit_time="t", progress=progress,
               gpu_mem_mb=gpu_mem_mb)


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def test_filters_match():
    f = Filters(state="running")
    assert f.match(_job(state="RUNNING"))
    assert not f.match(_job(state="PENDING"))
    assert Filters(partition="gpu").match(_job(partition="gpu"))
    assert not Filters(partition="cpu").match(_job(partition="gpu"))
    assert Filters(search="abl").match(_job(name="ablation"))
    assert not Filters(search="zzz").match(_job(name="ablation"))


def test_filters_label():
    assert Filters().label() == "none"
    lbl = Filters(state="running", host="H", partition="gpu", search="x").label()
    assert "state=running" in lbl and "cluster=H" in lbl and "search='x'" in lbl


# --------------------------------------------------------------------------- #
# Small renderable helpers
# --------------------------------------------------------------------------- #
def test_fmt_mem():
    assert render._fmt_mem(None) == "" and render._fmt_mem(0) == ""
    assert render._fmt_mem(512) == "512M"
    assert render._fmt_mem(12000).endswith("G")


def test_progress_bar():
    assert "no limit" in _plain(render._progress_bar(None))
    assert "50%" in _plain(render._progress_bar(0.5))


def test_resources_gpu_vs_slurm():
    assert "8c" in _plain(render._resources(_job()))               # slurm view
    gpu = _plain(render._resources(_job(gpu_mem_mb=12000, gpus=1)))  # gpu view
    assert "gpu" in gpu and "G" in gpu


def test_summary_text_counts():
    host = Host(name="H", jobs=[_job(state="RUNNING"), _job(state="PENDING")])
    out = _plain(render._summary_text(host))
    assert "1 run" in out and "1 pend" in out


# --------------------------------------------------------------------------- #
# Panels / collapsed lines
# --------------------------------------------------------------------------- #
def test_host_collapsed_ok_and_error():
    ok = render._host_collapsed(Host(name="Snellius", jobs=[_job()]), 2)
    txt = _plain(ok)
    assert "2 ▸" in txt and "Snellius" in txt and "run" in txt

    bad = render._host_collapsed(Host(name="Lisa", ok=False, error="timeout"), 5)
    assert "unreachable" in _plain(bad)


def test_host_panel_is_panel_with_index():
    panel = render._host_panel(Host(name="H", jobs=[_job()]), [_job()], 1)
    assert isinstance(panel, Panel)
    assert "1 ▾" in _plain(panel) and "H" in _plain(panel)


def test_host_panel_idle_messages():
    gpu_idle = render._host_panel(Host(name="G", kind="gpu"), [], 1)
    assert "GPU idle" in _plain(gpu_idle)
    slurm_empty = render._host_panel(Host(name="S"), [], 1)
    assert "no jobs queued" in _plain(slurm_empty)


# --------------------------------------------------------------------------- #
# render_body / render_header
# --------------------------------------------------------------------------- #
def _snapshot():
    return Snapshot(generated_at=0.0, hosts=[
        Host(name="A", jobs=[_job()]),
        Host(name="B", jobs=[_job(state="PENDING")]),
    ])


def test_render_body_states():
    assert "Connecting" in _plain(render_body(None, Filters()))
    empty = Snapshot(generated_at=0.0, hosts=[])
    assert "No hosts configured" in _plain(render_body(empty, Filters()))


def test_render_body_minimized_uses_collapsed_line():
    snap = _snapshot()
    group = render_body(snap, Filters(), minimized={"A"})
    assert isinstance(group, Group)
    assert isinstance(group.renderables[0], Text)    # A collapsed
    assert isinstance(group.renderables[1], Panel)   # B expanded


def test_render_body_host_filter_keeps_indices():
    snap = _snapshot()
    # Filtering to B should still show its number (2), proving stable indexing,
    # and host A's panel title (● A) should be absent.
    out = _plain(render_body(snap, Filters(host="B")))
    assert "2 ▾" in out and "● A" not in out


def test_render_body_no_match_message():
    snap = _snapshot()
    assert "No clusters match" in _plain(render_body(snap, Filters(host="zzz")))


def test_render_header():
    assert "loading" in _plain(render_header(None, Filters(), False))
    out = _plain(render_header(_snapshot(), Filters(), refreshing=True))
    assert "1 running" in out and "1 pending" in out and "refreshing" in out


def test_all_partitions_and_hosts():
    snap = _snapshot()
    assert all_hosts(snap) == ["A", "B"]
    assert all_partitions(snap) == ["gpu"]
    assert all_hosts(None) == [] and all_partitions(None) == []
