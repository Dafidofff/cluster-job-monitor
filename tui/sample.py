"""Synthetic snapshot for demos / UI testing without touching real clusters."""

from __future__ import annotations

import time

from core.collector import Host, Job, Snapshot


def _job(jobid, name, state, part, elapsed, limit, nodes, cpus, gpus,
         reason="", progress=None):
    return Job(
        jobid=jobid, name=name, state=state, partition=part, elapsed=elapsed,
        time_limit=limit, nodes=nodes, cpus=cpus, gpus=gpus, reason=reason,
        submit_time="2026-06-17T09:00:00", progress=progress,
    )


def make_demo_snapshot(_config=None) -> Snapshot:
    snellius = Host(name="Snellius", jobs=[
        _job("8123456", "diffusion-pretrain", "RUNNING", "gpu_a100",
             "1-04:12:00", "5-00:00:00", 2, 36, 8, progress=0.23),
        _job("8123457", "ablation-lr3e4", "RUNNING", "gpu",
             "18:42:10", "1-00:00:00", 1, 18, 4, progress=0.78),
        _job("8123460", "sweep-seed-7", "PENDING", "gpu_a100",
             "0:00", "1-00:00:00", 1, 18, 4, reason="Priority"),
        _job("8123461", "sweep-seed-8", "PENDING", "gpu_a100",
             "0:00", "1-00:00:00", 1, 18, 4, reason="Resources"),
    ])
    das6 = Host(name="DAS6", jobs=[
        _job("44219", "eval-medmnist", "RUNNING", "defq",
             "00:58:30", "01:00:00", 1, 8, 1, progress=0.97),
        _job("44220", "preprocess", "COMPLETING", "defq",
             "02:03:00", "04:00:00", 1, 4, 0),
    ])
    desktop = Host(name="Desktop", jobs=[
        _job("312", "local-notebook", "RUNNING", "local",
             "03:21:00", "", 1, 2, 1, progress=None),
    ])
    unreachable = Host(name="LISA", ok=False,
                       error="ssh: connect to host lisa port 22: Operation timed out")
    return Snapshot(
        generated_at=time.time(),
        hosts=[snellius, das6, desktop, unreachable],
    )
