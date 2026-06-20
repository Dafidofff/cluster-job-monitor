"""Synthetic snapshot for demos / UI testing without touching real clusters."""

from __future__ import annotations

import time

from core.collector import Host, Job, Partition, Snapshot


def _job(jobid, name, state, part, elapsed, limit, nodes, cpus, gpus,
         reason="", progress=None):
    return Job(
        jobid=jobid, name=name, state=state, partition=part, elapsed=elapsed,
        time_limit=limit, nodes=nodes, cpus=cpus, gpus=gpus, reason=reason,
        submit_time="2026-06-17T09:00:00", progress=progress,
    )


def make_demo_snapshot(_config=None) -> Snapshot:
    snellius = Host(
        name="Snellius",
        cpus_free=224, cpus_total=768, gpus_free=14, gpus_total=48,
        partitions=[
            Partition(name="gpu_a100", cpus_free=96, cpus_alloc=416, cpus_total=512,
                      gpus_free=4, gpus_alloc=28, gpus_total=32,
                      nodes_idle=0, nodes_mixed=6, nodes_alloc=2, nodes_total=8,
                      my_running=1, my_pending=2),
            Partition(name="gpu", cpus_free=128, cpus_alloc=128, cpus_total=256,
                      gpus_free=10, gpus_alloc=6, gpus_total=16,
                      nodes_idle=2, nodes_mixed=2, nodes_total=4,
                      my_running=1, my_pending=0),
        ],
        jobs=[
            _job("8123456", "diffusion-pretrain", "RUNNING", "gpu_a100",
                 "1-04:12:00", "5-00:00:00", 2, 36, 8, progress=0.23),
            _job("8123457", "ablation-lr3e4", "RUNNING", "gpu",
                 "18:42:10", "1-00:00:00", 1, 18, 4, progress=0.78),
            _job("8123460", "sweep-seed-7", "PENDING", "gpu_a100",
                 "0:00", "1-00:00:00", 1, 18, 4, reason="Priority"),
            _job("8123461", "sweep-seed-8", "PENDING", "gpu_a100",
                 "0:00", "1-00:00:00", 1, 18, 4, reason="Resources"),
        ],
    )
    das6 = Host(
        name="DAS6",
        cpus_free=200, cpus_total=256, gpus_free=12, gpus_total=16,
        partitions=[
            Partition(name="defq", cpus_free=200, cpus_alloc=56, cpus_total=256,
                      gpus_free=12, gpus_alloc=4, gpus_total=16,
                      nodes_idle=3, nodes_mixed=1, nodes_other=1, nodes_total=5,
                      my_running=1, my_pending=0),
        ],
        jobs=[
            _job("44219", "eval-medmnist", "RUNNING", "defq",
                 "00:58:30", "01:00:00", 1, 8, 1, progress=0.97),
            _job("44220", "preprocess", "COMPLETING", "defq",
                 "02:03:00", "04:00:00", 1, 4, 0),
        ],
    )
    desktop = Host(name="Desktop", jobs=[
        _job("312", "local-notebook", "RUNNING", "local",
             "03:21:00", "", 1, 2, 1, progress=None),
    ])
    larry = Host(name="Larry (desktop GPU)", kind="gpu",
                 note="RTX 3090 · 73% util · 17.8/24 GB",
                 gpus_total=1, gpus_free=0, jobs=[
        Job(jobid="40123", name="diffusion-v2", state="RUNNING", partition="GPU 0",
            elapsed="2-03:14:05", time_limit="", nodes=1, cpus=0, gpus=1,
            reason="", submit_time="", gpu_mem_mb=12000, user="david"),
        Job(jobid="40555", name="eval.py", state="RUNNING", partition="GPU 0",
            elapsed="11:02", time_limit="", nodes=1, cpus=0, gpus=1,
            reason="", submit_time="", gpu_mem_mb=6200, user="david"),
    ])
    unreachable = Host(name="LISA", ok=False,
                       error="ssh: connect to host lisa port 22: Operation timed out")
    return Snapshot(
        generated_at=time.time(),
        hosts=[snellius, das6, desktop, larry, unreachable],
    )
