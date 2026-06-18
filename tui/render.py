"""Rich renderables shared by the Textual app and the ``--once`` text mode.

Keeping the rendering here (pure: snapshot + filters -> Rich renderable) means
the live TUI and the one-shot print produce identical-looking output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from rich.box import ROUNDED
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.collector import Host, Job, Snapshot

# Bucket -> (label colour, chip style)
_STATE_STYLE = {
    "running": "bold green",
    "pending": "bold yellow",
    "other": "bold cyan",
}
_DOT = "●"  # ●


@dataclass
class Filters:
    """Active view filters. None / empty means 'show all'."""

    state: Optional[str] = None      # 'running' | 'pending' | 'other'
    host: Optional[str] = None       # host name
    partition: Optional[str] = None  # partition name
    search: str = ""                 # case-insensitive substring of job name

    def label(self) -> str:
        bits = []
        if self.state:
            bits.append(f"state={self.state}")
        if self.host:
            bits.append(f"cluster={self.host}")
        if self.partition:
            bits.append(f"part={self.partition}")
        if self.search:
            bits.append(f"search='{self.search}'")
        return "  ".join(bits) if bits else "none"

    def match(self, job: Job) -> bool:
        if self.state and job.bucket != self.state:
            return False
        if self.partition and job.partition != self.partition:
            return False
        if self.search and self.search.lower() not in job.name.lower():
            return False
        return True


def _progress_bar(frac: Optional[float], width: int = 12) -> Text:
    if frac is None:
        return Text("  no limit", style="dim")
    filled = int(round(frac * width))
    filled = max(0, min(width, filled))
    style = "green" if frac < 0.75 else "yellow" if frac < 0.95 else "red"
    bar = Text()
    bar.append("█" * filled, style=style)
    bar.append("░" * (width - filled), style="grey37")
    bar.append(f" {int(frac * 100):>3d}%", style=style)
    return bar


def _fmt_mem(mb: Optional[int]) -> str:
    if not mb:
        return ""
    return f"{mb / 1024:.1f}G" if mb >= 1024 else f"{mb}M"


def _resources(job: Job) -> Text:
    t = Text()
    # GPU-process view (non-SLURM hosts): show GPU count + memory used.
    if job.gpu_mem_mb is not None:
        t.append(f"{job.gpus}", style="bold magenta")
        t.append("gpu ", style="magenta")
        t.append(_fmt_mem(job.gpu_mem_mb), style="grey70")
        return t
    t.append(f"{job.nodes}n ", style="grey70")
    t.append(f"{job.cpus}c", style="grey70")
    if job.gpus:
        t.append(f" {job.gpus}", style="bold magenta")
        t.append("gpu", style="magenta")
    return t


def _jobs_table(jobs: list[Job]) -> Table:
    table = Table(
        box=None, expand=True, pad_edge=False, show_edge=False,
        header_style="dim", padding=(0, 1),
    )
    table.add_column("", width=2)                       # state dot
    table.add_column("JOB ID", style="grey50", no_wrap=True)
    table.add_column("NAME", ratio=2, no_wrap=True, overflow="ellipsis")
    table.add_column("PART", style="grey70", no_wrap=True)
    table.add_column("RES", no_wrap=True)
    table.add_column("ELAPSED / LIMIT", no_wrap=True)
    table.add_column("INFO", ratio=1, no_wrap=True, overflow="ellipsis")

    for job in jobs:
        style = _STATE_STYLE.get(job.bucket, "white")
        dot = Text(_DOT, style=style)
        if job.bucket == "running":
            if job.progress is not None:
                time_cell: RenderableType = _progress_bar(job.progress)
            elif job.elapsed:
                time_cell = Text(job.elapsed, style="dim")
            else:
                time_cell = Text("running", style="dim")
            info = Text(job.user or job.state.title(), style=style)
        elif job.bucket == "pending":
            time_cell = Text(f"limit {job.time_limit}", style="dim")
            info = Text(job.reason or "Pending", style="yellow")
        else:
            time_cell = Text(job.elapsed or "-", style="dim")
            info = Text(job.state.title(), style=style)
        table.add_row(
            dot, job.jobid, job.name, job.partition,
            _resources(job), time_cell, info,
        )
    return table


def _summary_text(host: Host) -> Text:
    """Compact per-host counts, shared by the expanded subtitle and the
    collapsed one-liner so the two views never drift."""
    t = Text()
    t.append(f"{host.running} run", style="green")
    if host.kind != "gpu":
        t.append("  ")
        t.append(f"{host.pending} pend", style="yellow")
    if host.other:
        t.append("  ")
        t.append(f"{host.other} other", style="cyan")
    if host.cpus_in_use:
        t.append("   ")
        t.append(f"{host.cpus_in_use} cpu", style="grey70")
    if host.gpus_in_use and host.kind != "gpu":
        t.append("  ")
        t.append(f"{host.gpus_in_use} gpu", style="magenta")
    if host.note:
        t.append("   ")
        t.append(host.note, style="grey62")
    return t


def _host_collapsed(host: Host, index: int) -> Text:
    """A minimised cluster: one line, no job table."""
    t = Text()
    t.append(f"{index} ▸ ", style="grey50")
    t.append(f"{_DOT} ", style="green" if host.ok else "red")
    t.append(host.name, style="bold")
    if not host.ok:
        t.append(f"   unreachable — {host.error}", style="red")
    else:
        t.append("   ")
        t.append_text(_summary_text(host))
    return t


def _host_panel(host: Host, jobs: list[Job], index: int) -> Panel:
    title = Text()
    title.append(f"{index} ▾ ", style="grey50")
    title.append(f"{_DOT} ", style="green" if host.ok else "red")
    title.append(host.name, style="bold")

    subtitle = _summary_text(host)

    if not host.ok:
        body: RenderableType = Text(f"unreachable — {host.error}", style="red")
    elif not jobs:
        if host.kind == "gpu":
            body = Text("no jobs match" if host.jobs else "GPU idle — no compute processes",
                        style="dim italic")
        else:
            body = Text("no jobs match" if host.jobs else "no jobs queued",
                        style="dim italic")
    else:
        body = _jobs_table(jobs)

    border = "green" if host.ok else "red"
    return Panel(
        body, title=title, subtitle=subtitle, subtitle_align="left",
        title_align="left", box=ROUNDED, border_style=border, padding=(0, 1),
    )


def render_header(snapshot: Optional[Snapshot], filters: Filters,
                  refreshing: bool) -> Text:
    t = Text()
    t.append("  CLUSTER JOBS  ", style="bold reverse")
    t.append("   ")
    if snapshot is None:
        t.append("loading…", style="dim")
        return t

    t.append(f"{snapshot.total_running} running", style="bold green")
    t.append("   ")
    t.append(f"{snapshot.total_pending} pending", style="bold yellow")
    t.append("      ")

    age = time.time() - snapshot.generated_at
    age_style = "green" if age < 60 else "yellow" if age < 180 else "red"
    when = "just now" if age < 2 else f"{int(age)}s ago"
    t.append("updated ", style="dim")
    t.append(when, style=age_style)
    if refreshing:
        t.append("  ↻ refreshing", style="cyan")

    t.append("      filter: ", style="dim")
    t.append(filters.label(), style="grey70")
    return t


def render_body(snapshot: Optional[Snapshot], filters: Filters,
                minimized: Optional[set[str]] = None) -> RenderableType:
    if snapshot is None:
        return Text("\n  Connecting to clusters…", style="dim")
    if not snapshot.hosts:
        return Text("\n  No hosts configured.", style="dim")

    minimized = minimized or set()
    items: list[RenderableType] = []
    # Enumerate before filtering so the index next to each cluster is stable.
    for index, host in enumerate(snapshot.hosts, 1):
        if filters.host and host.name != filters.host:
            continue
        if host.name in minimized:
            items.append(_host_collapsed(host, index))
        else:
            jobs = [j for j in host.jobs if filters.match(j)]
            items.append(_host_panel(host, jobs, index))
    if not items:
        return Text("\n  No clusters match the current filter.", style="dim")
    return Group(*items)


def all_partitions(snapshot: Optional[Snapshot]) -> list[str]:
    if snapshot is None:
        return []
    parts = {j.partition for h in snapshot.hosts for j in h.jobs if j.partition}
    return sorted(parts)


def all_hosts(snapshot: Optional[Snapshot]) -> list[str]:
    if snapshot is None:
        return []
    return [h.name for h in snapshot.hosts]
