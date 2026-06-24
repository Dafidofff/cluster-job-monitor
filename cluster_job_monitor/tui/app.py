"""Textual terminal dashboard for cross-cluster SLURM jobs (read-only)."""

from __future__ import annotations

import asyncio
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

from cluster_job_monitor.collector import Snapshot, collect
from .render import (
    Filters, all_hosts, all_partitions, render_body, render_header,
)

# Order used when cycling the state filter with `f`.
_STATE_CYCLE = [None, "running", "pending", "other"]


class JobMonitorApp(App):
    CSS = """
    Screen { background: $background; }
    #header { height: 1; padding: 0 1; }
    #body { padding: 0 1; }
    #search { dock: bottom; display: none; border: tall $accent; }
    #search.visible { display: block; }
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("f", "cycle_state", "State"),
        Binding("c", "cycle_host", "Cluster"),
        Binding("p", "cycle_partition", "Partition"),
        Binding("slash", "search", "Search"),
        Binding("m", "toggle_all_min", "Min all"),
        Binding("escape", "clear", "Clear filters"),
        Binding("q", "quit", "Quit"),
        *[Binding(str(n), f"toggle_min('{n}')", f"Min {n}", show=False)
          for n in range(1, 10)],
    ]

    def __init__(self, config: dict, collect_fn=collect) -> None:
        super().__init__()
        self.config = config
        self.collect_fn = collect_fn
        self.snapshot: Optional[Snapshot] = None
        self.filters = Filters()
        self._refreshing = False
        # Clusters collapsed to a one-line summary. Seeded from config
        # ("minimized": true), then toggled live with number keys / `m`.
        self.minimized: set[str] = {
            h["name"] for h in config.get("hosts", [])
            if h.get("minimized") and h.get("name")
        }

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        with VerticalScroll():
            yield Static(id="body")
        yield Input(placeholder="filter job name… (Enter to apply, Esc to close)",
                    id="search")

    def on_mount(self) -> None:
        self._render()
        interval = float(self.config.get("refresh_seconds", 30))
        self.set_interval(interval, self.refresh_data)
        self.set_interval(1.0, self._tick_header)  # keep "updated Xs ago" live
        self.call_after_refresh(self.refresh_data)

    # ---- data ---------------------------------------------------------- #
    async def refresh_data(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        self._render_header()
        try:
            self.snapshot = await asyncio.to_thread(self.collect_fn, self.config)
        except Exception as exc:  # never let a refresh crash the UI
            self.notify(f"collection failed: {exc}", severity="error")
        finally:
            self._refreshing = False
        self._render()

    # ---- rendering ----------------------------------------------------- #
    def _render(self) -> None:
        self._render_header()
        self.query_one("#body", Static).update(
            render_body(self.snapshot, self.filters, self.minimized)
        )

    def _render_header(self) -> None:
        self.query_one("#header", Static).update(
            render_header(self.snapshot, self.filters, self._refreshing)
        )

    def _tick_header(self) -> None:
        if not self._refreshing:
            self._render_header()

    # ---- actions ------------------------------------------------------- #
    def action_refresh(self) -> None:
        self.run_worker(self.refresh_data(), exclusive=False)

    def action_cycle_state(self) -> None:
        i = _STATE_CYCLE.index(self.filters.state) if self.filters.state in _STATE_CYCLE else 0
        self.filters.state = _STATE_CYCLE[(i + 1) % len(_STATE_CYCLE)]
        self._render()

    def action_cycle_host(self) -> None:
        self.filters.host = _cycle(all_hosts(self.snapshot), self.filters.host)
        self._render()

    def action_cycle_partition(self) -> None:
        self.filters.partition = _cycle(
            all_partitions(self.snapshot), self.filters.partition
        )
        self._render()

    def action_toggle_min(self, num: str) -> None:
        hosts = self.snapshot.hosts if self.snapshot else []
        idx = int(num) - 1
        if 0 <= idx < len(hosts):
            name = hosts[idx].name
            self.minimized.symmetric_difference_update({name})
            self._render()

    def action_toggle_all_min(self) -> None:
        if not self.snapshot:
            return
        names = {h.name for h in self.snapshot.hosts}
        # If anything is currently expanded, collapse everything; else expand all.
        if names - self.minimized:
            self.minimized |= names
        else:
            self.minimized.clear()
        self._render()

    def action_search(self) -> None:
        box = self.query_one("#search", Input)
        box.add_class("visible")
        box.value = self.filters.search
        box.focus()

    def action_clear(self) -> None:
        self.filters = Filters()
        box = self.query_one("#search", Input)
        box.remove_class("visible")
        self._render()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.filters.search = event.value.strip()
        box = self.query_one("#search", Input)
        box.remove_class("visible")
        self.set_focus(None)
        self._render()


def _cycle(values: list[str], current: Optional[str]) -> Optional[str]:
    """Cycle None -> values[0] -> ... -> values[-1] -> None."""
    options: list[Optional[str]] = [None, *values]
    try:
        i = options.index(current)
    except ValueError:
        i = 0
    return options[(i + 1) % len(options)]
