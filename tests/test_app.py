"""Tests for the Textual app: filters and the collapsible-cluster feature."""

from __future__ import annotations

from cluster_job_monitor.tui.app import JobMonitorApp, _cycle
from cluster_job_monitor.tui.sample import make_demo_snapshot


def test_cycle_helper():
    assert _cycle(["a", "b"], None) == "a"
    assert _cycle(["a", "b"], "a") == "b"
    assert _cycle(["a", "b"], "b") is None      # wraps back to "show all"


def test_minimized_seeded_from_config():
    cfg = {"hosts": [
        {"name": "Keep", "ssh": "k"},
        {"name": "Hide", "ssh": "h", "minimized": True},
    ]}
    app = JobMonitorApp(cfg)
    assert app.minimized == {"Hide"}


def _demo_app():
    return JobMonitorApp({"refresh_seconds": 5, "hosts": []},
                         collect_fn=make_demo_snapshot)


async def test_filters_cycle_and_search():
    app = _demo_app()
    async with app.run_test(size=(135, 45)) as pilot:
        await app.refresh_data()
        await pilot.pause()
        assert app.snapshot is not None and len(app.snapshot.hosts) == 5

        await pilot.press("f")            # state filter -> running
        assert app.filters.state == "running"
        await pilot.press("c")            # cluster filter -> first host
        assert app.filters.host == app.snapshot.hosts[0].name
        await pilot.press("p")            # partition filter -> some partition
        assert app.filters.partition is not None

        await pilot.press("slash")        # open search
        from textual.widgets import Input
        box = app.query_one("#search", Input)
        box.value = "diffusion"
        await box.action_submit()
        await pilot.pause()
        assert app.filters.search == "diffusion"

        await pilot.press("escape")       # clear all filters
        assert app.filters.label() == "none"


async def test_minimise_toggles():
    app = _demo_app()
    async with app.run_test(size=(135, 45)) as pilot:
        await app.refresh_data()
        await pilot.pause()
        names = [h.name for h in app.snapshot.hosts]

        await pilot.press("2")            # collapse host #2
        assert app.minimized == {names[1]}
        await pilot.press("2")            # expand host #2
        assert app.minimized == set()

        await pilot.press("m")            # collapse all
        assert app.minimized == set(names)
        await pilot.press("m")            # expand all
        assert app.minimized == set()


async def test_minimise_out_of_range_is_noop():
    app = _demo_app()
    async with app.run_test(size=(135, 45)) as pilot:
        await app.refresh_data()
        await pilot.pause()
        await pilot.press("9")            # only 5 hosts -> ignored
        assert app.minimized == set()


async def test_manual_refresh_action():
    app = _demo_app()
    async with app.run_test(size=(135, 45)) as pilot:
        await pilot.press("r")
        await pilot.pause()
        assert app.snapshot is not None
