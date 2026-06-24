"""Tests for the MCP wrapper.

The tool logic is importable without the MCP SDK, so most tests run anywhere;
the registration test is skipped when ``mcp`` isn't installed.
"""

from __future__ import annotations

from unittest import mock

import pytest

import mcp_server
from core.collector import Host, Partition, Snapshot


def test_cluster_overview_missing_config(monkeypatch):
    monkeypatch.setenv("CLUSTER_MONITOR_CONFIG", "/no/such/config.json")
    out = mcp_server.cluster_overview()
    assert out["clusters"] == [] and "config not found" in out["error"]


def test_my_jobs_missing_config(monkeypatch):
    monkeypatch.setenv("CLUSTER_MONITOR_CONFIG", "/no/such/config.json")
    out = mcp_server.my_jobs()
    assert out["hosts"] == [] and "config not found" in out["error"]


def test_cluster_overview_success(monkeypatch):
    snap = Snapshot(generated_at=1.0, hosts=[
        Host(name="C", cpus_free=10, cpus_total=20, gpus_free=2, gpus_total=4,
             partitions=[Partition(name="gpu", cpus_free=10, cpus_total=20,
                                   gpus_free=2, gpus_total=4, my_running=1)]),
    ])
    monkeypatch.setattr(mcp_server, "_load", lambda: {"hosts": []})
    monkeypatch.setattr("core.collector.collect", lambda *a, **k: snap)
    out = mcp_server.cluster_overview()
    assert out["clusters"][0]["name"] == "C"
    assert out["clusters"][0]["free"] == {"cpus": 10, "gpus": 2}
    assert out["clusters"][0]["partitions"][0]["my_running"] == 1


def test_my_jobs_success(monkeypatch):
    snap = Snapshot(generated_at=1.0, hosts=[Host(name="C")])
    monkeypatch.setattr(mcp_server, "_load", lambda: {"hosts": []})
    monkeypatch.setattr(mcp_server, "collect", lambda *a, **k: snap)
    out = mcp_server.my_jobs()
    assert out["hosts"][0]["name"] == "C" and "totals" in out


def test_tools_report_invalid_config(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")  # no "hosts" -> load_config raises ValueError
    monkeypatch.setenv("CLUSTER_MONITOR_CONFIG", str(bad))
    assert "invalid config" in mcp_server.cluster_overview()["error"]
    assert "invalid config" in mcp_server.my_jobs()["error"]


def test_config_path_env_override(monkeypatch):
    monkeypatch.setenv("CLUSTER_MONITOR_CONFIG", "/custom/path.json")
    assert mcp_server._config_path() == "/custom/path.json"


def test_build_server_registers_tools():
    pytest.importorskip("mcp")
    server = mcp_server.build_server()
    assert server is not None
    # Best-effort introspection of the registered tools (ToolManager.list_tools
    # is the stable sync accessor); fall back to a smoke test if the internal
    # layout differs across FastMCP versions.
    mgr = getattr(server, "_tool_manager", None)
    if mgr is not None and hasattr(mgr, "list_tools"):
        names = {t.name for t in mgr.list_tools()}
        assert {"cluster_overview", "my_jobs"} <= names
