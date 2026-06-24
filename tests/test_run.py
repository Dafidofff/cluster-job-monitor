"""Tests for the run.py entry point (arg handling, --once/--demo)."""

from __future__ import annotations

import json

import run


def test_once_demo(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run.py", "--once", "--demo"])
    assert run.main() == 0
    out = capsys.readouterr().out
    assert "CLUSTER JOBS" in out and "Larry" in out


def test_once_demo_json(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run.py", "--once", "--demo", "--json"])
    assert run.main() == 0
    data = json.loads(capsys.readouterr().out)
    assert "hosts" in data and data["totals"]["running"] >= 1


def test_overview_demo_json(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run.py", "--overview", "--demo", "--json"])
    assert run.main() == 0
    data = json.loads(capsys.readouterr().out)
    names = [c["name"] for c in data["clusters"]]
    assert "Snellius" in names
    snellius = next(c for c in data["clusters"] if c["name"] == "Snellius")
    assert snellius["free"]["gpus"] == 14
    assert snellius["partitions"][0]["name"] == "gpu_a100"


def test_overview_demo_table(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run.py", "--overview", "--demo"])
    assert run.main() == 0
    out = capsys.readouterr().out
    assert "Snellius" in out and "PARTITION" in out and "gpu_a100" in out


def test_once_real_config(monkeypatch, capsys, tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"hosts": [
        {"name": "Hidden", "ssh": "x", "minimized": True},
    ]}))
    # No real SSH happens fast here, but the host is collapsed via config and
    # squeue is absent -> the run must still complete and print the dashboard.
    monkeypatch.setattr("sys.argv", ["run.py", "--once", "--config", str(cfg)])
    assert run.main() == 0
    assert "Hidden" in capsys.readouterr().out


def test_missing_config_returns_2(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run.py", "--config", "/no/such/file.json"])
    assert run.main() == 2
    assert "config not found" in capsys.readouterr().err


def test_invalid_config_returns_2(monkeypatch, capsys, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")  # no "hosts"
    monkeypatch.setattr("sys.argv", ["run.py", "--config", str(bad)])
    assert run.main() == 2
    assert "invalid config" in capsys.readouterr().err
