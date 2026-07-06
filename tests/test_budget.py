"""Tests for the budget/accounting feature.

Covers the two data sources folded into the capacity overview:
- native SLURM ``sacctmgr`` (granted cap) + ``sreport`` (used hours), and
- the SURF-specific, best-effort ``budget-overview`` text parser.

Parsing functions are exercised standalone with raw-text fixtures; the
command construction and ``_collect_host``/``collect_overview`` paths follow the
same subprocess-mocking idioms as ``tests/test_overview.py``.
"""

from __future__ import annotations

import json
import subprocess
import types

import pytest

from cluster_job_monitor import collector
from cluster_job_monitor.collector import (
    Budget, Host, Snapshot, build_overview, collect, collect_overview,
    parse_budget_overview_output, parse_sacctmgr_and_sreport,
    parse_sacctmgr_output, parse_sreport_output,
)
from cluster_job_monitor.collector import _build_accounting_cmd, _parse_tres_mins


# --------------------------------------------------------------------------- #
# Budget dataclass
# --------------------------------------------------------------------------- #
def test_budget_pct_used_and_to_dict():
    b = Budget(account="prjs0123", source="sacctmgr", unit="core-hours",
               granted=1000.0, used=250.0, remaining=750.0)
    assert b.pct_used == 0.25
    d = b.to_dict()
    assert d["account"] == "prjs0123" and d["source"] == "sacctmgr"
    assert d["granted"] == 1000.0 and d["remaining"] == 750.0
    assert d["pct_used"] == 0.25


@pytest.mark.parametrize("granted,used,expected", [
    (0.0, 10.0, None),      # zero grant -> no division, pct is None
    (None, 10.0, None),     # unknown grant (unlimited/external billing)
    (100.0, None, None),    # unknown usage
    (100.0, 250.0, 1.0),    # over budget clamps to 1.0
    (100.0, -5.0, 0.0),     # never negative
])
def test_budget_pct_used_edge_cases(granted, used, expected):
    b = Budget(account="a", source="sacctmgr", unit="core-hours",
               granted=granted, used=used)
    assert b.pct_used == expected


# --------------------------------------------------------------------------- #
# TRES-minutes helper
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [
    ("cpu=175200000,gres/gpu=8760000", 175200000.0),
    ("gres/gpu=8760000,cpu=60", 60.0),
    ("", None),
    ("   ", None),
    ("mem=1024", None),          # no cpu key
    ("cpu=notanumber", None),
])
def test_parse_tres_mins(value, expected):
    assert _parse_tres_mins(value) == expected


# --------------------------------------------------------------------------- #
# sacctmgr parsing
# --------------------------------------------------------------------------- #
def test_parse_sacctmgr_uses_grptresmins_in_hours():
    # 6000 cpu-minutes -> 100 cpu-hours.
    text = "prjs0123|cpu=6000|cpu=3000\n"
    assert parse_sacctmgr_output(text) == {"prjs0123": 100.0}


def test_parse_sacctmgr_falls_back_to_run_mins():
    # GrpTRESMins unset -> use GrpTRESRunMins (1200 min -> 20 h).
    text = "acct|(null)|cpu=1200\n"
    # "(null)" has no cpu= pair, so treated as unset -> fall back.
    assert parse_sacctmgr_output(text) == {"acct": 20.0}


def test_parse_sacctmgr_both_unset_is_none():
    text = "acct||\n"
    assert parse_sacctmgr_output(text) == {"acct": None}


def test_parse_sacctmgr_skips_malformed_and_keeps_multiple():
    text = (
        "\n"
        "short|line\n"                 # < 3 fields -> skipped
        "|cpu=60|cpu=60\n"             # blank account -> skipped
        "a|cpu=6000|\n"
        "b|cpu=12000|\n"
    )
    assert parse_sacctmgr_output(text) == {"a": 100.0, "b": 200.0}


# --------------------------------------------------------------------------- #
# sreport parsing
# --------------------------------------------------------------------------- #
def test_parse_sreport_sums_rows_per_account():
    text = (
        "prjs0123|40\n"
        "prjs0123|10\n"       # second user under same account -> summed
        "other|5\n"
    )
    assert parse_sreport_output(text) == {"prjs0123": 50.0, "other": 5.0}


def test_parse_sreport_skips_non_numeric_and_blank():
    text = (
        "\n"
        "acct|notanumber\n"
        "acct|12.5\n"
        "short\n"
    )
    assert parse_sreport_output(text) == {"acct": 12.5}


# --------------------------------------------------------------------------- #
# Combining sacctmgr + sreport
# --------------------------------------------------------------------------- #
def test_parse_sacctmgr_and_sreport_combines_to_remaining():
    budgets = parse_sacctmgr_and_sreport("prjs0123|cpu=6000|\n", "prjs0123|40\n")
    assert len(budgets) == 1
    b = budgets[0]
    assert b.account == "prjs0123" and b.source == "sacctmgr"
    assert b.unit == "core-hours"
    assert b.granted == 100.0 and b.used == 40.0 and b.remaining == 60.0


def test_parse_sacctmgr_and_sreport_one_sided_presence():
    # Account with usage but no cap (granted None) -> remaining None.
    budgets = parse_sacctmgr_and_sreport("prjs0123||\n", "prjs0123|40\n")
    b = budgets[0]
    assert b.granted is None and b.used == 40.0 and b.remaining is None

    # Account present only in sreport (not in sacctmgr) still yields a Budget.
    budgets = parse_sacctmgr_and_sreport("", "ghost|5\n")
    assert budgets[0].account == "ghost" and budgets[0].granted is None


def test_parse_sacctmgr_and_sreport_empty():
    assert parse_sacctmgr_and_sreport("", "") == []


# --------------------------------------------------------------------------- #
# budget-overview best-effort parsing
# --------------------------------------------------------------------------- #
def test_parse_budget_overview_three_numbers():
    text = (
        "Account   Granted   Used   Remaining\n"   # header -> skipped
        "prjs0123  100000    45000  55000  SBU\n"
    )
    budgets = parse_budget_overview_output(text)
    assert len(budgets) == 1
    b = budgets[0]
    assert b.account == "prjs0123" and b.source == "budget-overview"
    assert b.granted == 100000.0 and b.used == 45000.0 and b.remaining == 55000.0
    assert b.unit == "sbu"
    assert "best-effort" in b.note


def test_parse_budget_overview_two_numbers_infers_remaining():
    # Two numbers read as used, granted; remaining inferred.
    text = "myproj 42 1000 core-hours\n"
    b = parse_budget_overview_output(text)[0]
    assert b.used == 42.0 and b.granted == 1000.0 and b.remaining == 958.0
    assert b.unit == "core-hours"


def test_parse_budget_overview_pipe_delimited_and_thousands():
    text = "prjs0123 | 1,000,000 | 250,000 | 750,000 | hours\n"
    b = parse_budget_overview_output(text)[0]
    assert b.granted == 1000000.0 and b.used == 250000.0 and b.remaining == 750000.0
    assert b.unit == "hours"


def test_parse_budget_overview_unit_unknown_when_absent():
    b = parse_budget_overview_output("proj 10 100 90\n")[0]
    assert b.unit == "unknown"


def test_parse_budget_overview_skips_garbage_never_raises():
    text = (
        "===== Budget overview for user foo =====\n"
        "(no active budgets)\n"
        "Contact servicedesk@surf.nl for questions.\n"
        "\n"
    )
    assert parse_budget_overview_output(text) == []
    # Total garbage / binary-ish text must not raise.
    assert parse_budget_overview_output("!@#$%^&*()\n\n   \n") == []


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #
def test_build_accounting_cmd_empty_without_config():
    assert _build_accounting_cmd({"name": "h"}) == ""
    assert _build_accounting_cmd({"accounting": ""}) == ""
    assert _build_accounting_cmd({"accounting": "bogus-mode"}) == ""


def test_build_accounting_cmd_sacctmgr_shape():
    cmd = _build_accounting_cmd({"accounting": "sacctmgr"})
    assert "sacctmgr show associations -n -P" in cmd
    assert "sreport -n -P -t hours cluster AccountUtilizationByUser" in cmd
    assert collector.SACCTMGR_SENTINEL in cmd and collector.SREPORT_SENTINEL in cmd
    assert "Accounts=" not in cmd            # no account filter when unset


def test_build_accounting_cmd_sacctmgr_account_filter():
    cmd = _build_accounting_cmd({"accounting": "sacctmgr",
                                 "accounting_account": "prjs0123"})
    assert "Accounts=prjs0123" in cmd        # appears on both sacctmgr + sreport
    assert cmd.count("Accounts=prjs0123") == 2


def test_build_accounting_cmd_budget_overview_shape_and_args():
    cmd = _build_accounting_cmd({"accounting": "budget-overview"})
    assert cmd == f"echo '{collector.BUDGET_OVERVIEW_SENTINEL}'; budget-overview 2>/dev/null || true"
    cmd2 = _build_accounting_cmd({"accounting": "budget-overview",
                                  "accounting_args": "-p gpu"})
    assert "budget-overview -p gpu 2>/dev/null" in cmd2


def test_build_squeue_cmd_includes_accounting_and_keeps_exit_last():
    acct = _build_accounting_cmd({"accounting": "sacctmgr"})
    full = collector._build_squeue_cmd(with_partitions=True, accounting_cmd=acct)
    assert collector.SINFO_SENTINEL in full and collector.QUEUE_SENTINEL in full
    assert collector.SACCTMGR_SENTINEL in full
    # sinfo, then queue, then accounting, then exit.
    assert (full.index(collector.SINFO_SENTINEL)
            < full.index(collector.QUEUE_SENTINEL)
            < full.index(collector.SACCTMGR_SENTINEL)
            < full.index("exit $rc"))


def test_build_squeue_cmd_accounting_only_without_partitions():
    acct = _build_accounting_cmd({"accounting": "budget-overview"})
    cmd = collector._build_squeue_cmd(with_partitions=False, accounting_cmd=acct)
    assert collector.BUDGET_OVERVIEW_SENTINEL in cmd
    assert collector.SINFO_SENTINEL not in cmd     # no capacity probe requested
    assert "exit $rc" in cmd


def test_combined_command_tolerates_failing_accounting():
    # squeue + sinfo OK, accounting tools missing -> overall success, jobs kept.
    acct = _build_accounting_cmd({"accounting": "sacctmgr"})
    cmd = (
        "squeue() { echo JOBLINE; return 0; }; sinfo() { echo SINFODATA; }; "
        "sacctmgr() { return 127; }; sreport() { return 127; }; "
        + collector._build_squeue_cmd(True, acct)
    )
    proc = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "JOBLINE" in proc.stdout
    assert collector.SACCTMGR_SENTINEL in proc.stdout


# --------------------------------------------------------------------------- #
# _collect_host / collect_overview integration (subprocess mocked)
# --------------------------------------------------------------------------- #
_SQUEUE = "1|train|RUNNING|gpu_a100|1:00|2:00|1|8|gres/gpu:2|node01|t\n"
_SINFO = "node01 gpu_a100 idle 0/128/0/128 gpu:a100:8 gpu:a100:0\n"
_QUEUE = "gpu_a100|RUNNING|1:00:00\n"


def _combined(accounting: str) -> str:
    return (_SQUEUE
            + f"{collector.SINFO_SENTINEL}\n" + _SINFO
            + f"{collector.QUEUE_SENTINEL}\n" + _QUEUE
            + accounting)


def _fake_run(stdout: str):
    def run(argv, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    return run


def test_collect_host_sacctmgr_populates_budgets(monkeypatch):
    acct = (f"{collector.SACCTMGR_SENTINEL}\nprjs0123|cpu=6000|\n"
            f"{collector.SREPORT_SENTINEL}\nprjs0123|40\n")
    monkeypatch.setattr(collector.subprocess, "run",
                        _fake_run(_combined(acct)))
    snap = collect({"hosts": [{"name": "Snellius", "ssh": "s",
                               "accounting": "sacctmgr"}]},
                   with_partitions=True)
    h = snap.hosts[0]
    # Budget parsed, and the pipe-delimited budget rows did NOT leak into the
    # queue parse or become phantom partitions.
    assert len(h.budgets) == 1
    b = h.budgets[0]
    assert b.account == "prjs0123" and b.granted == 100.0 and b.used == 40.0
    assert b.remaining == 60.0
    assert h.jobs[0].jobid == "1"                 # squeue still parsed
    assert [p.name for p in h.partitions] == ["gpu_a100"]


def test_collect_host_budget_overview_populates_budgets(monkeypatch):
    acct = (f"{collector.BUDGET_OVERVIEW_SENTINEL}\n"
            "prjs0123 100000 45000 55000 SBU\n")
    monkeypatch.setattr(collector.subprocess, "run",
                        _fake_run(_combined(acct)))
    snap = collect({"hosts": [{"name": "Snellius", "ssh": "s",
                               "accounting": "budget-overview"}]},
                   with_partitions=True)
    b = snap.hosts[0].budgets[0]
    assert b.source == "budget-overview" and b.granted == 100000.0
    assert b.unit == "sbu"


def test_collect_host_no_accounting_field(monkeypatch):
    captured = {}

    def run(argv, **kwargs):
        captured["argv"] = argv
        return types.SimpleNamespace(returncode=0, stdout=_combined(""), stderr="")

    monkeypatch.setattr(collector.subprocess, "run", run)
    snap = collect({"hosts": [{"name": "Plain", "ssh": "s"}]},
                   with_partitions=True)
    assert snap.hosts[0].budgets == []
    # The command sent to the host must not include any accounting probe.
    sent = " ".join(captured["argv"])
    assert "sacctmgr" not in sent and "budget-overview" not in sent


def test_collect_host_accounting_tool_absent_yields_empty(monkeypatch):
    # Sentinels present but the tools produced nothing (2>/dev/null || true) ->
    # no budgets, host still ok.
    acct = (f"{collector.SACCTMGR_SENTINEL}\n{collector.SREPORT_SENTINEL}\n")
    monkeypatch.setattr(collector.subprocess, "run",
                        _fake_run(_combined(acct)))
    snap = collect({"hosts": [{"name": "S", "ssh": "s",
                               "accounting": "sacctmgr"}]},
                   with_partitions=True)
    assert snap.hosts[0].ok is True
    assert snap.hosts[0].budgets == []


def test_collect_overview_includes_budget(monkeypatch):
    acct = (f"{collector.SACCTMGR_SENTINEL}\nprjs0123|cpu=6000|\n"
            f"{collector.SREPORT_SENTINEL}\nprjs0123|40\n")
    monkeypatch.setattr(collector.subprocess, "run",
                        _fake_run(_combined(acct)))
    ov = collect_overview({"hosts": [{"name": "Snellius", "ssh": "s",
                                      "accounting": "sacctmgr"}]})
    json.dumps(ov)                                # must be serialisable
    budget = ov["clusters"][0]["budget"]
    assert budget[0]["account"] == "prjs0123" and budget[0]["remaining"] == 60.0
    assert budget[0]["pct_used"] == 0.4


def test_build_overview_budget_empty_by_default():
    snap = Snapshot(generated_at=1.0, hosts=[Host(name="C")])
    ov = build_overview(snap)
    assert ov["clusters"][0]["budget"] == []
