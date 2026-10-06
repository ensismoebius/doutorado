#!/usr/bin/env python3
"""Tests for the remote-GA collection path.

Covers the three things that were actually broken:
  1. reading the last record of a remote JSONL whose last line has no newline
     (server.py::_last_jsonl_record) — this was the HTTP 500.
  2. parsing SSH stdout that carries a login-banner ahead of the JSON payload
     (collect_ga_local.py::_parse_remote_stdout).
  3. the remote command's conda activation and shell-quoting
     (collect_ga_local.py::build_remote_cmd).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline", "meeting01"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline", "meeting01", "dashboard"))

from collect_ga_local import (  # noqa: E402
    _finite_or_none,
    _parse_remote_stdout,
    _ssh_error_tail,
    append_record,
    build_remote_cmd,
)
from ga_cache import GaSearchState  # noqa: E402
from gridunesp_config import GridUnespConfig, load_config  # noqa: E402
from server import MAX_GA_FINISHED, SseDedupe, _last_jsonl_record  # noqa: E402


# ── SseDedupe: a tab left open must not re-send identical panels ─────────────


def _events(frames):
    return [f.split("\n", 1)[0].removeprefix("event: ") for f in frames]


def test_sse_first_tick_sends_every_present_panel():
    d = SseDedupe()
    frames = d.frames({"summary": {"a": 1}, "fold_grid": [1], "ga_summary": []})
    assert _events(frames) == ["summary", "fold_grid", "ga_summary"]


def test_sse_second_identical_tick_sends_nothing():
    d = SseDedupe()
    snap = {"summary": {"a": 1}, "aggregation": {"x": 2}}
    assert d.frames(snap), "first tick must send"
    assert d.frames(dict(snap)) == [], "unchanged panels must be suppressed"


def test_sse_resends_only_the_panel_that_moved():
    d = SseDedupe()
    d.frames({"summary": {"a": 1}, "aggregation": {"x": 2}})
    frames = d.frames({"summary": {"a": 2}, "aggregation": {"x": 2}})
    assert _events(frames) == ["summary"]


def test_sse_omits_absent_panels():
    d = SseDedupe()
    assert _events(d.frames({"summary": {"a": 1}})) == ["summary"]
    assert _events(d.frames({})) == []


def test_sse_sanitizes_nan_instead_of_emitting_invalid_json():
    d = SseDedupe()
    frames = d.frames({"summary": {"mse": float("nan")}})
    assert len(frames) == 1
    assert "NaN" not in frames[0]
    json.loads(frames[0].split("data: ", 1)[1])


def test_sse_skips_an_unserializable_panel_without_raising():
    d = SseDedupe()
    frames = d.frames({"summary": object(), "fold_grid": {"ok": 1}})
    assert _events(frames) == ["fold_grid"]


def test_sse_frame_shape_is_parseable():
    d = SseDedupe()
    f = d.frames({"summary": {"a": 1}})[0]
    assert f.startswith("event: summary\ndata: ")
    assert f.endswith("\n\n")


# ── _last_jsonl_record ───────────────────────────────────────────────────────


def test_last_record_without_trailing_newline(tmp_path):
    """The reported bug: splitlines()[-1] raised IndexError on an empty tail."""
    p = tmp_path / "ga.jsonl"
    p.write_text('{"cells": [1]}', encoding="utf-8")  # no '\n'
    assert _last_jsonl_record(str(p)) == {"cells": [1]}


def test_last_record_picks_the_final_record(tmp_path):
    p = tmp_path / "ga.jsonl"
    p.write_text('{"n": 1}\n{"n": 2}\n{"n": 3}\n', encoding="utf-8")
    assert _last_jsonl_record(str(p))["n"] == 3


def test_last_record_skips_a_corrupt_final_line(tmp_path):
    """A half-written tail record must not mask the last good one."""
    p = tmp_path / "ga.jsonl"
    p.write_text('{"n": 1}\n{"n": 2, "partia', encoding="utf-8")
    assert _last_jsonl_record(str(p))["n"] == 1


def test_last_record_tolerates_nan(tmp_path):
    p = tmp_path / "ga.jsonl"
    p.write_text('{"best": NaN, "n": 1}\n', encoding="utf-8")
    rec = _last_jsonl_record(str(p))
    assert rec["best"] is None
    assert rec["n"] == 1


def test_last_record_empty_file(tmp_path):
    p = tmp_path / "ga.jsonl"
    p.write_text("", encoding="utf-8")
    assert _last_jsonl_record(str(p)) is None


def test_last_record_missing_file(tmp_path):
    assert _last_jsonl_record(str(tmp_path / "nope.jsonl")) is None


# ── append_record / _parse_remote_stdout round trip ──────────────────────────


def test_append_then_read_round_trip(tmp_path):
    """What append_record writes must be readable by _last_jsonl_record."""
    p = tmp_path / "ga.jsonl"
    append_record(str(p), {"n": 1, "val": 0.5})
    append_record(str(p), {"n": 2, "val": 0.25})
    assert _last_jsonl_record(str(p))["n"] == 2


def test_append_record_refuses_non_finite(tmp_path):
    """allow_nan=False keeps the file valid JSON, as the SSE contract requires."""
    p = tmp_path / "ga.jsonl"
    with pytest.raises(ValueError):
        append_record(str(p), {"best": float("nan")})


# ── _parse_remote_stdout ─────────────────────────────────────────────────────


def test_parse_plain_payload():
    assert _parse_remote_stdout('{"cells": [1], "ts": 9}')["ts"] == 9


def test_parse_tolerates_login_banner():
    out = (
        "Last login: Tue Sep 23 10:00:00 2025 from 10.0.0.5\n"
        "*********************************************************\n"
        '{"cells": [{"dataset": "fsdd"}], "ts": 9}\n'
    )
    assert _parse_remote_stdout(out)["cells"][0]["dataset"] == "fsdd"


def test_parse_uses_the_last_object():
    out = '{"n": 1}\n{"n": 2}\n'
    assert _parse_remote_stdout(out)["n"] == 2


def test_parse_rejects_non_object():
    with pytest.raises(ValueError):
        _parse_remote_stdout("[1, 2, 3]\n")


def test_parse_rejects_empty():
    with pytest.raises(ValueError):
        _parse_remote_stdout("")


# ── _ssh_error_tail ──────────────────────────────────────────────────────────


def _noisy_stderr(traceback_text):
    return (
        "** WARNING: connection is not using a post-quantum key exchange algorithm.\n"
        "** This session may be vulnerable to \"store now, decrypt later\" attacks.\n"
        "** See https://openssh.com/pq.html\n"
        "========================================================================================\n"
        "\tYou are using Miniconda with libmamba-server. Installation proccess\n"
        "\tis supposed to be faster than common Conda modules.\n"
        "========================================================================================\n"
        "\tAfter creation of environment, try activation via command\n"
        f"{traceback_text}\n"
    )


def test_ssh_error_tail_surfaces_the_traceback():
    """The banner eats ~500 bytes on every connection; a head-truncation shows
    only the banner and hides why the collection failed."""
    tb = "Traceback (most recent call last):\n  File \"<stdin>\", line 3\nImportError: cannot import name 'collect_ga'"
    tail = _ssh_error_tail(_noisy_stderr(tb))
    assert "ImportError" in tail
    assert "cannot import name 'collect_ga'" in tail
    assert "post-quantum" not in tail
    assert "libmamba-server" not in tail


def test_ssh_error_tail_keeps_the_end_when_very_long():
    """Tracebacks live at the end, so an over-long message is clipped there."""
    filler = "\n".join(f"line {i}" for i in range(500))
    tail = _ssh_error_tail(_noisy_stderr(filler + "\nImportError: boom"), limit=200)
    assert tail.endswith("ImportError: boom")
    assert len(tail) <= 200


def test_ssh_error_tail_empty_stderr():
    assert _ssh_error_tail("") == ""
    assert _ssh_error_tail("=====\n  \n") == ""


# ── build_remote_cmd ─────────────────────────────────────────────────────────


def _cfg(**kw):
    kw.setdefault("user", "u")
    return GridUnespConfig(**kw)


def test_remote_cmd_activates_the_remote_conda_env():
    cmd = build_remote_cmd(_cfg(), "results/meeting01", "t")
    assert "module load miniconda/24.4.0-libmamba" in cmd
    assert 'eval "$(conda shell.bash hook)"' in cmd
    assert "conda activate meeting01-build" in cmd
    # The chain must be &&-linked, or a failed activation falls through to the
    # system python3 (3.6.8), which cannot parse `from __future__`.
    assert "&& python3 -" in cmd


def test_remote_cmd_passes_python_over_a_quoted_heredoc():
    """`python3 -c` with nested quotes is the classic way this breaks."""
    cmd = build_remote_cmd(_cfg(), "results/meeting01", "t")
    assert "python3 - <<'MEETING01_PY_EOF'" in cmd
    assert "python3 -c" not in cmd


def test_remote_cmd_passes_results_path_verbatim():
    """One quoting context per value: repr() in the Python body, NOT a
    shlex.quote'd string re-wrapped by repr() (which would embed literal
    quote characters in the path the remote actually receives)."""
    cfg = _cfg(remote_dir="/tmp/repo")
    cmd = build_remote_cmd(cfg, "results/meeting01", "t")
    assert "collect_ga('/tmp/repo/results/meeting01', 't')" in cmd
    # No doubled-up quoting artefacts.
    assert "\\'" not in cmd


def test_remote_cmd_is_injection_safe_when_executed(tmp_path):
    """End-to-end: run the generated command through bash with a stub
    collect_ga, using a results_dir that tries to break out.  Asserts both
    that nothing executes and that the path arrives intact."""
    import subprocess

    canary = tmp_path / "PWNED"
    root = tmp_path / "repo"
    (root / "scripts" / "pipeline" / "meeting01").mkdir(parents=True)
    stub = root / "scripts" / "pipeline" / "meeting01" / "gridunesp_status_remote.py"
    # collect_ga must RETURN its payload — the outer json.dump serialises that
    # return value, so a stub that prints instead yields `null`.
    stub.write_text(
        "def collect_ga(results_dir, run_tag):\n"
        "    return {'got': results_dir, 'tag': run_tag}\n",
        encoding="utf-8",
    )

    hostile = f"x; touch {canary}"
    cfg = _cfg(remote_dir=str(root))
    cmd = build_remote_cmd(cfg, hostile, "t").replace(cfg.remote_activate(), "true")

    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert not canary.exists(), "shell injection executed"

    # The path must reach the remote Python intact, with no stray quote chars.
    rec = _parse_remote_stdout(proc.stdout)
    assert rec["got"] == os.path.join(str(root), hostile)
    assert rec["tag"] == "t"


def test_remote_cmd_quotes_a_remote_root_with_spaces():
    """remote_dir lands in *shell* position, so it does need shlex.quote()."""
    cfg = _cfg(remote_dir="/tmp/repo with space")
    assert "cd '/tmp/repo with space'" in build_remote_cmd(cfg, "r", "t")


def test_remote_cmd_embeds_the_full_remote_results_path():
    cfg = _cfg(remote_dir="/tmp/repo")
    cmd = build_remote_cmd(cfg, "results/meeting01", "t")
    assert "'/tmp/repo/results/meeting01'" in cmd


# ── ssh_argv / ssh_env: the two auth modes stay mutually exclusive ───────────


def test_password_mode_uses_sshpass_and_exports_ssHPass():
    cfg = _cfg(user="u", password="hunter2")
    argv = cfg.ssh_argv("CMD")
    assert argv[0] == "sshpass"
    assert "-e" in argv
    assert "CMD" == argv[-1]
    assert cfg.ssh_env()["SSHPASS"] == "hunter2"


def test_key_mode_uses_bare_ssh_with_batchmode():
    cfg = _cfg(user="u", password=None)
    argv = cfg.ssh_argv("CMD")
    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv
    assert cfg.ssh_env().get("SSHPASS") is None


def test_default_remote_dir_is_the_repo_checkout():
    assert _cfg().remote_dir == "software/nn"


# ── GaSearchState.progress: the series the convergence chart plots ───────────


def _ind(gen, mse, cost, feasible=True):
    return {
        "born_generation": gen,
        "val_mse": mse,
        "inference_cost": cost,
        "feasible": feasible,
    }


def test_progress_is_one_row_per_generation():
    s = GaSearchState("fsdd", 0, 0, "snn")
    for gen, mse, cost in [(0, 1.0, 50), (0, 0.8, 40), (1, 0.5, 30), (2, 0.2, 20)]:
        s.add(_ind(gen, mse, cost))
    rows = s.progress
    assert [r["gen"] for r in rows] == [0, 1, 2]
    assert rows[0]["n"] == 2
    assert rows[0]["best_val_mse"] == 0.8  # min over gen 0
    assert rows[0]["best_cost"] == 40
    assert rows[2]["best_val_mse"] == 0.2


def test_progress_excludes_infeasible_from_the_best():
    s = GaSearchState("fsdd", 0, 0, "snn")
    s.add(_ind(0, 0.1, 5, feasible=False))  # better but not a solution
    s.add(_ind(0, 0.9, 90, feasible=True))
    row = s.progress[0]
    assert row["best_val_mse"] == 0.9
    assert row["best_cost"] == 90
    assert row["n"] == 2  # n counts every individual, feasible or not


def test_progress_reports_none_for_a_generation_with_no_feasible_member():
    s = GaSearchState("fsdd", 0, 0, "snn")
    s.add(_ind(3, None, None, feasible=False))
    row = s.progress[0]
    assert row["best_val_mse"] is None
    assert row["best_cost"] is None


def test_progress_is_empty_for_a_fresh_state():
    assert GaSearchState("fsdd", 0, 0, "snn").progress == []


def test_summary_exposes_progress_and_stays_json_serializable():
    s = GaSearchState("fsdd", 0, 0, "snn")
    s.add(_ind(0, 1.0, 10))
    s.add(_ind(1, float("nan"), 5))
    summary = s.summary()
    assert "progress" in summary
    json.dumps(summary, allow_nan=False)  # must not raise


# ── _finite_or_none ──────────────────────────────────────────────────────────


def test_finite_or_none_recurses():
    out = _finite_or_none({"a": [float("inf"), {"b": float("nan")}], "c": 1.5})
    assert out == {"a": [None, {"b": None}], "c": 1.5}
    json.dumps(out, allow_nan=False)


def test_finite_or_none_passes_through_non_floats():
    assert _finite_or_none({"s": "x", "i": 3, "n": None}) == {"s": "x", "i": 3, "n": None}


def test_max_ga_finished_is_a_sane_bound():
    """480 cells under the standard profile; the cap must be far below that."""
    assert 0 < MAX_GA_FINISHED < 480


# ── load_config: an HTTP request must never block on a terminal prompt ───────
# The dashboard serves POST /api/ga/remote/collect from a FastAPI worker whose
# stdin is a pipe (/dev/null), not a terminal.  If load_config ever prompts
# there, the request hangs until a client timeout instead of returning an
# error.  Both prompt sites are gated on sys.stdin.isatty(); these tests pin
# that guarantee down so a future "helpful" prompt cannot silently reintroduce
# a hang.


@pytest.fixture
def clean_env(monkeypatch):
    """No .env file, no inherited GRIDUNESP_* variables, stdin is not a TTY."""
    import gridunesp_config

    monkeypatch.setattr(gridunesp_config, "_read_env_file", lambda: {})
    for key in list(os.environ):
        if key.startswith("GRIDUNESP_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    return monkeypatch


def test_load_config_without_user_raises_instead_of_prompting(clean_env):
    with pytest.raises(SystemExit) as exc:
        load_config()
    assert "GRIDUNESP_USER" in str(exc.value)


def test_load_config_key_auth_is_noninteractive(clean_env):
    clean_env.setenv("GRIDUNESP_USER", "tester")
    cfg = load_config()
    assert cfg.user == "tester"
    assert cfg.password is None  # key auth, BatchMode=yes, no prompt


def test_load_config_reads_password_when_present(clean_env):
    clean_env.setenv("GRIDUNESP_USER", "tester")
    clean_env.setenv("GRIDUNESP_PASSWORD", "s3cret")
    cfg = load_config()
    assert (cfg.user, cfg.password) == ("tester", "s3cret")


def test_load_config_does_not_write_env_file_when_noninteractive(clean_env, tmp_path):
    """A key-auth load must not persist anything, even mid-request."""
    import gridunesp_config

    env_path = tmp_path / ".env"
    monkey = gridunesp_config.ENV_FILE
    clean_env.setattr(gridunesp_config, "ENV_FILE", str(env_path))
    clean_env.setenv("GRIDUNESP_USER", "tester")
    load_config()
    assert not env_path.exists()
    assert monkey  # original constant untouched
