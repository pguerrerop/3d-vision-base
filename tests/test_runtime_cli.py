from __future__ import annotations

import json
from types import SimpleNamespace

from scripts import runtime as runtime_cli


def test_runtime_cli_list(monkeypatch, capsys) -> None:
    class FakeSupervisor:
        def list_processes(self):
            return [{"process_id": "trispector_ftp", "status": "stopped"}]

    monkeypatch.setattr(runtime_cli, "get_runtime_supervisor", lambda **_kwargs: FakeSupervisor())
    monkeypatch.setattr("sys.argv", ["runtime.py", "list"])
    exit_code = runtime_cli.main()
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert exit_code == 0
    assert payload["processes"][0]["process_id"] == "trispector_ftp"


def test_runtime_cli_logs(monkeypatch, capsys) -> None:
    class FakeSupervisor:
        def tail_logs(self, process_id: str, limit: int = 200):
            return {"process_id": process_id, "lines": [f"tail={limit}"]}

    monkeypatch.setattr(runtime_cli, "get_runtime_supervisor", lambda **_kwargs: FakeSupervisor())
    monkeypatch.setattr("sys.argv", ["runtime.py", "logs", "trispector_ftp", "--limit", "7"])
    exit_code = runtime_cli.main()
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert exit_code == 0
    assert payload["lines"] == ["tail=7"]


def test_runtime_cli_start_foreground(monkeypatch) -> None:
    class FakeProc:
        def wait(self):
            return 0

        def terminate(self):
            return None

    class FakeSupervisor:
        def build_start_command(self, process_id: str):
            return ["python3", "-m", "vision_3d_acquisition.runtime.process_runner", process_id]

    monkeypatch.setattr(runtime_cli, "get_runtime_supervisor", lambda **_kwargs: FakeSupervisor())
    monkeypatch.setattr(runtime_cli.subprocess, "Popen", lambda *_args, **_kwargs: FakeProc())
    monkeypatch.setattr("sys.argv", ["runtime.py", "start", "trispector_ftp", "--foreground"])
    exit_code = runtime_cli.main()
    assert exit_code == 0


def test_runtime_cli_selftest(monkeypatch, capsys) -> None:
    class FakeSupervisor:
        def ftp_selftest(self, process_id: str):
            return {"ok": True, "process_id": process_id, "diagnostics": ["connected", "authenticated", "uploaded", "verified"]}

    monkeypatch.setattr(runtime_cli, "get_runtime_supervisor", lambda **_kwargs: FakeSupervisor())
    monkeypatch.setattr("sys.argv", ["runtime.py", "selftest", "trispector_ftp"])
    exit_code = runtime_cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True


def test_runtime_cli_start_foreground(monkeypatch) -> None:
    class FakeSupervisor:
        def build_start_command(self, process_id: str):
            assert process_id == "trispector_ftp"
            return ["python3", "-c", "print('ok')"]

    class FakeProc:
        def wait(self):
            return 0

    monkeypatch.setattr(runtime_cli, "get_runtime_supervisor", lambda **_kwargs: FakeSupervisor())
    monkeypatch.setattr(runtime_cli.subprocess, "Popen", lambda _cmd: FakeProc())
    monkeypatch.setattr("sys.argv", ["runtime.py", "start", "trispector_ftp", "--foreground"])
    exit_code = runtime_cli.main()
    assert exit_code == 0
