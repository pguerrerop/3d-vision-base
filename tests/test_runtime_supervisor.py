from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.runtime.models import RuntimeProcessDefinition
from vision_3d_acquisition.runtime.process_runner import TriSpectorFtpRuntime
from vision_3d_acquisition.runtime.supervisor import RuntimeSupervisor


def make_settings(data_dir: Path) -> ApiSettings:
    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
        datasets_dir=data_dir / "datasets",
    )
    settings.ensure_directories()
    return settings


def test_runtime_supervisor_process_lifecycle_and_restart(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")

    def fake_registry(_settings: ApiSettings) -> dict[str, RuntimeProcessDefinition]:
        return {
            "demo_worker": RuntimeProcessDefinition(
                process_id="demo_worker",
                process_type="worker",
                command=[sys.executable, "-c", "import time; print('worker started'); time.sleep(60)"],
                enabled=True,
                source_id="worker_0",
                config={"kind": "demo"},
            )
        }

    monkeypatch.setattr("vision_3d_acquisition.runtime.supervisor.build_runtime_registry", fake_registry)
    supervisor = RuntimeSupervisor(settings)
    started = supervisor.start("demo_worker")
    assert started["status"] == "running"
    assert started["desired_state"] == "running"
    assert started["observed_state"] == "running"
    assert started["pid"] is not None

    restarted = supervisor.restart("demo_worker")
    assert restarted["status"] == "running"
    assert restarted["restart_count"] == 1

    events = supervisor.events("demo_worker", limit=10)["events"]
    assert any(item["event_type"] == "PROCESS_STARTED" for item in events)

    stopped = supervisor.stop("demo_worker")
    assert stopped["status"] == "stopped"
    assert stopped["desired_state"] == "stopped"
    assert stopped["observed_state"] == "stopped"
    assert stopped["pid"] is None
    logs = supervisor.tail_logs("demo_worker", limit=20)["lines"]
    assert any("PROCESS_STOPPED" in line or "PROCESS_EXITED" in line for line in logs)


def test_trispector_runtime_stability_and_adapter_invocation(tmp_path: Path, monkeypatch) -> None:
    upload_root = tmp_path / "uploads"
    incoming_dir = upload_root / "incoming"
    incoming_dir.mkdir(parents=True)
    upload = incoming_dir / "capture_001.png"
    upload.write_bytes(b"png-bytes")
    runtime = TriSpectorFtpRuntime(
        process_id="trispector_ftp",
        source_id="trispector_ftp_0",
        ftp_command="python -m pyftpdlib -p 2121",
        upload_dir=incoming_dir,
        poll_interval_sec=0.1,
        stable_checks=1,
    )

    events: list[tuple[str, str]] = []
    monkeypatch.setattr(runtime, "emit", lambda event_type, message, **_kwargs: events.append((event_type, message)))
    monkeypatch.setattr(runtime, "_start_ftp_server", lambda: None)
    monkeypatch.setattr(runtime, "_stop_ftp_server", lambda: None)
    monkeypatch.setattr(runtime, "_check_ftp_health", lambda: None)
    monkeypatch.setattr(runtime, "_emit_heartbeat", lambda: None)

    invocations: list[str] = []

    def fake_parse(path: Path) -> dict[str, object]:
        invocations.append(path.name)
        return {"take_id": "take_001", "processing": {"status": "processing_triggered"}}

    monkeypatch.setattr(runtime._adapter, "parse_and_register_trispector_upload", fake_parse)
    runtime._scan_and_process_uploads()
    runtime._scan_and_process_uploads()

    assert invocations == ["capture_001.png"]
    assert (runtime.processed_dir / "capture_001.png").is_file()
    assert any(item[0] == "FILE_STABLE" for item in events)
    assert any(item[0] == "TAKE_REGISTERED" for item in events)
    assert any(item[0] == "PROCESSING_TRIGGERED" for item in events)


def test_runtime_supervisor_startup_failure_sets_failed_startup(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")

    def fake_registry(_settings: ApiSettings) -> dict[str, RuntimeProcessDefinition]:
        return {
            "demo_worker": RuntimeProcessDefinition(
                process_id="demo_worker",
                process_type="worker",
                command=[sys.executable, "-c", "import sys; sys.exit(7)"],
                enabled=True,
                source_id="worker_0",
                config={},
            )
        }

    monkeypatch.setattr("vision_3d_acquisition.runtime.supervisor.build_runtime_registry", fake_registry)
    supervisor = RuntimeSupervisor(settings)
    try:
        supervisor.start("demo_worker")
        assert False, "expected startup failure"
    except RuntimeError:
        pass
    status = supervisor.status("demo_worker")
    assert status["status"] == "failed_startup"
    assert status["observed_state"] == "failed_startup"


def test_runtime_supervisor_validation_failure_on_missing_executable(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")

    def fake_registry(_settings: ApiSettings) -> dict[str, RuntimeProcessDefinition]:
        return {
            "demo_worker": RuntimeProcessDefinition(
                process_id="demo_worker",
                process_type="worker",
                command=["definitely_missing_executable_binary_xyz"],
                enabled=True,
                source_id="worker_0",
                config={},
            )
        }

    monkeypatch.setattr("vision_3d_acquisition.runtime.supervisor.build_runtime_registry", fake_registry)
    supervisor = RuntimeSupervisor(settings)
    try:
        supervisor.start("demo_worker")
        assert False, "expected validation failure"
    except ValueError:
        pass
    events = supervisor.events("demo_worker", limit=20)["events"]
    assert any(item["event_type"] == "PROCESS_VALIDATION_FAILED" for item in events)


def test_runtime_supervisor_recovers_orphaned_pid(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")

    def fake_registry(_settings: ApiSettings) -> dict[str, RuntimeProcessDefinition]:
        return {
            "demo_worker": RuntimeProcessDefinition(
                process_id="demo_worker",
                process_type="worker",
                command=["definitely_not_running_binary", "--foo"],
                enabled=True,
                source_id="worker_0",
                config={},
            )
        }

    process_dir = settings.data_dir / "runtime" / "processes" / "demo_worker"
    process_dir.mkdir(parents=True, exist_ok=True)
    (process_dir / "pid").write_text(
        '{"owner":"sensor_studio_runtime_supervisor_v1","pid":'
        + str(os.getpid())
        + ',"process_id":"demo_worker","command":["definitely_not_running_binary","--foo"]}',
        encoding="utf-8",
    )
    monkeypatch.setattr("vision_3d_acquisition.runtime.supervisor.build_runtime_registry", fake_registry)
    supervisor = RuntimeSupervisor(settings)
    status = supervisor.status("demo_worker")
    assert status["observed_state"] == "orphaned"
    assert status["status"] == "orphaned"


def test_runtime_supervisor_recovers_running_pid_across_instances(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    proc = subprocess.Popen(["sleep", "20"])
    try:
        command = ["sleep", "20"]

        def fake_registry(_settings: ApiSettings) -> dict[str, RuntimeProcessDefinition]:
            return {
                "demo_worker": RuntimeProcessDefinition(
                    process_id="demo_worker",
                    process_type="worker",
                    command=command,
                    enabled=True,
                    source_id="worker_0",
                    config={},
                )
            }

        process_dir = settings.data_dir / "runtime" / "processes" / "demo_worker"
        process_dir.mkdir(parents=True, exist_ok=True)
        (process_dir / "pid").write_text(
            '{"owner":"sensor_studio_runtime_supervisor_v1","pid":'
            + str(proc.pid)
            + ',"process_id":"demo_worker","command_signature":"'
            + " ".join(command[:8])
            + '"}',
            encoding="utf-8",
        )
        (process_dir / "instance.json").write_text(
            '{"process_id":"demo_worker","desired_state":"running","cwd":"'
            + str(Path.cwd())
            + '","command_signature":"'
            + " ".join(command[:8])
            + '"}',
            encoding="utf-8",
        )
        monkeypatch.setattr("vision_3d_acquisition.runtime.supervisor.build_runtime_registry", fake_registry)
        monkeypatch.setattr(RuntimeSupervisor, "_pid_owned_by_definition", lambda _self, _pid, _defn, _record=None, _metadata=None: True)
        supervisor = RuntimeSupervisor(settings)
        status = supervisor.status("demo_worker")
        assert status["observed_state"] == "running"
        assert status["status"] == "running"
        assert status["pid"] == proc.pid
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_runtime_supervisor_loads_persisted_event_history(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")

    def fake_registry(_settings: ApiSettings) -> dict[str, RuntimeProcessDefinition]:
        return {
            "demo_worker": RuntimeProcessDefinition(
                process_id="demo_worker",
                process_type="worker",
                command=[sys.executable, "-c", "import time; time.sleep(1.2)"],
                enabled=True,
                source_id="worker_0",
                config={},
            )
        }

    monkeypatch.setattr("vision_3d_acquisition.runtime.supervisor.build_runtime_registry", fake_registry)
    supervisor = RuntimeSupervisor(settings)
    supervisor.start("demo_worker")
    supervisor.stop("demo_worker")

    second = RuntimeSupervisor(settings)
    events = second.events("demo_worker", limit=20)["events"]
    assert any(item["event_type"] == "PROCESS_STARTED" for item in events)


def test_runtime_supervisor_normalizes_ftp_command_and_injects_root(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    upload_dir = tmp_path / "incoming"
    upload_dir.mkdir(parents=True)

    def fake_registry(_settings: ApiSettings) -> dict[str, RuntimeProcessDefinition]:
        return {
            "trispector_ftp": RuntimeProcessDefinition(
                process_id="trispector_ftp",
                process_type="trispector_ftp_runtime",
                command=[sys.executable, "-m", "worker", "--ftp-command", "python -m pyftpdlib -p 2121", "--upload-dir", str(upload_dir)],
                enabled=True,
                source_id="trispector_ftp_0",
                config={"command": "python -m pyftpdlib -p 2121", "upload_dir": str(upload_dir), "auth_mode": "anonymous"},
            )
        }

    monkeypatch.setattr("vision_3d_acquisition.runtime.supervisor.build_runtime_registry", fake_registry)
    monkeypatch.setattr("vision_3d_acquisition.runtime.supervisor.sp.check_call", lambda *args, **kwargs: 0)
    monkeypatch.setattr(RuntimeSupervisor, "_port_available", lambda _self, _port: True)
    supervisor = RuntimeSupervisor(settings)
    command = supervisor.build_start_command("trispector_ftp")
    ftp_index = command.index("--ftp-command") + 1
    assert command[ftp_index].startswith(sys.executable)
    assert "-d" in command[ftp_index]
    assert "-w" in command[ftp_index]
    events = supervisor.events("trispector_ftp", limit=20)["events"]
    event_types = {item["event_type"] for item in events}
    assert "FTP_COMMAND_NORMALIZED" in event_types
    assert "FTP_ROOT_INJECTED" in event_types
    assert "FTP_WRITE_ENABLED" in event_types
    assert "FTP_AUTH_ANONYMOUS" in event_types


def test_runtime_supervisor_validation_failure_emits_event(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")

    def fake_registry(_settings: ApiSettings) -> dict[str, RuntimeProcessDefinition]:
        return {
            "trispector_ftp": RuntimeProcessDefinition(
                process_id="trispector_ftp",
                process_type="trispector_ftp_runtime",
                command=[sys.executable, "-m", "worker", "--ftp-command", "python -m pyftpdlib -p 2121"],
                enabled=True,
                source_id="trispector_ftp_0",
                config={"command": "python -m pyftpdlib -p 2121", "upload_dir": str(tmp_path / "missing")},
            )
        }

    monkeypatch.setattr("vision_3d_acquisition.runtime.supervisor.build_runtime_registry", fake_registry)
    supervisor = RuntimeSupervisor(settings)
    try:
        supervisor.build_start_command("trispector_ftp")
        assert False, "expected validation failure"
    except ValueError:
        pass
    events = supervisor.events("trispector_ftp", limit=20)["events"]
    assert any(item["event_type"] == "PROCESS_VALIDATION_FAILED" for item in events)


def test_runtime_supervisor_applies_user_password_auth_mode(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    upload_dir = tmp_path / "incoming"
    upload_dir.mkdir(parents=True)

    def fake_registry(_settings: ApiSettings) -> dict[str, RuntimeProcessDefinition]:
        return {
            "trispector_ftp": RuntimeProcessDefinition(
                process_id="trispector_ftp",
                process_type="trispector_ftp_runtime",
                command=[sys.executable, "-m", "worker", "--ftp-command", "python -m pyftpdlib -p 2121", "--upload-dir", str(upload_dir)],
                enabled=True,
                source_id="trispector_ftp_0",
                config={
                    "command": "python -m pyftpdlib -p 2121",
                    "upload_dir": str(upload_dir),
                    "auth_mode": "user_password",
                    "username": "trispector",
                    "password": "secret",
                },
            )
        }

    monkeypatch.setattr("vision_3d_acquisition.runtime.supervisor.build_runtime_registry", fake_registry)
    monkeypatch.setattr("vision_3d_acquisition.runtime.supervisor.sp.check_call", lambda *args, **kwargs: 0)
    monkeypatch.setattr(RuntimeSupervisor, "_port_available", lambda _self, _port: True)
    supervisor = RuntimeSupervisor(settings)
    command = supervisor.build_start_command("trispector_ftp")
    ftp_command = command[command.index("--ftp-command") + 1]
    assert "-u" in ftp_command
    assert "-P" in ftp_command
    events = supervisor.events("trispector_ftp", limit=20)["events"]
    assert any(item["event_type"] == "FTP_AUTH_USER_PASSWORD" for item in events)
