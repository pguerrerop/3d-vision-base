from __future__ import annotations

import atexit
import ftplib
import io
import json
import os
import signal
import socket
import subprocess as sp
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from vision_3d_acquisition.api.settings import ApiSettings, get_settings
from vision_3d_acquisition.processes.bindings import ProcessBindingService
from vision_3d_acquisition.runtime.commanding import normalize_ftp_command, resolve_python_executable
from vision_3d_acquisition.runtime.logging import (
    RuntimeLogStore,
    parse_event_json_line,
    parse_event_line,
    read_last_lines,
    write_line,
)
from vision_3d_acquisition.runtime.models import RuntimeProcessDefinition, RuntimeProcessEvent, RuntimeProcessInstance, utc_now_iso
from vision_3d_acquisition.runtime.registry import build_runtime_registry

PID_RECORD_OWNER = "sensor_studio_runtime_supervisor_v1"
STARTUP_GRACE_SECONDS = 0.8
SHUTDOWN_TIMEOUT_SECONDS = 6.0
FORCE_KILL_WAIT_SECONDS = 3.0
MAX_AUTO_RESTARTS = 3


class RuntimeSupervisor:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._definitions = build_runtime_registry(settings)
        self._instances: dict[str, RuntimeProcessInstance] = {
            process_id: RuntimeProcessInstance(
                process_id=definition.process_id,
                process_type=definition.process_type,
                pid=None,
                desired_state="stopped",
                observed_state="stopped",
                status_label="stopped",
                started_at=None,
                stopped_at=None,
                restart_count=0,
                config=dict(definition.config),
                source_id=definition.source_id,
                health="unknown",
                last_heartbeat=None,
                last_event_summary=None,
            )
            for process_id, definition in self._definitions.items()
        }
        self._stores: dict[str, RuntimeLogStore] = {process_id: RuntimeLogStore() for process_id in self._definitions.keys()}
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._waiters: dict[str, threading.Thread] = {}
        self._recent_events: deque[RuntimeProcessEvent] = deque(maxlen=1200)
        self._base_dir = self.settings.data_dir / "runtime" / "processes"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_history()
        self._recover_known_pids()

    def refresh_registry(self) -> None:
        with self._lock:
            next_definitions = build_runtime_registry(self.settings)
            self._definitions = next_definitions
            for process_id, definition in next_definitions.items():
                if process_id not in self._instances:
                    self._instances[process_id] = RuntimeProcessInstance(
                        process_id=definition.process_id,
                        process_type=definition.process_type,
                        pid=None,
                        desired_state="stopped",
                        observed_state="stopped",
                        status_label="stopped",
                        started_at=None,
                        stopped_at=None,
                        restart_count=0,
                        config=dict(definition.config),
                        source_id=definition.source_id,
                        health="unknown",
                        last_heartbeat=None,
                        last_event_summary=None,
                    )
                    self._stores[process_id] = RuntimeLogStore()
                else:
                    instance = self._instances[process_id]
                    instance.process_type = definition.process_type
                    instance.config = dict(definition.config)
                    instance.source_id = definition.source_id
                    instance.status_label = self._status_label(instance)

    def list_processes(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._instances[key].to_dict() for key in sorted(self._instances.keys())]

    def status(self, process_id: str) -> dict[str, Any]:
        with self._lock:
            return self._require_instance(process_id).to_dict()

    def build_start_command(self, process_id: str) -> list[str]:
        with self._lock:
            self.refresh_registry()
            definition = self._require_definition(process_id)
            if not definition.enabled:
                raise ValueError(f"Runtime process '{process_id}' is disabled in runtime config.")
            prepared = self._prepare_definition_for_start(process_id, definition)
            self._validate_process_definition(process_id, prepared)
            return list(prepared.command)

    def start(self, process_id: str) -> dict[str, Any]:
        with self._lock:
            self.refresh_registry()
            definition = self._require_definition(process_id)
            instance = self._require_instance(process_id)
            if instance.observed_state in {"starting", "running"} and process_id in self._procs:
                return instance.to_dict()
            if not definition.enabled:
                raise ValueError(f"Runtime process '{process_id}' is disabled in runtime config.")

            definition = self._prepare_definition_for_start(process_id, definition)
            self._validate_process_definition(process_id, definition)
            instance.desired_state = "running"
            instance.observed_state = "starting"
            instance.status_label = self._status_label(instance)
            instance.stopped_at = None
            instance.last_event_summary = "Launching runtime process."
            proc = subprocess.Popen(
                definition.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._procs[process_id] = proc
            instance.pid = proc.pid
            instance.observed_state = "running"
            instance.started_at = utc_now_iso()
            instance.health = "starting"
            instance.status_label = self._status_label(instance)
            self._write_pid_file(process_id, proc.pid, definition)
            self._write_process_metadata(process_id, instance, definition)
            self._append_event(
                RuntimeProcessEvent(
                    timestamp=utc_now_iso(),
                    process_id=process_id,
                    severity="info",
                    event_type="PROCESS_STARTED",
                    message=f"Process {process_id} launched with pid {proc.pid}.",
                    metadata={"pid": proc.pid, "command": definition.command},
                )
            )
            self._start_stream_threads(process_id, proc)
            self._start_waiter_thread(process_id, proc)
        self._validate_startup(process_id)
        with self._lock:
            return self._require_instance(process_id).to_dict()

    def stop(self, process_id: str) -> dict[str, Any]:
        definition = None
        with self._lock:
            instance = self._require_instance(process_id)
            definition = self._require_definition(process_id)
            instance.desired_state = "stopped"
            proc = self._procs.get(process_id)
            if proc is None or proc.poll() is not None:
                pid_record = self._read_pid_file(process_id)
                metadata = self._read_process_metadata(process_id)
                if not self._pid_record_matches_process(pid_record, process_id, metadata):
                    self._delete_pid_file(process_id)
                    self._mark_stopped(process_id)
                    return instance.to_dict()
                pid = int(pid_record.get("pid")) if isinstance(pid_record.get("pid"), int) else None
                if pid is not None and self._pid_running(pid):
                    if self._pid_owned_by_definition(pid, definition, pid_record, metadata):
                        self._graceful_stop_pid(pid)
                    else:
                        instance.observed_state = "orphaned"
                        instance.status_label = self._status_label(instance)
                        self._append_event(
                            RuntimeProcessEvent(
                                timestamp=utc_now_iso(),
                                process_id=process_id,
                                severity="warning",
                                event_type="STOP_SKIPPED_OWNERSHIP_MISMATCH",
                                message=f"PID {pid} does not match expected process ownership; stop skipped.",
                                metadata={"pid": pid},
                            )
                        )
                        return instance.to_dict()
                instance.observed_state = "stopped"
                instance.status_label = self._status_label(instance)
                self._mark_stopped(process_id)
                return instance.to_dict()

            instance.observed_state = "stopping"
            instance.status_label = self._status_label(instance)
            proc.terminate()
        try:
            proc.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._append_event(
                RuntimeProcessEvent(
                    timestamp=utc_now_iso(),
                    process_id=process_id,
                    severity="warning",
                    event_type="PROCESS_SHUTDOWN_TIMEOUT",
                    message=f"Graceful shutdown timed out after {SHUTDOWN_TIMEOUT_SECONDS:.1f}s; force killing.",
                    metadata={},
                )
            )
            proc.kill()
            proc.wait(timeout=FORCE_KILL_WAIT_SECONDS)

        with self._lock:
            self._mark_stopped(process_id)
            self._append_event(
                RuntimeProcessEvent(
                    timestamp=utc_now_iso(),
                    process_id=process_id,
                    severity="info",
                    event_type="PROCESS_STOPPED",
                    message=f"Process {process_id} stopped.",
                    metadata={},
                )
            )
            return self._require_instance(process_id).to_dict()

    def restart(self, process_id: str) -> dict[str, Any]:
        self.stop(process_id)
        with self._lock:
            self._require_instance(process_id).restart_count += 1
        return self.start(process_id)

    def tail_logs(self, process_id: str, limit: int = 200) -> dict[str, Any]:
        with self._lock:
            self._require_instance(process_id)
            logs = self._stores[process_id].tail_logs(limit)
            return {"process_id": process_id, "lines": logs}

    def events(self, process_id: str, limit: int = 200) -> dict[str, Any]:
        with self._lock:
            self._require_instance(process_id)
            events = self._stores[process_id].tail_events(limit)
            return {"process_id": process_id, "events": events}

    def shutdown(self) -> None:
        for process_id in list(self._definitions.keys()):
            try:
                self.stop(process_id)
            except Exception:  # noqa: BLE001
                continue

    def ftp_selftest(self, process_id: str) -> dict[str, Any]:
        with self._lock:
            definition = self._require_definition(process_id)
            config = dict(definition.config)
            command = str(config.get("command") or "")
            auth_mode = str(config.get("auth_mode") or "anonymous")
            username = str(config.get("username") or "") or None
            password = str(config.get("password") or "") or None
            upload_dir = Path(str(config.get("upload_dir") or "")).expanduser()
        port = self._extract_ftp_port(command) or 2121
        upload_dir.mkdir(parents=True, exist_ok=True)
        test_name = f"sensor_studio_selftest_{int(time.time())}.txt"
        payload = f"sensor_studio selftest {utc_now_iso()}\n".encode("utf-8")
        ftp = ftplib.FTP()
        diagnostics: list[str] = []
        try:
            ftp.connect(host="127.0.0.1", port=port, timeout=5.0)
            diagnostics.append("connected")
            if auth_mode == "user_password":
                ftp.login(user=username or "", passwd=password or "")
            else:
                ftp.login()
            diagnostics.append("authenticated")
            ftp.storbinary(f"STOR {test_name}", io.BytesIO(payload))
            diagnostics.append("uploaded")
            uploaded_path = upload_dir / test_name
            exists = uploaded_path.is_file()
            if exists:
                uploaded_path.unlink(missing_ok=True)
            diagnostics.append("verified" if exists else "missing_after_upload")
            ftp.quit()
            return {
                "ok": bool(exists),
                "process_id": process_id,
                "host": "127.0.0.1",
                "port": port,
                "auth_mode": auth_mode,
                "upload_dir": str(upload_dir),
                "diagnostics": diagnostics,
                "test_file": test_name,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "process_id": process_id,
                "host": "127.0.0.1",
                "port": port,
                "auth_mode": auth_mode,
                "upload_dir": str(upload_dir),
                "diagnostics": diagnostics,
                "error": str(exc),
                "test_file": test_name,
            }

    def health_checklist(self, process_id: str) -> dict[str, Any]:
        with self._lock:
            definition = self._require_definition(process_id)
            instance = self._require_instance(process_id)
            config = dict(definition.config)
            events = self._stores.get(process_id, RuntimeLogStore()).events

        upload_dir = Path(str(config.get("upload_dir") or "")).expanduser()
        upload_exists = upload_dir.is_dir()
        upload_writable = upload_exists and os.access(upload_dir, os.W_OK)
        ftp_command = str(config.get("command") or "").strip()
        source_id = str(config.get("source_id") or instance.source_id or "").strip()
        port_available = self._port_available(self._extract_ftp_port(ftp_command))

        binding_found = False
        binding_warning = None
        if source_id:
            try:
                binding = ProcessBindingService(self.settings).resolve_process_binding(source_id, "heightmap", "acquisition_inspection")
                binding_found = binding is not None
                if not binding_found:
                    binding_warning = "No active ProcessBinding for source/modality/purpose."
            except Exception as exc:  # noqa: BLE001
                binding_warning = str(exc)
        else:
            binding_warning = "Missing source_id."

        last_success = None
        for event in reversed(list(events)):
            if event.event_type == "TAKE_REGISTERED":
                last_success = event.timestamp
                break

        return {
            "process_id": process_id,
            "items": [
                {"id": "upload_dir_exists_writable", "ok": bool(upload_exists and upload_writable), "label": "upload dir exists/writable", "details": str(upload_dir)},
                {"id": "ftp_command_configured", "ok": bool(ftp_command), "label": "FTP command configured", "details": ftp_command or "missing"},
                {"id": "port_available", "ok": bool(port_available), "label": "port available", "details": str(self._extract_ftp_port(ftp_command) or "unknown")},
                {"id": "source_id_configured", "ok": bool(source_id), "label": "source_id configured", "details": source_id or "missing"},
                {
                    "id": "binding_found_or_warning",
                    "ok": bool(binding_found),
                    "label": "binding found or warning",
                    "details": "binding_found" if binding_found else str(binding_warning or "binding missing"),
                },
                {"id": "last_successful_upload_time", "ok": bool(last_success), "label": "last successful upload time", "details": last_success or "none"},
            ],
        }

    def ftp_status(self, process_id: str) -> dict[str, Any]:
        with self._lock:
            definition = self._require_definition(process_id)
            instance = self._require_instance(process_id)
            config = dict(definition.config)
            events = list(self._stores.get(process_id, RuntimeLogStore()).events)
        command = str(config.get("command") or "")
        port = self._extract_ftp_port(command) or 2121
        upload_dir = Path(str(config.get("upload_dir") or "")).expanduser()
        listening = bool(instance.observed_state == "running" and not self._port_available(port))
        recent_uploads: list[dict[str, Any]] = []
        active_clients: dict[str, dict[str, Any]] = {}
        last_upload_timestamp: str | None = None
        for event in reversed(events):
            if event.event_type == "FTP_UPLOAD_COMPLETED":
                if last_upload_timestamp is None:
                    last_upload_timestamp = event.timestamp
                recent_uploads.append(
                    {
                        "timestamp": event.timestamp,
                        "filename": event.metadata.get("filename"),
                        "size": event.metadata.get("upload_size"),
                        "client_ip": event.metadata.get("client_ip"),
                        "username": event.metadata.get("username"),
                    }
                )
                if len(recent_uploads) >= 20:
                    break
            if event.event_type in {"FTP_CLIENT_CONNECTED", "FTP_CLIENT_LOGIN_SUCCESS"}:
                ip = str(event.metadata.get("client_ip") or "")
                if ip and ip not in active_clients:
                    active_clients[ip] = {"client_ip": ip, "username": event.metadata.get("username"), "last_event": event.event_type}
        return {
            "process_id": process_id,
            "listening": listening,
            "host": "0.0.0.0",
            "port": port,
            "auth_mode": str(config.get("auth_mode") or "anonymous"),
            "upload_dir": str(upload_dir),
            "writable": bool(upload_dir.is_dir() and os.access(upload_dir, os.W_OK)),
            "active_clients": list(active_clients.values()),
            "recent_uploads": list(reversed(recent_uploads)),
            "last_upload_timestamp": last_upload_timestamp,
        }

    def _start_stream_threads(self, process_id: str, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is not None:
            thread = threading.Thread(
                target=self._stream_reader,
                args=(process_id, "stdout", proc.stdout),
                daemon=True,
            )
            thread.start()
        if proc.stderr is not None:
            thread = threading.Thread(
                target=self._stream_reader,
                args=(process_id, "stderr", proc.stderr),
                daemon=True,
            )
            thread.start()

    def _start_waiter_thread(self, process_id: str, proc: subprocess.Popen[str]) -> None:
        thread = threading.Thread(target=self._wait_for_exit, args=(process_id, proc), daemon=True)
        self._waiters[process_id] = thread
        thread.start()

    def _stream_reader(self, process_id: str, stream_name: str, stream: Any) -> None:
        for line in iter(stream.readline, ""):
            stripped = line.rstrip("\n")
            event = parse_event_line(stripped)
            if event is not None and event.process_id == process_id:
                with self._lock:
                    self._append_event(event)
                continue
            with self._lock:
                store = self._stores.get(process_id)
                if store is not None:
                    payload = f"[{stream_name}] {stripped}"
                    store.append_log(payload)
                    write_line(self._log_path(process_id), payload)
        stream.close()

    def _wait_for_exit(self, process_id: str, proc: subprocess.Popen[str]) -> None:
        return_code = proc.wait()
        should_restart = False
        with self._lock:
            instance = self._instances.get(process_id)
            if instance is None:
                return
            instance.pid = None
            instance.stopped_at = utc_now_iso()
            if instance.observed_state != "stopped":
                instance.observed_state = "crashed" if return_code != 0 else "stopped"
            instance.health = "error" if return_code != 0 else "stopped"
            instance.status_label = self._status_label(instance)
            self._procs.pop(process_id, None)
            self._delete_pid_file(process_id)
            self._append_event(
                RuntimeProcessEvent(
                    timestamp=utc_now_iso(),
                    process_id=process_id,
                    severity="error" if return_code != 0 else "info",
                    event_type="PROCESS_EXITED",
                    message=f"Process exited with code {return_code}.",
                    metadata={"exit_code": return_code},
                )
            )
            should_restart = (
                return_code != 0
                and instance.desired_state == "running"
                and instance.restart_count < MAX_AUTO_RESTARTS
            )
            if should_restart:
                instance.restart_count += 1
                instance.observed_state = "starting"
                instance.status_label = self._status_label(instance)
                self._append_event(
                    RuntimeProcessEvent(
                        timestamp=utc_now_iso(),
                        process_id=process_id,
                        severity="warning",
                        event_type="PROCESS_RESTART_SCHEDULED",
                        message=f"Process crashed; scheduling restart attempt {instance.restart_count}/{MAX_AUTO_RESTARTS}.",
                        metadata={"exit_code": return_code, "restart_count": instance.restart_count},
                    )
                )
        if should_restart:
            try:
                self.start(process_id)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    instance = self._instances.get(process_id)
                    if instance is not None:
                        instance.observed_state = "failed_startup"
                        instance.health = "error"
                        instance.status_label = self._status_label(instance)
                        self._append_event(
                            RuntimeProcessEvent(
                                timestamp=utc_now_iso(),
                                process_id=process_id,
                                severity="error",
                                event_type="PROCESS_RESTART_FAILED",
                                message=f"Restart attempt failed: {exc}",
                                metadata={},
                            )
                        )

    def _append_event(self, event: RuntimeProcessEvent) -> None:
        store = self._stores.setdefault(event.process_id, RuntimeLogStore())
        store.append_event(event)
        store.append_log(f"[event:{event.severity}] {event.event_type} {event.message}")
        write_line(self._event_path(event.process_id), json.dumps(event.to_dict(), ensure_ascii=True))
        write_line(self._log_path(event.process_id), f"[event:{event.severity}] {event.event_type} {event.message}")
        self._recent_events.append(event)
        instance = self._instances.get(event.process_id)
        if instance is None:
            return
        instance.last_event_summary = event.message
        if event.event_type == "HEARTBEAT":
            instance.last_heartbeat = event.timestamp
            instance.health = "healthy"
        elif event.severity == "error":
            instance.health = "error"
        elif instance.health in {"unknown", "starting"}:
            instance.health = "healthy"
        instance.status_label = self._status_label(instance)

    def _load_persisted_history(self) -> None:
        for process_id in self._definitions.keys():
            store = self._stores.setdefault(process_id, RuntimeLogStore())
            for line in read_last_lines(self._log_path(process_id), store.max_lines):
                store.append_log(line)
            for line in read_last_lines(self._event_path(process_id), store.max_events):
                event = parse_event_json_line(line)
                if event is not None:
                    store.append_event(event)
                    self._recent_events.append(event)
            if store.events:
                latest = list(store.events)[-1]
                instance = self._instances.get(process_id)
                if instance is not None:
                    instance.last_event_summary = latest.message
                    if latest.event_type == "HEARTBEAT":
                        instance.last_heartbeat = latest.timestamp

    def _validate_startup(self, process_id: str) -> None:
        deadline = time.monotonic() + STARTUP_GRACE_SECONDS
        proc = None
        with self._lock:
            proc = self._procs.get(process_id)
        if proc is None:
            return
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        return_code = proc.poll()
        if return_code is None:
            return
        with self._lock:
            instance = self._require_instance(process_id)
            instance.desired_state = "stopped"
            instance.observed_state = "failed_startup"
            instance.health = "error"
            instance.status_label = self._status_label(instance)
            self._procs.pop(process_id, None)
            self._delete_pid_file(process_id)
            self._append_event(
                RuntimeProcessEvent(
                    timestamp=utc_now_iso(),
                    process_id=process_id,
                    severity="error",
                    event_type="PROCESS_START_FAILED",
                    message=f"Process exited during startup with code {return_code}.",
                    metadata={"exit_code": return_code},
                )
            )
        raise RuntimeError(f"Runtime process '{process_id}' failed to start (exit code {return_code}).")

    def _recover_known_pids(self) -> None:
        for process_id, instance in self._instances.items():
            definition = self._definitions.get(process_id)
            if definition is None:
                continue
            record = self._read_pid_file(process_id)
            metadata = self._read_process_metadata(process_id)
            if not self._pid_record_matches_process(record, process_id, metadata):
                self._delete_pid_file(process_id)
                continue
            pid = record.get("pid") if isinstance(record.get("pid"), int) else None
            if pid is None:
                continue
            if self._pid_running(pid) and self._pid_owned_by_definition(pid, definition, record, metadata):
                instance.pid = pid
                instance.desired_state = "running" if str((metadata or {}).get("desired_state") or "running") == "running" else "stopped"
                instance.observed_state = "running"
                instance.health = "healthy"
                instance.started_at = str((metadata or {}).get("started_at") or instance.started_at or utc_now_iso())
                instance.status_label = self._status_label(instance)
                self._append_event(
                    RuntimeProcessEvent(
                        timestamp=utc_now_iso(),
                        process_id=process_id,
                        severity="info",
                        event_type="PROCESS_RECOVERED",
                        message=f"Recovered running process from pid metadata (pid {pid}).",
                        metadata={"pid": pid},
                    )
                )
            elif self._pid_running(pid):
                instance.pid = pid
                instance.desired_state = "stopped"
                instance.observed_state = "orphaned"
                instance.health = "warning"
                instance.status_label = self._status_label(instance)
                self._append_event(
                    RuntimeProcessEvent(
                        timestamp=utc_now_iso(),
                        process_id=process_id,
                        severity="warning",
                        event_type="STALE_PID_OWNERSHIP_MISMATCH",
                        message=f"Recovered PID {pid} is running but ownership check failed.",
                        metadata={"pid": pid},
                    )
                )
            else:
                self._delete_pid_file(process_id)
                instance.pid = None
                instance.observed_state = "stopped"
                instance.status_label = self._status_label(instance)

    def _write_pid_file(self, process_id: str, pid: int, definition: RuntimeProcessDefinition) -> None:
        path = self._pid_path(process_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "owner": PID_RECORD_OWNER,
            "pid": pid,
            "process_id": process_id,
            "process_type": definition.process_type,
            "command": definition.command,
            "command_signature": self._command_signature(definition.command),
            "cwd": str(Path.cwd()),
            "source_id": definition.source_id,
            "written_at": utc_now_iso(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _read_pid_file(self, process_id: str) -> dict[str, Any]:
        path = self._pid_path(process_id)
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def _delete_pid_file(self, process_id: str) -> None:
        path = self._pid_path(process_id)
        if path.is_file():
            path.unlink(missing_ok=True)

    def _pid_path(self, process_id: str) -> Path:
        return self._base_dir / process_id / "pid"

    def _pid_running(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _pid_record_matches_process(self, record: dict[str, Any], process_id: str, metadata: dict[str, Any] | None = None) -> bool:
        signature = str(record.get("command_signature") or "")
        metadata_signature = str((metadata or {}).get("command_signature") or "")
        return (
            str(record.get("owner") or "") == PID_RECORD_OWNER
            and str(record.get("process_id") or "") == process_id
            and isinstance(record.get("pid"), int)
            and (not signature or not metadata_signature or signature == metadata_signature)
        )

    def _pid_owned_by_definition(
        self,
        pid: int,
        definition: RuntimeProcessDefinition,
        pid_record: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        try:
            output = sp.check_output(["ps", "-p", str(pid), "-o", "command="], text=True).strip()
        except Exception:  # noqa: BLE001
            return False
        if not output:
            return False
        required_tokens = [token for token in definition.command if token and len(token) > 2][:8]
        record_signature = str((pid_record or {}).get("command_signature") or "")
        metadata_signature = str((metadata or {}).get("command_signature") or "")
        cwd_matches = True
        expected_cwd = str((metadata or {}).get("cwd") or "")
        if expected_cwd:
            cwd_matches = self._pid_cwd_matches(pid, expected_cwd)
        if record_signature and record_signature in output and cwd_matches:
            return True
        if metadata_signature and metadata_signature in output and cwd_matches:
            return True
        if not required_tokens:
            return True
        hits = 0
        for token in required_tokens:
            basename = os.path.basename(token)
            if token in output or basename in output:
                hits += 1
        min_hits = 1 if len(required_tokens) <= 3 else min(2, len(required_tokens))
        return cwd_matches and hits >= min_hits

    def _pid_cwd_matches(self, pid: int, expected_cwd: str) -> bool:
        try:
            output = sp.check_output(["ps", "-p", str(pid), "-o", "cwd="], text=True).strip()
        except Exception:  # noqa: BLE001
            return True
        if not output:
            return True
        if output in {"-", "?"}:
            return True
        return output == expected_cwd

    def _graceful_stop_pid(self, pid: int) -> None:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not self._pid_running(pid):
                return
            time.sleep(0.1)
        if self._pid_running(pid):
            os.kill(pid, signal.SIGKILL)

    def _mark_stopped(self, process_id: str) -> None:
        instance = self._require_instance(process_id)
        instance.observed_state = "stopped"
        instance.status_label = self._status_label(instance)
        instance.stopped_at = utc_now_iso()
        instance.pid = None
        instance.health = "stopped"
        self._delete_pid_file(process_id)
        self._write_process_metadata(process_id, instance, self._require_definition(process_id))
        self._procs.pop(process_id, None)

    def _status_label(self, instance: RuntimeProcessInstance) -> str:
        if instance.desired_state == "running" and instance.observed_state == "running":
            return "running"
        if instance.desired_state == "running" and instance.observed_state == "starting":
            return "starting"
        if instance.desired_state == "running" and instance.observed_state == "crashed":
            return "crashed_restart_pending"
        if instance.desired_state == "running" and instance.observed_state == "failed_startup":
            return "failed_startup"
        if instance.desired_state == "stopped" and instance.observed_state == "failed_startup":
            return "failed_startup"
        if instance.desired_state == "stopped" and instance.observed_state == "stopping":
            return "stopping"
        if instance.desired_state == "stopped" and instance.observed_state == "orphaned":
            return "orphaned"
        if instance.desired_state == "stopped" and instance.observed_state == "stopped":
            return "stopped"
        return f"{instance.desired_state}_{instance.observed_state}"

    def _extract_ftp_port(self, command: str) -> int | None:
        tokens = command.split()
        if "-p" in tokens:
            idx = tokens.index("-p")
            if idx + 1 < len(tokens):
                try:
                    return int(tokens[idx + 1])
                except ValueError:
                    return None
        return None

    def _port_available(self, port: int | None) -> bool | None:
        if port is None:
            return None
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
        finally:
            sock.close()
        return True

    def _require_definition(self, process_id: str):
        definition = self._definitions.get(process_id)
        if definition is None:
            raise KeyError(f"Unknown runtime process: {process_id}")
        return definition

    def _require_instance(self, process_id: str) -> RuntimeProcessInstance:
        instance = self._instances.get(process_id)
        if instance is None:
            raise KeyError(f"Unknown runtime process: {process_id}")
        return instance

    def _log_path(self, process_id: str) -> Path:
        return self._base_dir / process_id / "runtime.log"

    def _event_path(self, process_id: str) -> Path:
        return self._base_dir / process_id / "events.log"

    def _metadata_path(self, process_id: str) -> Path:
        return self._base_dir / process_id / "instance.json"

    def _command_signature(self, command: list[str]) -> str:
        parts = [part for part in command if part][:8]
        return " ".join(parts)

    def _write_process_metadata(self, process_id: str, instance: RuntimeProcessInstance, definition: RuntimeProcessDefinition) -> None:
        path = self._metadata_path(process_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "process_id": process_id,
            "process_type": definition.process_type,
            "desired_state": instance.desired_state,
            "started_at": instance.started_at,
            "stopped_at": instance.stopped_at,
            "cwd": str(Path.cwd()),
            "command_signature": self._command_signature(definition.command),
            "written_at": utc_now_iso(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _read_process_metadata(self, process_id: str) -> dict[str, Any]:
        path = self._metadata_path(process_id)
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        return payload if isinstance(payload, dict) else {}

    def _prepare_definition_for_start(self, process_id: str, definition: RuntimeProcessDefinition) -> RuntimeProcessDefinition:
        if definition.process_type != "trispector_ftp_runtime":
            return definition
        config = dict(definition.config)
        upload_dir = Path(str(config.get("upload_dir") or "")).expanduser()
        raw_ftp_command = str(config.get("command") or "").strip()
        python_resolution = resolve_python_executable()
        if python_resolution.warning:
            self._append_event(
                RuntimeProcessEvent(
                    timestamp=utc_now_iso(),
                    process_id=process_id,
                    severity="warning",
                    event_type="PYTHON_EXECUTABLE_FALLBACK",
                    message=python_resolution.warning,
                    metadata={"resolved_executable": python_resolution.executable},
                )
            )
        normalized = normalize_ftp_command(
            raw_ftp_command,
            upload_dir,
            python_executable=python_resolution.executable,
            auth_mode=str(config.get("auth_mode") or "anonymous"),
            username=(str(config.get("username") or "") or None),
            password=(str(config.get("password") or "") or None),
        )
        if normalized.normalized_python:
            self._append_event(
                RuntimeProcessEvent(
                    timestamp=utc_now_iso(),
                    process_id=process_id,
                    severity="info",
                    event_type="FTP_COMMAND_NORMALIZED",
                    message="Normalized FTP command interpreter to resolved Python executable.",
                    metadata={"command": normalized.command},
                )
            )
        if normalized.injected_root:
            self._append_event(
                RuntimeProcessEvent(
                    timestamp=utc_now_iso(),
                    process_id=process_id,
                    severity="info",
                    event_type="FTP_ROOT_INJECTED",
                    message="Injected pyftpdlib upload root (-d) into FTP command.",
                    metadata={"upload_dir": str(upload_dir), "command": normalized.command},
                )
            )
        if normalized.injected_write:
            self._append_event(
                RuntimeProcessEvent(
                    timestamp=utc_now_iso(),
                    process_id=process_id,
                    severity="info",
                    event_type="FTP_WRITE_ENABLED",
                    message="Injected pyftpdlib write access flag (-w).",
                    metadata={"command": normalized.command},
                )
            )
        if normalized.auth_mode == "user_password":
            self._append_event(
                RuntimeProcessEvent(
                    timestamp=utc_now_iso(),
                    process_id=process_id,
                    severity="info",
                    event_type="FTP_AUTH_USER_PASSWORD",
                    message="FTP auth mode set to user/password.",
                    metadata={"username": str(config.get("username") or "")},
                )
            )
        else:
            self._append_event(
                RuntimeProcessEvent(
                    timestamp=utc_now_iso(),
                    process_id=process_id,
                    severity="info",
                    event_type="FTP_AUTH_ANONYMOUS",
                    message="FTP auth mode set to anonymous.",
                    metadata={},
                )
            )
        if normalized.root_mismatch:
            self._append_event(
                RuntimeProcessEvent(
                    timestamp=utc_now_iso(),
                    process_id=process_id,
                    severity="warning",
                    event_type="FTP_ROOT_MISMATCH",
                    message="Configured pyftpdlib root differs from runtime upload_dir.",
                    metadata={"upload_dir": str(upload_dir), "command": normalized.command},
                )
            )
        next_command = list(definition.command)
        if "--ftp-command" in next_command:
            idx = next_command.index("--ftp-command")
            if idx + 1 < len(next_command):
                next_command[idx + 1] = normalized.command
        config["command"] = normalized.command
        config["auth_mode"] = normalized.auth_mode
        return RuntimeProcessDefinition(
            process_id=definition.process_id,
            process_type=definition.process_type,
            command=next_command,
            source_id=definition.source_id,
            enabled=definition.enabled,
            config=config,
        )

    def _validate_process_definition(self, process_id: str, definition: RuntimeProcessDefinition) -> None:
        executable = definition.command[0] if definition.command else ""
        if not executable:
            self._emit_validation_failure(process_id, "missing executable")
        if not Path(executable).exists() and executable != sys.executable:
            self._emit_validation_failure(process_id, f"command executable not found: {executable}")
        if definition.process_type != "trispector_ftp_runtime":
            return
        config = dict(definition.config)
        upload_dir = Path(str(config.get("upload_dir") or "")).expanduser()
        if not upload_dir.is_dir():
            self._emit_validation_failure(process_id, f"upload_dir does not exist: {upload_dir}")
        if not os.access(upload_dir, os.W_OK):
            self._emit_validation_failure(process_id, f"upload_dir is not writable: {upload_dir}")
        command = str(config.get("command") or "")
        if not command.strip():
            self._emit_validation_failure(process_id, "ftp command is empty")
        if "-m pyftpdlib" in command:
            try:
                sp.check_call([definition.command[0], "-c", "import pyftpdlib"], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
            except Exception:
                self._emit_validation_failure(process_id, "pyftpdlib is not importable in resolved interpreter")
        port = self._extract_ftp_port(command)
        port_available = self._port_available(port)
        if port_available is False:
            self._emit_validation_failure(process_id, f"ftp port is unavailable: {port}")

    def _emit_validation_failure(self, process_id: str, reason: str) -> None:
        self._append_event(
            RuntimeProcessEvent(
                timestamp=utc_now_iso(),
                process_id=process_id,
                severity="error",
                event_type="PROCESS_VALIDATION_FAILED",
                message=reason,
                metadata={},
            )
        )
        raise ValueError(f"Runtime process validation failed for '{process_id}': {reason}")


_SUPERVISOR: RuntimeSupervisor | None = None
_SUPERVISOR_LOCK = threading.Lock()
_SUPERVISOR_ATEXIT_REGISTERED = False


def get_runtime_supervisor(*, register_atexit: bool = True) -> RuntimeSupervisor:
    global _SUPERVISOR
    global _SUPERVISOR_ATEXIT_REGISTERED
    with _SUPERVISOR_LOCK:
        if _SUPERVISOR is None:
            _SUPERVISOR = RuntimeSupervisor(get_settings())
        if register_atexit and not _SUPERVISOR_ATEXIT_REGISTERED:
            atexit.register(_SUPERVISOR.shutdown)
            _SUPERVISOR_ATEXIT_REGISTERED = True
        return _SUPERVISOR
