"""A single project GPU slot and conservative GPU wall-time accounting."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .io import write_json


@contextmanager
def gpu_slot(run_dir, limits):
    artifacts = Path("artifacts")
    with (artifacts / "gpu.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another project GPU task holds the GPU slot") from exc
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, check=True)
        foreign = [int(p.strip()) for p in result.stdout.splitlines() if p.strip().isdigit() and int(p.strip()) != os.getpid()]
        if foreign:
            raise RuntimeError(f"GPU already has compute processes {foreign}; no processes were changed")
        ledger_path = artifacts / "resource_ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"scope": "bootstrap_100_queries", "gpu_runs": []}
        consumed = sum(row["gpu_wall_seconds"] for row in ledger["gpu_runs"])
        remaining = limits["gpu_wall_seconds"] - consumed
        if remaining <= 0:
            raise RuntimeError("First implementation GPU budget exhausted")
        disk_bytes = int(subprocess.check_output(["du", "-sB1", "."], text=True).split()[0])
        if disk_bytes >= limits["project_disk_bytes"]:
            raise RuntimeError("Project disk budget exhausted")
        started = time.monotonic()
        row = {"run_dir": str(Path(run_dir).resolve()), "pid": os.getpid(), "started_unix": time.time(), "status": "running", "disk_bytes_before": disk_bytes}
        telemetry_path = Path(run_dir) / "gpu_telemetry.csv"
        telemetry_file = telemetry_path.open("w")
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw", "--format=csv", "--loop-ms=200"],
                                   stdout=telemetry_file, stderr=subprocess.DEVNULL, env={**os.environ, "TZ": "UTC"})
        row.update(telemetry_pid=monitor.pid, telemetry_path=str(telemetry_path), telemetry_period_ms=200)
        write_json(Path(run_dir) / "gpu_receipt.json", row)
        try:
            yield remaining
            row["status"] = "finished"
        except BaseException as exc:
            row.update(status="failed", error=str(exc))
            raise
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
            telemetry_file.close()
            row.update(gpu_wall_seconds=time.monotonic() - started, finished_unix=time.time())
            ledger["gpu_runs"].append(row)
            ledger["gpu_wall_seconds_total"] = sum(item["gpu_wall_seconds"] for item in ledger["gpu_runs"])
            ledger["gpu_wall_seconds_limit"] = limits["gpu_wall_seconds"]
            write_json(ledger_path, ledger)
            write_json(Path(run_dir) / "gpu_receipt.json", row)
            fcntl.flock(lock, fcntl.LOCK_UN)
