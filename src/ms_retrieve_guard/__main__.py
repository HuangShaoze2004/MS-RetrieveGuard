"""Small explicit commands; sealed-role model evaluation has no CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import yaml

from .io import sha256_file, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "audit", "build-smoke", "smoke", "select-smoke"])
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("scope") != "bootstrap_100_queries":
        raise ValueError("Only the approved bootstrap scope is implemented")
    data_config = config["data"]
    run_dir = Path(args.run_dir or f"artifacts/runs/{args.command}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / f"{args.command}_process.json"
    if status_path.exists():
        raise ValueError(f"Existing run receipt: {status_path}; use a new run directory")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    # A dirty tree is recorded by content, including new source files, not just git diff.
    source_paths = sorted(Path("src").rglob("*.py")) + sorted(Path("configs").rglob("*.yaml"))
    snapshot = {str(p): sha256_file(p) for p in source_paths}
    receipt = {"status": "running", "command": args.command, "pid": os.getpid(), "started_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               "started_unix": time.time(), "git_commit": git, "git_status": dirty, "source_sha256": snapshot,
               "config": config, "config_sha256": sha256_file(config_path), "python": sys.executable,
               "sealed_roles": ["calibration", "test"]}
    write_json(status_path, receipt)
    write_json(run_dir / "config.json", config)
    started = time.monotonic()
    try:
        if args.command == "download":
            from .io import download_data
            result = download_data("artifacts/raw", "manifests/downloads.json")
        elif args.command == "audit":
            from .data import audit_data
            result = audit_data(data_config["tsv"], data_config["candidates"], data_config["output_dir"], data_config)
        elif args.command == "build-smoke":
            from .data import build_smoke
            result = build_smoke(data_config["output_dir"], data_config["smoke_path"], data_config)
        elif args.command == "smoke":
            from .rankers import run_smoke
            from .runtime import gpu_slot
            with gpu_slot(run_dir, config["limits"]) as remaining:
                ranker_config = dict(config["ranker"])
                ranker_config["max_gpu_seconds"] = min(ranker_config["max_gpu_seconds"], remaining)
                result = run_smoke(data_config["smoke_path"], run_dir, ranker_config)
        else:
            from .selection import run_selection_smoke
            result = run_selection_smoke(data_config["smoke_path"], run_dir / "scores.npz", run_dir, config["selection"])
        receipt.update(status="budget_stopped" if result.get("budget_stopped") else "complete", elapsed_seconds=time.monotonic() - started, finished_unix=time.time())
        write_json(run_dir / f"{args.command}_result.json", result)
        write_json(status_path, receipt)
        print(json.dumps({"status": receipt["status"], "command": args.command, "run_dir": str(run_dir), "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)
    except BaseException as exc:
        receipt.update(status="failed", elapsed_seconds=time.monotonic() - started, finished_unix=time.time(),
                       exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
        write_json(status_path, receipt)
        raise


if __name__ == "__main__":
    main()
