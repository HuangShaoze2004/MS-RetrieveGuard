"""Small, revision-pinned artifact IO with byte limits and content verification."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

REVISION = "d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a"
DATASET = "roman-bushuiev/MassSpecGym"
FILES = {
    "data/MassSpecGym1.5.tsv": 261197365,
    "data/molecules/MassSpecGym1.5_retrieval_candidates_formula.json": 370306399,
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def download_data(raw_dir: str | Path, manifest_path: str | Path, max_bytes: int = 700000000) -> dict:
    """Download only the two approved files; verify HF LFS SHA256 and exact size."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/data?recursive=true"
    req = urllib.request.Request(api, headers={"User-Agent": "MS-RetrieveGuard/0.1"})
    with urllib.request.urlopen(req, timeout=60) as response:
        entries = json.load(response)
    records = {item["path"]: item for item in entries}
    if sum(FILES.values()) > max_bytes:
        raise ValueError("Approved data exceeds download byte budget")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "dataset": DATASET, "revision": REVISION, "files": [], "network_data_bytes": 0, "attempts": []}
    if manifest.get("revision") != REVISION or manifest.get("dataset") != DATASET:
        raise ValueError("Existing download ledger belongs to a different dataset revision")
    manifest.setdefault("attempts", [])
    for remote_path, expected_size in FILES.items():
        entry = records[remote_path]
        lfs = entry.get("lfs", {})
        expected_sha = lfs.get("oid") or lfs.get("sha256")
        if expected_sha and expected_sha.startswith("sha256:"):
            expected_sha = expected_sha.split(":", 1)[1]
        if not expected_sha or len(expected_sha) != 64:
            raise ValueError(f"Missing upstream SHA256 for {remote_path}")
        if entry.get("size") != expected_size:
            raise ValueError(f"Unexpected revision-pinned size for {remote_path}")
        target = raw_dir / Path(remote_path).name
        started = time.monotonic()
        downloaded = 0
        if not (target.exists() and target.stat().st_size == expected_size and sha256_file(target) == expected_sha):
            if manifest["network_data_bytes"] + expected_size > max_bytes:
                raise ValueError("Cumulative data download budget would be exceeded")
            partial = target.with_suffix(target.suffix + ".partial")
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{remote_path}?download=true"
            request = urllib.request.Request(url, headers={"User-Agent": "MS-RetrieveGuard/0.1", "Cache-Control": "no-cache"})
            attempt = {"remote_path": remote_path, "started_unix": time.time(), "status": "running", "bytes": 0}
            manifest["attempts"].append(attempt)
            write_json(manifest_path, manifest)
            try:
                with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
                    if response.status != 200:
                        raise ValueError(f"Expected full HTTP 200 download, got {response.status}")
                    last_persisted = 0
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        downloaded += len(chunk)
                        manifest["network_data_bytes"] += len(chunk)
                        attempt["bytes"] = downloaded
                        if downloaded > expected_size or manifest["network_data_bytes"] > max_bytes:
                            raise ValueError("Download exceeded approved byte limit")
                        output.write(chunk)
                        if downloaded - last_persisted >= 8 * 1024 * 1024:
                            write_json(manifest_path, manifest)
                            last_persisted = downloaded
                attempt["status"] = "downloaded"
            except BaseException as exc:
                attempt.update(status="failed", error=str(exc), finished_unix=time.time())
                raise
            finally:
                write_json(manifest_path, manifest)
            if partial.stat().st_size != expected_size or sha256_file(partial) != expected_sha:
                raise ValueError(f"Downloaded size/hash mismatch: {remote_path}")
            os.replace(partial, target)
        manifest["files"] = [record for record in manifest["files"] if record["remote_path"] != remote_path]
        manifest["files"].append({"remote_path": remote_path, "local_path": str(target), "bytes": expected_size,
                                   "sha256": expected_sha, "verified": True, "downloaded_bytes_this_run": downloaded,
                                   "seconds": time.monotonic() - started})
        write_json(manifest_path, manifest)
        print(json.dumps({"event": "download_verified", "file": target.name, "bytes": expected_size}), flush=True)
    return manifest
