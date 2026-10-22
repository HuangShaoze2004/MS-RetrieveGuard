"""Small, explicitly labelled retrieval smoke experiment.

The MLP shape follows the MIT-licensed ms-mole implementation (upstream
f81558c4cb37ca2f3d4300eff8e3785eea53c9e3); this is a local reimplementation.
No upstream trainer, split handling, candidate-zero label, or test hook is used.
Repeated training steps benchmark a small subset and are not a full baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROLE_CONTRACT = {"train": "train", "fit": "fit_selector", "eval": "dev_ranker"}
SEALED_ROLES = {"calibration", "test", "pilot_test"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_npz(npz_path: str | Path, *, input_dim: int = 10050, output_dim: int = 4096) -> dict:
    """Validate role metadata before reading spectra, labels, or fingerprints.

    This is a defensive gate, not proof that an upstream split was correctly
    produced. The split/candidate construction manifest remains authoritative.
    """
    path = Path(npz_path)
    npz_hash = _sha256(path)
    manifest_report = None
    for candidate in (path.with_name("smoke_manifest.json"), path.with_suffix(".manifest.json")):
        if candidate.exists():
            manifest = json.loads(candidate.read_text())
            expected = manifest.get("npz_sha256")
            if expected is not None and expected != npz_hash:
                raise ValueError(f"NPZ checksum disagrees with {candidate}")
            manifest_report = {"path": str(candidate.resolve()), "sha256": _sha256(candidate),
                               "npz_checksum_checked": expected is not None}
            break
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            if name.endswith("_x") and name[:-2] not in ROLE_CONTRACT:
                raise ValueError(f"Unexpected or sealed spectral role in smoke bundle: {name}")
            if name.endswith("_role"):
                values = np.asarray(data[name]).reshape(-1)
                if any(str(value) in SEALED_ROLES for value in values):
                    raise ValueError(f"Sealed role in smoke bundle: {name}")
        for prefix, expected in ROLE_CONTRACT.items():
            role_key = prefix + "_role"
            if role_key not in data.files:
                raise ValueError(f"Missing explicit role metadata: {role_key}")
            role = np.asarray(data[role_key]).reshape(-1)
            if role.size != 1 or str(role[0]) != expected:
                raise ValueError(f"Expected {role_key}={expected!r}; got {role.tolist()!r}")
        bitorder = str(np.asarray(data["fingerprint_bitorder"]).item()) if "fingerprint_bitorder" in data.files else "little"
        if bitorder not in {"little", "big"}:
            raise ValueError("fingerprint_bitorder must be little or big")
        report: dict[str, Any] = {"npz_sha256": npz_hash, "fingerprint_bitorder": bitorder,
                                  "manifest": manifest_report, "roles": {}}
        role_identities = []
        packed_dim = (output_dim + 7) // 8
        for prefix, role in ROLE_CONTRACT.items():
            x = data[prefix + "_x"]
            truth = data[prefix + "_truth_keys"].astype(str)
            ids = data[prefix + "_query_ids"].astype(str)
            fp = data[prefix + "_fp"]
            n = len(truth)
            if n == 0 or x.shape != (n, input_dim) or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError(f"{prefix}: invalid or empty finite float32 input matrix")
            if ids.shape != (n,) or truth.shape != (n,) or any(not key for key in truth):
                raise ValueError(f"{prefix}: invalid query identifiers")
            if len(set(ids)) != n:
                raise ValueError(f"{prefix}: duplicate query IDs")
            if len(set(truth)) != n:
                raise ValueError(f"{prefix}: duplicate molecule identity in one-query-per-molecule smoke")
            if fp.shape != (n, packed_dim) or fp.dtype != np.uint8:
                raise ValueError(f"{prefix}: invalid packed true fingerprints")
            variants = {}
            keys_by_variant = {}
            packed_by_variant = {}
            for variant in ("present", "absent"):
                candidates = data[f"{prefix}_candidates_{variant}"]
                keys = data[f"{prefix}_keys_{variant}"].astype(str)
                if candidates.dtype != np.uint8 or candidates.ndim != 3 or candidates.shape[0] != n or candidates.shape[2] != packed_dim:
                    raise ValueError(f"{prefix}/{variant}: invalid packed candidates")
                k = candidates.shape[1]
                if k < 2 or keys.shape != (n, k):
                    raise ValueError(f"{prefix}/{variant}: invalid candidate identities or K")
                if any(len(set(row)) != k or any(not key for key in row) for row in keys):
                    raise ValueError(f"{prefix}/{variant}: duplicate or empty candidate identity")
                match_counts = np.sum(keys == truth[:, None], axis=1)
                expected_count = 1 if variant == "present" else 0
                if not np.all(match_counts == expected_count):
                    raise ValueError(f"{prefix}/{variant}: truth identity membership violated")
                if variant == "present":
                    truth_positions = (keys == truth[:, None]).argmax(axis=1)
                    if not np.array_equal(candidates[np.arange(n), truth_positions], fp):
                        raise ValueError(f"{prefix}: true fingerprint does not match identity-indexed candidate vector")
                variants[variant] = k
                keys_by_variant[variant] = keys
                packed_by_variant[variant] = candidates
            if variants["present"] != variants["absent"]:
                raise ValueError(f"{prefix}: present and absent K differ")
            for i in range(n):
                present_negatives = set(keys_by_variant["present"][i]) - {truth[i]}
                if not present_negatives.issubset(set(keys_by_variant["absent"][i])):
                    raise ValueError(f"{prefix}: pair does not share K-1 negatives")
                negative_positions = np.flatnonzero(keys_by_variant["present"][i] != truth[i])
                absent_positions = {key: j for j, key in enumerate(keys_by_variant["absent"][i])}
                matching_positions = [absent_positions[keys_by_variant["present"][i, j]]
                                      for j in negative_positions]
                # Compare packed vectors one row at a time, without expanding
                # the full molecule x candidate x fingerprint bit matrix.
                if not np.array_equal(packed_by_variant["present"][i, negative_positions],
                                      packed_by_variant["absent"][i, matching_positions]):
                    raise ValueError(f"{prefix}: shared negative identity has inconsistent fingerprint")
            role_identities.append(set(truth))
            report["roles"][prefix] = {"role": role, "queries": n, "molecules": len(set(truth)),
                                        "K": variants["present"]}
        for i in range(len(role_identities)):
            for j in range(i):
                if role_identities[i] & role_identities[j]:
                    raise ValueError("Molecular identity overlap across train/fit/eval")
    return report


class MoleculeMLP(nn.Module):
    def __init__(self, input_dim: int = 10050, hidden_dim: int = 1024, output_dim: int = 4096):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.25), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.25), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, output_dim), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def explicit_target_indices(candidate_keys: np.ndarray, truth_keys: np.ndarray) -> np.ndarray:
    """Fingerprint collisions never define correctness or listwise targets."""
    matches = candidate_keys == truth_keys[:, None]
    if not np.all(matches.sum(axis=1) == 1):
        raise ValueError("Training requires exactly one explicit true identity per candidate row")
    return matches.argmax(axis=1).astype(np.int64)


def _unpack(packed: np.ndarray, output_dim: int, bitorder: str, device: torch.device) -> torch.Tensor:
    unpacked = np.unpackbits(packed, axis=-1, bitorder=bitorder)[..., :output_dim]
    return torch.from_numpy(unpacked).to(device=device, dtype=torch.float32)


def candidate_cosine_scores(predictions: np.ndarray, packed_candidates: np.ndarray, *,
                            bitorder: str = "little", batch_size: int = 32,
                            device: str | torch.device = "cpu") -> np.ndarray:
    device = torch.device(device)
    if len(predictions) != len(packed_candidates):
        raise ValueError("Predictions and candidate rows disagree")
    scores = np.empty(packed_candidates.shape[:2], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(predictions), batch_size):
            end = start + batch_size
            p = torch.as_tensor(predictions[start:end], dtype=torch.float32, device=device)
            c = _unpack(packed_candidates[start:end], p.shape[-1], bitorder, device)
            scores[start:end] = torch.einsum("bd,bkd->bk", F.normalize(p, dim=-1), F.normalize(c, dim=-1)).cpu().numpy()
    return scores


def nearest_neighbor_scores(train_x: np.ndarray, train_fp: np.ndarray, train_keys: np.ndarray,
                            train_query_ids: np.ndarray, query_x: np.ndarray, packed_candidates: np.ndarray,
                            *, bitorder: str = "little", batch_size: int = 32,
                            device: str | torch.device = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Training-library cosine NN with identity/query-ID deterministic ties."""
    device = torch.device(device)
    if len(train_x) == 0:
        raise ValueError("Empty nearest-neighbor reference library")
    order = np.lexsort((train_query_ids.astype(str), train_keys.astype(str)))
    with torch.no_grad():
        reference = F.normalize(torch.as_tensor(train_x[order], dtype=torch.float32, device=device), dim=-1)
        indices = []
        for start in range(0, len(query_x), batch_size):
            query = F.normalize(torch.as_tensor(query_x[start:start + batch_size], dtype=torch.float32, device=device), dim=-1)
            # argmax chooses the first in the identity-sorted library for exact ties.
            positions = (query @ reference.T).argmax(dim=1).cpu().numpy()
            indices.append(order[positions])
    nearest = np.concatenate(indices)
    predictions = np.unpackbits(train_fp[nearest], axis=-1, bitorder=bitorder).astype(np.float32)
    return candidate_cosine_scores(predictions, packed_candidates, bitorder=bitorder,
                                   batch_size=batch_size, device=device), nearest


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                     device: torch.device) -> dict:
    # Only checkpoints created by this local run are loaded. No public pickle is trusted.
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if device.type == "cuda" and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng_state"]])
    random.setstate(state["python_rng_state"])
    return state


def run_smoke(npz_path: str | Path, output_dir: str | Path, config: dict | None = None) -> dict:
    """Train/benchmark and score only explicitly permitted unsealed roles.

    Config permits smaller dimensions/steps for CPU integration tests. Production
    defaults implement the 100-query pilot; GPU orchestration belongs to the CLI.
    """
    config = dict(config or {})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if (output / "checkpoint.pt").exists() and not config.get("resume_from"):
        raise FileExistsError("Run directory already has a checkpoint; use a new directory or explicit resume_from")
    started = time.monotonic()
    _json(status_path, {"status": "running", "pid": os.getpid(), "started_unix": time.time()})
    try:
        result = _run_smoke(Path(npz_path), output, config, started)
        _json(status_path, {"status": "budget_stopped" if result["budget_stopped"] else "complete",
                            "pid": os.getpid(), "ended_unix": time.time(),
                            "elapsed_seconds": time.monotonic() - started})
        return result
    except BaseException as exc:
        _json(status_path, {"status": "failed", "pid": os.getpid(), "ended_unix": time.time(),
                            "elapsed_seconds": time.monotonic() - started,
                            "error_type": type(exc).__name__, "error": str(exc)})
        raise


def _run_smoke(npz_path: Path, output: Path, config: dict, started: float) -> dict:
    input_read_started = time.monotonic()
    input_dim = int(config.get("input_dim", 10050))
    output_dim = int(config.get("output_dim", 4096))
    audit = validate_npz(npz_path, input_dim=input_dim, output_dim=output_dim)
    _json(output / "input_validation.json", audit)
    seed = int(config.get("seed", 0))
    device_name = config.get("device", "auto")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    torch.set_num_threads(int(config.get("torch_num_threads", 8)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    generator = np.random.default_rng(seed)
    with np.load(npz_path, allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files if not key.endswith("_role")}
    input_read_seconds = time.monotonic() - input_read_started
    bitorder = audit["fingerprint_bitorder"]
    hidden_dim = int(config.get("hidden_dim", 1024))
    model = MoleculeMLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-4)),
                                 weight_decay=float(config.get("weight_decay", 0.0)))
    temperature = float(config.get("temperature", 1.0))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    train_steps = int(config.get("train_steps", 1000))
    batch_size = int(config.get("batch_size", 64))
    if train_steps < 1 or batch_size < 1:
        raise ValueError("train_steps and batch_size must be positive")
    max_gpu_seconds = float(config.get("max_gpu_seconds", 3600))
    if not math.isfinite(max_gpu_seconds) or max_gpu_seconds <= 0 or max_gpu_seconds > 7200:
        raise ValueError("max_gpu_seconds must be in (0, 7200] for the first package")
    start_step = 0
    if config.get("resume_from"):
        resumed = _load_checkpoint(Path(config["resume_from"]), model, optimizer, device)
        if resumed["npz_sha256"] != audit["npz_sha256"]:
            raise ValueError("Cannot resume against a different NPZ")
        generator.bit_generator.state = resumed["numpy_generator_state"]
        start_step = int(resumed["step"])
    target_indices = explicit_target_indices(arrays["train_keys_present"], arrays["train_truth_keys"])
    train_x = arrays["train_x"]
    train_candidates = arrays["train_candidates_present"]
    effective_config = {**config, "input_dim": input_dim, "output_dim": output_dim,
                        "hidden_dim": hidden_dim, "seed": seed, "device": str(device),
                        "train_steps": train_steps, "batch_size": batch_size, "temperature": temperature,
                        "learning_rate": float(config.get("learning_rate", 1e-4)),
                        "weight_decay": float(config.get("weight_decay", 0)),
                        "grad_clip": float(config.get("grad_clip", 1.0)),
                        "max_gpu_seconds": max_gpu_seconds, "fingerprint_bitorder": bitorder}
    config_serialized = json.dumps(effective_config, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_serialized.encode()).hexdigest()
    _json(output / "effective_config.json", effective_config)
    if config.get("resume_from"):
        immutable = ("input_dim", "output_dim", "hidden_dim", "seed", "temperature", "batch_size",
                     "learning_rate", "weight_decay", "grad_clip", "fingerprint_bitorder")
        changed = [name for name in immutable if resumed["config"].get(name) != effective_config.get(name)]
        if changed:
            raise ValueError(f"Resume config changes immutable training parameters: {changed}")

    def update(indices: np.ndarray) -> float:
        model.train()
        x = torch.as_tensor(train_x[indices], dtype=torch.float32, device=device)
        candidates = _unpack(train_candidates[indices], output_dim, bitorder, device)
        targets = torch.as_tensor(target_indices[indices], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        predicted = F.normalize(model(x), dim=-1)
        logits = torch.einsum("bd,bkd->bk", predicted, F.normalize(candidates, dim=-1)) / temperature
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite listwise training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)), error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach().cpu())

    def save_checkpoint(step: int) -> None:
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                 "numpy_generator_state": generator.bit_generator.state,
                 "torch_rng_state": torch.get_rng_state(),
                 "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                 "python_rng_state": random.getstate(), "config": effective_config,
                 "config_sha256": config_hash, "npz_sha256": audit["npz_sha256"]}
        temporary = output / "checkpoint.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(output / "checkpoint.pt")

    step_times = []
    losses = []
    completed = start_step
    budget_stopped = False
    _sync(device)
    training_started = time.monotonic()
    with (output / "train.jsonl").open("a") as log:
        for step in range(start_step, train_steps):
            elapsed = time.monotonic() - started
            # Keep time for checkpoint, restore canary and scoring inside the cap.
            reserve = max(30.0, 5.0 * (np.median(step_times[-50:]) if step_times else 0.0))
            if elapsed + reserve >= max_gpu_seconds:
                budget_stopped = True
                break
            indices = generator.integers(0, len(train_x), size=min(batch_size, len(train_x)))
            _sync(device)
            step_start = time.monotonic()
            loss = update(indices)
            _sync(device)
            elapsed_step = time.monotonic() - step_start
            step_times.append(elapsed_step)
            losses.append(loss)
            completed = step + 1
            log.write(json.dumps({"step": completed, "loss": loss, "seconds": elapsed_step,
                                  "elapsed_seconds": time.monotonic() - started}) + "\n")
            log.flush()
            if completed % int(config.get("checkpoint_every", 100)) == 0:
                save_checkpoint(completed)
                _json(output / "status.json", {"status": "running", "pid": os.getpid(), "step": completed,
                                               "heartbeat_unix": time.time()})
    _sync(device)
    training_seconds = time.monotonic() - training_started
    save_checkpoint(completed)
    if completed == 0:
        raise RuntimeError("Time budget prevented even one training step; checkpoint saved")

    # Validate prediction restoration and a genuine optimizer/RNG resume update.
    check_x = torch.as_tensor(arrays["eval_x"][:min(batch_size, len(arrays["eval_x"]))], device=device)
    model.eval()
    with torch.no_grad():
        prediction_before = model(check_x).cpu().numpy()
    checkpoint_path = output / "checkpoint.pt"
    _load_checkpoint(checkpoint_path, model, optimizer, device)
    model.eval()
    with torch.no_grad():
        prediction_after = model(check_x).cpu().numpy()
    prediction_max_abs = float(np.max(np.abs(prediction_before - prediction_after)))
    canary_indices = np.arange(min(batch_size, len(train_x)))
    canary_loss_first = update(canary_indices)
    canary_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    _load_checkpoint(checkpoint_path, model, optimizer, device)
    canary_loss_resumed = update(canary_indices)
    resume_max_abs = max(float(torch.max(torch.abs(value.detach().cpu() - canary_state[key])))
                         for key, value in model.state_dict().items())
    checkpoint_validation = {"prediction_max_abs": prediction_max_abs,
                             "optimizer_resume_parameter_max_abs": resume_max_abs,
                             "optimizer_resume_loss_abs": abs(canary_loss_first - canary_loss_resumed),
                             "prediction_equal": bool(np.allclose(prediction_before, prediction_after, atol=1e-7, rtol=0)),
                             "optimizer_resume_equal": resume_max_abs <= 1e-7,
                             "note": "Canary updates are discarded; scored model is the saved checkpoint."}
    _json(output / "checkpoint_validation.json", checkpoint_validation)
    if not checkpoint_validation["prediction_equal"] or not checkpoint_validation["optimizer_resume_equal"]:
        raise RuntimeError("Checkpoint prediction/optimizer resume validation failed")
    _load_checkpoint(checkpoint_path, model, optimizer, device)
    model.eval()

    score_arrays = {}
    score_times = {}
    nearest_rows = {}
    score_batch = int(config.get("score_batch_size", 16))
    with torch.no_grad():
        for prefix in ("fit", "eval"):
            _sync(device)
            score_start = time.monotonic()
            x = arrays[prefix + "_x"]
            chunks = [model(torch.as_tensor(x[start:start + score_batch], device=device)).cpu().numpy()
                      for start in range(0, len(x), score_batch)]
            predictions = np.concatenate(chunks)
            for variant in ("present", "absent"):
                score_arrays[f"mlp_{prefix}_{variant}"] = candidate_cosine_scores(
                    predictions, arrays[f"{prefix}_candidates_{variant}"], bitorder=bitorder,
                    batch_size=score_batch, device=device)
            _sync(device)
            score_times["mlp_" + prefix] = time.monotonic() - score_start
            _sync(device)
            score_start = time.monotonic()
            for variant in ("present", "absent"):
                scores, nearest = nearest_neighbor_scores(
                    train_x, arrays["train_fp"], arrays["train_truth_keys"], arrays["train_query_ids"],
                    x, arrays[f"{prefix}_candidates_{variant}"], bitorder=bitorder,
                    batch_size=score_batch, device=device)
                score_arrays[f"nn_{prefix}_{variant}"] = scores
                nearest_rows[prefix] = nearest
            _sync(device)
            score_times["nn_" + prefix] = time.monotonic() - score_start
            if time.monotonic() - started >= max_gpu_seconds:
                raise RuntimeError("GPU wall-time cap reached during scoring; no complete-run claim")
    score_arrays.update({f"nn_{prefix}_train_indices": value for prefix, value in nearest_rows.items()})
    score_arrays.update({f"{prefix}_query_ids": arrays[f"{prefix}_query_ids"] for prefix in ("fit", "eval")})
    score_arrays["input_npz_sha256"] = np.asarray(audit["npz_sha256"])
    np.savez_compressed(output / "scores.npz", **score_arrays)
    warmup = min(50, max(0, len(step_times) // 10))
    stable = step_times[warmup:] or step_times
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    report = {"experiment_kind": "small_subset_engineering_smoke_not_full_baseline",
              "completed_steps": completed, "requested_steps": train_steps, "budget_stopped": budget_stopped,
              "device": str(device), "torch_version": torch.__version__, "numpy_version": np.__version__,
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
              "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
              "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
              "input_validation_and_read_seconds": input_read_seconds,
              "input_npz_bytes": npz_path.stat().st_size,
              "loaded_array_bytes": sum(value.nbytes for value in arrays.values()),
              "training_seconds": training_seconds, "scoring_seconds": score_times,
              "training_step_seconds_median": float(np.median(stable)) if stable else None,
              "training_step_seconds_p95": float(np.quantile(stable, 0.95)) if stable else None,
              "benchmark_warmup_steps_excluded": warmup,
              "sampled_batch_size": min(batch_size, len(train_x)),
              "first_loss": losses[0] if losses else None, "final_loss": losses[-1] if losses else None,
              "checkpoint_sha256": _sha256(checkpoint_path), "config_sha256": config_hash,
              "npz_sha256": audit["npz_sha256"], "source_commit": git_commit,
              "ranker_source_sha256": _sha256(Path(__file__)),
              "scores_path": str((output / "scores.npz").resolve()),
              "checkpoint_path": str(checkpoint_path.resolve()),
              "checkpoint_validation": checkpoint_validation,
              "elapsed_seconds": time.monotonic() - started,
              "role_audit": audit["roles"],
              "caveat": "Fit/eval scores are development outputs. No calibration/test scores were loaded or produced."}
    _json(output / "benchmark.json", report)
    return report


train_smoke = run_smoke
