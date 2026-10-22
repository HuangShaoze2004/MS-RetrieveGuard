"""Read-only exact-spectrum audit, including sealed roles for leakage audit only.

This module does not bin spectra, normalize intensities, compute embeddings,
load model inputs from an NPZ, or compute retrieval metrics. Identical peak
lists alone are not proof of copied observations or leakage: a one-peak list
may simply describe a common fragment. The smoke gate is deliberately narrower.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time

from .io import sha256_file, write_json


HASH_KINDS = ("raw_peaks", "numeric_peaks", "composite")
SMOKE_ROLES = {"train": "train", "fit": "fit_selector", "eval": "dev_ranker"}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _numbers(value: str) -> list[float]:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not text:
        raise ValueError("empty_peak_list")
    values = [float(token) for token in re.split(r"[\s,]+", text)]
    if not all(math.isfinite(number) and number >= 0 for number in values):
        raise ValueError("nonfinite_or_negative_peak_value")
    return values


def _number_text(value: float) -> str:
    # Canonicalize signed zero; otherwise preserve the exact parsed float64.
    return (0.0 if value == 0 else value).hex()


def _numeric_peak_hash(mzs: str, intensities: str) -> tuple[str, int]:
    masses, heights = _numbers(mzs), _numbers(intensities)
    if len(masses) != len(heights):
        raise ValueError("unaligned_peak_lists")
    # Reordering and numeric spelling are immaterial; peak multiplicity, exact
    # float values and intensity scaling remain material. No model features.
    peaks = sorted(zip(masses, heights, strict=True))
    serialized = json.dumps([[_number_text(m), _number_text(i)] for m, i in peaks],
                            separators=(",", ":"))
    return _digest(serialized), len(peaks)


def _peak_bucket(count) -> str:
    if count is None:
        return "invalid_numeric"
    if count == 1:
        return "single_peak"
    if count < 4:
        return "two_or_three_peaks"
    return "at_least_four_peaks"


def _cross_role_identity_pairs(records) -> bool:
    labels = {(r["identity"], r["role"]) for r in records}
    return any(a[0] != b[0] and a[1] != b[1] for a in labels for b in labels)


def _group_summary(groups, records) -> dict:
    duplicates = [[records[i] for i in indices] for indices in groups.values() if len(indices) > 1]
    cross_identity = [g for g in duplicates if len({r["identity"] for r in g}) > 1]
    cross_role = [g for g in duplicates if len({r["role"] for r in g}) > 1]
    cross_both = [g for g in duplicates if _cross_role_identity_pairs(g)]
    same_identity_groups = []
    same_identity_rows = 0
    for group in duplicates:
        counts = Counter(r["identity"] for r in group)
        if any(count > 1 for count in counts.values()):
            same_identity_groups.append(group)
            same_identity_rows += sum(count for count in counts.values() if count > 1)
    return {"hashed_rows": sum(map(len, groups.values())), "unique_hashes": len(groups),
            "duplicate_groups": len(duplicates),
            "rows_in_duplicate_groups": sum(map(len, duplicates)),
            "groups_with_within_identity_repeats": len(same_identity_groups),
            "rows_in_within_identity_repeats": same_identity_rows,
            "cross_identity_groups": len(cross_identity),
            "rows_in_cross_identity_groups": sum(map(len, cross_identity)),
            "cross_role_groups": len(cross_role),
            "rows_in_cross_role_groups": sum(map(len, cross_role)),
            "cross_role_cross_identity_groups": len(cross_both),
            "rows_in_cross_role_cross_identity_groups": sum(map(len, cross_both)),
            "cross_role_cross_identity_groups_by_peak_count": dict(Counter(
                _peak_bucket(g[0]["peak_count"]) for g in cross_both)),
            "cross_role_cross_identity_rows_by_peak_count": dict(Counter(
                _peak_bucket(r["peak_count"]) for g in cross_both for r in g))}


def _selected_queries(smoke_npz, query_metadata) -> dict:
    if smoke_npz is None:
        return {}
    # np.load is lazy: only scalar role metadata and query-ID arrays are read.
    # In particular, no *_x, fingerprint, candidate, score or truth array loads.
    import numpy as np

    selected = {}
    with np.load(smoke_npz, allow_pickle=False) as data:
        for prefix, role in SMOKE_ROLES.items():
            supplied_role = np.asarray(data[f"{prefix}_role"])
            if supplied_role.size != 1 or str(supplied_role.reshape(-1)[0]) != role:
                raise ValueError("Unexpected or sealed smoke role in duplicate audit")
        for prefix, role in SMOKE_ROLES.items():
            ids = np.asarray(data[f"{prefix}_query_ids"])
            if ids.ndim != 1:
                raise ValueError("Smoke query IDs must be one-dimensional")
            for supplied_id in ids:
                query_id = str(supplied_id)
                if query_id in selected:
                    raise ValueError("Duplicate query ID across smoke roles")
                if query_id not in query_metadata or query_metadata[query_id][1] != role:
                    raise ValueError(f"Smoke query role disagrees with identity audit: {query_id}")
                selected[query_id] = role
    return selected


def audit_spectrum_duplicates(tsv_path, audit_sqlite, output_dir, smoke_npz=None) -> dict:
    """Audit exact duplicates without changing the identity database.

    selected_cross_role_count counts UNIQUE selected queries participating in a
    cross-role, cross-identity composite duplicate with >=4 peaks. It is a smoke
    blocking rule for strongly suspicious duplication, not proof of leakage.
    All other duplicate counts are diagnostic and never trigger this gate.
    """
    tsv_path, audit_sqlite, out = Path(tsv_path), Path(audit_sqlite), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "duplicate_summary.json"
    started = time.monotonic()
    write_json(summary_path, {"status": "running", "scope": "identity_and_leakage_audit_only"})
    try:
        with sqlite3.connect(f"file:{audit_sqlite.resolve()}?mode=ro", uri=True) as db:
            metadata = {qid: (identity, role, fold) for qid, identity, role, fold in db.execute(
                "SELECT query_id,identity,role,original_fold FROM queries WHERE reason IS NULL AND role IS NOT NULL AND identity IS NOT NULL")}
            excluded = db.execute("SELECT COUNT(*) FROM queries WHERE reason IS NOT NULL OR role IS NULL OR identity IS NULL").fetchone()[0]
        selected = _selected_queries(smoke_npz, metadata)
        records, seen = [], set()
        groups = {kind: defaultdict(list) for kind in HASH_KINDS}
        failures = Counter()
        composite_missing = 0
        fields = ["query_id", "identity", "role", "original_fold", "peak_count", "peak_bucket",
                  "precursor_mz", "adduct", "instrument_type", "collision_energy",
                  "raw_peaks_sha256", "numeric_peaks_sha256", "composite_sha256", "numeric_error"]
        with tsv_path.open(newline="") as stream, (out / "spectrum_hashes.csv").open("w", newline="") as target:
            reader = csv.DictReader(stream, delimiter="\t")
            required = {"identifier", "mzs", "intensities", "precursor_mz", "adduct"}
            if not required <= set(reader.fieldnames or []):
                raise ValueError("Missing columns for spectrum duplicate audit")
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                query_id = row["identifier"]
                if query_id not in metadata:
                    continue
                if query_id in seen:
                    raise ValueError(f"Duplicate source query ID: {query_id}")
                seen.add(query_id)
                identity, role, fold = metadata[query_id]
                raw_hash = _digest(row["mzs"] + "\0" + row["intensities"])
                record = {"query_id": query_id, "identity": identity, "role": role,
                          "original_fold": fold, "peak_count": None,
                          "precursor_mz": row["precursor_mz"], "adduct": row["adduct"].strip(),
                          "instrument_type": row.get("instrument_type", ""),
                          "collision_energy": row.get("collision_energy", ""),
                          "raw_peaks_sha256": raw_hash, "numeric_peaks_sha256": "",
                          "composite_sha256": "", "numeric_error": ""}
                try:
                    numeric_hash, count = _numeric_peak_hash(row["mzs"], row["intensities"])
                    record.update(numeric_peaks_sha256=numeric_hash, peak_count=count)
                    try:
                        precursor = float(row["precursor_mz"])
                        if not math.isfinite(precursor) or precursor <= 0 or not record["adduct"]:
                            raise ValueError("missing_or_invalid_precursor_or_adduct")
                        record["composite_sha256"] = _digest(json.dumps(
                            [numeric_hash, _number_text(precursor), record["adduct"]], separators=(",", ":")))
                    except (ValueError, TypeError):
                        composite_missing += 1
                except (ValueError, TypeError) as exc:
                    record["numeric_error"] = str(exc)
                    failures[str(exc)] += 1
                    composite_missing += 1
                record["peak_bucket"] = _peak_bucket(record["peak_count"])
                index = len(records)
                records.append(record)
                for kind in HASH_KINDS:
                    if record[f"{kind}_sha256"]:
                        groups[kind][record[f"{kind}_sha256"]].append(index)
                writer.writerow(record)
        missing = set(metadata) - seen
        if missing:
            raise ValueError(f"Source TSV missing {len(missing)} identity-audited query IDs")
        selected_groups = {kind: {} for kind in HASH_KINDS}
        for kind in HASH_KINDS:
            for digest, indices in groups[kind].items():
                subset = [i for i in indices if records[i]["query_id"] in selected]
                if subset:
                    selected_groups[kind][digest] = subset
        block_groups, blocked_ids = [], set()
        for digest, indices in selected_groups["composite"].items():
            subset = [records[i] for i in indices]
            if subset[0]["peak_count"] >= 4 and _cross_role_identity_pairs(subset):
                block_groups.append(digest)
                # Count only rows that have an actual different-role/identity peer.
                for row in subset:
                    if any(row["role"] != peer["role"] and row["identity"] != peer["identity"] for peer in subset):
                        blocked_ids.add(row["query_id"])
        with (out / "duplicate_groups.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["hash_kind", "group_sha256", "group_size", "cross_role_cross_identity", "selected_for_smoke", *fields])
            writer.writeheader()
            for kind in HASH_KINDS:
                for digest, indices in sorted(groups[kind].items()):
                    if len(indices) < 2:
                        continue
                    subset = [records[i] for i in indices]
                    cross = _cross_role_identity_pairs(subset)
                    for row in subset:
                        writer.writerow({"hash_kind": kind, "group_sha256": digest, "group_size": len(indices),
                                         "cross_role_cross_identity": cross,
                                         "selected_for_smoke": row["query_id"] in selected, **row})
        with (out / "selected_blocking_conflicts.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in records if row["query_id"] in blocked_ids)
        summary = {"status": "complete", "schema_version": 1,
                   "scope": "identity_and_leakage_audit_only",
                   "model_inputs_produced": False, "model_scores_or_metrics_computed": False,
                   "sealed_roles_use": "Exact duplicate leakage audit only; no embedding, binning or model scoring.",
                   "rules": {"raw_peaks": "SHA256 of the exact mzs string, a NUL, and exact intensities string; matches preliminary 133-group/371-row audit method.",
                             "numeric_peaks": "Parse all tokens as finite nonnegative float64, sort (mz,intensity) pairs, serialize float.hex pairs as compact JSON; signed zero normalized, no rounding or binning, preserve repeated peaks.",
                             "intensity_normalization": "none; spectra differing by an intensity scale factor are not declared equal",
                             "composite": "SHA256 of numeric peak hash + exact parsed float64 precursor_mz + stripped adduct; missing precursor/adduct excluded. Instrument and collision_energy recorded as diagnostics, not equality keys.",
                             "peak_buckets": "Mutually exclusive single_peak, two_or_three_peaks, at_least_four_peaks; invalid_numeric separate.",
                             "blocking": "Only selected cross-role AND cross-identity composite-equal queries with >=4 peaks block the smoke; equality alone does not prove copied data or leakage."},
                   "denominators": {"identity_audited_rows": len(metadata), "excluded_identity_audit_rows": excluded,
                                    "source_rows_audited": len(records), "numeric_valid_rows": len(records) - sum(failures.values()),
                                    "numeric_invalid_rows": sum(failures.values()), "composite_eligible_rows": len(records) - composite_missing,
                                    "roles": dict(Counter(r["role"] for r in records)),
                                    "selected_queries": len(selected), "selected_roles": dict(Counter(selected.values()))},
                   "numeric_failures": dict(failures),
                   "duplicates": {kind: _group_summary(groups[kind], records) for kind in HASH_KINDS},
                   "selected_duplicates": {kind: _group_summary(selected_groups[kind], records) for kind in HASH_KINDS},
                   "selected_cross_role_count": len(blocked_ids),
                   "selected_cross_role_count_unit": "unique selected query IDs in blocking composite groups, not independent samples or peak-only groups",
                   "selected_cross_role_group_count": len(block_groups),
                   "selected_conflicting_query_ids": sorted(blocked_ids),
                   "selected_gate_passed": not blocked_ids,
                   "sources": {"tsv": str(tsv_path.resolve()), "tsv_sha256": sha256_file(tsv_path),
                               "audit_sqlite": str(audit_sqlite.resolve()),
                               "identity_metadata_sha256": _digest(json.dumps(sorted((key, *value) for key, value in metadata.items()), separators=(",", ":"))),
                               "smoke_npz": str(Path(smoke_npz).resolve()) if smoke_npz is not None else None,
                               "smoke_npz_sha256": sha256_file(smoke_npz) if smoke_npz is not None else None},
                   "outputs": {name: {"path": str((out / name).resolve()), "sha256": sha256_file(out / name)} for name in ["spectrum_hashes.csv", "duplicate_groups.csv", "selected_blocking_conflicts.csv"]},
                   "limitations": ["Exact equality audit only; no near-duplicate search or numerical tolerance.",
                                   "Shared low-information/common-fragment peak lists do not establish leakage.",
                                   "Metadata cannot establish whether observations were copied; source-level verification may be required.",
                                   "This audit does not inspect candidate pools or change molecule roles/SQLite records."],
                   "elapsed_seconds": time.monotonic() - started}
        write_json(summary_path, summary)
        return summary
    except BaseException as exc:
        write_json(summary_path, {"status": "failed", "scope": "identity_and_leakage_audit_only",
                                  "error_type": type(exc).__name__, "error": str(exc),
                                  "elapsed_seconds": time.monotonic() - started})
        raise
