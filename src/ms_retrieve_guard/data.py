"""Identity-first audit and a small, sealed-role-safe experiment input builder."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import multiprocessing
from pathlib import Path
import sqlite3
import time

import ijson
import numpy as np

from .chemistry import CHEMISTRY_VERSION, FP_BITORDER, chemistry_metadata, fingerprint, standardize

SEALED = frozenset({"calibration", "test"})
DEV_ROLES = ("dev_ranker", "fit_selector", "calibration", "pilot_dev")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash(*parts) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _progress(out: Path, stage: str, **kwargs) -> None:
    record = {"timestamp": time.time(), "stage": stage, **kwargs}
    with (out / "audit_progress.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def validation_roles(identities, seed: int = 1729) -> dict[str, str]:
    """Exact deterministic 15/25/40/20 allocation, independent of input order."""
    ordered = sorted(set(identities), key=lambda x: (_hash(seed, "split", x), x))
    n = len(ordered)
    bounds = [int(n * .15), int(n * .40), int(n * .80), n]
    roles = {}
    start = 0
    for role, stop in zip(DEV_ROLES, bounds):
        roles.update({key: role for key in ordered[start:stop]})
        start = stop
    return roles


def bin_spectrum(mzs, intensities, *, n_bins: int = 10050, bin_width: float = .1) -> np.ndarray:
    """10 <= m/z < 1005; normalize peaks, sum bins, normalize summed bins."""
    if isinstance(mzs, str):
        mzs = np.fromstring(mzs.strip("[]"), sep="," if "," in mzs else " ", dtype=np.float64)
    if isinstance(intensities, str):
        intensities = np.fromstring(intensities.strip("[]"), sep="," if "," in intensities else " ", dtype=np.float64)
    mzs = np.asarray(mzs, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    if mzs.ndim != 1 or mzs.shape != intensities.shape or not len(mzs):
        raise ValueError("invalid_peak_shapes")
    if not np.isfinite(mzs).all() or not np.isfinite(intensities).all() or (intensities < 0).any() or (mzs < 0).any():
        raise ValueError("invalid_peak_values")
    valid = (mzs >= 10.) & (mzs < n_bins * bin_width) & (intensities > 0)
    if not valid.any():
        raise ValueError("empty_binned_spectrum")
    values = intensities[valid] / intensities[valid].max()
    bins = np.floor(mzs[valid] / bin_width).astype(np.int64)
    x = np.bincount(bins, weights=values, minlength=n_bins).astype(np.float32)
    x /= x.max()
    return x


def paired_candidate_keys(truth: str, negatives, k: int, *, seed: int = 42, mechanism: str = "U") -> tuple[list[str], list[str]]:
    """Paired sets share K-1 negatives; absent has an additional distinct negative."""
    if mechanism != "U":
        raise ValueError("Smoke implements U only; H/D remain frozen future mechanisms")
    unique = sorted(set(negatives) - {truth}, key=lambda key: (_hash(seed, "negative", truth, key), key))
    if len(unique) < k:
        raise ValueError("insufficient_distinct_negatives")
    present, absent = unique[:k - 1] + [truth], unique[:k]
    # Stable random permutations do not preserve truth-at-zero or candidate input order.
    present.sort(key=lambda key: (_hash(seed, "present_order", truth, key), key))
    absent.sort(key=lambda key: (_hash(seed, "absent_order", truth, key), key))
    return present, absent


def _connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-131072")
    return db


def _init_db(db):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS chemistry(raw_smiles TEXT PRIMARY KEY, identity TEXT, smiles TEXT,
        formula TEXT, exact_mass REAL, error TEXT);
    CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE IF NOT EXISTS queries(query_id TEXT PRIMARY KEY, identity TEXT, raw_smiles TEXT,
        original_fold TEXT, role TEXT, reason TEXT, formula TEXT, parent_mass REAL,
        precursor_mz REAL, adduct TEXT, mzs TEXT, intensities TEXT);
    CREATE INDEX IF NOT EXISTS query_identity ON queries(identity);
    CREATE TABLE IF NOT EXISTS molecules(identity TEXT PRIMARY KEY, role TEXT, original_fold TEXT,
        representative_id TEXT, smiles TEXT, formula TEXT, exact_mass REAL, negative_count INTEGER DEFAULT 0,
        truth_in_original_pool INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS pools(query_identity TEXT, candidate_identity TEXT,
        PRIMARY KEY(query_identity,candidate_identity));
    CREATE TABLE IF NOT EXISTS structures(identity TEXT, smiles TEXT, formula TEXT, exact_mass REAL,
        PRIMARY KEY(identity,formula));
    """)


class ChemistryCache:
    def __init__(self, db):
        self.db = db
        self.memo = {}

    def _remember(self, raw, row, *, insert=False):
        if insert:
            self.db.execute("INSERT INTO chemistry VALUES(?,?,?,?,?,?)", (raw, *row))
        if row[0]:
            self.db.execute("INSERT INTO structures VALUES(?,?,?,?) ON CONFLICT(identity,formula) DO UPDATE SET smiles=CASE WHEN excluded.smiles<structures.smiles THEN excluded.smiles ELSE structures.smiles END, exact_mass=CASE WHEN excluded.smiles<structures.smiles THEN excluded.exact_mass ELSE structures.exact_mass END", row[:4])
        # Bound Python memory; SQLite is the complete cache.
        if len(self.memo) >= 100000:
            self.memo.clear()
        self.memo[raw] = row
        return row

    def get(self, raw: str):
        if raw in self.memo:
            return self.memo[raw]
        row = self.db.execute("SELECT identity,smiles,formula,exact_mass,error FROM chemistry WHERE raw_smiles=?", (raw,)).fetchone()
        return self._remember(raw, _standardize_row(raw) if row is None else row, insert=row is None)

    def prefill(self, raw_smiles, executor=None, *, batch_size=1024):
        """Normalize missing candidates in bounded batches; only parent touches SQL.

        Batched SQLite reads preserve restart benefits. Process results are consumed
        in input order; representative selection remains a commutative lexical min.
        """
        ordered = list(dict.fromkeys(raw_smiles))
        for start in range(0, len(ordered), batch_size):
            batch = [raw for raw in ordered[start:start+batch_size] if raw not in self.memo]
            cached = {}
            for offset in range(0, len(batch), 900):
                part = batch[offset:offset+900]
                marks = ",".join("?" for _ in part)
                for raw, *values in self.db.execute(f"SELECT raw_smiles,identity,smiles,formula,exact_mass,error FROM chemistry WHERE raw_smiles IN ({marks})", part):
                    cached[raw] = tuple(values)
            for raw, row in cached.items():
                self._remember(raw, row)
            missing = [raw for raw in batch if raw not in cached]
            normalized = executor.map(_standardize_row, missing, chunksize=32) if executor is not None else map(_standardize_row, missing)
            for raw, row in zip(missing, normalized):
                self._remember(raw, row, insert=True)


def _standardize_row(raw):
    """Spawn-worker entrypoint: no database, files, spectra, or model imports."""
    try:
        mol = standardize(raw)
        return (mol.identity, mol.smiles, mol.formula, mol.exact_mass, None)
    except Exception as exc:
        return (None, None, None, None, str(exc)[:180])


def _number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def audit_data(tsv_path, candidates_path, output_dir, config: dict | None = None) -> dict:
    """Audit all roles, without loading/scoring sealed spectra or computing embeddings.

    Formula candidates only. Cached chemistry can resume after failure; audit outputs
    are rebuilt so partial molecule/query tables are never mistaken for complete.
    """
    config = config or {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tsv_path, candidates_path = Path(tsv_path), Path(candidates_path)
    seed = int(config.get("seed", config.get("split_seed", 1729)))
    mass_tol = float(config.get("mass_tolerance_da", .1))
    paths = {"tsv": str(tsv_path.resolve()), "candidates": str(candidates_path.resolve())}
    checksums = {"tsv": sha256_file(tsv_path), "candidates": sha256_file(candidates_path)}
    db = _connect(out / "audit.sqlite")
    _init_db(db)
    cache_signature = json.dumps(chemistry_metadata(), sort_keys=True)
    previous = db.execute("SELECT value FROM metadata WHERE key='chemistry_version'").fetchone()
    if previous and previous[0] != cache_signature:
        raise ValueError("Chemistry version changed; use a fresh audit directory")
    _write_json(out / "audit_summary.json", {"status": "running", "source_sha256": checksums})
    db.execute("INSERT OR REPLACE INTO metadata VALUES('chemistry_version',?)", (cache_signature,))
    db.execute("DELETE FROM queries")
    db.execute("DELETE FROM molecules")
    db.execute("DELETE FROM pools")
    db.commit()
    cache = ChemistryCache(db)
    reasons = Counter()
    count = 0
    _progress(out, "query_identity_start", checksums=checksums)
    with tsv_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"identifier", "smiles", "inchikey", "formula", "parent_mass", "fold", "mzs", "intensities"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Missing required TSV columns: {sorted(required-set(reader.fieldnames or []))}")
        for row in reader:
            identity, smiles, formula, mass, err = cache.get(row["smiles"])
            problems = []
            if err:
                problems.append("chemistry:" + err)
            else:
                if row["inchikey"].split("-")[0] != identity:
                    problems.append("identity_mismatch")
                if row["formula"] != formula:
                    problems.append("formula_mismatch")
                supplied_mass = _number(row["parent_mass"])
                if supplied_mass is None or abs(supplied_mass - mass) > mass_tol:
                    problems.append("mass_mismatch")
            fold = row["fold"]
            if fold not in {"train", "val", "test"}:
                problems.append("unknown_fold")
            reason = ";".join(problems) or None
            reasons.update(problems)
            # Peak fields intentionally not retained until unsealed roles are assigned.
            db.execute("INSERT INTO queries VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                row["identifier"], identity, row["smiles"], fold, None, reason,
                row["formula"], _number(row["parent_mass"]), _number(row.get("precursor_mz")), row.get("adduct"), None, None))
            count += 1
            if count % 10000 == 0:
                db.commit()
                _progress(out, "query_identity", rows=count, chemistry_cache=db.execute("SELECT COUNT(*) FROM chemistry").fetchone()[0])
    # All instances of a recomputed identity spanning upstream folds are excluded.
    crossing = [r[0] for r in db.execute("SELECT identity FROM queries WHERE identity IS NOT NULL GROUP BY identity HAVING COUNT(DISTINCT original_fold)>1")]
    for identity in crossing:
        db.execute("UPDATE queries SET reason=CASE WHEN reason IS NULL THEN 'cross_fold_identity' ELSE reason||';cross_fold_identity' END WHERE identity=?", (identity,))
    conflicting_query_identities = [r[0] for r in db.execute("""
        SELECT q.identity FROM queries q JOIN chemistry c ON q.raw_smiles=c.raw_smiles
        WHERE q.identity IS NOT NULL GROUP BY q.identity
        HAVING COUNT(DISTINCT c.formula)>1 OR MAX(c.exact_mass)-MIN(c.exact_mass)>?
        """, (mass_tol,))]
    for identity in conflicting_query_identities:
        db.execute("UPDATE queries SET reason=CASE WHEN reason IS NULL THEN 'identity_formula_mass_conflict' ELSE reason||';identity_formula_mass_conflict' END WHERE identity=?", (identity,))
    valid_val = [r[0] for r in db.execute("SELECT DISTINCT identity FROM queries WHERE original_fold='val' AND reason IS NULL")]
    role_map = validation_roles(valid_val, seed)
    db.execute("UPDATE queries SET role=original_fold WHERE reason IS NULL AND original_fold IN ('train','test')")
    db.executemany("UPDATE queries SET role=? WHERE identity=? AND reason IS NULL", [(role, key) for key, role in role_map.items()])
    rows = db.execute("SELECT identity,role,original_fold FROM queries WHERE reason IS NULL GROUP BY identity").fetchall()
    for identity, role, fold in rows:
        ids = [r[0] for r in db.execute("SELECT query_id FROM queries WHERE identity=? AND reason IS NULL", (identity,))]
        representative = min(ids, key=lambda q: (_hash(seed, "spectrum", identity, q), q))
        query_formula = db.execute("SELECT c.formula FROM queries q JOIN chemistry c ON q.raw_smiles=c.raw_smiles WHERE q.identity=? AND q.reason IS NULL LIMIT 1", (identity,)).fetchone()[0]
        structure = db.execute("SELECT smiles,formula,exact_mass FROM structures WHERE identity=? AND formula=?", (identity,query_formula)).fetchone()
        db.execute("INSERT INTO molecules(identity,role,original_fold,representative_id,smiles,formula,exact_mass) VALUES(?,?,?,?,?,?,?)", (identity, role, fold, representative, *structure))
    db.commit()
    _progress(out, "roles_assigned", rows=count, identities=len(rows), crossing_identities=len(crossing))
    # Spectra are only copied for nonsealed roles; this pass never loads sealed peaks into numpy.
    allowed_ids = {r[0] for r in db.execute("SELECT query_id FROM queries WHERE role IN ('train','dev_ranker','fit_selector','pilot_dev') AND reason IS NULL")}
    with tsv_path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["identifier"] in allowed_ids:
                db.execute("UPDATE queries SET mzs=?,intensities=? WHERE query_id=?", (row["mzs"], row["intensities"], row["identifier"]))
    db.commit()
    del allowed_ids
    # All aliases of a true identity aggregate into the same de-duplicated pool.
    candidates_seen, candidate_errors, formula_rejected, alias_entries = 0, 0, 0, 0
    workers = int(config.get("chemistry_workers", 4))
    if not 1 <= workers <= 8:
        raise ValueError("chemistry_workers must be between 1 and 8")
    chemistry_batch_size = int(config.get("chemistry_batch_size", 1024))
    if not 1 <= chemistry_batch_size <= 4096:
        raise ValueError("chemistry_batch_size must be between 1 and 4096")
    _progress(out, "candidate_identity_start", chemistry_workers=workers, chemistry_batch_size=chemistry_batch_size)
    valid_molecules = {row[0]: row[1] for row in db.execute("SELECT identity,formula FROM molecules")}
    pool_context = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) if workers > 1 else nullcontext(None)
    with pool_context as executor, candidates_path.open("rb") as f:
        for query_smiles, candidates in ijson.kvitems(f, ""):
            query = cache.get(query_smiles)
            if not query[0] or query[0] not in valid_molecules:
                continue
            if not isinstance(candidates, list):
                raise ValueError("Formula candidate JSON must map true SMILES to arrays")
            cache.prefill(candidates, executor, batch_size=chemistry_batch_size)
            alias_entries += 1
            target_formula = valid_molecules[query[0]]
            batch = []
            for candidate in candidates:
                candidates_seen += 1
                cand = cache.get(candidate)
                if cand[4]:
                    candidate_errors += 1
                    continue
                if cand[2] != target_formula:
                    formula_rejected += 1
                    continue
                if cand[0] != query[0]:
                    batch.append((query[0], cand[0]))
                else:
                    db.execute("UPDATE molecules SET truth_in_original_pool=1 WHERE identity=?", (query[0],))
            db.executemany("INSERT OR IGNORE INTO pools VALUES(?,?)", batch)
            if alias_entries % 200 == 0:
                db.commit()
                _progress(out, "candidate_identity", query_alias_entries=alias_entries,
                          candidate_occurrences=candidates_seen,
                          distinct_smiles=db.execute("SELECT COUNT(*) FROM chemistry").fetchone()[0])
    db.execute("UPDATE molecules SET negative_count=(SELECT COUNT(*) FROM pools WHERE query_identity=molecules.identity)")
    db.commit()
    retention = {}
    for role, total in db.execute("SELECT role,COUNT(*) FROM molecules GROUP BY role"):
        retention[role] = {"total_molecules": total, "by_k": {}}
        retention[role]["truth_missing_from_original_pool"] = db.execute("SELECT COUNT(*) FROM molecules WHERE role=? AND truth_in_original_pool=0", (role,)).fetchone()[0]
        for k in config.get("k_grid", config.get("k_values", [32, 64, 128])):
            eligible = db.execute("SELECT COUNT(*) FROM molecules WHERE role=? AND negative_count>=? AND truth_in_original_pool=1", (role, k)).fetchone()[0]
            retention[role]["by_k"][str(k)] = {"eligible_molecules": eligible, "retained_fraction": eligible / total if total else None}
    minimums = config.get("main_k_minimums", {"calibration": 800, "pilot_dev": 300, "fit_selector": 400, "dev_ranker": 200})
    passing = [k for k in config.get("k_grid", config.get("k_values", [32,64,128])) if all(retention.get(role, {}).get("by_k", {}).get(str(k), {}).get("eligible_molecules", 0) >= minimum for role, minimum in minimums.items())]
    main_k = max(passing) if passing else min(config.get("k_grid", config.get("k_values", [32,64,128])))
    with (out / "split_manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["identity", "role", "original_fold", "representative_query_id", "negative_count", "truth_in_original_pool"])
        writer.writerows(db.execute("SELECT identity,role,original_fold,representative_id,negative_count,truth_in_original_pool FROM molecules ORDER BY identity"))
    with (out / "quarantine.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "identity", "original_fold", "reason"])
        writer.writerows(db.execute("SELECT query_id,identity,original_fold,reason FROM queries WHERE reason IS NOT NULL ORDER BY query_id"))
    quarantined_rows = db.execute("SELECT COUNT(*) FROM queries WHERE reason IS NOT NULL").fetchone()[0]
    reasons = Counter()
    for (reason,) in db.execute("SELECT reason FROM queries WHERE reason IS NOT NULL"):
        reasons.update(reason.split(";"))
    summary = {"status": "complete", "information_condition": "known_formula", "paths": paths,
               "source_sha256": checksums, "seed": seed, "mass_tolerance_da": mass_tol,
               "chemistry": chemistry_metadata(), "query_rows": count, "molecules": len(rows),
               "quarantined_rows": quarantined_rows, "row_reason_counts": dict(reasons),
               "cross_fold_identities": len(crossing), "candidate_alias_entries": alias_entries,
               "conflicting_query_identities": len(conflicting_query_identities),
               "candidate_identity_multiple_formula_count": db.execute("SELECT COUNT(*) FROM (SELECT identity FROM structures GROUP BY identity HAVING COUNT(*)>1)").fetchone()[0],
               "candidate_occurrences_processed": candidates_seen, "candidate_chemistry_errors": candidate_errors,
               "chemistry_execution": {"workers": workers, "batch_size": chemistry_batch_size, "start_method": "spawn" if workers > 1 else "serial"},
               "candidate_formula_rejected": formula_rejected,
               "retention": retention, "main_k": main_k, "certification_sample_gate_passed": bool(passing),
               "sealed_roles": sorted(SEALED), "sealed_spectra_stored": False,
               "candidate_labels": "explicit standardized identity; no fingerprint matching",
               "manifest_sha256": sha256_file(out / "split_manifest.csv")}
    _write_json(out / "retention.json", retention)
    _write_json(out / "audit_summary.json", summary)
    db.execute("INSERT OR REPLACE INTO metadata VALUES('audit_summary',?)", (json.dumps(summary),))
    db.commit()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    _progress(out, "complete", main_k=main_k, sample_gate=bool(passing))
    return summary


def build_smoke(audit_dir, output_path, config: dict | None = None) -> dict:
    """Build only train/fit-selector/dev-ranker arrays, using one spectrum/molecule."""
    config = config or {}
    audit_dir, output_path = Path(audit_dir), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads((audit_dir / "audit_summary.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError("Audit must be complete before smoke construction")
    db = sqlite3.connect(f"file:{(audit_dir / 'audit.sqlite').resolve()}?mode=ro", uri=True)
    seed, candidate_seed = int(config.get("seed", config.get("split_seed",1729))), int(config.get("candidate_seed", 42))
    if config.get("candidate_mechanism", "U") != "U":
        raise ValueError("Smoke implements U only; H/D are not enabled")
    k = int(config.get("k", summary["main_k"]))
    specs = [("train", "train", int(config.get("n_train", 1000))),
             ("fit", "fit_selector", int(config.get("n_fit", 100))),
             ("eval", "dev_ranker", int(config.get("n_eval", 100)))]
    result = {"fingerprint_bitorder": np.array(FP_BITORDER)}
    manifest = {"status": "complete", "schema_version": 1, "candidate_mechanism": "U",
                "information_condition": "known_formula", "main_k": k, "seed": seed,
                "candidate_seed": candidate_seed, "chemistry": summary["chemistry"],
                "spectrum_binning": {"lower_mz_inclusive":10., "upper_mz_exclusive":1005., "bin_width":.1, "n_bins":10050},
                "audit_manifest_sha256": summary["manifest_sha256"], "source_sha256": summary["source_sha256"],
                "roles": {}, "excluded_spectrum_reasons": {}}
    fp_cache = {}

    def get_fp(key, formula):
        cache_key = (key, formula)
        if cache_key not in fp_cache:
            row = db.execute("SELECT smiles FROM structures WHERE identity=? AND formula=?", (key,formula)).fetchone()
            if row is None:
                raise ValueError(f"Missing candidate structure {key}")
            fp_cache[cache_key] = fingerprint(row[0])
        return fp_cache[cache_key]

    for prefix, role, requested in specs:
        if role in SEALED:
            raise ValueError("Sealed roles forbidden in smoke")
        eligible = db.execute("SELECT identity,representative_id,formula FROM molecules WHERE role=? AND negative_count>=? AND truth_in_original_pool=1", (role, k)).fetchall()
        eligible.sort(key=lambda row: (_hash(seed, "smoke", role, row[0]), row[0]))
        arrays = {name: [] for name in ["x", "candidates_present", "candidates_absent", "keys_present", "keys_absent", "truth_keys", "query_ids", "fp"]}
        bad = Counter()
        for key, query_id, formula in eligible:
            row = db.execute("SELECT role,mzs,intensities FROM queries WHERE query_id=? AND reason IS NULL", (query_id,)).fetchone()
            if row is None or row[0] != role or row[1] is None:
                raise ValueError("Role/spectrum manifest inconsistency")
            try:
                x = bin_spectrum(row[1], row[2])
            except ValueError as exc:
                bad[str(exc)] += 1
                continue
            negatives = [r[0] for r in db.execute("SELECT candidate_identity FROM pools WHERE query_identity=?", (key,))]
            present, absent = paired_candidate_keys(key, negatives, k, seed=candidate_seed)
            arrays["x"].append(x)
            arrays["candidates_present"].append(np.stack([get_fp(c,formula) for c in present]))
            arrays["candidates_absent"].append(np.stack([get_fp(c,formula) for c in absent]))
            arrays["keys_present"].append(present)
            arrays["keys_absent"].append(absent)
            arrays["truth_keys"].append(key)
            arrays["query_ids"].append(query_id)
            arrays["fp"].append(get_fp(key,formula))
            if len(arrays["x"]) == requested:
                break
        if len(arrays["x"]) != requested:
            raise ValueError(f"Need exactly {requested} eligible {role} molecules, got {len(arrays['x'])}; reduce configured count explicitly")
        for name, values in arrays.items():
            result[f"{prefix}_{name}"] = np.asarray(values)
        result[f"{prefix}_role"] = np.array(role)
        manifest["roles"][prefix] = {"role": role, "n_molecules": requested, "identities": arrays["truth_keys"],
                                     "query_ids": arrays["query_ids"], "one_spectrum_per_molecule": True,
                                     "candidate_keys_sha256": _hash(json.dumps(arrays["keys_present"]), json.dumps(arrays["keys_absent"]))}
        manifest["excluded_spectrum_reasons"][role] = dict(bad)
    role_sets = [set(manifest["roles"][prefix]["identities"]) for prefix, _, _ in specs]
    if any(a & b for i,a in enumerate(role_sets) for b in role_sets[i+1:]):
        raise ValueError("Cross-role identity leakage")
    # Public candidate reuse is permitted, but it is an explicit structure
    # exposure; it never authorizes loading a sealed query spectrum.
    candidate_identities = {key for key, _ in fp_cache}
    role_exposure = Counter()
    for key, role in db.execute("SELECT identity,role FROM molecules"):
        if key in candidate_identities:
            role_exposure[role] += 1
    manifest["candidate_structure_overlap_with_query_roles"] = dict(role_exposure)
    manifest["sealed_query_spectra_loaded"] = False
    np.savez_compressed(output_path, **result)
    manifest["output_path"] = str(output_path.resolve())
    manifest["output_sha256"] = sha256_file(output_path)
    manifest["npz_sha256"] = manifest["output_sha256"]
    manifest["npz_arrays"] = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key,value in result.items()}
    _write_json(output_path.with_name("smoke_manifest.json"), manifest)
    db.close()
    return manifest
