"""Exact equality is diagnostic; only the declared composite gate blocks smoke."""
import csv
import json
import sqlite3

import numpy as np
import pytest

from ms_retrieve_guard.duplicates import _numeric_peak_hash, audit_spectrum_duplicates
from ms_retrieve_guard.io import sha256_file


def _fixture(tmp_path, rows):
    fields = ["identifier", "mzs", "intensities", "precursor_mz", "adduct", "instrument_type", "collision_energy"]
    db_path = tmp_path / "audit.sqlite"
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE queries(query_id TEXT, identity TEXT, role TEXT, original_fold TEXT, reason TEXT)")
        for row in rows:
            db.execute("INSERT INTO queries VALUES(?,?,?,?,?)", (row["identifier"], row["identity"], row["role"], row["role"], row.get("reason")))
    tsv = tmp_path / "data.tsv"
    with tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    return tsv, db_path


def _row(qid, identity, role, mzs="10,20,30,40", intensities="1,2,3,4", precursor="100", adduct="[M+H]+"):
    return {"identifier": qid, "identity": identity, "role": role, "mzs": mzs,
            "intensities": intensities, "precursor_mz": precursor, "adduct": adduct}


def _smoke(tmp_path, train, fit, evaluation):
    p = tmp_path / "smoke.npz"
    arrays = {}
    for prefix, role, ids in [("train", "train", train), ("fit", "fit_selector", fit), ("eval", "dev_ranker", evaluation)]:
        arrays[f"{prefix}_role"] = np.array(role)
        arrays[f"{prefix}_query_ids"] = np.array(ids)
        # Spectral/model arrays are deliberately absent. Audit must only use IDs.
    np.savez(p, **arrays)
    return p


def test_numeric_hash_removes_order_and_spelling_but_not_intensity_scaling():
    a, n = _numeric_peak_hash("10,20,-0", "1,2,0")
    b, _ = _numeric_peak_hash("0.0 20.000 1e1", "-0.0 2.0 1.00")
    c, _ = _numeric_peak_hash("10,20,0", "2,4,0")
    assert n == 3 and a == b and a != c
    with pytest.raises(ValueError):
        _numeric_peak_hash("10,garbage", "1,2")
    with pytest.raises(ValueError):
        _numeric_peak_hash("10,nan", "1,2")
    with pytest.raises(ValueError):
        _numeric_peak_hash("10,20", "1")


def test_cross_role_composite_gate_uses_only_selected_distinct_identity_multipeak_queries(tmp_path):
    rows = [_row("t", "A", "train"), _row("f", "B", "fit_selector", "40 20 30 10", "4 2 3 1"),
            _row("d", "C", "dev_ranker", precursor="101"), _row("sealed", "D", "calibration"),
            _row("same", "A", "train")]
    tsv, db = _fixture(tmp_path, rows)
    original_db_hash = sha256_file(db)
    smoke = _smoke(tmp_path, ["t"], ["f"], ["d"])
    report = audit_spectrum_duplicates(tsv, db, tmp_path / "out", smoke)
    assert sha256_file(db) == original_db_hash
    assert report["selected_cross_role_count"] == 2
    assert report["selected_conflicting_query_ids"] == ["f", "t"]
    assert not report["selected_gate_passed"]
    assert report["duplicates"]["raw_peaks"]["groups_with_within_identity_repeats"] == 1
    assert report["duplicates"]["numeric_peaks"]["cross_role_cross_identity_groups"] == 1
    assert report["duplicates"]["composite"]["hashed_rows"] == 5
    assert report["denominators"]["selected_queries"] == 3
    assert not report["model_inputs_produced"] and not report["model_scores_or_metrics_computed"]
    assert "sealed" not in report["selected_conflicting_query_ids"]


def test_single_peak_common_fragments_and_missing_precursor_do_not_block(tmp_path):
    rows = [_row("t", "A", "train", "10", "1"), _row("f", "B", "fit_selector", "10", "1"),
            _row("d", "C", "dev_ranker", "10", "1", precursor="")]
    tsv, db = _fixture(tmp_path, rows)
    report = audit_spectrum_duplicates(tsv, db, tmp_path / "out", _smoke(tmp_path, ["t"], ["f"], ["d"]))
    assert report["selected_cross_role_count"] == 0 and report["selected_gate_passed"]
    assert report["duplicates"]["numeric_peaks"]["cross_role_cross_identity_groups_by_peak_count"] == {"single_peak": 1}
    assert report["denominators"]["composite_eligible_rows"] == 2


def test_multipeak_peak_only_overlap_without_matching_precursor_adduct_is_diagnostic(tmp_path):
    rows = [_row("t", "A", "train"), _row("f", "B", "fit_selector", precursor="101"),
            _row("d", "C", "dev_ranker", adduct="[M+Na]+")]
    tsv, db = _fixture(tmp_path, rows)
    report = audit_spectrum_duplicates(tsv, db, tmp_path / "out", _smoke(tmp_path, ["t"], ["f"], ["d"]))
    assert report["duplicates"]["numeric_peaks"]["cross_role_cross_identity_groups_by_peak_count"] == {"at_least_four_peaks": 1}
    assert report["duplicates"]["composite"]["duplicate_groups"] == 0
    assert report["selected_gate_passed"] and report["selected_cross_role_count"] == 0


def test_numeric_invalid_sealed_row_is_reported_without_scoring_or_db_changes(tmp_path):
    rows = [_row("good", "A", "train"), _row("bad", "B", "test", "INVALID", "1"),
            dict(_row("excluded", "C", "calibration"), reason="identity_mismatch")]
    tsv, db = _fixture(tmp_path, rows)
    report = audit_spectrum_duplicates(tsv, db, tmp_path / "out")
    assert report["denominators"]["numeric_invalid_rows"] == 1
    assert report["denominators"]["excluded_identity_audit_rows"] == 1
    assert report["selected_cross_role_count"] == 0
    assert report["duplicates"]["raw_peaks"]["hashed_rows"] == 2


def test_sealed_smoke_role_rejected_before_any_model_input_access(tmp_path):
    tsv, db = _fixture(tmp_path, [_row("t", "A", "train")])
    p = tmp_path / "sealed.npz"
    np.savez(p, train_role=np.array("calibration"))
    with pytest.raises(ValueError, match="sealed"):
        audit_spectrum_duplicates(tsv, db, tmp_path / "out", p)
    assert json.loads((tmp_path / "out/duplicate_summary.json").read_text())["status"] == "failed"
