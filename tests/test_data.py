"""Tests for scientific invariants, not training convergence."""
import csv
import json

import numpy as np
import pytest

from ms_retrieve_guard.chemistry import fingerprint, standardize
from ms_retrieve_guard.data import audit_data, bin_spectrum, build_smoke, paired_candidate_keys, validation_roles


def test_stereo_identity_and_conservative_fragment_charge():
    assert standardize("C[C@H](O)CC").identity == standardize("C[C@@H](O)CC").identity
    assert "+" in standardize("C[NH3+]").smiles
    with pytest.raises(ValueError, match="multicomponent"):
        standardize("CCO.[Na+]")
    assert fingerprint("CCO").shape == (512,)


def test_group_split_is_exact_and_order_invariant():
    keys = [f"m{i:04d}" for i in range(100)]
    a = validation_roles(keys, 1729)
    assert a == validation_roles(keys[::-1] + keys, 1729)
    assert {role: list(a.values()).count(role) for role in set(a.values())} == {
        "dev_ranker": 15, "fit_selector": 25, "calibration": 40, "pilot_dev": 20}


def test_pairs_exclude_all_true_aliases_and_ignore_input_order():
    negatives = [f"negative{i}" for i in range(9)] + ["truth", "truth", "negative0"]
    present, absent = paired_candidate_keys("truth", negatives, 4)
    assert len(present) == len(absent) == 4
    assert present.count("truth") == 1 and "truth" not in absent
    assert len(set(present) & set(absent)) == 3
    assert (present, absent) == paired_candidate_keys("truth", negatives[::-1], 4)
    with pytest.raises(ValueError, match="insufficient"):
        paired_candidate_keys("truth", ["truth", "x"], 2)


def test_binning_validates_and_aggregates():
    x = bin_spectrum([10.01,10.02,30.,5.], [2.,2.,1.,100.])
    assert x.shape == (10050,) and x.dtype == np.float32
    assert x[100] == 1. and x[300] == .25 and x[:100].sum() == 0
    with pytest.raises(ValueError):
        bin_spectrum([1.], [float("nan")])
    with pytest.raises(ValueError):
        bin_spectrum([1005.], [1.])


def test_audit_identity_crossfold_filtering_and_sealed_spectra(tmp_path):
    smiles = ["CCO", "CCCO", "CCCCO", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC", "CCCCCCCCCCC", "CCCCCCCCCCCC"]
    # Distinct formulas mean no eligible negatives: retention must say zero, not pad.
    data = []
    for i,s in enumerate(smiles):
        mol = standardize(s)
        data.append({"identifier": f"q{i}", "smiles": s, "inchikey": mol.identity,
                     "formula": mol.formula, "parent_mass": str(mol.exact_mass), "fold": "val",
                     "mzs": "10,20", "intensities": "1,2"})
    duplicate = dict(data[0], identifier="cross", fold="train")
    data.append(duplicate)
    malformed = dict(data[1], identifier="wrong", inchikey="WRONGIDENTITYX")
    data.append(malformed)
    tsv = tmp_path / "small.tsv"
    with tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(data)
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({s: [s] for s in smiles}))
    summary = audit_data(tsv, candidates, tmp_path / "audit", {"k_grid": [2], "chemistry_workers":1})
    assert summary["cross_fold_identities"] == 1
    assert summary["quarantined_rows"] == 3
    assert summary["row_reason_counts"]["identity_mismatch"] == 1
    assert not summary["certification_sample_gate_passed"]
    import sqlite3
    db = sqlite3.connect(tmp_path / "audit" / "audit.sqlite")
    assert db.execute("SELECT COUNT(*) FROM queries WHERE role='calibration' AND mzs IS NOT NULL").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM molecules WHERE negative_count!=0").fetchone()[0] == 0
    with pytest.raises(ValueError, match="exactly"):
        build_smoke(tmp_path / "audit", tmp_path / "smoke.npz", {"k": 2, "n_train": 1, "n_fit": 1, "n_eval": 1})


def test_smoke_explicit_labels_roles_and_sealed_peak_sentinel(tmp_path):
    val = ["CCCCO", "CCC(O)C", "CC(C)CO", "CC(C)(C)O", "CCOCC", "COCCC", "COC(C)C"]
    train = ["CCCCC", "CC(C)CC", "CC(C)(C)C"]
    role_map = validation_roles([standardize(s).identity for s in val])
    data = []
    for i,s in enumerate(train + val):
        mol = standardize(s)
        sealed = role_map.get(mol.identity) == "calibration"
        data.append({"identifier": f"q{i}", "smiles": s, "inchikey": mol.identity,
                     "formula": mol.formula, "parent_mass": str(mol.exact_mass),
                     "fold": "train" if s in train else "val",
                     "mzs": "SEALED_DO_NOT_PARSE" if sealed else "10,20", "intensities": "1,2"})
    tsv = tmp_path / "data.tsv"
    with tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(data)
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({s: train if s in train else val for s in train+val}))
    audit_data(tsv, candidates, tmp_path / "audit", {"k_grid": [2], "chemistry_workers":1})
    manifest = build_smoke(tmp_path / "audit", tmp_path / "smoke.npz", {"k": 2, "n_train": 1, "n_fit": 1, "n_eval": 1})
    assert manifest["npz_sha256"] == manifest["output_sha256"]
    with np.load(tmp_path / "smoke.npz", allow_pickle=False) as arrays:
        assert str(arrays["fingerprint_bitorder"]) == "little"
        for prefix,role in [("train","train"), ("fit","fit_selector"), ("eval","dev_ranker")]:
            assert str(arrays[prefix+"_role"]) == role
            assert arrays[prefix+"_x"].shape == (1,10050)
            assert arrays[prefix+"_candidates_present"].shape == (1,2,512)
            truth = arrays[prefix+"_truth_keys"][0]
            assert list(arrays[prefix+"_keys_present"][0]).count(truth) == 1
            assert truth not in arrays[prefix+"_keys_absent"][0]


def test_original_truth_membership_alias_union_and_protonation_conflict(tmp_path):
    val = ["CCCCO", "CCC(O)C", "CC(C)CO", "CC(C)(C)O"]
    acids = ["CC(=O)O", "CC(=O)[O-]"]
    assert standardize(acids[0]).identity == standardize(acids[1]).identity
    data = []
    for i,s in enumerate(val+acids):
        mol = standardize(s)
        data.append({"identifier": f"q{i}", "smiles": s, "inchikey": mol.identity,
                     "formula": mol.formula, "parent_mass": str(mol.exact_mass), "fold": "val",
                     "mzs": "10,20", "intensities": "1,2"})
    tsv = tmp_path / "data.tsv"
    with tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(data)
    # Missing source truth must remain ineligible, even though a synthetic present
    # could have been constructed. Stereoisomer query aliases aggregate pools.
    pools = {val[0]: val[1:], val[1]: [val[0]], "C[C@H](O)CC": ["C[C@@H](O)CC", val[2]],
             val[2]: val, val[3]: val}
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(pools))
    summary = audit_data(tsv, candidates, tmp_path / "audit", {"k_grid": [2], "chemistry_workers":1})
    assert summary["conflicting_query_identities"] == 1
    import sqlite3
    db = sqlite3.connect(tmp_path / "audit" / "audit.sqlite")
    missing = db.execute("SELECT negative_count,truth_in_original_pool FROM molecules WHERE identity=?", (standardize(val[0]).identity,)).fetchone()
    merged = db.execute("SELECT negative_count,truth_in_original_pool FROM molecules WHERE identity=?", (standardize(val[1]).identity,)).fetchone()
    assert missing == (3,0)
    assert merged == (2,1)
    assert db.execute("SELECT COUNT(*) FROM queries WHERE reason LIKE '%identity_formula_mass_conflict%'").fetchone()[0] == 2


def test_spawn_prefill_exactly_matches_serial_and_resumes_cache(tmp_path):
    import multiprocessing
    import sqlite3
    from concurrent.futures import ProcessPoolExecutor
    from ms_retrieve_guard.data import ChemistryCache, _init_db
    smiles = ["CCO", "OCC", "C[C@H](O)CC", "C[C@@H](O)CC", "invalid!", "CCO.[Na+]", "CC(=O)O", "CC(=O)[O-]"]
    connections = []
    for mode in ["serial", "parallel"]:
        db = sqlite3.connect(tmp_path / (mode+".sqlite"))
        _init_db(db)
        cache = ChemistryCache(db)
        # Include a preexisting cached entry, duplicates, and invalid structures.
        cache.get(smiles[0])
        if mode == "parallel":
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as executor:
                cache.prefill(smiles[::-1]+smiles, executor, batch_size=3)
        else:
            cache.prefill(smiles+smiles, batch_size=3)
        db.commit()
        before = db.execute("SELECT COUNT(*) FROM chemistry").fetchone()[0]
        # A fresh cache instance must consume persistent results without inserts.
        ChemistryCache(db).prefill(smiles[::-1], batch_size=2)
        assert db.execute("SELECT COUNT(*) FROM chemistry").fetchone()[0] == before
        connections.append(db)
    for table in ["chemistry", "structures"]:
        assert connections[0].execute(f"SELECT * FROM {table} ORDER BY 1,2").fetchall() == connections[1].execute(f"SELECT * FROM {table} ORDER BY 1,2").fetchall()
