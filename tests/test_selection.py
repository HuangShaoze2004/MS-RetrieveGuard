import hashlib
from pathlib import Path

import numpy as np
import pytest

from ms_retrieve_guard.selection import run_selection_smoke


def _inputs(tmp_path, roles=None):
    values = {}
    scores = {}
    for prefix, expected in [("fit", "fit_selector"), ("eval", "dev_ranker")]:
        values[f"{prefix}_role"] = np.asarray((roles or {}).get(prefix, expected))
        values[f"{prefix}_truth_keys"] = np.asarray([f"{prefix}a", f"{prefix}c"])
        values[f"{prefix}_query_ids"] = np.asarray([f"{prefix}0", f"{prefix}1"])
        values[f"{prefix}_keys_present"] = np.asarray([[f"{prefix}a", "b"], ["d", f"{prefix}c"]])
        values[f"{prefix}_keys_absent"] = np.asarray([["b", "e"], ["d", "e"]])
        scores[f"{prefix}_query_ids"] = values[f"{prefix}_query_ids"]
        for model in ("mlp", "nn"):
            for scenario in ("present", "absent"):
                scores[f"{model}_{prefix}_{scenario}"] = np.asarray([[3., 1.], [1., 3.]])
    data_path = tmp_path / "input.npz"
    score_path = tmp_path / "scores.npz"
    np.savez(data_path, **values)
    scores["input_npz_sha256"] = np.asarray(hashlib.sha256(data_path.read_bytes()).hexdigest())
    np.savez(score_path, **scores)
    return data_path, score_path, scores


def test_score_identity_binding_rejects_wrong_bundle_or_rows(tmp_path):
    data_path, score_path, scores = _inputs(tmp_path)
    scores["input_npz_sha256"] = np.asarray("0" * 64)
    np.savez(score_path, **scores)
    with pytest.raises(ValueError, match="not bound"):
        run_selection_smoke(data_path, score_path, tmp_path / "out", {})
    data_path, score_path, scores = _inputs(tmp_path)
    scores["eval_query_ids"] = scores["eval_query_ids"][::-1]
    np.savez(score_path, **scores)
    with pytest.raises(ValueError, match="misaligned"):
        run_selection_smoke(data_path, score_path, tmp_path / "out", {})


def test_sealed_or_mixed_role_and_formal_calibration_are_rejected(tmp_path):
    data_path, score_path, _ = _inputs(tmp_path, {"fit": ["fit_selector", "calibration"]})
    with pytest.raises(ValueError, match="Sealed"):
        run_selection_smoke(data_path, score_path, tmp_path / "out", {})
    with pytest.raises(ValueError, match="cannot perform formal"):
        run_selection_smoke(data_path, score_path, tmp_path / "out", {"formal_calibration": True})


def test_uninformative_absence_scores_reject_everything_without_zero_risk_claim(tmp_path):
    data_path, score_path, _ = _inputs(tmp_path)
    report = run_selection_smoke(data_path, score_path, tmp_path / "out", {})
    assert not report["formal_calibration_accessed"] and not report["test_accessed"]
    for model in report["models"].values():
        for selector in model["selectors"].values():
            assert selector["rule"]["reject_all"]
            assert selector["eval"]["n_present"] == 2
            assert selector["eval"]["n_absent"] == 2
            assert selector["eval"]["present_accepted_error_rate"] is None
            assert selector["eval"]["unknown_far"] == 0
