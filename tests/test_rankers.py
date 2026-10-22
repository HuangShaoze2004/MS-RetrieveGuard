"""Scientific failure modes: sealed roles, explicit labels, ties, resuming."""

import json

import numpy as np
import pytest

from ms_retrieve_guard.rankers import (
    candidate_cosine_scores,
    explicit_target_indices,
    nearest_neighbor_scores,
    run_smoke,
    validate_npz,
)


def _bundle(tmp_path, **overrides):
    rng = np.random.default_rng(9)
    values = {"fingerprint_bitorder": np.array("little")}
    for prefix, role in (("train", "train"), ("fit", "fit_selector"), ("eval", "dev_ranker")):
        n = 4
        truth = np.array([f"{prefix}_truth_{i}" for i in range(n)], dtype="U40")
        values[prefix + "_role"] = np.array(role)
        values[prefix + "_x"] = rng.random((n, 8), dtype=np.float32)
        values[prefix + "_truth_keys"] = truth
        values[prefix + "_query_ids"] = np.array([f"{prefix}_{i}" for i in range(n)])
        values[prefix + "_fp"] = np.packbits(rng.integers(0, 2, (n, 16), dtype=np.uint8), axis=-1, bitorder="little")
        values[prefix + "_keys_present"] = np.array([[f"decoy_{i}", key] for i, key in enumerate(truth)], dtype="U40")
        values[prefix + "_keys_absent"] = np.array([[f"decoy_{i}", f"other_{i}"] for i in range(n)], dtype="U40")
        for variant in ("present", "absent"):
            values[f"{prefix}_candidates_{variant}"] = np.packbits(
                rng.integers(0, 2, (n, 2, 16), dtype=np.uint8), axis=-1, bitorder="little")
        values[prefix + "_candidates_present"][:, 1] = values[prefix + "_fp"]
        values[prefix + "_candidates_present"][:, 0] = values[prefix + "_fp"] ^ np.uint8(255)
        values[prefix + "_candidates_absent"][:, 0] = values[prefix + "_candidates_present"][:, 0]
    values.update(overrides)
    path = tmp_path / "smoke.npz"
    np.savez(path, **values)
    return path


def test_true_identity_labels_survive_fingerprint_collisions_and_permutation():
    candidates = np.array([["false_identity", "true_identity", "third"]])
    truth = np.array(["true_identity"])
    assert explicit_target_indices(candidates, truth).tolist() == [1]
    permutation = [2, 0, 1]
    assert explicit_target_indices(candidates[:, permutation], truth).tolist() == [2]
    # Equal fingerprints may generate equal scores, but cannot change the label.
    scores = candidate_cosine_scores(np.ones((1, 8), dtype=np.float32), np.full((1, 3, 1), 255, dtype=np.uint8))
    assert np.all(scores == scores[0, 0])
    with pytest.raises(ValueError, match="exactly one"):
        explicit_target_indices(np.array([["wrong", "wrong"]]), truth)


def test_nearest_neighbor_ties_are_identity_stable_under_reference_permutation():
    x = np.array([[1.0, 0], [1.0, 0]], dtype=np.float32)
    fingerprints = np.array([[1], [2]], dtype=np.uint8)
    keys = np.array(["z", "a"])
    ids = np.array(["first", "second"])
    query = x[:1]
    candidates = np.array([[[1], [2]]], dtype=np.uint8)
    scores, nearest = nearest_neighbor_scores(x, fingerprints, keys, ids, query, candidates)
    perm = np.array([1, 0])
    shuffled, shuffled_nearest = nearest_neighbor_scores(x[perm], fingerprints[perm], keys[perm], ids[perm], query, candidates)
    assert keys[nearest].tolist() == ["a"]
    assert keys[perm][shuffled_nearest].tolist() == ["a"]
    np.testing.assert_array_equal(scores, shuffled)
    np.testing.assert_allclose(scores, [[0, 1]])


def test_reject_sealed_role_before_required_arrays_are_loaded(tmp_path):
    path = tmp_path / "sealed.npz"
    np.savez(path, train_role=np.array("train"), fit_role=np.array("calibration"), eval_role=np.array("dev_ranker"))
    with pytest.raises(ValueError, match="Sealed role"):
        validate_npz(path, input_dim=8, output_dim=16)


def test_checksum_manifest_catches_wrong_candidate_bundle(tmp_path):
    path = _bundle(tmp_path)
    (tmp_path / "smoke_manifest.json").write_text(json.dumps({"npz_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="checksum"):
        validate_npz(path, input_dim=8, output_dim=16)


def test_swapped_candidate_vectors_without_keys_are_rejected(tmp_path):
    path = _bundle(tmp_path)
    with np.load(path) as data:
        values = {key: data[key] for key in data.files}
    values["eval_candidates_present"] = values["eval_candidates_present"][:, [1, 0]]
    np.savez(path, **values)
    with pytest.raises(ValueError, match="true fingerprint"):
        validate_npz(path, input_dim=8, output_dim=16)


def test_duplicate_molecule_with_distinct_query_ids_is_rejected(tmp_path):
    path = _bundle(tmp_path)
    with np.load(path) as data:
        values = {key: data[key] for key in data.files}
    values["fit_truth_keys"][1] = values["fit_truth_keys"][0]
    np.savez(path, **values)
    with pytest.raises(ValueError, match="duplicate molecule identity"):
        validate_npz(path, input_dim=8, output_dim=16)


def test_shared_negative_fingerprint_change_is_rejected(tmp_path):
    path = _bundle(tmp_path)
    with np.load(path) as data:
        values = {key: data[key] for key in data.files}
    values["eval_candidates_absent"][0, 0, 0] ^= np.uint8(1)
    np.savez(path, **values)
    with pytest.raises(ValueError, match="shared negative identity"):
        validate_npz(path, input_dim=8, output_dim=16)


def test_cross_role_molecule_overlap_and_absent_truth_are_rejected(tmp_path):
    path = _bundle(tmp_path)
    with np.load(path) as data:
        values = {key: data[key] for key in data.files}
    values["eval_truth_keys"][0] = values["train_truth_keys"][0]
    values["eval_keys_present"][0, 1] = values["train_truth_keys"][0]
    np.savez(path, **values)
    with pytest.raises(ValueError, match="overlap"):
        validate_npz(path, input_dim=8, output_dim=16)
    path = _bundle(tmp_path)
    with np.load(path) as data:
        values = {key: data[key] for key in data.files}
    values["eval_keys_absent"][0, 1] = values["eval_truth_keys"][0]
    np.savez(path, **values)
    with pytest.raises(ValueError, match="truth identity membership"):
        validate_npz(path, input_dim=8, output_dim=16)


def test_mlp_nn_smoke_saves_restorable_checkpoint_without_sealed_scores(tmp_path):
    path = _bundle(tmp_path)
    output = tmp_path / "run"
    result = run_smoke(path, output, {"input_dim": 8, "output_dim": 16, "hidden_dim": 12,
                                    "train_steps": 3, "batch_size": 2, "device": "cpu",
                                    "torch_num_threads": 1, "checkpoint_every": 2})
    assert result["completed_steps"] == 3
    assert result["checkpoint_validation"]["optimizer_resume_equal"]
    assert result["checkpoint_validation"]["prediction_equal"]
    assert result["score_permutation_validation"]["passed"]
    assert len(result["score_permutation_validation"]["cases"]) == 8
    assert json.loads((output / "score_permutation_validation.json").read_text())["passed"]
    with np.load(output / "scores.npz") as data:
        assert data["mlp_eval_present"].shape == (4, 2)
        assert data["nn_fit_absent"].shape == (4, 2)
        assert not any("test" in key or "calibration" in key for key in data.files)
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    resumed = run_smoke(path, tmp_path / "resumed", {"input_dim": 8, "output_dim": 16, "hidden_dim": 12,
                        "train_steps": 4, "batch_size": 2, "device": "cpu", "torch_num_threads": 1,
                        "resume_from": str(output / "checkpoint.pt")})
    assert resumed["completed_steps"] == 4
