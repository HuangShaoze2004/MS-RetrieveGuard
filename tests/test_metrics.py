import numpy as np
import pytest
from ms_retrieve_guard.metrics import binomial_upper, confidence_scores, rank_order, retrieval_metrics, selective_metrics
from ms_retrieve_guard.selection import fit_temperature, select_development_threshold


def test_ties_are_permutation_invariant_and_not_fingerprint_labels():
    keys = np.array(["A", "B", "C"])
    scores = np.array([.8, .8, .1])
    original = keys[rank_order(scores, keys, "q")]
    permutation = np.array([2, 0, 1])
    assert original.tolist() == keys[permutation][rank_order(scores[permutation], keys[permutation], "q")].tolist()
    # Equal score/fingerprint does not turn another identity into a second truth.
    result = retrieval_metrics([scores], [keys], ["C"], ["q"])
    assert result["recall_at_1"] == 0
    with pytest.raises(ValueError):
        retrieval_metrics([scores], [keys], ["ABSENT"], ["q"])


def test_metric_denominators_and_zero_acceptance():
    result = selective_metrics([True, False, False, False], [True, False, True, False], [False, False, True, True])
    assert result["acceptance_rate"] == .5
    assert result["present_accepted_error_rate"] == 0
    assert result["unknown_far"] == .5
    assert result["accepted_error_rate"] == .5
    result = selective_metrics([False], [False], [False])
    assert result["present_accepted_error_rate"] is None
    assert result["accepted_error_rate"] is None
    assert result["unknown_far"] is None
    with pytest.raises(ValueError):
        selective_metrics([True], [True], [True])


def test_binomial_small_samples_and_undefined():
    assert binomial_upper(0, 0, .05) is None
    assert binomial_upper(1, 1, .05) == 1
    assert binomial_upper(0, 59, .05 / 24) <= .1
    assert binomial_upper(0, 58, .05 / 24) > .1
    assert binomial_upper(0, 121, .05 / 24) <= .05
    assert binomial_upper(0, 120, .05 / 24) > .05
    assert binomial_upper(0, 10, .05) == pytest.approx(1 - .05 ** .1)


def test_candidate_removal_requires_recomputing_probabilities():
    original = confidence_scores(np.array([[3., 2., 1.]]))["max_probability"][0]
    removed = confidence_scores(np.array([[2., 1.]]))["max_probability"][0]
    assert original != removed
    assert confidence_scores(np.ones((1, 4)))["negative_entropy"][0] == pytest.approx(-1)


def test_temperature_is_present_only_and_does_not_change_ranking():
    scores = np.array([[.6, .1], [.1, .7], [.4, .3]])
    keys = np.array([["A", "B"], ["C", "D"], ["E", "F"]])
    result = fit_temperature(scores, keys, ["A", "D", "E"])
    assert result["nll_after"] < result["nll_before"]
    assert (np.argmax(scores, 1) == np.argmax(scores / result["temperature"], 1)).all()
    with pytest.raises(ValueError):
        fit_temperature(scores, keys, ["MISSING", "D", "E"])


def test_impossible_development_rule_rejects_all_honestly():
    rule = select_development_threshold(np.ones(4), np.ones(4), np.zeros(4, dtype=bool), {})
    assert rule["reject_all"]
    assert rule["fit_present_risk"] is None
    assert not rule["certified"]
